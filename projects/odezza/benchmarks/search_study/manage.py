# SPDX-FileCopyrightText: 2026 Charles Durham
# SPDX-License-Identifier: MIT
#
# MIT License
#
# Copyright (c) 2026 Charles Durham
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Deploy a separate supplement, then supervise/report it from mac1 tmux."""
import argparse
import fcntl
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time
from benchmarks.lm_tuning.common import REPO, file_hash, load, save
from benchmarks.lm_tuning.deploy import ssh, lanes, python_for
from benchmarks.lm_tuning.coordinator import collect
from .report import report

HOSTS=['rack1','rohini','ada']

def deploy(root):
    plan=load(root/'plan.json');remote='/home/cdurham/odezza/scratch/search_study_20260910'
    sources={str(p.relative_to(REPO)):str(p.relative_to(REPO)) for p in (REPO/'benchmarks/search_study').glob('*.py')}
    for name in ['common.py','fixed_capacity.py']:
        sources['benchmarks/search_study/helpers/'+name]='scratch/trajectory_benchmarks/'+name
    hashes={target:dict(source=source,sha256=file_hash(REPO/source)) for target,source in sources.items()}
    save(root/'source-hashes.json',hashes)
    with tarfile.open(root/'source.tar.gz','w:gz') as archive:
        for target,source in sorted(sources.items()):
            data=(REPO/source).read_bytes();member=tarfile.TarInfo(target);member.size=len(data);member.mode=0o644
            archive.addfile(member,io.BytesIO(data))
    save(root/'execution.json',dict(remote=remote,runtime=plan['runtime_remote'],source_archive_sha256=file_hash(root/'source.tar.gz')))
    for host in HOSTS:
        ssh(host,['mkdir',remote])
        subprocess.run(['scp','-q',str(root/'source.tar.gz'),host+':'+remote+'/source.tar.gz'],check=True)
        ssh(host,['tar','-xzf',remote+'/source.tar.gz','-C',remote])
    for host,device,lane in lanes(HOSTS):
        destination=remote+'/queues/'+lane+'/groups';ssh(host,['mkdir','-p',destination])
        files=[root/'groups'/(c['group_id']+'.json') for c in plan['selections'] if c['lane']==lane]
        for p in files:
            if file_hash(p)!=plan['group_hashes'][p.stem]:raise ValueError('Changed group')
        subprocess.run(['scp','-q',*map(str,files),host+':'+destination+'/'],check=True)
    # Preparation uses all public cases but stops before creating any search.
    destination=remote+'/preflight';ssh('rack1',['mkdir','-p',destination+'/groups'])
    files=[root/'groups'/(c['group_id']+'.json') for c in plan['selections']]
    subprocess.run(['scp','-q',*map(str,files),'rack1:'+destination+'/groups/'],check=True)
    code='import os,runpy,sys;os.chdir('+repr(remote)+');sys.argv=["benchmark_worker",*sys.argv[1:]];runpy.run_module("benchmarks.search_study.benchmark_worker",run_name="__main__")'
    r=ssh('rack1',['env','PYTHONPATH='+remote+':'+plan['runtime_remote'],'PYTHONDONTWRITEBYTECODE=1',
        python_for('rack1'),'-B','-c',code,'--root',destination,'--device','0','--deadline',str(time.time()+180),'--preflight'],timeout=180)
    (root/'preflight.log').write_text(r.stdout+r.stderr)
    for name in ['status.json','results.jsonl']:
        subprocess.run(['scp','-q','rack1:'+destination+'/'+name,str(root/('preflight-'+name))],check=True)
    results=[json.loads(s) for s in (root/'preflight-results.jsonl').read_text().splitlines()]
    save(root/'preflight-summary.json',dict(trials=len(results),prepared=sum(r['status']=='prepared_without_submission' for r in results),
        rejected=[{k:r.get(k) for k in ('case_id','trial_id','status','error')} for r in results if r['status']!='prepared_without_submission']))
    if len(results)!=plan['expected_trials'] or any(r['status']!='prepared_without_submission' for r in results):
        raise ValueError('Benchmark preparation rejected cases; investigate retained preflight before launch')
    print(json.dumps(dict(status='deployed_and_preflight_passed',trials=len(results))))

def start(root,plan,execution):
    for host,device,lane in lanes(HOSTS):
        receipt=root/'hosts'/lane/'launch.json'
        if receipt.exists():continue
        unit='odezza-search-study-20260910-'+lane.replace('_','-')
        command=['systemd-run','--user','--unit='+unit,'--property=KillMode=control-group',
            '--property=RuntimeMaxSec='+str(max(1,int(plan['random_deadline']-time.time()+30))),
            '--property=TimeoutStopSec=20','--working-directory='+execution['remote'],
            '--setenv=PYTHONPATH='+execution['remote']+':'+execution['runtime'],
            '--setenv=CUDA_MODULE_LOADING=EAGER','--setenv=ODEZZA_CORE_LIBRARY='+execution['runtime']+'/build/libodezza.so',
            '--setenv=PYTHONDONTWRITEBYTECODE=1','--setenv=ODEZZA_TELEGRAM_DISABLE=1',
            python_for(host),'-B','-m','benchmarks.search_study.benchmark_worker',
            '--root',execution['remote']+'/queues/'+lane,'--device',str(device),'--deadline',str(plan['random_deadline'])]
        result=ssh(host,command,timeout=30);save(receipt,dict(at=time.time(),unit=unit,stdout=result.stdout,stderr=result.stderr))

def run(root):
    lock=(root/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'completion.json').exists():raise ValueError('Already terminal; use report')
    plan=load(root/'plan.json');execution=load(root/'execution.json');random=Path(plan['random_root'])
    for target,source in load(root/'source-hashes.json').items():
        if file_hash(REPO/source['source'])!=source['sha256']:raise ValueError('Supplement source changed: '+target)
    for gid,want in plan['group_hashes'].items():
        if file_hash(root/'groups'/(gid+'.json'))!=want:raise ValueError('Benchmark group changed')
    check=load(root/'preflight-summary.json')
    if check['prepared']!=plan['expected_trials']:raise ValueError('Unresolved API preparation failures')
    save(root/'supervisor.json',dict(started_at=time.time(),phase='waiting_for_random_campaign'))
    while not (random/'completion.json').exists() and time.time()<plan['random_deadline']+90:
        report(root);time.sleep(20)
    previous=load(random/'completion.json') if (random/'completion.json').exists() else {}
    repaired=(root/'repair-validation.json').exists() and load(root/'repair-validation.json').get('passed') is True
    # A diagnosed host-side failure may leave a random queue terminal early.
    # Never reinterpret it as a completed trial or mix its rows with repairs.
    lanes_state=load(random/'status.json').get('lanes',{}) if (random/'status.json').exists() else {}
    known_failure=repaired and len(lanes_state)==4 and all(
        s.get('finished_at') and (s.get('status')=='complete' or
        (s.get('status')=='failed' and s.get('error')=='Fitting campaign failed: division by zero'))
        for s in lanes_state.values())
    if previous.get('status')!='complete' and not known_failure:
        save(root/'completion.json',dict(status='benchmark_not_started',reason='random_campaign_failed_or_unconfirmed',at=time.time()))
    elif time.time()+750>=plan['random_deadline']:
        save(root/'completion.json',dict(status='benchmark_not_started',reason='insufficient_remaining_campaign_budget',at=time.time()))
    else:
        for host,device,lane in lanes(HOSTS):
            unit=load(random/'hosts'/lane/'launch.json')['unit']
            result=subprocess.run(['ssh','-o','BatchMode=yes',host,'systemctl','--user','is-active',unit],capture_output=True,text=True)
            if result.stdout.strip() not in ('inactive','failed'):raise RuntimeError('Random GPU worker is not confirmed stopped: '+lane)
        start(root,plan,execution)
        save(root/'supervisor.json',dict(at=time.time(),phase='benchmark_running'))
        while True:
            states=collect(root,execution['remote'],lanes(HOSTS));report(root)
            if all(s.get('finished_at') for s in states.values()) or time.time()>plan['random_deadline']+90:break
            time.sleep(15)
        rows=report(root)['benchmark_completed']
        complete=all(s.get('status')=='complete' and s.get('finished_at') for s in states.values()) and rows==plan['expected_trials']
        save(root/'completion.json',dict(status='complete' if complete else 'partial_or_failed',at=time.time(),lanes=states,reported_trials=rows))
    result=report(root)
    sys.path.insert(0,str(REPO/'scratch/fitting_batch_trial'))
    from telegram_notify import send
    message=(f"Odezza search-design report ready. Random: {result['random_completed']}/{result['random_expected']} trials; "
        f"benchmarks: {result['benchmark_completed']}/{result['benchmark_expected']}. "
        f"Status: {load(root/'completion.json')['status']}. Report: {root}/REPORT.md")
    for attempt in range(3):
        try:save(root/'telegram-complete.json',dict(message_id=send(message),message=message));break
        except RuntimeError as e:save(root/'telegram-error.json',dict(attempt=attempt,error=str(e)));time.sleep(10)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['deploy','run','report']);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve()
    {'deploy':deploy,'run':run,'report':report}[a.action](root)
