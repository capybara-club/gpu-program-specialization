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
"""Freeze the diagnosed Python-only repair separately from the active study."""
import argparse
import io
from pathlib import Path
import subprocess
import tarfile
from benchmarks.lm_tuning.common import REPO, load, save, file_hash
from benchmarks.lm_tuning.deploy import ssh, python_for

HOSTS=('rack1','rohini','ada')
PATCHES=('scratch/fitting_batch_trial/runtime/odezza/lm_ast.py',
         'scratch/fitting_batch_trial/lm_toggle/service.py','scratch/fitting_batch_trial/lm_toggle/native_service.py',
         'scratch/fitting_batch_trial/lm_toggle/test_invalid_candidates.py',
         'scratch/fitting_batch_trial/lm_toggle/test_planner.py','tests/test_lm.py')

def deploy(root, version):
    if list((root/'hosts').glob('*/launch.json')):raise ValueError('Supplement already launched')
    execution=load(root/'execution.json');original=load(root/'plan.json')['runtime_remote'];remote=execution['remote'];runtime=remote+'/runtime-'+version
    save(root/('execution-before-'+version+'.json'),execution)
    sources={str(p.relative_to(REPO)):str(p.relative_to(REPO)) for p in (REPO/'benchmarks/search_study').glob('*.py')}
    for name in ('common.py','fixed_capacity.py'):
        sources['benchmarks/search_study/helpers/'+name]='scratch/trajectory_benchmarks/'+name
    sources.update({'runtime-'+version+'/'+p:p for p in PATCHES})
    hashes={target:dict(source=source,sha256=file_hash(REPO/source)) for target,source in sources.items()}
    archive=root/('source-'+version+'.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:
        for target,source in sorted(sources.items()):
            data=(REPO/source).read_bytes();member=tarfile.TarInfo(target);member.size=len(data);member.mode=0o644
            tar.addfile(member,io.BytesIO(data))
    save(root/('source-hashes-before-'+version+'.json'),load(root/'source-hashes.json'))
    receipts={}
    for host in HOSTS:
        ssh(host,['mkdir',runtime])
        ssh(host,['tar','-xzf',original+'/source.tar.gz','-C',runtime])
        for part in ('build','scratch/fitting_batch_trial/bin'):
            ssh(host,['mkdir','-p',runtime+'/'+part])
        binaries=('build/libodezza.so','scratch/fitting_batch_trial/bin/odezza-fit-core-run',
                  'scratch/fitting_batch_trial/bin/libodezza_expand.so')
        for p in binaries:ssh(host,['cp',original+'/'+p,runtime+'/'+p])
        subprocess.run(['scp','-q',str(archive),host+':'+remote+'/'+archive.name],check=True)
        ssh(host,['tar','-xzf',remote+'/'+archive.name,'-C',remote])
        code='''import hashlib,json
from pathlib import Path
old=Path(%r);new=Path(%r)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
files=%r
assert all(sha(old/p)==sha(new/p) for p in files)
print(json.dumps(dict(original_archive=sha(old/'source.tar.gz'),binaries={p:sha(new/p) for p in files})))
'''%(original,runtime,binaries)
        import json
        receipts[host]=json.loads(ssh(host,[python_for(host),'-B','-c',code]).stdout)
    save(root/('repair-deployment-'+version+'.json'),dict(original_runtime=original,runtime=runtime,patches={p:file_hash(REPO/p) for p in PATCHES},hosts=receipts))
    save(root/'source-hashes.json',hashes)
    save(root/'execution.json',dict(execution,runtime=runtime,source_archive_sha256=file_hash(archive),repair='preserve literal-zero division during derivative preparation'))
    # Repeat all public API preparations against the repaired, frozen runtime.
    plan=load(root/'plan.json');dest=remote+'/preflight-'+version
    ssh('rack1',['mkdir','-p',dest+'/groups'])
    files=[root/'groups'/(c['group_id']+'.json') for c in plan['selections']]
    subprocess.run(['scp','-q',*map(str,files),'rack1:'+dest+'/groups/'],check=True)
    code='import os,runpy,sys;os.chdir('+repr(remote)+');sys.argv=["worker",*sys.argv[1:]];runpy.run_module("benchmarks.search_study.benchmark_worker",run_name="__main__")'
    import time
    result=ssh('rack1',['env','PYTHONPATH='+remote+':'+runtime,'PYTHONDONTWRITEBYTECODE=1',python_for('rack1'),'-B','-c',code,
        '--root',dest,'--device','0','--deadline',str(time.time()+180),'--preflight'],timeout=180)
    (root/('preflight-'+version+'.log')).write_text(result.stdout+result.stderr)
    result_path=root/('preflight-'+version+'-results.jsonl')
    subprocess.run(['scp','-q','rack1:'+dest+'/results.jsonl',str(result_path)],check=True)
    rows=[json.loads(line) for line in result_path.read_text().splitlines()]
    if len(rows)!=plan['expected_trials'] or any(r['status']!='prepared_without_submission' for r in rows):
        raise ValueError('Repaired API preparation did not pass every selected policy')
    save(root/'preflight-summary.json',dict(trials=len(rows),prepared=len(rows),rejected=[],runtime=runtime))
    print('Separate repaired runtime deployed; all API preparations passed. GPU regression still required.')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--version',required=True,choices=['v2','v3','v4','v5'])
    args=parser.parse_args();deploy(args.root.resolve(),args.version)
