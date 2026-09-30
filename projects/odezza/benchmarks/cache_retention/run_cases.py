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
"""Matched retention-memory campaign through the live service; no hidden truths."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'service_trial'))
from client import Client

def cases():
    g={'version':1,'states':['x0'],'integration':{'method':'rk4','dt':.02},
       'rng_banks':{'b':{'base':'uniform01','count':64,'seed':31,'scope':'run'}},
       'rng':{f'c{i}':{'bank':'b','axis':'trial','stream':f'c{i}',
                      'transform':{'kind':'affine','scale':.01,'shift':-.005}} for i in range(16)},
       'rules':{'R(z)':['z','sin(z)','cos(z)','tanh(z)','pow(z,2)','z/(1+pow(z,2))','sin(sin(z))','cos(sin(z))',
                           {'expr':'z','tags':['late']}]},
       'rhs':{'x0':'('+'+'.join(f'rng.c{i}' for i in range(16))+')*(R(x0)+2*R(x0)+3*R(x0)+4*R(x0))'},
       'families':[{'id':'mixed','tags':['all']}],
       'expansion':{'max_nodes':128,'max_depth':24},
       'limits':{'max_skeletons':10000,'max_variants':10000,'max_configurations':1000000},
       'retain':{'global':{'k':256,'unit':'evaluation_row'},'per_family':{'k':64,'unit':'evaluation_row'},
                 'by_tag':{'all':{'k':16,'unit':'evaluation_row'},'late':{'k':16,'unit':'evaluation_row'}}}}
    p={'states':['x0'],'trajectories':[{'initial':[.2],'times':[0,.02,.04],'values':[[.2],[.2],[.2]]}]}
    for chunk in [4096,1024]:
        yield f'tagged_16_slots_chunk_{chunk}',{'problem':p,'grammar':g,'execution':{'batch_variants':64,'module_systems':32,
                'max_chunk_configurations':chunk,'max_seconds':120}}

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--port',type=int,default=14222);a=p.parse_args()
    runs=[]
    for name,request in cases():
        c=Client(port=a.port)
        try:
            t=time.perf_counter();job=c.submit(json.dumps(request).encode());handle=job['handle'];deadline=time.monotonic()+150
            while True:
                s=c.status(handle)
                if s['state']=='terminal':break
                assert time.monotonic()<deadline,s
                time.sleep(.1)
            r=c.result(handle);runs.append({'name':name,'request':request,'elapsed':time.perf_counter()-t,'report':r})
            Path(a.output).write_text(json.dumps(runs,indent=2)+'\n')
            print(name,r['status'],r.get('error'),r.get('timing',{}).get('total_seconds'),r.get('memory',{}).get('classes',{}).get('retained'),flush=True)
            c.rpc('release.'+handle)
            assert r['status']=='complete',r
        finally:c.close()
if __name__=='__main__':main()
