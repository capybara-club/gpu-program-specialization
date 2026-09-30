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
"""New service policy checks. Run against an isolated broker with no workers."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from client import Client


def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,required=True)
    p.add_argument('--output',required=True);p.add_argument('--reject-only',action='store_true')
    a=p.parse_args();c=Client(port=a.port)
    base=json.loads((Path(__file__).resolve().parents[1]/'examples/grammar/submit.json').read_text())
    checks=[]
    def run(label,change,error=None):
        q=deepcopy(base);change(q);h=c.submit(json.dumps(q).encode())['handle']
        try:
            s=c.status(h);assert s['attempts']==0,s
            if error:
                r=c.result(h)
                assert r['error']==error and r['completed_configurations']==0,r
                checks.append(dict(label=label,status=s,report=r))
            else:
                assert s['state']=='queued',s
                checks.append(dict(label=label,status=s));c.rpc('cancel.'+h)
        finally:c.rpc('release.'+h)
    try:
        health=c.health();assert health['admission_limits']['configurations']==10**12
        for field in ('skeletons','variants','configurations','derivations','expansion_steps'):
            run(field,lambda q,f=field:q['grammar'].update(limits={'max_'+f:health['admission_limits'][f]+1}), 'search_'+field+'_limit')
        run('family conflict',lambda q:q['grammar'].update(limits={'max_configurations':15},
            families=[{'id':'a','limits':{'max_configurations':8}},{'id':'b','limits':{'max_configurations':8}}]),'invalid_search_allocation')
        def costly(q):
            q['grammar']['limits']={'max_configurations':10**12}
            q['grammar']['integration']['dt']=.0001
        run('aggregate rollout work',costly,'rollout_work_limit')
        run('host minimum',lambda q:q['execution'].update(max_host_bytes=1024),'host_minimum_exceeds_reservation')
        run('host ceiling',lambda q:q['execution'].update(max_host_bytes=2147483649),'host_reservation_limit')
        if not a.reject_only:
            run('billion configs',lambda q:q['grammar'].update(limits={'max_skeletons':1000000,'max_variants':1000000,'max_configurations':2048000000}))
            assert checks[-1]['status']['allocation']['configurations']==2048000000
        Path(a.output).write_text(json.dumps(dict(passed=True,health=health,checks=checks),indent=2)+'\n')
        print('PASS',len(checks),'prequeue allocation checks; zero worker claims')
    finally:c.close()


if __name__=='__main__':main()
