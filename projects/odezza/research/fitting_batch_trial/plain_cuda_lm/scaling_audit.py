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
"""Audit smaller-state timing equivalence; retain and replay discrepancies."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'assignment_search'))
from verify import verify

def main():
    p=argparse.ArgumentParser();p.add_argument('runs',type=Path);a=p.parse_args();audit=[]
    for folder in sorted(a.runs.iterdir()):
        if not folder.is_dir():continue
        base=json.loads((folder/'gpu-w1.json').read_text());case=json.loads((folder/'case.json').read_text())
        for width in (2,4,8):
            other=json.loads((folder/f'gpu-w{width}.json').read_text())
            different={k:sum(x!=y for x,y in zip(base[k],other[k])) for k in
                       ('coefficients','initial_mse','mse','iterations','accepted','attempts')}
            row=dict(case=folder.name,width=width,different_elements=different,exact=not any(different.values()))
            if not row['exact']:
                index=max(range(base['count']),key=lambda i:abs(base['mse'][i]-other['mse'][i]));checks=[]
                for w,r in ((1,base),(width,other)):
                    params=r['coefficients'][index*case['p']:(index+1)*case['p']]
                    req=dict(rhs=case['fixed']+[case['variants'][index%len(case['variants'])]],parameter_values=params,rows=case['rows'],target=1e-10)
                    check=verify(req)
                    checks.append(dict(width=w,fit_index=index,reported_mse=r['mse'][index],parameter_values=params,training_replay=check))
                row.update(status='non_equivalent_fitting_paths',diagnostic=checks,
                           cause='Not isolated. Do not assume harmless roundoff or claim identical optimizer work.',
                           next_action='Compare per-iteration sensitivities, normal equations and proposals before admitting this cell to a topology-only speed comparison.')
            audit.append(row)
    (a.runs/'audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
    print(json.dumps(audit))

if __name__=='__main__':main()
