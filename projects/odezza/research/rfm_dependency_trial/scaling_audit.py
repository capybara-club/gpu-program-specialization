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
"""Audit missed rankings after all predictions are frozen; no fitting feedback."""
import argparse
import json
from pathlib import Path
import numpy as np
import sympy as sp
from generate import symbolic
from learn import view


def run(root):
    root=Path(root);cfg=json.loads((root/'private/config.json').read_text());rows=[]
    for n in cfg['state_counts']:
        symbols=sp.symbols('x0:'+str(n))
        for i in range(cfg['development_systems'],cfg['development_systems']+cfg['systems']):
            case=f'{i:03d}';data=root/f'n{n}'
            private=json.loads((data/f'private/{case}.json').read_text())
            r=json.loads((data/f'results/{case}-m64-t0.5.json').read_text())
            observations=json.loads((data/f'public/{case}.json').read_text())
            x,y,_,_=view(observations,64,.5)
            for j,truth in enumerate(private['support']):
                support=np.array(truth,dtype=bool);scores={};ranks={}
                for mode,values in r['methods'].items():
                    d=np.diag(np.array(values['matrices'][j]));d=d/max(d.max(),1e-30)
                    scores[mode]=d.tolist()
                    order=np.argsort(-d,kind='stable');rank=np.empty(n,dtype=int);rank[order]=np.arange(1,n+1)
                    ranks[mode]=int(rank[support].max())
                if ranks['diagonal']<=support.sum():continue
                f=cfg['rhs_scale']*symbolic(private['asts'][j],symbols)
                grad=np.zeros_like(x)
                for k in np.flatnonzero(support):
                    grad[:,k]=sp.lambdify(symbols,sp.diff(f,symbols[k]),'numpy')(*x.T)
                grad*=np.array(r['scales']['x'])[None,:]/r['scales']['y'][j]
                oracle=(grad**2).mean(0);oracle/=max(oracle.max(),1e-30)
                actual=sp.lambdify(symbols,f,'numpy')(*x.T)
                label_rmse=float(np.sqrt(np.mean((y[:,j]-actual)**2))/max(np.sqrt(np.mean(actual**2)),1e-8))
                rows.append({'states':n,'id':case,'rhs_index':j,'rhs':str(f),'support':list(map(int,np.flatnonzero(support))),
                             'worst_true_rank':ranks,'estimated_scores':scores,'oracle_sensitivity_scores':oracle.tolist(),
                             'relative_derivative_label_rmse':label_rmse})
    (root/'report/outlier-audit.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    for row in sorted(rows,key=lambda r:r['worst_true_rank']['diagonal'],reverse=True)[:5]:
        print(json.dumps({**{k:row[k] for k in ['states','id','rhs_index','rhs','worst_true_rank','relative_derivative_label_rmse']},
              'true_inputs':[{ 'state':k,'oracle':row['oracle_sensitivity_scores'][k],
                  **{m:v[k] for m,v in row['estimated_scores'].items()}} for k in row['support']]}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();run(a.root)
