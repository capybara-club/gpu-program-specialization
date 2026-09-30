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
"""Post-freeze derivative-label diagnostic. Never used by learner or calibration."""
import json
from pathlib import Path
import numpy as np
import sympy as sp
from generate import symbolic
from learn import view


def main():
    root=Path('pilot-01'); cfg=json.loads((root/'manifest.json').read_text())
    rows=[]
    for case in cfg['cases']:
        if case['split']!='test':continue
        data=json.loads((root/case['path']).read_text())
        private=json.loads((root/'private'/(case['id']+'.json')).read_text())
        syms=sp.symbols('x0:'+str(cfg['config']['states']))
        funcs=[sp.lambdify(syms,cfg['config']['rhs_scale']*symbolic(a,syms),'numpy') for a in private['asts']]
        for count,duration in [(1,4.),(16,.5),(16,4.)]:
            x,y,_,_=view(data,count,duration)
            actual=np.column_stack([np.broadcast_to(f(*x.T),len(x)) for f in funcs])
            rmse=np.sqrt(np.mean((y-actual)**2,axis=0))
            relative=rmse/np.maximum(np.sqrt(np.mean(actual**2,axis=0)),1e-8)
            rows.append(dict(id=case['id'],count=count,duration=duration,rmse=rmse.tolist(),relative_rmse=relative.tolist()))
    Path('report-01/derivative-audit.json').write_text(json.dumps(rows,indent=2)+'\n')
    for count,duration in [(1,4.),(16,.5),(16,4.)]:
        values=np.concatenate([r['relative_rmse'] for r in rows if (r['count'],r['duration'])==(count,duration)])
        print(count,duration,'median/p95/max relative derivative RMSE',np.quantile(values,[.5,.95,1.]))


if __name__=='__main__':main()
