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
"""Run a complete autonomous model from JSON on the separate GPU prototype."""
import argparse
import json
from pathlib import Path
from prototype import Solver
from audit import write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request',type=Path); parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',type=int,default=0)
    args=parser.parse_args(); request=json.loads(args.request.read_text())
    if request.get('schema')!='odezza.rosenbrock-trial.v1': raise ValueError('unknown request schema')
    if set(request)-{'schema','states','rhs','trajectories','solver'}: raise ValueError('unknown request field')
    settings=request.get('solver',{})
    if set(settings)-{'rtol','atol','h0','max_attempts','predictions'}: raise ValueError('unknown solver field')
    solver=Solver(request['rhs'],request['states'])
    result=solver.run(request['trajectories'],device=args.device,**settings)
    result.update(schema='odezza.rosenbrock-trial-result.v1',compile_seconds=solver.compile_seconds,
                  load_seconds=solver.load_seconds,compiled=solver.compiled,
                  complete=all(s['status']==0 for s in result['stats']))
    write(args.out,result)
    print(json.dumps(dict(complete=result['complete'],kernel_ms=result['kernel_ms'],output=str(args.out))))
    if not result['complete']: raise SystemExit(2)


if __name__=='__main__': main()
