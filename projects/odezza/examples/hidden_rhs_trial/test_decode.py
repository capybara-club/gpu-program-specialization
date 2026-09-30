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
"""Regression: preserve structural slots when coefficients round to zero/one."""
import json
from pathlib import Path
import numpy as np
import native
from engine import compile_model
if __name__=='__main__':
 repo=Path(__file__).resolve().parents[2];count=0
 for base in ['recovery28_odezza','recovery28_prepared_diverse_replay']:
  for f in (repo/'scratch'/base).glob('*/report.json'):
   report=json.loads(f.read_text())
   for c in report.get('candidates',{}).values():
    e,p=native.decode(c);raw=[native.KNOWN['x0'],native.KNOWN['x1']]+[native.unpack(c,x) for x in c['resolved_programs'][2:]];X=np.random.default_rng(2818).uniform(-1.5,1.5,(4,6));expected=compile_model(raw,len(c['values']))(X,c['values'])[0];actual=compile_model(e,16)(X,p)[0];assert np.max(abs(expected-actual))<1e-6;count+=1
 assert native.parse('1*sin(1*x0*x1)')[0]=='*'
 assert native.parse('0*sin(1*x0*x1)')[0]=='*'
 print('Decoded and numerically checked',count,'retained candidates; zero/one structure preserved')
