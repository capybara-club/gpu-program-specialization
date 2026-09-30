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
import json,math,struct,subprocess,sys,tempfile
from pathlib import Path
from secant_sr_ast import Expression as E
from secant_sr import fit,write_dataset
x=E.input(0);y=E.input(1);c=E.bank(0);d=E.bank(1)
t=E('mul',(E.toggle4(x,c,y,d,0,1),E.toggle2(c,d,0)))
assert E.decode(t.encode())==t
affine=E.affine_bank(1,-.25,2.5)
assert affine.encode()==b'\xbf\x01'+struct.pack('<ff',-.25,2.5)+b'\x83'
transformed=E.toggle4(x,affine,c,E.affine_bank(0,0,-1.25),0,1)
assert E.decode(transformed.encode())==transformed
for p in range(4):
    assert transformed.evaluate((4.,5.),(2.,3.),p)==transformed.resolve((2.,3.),p).evaluate((4.,5.))
for cut in range(1,10):
    try:E.decode(affine.encode()[:cut])
    except ValueError:pass
    else:raise AssertionError('truncated affine accepted')
for p in range(4):
    resolved=t.resolve((2.,3.),p)
    assert t.evaluate((4.,5.),(2.,3.),p)==resolved.evaluate((4.,5.))
    assert E.decode(resolved.encode())==resolved
for data in [b'',b'\xbc',b'\xb7\0\x83',b'\x85\x83',t.encode()+b'\0']:
    try:E.decode(data)
    except ValueError:pass
    else:raise AssertionError('invalid bytecode accepted')
for operation in [lambda:E.toggle2(E('sin',(x,)),y,0),lambda:E.toggle4(x,y,c,d,1,1),lambda:E.bank(-1),lambda:E.constant(float('inf'))]:
    try:operation()
    except ValueError:pass
    else:raise AssertionError('invalid expression accepted')
exe=sys.argv[1]
X=[[i/16-.5,(i*7%31)/31-.5] for i in range(64)]
y=[a+b for a,b in X]
result=fit(X,y,executable=exe,backend='cpu',population=64,banks=2,constants=0,toggle_bits=3,generations=20)
assert result['solved'],result
assert result['validation_mse']<1e-10
resolved=E.decode(bytes.fromhex(result['resolved_ast_hex']))
for row in X:assert abs(resolved.evaluate(row)-sum(row))<1e-5
family=E.decode(bytes.fromhex(result['genotype_hex']))
assert family.resolve(result['coefficients'],result['permutation']).encode()==resolved.encode()
repeat=fit(X,y,executable=exe,backend='cpu',population=64,banks=2,constants=0,toggle_bits=3,generations=20)
assert repeat['resolved_ast_hex']==result['resolved_ast_hex']
assert repeat['configurations']==result['configurations']
# Exercise the client, fitting before breeding, and byte-exact winner replay.
refined=fit(X,[2.75*a-.125 for a,b in X],executable=exe,backend='cpu',
            population=64,banks=16,constants=2,toggle_bits=3,generations=4,
            refine_rounds=4,refine_budget=16,refine_scale=1,stop_nmse=0)
assert int(refined['refinement']['configurations'])>0,refined
assert refined['fitted_leaves'],refined
assert refined['coefficient_semantics']=='raw_bank_vector'
genotype=E.decode(bytes.fromhex(refined['genotype_hex']))
assert genotype.resolve(refined['coefficients'],refined['permutation']).encode()==bytes.fromhex(refined['resolved_ast_hex'])
assert refined['score_audit']['accepted']
# Constant targets have a defined SSE criterion even though R2 is degenerate.
constant=fit(X,[1]*64,executable=exe,backend='cpu',population=64,banks=1,constants=0,toggle_bits=2,generations=10)
assert constant['solved'],constant
for flag,value in [('population','-1'),('population','0'),('banks','0'),('score-mib','0'),('max-nodes','0')]:
    p=subprocess.run([exe,'--backend','cpu','--'+flag,value],capture_output=True,text=True)
    assert p.returncode>0,(flag,p.returncode)
print('Python toggle encoding/replay, custom dataset fitting, deterministic split, constant targets and invalid CLI passed')
