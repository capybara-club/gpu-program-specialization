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
"""Budgeted training-only finalist polishing; no feedback into GP evolution."""
import copy
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import time

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'polish'))
from polish import Expression, prepare, fit, materialize, mse


def expression_text(node):
    if node.op == 'constant': return f'{node.value:.9g}'
    if node.op == 'input': return 'x'+str(node.value)
    binary={'add':'+','sub':'-','mul':'*','div':'/'}
    if node.op in binary:
        return '('+expression_text(node.args[0])+' '+binary[node.op]+' '+expression_text(node.args[1])+')'
    return node.op+'('+', '.join(map(expression_text,node.args))+')'


def polish(record, job, data, replay, deadline, starts=2):
    """deadline caps this phase and never exceeds the end-to-end fit deadline."""
    begin=time.monotonic()
    out=copy.deepcopy(record)
    out['search_result']=copy.deepcopy(record)
    out.pop('native_result',None)
    out['final_model']=dict(record['native_result'],stage='search')
    out['resolved_ast_hex']=record['native_result']['resolved_ast_hex']
    out['strict_success']=record['native_result']['solved']
    report=dict(method='scipy_f64_lm_finalists',policy='tied_parameters_fixed_literals',
                candidates=[],accepted=False,skipped=None,search_validation_used_for_selection=False)
    out['final_polish']=report
    source=record['native_result']
    if source['train_nmse'] <= 1e-6 or begin >= deadline-.15:
        report['skipped']='training_threshold' if source['train_nmse']<=1e-6 else 'time_budget'
        report['seconds']=time.monotonic()-begin
        return out
    raw=Path(data).read_bytes()
    magic,version,inputs,rows,test=struct.unpack_from('<8sIIQQ',raw)
    if (magic,version,inputs,rows,test)!=(b'SECSRDS\0',1,job['num_inputs'],job['num_train_rows'],job['num_validation_rows']):
        raise ValueError('dataset header mismatch')
    # Only training rows are made available to the optimizer.
    values=np.frombuffer(raw,dtype='<f4',offset=32,count=(inputs+1)*rows)
    x=values[:inputs*rows].reshape(inputs,rows).T;y=values[inputs*rows:]
    candidates=[dict(source,index=-1)]+[dict(v,index=i) for i,v in enumerate((record.get('finalists') or {}).get('models',[]))]
    unique=[];seen=set()
    for candidate in candidates:
        key=candidate['resolved_ast_hex']
        if key not in seen:unique.append(candidate);seen.add(key)
    incumbent=Expression.decode(bytes.fromhex(source['resolved_ast_hex']))
    incumbent_code,unused=prepare(incumbent,[],0,'tied')
    incumbent_mse=mse(incumbent_code,x,y,unused)
    best_mse,best_model,best_index=incumbent_mse,incumbent,None
    selected_initial_mse=None
    for index,candidate in enumerate(unique):
        remaining=deadline-time.monotonic()-.15
        if remaining<=0:report['skipped']='time_budget';break
        genome=Expression.decode(bytes.fromhex(candidate['genotype_hex']))
        code,center=prepare(genome,candidate['coefficients'],candidate['permutation'],'tied',candidate.get('fitted_leaves'))
        if materialize(code,center).encode().hex()!=candidate['resolved_ast_hex']:
            raise ValueError('finalist binding/parameter replay failed')
        allowance=remaining/(len(unique)-index)
        if not math.isfinite(mse(code,x,y,center)):
            report['candidates'].append(dict(index=candidate['index'],skipped='nonfinite_numpy_center'))
            continue
        result=fit(code,center,x,y,job['seed']+index,allowance,starts=starts,evaluations=200)
        report['candidates'].append(dict(index=candidate['index'],allowance_seconds=allowance,
            parameter_count=len(center),**result))
        if result['train_mse']<best_mse:
            best_mse=result['train_mse'];best_model=materialize(code,result['parameters']);best_index=candidate['index']
            selected_initial_mse=result['initial_mse']
    report['selected_archive_index']=best_index
    report['numpy_train_mse']=best_mse
    report['selected_initial_mse']=selected_initial_mse
    report['coefficient_improved_selected']=selected_initial_mse is not None and best_mse<selected_initial_mse
    if best_index is not None:
        # Final native replay reports holdout only after training has chosen a model.
        # The acceptance guard below reads only native training error.
        run=subprocess.run([str(replay),str(data),best_model.encode().hex()],capture_output=True,text=True,check=True,timeout=30)
        native=json.loads(run.stdout)
        gap=abs(math.sqrt(native['train_mse'])-math.sqrt(best_mse))/max(1,float(np.sqrt(np.mean(y.astype(np.float64)**2))),math.sqrt(best_mse))
        if gap>2e-5:raise ValueError('polished winner failed native CPU numerical audit')
        report['native_audit']=dict(relative_rmse_gap=gap,tolerance=2e-5,accepted=True)
        if native['train_mse']<record['train_mse']:
            report['accepted']=True
            out['resolved_ast_hex']=best_model.encode().hex()
            out['best_expression']=expression_text(best_model)
            out.update(native)
            out['train_r2']=1-native['train_mse']/job['train_variance']
            out['validation_r2']=1-native['validation_mse']/job['test_variance']
            out['accuracy_solution']=int(out['validation_r2']>.999)
            out['strict_success']=native['train_mse']/job['train_variance']<=1e-6 and native['validation_mse']/job['test_variance']<=1e-6
            out['final_model']=dict(stage='cpu_final_polish',resolved_ast_hex=out['resolved_ast_hex'],
                expression=out['best_expression'],**native,score_audit=report['native_audit'],
                symbolic_equivalence='not_checked',source_archive_index=best_index)
        else:report['skipped']='native_training_not_improved'
    report['seconds']=time.monotonic()-begin
    return out
