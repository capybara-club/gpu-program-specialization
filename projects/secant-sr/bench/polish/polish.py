#!/usr/bin/env python3
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
"""Fixed-structure, training-only CPU least-squares diagnostic for retained winners."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import time

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'python'))
from secant_sr_ast import Expression, BINARY, UNARY


def prepare(genome, constants, permutation, mode, fitted_leaves=None):
    """Parameter sharing is explicit; unchosen toggle branches are absent."""
    instructions, centers, keys = [], [], {}
    # Since 0.3.3, zero-scale affine instructions use slot zero. Persistent
    # fitting identities belong to the host metadata, not those runtime slots.
    identities = {}
    if fitted_leaves is not None:
        metadata = {(v['node'], v['alternative']): v for v in fitted_leaves}
        logical = [0]
        consumed = set()
        def index(node):
            if node.op not in {'toggle2', 'toggle4', 'input', 'constant', 'bank', 'affine_bank'}:
                for child in node.args:
                    index(child)
            choices = node.args if node.op in {'toggle2', 'toggle4'} else (node,)
            for choice, leaf in enumerate(choices):
                key = (logical[0], choice)
                if leaf.op == 'affine_bank':
                    value = metadata.get(key)
                    if value is None or struct.pack('<f', value['value']) != struct.pack('<f', leaf.value[2]):
                        raise ValueError('missing or inconsistent fitted parameter metadata')
                    identities[id(leaf)] = value['slot']
                    consumed.add(key)
            logical[0] += 1
        index(genome)
        if consumed != set(metadata):
            raise ValueError('unmatched fitted parameter metadata')
    def visit(node):
        if node.op in {'toggle2', 'toggle4'}:
            low = node.value if node.op == 'toggle2' else node.value[0]
            choice = (permutation >> low) & 1
            if node.op == 'toggle4':
                choice |= ((permutation >> node.value[1]) & 1) << 1
            return visit(node.args[choice])
        if node.op in {'constant', 'bank', 'affine_bank'}:
            if node.op == 'constant':
                value, key = node.value, None
            elif node.op == 'bank':
                value, key = constants[node.value], ('bank', node.value)
            else:
                slot, scale, offset = node.value
                if scale != 0:
                    raise ValueError('retained fitted genome must have zero affine scale')
                value, key = offset, ('fitted', identities.get(id(node), slot), struct.pack('<f', offset).hex())
            if mode == 'all_literals':
                key = ('occurrence', len(instructions))
            if key is not None:
                if key not in keys:
                    keys[key] = len(centers)
                    centers.append(value)
                instructions.append(('parameter', keys[key]))
            else:
                instructions.append(('constant', value))
        elif node.op == 'input':
            instructions.append(('input', node.value))
        else:
            for child in node.args:
                visit(child)
            instructions.append((node.op, None))
    if mode not in {'tied', 'all_literals'}:
        raise ValueError('unknown parameter policy')
    visit(genome)
    return instructions, np.asarray(centers, dtype=np.float64)


def evaluate(code, x, parameters, jacobian=False, dtype=np.float64):
    n, p = len(x), len(parameters)
    stack = []
    def leaf(value, slot=None):
        values = np.full(n, value, dtype=dtype) if np.ndim(value) == 0 else np.asarray(value, dtype=dtype)
        derivative = np.zeros((n, p)) if jacobian else None
        if jacobian and slot is not None:
            derivative[:, slot] = 1
        return values, derivative
    with np.errstate(all='ignore'):
        for op, value in code:
            if op == 'input':
                stack.append(leaf(x[:, value]))
                continue
            if op == 'parameter':
                stack.append(leaf(parameters[value], value))
                continue
            if op == 'constant':
                stack.append(leaf(value))
                continue
            b, db = stack.pop()
            if op in BINARY:
                a, da = stack.pop()
                if op == 'add': out, deriv = a+b, (da+db if jacobian else None)
                elif op == 'sub': out, deriv = a-b, (da-db if jacobian else None)
                elif op == 'mul': out, deriv = a*b, (da*b[:, None]+db*a[:, None] if jacobian else None)
                elif op == 'div': out, deriv = a/b, ((da*b[:, None]-db*a[:, None])/(b*b)[:, None] if jacobian else None)
                elif op in {'min', 'max'}:
                    choose = a <= b if op == 'min' else a >= b
                    out, deriv = np.where(choose, a, b), (np.where(choose[:, None], da, db) if jacobian else None)
                else: raise ValueError(op)
            else:
                functions = {'neg':np.negative, 'sqrt':np.sqrt, 'sin':np.sin, 'cos':np.cos,
                             'exp':np.exp, 'log':np.log, 'exp2':np.exp2, 'log2':np.log2,
                             'tanh':np.tanh, 'abs':np.abs, 'rcp':lambda v:1/v,
                             'rsqrt':lambda v:1/np.sqrt(v)}
                out = functions[op](b)
                deriv = None
                if jacobian:
                    if op == 'neg': slope = -np.ones(n)
                    elif op == 'sqrt': slope = np.divide(.5, out, out=np.zeros(n), where=out != 0)
                    elif op == 'sin': slope = np.cos(b)
                    elif op == 'cos': slope = -np.sin(b)
                    elif op == 'exp': slope = out
                    elif op == 'log': slope = 1/b
                    elif op == 'exp2': slope = out*np.log(2)
                    elif op == 'log2': slope = 1/(b*np.log(2))
                    elif op == 'tanh': slope = 1-out*out
                    elif op == 'abs': slope = np.sign(b)
                    elif op == 'rcp': slope = -1/(b*b)
                    elif op == 'rsqrt': slope = -.5/(b*np.sqrt(b))
                    else: raise ValueError(op)
                    deriv = db*slope[:, None]
                    deriv[db == 0] = 0  # Constant subtrees need no derivative at a singularity.
            stack.append((out, deriv))
    if len(stack) != 1:
        raise ValueError('invalid program')
    return stack[0]


def materialize(code, parameters):
    stack = []
    for op, value in code:
        if op == 'input': node = Expression.input(value)
        elif op in {'constant', 'parameter'}:
            node = Expression.constant(parameters[value] if op == 'parameter' else value)
        else:
            n = 2 if op in BINARY else 1
            args = tuple(stack[-n:]); del stack[-n:]
            node = Expression(op, args)
        stack.append(node)
    return stack[0]


def mse(code, x, y, parameters):
    predictions, _ = evaluate(code, x, parameters, dtype=np.float32)
    if not np.isfinite(predictions).all():
        return math.inf
    residual = predictions.astype(np.float64)-y
    return float(np.mean(residual*residual))


def fit(code, center, x, y, seed, seconds, starts=8, evaluations=200):
    started = time.monotonic()
    best, best_mse = center.copy(), mse(code, x, y, center)
    initial_mse = best_mse
    trace = []
    if not len(center):
        return dict(parameters=best.tolist(), initial_mse=initial_mse, train_mse=best_mse, starts=trace, seconds=0)
    scale = max(float(np.std(y)), 1e-30)
    rng = np.random.default_rng(seed)
    class Deadline(Exception): pass
    cache = {}
    def compute(theta):
        nonlocal best, best_mse
        if time.monotonic()-started > seconds:
            raise Deadline()
        if 'theta' not in cache or not np.array_equal(theta, cache['theta']):
            v, j = evaluate(code, x, theta, jacobian=True)
            invalid = ~np.isfinite(v) | ~np.isfinite(j).all(axis=1)
            residual = np.clip((v-y)/scale, -1e6, 1e6)
            residual[invalid] = 1e6
            j = np.clip(j/scale, -1e100, 1e100)
            j[invalid] = 0
            cache.update(theta=theta.copy(), residual=residual, jacobian=j)
            if not invalid.any() and np.mean((v-y)**2) < best_mse:
                with np.errstate(all='ignore'):
                    quantized = theta.astype(np.float32).astype(np.float64)
                value = mse(code, x, y, quantized) if np.isfinite(quantized).all() else math.inf
                if value < best_mse:
                    best, best_mse = quantized.copy(), value
        return cache
    for i in range(starts):
        theta = center.copy() if not i else center+rng.normal(size=len(center))*np.maximum(1, np.abs(center))*[.01,.1,.5,1][(i-1)%4]
        try:
            result = least_squares(lambda t:compute(t)['residual'], theta,
                                   jac=lambda t:compute(t)['jacobian'], method='lm',
                                   max_nfev=evaluations, x_scale='jac', ftol=1e-10, xtol=1e-10, gtol=1e-10)
            with np.errstate(all='ignore'):
                quantized = result.x.astype(np.float32).astype(np.float64)
            score = mse(code, x, y, quantized) if np.isfinite(quantized).all() else math.inf
            if score < best_mse:
                best, best_mse = quantized, score
            trace.append(dict(start=i, status=int(result.status), nfev=result.nfev, train_mse=score if math.isfinite(score) else None))
        except Deadline:
            trace.append(dict(start=i, status='time_budget'))
            break
        except (ValueError, FloatingPointError) as error:
            trace.append(dict(start=i, status=type(error).__name__))
    return dict(parameters=best.tolist(), initial_mse=initial_mse, train_mse=best_mse,
                starts=trace, seconds=time.monotonic()-started)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=15)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(a.manifest.read_text())
    result = dict(status='running', started_unix=time.time(), manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), seconds_per_mode=a.seconds,
                  protocol='train-only f64 LM; original structure/operators fixed; final f32 coefficient quantization and f32 prediction scoring', cases=[])
    def save():
        (a.output/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    save()
    try:
        for item in manifest['jobs']:
            job, source = item['job'], item['source']['native_result']
            raw = (a.data/job['path']).read_bytes()
            if hashlib.sha256(raw).hexdigest() != job['prepared_sha256']:
                raise ValueError('dataset hash mismatch')
            magic, version, inputs, train, test = struct.unpack_from('<8sIIQQ', raw)
            if (magic,version,inputs,train,test) != (b'SECSRDS\0',1,job['num_inputs'],job['num_train_rows'],job['num_validation_rows']):
                raise ValueError('dataset dimensions')
            values = np.frombuffer(raw, dtype='<f4', offset=32)
            split = train*(inputs+1)
            x = values[:train*inputs].reshape(inputs,train).T
            y = values[train*inputs:split]
            vx = values[split:split+test*inputs].reshape(inputs,test).T
            vy = values[split+test*inputs:]
            if len(vy) != test: raise ValueError('dataset length')
            genome = Expression.decode(bytes.fromhex(source['genotype_hex']))
            row = dict(id=job['id'], original_validation_r2=item['source']['validation_r2'], modes={})
            for mode in ['tied', 'all_literals']:
                code, center = prepare(genome, source['coefficients'], source['permutation'], mode, source.get('fitted_leaves'))
                original = materialize(code,center).encode().hex()
                if original != source['resolved_ast_hex']: raise ValueError('coefficient identity mismatch')
                fitted = fit(code, center, x, y, job['seed'], a.seconds)
                # Holdout is accessed only after training selects the final parameters.
                validation = mse(code, vx, vy, fitted['parameters'])
                baseline = source['train_mse']
                gap = abs(math.sqrt(fitted['initial_mse'])-math.sqrt(baseline))/max(1,math.sqrt(float(np.mean(y.astype(np.float64)**2))),math.sqrt(baseline))
                if gap > 2e-5: raise ValueError('baseline numerical audit failed')
                fitted.update(parameter_count=len(center), baseline_relative_rmse_gap=gap,
                              validation_mse=validation if math.isfinite(validation) else None,
                              validation_r2=1-validation/job['test_variance'] if math.isfinite(validation) else None,
                              recovered=math.isfinite(validation) and 1-validation/job['test_variance']>.999,
                              resolved_ast_hex=materialize(code,fitted['parameters']).encode().hex())
                row['modes'][mode]=fitted
                print(job['id'],mode,'parameters',len(center),'test R2',fitted['validation_r2'],'seconds',fitted['seconds'],flush=True)
            result['cases'].append(row); save()
        result['status']='complete'
    except BaseException as error:
        result.update(status='failed', error=str(error))
        raise
    finally:
        result['finished_unix']=time.time();save()


if __name__ == '__main__':
    main()
