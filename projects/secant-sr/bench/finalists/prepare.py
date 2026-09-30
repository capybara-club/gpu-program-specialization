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
"""Prepare fresh official splits from already-present, hash-verified data."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import sklearn
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'python'))
from srbench_data import prepare_srbench_v2_groundtruth_dataset,HEADER
from srbench_v2 import SRBENCH_V2_SEEDS

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();old=json.loads(a.previous.read_text());seeds=[860,5390]
    if set(seeds)&{j['seed'] for j in old['jobs']} or not set(seeds)<=set(SRBENCH_V2_SEEDS):
        raise ValueError('expected fresh official seeds')
    if (a.output/'manifest.json').exists():raise ValueError('manifest already exists')
    a.output.mkdir(parents=True,exist_ok=True);sources={j['problem']:j for j in old['jobs']};jobs=[]
    for name,j in sorted(sources.items()):
        source=Path(j['source_path'])
        if hashlib.sha256(source.read_bytes()).hexdigest()!=j['source_sha256']:
            raise ValueError('original dataset hash mismatch')
        for seed in seeds:
            item=prepare_srbench_v2_groundtruth_dataset(source,a.output/'data',seed,0)
            raw=item.path.read_bytes();_,_,inputs,train,test=HEADER.unpack_from(raw)
            values=np.frombuffer(raw,dtype='<f4',offset=HEADER.size)
            if not np.isfinite(values).all():raise ValueError('nonfinite data')
            record=asdict(item);record.update(path=str(item.path.relative_to(a.output)),source_path=str(source),
                problem=name,seed=seed,id=f'{name}-s{seed}',prepared_sha256=hashlib.sha256(raw).hexdigest(),
                train_variance=float(np.var(values[inputs*train:(inputs+1)*train].astype(np.float64))),
                test_variance=float(np.var(values[-test:].astype(np.float64))))
            jobs.append(record)
    output=dict(old,jobs=jobs,problems=sorted(sources),seeds=seeds,
        selection='Same 21 diagnostic problems, fresh official seeds 860 and 5390; outcomes not yet observed',
        dependencies=dict(numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__))
    (a.output/'manifest.json').write_text(json.dumps(output,indent=2)+'\n')
    print('Prepared',len(jobs),'fresh cases',flush=True)

if __name__=='__main__':main()
