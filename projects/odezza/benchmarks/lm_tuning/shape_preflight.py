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
"""Check native template creation before admitting a timed campaign.

The current fitting adapter pads to three parameters, admits up to eight, and
uses the default 256-instruction patch capacity. This checks every such shape
for the requested state counts, including coefficients introduced by search.
Creation success does not guarantee that every expression can specialize.
"""
import argparse
from pathlib import Path
import sys
import time
from .common import REPO, save, file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--states', type=int, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--lanes-per-fit', type=int, choices=(1,2,4,8), default=1)
    parser.add_argument('--exact-shape', action='store_true', help='Disable the native adapter default of wider fallback')
    args = parser.parse_args()
    sys.path.insert(0, str(REPO / 'scratch/fitting_batch_trial'))
    import bootstrap
    from pipeline_runner.cuda import CudaOwner
    from odezza.native import LmPipeline, LmShape, NativeError
    owner = CudaOwner(0)
    rows = []
    report = dict(scope='template_creation_only', library_sha256=file_hash(args.library),
                  arch=owner.gpu.arch, status='running', shapes=rows)
    try:
        for states in sorted(set(args.states)):
            for parameters in range(3, 9):
                started = time.monotonic()
                row = dict(states=states, parameters=parameters, patch_capacity=256,
                           lanes_per_fit=args.lanes_per_fit, shape_fallback=not args.exact_shape)
                try:
                    with LmPipeline(args.library, int(owner.gpu.arch.split('_')[1]),
                                    LmShape(states, parameters, 256, args.lanes_per_fit),
                                    fallback_lanes=() if args.exact_shape else tuple(w for w in (2,4,8) if w>args.lanes_per_fit)) as pipeline:
                        row['shapes'] = pipeline.shape_report()
                    row['status'] = 'created'
                except NativeError as error:
                    row.update(status='unsupported_resource' if error.result in (7, 11, 12)
                               else 'error', native_result=error.result, error=str(error))
                row['seconds'] = time.monotonic() - started
                rows.append(row)
                save(args.out, report)
                if row['status'] == 'error':
                    report['status'] = 'failed'
                    save(args.out, report)
                    raise RuntimeError('Unexpected native shape failure; inspect ' + str(args.out))
        report['status'] = 'passed'
        save(args.out, report)
        print('Template preflight:', len(rows), 'shapes;',
              sum(r['status'] == 'unsupported_resource' for r in rows), 'resource rejections')
    finally:
        owner.close()


if __name__ == '__main__':
    main()
