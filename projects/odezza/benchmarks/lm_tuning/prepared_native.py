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
"""Large prepared banks through the public C99 LM ABI, with bounded buffers.

The service planner sees one row per candidate; the execution descriptor gets
the shared full bank. This avoids materializing millions of JSON candidate rows.
No CUDA source, optimization math, or specializer lives in this module.
"""
import math
import time
from .common import save
from .batching import batches


def run(adapter, spec, bank, options, directory, deadline):
    import refinement as rf
    from lm_toggle.service import prepare_groups, data_for
    from pipeline_runner.native_fit import DeviceFit
    from odezza.native import LmShape, LmPipeline
    from odezza.lm_model import LMKernelShape
    from lm_toggle.settings import native_shape_key
    v = rf.validate(spec)
    groups, fixed = prepare_groups(v, {'starts_per_candidate': 1, 'toggle_width': options['toggle_width']}, 'prepared-bank')
    if fixed:
        raise ValueError('Prepared fitting calibration requires at least one active coefficient')
    data = data_for(v)
    records = []
    profiles = []
    total_trials = 0
    complete = True
    for group in groups:
        if time.monotonic() >= deadline:
            complete = False
            break
        shape = LMKernelShape.from_dict(group['plan']['shape'])
        key = native_shape_key(shape, options)
        hit = key in adapter.handles
        start = time.monotonic()
        if not hit:
            adapter.handles[key] = LmPipeline(adapter.library, int(adapter.owner.gpu.arch.split('_')[1]), LmShape(*key[:4]), fallback_lanes=key[4])
        handle = adapter.handles[key]
        preparation = time.monotonic() - start
        active = group['active']
        starts = [[row[i] for i in active] + [0.] * (shape.optimized_constant_count - len(active)) for row in bank]
        packs = group['plan']['packs']
        for batch_index, batch in enumerate(batches(packs, len(starts), options)):
            if time.monotonic() >= deadline:
                complete = False
                break
            prepare_buffers = time.monotonic()
            selected = [dict(pack, starts=starts, fit_count=len(starts) * pack['permutation_count']) for pack in batch]
            if any(pack['fit_count'] > 1048576 for pack in selected):
                raise ValueError('Prepared pack exceeds one million fits')
            jobs = []
            try:
                for pack in selected:
                    job = DeviceFit(adapter.owner, pack, shape, dict(data, lower=group['lower'], upper=group['upper']),
                                    dict(v['refinement'], steps=v['execution']['steps_per_observation'],
                                         initial_damping=options['initial_damping'], damping_attempts=options['damping_attempts']))
                    jobs.append(job)
                    for buffer in job.inputs:
                        adapter.owner.driver.upload(buffer, adapter.transfer_stream)
                adapter.owner.driver.event_record(adapter.ready, adapter.transfer_stream)
                for job in jobs:
                    job.descriptor.input_ready_event = adapter.ready.value
                buffers_seconds = time.monotonic() - prepare_buffers
                native = handle.run([job.descriptor for job in jobs])
                download_started = time.monotonic()
                for job in jobs:
                    for buffer in job.outputs:
                        adapter.owner.driver.download(buffer, adapter.transfer_stream)
                adapter.owner.driver.event_record(adapter.ready, adapter.transfer_stream)
                adapter.owner.driver.event_wait(adapter.ready)
                profiles.append(dict(native, template_cache_hit=hit or batch_index > 0, module_cache_hit=False,
                    prepare_seconds=preparation if batch_index == 0 else 0., state_count=shape.state_count,
                    optimized_parameter_count=len(active), compiled_parameter_count=shape.optimized_constant_count,
                    bank_rows=len(bank), execution_path='prepared_public_c99', requested_lanes_per_fit=key[3], cta_threads=32,
                    buffer_prepare_upload_seconds=buffers_seconds,
                    download_fence_seconds=time.monotonic()-download_started,
                    submitted_packs=len(jobs), pack_fit_counts=[p['fit_count'] for p in selected],
                    submission_packs_requested=options.get('submission_packs',2),
                    buffer_fit_limit=options.get('buffer_fit_limit',4194304)))
                for pack, job in zip(selected, jobs):
                    coefficients, initial, mse, iterations, accepted, factors, evaluations, invalid = [buffer[2] for buffer in job.outputs]
                    total_trials += sum(evaluations)
                    for permutation, candidate in enumerate(pack['candidates']):
                        indices = range(permutation, pack['fit_count'], pack['permutation_count'])
                        finite = (i for i in indices if math.isfinite(mse[i]) and 0 <= mse[i] < 3e38)
                        best = min(finite, key=lambda i: (mse[i], i), default=None)
                        if best is None:
                            continue
                        for item_index in candidate['tags']['item_indices']:
                            item = v['items'][item_index]
                            values = item['rows'][0][:]
                            for compact, original in enumerate(active):
                                values[original] = coefficients[best * shape.optimized_constant_count + compact]
                            records.append(dict(candidate_id=item['id'], program_hex=item['program'].hex(), parameter_values=values,
                                mse=mse[best], initial_mse=initial[best], iterations=iterations[best],
                                start_index=best // pack['permutation_count'], permutation=permutation,
                                factorizations=sum(factors[i] for i in indices), evaluations=sum(evaluations[i] for i in indices),
                                invalid=sum(invalid[i] for i in indices)))
            except Exception:
                if not handle.fatal:
                    try:
                        adapter.owner.driver.event_record(adapter.ready, adapter.transfer_stream)
                        adapter.owner.driver.event_wait(adapter.ready)
                    except Exception:
                        for job in jobs:
                            job.fatal = True
                        adapter.pending.extend(jobs)
                        raise
                else:
                    for job in jobs:
                        job.fatal = True
                    adapter.pending.extend(jobs)
                raise
            finally:
                for job in jobs:
                    job.close()
    report = dict(status='complete', work_complete=complete, candidates=records,
                  counts={'coefficient_trials': total_trials, 'starts': sum(p['fit_count'] for p in profiles)},
                  lm_profile=profiles, timing={'native_run_seconds': sum(p['total_seconds'] for p in profiles)},
                  execution_path='prepared_public_c99', requested_fits=len(bank) * len(v['candidates']))
    save(directory / 'report.json', report)
    return report
