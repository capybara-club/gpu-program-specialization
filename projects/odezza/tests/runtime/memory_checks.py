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
"""Native memory budgets, pool reuse and multi-GPU accounting; no search controller."""
import argparse
from copy import deepcopy
import ctypes as C
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
P = C.c_void_p


class Runtime:
    def __init__(self, devices=(0,), pooled=True, attempt=True, library=None, host_limit=None):
        self.lib = C.CDLL(str(library or ROOT/'build/runtime/libodezza_runtime.so'))
        l = self.lib
        l.odz_runtime_create_devices.argtypes = [C.POINTER(C.c_uint), C.c_size_t, C.c_char_p, C.POINTER(P)]
        l.odz_runtime_attempt_inputs.argtypes = [P]
        l.odz_runtime_reserve_buffers.argtypes = [P, C.c_size_t, C.c_size_t, C.c_size_t]
        l.odz_runtime_destroy.argtypes = [P]
        l.odz_job_create.argtypes = [C.c_char_p, C.c_size_t, C.POINTER(P)]
        l.odz_job_run.argtypes = [P, P]
        l.odz_job_report.argtypes = [P, P, C.c_size_t, C.POINTER(C.c_size_t)]
        l.odz_job_status.argtypes = l.odz_job_report.argtypes
        l.odz_job_destroy.argtypes = [P]
        self.handle = P()
        assert not l.odz_runtime_create_devices((C.c_uint*len(devices))(*devices), len(devices),
                                                b'/tmp/odezza-service-trial-cache', C.byref(self.handle))
        if attempt: assert not l.odz_runtime_attempt_inputs(self.handle)
        if pooled: assert not l.odz_runtime_reserve_buffers(self.handle, 64<<20, 256<<20, 64<<20)
        if host_limit is not None:
            l.odz_runtime_host_limit.argtypes = [P, C.c_size_t]
            assert not l.odz_runtime_host_limit(self.handle, host_limit)

    def run(self, request):
        raw = json.dumps(request, separators=(',', ':')).encode()
        job = P()
        assert not self.lib.odz_job_create(raw, len(raw), C.byref(job))
        start = time.perf_counter()
        try:
            self.lib.odz_job_run(self.handle, job)
            size = C.c_size_t()
            assert not self.lib.odz_job_report(job, None, 0, C.byref(size))
            out = C.create_string_buffer(size.value)
            assert not self.lib.odz_job_report(job, out, len(out), C.byref(size))
            report = json.loads(out.value)
            # Sizing/reporting is repeatable and cannot grow retained storage.
            assert not self.lib.odz_job_report(job, out, len(out), C.byref(size))
            assert json.loads(out.value) == report
            return dict(elapsed_seconds=time.perf_counter()-start, report=report)
        finally:
            self.lib.odz_job_destroy(job)

    def close(self):
        self.lib.odz_runtime_destroy(self.handle)


def problem():
    return dict(problem=dict(states=['x', 'y'], known_rhs={'y': '0'}, trajectories=[
        dict(initial=[0, 0], times=[0, .125], values=[[0, 0], [0, 0]])]),
        grammar=dict(version=1, states=['x', 'y'], rhs={'x': 'rng.a*x+rng.b*y'},
            integration=dict(method='rk4', dt=.125),
            rng_banks={name: dict(base='uniform01', count=65536, seed=i) for i, name in enumerate(('a', 'b'))},
            rng={name: dict(bank=name, axis='rows') for name in ('a', 'b')},
            limits=dict(max_skeletons=1, max_variants=1, max_configurations=4),
            retain={'global': {'k': 1, 'unit': 'variant'}}),
        execution=dict(batch_variants=1, module_systems=4, dedup_bytes_per_family=1<<20))


def checked(record, success=True):
    r = record['report'];m = r['memory']
    assert r['status'] == ('complete' if success else 'failed'), r.get('error')
    assert not r['runtime_quarantined'], r
    assert m['charged_live_bytes'] <= m['charged_peak_bytes'] <= m['host_budget_bytes'], m
    assert sum(c['live_bytes'] for c in m['classes'].values()) == m['charged_live_bytes']
    assert m['classes']['tile_scratch']['live_bytes'] == 0
    assert m['classes']['pipeline_workspace']['live_bytes'] == 0
    if success:
        assert m['classes']['pipeline_workspace']['peak_bytes'] > 0
        assert m['rss_sample_valid'] and m['process_rss_bytes'] > 0
        for d in r['execution']['devices']:
            a=d['memory']
            assert a['allocated_device_bytes'] <= a['peak_allocated_device_bytes']
            assert a['device_sample_valid'] and a['device_used_bytes'] <= a['device_total_bytes']
    return r


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--benchmark-only',action='store_true');p.add_argument('--library')
    p.add_argument('--skip-benchmark',action='store_true')
    a=p.parse_args();records=[]
    def save(name, rec):
        records.append(dict(name=name, **rec))
        Path(a.output).write_text(json.dumps(dict(passed=False, records=records), indent=2)+'\n')
        print(name, rec['report']['status'], rec['elapsed_seconds'], flush=True)
    if not a.benchmark_only:
        for attempt in (False, True):
            runtime=Runtime(attempt=attempt)
            try:
                warm=runtime.run(problem());checked(warm);save(f'warm_pools_{attempt}',warm)
                smaller=problem();smaller['grammar']['rhs']['x']='rng.b*x'
                smaller['execution']['max_bank_bytes']=65536*4
                result=runtime.run(smaller);r=checked(result);save(f'smaller_pool_limit_{attempt}',result)
                assert r['cache']['numeric_pool_evictions'] >= 1
                assert r['execution']['devices'][0]['memory']['numeric_pool_bytes'] <= 65536*4
                if not attempt: assert r['timing']['rng_seconds'] == 0, 'expected retained bank cache hit'
                budget=deepcopy(smaller)
                # Enough for parsed arenas, insufficient for the newly charged pipeline workspace.
                m=r['memory'];budget['execution']['max_host_bytes']=sum(
                    m['classes'][k]['peak_bytes'] for k in ('request','arenas','retained'))
                failed=runtime.run(budget);f=checked(failed,False);save(f'workspace_budget_failure_{attempt}',failed)
                assert f['memory']['denied_reservations'] >= 1
                recovery=runtime.run(smaller);rr=checked(recovery);save(f'after_budget_failure_{attempt}',recovery)
                assert rr['candidates']==r['candidates'] and rr['leaderboards']==r['leaderboards']
                no_bank=deepcopy(smaller);no_bank['grammar']['rhs']['x']='x'
                no_bank['execution']['max_bank_bytes']=1
                record=runtime.run(no_bank);r=checked(record);save(f'no_slots_trim_{attempt}',record)
                assert r['execution']['devices'][0]['memory']['numeric_pool_bytes']==0
            finally:runtime.close()
        runtime=Runtime(host_limit=2<<30)
        try:
            req=problem();req['execution']['max_host_bytes']=(2<<30)+1
            record=runtime.run(req);r=checked(record,False);save('operator_host_ceiling',record)
            assert 'worker host ceiling' in r['error'] and r['counts']['completed_configurations']==0
            record=runtime.run(problem());checked(record);save('after_operator_rejection',record)
        finally:runtime.close()
        runtime=Runtime(pooled=False)
        try:
            large=problem();large['grammar']['limits']['max_configurations']=65536
            record=runtime.run(large);r=checked(record);save('unpooled_large',record)
            large_bytes=r['execution']['devices'][0]['memory']['allocated_device_bytes']
            small=problem();small['execution']['max_device_bytes']=4096
            record=runtime.run(small);r=checked(record);save('unpooled_small_budget',record)
            small_bytes=r['execution']['devices'][0]['memory']['allocated_device_bytes']
            assert small_bytes<large_bytes-256*1024,(small_bytes,large_bytes)
            fresh=Runtime(pooled=False)
            try:
                reference=fresh.run(small);ref=checked(reference);save('unpooled_small_fresh',reference)
                assert r['candidates']==ref['candidates'] and r['leaderboards']==ref['leaderboards']
                assert small_bytes==ref['execution']['devices'][0]['memory']['allocated_device_bytes']
            finally:fresh.close()
        finally:runtime.close()
    if a.skip_benchmark:
        Path(a.output).write_text(json.dumps(dict(passed=True, records=records), indent=2)+'\n')
        return
    runtime=Runtime(devices=(0,1),library=a.library)
    try:
        req=json.loads((ROOT/'service_trial/validation/million-2048-request.json').read_text())
        for i in range(4):
            rec=runtime.run(req);r=rec['report']
            if not a.library: checked(rec)
            assert r['status']=='complete' and r['counts']['completed_configurations']==2048000000
            assert all(d['completed_configurations']>0 for d in r['execution']['devices'])
            save(f'million_dual_{i}',rec)
    finally:runtime.close()
    Path(a.output).write_text(json.dumps(dict(passed=True, records=records), indent=2)+'\n')


if __name__=='__main__':main()
