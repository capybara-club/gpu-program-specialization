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
"""Queue/protocol regression tests against mac3, with GPU workers stopped.

Synthetic reports below test service ownership, not CUDA numerical correctness.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import uuid
from client import Client, CHUNK


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--port', type=int, default=14222)
    args = parser.parse_args()
    c = Client(port=args.port)
    checks = []
    handles = []
    base = (Path(__file__).resolve().parents[1] / 'examples/grammar/submit.json').read_bytes()

    def expect_error(op, data=b'', code=None):
        try:
            c.rpc(op, data)
        except RuntimeError as e:
            if code:
                assert str(e) == code, str(e)
            return
        raise AssertionError('expected failure: ' + op)

    def submit(raw=base):
        r = c.submit(raw)
        handles.append(r['handle'])
        return r['handle']

    try:
        # Multi-megabyte raw request arrives byte-for-byte; no Python AST conversion.
        raw = base + b' ' * (1300 * 1024)
        h = submit(raw)
        generation, job = h.split('.')
        same = c.submit(raw, job_id=job, generation=generation)
        assert same['state'] == 'queued'
        expect_error('put.' + h + '.0', b'!', 'idempotency_conflict')
        attempt, worker = uuid.uuid4().hex, uuid.uuid4().hex
        c.rpc(f'claim.{h}.{attempt}.{worker}')
        got = bytearray()
        while True:
            part = c.rpc(f'request.{h}.{attempt}.{len(got)}')
            got.extend(part)
            if len(part) < CHUNK:
                break
        assert bytes(got) == raw
        checks.append('chunked_1.3MiB_request_exact_bytes_and_idempotency')
        expect_error(f'report.{h}.{uuid.uuid4().hex}.0', b'{}', 'stale_attempt_or_bad_operation')
        expect_error(f'claim.{h}.{uuid.uuid4().hex}.{worker}', code='attempt_busy')
        checks.append('active_attempt_exclusion_and_stale_write_rejection')
        report = b'{"status":"complete","test_only":true}' + b' ' * (2 * 1024 * 1024)
        status_times = []

        def upload_report():
            other = Client(port=args.port)
            try:
                for start in range(0, len(report), CHUNK):
                    other.rpc(f'report.{h}.{attempt}.{start}', report[start:start + CHUNK])
                other.rpc(f'commit.{h}.{attempt}.{len(report)}')
            finally:
                other.close()

        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(upload_report)
            while not future.done():
                start = time.perf_counter()
                c.status(h)
                status_times.append(time.perf_counter() - start)
            future.result()
        assert max(status_times, default=0) < 1
        assert c.result(h) == {'status': 'complete', 'test_only': True}
        c.rpc('release.' + h)
        expect_error(f'begin.{h}.{len(raw)}', code='released_job')
        checks.append('chunked_report_commit_immediate_release_and_control_concurrency')
        h = submit()
        cancelled = json.loads(c.rpc('cancel.' + h))
        assert cancelled['state'] == 'terminal'
        assert c.result(h)['status'] == 'cancelled'
        c.rpc('release.' + h)
        checks.append('queued_cancel_without_execution')
        for execution, grammar, expected in ((1800, None, 1800), (1800, 17, 17)):
            request = json.loads(base)
            request['execution']['max_seconds'] = execution
            if grammar is not None:
                request['grammar']['limits'] = {'max_seconds': grammar}
            h = submit(json.dumps(request).encode())
            assert c.status(h)['execution_budget_seconds'] == expected
            assert any(row['job_id'] == h.split('.')[1] for row in json.loads(c.rpc('jobs'))['jobs'])
            c.rpc('cancel.' + h)
            c.rpc('release.' + h)
        too_long = json.loads(base)
        too_long['execution']['max_seconds'] = 3601
        h = submit(json.dumps(too_long).encode())
        assert c.result(h)['error'] == 'invalid_envelope_or_budget_above_3600s'
        c.rpc('release.' + h)
        checks.append('effective_budgets_above_600s_smaller_grammar_limit_and_explicit_ceiling')
        limits = c.health()['integration_limits']
        assert limits == dict(points_per_configuration=65536, steps_per_observation=4096, steps_per_configuration=65536,
            base_work_units_per_tile=67108864,maximum_parallel_target_configurations=4194304,tile_policy='parallel-work-v1')
        for intervals, dt, rejected in ((16, 1/4096, False), (17, 1/4096, True), (1, 1/8192, True), (1, 1e-30, True), (65536, 1, True)):
            request = json.loads(base)
            request['problem']['trajectories'] = [dict(initial=[1,2], times=list(range(intervals+1)), values=[[1,2]]+[[0,0]]*intervals)]
            request['grammar']['integration']['dt'] = dt
            h = submit(json.dumps(request).encode())
            status = c.status(h)
            assert status['attempts'] == 0
            if rejected:
                assert status['state'] == 'terminal'
                report = c.result(h)
                assert report['error'] == 'integration_work_limit', report
                assert report['completed_configurations'] == 0
            else:
                assert status['state'] == 'queued'
                assert status['integration_work'] == dict(steps_per_observation=4096, steps_per_configuration=65536, trajectory_points=17, trajectory_count=1)
                c.rpc('cancel.'+h)
            c.rpc('release.'+h)
        checks.append('integration_work_bounds_exact_boundary_reject_before_claim_and_slot_reuse')
        # The ninth admission must fail without displacing any of eight jobs.
        hs = [submit() for _ in range(8)]
        try:
            submit()
        except RuntimeError as e:
            assert str(e) == 'admission_full'
        else:
            raise AssertionError('unbounded admission')
        for h in hs:
            assert c.status(h)['state'] == 'queued'
            c.rpc('cancel.' + h)
            c.rpc('release.' + h)
        checks.append('eight_slot_backpressure_without_dropping_accepted_jobs')
        # A disappeared claimant loses its write authority. New attempt counted.
        h = submit()
        old, new = uuid.uuid4().hex, uuid.uuid4().hex
        c.rpc(f'claim.{h}.{old}.{worker}')
        deadline = time.monotonic() + 20
        while c.status(h)['state'] != 'queued':
            assert time.monotonic() < deadline
            time.sleep(.25)
        c.rpc(f'claim.{h}.{new}.{worker}')
        expect_error(f'report.{h}.{old}.0', b'{}', 'stale_attempt_or_bad_operation')
        assert c.status(h)['attempts'] == 2
        fake = b'{"status":"failed","test_only":true}'
        c.rpc(f'report.{h}.{new}.0', fake)
        c.rpc(f'commit.{h}.{new}.{len(fake)}')
        c.rpc('release.' + h)
        checks.append('lease_expiry_replacement_and_old_attempt_rejection')
        result = dict(passed=True, checks=checks, largest_status_seconds=max(status_times, default=0),
                      evidence='real NATS broker and C service; synthetic worker claims/reports; no CUDA work')
        Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result))
    finally:
        for h in handles:
            try:
                c.rpc('cancel.' + h)
                c.rpc('release.' + h)
            except RuntimeError:
                pass
        c.close()


if __name__ == '__main__':
    main()
