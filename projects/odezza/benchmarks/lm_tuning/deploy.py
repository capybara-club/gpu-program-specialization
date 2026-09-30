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
"""Freeze project sources and build an isolated worker on existing CUDA hosts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
from .common import REPO, save, file_hash, load
from .protocol import validate


def ssh(host, args, **kwargs):
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host, shlex.join(map(str, args))],
                          check=True, text=True, capture_output=True, **kwargs)


def lanes(hosts):
    return [(host, device, host + '_' + str(device)) for host in hosts for device in ([0, 1] if host == 'rack1' else [0])]


def python_for(host):
    return '/home/cdurham/odezza/.venv/bin/python' if host == 'rack1' else '/home/cdurham/odezza/scratch/structural_search_trial/.venv/bin/python'


def freeze(root):
    excluded = {'validation', 'runs', '.venv', 'bin', 'build', '__pycache__'}
    suffixes = {'.py', '.c', '.h', '.cu', '.cuh', '.exports', '.md', '.json'}
    files = set()
    for directory in ('core', 'app', 'benchmarks/lm_tuning', 'scratch/fitting_batch_trial'):
        for p in (REPO / directory).rglob('*'):
            relative = p.relative_to(REPO / directory)
            if any(part in excluded for part in relative.parts):
                continue
            if p.is_file() and (p.suffix in suffixes or p.name == 'Makefile'):
                # Research datasets are not execution dependencies.
                if directory == 'scratch/fitting_batch_trial' and p.suffix == '.json':
                    continue
                files.add(p)
    files.update(REPO / name for name in ('Makefile', 'ode_game.py', 'python/odezza/native.py',
                 'examples/grammar_game/depth3.json', 'examples/grammar_game/depth2.json',
                 'tests/o_lm_core_test.c', 'tests/o_lm_cuda_failures.c', 'tests/o_verify_public_exports.py',
                 'tests/o_lm_host_test.c', 'tests/o_lm_shape_validation.py', 'tests/test_native.py',
                 'tests/o_generate_scoring_cuda_test.c', 'tests/o_inspect_scoring_cubin_test.c',
                 'tests/o_nvrtc_compilation_test.c', 'tests/o_specialize_scoring_cubin_test.c',
                 'tests/o_scoring_pipeline_test.c', 'tests/o_scoring_template_test.c', 'tests/o_sass_test.c', 'tests/o_sass_corpus_test.c', 'tests/o_scoring_fixture.h'))
    hashes = {str(p.relative_to(REPO)): file_hash(p) for p in sorted(files)}
    save(root / 'source-hashes.json', hashes)
    with tarfile.open(root / 'source.tar.gz', 'w:gz') as archive:
        for path in sorted(files):
            archive.add(path, arcname=str(path.relative_to(REPO)), recursive=False)
    return hashes


def deploy_frozen(root, name, hosts, states):
    """Freeze/build an isolated execution tree; scheduling remains caller-owned."""
    freeze(root)
    remote = '/home/cdurham/odezza/scratch/' + name
    save(root / 'deployment.json', dict(remote=remote, hosts=hosts, lanes=lanes(hosts)))
    def host_build(host):
        directory = root / 'deployment' / host
        directory.mkdir(parents=True)
        ssh(host, ['mkdir', remote])
        subprocess.run(['scp', '-q', str(root / 'source.tar.gz'), host + ':' + remote + '/source.tar.gz'], check=True)
        ssh(host, ['tar', '-xzf', remote + '/source.tar.gz', '-C', remote])
        command = ['env', 'CUDA_MODULE_LOADING=EAGER', 'make', '-C', remote, '-j4', 'all', 'shared', 'test-lm-failures', 'test-public-api', 'test-c']
        result = ssh(host, command, timeout=180)
        (directory / 'core-build.log').write_text(result.stdout + result.stderr)
        result = ssh(host, ['make', '-C', remote + '/scratch/fitting_batch_trial', 'native', 'core-runner'], timeout=90)
        (directory / 'worker-build.log').write_text(result.stdout + result.stderr)
        preflight = ['env', 'CUDA_MODULE_LOADING=EAGER', python_for(host), '-B', '-c',
                     'import os,runpy,sys;os.chdir(' + repr(remote) + ');'
                     'sys.argv=["shape_preflight",*sys.argv[1:]];'
                     'runpy.run_module("benchmarks.lm_tuning.shape_preflight",run_name="__main__")',
                     '--states', *map(str, sorted(set(states) | {3})),
                     '--library', remote + '/build/libodezza.so', '--out', remote + '/shape-preflight.json']
        result = ssh(host, preflight, timeout=180)
        (directory / 'shape-preflight.log').write_text(result.stdout + result.stderr)
        subprocess.run(['scp', '-q', host + ':' + remote + '/shape-preflight.json',
                        str(directory / 'shape-preflight.json')], check=True)
        save(directory / 'status.json', {'status': 'built_and_native_tests_passed'})
        return host
    with ThreadPoolExecutor(max_workers=len(hosts)) as pool:
        for host in pool.map(host_build, hosts):
            print(json.dumps({'host': host, 'status': 'native_tests_passed'}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    protocol = validate(load(a.protocol))
    root = a.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    save(root / 'protocol.json', protocol)
    deploy_frozen(root, protocol['name'], protocol['hosts'], protocol['generation']['states'])


if __name__ == '__main__':
    main()
