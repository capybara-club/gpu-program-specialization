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
from pathlib import Path


BACKENDS = ("ptx", "cuda", "cubin")


def select_backends(available, requested=None):
    ordered_available = [
        backend for backend in BACKENDS
        if backend in available
    ]
    if requested is None:
        return ordered_available

    unknown = [
        backend for backend in requested
        if backend not in BACKENDS
    ]
    if unknown:
        raise RuntimeError(f"unknown backend: {unknown[0]}")

    missing = [
        backend for backend in requested
        if backend not in ordered_available
    ]
    if missing:
        raise RuntimeError(f"CSV inputs contain no {missing[0]} data")

    return [
        backend for backend in BACKENDS
        if backend in requested
    ]


def backend_csv_path(output, backend):
    output = Path(output)
    suffix = output.suffix or ".csv"
    stem = output.stem if output.suffix else output.name
    return output.with_name(f"{stem}_{backend}{suffix}")


def backend_opt_levels(backend, opt_levels):
    if backend == "cubin":
        return tuple(level for level in opt_levels if level == 1)
    return tuple(opt_levels)
