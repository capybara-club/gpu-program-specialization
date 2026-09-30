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
"""Native implementation modules for FusedSINDy."""

from __future__ import annotations

import ctypes
import importlib
import os
import sys
from pathlib import Path


def _candidate_cuda_lib_dirs():
    roots = []
    package_path = Path(__file__).resolve()
    if len(package_path.parents) >= 3:
        roots.append(package_path.parents[2])
    for entry in sys.path:
        if entry:
            roots.append(Path(entry))

    seen = set()
    for root in roots:
        try:
            root = root.resolve()
        except OSError:
            continue
        if root in seen:
            continue
        seen.add(root)
        for name in ("cu13", "cu12"):
            lib_dir = root / "nvidia" / name / "lib"
            if lib_dir.is_dir():
                yield lib_dir


def _preload_cuda_jit_libraries() -> None:
    mode = getattr(os, "RTLD_NOW", 0) | getattr(ctypes, "RTLD_GLOBAL", 0)
    for lib_dir in _candidate_cuda_lib_dirs():
        for major in ("13", "12"):
            nvjitlink = lib_dir / f"libnvJitLink.so.{major}"
            nvrtc = lib_dir / f"libnvrtc.so.{major}"
            if nvjitlink.exists() and nvrtc.exists():
                ctypes.CDLL(str(nvjitlink), mode=mode)
                ctypes.CDLL(str(nvrtc), mode=mode)
                return


def _load_native():
    _preload_cuda_jit_libraries()
    try:
        return importlib.import_module(__name__ + "._implicit_sindy")
    except ImportError as exc:
        message = str(exc)
        if "nvJitLink" in message or "nvrtc" in message or "nvPTXCompiler" in message:
            raise ImportError(
                "FusedSINDy could not load its CUDA JIT native extension. "
                "Install matching nvidia-cuda-nvrtc and nvidia-nvjitlink packages "
                "for the CUDA/PyTorch environment, or rebuild implicit-sindy in that "
                "same environment so the extension links against the same CUDA JIT "
                "libraries."
            ) from exc
        raise


_implicit_sindy = _load_native()

__all__ = ["_implicit_sindy"]
