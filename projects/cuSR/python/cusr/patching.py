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
from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np

from .ast import pack_programs
from .inspection import CubinInspection


class PatchEpilogue(str, Enum):
    SSE = "sse"
    VALUE = "value"


def _native_module() -> Any:
    try:
        import _cusr_native as _native
    except ImportError as error:
        raise RuntimeError("cuSR was built without its nanobind extension") from error
    return _native


def _native_program_matrix(programs: np.ndarray) -> np.ndarray:
    if (
        isinstance(programs, np.ndarray)
        and programs.dtype == np.uint64
        and programs.ndim == 2
        and programs.flags.c_contiguous
    ):
        return programs
    return pack_programs(programs)


def prepare_patch_layout(inspection: CubinInspection) -> Any:
    return _native_module().SassPatchLayout(inspection)


def patch_cubin_in_place(
    layout: Any,
    programs: np.ndarray,
    cubin: bytearray,
    *,
    epilogue: PatchEpilogue | str = PatchEpilogue.SSE,
    routines: np.ndarray | None = None,
) -> Any:
    try:
        epilogue = PatchEpilogue(epilogue)
    except ValueError as error:
        raise ValueError("epilogue must be 'sse' or 'value'") from error

    native = _native_module()
    native_epilogue = native.PatchEpilogue.SSE if epilogue is PatchEpilogue.SSE else native.PatchEpilogue.VALUE
    packed_programs = _native_program_matrix(programs)
    packed_routines = None if routines is None else _native_program_matrix(routines)
    return layout.patch(packed_programs, cubin, native_epilogue, packed_routines)
