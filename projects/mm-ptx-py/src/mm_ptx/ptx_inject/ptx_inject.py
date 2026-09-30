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
# SPDX-FileCopyrightText: 2026 Charles Durham
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Mapping

from ._impl import (
    PtxInjectResult,
    PtxInjectMutType,
    ptx_inject_result_to_string,
    ptx_inject_create,
    ptx_inject_destroy,
    ptx_inject_num_injects,
    ptx_inject_inject_info_by_index,
    ptx_inject_variable_info_by_index,
    ptx_inject_render_ptx,
)


# ---------------------------------------------------------------------------
# Errors & result checking
# ---------------------------------------------------------------------------

class PtxInjectError(RuntimeError):
    """Raised when a PTX Inject C API call fails."""


def _check_result(ret: PtxInjectResult) -> None:
    """Raise PtxInjectError if `ret` indicates failure."""
    if ret != PtxInjectResult.PTX_INJECT_SUCCESS:
        raise PtxInjectError(ptx_inject_result_to_string(ret))


# ---------------------------------------------------------------------------
# Mutability types (wrap C enum)
# ---------------------------------------------------------------------------

class MutType(Enum):
    """Argument mutability in an injection site (IN/MOD/OUT)."""
    OUT = PtxInjectMutType.PTX_INJECT_MUT_TYPE_OUT
    MOD = PtxInjectMutType.PTX_INJECT_MUT_TYPE_MOD
    IN = PtxInjectMutType.PTX_INJECT_MUT_TYPE_IN


# ---------------------------------------------------------------------------
# Simple data holders
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InjectArg:
    name: str
    mut_type: MutType
    data_type: str
    register_type: str
    reg: str


@dataclass(frozen=True)
class Inject:
    name: str
    num_args: int
    num_sites: int
    args: List[InjectArg]


# ---------------------------------------------------------------------------
# User-facing helpers
# ---------------------------------------------------------------------------

class PTXInject:
    """
    High-level wrapper around a PTX Inject handle.

    Typical usage:

        # 1) Compile CUDA with PTX_INJECT macros to PTX (nvcc or cuda.core).
        # 2) Build injector from annotated PTX.
        injector = PTXInject(annotated_ptx)

        # 3) Inspect sites.
        injector.print_injects()

        # 4) Render final PTX by supplying PTX text for each inject (dict by name).
        final_ptx = injector.render_ptx({"my_inject": "...", ...})

    This object is also a mapping of inject name -> {arg_name -> InjectArg} for convenience:
        injector["my_inject"]["x"].data_type
    """

    def __init__(self, annotated_ptx: str):
        ret, self._handle = ptx_inject_create(annotated_ptx)
        _check_result(ret)

        self.injects: List[Inject] = []
        ret, num_injects = ptx_inject_num_injects(self._handle)
        _check_result(ret)

        for inject_idx in range(num_injects):
            ret, inject_name, inject_num_args, num_sites = ptx_inject_inject_info_by_index(
                self._handle, inject_idx
            )
            _check_result(ret)

            args: List[InjectArg] = []
            for arg_idx in range(inject_num_args):
                ret, arg_name, reg_name, mut_type, register_type, data_type = (
                    ptx_inject_variable_info_by_index(
                        self._handle, inject_idx, arg_idx
                    )
                )
                _check_result(ret)
                args.append(
                    InjectArg(
                        name=arg_name,
                        mut_type=MutType(mut_type),
                        data_type=data_type,
                        register_type=register_type,
                        reg=reg_name,
                    )
                )
            self.injects.append(Inject(inject_name, inject_num_args, num_sites, args))

        # Fast lookup: inject_name -> (arg_name -> InjectArg)
        self._inject_lookup: Dict[str, Dict[str, InjectArg]] = {
            inj.name: {arg.name: arg for arg in inj.args} for inj in self.injects
        }

    # --- Mapping-like sugar ---
    def __getitem__(self, inject_name: str) -> Dict[str, InjectArg]:
        return self._inject_lookup[inject_name]

    # --- Nice printing for users ---
    def print_injects(self) -> None:
        if not self.injects:
            print("No inject sites found.")
            return

        for i, inject in enumerate(self.injects, start=1):
            print(f"Inject #{i}:")
            print(f"  Name:            {inject.name!r}")
            print(f"  Number of Args:  {inject.num_args}")
            print(f"  Number of Sites: {inject.num_sites}")
            if inject.args:
                print("  Arguments:")
                for j, arg in enumerate(inject.args, start=1):
                    print(f"    Arg #{j}:")
                    print(f"      CUDA Name:     {arg.name!r}")
                    print(f"      Mutation Type: {arg.mut_type.name}")
                    print(f"      Type Token:    {arg.data_type}")
                    print(f"      Register Type: {arg.register_type}")
                    print(f"      Register Name: {arg.reg!r}")
            else:
                print("  (No arguments.)")
            print()

    # --- Core render ---
    def render_ptx(self, ptx_stubs: Mapping[str, str]) -> str:
        """
        Render final PTX by supplying PTX text for each inject (keyed by inject name).

        Args:
            ptx_stubs: dict mapping inject_name -> PTX stub source

        Returns:
            Combined PTX string with stubs injected in the correct order.
        """
        ordered_stubs = [ptx_stubs[inject.name] for inject in self.injects]

        ret, required_size = ptx_inject_render_ptx(self._handle, ordered_stubs, None)
        _check_result(ret)

        buffer = bytearray(required_size + 1)
        ret, written_size = ptx_inject_render_ptx(self._handle, ordered_stubs, buffer)
        _check_result(ret)

        return buffer[:written_size].decode("utf-8")

    # --- Context manager & cleanup ---
    def close(self) -> None:
        if getattr(self, "_handle", None):
            ret = ptx_inject_destroy(self._handle)
            _check_result(ret)
            self._handle = None  # type: ignore[assignment]

    def __enter__(self) -> "PTXInject":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


__all__ = [
    "PtxInjectError",
    "MutType",
    "InjectArg",
    "Inject",
    "PTXInject",
]
