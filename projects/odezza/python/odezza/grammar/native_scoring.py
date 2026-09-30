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
"""ctypes layouts for the public scoring/reduction ABI; no grammar in C99."""
from __future__ import annotations
import ctypes as C
from pathlib import Path

from odezza.native import AstProgram, NativeError
from .cuda import D, U, Z, P


class Rhs(C.Structure):
    _fields_ = [("state_index", C.c_uint8), ("program", AstProgram)]


class System(C.Structure):
    _fields_ = [("rhs", C.POINTER(Rhs)), ("rhs_count", Z)]


class CreateInfo(C.Structure):
    _fields_ = [("sm", U), ("states", Z), ("state_capacity", Z), ("constants", U), ("constant_capacity", U),
                ("fixed_rhs", C.POINTER(Rhs)), ("fixed_count", Z), ("system_capacity", U),
                ("shared_patch", U), ("system_patch", U), ("workers", U), ("slots", U)]


class TemplateInfo(C.Structure):
    _fields_ = [(name, U) for name in ("sm", "states", "constants", "systems", "shared_patch", "system_patch")]


class Launch(C.Structure):
    _fields_ = [("banks", D), ("bank_count", U), ("toggle_bits", U), ("offsets", D), ("times", D),
                ("reference", D), ("trajectories", U), ("points", U), ("steps", U), ("scores", D),
                ("sampled", D), ("uniform", D), ("normal", D), ("pool_size", D), ("missing", U), ("ready", P)]


class RunReport(C.Structure):
    _fields_ = [("systems", Z), ("modules", Z), ("configurations", D), ("seconds", C.c_double)]


class ReductionSize(C.Structure):
    _fields_ = [("configurations", D), ("scores", D), ("groups", D), ("tiles", U),
                ("workspace", Z), ("winners", Z), ("counts", Z)]


class ReductionReport(C.Structure):
    _fields_ = [("seconds", C.c_double), ("groups", D), ("scores", D)]


class Winner(C.Structure):
    _fields_ = [("mse", C.c_float), ("reserved", U), ("index", D)]


class Counts(C.Structure):
    _fields_ = [("valid", D), ("invalid", D), ("negative", D)]


def library(path):
    lib = C.CDLL(str(Path(path).resolve(strict=True)))
    signatures = {
        "scoring_pipeline_create": [C.POINTER(CreateInfo), C.POINTER(P)],
        "scoring_pipeline_create_with_template": [C.POINTER(CreateInfo), P, C.POINTER(P)],
        "scoring_template_create": [C.POINTER(TemplateInfo), C.POINTER(P)],
        "scoring_template_destroy": [P],
        "scoring_template_write_error": [P, P, Z, C.POINTER(Z)],
        "scoring_pipeline_workspace_requirements": [P, C.POINTER(Z), C.POINTER(Z)],
        "scoring_pipeline_run": [P, C.POINTER(System), Z, C.POINTER(Launch), P, Z, C.POINTER(RunReport)],
        "scoring_pipeline_write_error": [P, P, Z, C.POINTER(Z)],
        "scoring_pipeline_destroy": [P],
        "score_reducer_create": [U, U, C.POINTER(P)], "score_reducer_destroy": [P],
        "score_reducer_write_error": [P, P, Z],
        "score_reduction_requirements": [C.POINTER(Launch), Z, C.c_int, U, C.POINTER(ReductionSize)],
        "score_reducer_run": [P, C.POINTER(Launch), Z, C.c_int, D, Z, D, Z, D, Z, C.POINTER(ReductionReport)],
        "score_reducer_gather": [P, C.POINTER(Launch), Z, C.c_int, U, D, Z, D, Z],
    }
    for name, args in signatures.items():
        fn = getattr(lib, "odezza_"+name); fn.argtypes = args; fn.restype = C.c_int
    return lib


class Template:
    def __init__(self, lib, sm, nstates, nslots, systems, patch):
        self.lib, self.handle = lib, P()
        self.capacity = nslots
        info = TemplateInfo(sm, nstates, nslots, systems, 64, patch)
        result = lib.odezza_scoring_template_create(C.byref(info), C.byref(self.handle))
        if result:
            text = C.create_string_buffer(4096); size = Z()
            lib.odezza_scoring_template_write_error(self.handle, text, len(text), C.byref(size))
            self.close()
            raise NativeError(result, text.value.decode(errors="replace"))

    def close(self):
        if self.handle:
            result = self.lib.odezza_scoring_template_destroy(self.handle)
            if result: raise NativeError(result, "Could not destroy scoring template")
            self.handle = P()


class Pipeline:
    def __init__(self, lib, gpu, nstates, nslots, system_capacity=64, patch_capacity=384, template=None):
        self.lib, self.gpu, self.handle = lib, gpu, P()
        info = CreateInfo(gpu.sm, nstates, nstates, nslots, template.capacity if template else max(1, nslots), None, 0,
                          system_capacity, 64, patch_capacity, 2, 2)
        result = lib.odezza_scoring_pipeline_create_with_template(C.byref(info), template.handle, C.byref(self.handle)) if template else lib.odezza_scoring_pipeline_create(C.byref(info), C.byref(self.handle))
        if result:
            error = self.error(result); self.close(); raise error
        try:
            size, alignment = Z(), Z()
            self.check(lib.odezza_scoring_pipeline_workspace_requirements(self.handle, C.byref(size), C.byref(alignment)))
            self.workspace = C.create_string_buffer(size.value+alignment.value)
            self.address = (C.addressof(self.workspace)+alignment.value-1)//alignment.value*alignment.value
            self.size = size.value
        except BaseException:
            self.close()
            raise

    def error(self, result):
        text = C.create_string_buffer(4096); size = Z()
        self.lib.odezza_scoring_pipeline_write_error(self.handle, text, len(text), C.byref(size))
        return NativeError(result, text.value.decode(errors="replace"))

    def check(self, result):
        if result:
            if result == 18: self.gpu.fatal = True
            raise self.error(result)

    def run(self, programs, launch):
        keep, items = [], []
        for system in programs:
            rhs = []
            for state, code in enumerate(system):
                buffer = C.create_string_buffer(code); keep.append(buffer)
                rhs.append(Rhs(state, AstProgram(C.addressof(buffer), len(code))))
            array = (Rhs*len(rhs))(*rhs); keep.append(array); items.append(System(array, len(rhs)))
        array = (System*len(items))(*items); report = RunReport()
        self.check(self.lib.odezza_scoring_pipeline_run(self.handle, array, len(items), C.byref(launch),
                                                      self.address, self.size, C.byref(report)))
        return {name: getattr(report, name) for name, _ in report._fields_}

    def close(self):
        if self.handle and not self.gpu.fatal:
            self.check(self.lib.odezza_scoring_pipeline_destroy(self.handle)); self.handle = P()


class Reducer:
    def __init__(self, lib, gpu, k=16):
        self.lib, self.gpu, self.k, self.handle = lib, gpu, k, P()
        result = lib.odezza_score_reducer_create(gpu.sm, k, C.byref(self.handle))
        if result:
            try: self.check(result)
            finally: self.close()

    def check(self, result):
        if result:
            if result == 18: self.gpu.fatal = True
            text = C.create_string_buffer(4096)
            self.lib.odezza_score_reducer_write_error(self.handle, text, len(text))
            raise NativeError(result, text.value.decode(errors="replace"))

    def requirements(self, launch, count):
        size = ReductionSize()
        self.check(self.lib.odezza_score_reduction_requirements(C.byref(launch), count, 1, self.k, C.byref(size)))
        return size

    def run(self, launch, systems, nslots):
        size = self.requirements(launch, systems)
        resources = []
        try:
            for n in (size.workspace, size.winners, size.counts, size.groups*self.k*nslots*4):
                resources.append(self.gpu.buffer(n))
            workspace, winners, counts, values = resources
            report = ReductionReport()
            self.check(self.lib.odezza_score_reducer_run(self.handle, C.byref(launch), systems, 1,
                workspace.ptr, size.workspace, winners.ptr, size.winners, counts.ptr, size.counts, C.byref(report)))
            if nslots:
                self.check(self.lib.odezza_score_reducer_gather(self.handle, C.byref(launch), systems, 1,
                    nslots, winners.ptr, size.winners, values.ptr, size.groups*self.k*nslots*4))
            ws = list((Winner*(size.groups*self.k)).from_buffer_copy(winners.read(size.winners)))
            cs = list((Counts*size.groups).from_buffer_copy(counts.read(size.counts)))
            vs = list((C.c_float*(size.groups*self.k*nslots)).from_buffer_copy(values.read(size.groups*self.k*nslots*4))) if nslots else []
            # Caller may consume selected scores before these buffers are released.
            return ws, cs, vs, report.seconds, resources
        except BaseException:
            for buffer in reversed(resources): buffer.close()
            raise

    def close(self):
        if self.handle and not self.gpu.fatal:
            self.check(self.lib.odezza_score_reducer_destroy(self.handle)); self.handle = P()
