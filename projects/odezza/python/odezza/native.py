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
"""Thin ctypes binding to core/odezza.h. No search or CUDA ownership policy.

The caller owns the current CUDA context and every device buffer. A selected
library path must exist; this binding never falls back to Python specialization.
"""
from __future__ import annotations
import ctypes as C
from pathlib import Path

U32=C.c_uint32
U64=C.c_uint64
DevicePointer=U64

class AstProgram(C.Structure):
    _fields_=[('bytes',C.c_void_p),('byte_count',C.c_size_t)]

class LmShape(C.Structure):
    _fields_=[('state_count',U32),('parameter_count',U32),('site_patch_capacity',U32),('lanes_per_fit',U32)]

class LmCreateInfo(C.Structure):
    _fields_=[('sm_version',U32),('shape',LmShape)]

class LmFit(C.Structure):
    _fields_=[('rhs',C.POINTER(AstProgram)),('starts_device',DevicePointer),('start_count',U64),('toggle_bit_count',U32),
              ('offsets_device',DevicePointer),('times_device',DevicePointer),('reference_device',DevicePointer),('weights_device',DevicePointer),
              ('trajectory_count',U32),('point_count',U32),('lower_device',DevicePointer),('upper_device',DevicePointer),
              ('steps_per_interval',U32),('max_iterations',U32),('max_damping_attempts',U32),
              ('initial_damping',C.c_float),('max_step',C.c_float),('target_mse',C.c_float),
              *[(name,DevicePointer) for name in ('parameters_device','initial_mse_device','mse_device','iterations_device','accepted_device','factorizations_device','evaluations_device','invalid_device')],
              ('input_ready_event',C.c_void_p)]

class LmReport(C.Structure):
    _fields_=[('fit_count',U64),('completed_system_count',C.c_size_t),('total_seconds',C.c_double),
              ('specialization_seconds',C.c_double),('module_load_seconds',C.c_double),('kernel_seconds',C.c_double),
              ('peak_inflight_modules',U32),('maximum_register_count',U32)]
    def as_dict(self):return {name:getattr(self,name) for name,_ in self._fields_}

class NativeError(RuntimeError):
    def __init__(self,result,message,pipeline=None):
        self.result=result
        # Retain a diagnostic handle if initialization cleanup itself failed.
        self.pipeline=pipeline
        super().__init__(f'Odezza C99 result {result}: {message}')

class LmShapeReport(C.Structure):
    _fields_=[*[(name,U32) for name in ('requested_lanes_per_fit','allowed_lanes_mask','available_lanes_mask','used_lanes_mask','attempted_lanes_mask')],
              ('template_registers',U32*4),('template_results',C.c_int*4),
              ('system_counts',U64*4),('register_rejections',U64*4),('template_prepare_seconds',C.c_double)]
    def as_dict(self):
        return {name:list(getattr(self,name)) if isinstance(getattr(self,name),C.Array) else getattr(self,name)
                for name,_ in self._fields_}

class LmPipeline:
    def __init__(self,library: str | Path,sm: int,shape: LmShape,*,fallback_lanes=()):
        fallback_lanes=tuple(fallback_lanes)
        if any(type(w) is not int or w not in (1,2,4,8) or w<=shape.lanes_per_fit for w in fallback_lanes):
            raise ValueError('Fallback lanes must be supported widths greater than the requested width')
        mask=0
        for w in fallback_lanes:mask|=w
        self.library=C.CDLL(str(Path(library).resolve(strict=True)))
        signatures={
            'create':([C.POINTER(LmCreateInfo),C.POINTER(C.c_void_p)],C.c_int),
            'create_with_fallback':([C.POINTER(LmCreateInfo),U32,C.POINTER(C.c_void_p)],C.c_int),
            'shape_report':([C.c_void_p,C.POINTER(LmShapeReport)],C.c_int),
            'workspace_requirements':([C.c_void_p,C.POINTER(C.c_size_t),C.POINTER(C.c_size_t)],C.c_int),
            'run':([C.c_void_p,C.POINTER(LmFit),C.c_size_t,C.c_void_p,C.c_size_t,C.POINTER(LmReport)],C.c_int),
            'write_error':([C.c_void_p,C.c_void_p,C.c_size_t,C.POINTER(C.c_size_t)],C.c_int),
            'destroy':([C.c_void_p],C.c_int),
        }
        for name,(args,result) in signatures.items():
            fn=getattr(self.library,'odezza_lm_pipeline_'+name);fn.argtypes=args;fn.restype=result
        self.handle=C.c_void_p();self.workspace=None;self.fatal=False
        result=self.library.odezza_lm_pipeline_create_with_fallback(C.byref(LmCreateInfo(sm,shape)),mask,C.byref(self.handle))
        if result:
            error=self.error(result)
            try:self.close()
            except NativeError as cleanup:raise cleanup from error
            raise error
        try:
            size=C.c_size_t();alignment=C.c_size_t()
            self.check(self.library.odezza_lm_pipeline_workspace_requirements(self.handle,C.byref(size),C.byref(alignment)))
            self.workspace=C.create_string_buffer(size.value+alignment.value-1)
            address=C.addressof(self.workspace)
            self.workspace_address=C.c_void_p((address+alignment.value-1)//alignment.value*alignment.value)
            self.workspace_size=size.value
        except BaseException:
            self.close()
            raise

    def error(self,result):
        buffer=C.create_string_buffer(1024);size=C.c_size_t()
        if self.handle:self.library.odezza_lm_pipeline_write_error(self.handle,buffer,len(buffer),C.byref(size))
        return NativeError(result,buffer.value.decode(errors='replace'),self)

    def check(self,result):
        if result:
            if result==18:self.fatal=True
            raise self.error(result)

    def run(self,fits):
        if not self.handle or self.fatal:raise RuntimeError('Native handle is closed or requires context recovery')
        items=(LmFit*len(fits))(*fits);report=LmReport()
        self.check(self.library.odezza_lm_pipeline_run(self.handle,items,len(items),self.workspace_address,self.workspace_size,C.byref(report)))
        return dict(report.as_dict(),shapes=self.shape_report())

    def shape_report(self):
        report=LmShapeReport()
        self.check(self.library.odezza_lm_pipeline_shape_report(self.handle,C.byref(report)))
        return report.as_dict()

    def close(self):
        if self.handle:
            self.check(self.library.odezza_lm_pipeline_destroy(self.handle))
            self.handle=C.c_void_p();self.workspace=None

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
