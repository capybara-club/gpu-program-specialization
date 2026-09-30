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
"""Application-owned CUDA resources; synchronization is scoped to events."""
from __future__ import annotations
import ctypes as C
import time

P, U, D, Z = C.c_void_p, C.c_uint32, C.c_uint64, C.c_size_t


def nvrtc_version():
    try: lib = C.CDLL("libnvrtc.so")
    except OSError: lib = C.CDLL("/usr/local/cuda/lib64/libnvrtc.so")
    lib.nvrtcVersion.argtypes = [C.POINTER(C.c_int), C.POINTER(C.c_int)]
    lib.nvrtcVersion.restype = C.c_int
    major, minor = C.c_int(), C.c_int()
    result = lib.nvrtcVersion(C.byref(major), C.byref(minor))
    if result: raise RuntimeError(f"Could not read NVRTC version ({result})")
    return [major.value, minor.value]


def compile_cuda(source, sm):
    try:
        lib = C.CDLL("libnvrtc.so")
    except OSError:
        lib = C.CDLL("/usr/local/cuda/lib64/libnvrtc.so")
    signatures = {"nvrtcCreateProgram": [C.POINTER(P), C.c_char_p, C.c_char_p, C.c_int, P, P],
                  "nvrtcCompileProgram": [P, C.c_int, C.POINTER(C.c_char_p)],
                  "nvrtcGetProgramLogSize": [P, C.POINTER(Z)], "nvrtcGetProgramLog": [P, P],
                  "nvrtcGetCUBINSize": [P, C.POINTER(Z)], "nvrtcGetCUBIN": [P, P],
                  "nvrtcDestroyProgram": [C.POINTER(P)]}
    for name, args in signatures.items():
        getattr(lib, name).argtypes = args
        getattr(lib, name).restype = C.c_int
    def check(name, *args):
        result = getattr(lib, name)(*args)
        if result:
            raise RuntimeError(f"{name}: NVRTC error {result}")
    program = P()
    check("nvrtcCreateProgram", C.byref(program), source.encode(), b"grammar_prelude.cu", 0, None, None)
    try:
        options = [b"--std=c++11", b"--fmad=false", f"--gpu-architecture=sm_{sm}".encode()]
        result = lib.nvrtcCompileProgram(program, len(options), (C.c_char_p*len(options))(*options))
        size = Z(); check("nvrtcGetProgramLogSize", program, C.byref(size))
        log = C.create_string_buffer(size.value); check("nvrtcGetProgramLog", program, log)
        if result:
            raise RuntimeError(f"Prelude NVRTC compilation failed ({result}): {log.value.decode()}")
        check("nvrtcGetCUBINSize", program, C.byref(size))
        cubin = C.create_string_buffer(size.value); check("nvrtcGetCUBIN", program, cubin)
        return cubin.raw
    finally:
        check("nvrtcDestroyProgram", C.byref(program))


class Buffer:
    def __init__(self, owner, size, data=None):
        self.owner, self.size, self.ptr, self.host = owner, max(1, size), D(), P()
        owner.call("cuMemAlloc_v2", C.byref(self.ptr), self.size)
        try:
            if data is not None:
                owner.call("cuMemAllocHost_v2", C.byref(self.host), self.size)
                if len(data) > self.size:
                    raise ValueError("Upload exceeds allocation")
                C.memmove(self.host, data, len(data))
                owner.call("cuMemcpyHtoDAsync_v2", self.ptr, self.host, len(data), owner.stream)
        except BaseException:
            self.close(); raise

    def read(self, size=None, offset=0):
        size = self.size-offset if size is None else size
        if size < 0 or offset < 0 or size+offset > self.size:
            raise ValueError("Download exceeds allocation")
        host = P()
        self.owner.call("cuMemAllocHost_v2", C.byref(host), max(1, size))
        try:
            self.owner.call("cuMemcpyDtoHAsync_v2", host, self.ptr.value+offset, size, self.owner.stream)
            self.owner.fence()
            return C.string_at(host, size)
        finally:
            if not self.owner.fatal:
                self.owner.call("cuMemFreeHost", host)

    def close(self):
        if self.owner.fatal:
            return
        if self.host:
            # An upload may still be pending after a host validation error.
            self.owner.fence()
            self.owner.call("cuMemFreeHost", self.host); self.host = P()
        if self.ptr:
            self.owner.call("cuMemFree_v2", self.ptr); self.ptr = D()


class CUDA:
    def __init__(self, device=0):
        self.lib = C.CDLL("libcuda.so.1")
        signatures = {
            "cuInit": [U], "cuDriverGetVersion": [C.POINTER(C.c_int)], "cuDeviceGet": [C.POINTER(C.c_int), C.c_int],
            "cuDeviceGetAttribute": [C.POINTER(C.c_int), C.c_int, C.c_int],
            "cuDevicePrimaryCtxRetain": [C.POINTER(P), C.c_int],
            "cuDevicePrimaryCtxRelease_v2": [C.c_int], "cuCtxSetCurrent": [P],
            "cuStreamCreate": [C.POINTER(P), U], "cuStreamDestroy_v2": [P],
            "cuModuleLoadData": [C.POINTER(P), P], "cuModuleUnload": [P],
            "cuModuleGetFunction": [C.POINTER(P), P, C.c_char_p],
            "cuMemAlloc_v2": [C.POINTER(D), Z], "cuMemFree_v2": [D],
            "cuMemAllocHost_v2": [C.POINTER(P), Z], "cuMemFreeHost": [P],
            "cuMemcpyHtoDAsync_v2": [D, P, Z, P], "cuMemcpyDtoHAsync_v2": [P, D, Z, P],
            "cuEventCreate": [C.POINTER(P), U], "cuEventRecord": [P, P],
            "cuEventSynchronize": [P], "cuEventElapsedTime": [C.POINTER(C.c_float), P, P],
            "cuEventDestroy_v2": [P], "cuLaunchKernel": [P, U, U, U, U, U, U, U, P, C.POINTER(P), P],
        }
        for name, args in signatures.items():
            getattr(self.lib, name).argtypes = args
            getattr(self.lib, name).restype = C.c_int
        self.fatal = False
        self.call("cuInit", 0)
        version = C.c_int(); self.call("cuDriverGetVersion", C.byref(version)); self.driver_version = version.value
        dev = C.c_int(); self.call("cuDeviceGet", C.byref(dev), device); self.device = dev.value
        self.context, self.stream = P(), P()
        self.call("cuDevicePrimaryCtxRetain", C.byref(self.context), self.device)
        self.call("cuCtxSetCurrent", self.context)
        self.call("cuStreamCreate", C.byref(self.stream), 1)
        self.events, self.modules = [], []
        self.ready, self.begin, self.end = self.event(), self.event(), self.event()

    def call(self, name, *args):
        if self.fatal:
            raise RuntimeError("CUDA owner requires process/context recovery")
        result = getattr(self.lib, name)(*args)
        if result:
            if name == "cuEventSynchronize":
                self.fatal = True
            raise RuntimeError(f"{name}: CUDA error {result}")

    def attribute(self, number):
        value = C.c_int(); self.call("cuDeviceGetAttribute", C.byref(value), number, self.device)
        return value.value

    @property
    def sm(self):
        return 10*self.attribute(75)+self.attribute(76)

    def event(self):
        event = P(); self.call("cuEventCreate", C.byref(event), 0); self.events.append(event)
        return event

    def record(self, event):
        self.call("cuEventRecord", event, self.stream)

    def fence(self):
        self.record(self.ready); self.call("cuEventSynchronize", self.ready)

    def buffer(self, size, data=None):
        return Buffer(self, size, data)

    def module(self, source):
        module = P(); cubin = C.create_string_buffer(compile_cuda(source, self.sm))
        self.call("cuModuleLoadData", C.byref(module), C.cast(cubin, P)); self.modules.append(module)
        return module

    def function(self, module, name):
        fn = P(); self.call("cuModuleGetFunction", C.byref(fn), module, name.encode()); return fn

    def launch(self, fn, count, args):
        if not count:
            return
        params = (P*len(args))(*[C.cast(C.byref(v), P) for v in args])
        self.call("cuLaunchKernel", fn, min(65535, (count+255)//256), 1, 1, 256, 1, 1, 0, self.stream, params, None)

    def timed(self, callback):
        self.record(self.begin); callback(); self.record(self.end)
        self.call("cuEventSynchronize", self.end)
        ms = C.c_float(); self.call("cuEventElapsedTime", C.byref(ms), self.begin, self.end)
        return ms.value / 1000

    def close(self):
        if self.fatal:
            raise RuntimeError("Unfenced CUDA work: retain allocations and restart this worker")
        self.fence()
        for module in reversed(self.modules): self.call("cuModuleUnload", module)
        for event in reversed(self.events): self.call("cuEventDestroy_v2", event)
        self.call("cuStreamDestroy_v2", self.stream)
        self.call("cuDevicePrimaryCtxRelease_v2", self.device)
