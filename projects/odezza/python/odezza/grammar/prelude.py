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
"""Reusable Philox generation and mixed-radix coefficient preparation on GPU."""
from __future__ import annotations
from collections import OrderedDict
import ctypes as C
import hashlib
import math
import struct

from .cuda import D, U
from .lowering import CompatibilityError, f32

PROFILE = "odezza.grammar.philox4x32-10.v1"


def bank_address(request):
    # Compiler key excludes length; both run and skeleton scope are already in it.
    digest = hashlib.sha256((PROFILE + ":" + request["key"]).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "little"), int.from_bytes(digest[8:16], "little") & (2**63-1)


class Slot(C.Structure):
    _fields_ = [("data", D), ("divisor", D), ("count", D), ("kind", U),
                ("a", C.c_int), ("b", C.c_int), ("v0", C.c_float), ("v1", C.c_float)]


SOURCE = r'''
typedef unsigned long long U64;
struct Slot { U64 data,divisor,count; unsigned kind; int a,b; float v0,v1; };
__device__ uint4 philox(U64 block,U64 seed,U64 domain) {
    uint4 c=make_uint4((unsigned)block,(unsigned)(block>>32),(unsigned)domain,(unsigned)(domain>>32));
    unsigned k0=(unsigned)seed,k1=(unsigned)(seed>>32);
    for(unsigned r=0;r<10;++r) {
        unsigned hi0=__umulhi(0xd2511f53u,c.x),lo0=0xd2511f53u*c.x;
        unsigned hi1=__umulhi(0xcd9e8d57u,c.z),lo1=0xcd9e8d57u*c.z;
        c=make_uint4(hi1^c.y^k0,lo1,hi0^c.w^k1,lo0);
        k0+=0x9e3779b9u;k1+=0xbb67ae85u;
    }
    return c;
}
__device__ float unit(unsigned x) { return ((float)(x>>9)+0.5f)*0.00000011920928955078125f; }
extern "C" __global__ void grammar_rng(float *out,U64 count,U64 seed,U64 stream,unsigned normal) {
    for(U64 b=(U64)blockIdx.x*blockDim.x+threadIdx.x;b<(count+3)/4;b+=(U64)gridDim.x*blockDim.x) {
        uint4 bits=philox(b,seed,(stream<<1)|normal);
        float u[4]={unit(bits.x),unit(bits.y),unit(bits.z),unit(bits.w)};
        if(normal) for(unsigned p=0;p<4;p+=2) {
            float radius=sqrtf(-2.0f*logf(u[p])),s,c;
            sincosf(6.2831853071795864769f*u[p+1],&s,&c);
            u[p]=radius*c;u[p+1]=radius*s;
        }
        for(unsigned j=0;j<4;++j) if(4*b+j<count) out[4*b+j]=u[j];
    }
}
extern "C" __global__ void grammar_prelude(const Slot *slots,float *out,unsigned nslots,unsigned systems,U64 start,unsigned banks) {
    for(U64 item=(U64)blockIdx.x*blockDim.x+threadIdx.x;item<(U64)systems*banks;item+=(U64)gridDim.x*blockDim.x) {
        unsigned sys=item/banks;U64 bank=start+item%banks;float *row=out+item*nslots;
        for(unsigned s=0;s<nslots;++s) {
            Slot d=slots[sys*nslots+s];
            if(d.kind==0) { row[s]=d.v0;continue; }
            float u=((const float*)d.data)[(bank/d.divisor)%d.count];
            if(d.kind==1) { row[s]=u;continue; }
            float a=d.a<0?d.v0:row[d.a],b=d.b<0?d.v1:row[d.b];
            float v=u;
            if(d.kind==2) v=a*u+b;
            if(d.kind==3) v=(1.0f-u)*a+u*b;
            if(d.kind==4) v=a+b*u;
            if(d.kind==5) v=expf((1.0f-u)*logf(a)+u*logf(b));
            row[s]=v;
        }
    }
}
struct Winner { float mse; unsigned reserved; U64 index; };
extern "C" __global__ void grammar_consume(float *scores,const Winner *w,U64 count) {
    for(U64 i=(U64)blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=(U64)gridDim.x*blockDim.x)
        if(w[i].index!=~0ull) scores[w[i].index]=3.40282346638528859812e38f;
}
'''


class Prelude:
    def __init__(self, gpu, max_bank_bytes=256*1024*1024):
        self.gpu, self.max_bank_bytes = gpu, max_bank_bytes
        module = gpu.module(SOURCE)
        self.fill = gpu.function(module, "grammar_rng")
        self.prepare = gpu.function(module, "grammar_prelude")
        self.consume = gpu.function(module, "grammar_consume")
        self.cache = OrderedDict()
        self.kernel_seconds = 0.
        self.generation_seconds = 0.

    def bank(self, request):
        key = request["key"]
        count = request["count"]
        if count*4 > self.max_bank_bytes:
            raise CompatibilityError("One RNG bank exceeds execution.max_bank_bytes")
        if key in self.cache and self.cache[key].size >= count*4:
            self.cache.move_to_end(key)
            return self.cache[key]
        # Eviction is performed only between batches; all current descriptors stay valid.
        if key in self.cache:
            self.cache.pop(key).close()
        buffer = self.gpu.buffer(count*4)
        seed, stream = bank_address(request)
        self.generation_seconds += self.gpu.timed(lambda: self.gpu.launch(self.fill, (count+3)//4,
            [buffer.ptr, D(count), D(seed), D(stream), U(request["base"] == "normal01")]))
        self.cache[key] = buffer
        return buffer

    def trim(self):
        while sum(b.size for b in self.cache.values()) > self.max_bank_bytes:
            self.cache.popitem(last=False)[1].close()

    def descriptors(self, variants, layouts):
        resources, slots = [], []
        try:
            requests = {}
            for variant in variants:
                for request in variant["pools"]["rng_bank_requests"]:
                    if request["count"] > requests.get(request["key"], {}).get("count", 0):
                        requests[request["key"]] = request
            if sum(r["count"]*4 for r in requests.values()) > self.max_bank_bytes:
                raise CompatibilityError("Batch RNG banks exceed execution.max_bank_bytes; reduce batch_variants or increase pool budget")
            def projected_bytes():
                return (sum(b.size for key, b in self.cache.items() if key not in requests) +
                        sum(max(r["count"]*4, self.cache[key].size if key in self.cache else 0)
                            for key, r in requests.items()))
            while projected_bytes() > self.max_bank_bytes and self.cache:
                self.cache.popitem(last=False)[1].close()
            for request in requests.values(): self.bank(request)
            for variant, layout in zip(variants, layouts):
                pools = variant["pools"]
                axes, divisor = {}, 1
                for axis in reversed(pools["pool_axes"]):
                    if axis["kind"] != "leaf":
                        axes[axis["id"]] = (divisor, axis["count"])
                        divisor *= axis["count"]
                banks = {r["id"]: r for r in pools["rng_bank_requests"]}
                # Ensure every count extension occurs before any descriptor uses its pointer.
                index = {p: i for i, p in enumerate(layout.slots)}
                for namespace, name in layout.slots:
                    d = Slot(0, 1, 1, 0, -1, -1, 0., 0.)
                    if namespace == "param":
                        d.v0 = f32(pools["parameter_initials"][name])
                    elif namespace == "const":
                        spec = pools["constants"][name]
                        if spec["kind"] == "fixed":
                            d.v0 = f32(spec["value"])
                        else:
                            values = [f32(v) for v in spec["values"]]
                            raw = struct.pack("<"+"f"*len(values), *values)
                            b = self.gpu.buffer(len(raw), raw); resources.append(b)
                            d.kind, d.data = 1, b.ptr.value
                            d.divisor, d.count = axes[spec["axis"]]
                    else:
                        spec = pools["rng_bindings"][name]
                        request = banks[spec["bank_id"]]
                        d.data = self.cache[request["key"]].ptr.value
                        d.divisor, d.count = axes[spec["axis"]]
                        transform = spec["transform"]
                        kind = transform["kind"]
                        fields = {"identity": (), "affine": ("scale", "shift"), "uniform": ("low", "high"),
                                  "normal": ("mean", "std"), "log_uniform": ("low", "high")}[kind]
                        d.kind = {"identity": 1, "affine": 2, "uniform": 3, "normal": 4, "log_uniform": 5}[kind]
                        for field, ref, value in zip(fields, ("a", "b"), ("v0", "v1")):
                            v = transform[field]
                            if isinstance(v, dict): setattr(d, ref, index[("const", v["const"])])
                            else: setattr(d, value, f32(v))
                    slots.append(d)
            array = (Slot*len(slots))(*slots)
            buffer = self.gpu.buffer(C.sizeof(array), bytes(array)); resources.append(buffer)
            return buffer, resources
        except BaseException:
            for r in reversed(resources): r.close()
            raise

    def run(self, descriptors, output, nslots, systems, start, banks):
        elapsed = self.gpu.timed(lambda: self.gpu.launch(self.prepare, systems*banks,
            [descriptors.ptr, output.ptr, U(nslots), U(systems), D(start), U(banks)]))
        self.kernel_seconds += elapsed
        return elapsed

    def close(self):
        for buffer in self.cache.values(): buffer.close()
        self.cache.clear()
