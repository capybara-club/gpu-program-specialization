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

import sys
import os

import cv2

import mm_ptx.ptx_inject as ptx_inject
import mm_ptx.stack_ptx as stack_ptx
from mm_ptx import get_ptx_inject_header

from cuda.core import LaunchConfig, launch

import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stack_ptx_default_types import Stack, PtxInstruction
from stack_ptx_default_types import compiler as stack_ptx_compiler
from compiler_helper import NvCompilerHelper

from helpers import show_image, render_video

ptx_header = get_ptx_inject_header().replace("\\", "/")

cuda_code = (
    f"#include \"{ptx_header}\"\n"
    r"""

#define PI 3.1415926535897932384626433832795L

typedef unsigned char u8;

__device__
u8 float_to_byte(float val) {
  return ((val <= 0.0f) ? 0 : ((val > (1.0f - 0.5f / 255.0f)) ? 255 : (u8)((255.0f * val) + 0.5f)));
}

__device__
uchar4 color_float_to_byte(float3 c)
{
    uchar4 b;
    b.x = float_to_byte(c.x);
    b.y = float_to_byte(c.y);
    b.z = float_to_byte(c.z);
    b.w = 255;
    return b;
}

__device__
float apply_log_filter(bool apply, float n, float log_multiplier) {
    return apply ? log (n * log_multiplier + 1) : n;
}

__device__
float clamp(float a, float mn, float mx) {
  return min(max(a, mn), mx);
}

__device__
float3 hsl_to_rgb(float3 hsl) {
    float nr, ng, nb, chroma, h, s, l;

    h = hsl.x;
    s = hsl.y;
    l = hsl.z;

    nr = fabsf(h * 6.0f - 3.0f) - 1.0f;
    ng = 2.0f - fabsf(h * 6.0f - 2.0f);
    nb = 2.0f - fabsf(h * 6.0f - 4.0f);

    nr = clamp(nr, 0.0f, 1.0f);
    nb = clamp(nb, 0.0f, 1.0f);
    ng = clamp(ng, 0.0f, 1.0f);

    chroma = (1.0f - fabsf(2.0f * l - 1.0f)) * s;
    float3 rgb;
    rgb.x = (nr - 0.5f) * chroma + l;
    rgb.y = (ng - 0.5f) * chroma + l;
    rgb.z = (nb - 0.5f) * chroma + l;
    return rgb;
}

__device__
uchar4 rgb_to_bgr(uchar4 rgb) {
    uchar4 bgr;
    bgr.x = rgb.z;
    bgr.y = rgb.y;
    bgr.z = rgb.x;
    bgr.w = rgb.w;
    return bgr;
}

extern "C"
__global__ 
void
kernel(
    size_t height, 
    size_t width,
    float light,
    float saturation,
    float max_norm,
    float time_step,
    unsigned int* __restrict__ data
) {
    uchar4* data_ptr = reinterpret_cast<uchar4*>(data);
    float log_multiplier = 2.0;
    size_t batch_num = blockIdx.z;
    size_t batch_offset = batch_num * height * width;
    for (size_t i = blockDim.x * blockIdx.x + threadIdx.x; i < height * width; i += gridDim.x * blockDim.x) {
        size_t w = (i % width);
        size_t h = height - (i / width);
        float w_norm = 2*w/ (float)width - 1.0;
        float h_norm = 2*h/ (float)height - 1.0;
        w_norm *= 8.0;
        h_norm *= 8.0;
        float t = (float)batch_num * time_step;
        float f = 0.0f;
        float g = 0.0f;
        PTX_INJECT("func",
            PTX_IN (F32, t, t),
            PTX_IN (F32, w_norm, w_norm),
            PTX_IN (F32, h_norm, h_norm),
            PTX_OUT(F32, f, f),
            PTX_OUT(F32, g, g)
        );
        float theta = atan2(g,f);
        float norm = sqrt(f*f + g*g);
        float log_max_norm = apply_log_filter(true, max_norm, log_multiplier);
        float3 hsl;
        hsl.x = (theta + (float)PI) / (2.0*(float)PI);
        hsl.y = saturation;
        hsl.z = light * apply_log_filter(true, norm, log_multiplier) / log_max_norm;
        float3 rgb = hsl_to_rgb(hsl);
        uchar4 rgb_bytes = color_float_to_byte(rgb);
        uchar4 bgr_bytes = rgb_to_bgr(rgb_bytes);
        data_ptr[batch_offset + i] = bgr_bytes;
    }
}
"""
)

nv_compiler = NvCompilerHelper()
dev = nv_compiler.dev

pt_stream = torch.cuda.current_stream()
print(f"PyTorch stream: {pt_stream}")

class PyTorchStreamWrapper:
    def __init__(self, pt_stream):
        self.pt_stream = pt_stream

    def __cuda_stream__(self):
        stream_id = self.pt_stream.cuda_stream
        return (0, stream_id)

s = dev.create_stream(PyTorchStreamWrapper(pt_stream))

annotated_ptx = nv_compiler.cuda_to_ptx(cuda_code)

inject = ptx_inject.PTXInject(annotated_ptx)

inject.print_injects()

func = inject['func']

assert( func['t'].mut_type == ptx_inject.MutType.IN )
assert( func['t'].data_type == "F32" )

assert( func['w_norm'].mut_type == ptx_inject.MutType.IN )
assert( func['w_norm'].data_type == "F32" )

assert( func['h_norm'].mut_type == ptx_inject.MutType.IN )
assert( func['h_norm'].data_type == "F32" )

assert( func['f'].mut_type == ptx_inject.MutType.OUT )
assert( func['f'].data_type == "F32" )

assert( func['g'].mut_type == ptx_inject.MutType.OUT )
assert( func['g'].data_type == "F32" )

registry = stack_ptx.RegisterRegistry()
registry.add(func['t'].reg,         Stack.f32, name = 't')
registry.add(func['w_norm'].reg,    Stack.f32, name = 'x')
registry.add(func['h_norm'].reg,    Stack.f32, name = 'y')
registry.add(func['f'].reg,         Stack.f32, name = 'f')
registry.add(func['g'].reg,         Stack.f32, name = 'g')
registry.freeze()

A = 1.295
B = 0.560
C = 0.395
D = 0.440
E = 0.250
F = 0.000
G = 0.000
H = 0.000

instructions = [
    registry.t,
    PtxInstruction.sin_approx_ftz_f32,
    registry.x,
    PtxInstruction.mul_ftz_f32,
    Stack.f32.constant(A),
    PtxInstruction.mul_ftz_f32,
    PtxInstruction.sin_approx_ftz_f32,

    registry.t,
    PtxInstruction.sin_approx_ftz_f32,
    Stack.f32.constant(G),
    PtxInstruction.mul_ftz_f32,
    PtxInstruction.add_ftz_f32,

    registry.y,
    registry.x,
    PtxInstruction.mul_ftz_f32,
    Stack.f32.constant(B),
    PtxInstruction.mul_ftz_f32,
    registry.t,
    PtxInstruction.sin_approx_ftz_f32,
    Stack.f32.constant(E),
    PtxInstruction.mul_ftz_f32,
    PtxInstruction.add_ftz_f32,
    PtxInstruction.sin_approx_ftz_f32,
    PtxInstruction.add_ftz_f32,

    registry.t,
    PtxInstruction.sin_approx_ftz_f32,
    registry.y,
    PtxInstruction.mul_ftz_f32,
    Stack.f32.constant(C),
    PtxInstruction.mul_ftz_f32,
    PtxInstruction.cos_approx_ftz_f32,

    registry.t,
    PtxInstruction.cos_approx_ftz_f32,
    Stack.f32.constant(H),
    PtxInstruction.mul_ftz_f32,
    PtxInstruction.add_ftz_f32,

    registry.y,
    registry.x,
    PtxInstruction.mul_ftz_f32,
    Stack.f32.constant(D),
    PtxInstruction.mul_ftz_f32,
    registry.t,
    PtxInstruction.sin_approx_ftz_f32,
    Stack.f32.constant(F),
    PtxInstruction.mul_ftz_f32,
    PtxInstruction.add_ftz_f32,
    PtxInstruction.cos_approx_ftz_f32,
    PtxInstruction.add_ftz_f32
]

requests = [registry.f, registry.g]

ptx_stub = \
    stack_ptx_compiler.compile(
        registry=registry,
        instructions=instructions, 
        requests=requests,
        execution_limit=100,
        max_ast_size=100,
        max_ast_to_visit_stack_depth=20,
        stack_size=128,
        max_frame_depth=4,
        store_size=16
    )

print(ptx_stub)

ptx_stubs = {
    'func' : ptx_stub
}

rendered_ptx = inject.render_ptx(ptx_stubs)

mod = nv_compiler.ptx_to_cubin(rendered_ptx)

ker = mod.get_kernel("kernel")

def run_kernel(
    height = 1024, 
    width = 1024,
    num_batches = 1024,
    light = 0.5,
    saturation = 0.6,
    max_norm = 3.0,
    time_step = 0.01
):
    n = height * width
    out = torch.zeros((num_batches, n), dtype=torch.uint32, device="cuda")

    block = int(256)
    grid = (int(n // 256), 1, num_batches)
    config = LaunchConfig(grid=grid, block=block)

    ker_args = (
        height,
        width,
        np.float32(light),
        np.float32(saturation),
        np.float32(max_norm),
        np.float32(time_step),
        out.data_ptr()
    )

    launch(s, config, ker, *ker_args)

    s.sync()

    out = out.view(torch.uint8).reshape(num_batches, height, width, 4)
    return out

tensor = run_kernel(height=1024, width=1024, num_batches=40, time_step=0.2)

this_dir = os.path.dirname(__file__)

show_image('domain_coloring_output.png', tensor, image_num=39)

tensor = run_kernel(height=1024, width=1024, num_batches=240, time_step=0.01)
print("Generating video")
render_video('domain_coloring_video.mp4', tensor)
