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
"""Python CUDA source-generator reference."""

from __future__ import annotations

from .manifest import finalize_cuda_source, make_manifest
from .model import KERNEL_NAME, KernelShape


FIRST_MARKER_BITS = 0x7FC0A11E
SHARED_BEGIN_MARKER_BITS = 0x7FC1A11E
SHARED_END_MARKER_BITS = 0x7FC2A11E
OUTPUT_LIVE_MARKER_BITS = 0x7FC3A11E


def _cuda_float(value: float) -> str:
    rendered = format(float(value), ".9g")
    if "." not in rendered and "e" not in rendered.lower():
        rendered += ".0"
    return rendered + "f"


def _marker_assembly(shape: KernelShape) -> tuple[str, str, str]:
    marked = [f"marked{index}" for index in range(shape.input_count)]
    outputs = [f"rhs{index}" for index in range(shape.output_count)]
    boundary_begin_operand = shape.input_count + shape.output_count
    boundary_end_operand = boundary_begin_operand + 1
    output_operand_count = shape.input_count + shape.output_count + 3
    keepalive_operand = output_operand_count - 1
    input_operand_start = output_operand_count
    permutation_operand = input_operand_start + shape.input_count
    dispatch_operand = permutation_operand + 1
    targets = ", ".join(f"odezza_system_{index}" for index in range(shape.system_capacity))
    arena_capacity = shape.arena_instruction_count
    placeholder_instructions = 2 * shape.system_capacity
    if arena_capacity < placeholder_instructions:
        raise ValueError("the specialization arena needs at least two instructions per system target")

    lines = ["brkpt;"]
    for index in range(shape.input_count):
        marker = FIRST_MARKER_BITS + index
        lines.append(f"add.rn.ftz.f32 %{index}, %{input_operand_start + index}, 0f{marker:08x};")
    lines.append("brkpt;")
    for index in range(1, shape.input_count):
        lines.append(f"add.rn.ftz.f32 %0, %0, %{index};")
    lines.extend(
        (
            ".reg .pred odezza_toggle_predicate;",
            ".reg .b32 odezza_toggle_masked;",
            f"and.b32 odezza_toggle_masked, %{permutation_operand}, 1;",
            "setp.ne.u32 odezza_toggle_predicate, odezza_toggle_masked, 0;",
            "selp.f32 %0, %0, %1, odezza_toggle_predicate;",
        )
    )
    for index in range(shape.output_count):
        marker = OUTPUT_LIVE_MARKER_BITS + index
        lines.append(
            f"add.rn.ftz.f32 %{shape.input_count + index}, %0, 0f{marker:08x};"
        )
    lines.append(f"add.rn.ftz.f32 %{boundary_begin_operand}, %{input_operand_start}, 0f{SHARED_BEGIN_MARKER_BITS:08x};")
    lines.extend("brkpt;" for _ in range(shape.shared_patch_capacity))
    lines.append(f"add.rn.ftz.f32 %{boundary_end_operand}, %{input_operand_start}, 0f{SHARED_END_MARKER_BITS:08x};")
    lines.extend(
        (
            f"odezza_system_targets: .branchtargets {targets};",
            f"brx.idx.uni %{dispatch_operand}, odezza_system_targets;",
        )
    )
    for index in range(shape.system_capacity):
        lines.extend((f"odezza_system_{index}:", "brkpt;"))
        if index == shape.system_capacity - 1:
            lines.extend("brkpt;" for _ in range(arena_capacity - placeholder_instructions))
        lines.append("bra.uni odezza_system_continue;")
    lines.append("odezza_system_continue:")

    for index in range(shape.output_count):
        marker = FIRST_MARKER_BITS + shape.input_count + index
        lines.append(
            f"add.rn.ftz.f32 %{shape.input_count + index}, %{shape.input_count + index}, 0f{marker:08x};"
        )
    lines.append("brkpt;")
    output_operands = [shape.input_count + index for index in range(shape.output_count)]
    if len(output_operands) == 1:
        lines.append(f"add.rn.ftz.f32 %{keepalive_operand}, %{output_operands[0]}, %{input_operand_start};")
        next_input = 1
    else:
        lines.append(f"add.rn.ftz.f32 %{keepalive_operand}, %{output_operands[0]}, %{output_operands[1]};")
        for operand in output_operands[2:]:
            lines.append(f"add.rn.ftz.f32 %{keepalive_operand}, %{keepalive_operand}, %{operand};")
        next_input = 0
    for index in range(next_input, shape.input_count):
        lines.append(f"add.rn.ftz.f32 %{keepalive_operand}, %{keepalive_operand}, %{input_operand_start + index};")
    lines.append(f"add.rn.ftz.f32 %{keepalive_operand}, %{keepalive_operand}, %{boundary_begin_operand};")
    lines.append(f"add.rn.ftz.f32 %{keepalive_operand}, %{keepalive_operand}, %{boundary_end_operand};")
    lines.extend(("mov.u32 keepalive_address, 0;", f"st.volatile.shared.f32 [keepalive_address], %{keepalive_operand};", "brkpt;"))

    assembly = "\n".join(f'                        "{line}\\n\\t"' for line in lines)
    constraints = ", ".join(
        f'"=&f"({name})'
        for name in (*marked, *outputs, "shared_boundary_begin", "shared_boundary_end", "keepalive")
    )
    inputs = ", ".join(
        [*(f'"f"(site_input{index})' for index in range(shape.input_count)), '"r"(permutation)', '"r"(system_index)']
    )
    return assembly, constraints, inputs


def generate_cuda(shape: KernelShape = KernelShape()) -> str:
    """Generate the multi-system scorer with a shared SASS prelude and compact branches."""

    marker, constraints, inputs = _marker_assembly(shape)
    input_declarations: list[str] = []
    for index in range(shape.state_count):
        input_declarations.append(f"                    const float site_input{index} = stage_state[{index}];")
    for index in range(shape.constant_count):
        input_index = shape.state_count + index
        input_declarations.append(f"                    const float site_input{input_index} = constant{index};")
    marked = "\n".join(f"                    float marked{index} __attribute__((unused));" for index in range(shape.input_count))
    rhs = "\n".join(f"                    float rhs{index};" for index in range(shape.output_count))
    constants = "\n".join(
        f"    const float constant{index} = constant_banks[((unsigned long long)system_index * constant_bank_count + constant_bank) * ODEZZA_CONSTANT_COUNT + {index}ull];"
        for index in range(shape.constant_count)
    )
    initial = "\n".join(
        f"        state[{index}] = reference[{index} * ODEZZA_TRAJECTORY_COUNT + trajectory];"
        for index in range(shape.state_count)
    )
    sums = "\n".join(
        f"                    sum_state[{index}] = fmaf(weight, rhs{index}, sum_state[{index}]);"
        for index in range(shape.state_count)
    )
    stages = "\n".join(
        f"                        stage_state[{index}] = fmaf(stage_h, rhs{index}, base_state[{index}]);"
        for index in range(shape.state_count)
    )
    finals = "\n".join(
        f"                state[{index}] = fmaf(sixth_h, sum_state[{index}], base_state[{index}]);"
        for index in range(shape.state_count)
    )
    errors = "\n".join(
        f"            const float error{index} = state[{index}] - reference[target_offset + {index} * target_plane + target];\n"
        f"            squared_error = fmaf(error{index}, error{index}, squared_error);"
        for index in range(shape.state_count)
    )
    finite = " && ".join(f"isfinite(state[{index}])" for index in range(shape.state_count))
    body = f'''/* Generated CUDA template. Do not edit by hand.
 * shape: scoring
 * ownership: grid-y selects one candidate system; each thread owns one configuration
 * integration: fused RK4 over dense aligned trajectories
 * specialization: common derivative prelude followed by uniform candidate-system dispatch
 * equations: every shared and candidate expression is inserted by the postorder specializer
 */
#define ODEZZA_STATE_COUNT {shape.state_count}
#define ODEZZA_CONSTANT_COUNT {shape.constant_count}
#define ODEZZA_TRAJECTORY_COUNT {shape.trajectory_count}
#define ODEZZA_OBSERVATION_COUNT {shape.observation_count}
#define ODEZZA_REFERENCE_FLOAT_COUNT {shape.reference_float_count}
#define ODEZZA_SYSTEM_CAPACITY {shape.system_capacity}u

extern "C" __global__ void {KERNEL_NAME}(
    const float *__restrict__ constant_banks,
    unsigned int constant_bank_count,
    const float *__restrict__ reference_data,
    unsigned int active_toggle_count,
    unsigned int active_system_count,
    unsigned int steps_per_observation,
    float *__restrict__ mse_out)
{{
    extern __shared__ float reference[];
    for (unsigned int index = threadIdx.x; index < ODEZZA_REFERENCE_FLOAT_COUNT; index += blockDim.x) {{
        reference[index] = reference_data[index];
    }}
    __syncthreads();

    const unsigned int system_index = blockIdx.y;
    const unsigned long long configuration = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long configuration_count = (unsigned long long)constant_bank_count << active_toggle_count;
    if (active_toggle_count > 32u || system_index >= active_system_count || system_index >= ODEZZA_SYSTEM_CAPACITY || configuration >= configuration_count || constant_bank_count == 0u || steps_per_observation == 0u) return;
    const unsigned int permutation = (unsigned int)configuration;
    const unsigned int constant_bank = (unsigned int)(configuration >> active_toggle_count);
{constants}

    const float h = {_cuda_float(shape.observation_interval)} / (float)steps_per_observation;
    const float half_h = 0.5f * h;
    const float sixth_h = h / 6.0f;
    float squared_error = 0.0f;
    int valid = 1;

#pragma unroll 1
    for (int trajectory = 0; trajectory < ODEZZA_TRAJECTORY_COUNT; ++trajectory) {{
        float state[ODEZZA_STATE_COUNT];
{initial}

#pragma unroll 1
        for (int observation = 0; observation < ODEZZA_OBSERVATION_COUNT; ++observation) {{
#pragma unroll 1
            for (unsigned int step = 0; step < steps_per_observation; ++step) {{
                float base_state[ODEZZA_STATE_COUNT];
                float stage_state[ODEZZA_STATE_COUNT];
                float sum_state[ODEZZA_STATE_COUNT];
#pragma unroll
                for (int component = 0; component < ODEZZA_STATE_COUNT; ++component) {{
                    base_state[component] = state[component];
                    stage_state[component] = state[component];
                    sum_state[component] = 0.0f;
                }}

#pragma unroll 1
                for (int rk_stage = 0; rk_stage < 4; ++rk_stage) {{
{chr(10).join(input_declarations)}
{marked}
{rhs}
                    float shared_boundary_begin __attribute__((unused));
                    float shared_boundary_end __attribute__((unused));
                    float keepalive __attribute__((unused));
                    asm volatile(
                        "{{\\n\\t"
                        ".reg .u32 keepalive_address;\\n\\t"
{marker}
                        "}}\\n\\t"
                        : {constraints}
                        : {inputs}
                        : "memory");

                    const float weight = (rk_stage == 0 || rk_stage == 3) ? 1.0f : 2.0f;
{sums}
                    if (rk_stage != 3) {{
                        const float stage_h = rk_stage == 2 ? h : half_h;
{stages}
                    }}
                }}
{finals}
            }}

            valid = valid && {finite};
            const int target = trajectory * ODEZZA_OBSERVATION_COUNT + observation;
            const int target_offset = ODEZZA_STATE_COUNT * ODEZZA_TRAJECTORY_COUNT;
            const int target_plane = ODEZZA_TRAJECTORY_COUNT * ODEZZA_OBSERVATION_COUNT;
{errors}
        }}
    }}

    const unsigned long long output_index = (unsigned long long)system_index * configuration_count + configuration;
    mse_out[output_index] = valid && isfinite(squared_error)
        ? squared_error / (float)(ODEZZA_STATE_COUNT * ODEZZA_TRAJECTORY_COUNT * ODEZZA_OBSERVATION_COUNT)
        : 3.402823466e+38F;
}}
'''
    return finalize_cuda_source(body, make_manifest(shape))
