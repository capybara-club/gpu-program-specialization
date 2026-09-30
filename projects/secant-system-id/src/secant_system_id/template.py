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

from .manifest import finalize_cuda_source, make_manifest
from .model import SystemModel
from .shape import KernelShape


FIRST_MARKER_BITS = 0x7FC0FFEE
KERNEL_NAME = "ssid_score"


def _marker_assembly(shape: KernelShape) -> tuple[str, str, str]:
    marked = [f"marked{index}" for index in range(shape.input_count)]
    outputs = [f"missing_output{index}" for index in range(shape.ast_count)]
    asm_output_count = shape.input_count + shape.ast_count + 1
    keepalive_operand = asm_output_count - 1
    input_operand_start = asm_output_count
    lines = ["brkpt;"]

    for index in range(shape.input_count):
        marker = FIRST_MARKER_BITS + index
        lines.append(
            f"add.rn.ftz.f32 %{index}, %{input_operand_start + index}, 0f{marker:08x};"
        )
    lines.append("brkpt;")
    for index in range(1, shape.input_count):
        lines.append(f"add.rn.ftz.f32 %0, %0, %{index};")
    for index in range(shape.ast_count):
        marker = FIRST_MARKER_BITS + shape.input_count + index
        lines.append(
            f"add.rn.ftz.f32 %{shape.input_count + index}, %0, 0f{marker:08x};"
        )
    lines.append("brkpt;")

    output_operands = [shape.input_count + index for index in range(shape.ast_count)]
    if len(output_operands) == 1:
        lines.append(
            f"add.rn.ftz.f32 %{keepalive_operand}, %{output_operands[0]}, %{input_operand_start};"
        )
        next_input = 1
    else:
        lines.append(
            f"add.rn.ftz.f32 %{keepalive_operand}, %{output_operands[0]}, %{output_operands[1]};"
        )
        for operand in output_operands[2:]:
            lines.append(
                f"add.rn.ftz.f32 %{keepalive_operand}, %{keepalive_operand}, %{operand};"
            )
        next_input = 0
    for index in range(next_input, shape.input_count):
        lines.append(
            f"add.rn.ftz.f32 %{keepalive_operand}, %{keepalive_operand}, %{input_operand_start + index};"
        )
    lines.extend(
        [
            "mov.u32 keepalive_address, 0;",
            f"st.volatile.shared.f32 [keepalive_address], %{keepalive_operand};",
        ]
    )
    lines.extend("brkpt;" for _ in range(shape.patch_capacity))
    template = "\n".join(f'                        "{line}\\n\\t"' for line in lines)

    output_variables = marked + outputs + ["keepalive"]
    constraints = ", ".join(f'"=&f"({name})' for name in output_variables)
    inputs = ", ".join(f'"f"(leaf{index})' for index in range(shape.input_count))
    return template, constraints, inputs


def generate_cuda(model: SystemModel, shape: KernelShape) -> str:
    """Render a typed known ODE backbone and dynamic missing sites as CUDA."""

    if model.state_count != shape.state_count:
        raise ValueError("model state count does not match the kernel shape")
    if model.ast_leaf_counts != shape.ast_leaf_counts:
        raise ValueError("model missing sites do not match the kernel AST layout")

    marker_template, constraints, input_constraints = _marker_assembly(shape)
    locals_decl = "\n".join(
        f"                    float marked{index} __attribute__((unused));"
        for index in range(shape.input_count)
    )
    outputs_decl = "\n".join(
        f"                    float missing_output{index};" for index in range(shape.ast_count)
    )
    binding_loads = "\n".join(
        f"    const unsigned int binding{index}_raw = leaf_bindings[{index}ull * bindings_leading_dimension + setting];\n"
        f"    const unsigned int binding{index} = binding{index}_raw < SSID_BANK_SLOT_COUNT ? binding{index}_raw : 0u;"
        for index in range(shape.input_count)
    )
    constant_stores = "\n".join(
        f"    bank[({shape.state_count}u + {index}u) * blockDim.x + threadIdx.x] = "
        f"settings[{index}ull * settings_leading_dimension + setting];"
        for index in range(shape.constant_count)
    )
    leaf_loads = "\n".join(
        f"                    const float leaf{index} = bank[binding{index} * blockDim.x + threadIdx.x];"
        for index in range(shape.input_count)
    )
    state_loads = "\n".join(
        f"        state[{index}] = reference[{index} * SSID_TRAJECTORY_COUNT + experiment];"
        for index in range(shape.state_count)
    )
    state_bank_stores = "\n".join(
        f"                    bank[{index}u * blockDim.x + threadIdx.x] = stage_state[{index}];"
        for index in range(shape.state_count)
    )
    derivative_lines = "\n".join(
        f"                    const float derivative{index} = {expression.render()};"
        for index, expression in enumerate(model.derivatives)
    )
    sum_lines = "\n".join(
        f"                    sum_state[{index}] = fmaf(weight, derivative{index}, sum_state[{index}]);"
        for index in range(shape.state_count)
    )
    stage_update_lines = "\n".join(
        f"                        stage_state[{index}] = fmaf(stage_h, derivative{index}, base_state[{index}]);"
        for index in range(shape.state_count)
    )
    final_update_lines = "\n".join(
        f"                state[{index}] = fmaf(sixth_h, sum_state[{index}], base_state[{index}]);"
        for index in range(shape.state_count)
    )
    validity_expression = " && ".join(
        f"isfinite(state[{index}])" for index in range(shape.state_count)
    )
    error_lines = "\n".join(
        f"            const float error{index} = state[{index}] - reference[target_offset + {index} * target_plane + target];\n"
        f"            squared_error = fmaf(error{index}, error{index}, squared_error);"
        for index in range(shape.state_count)
    )
    state_names = ", ".join(
        f"{index}={name}" for index, name in enumerate(model.state_names)
    )
    missing_names = ", ".join(
        f"{index}={site.name}[{site.leaf_count}]" for index, site in enumerate(model.missing_sites)
    )
    observation_interval = format(model.observation_interval, ".9g")
    if "." not in observation_interval and "e" not in observation_interval.lower():
        observation_interval += ".0"

    body = f'''/* Generated by secant_system_id.template. Do not edit by hand.
 * model: {model.name}
 * states: {state_names}
 * missing sites: {missing_names}
 */
#define SSID_STATE_COUNT {shape.state_count}
#define SSID_CONSTANT_COUNT {shape.constant_count}
#define SSID_TRAJECTORY_COUNT {shape.trajectory_count}
#define SSID_OBSERVATION_COUNT {shape.observation_count}
#define SSID_REFERENCE_FLOAT_COUNT {shape.reference_float_count}
#define SSID_BANK_SLOT_COUNT {shape.bank_slot_count}

extern "C" __global__ void {KERNEL_NAME}(
    const float *__restrict__ settings,
    unsigned long long settings_leading_dimension,
    const unsigned int *__restrict__ leaf_bindings,
    unsigned long long bindings_leading_dimension,
    unsigned long long num_settings,
    const float *__restrict__ reference_data,
    unsigned int steps_per_observation,
    float *__restrict__ mse_out)
{{
    extern __shared__ float shared_storage[];
    float *reference = shared_storage;
    volatile float *bank = shared_storage + SSID_REFERENCE_FLOAT_COUNT;
    for (unsigned int index = threadIdx.x; index < SSID_REFERENCE_FLOAT_COUNT; index += blockDim.x) {{
        reference[index] = reference_data[index];
    }}
    __syncthreads();

    const unsigned long long setting = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (setting >= num_settings || steps_per_observation == 0u) return;

{binding_loads}
{constant_stores}

    const float h = {observation_interval}f / (float)steps_per_observation;
    const float half_h = 0.5f * h;
    const float sixth_h = h / 6.0f;
    float squared_error = 0.0f;
    int valid = 1;

#pragma unroll 1
    for (int experiment = 0; experiment < SSID_TRAJECTORY_COUNT; ++experiment) {{
        float state[SSID_STATE_COUNT];
{state_loads}

#pragma unroll 1
        for (int observation = 0; observation < SSID_OBSERVATION_COUNT; ++observation) {{
#pragma unroll 1
            for (unsigned int step = 0; step < steps_per_observation; ++step) {{
                float base_state[SSID_STATE_COUNT];
                float stage_state[SSID_STATE_COUNT];
                float sum_state[SSID_STATE_COUNT];
#pragma unroll
                for (int component = 0; component < SSID_STATE_COUNT; ++component) {{
                    base_state[component] = state[component];
                    stage_state[component] = state[component];
                    sum_state[component] = 0.0f;
                }}

#pragma unroll 1
                for (int stage = 0; stage < 4; ++stage) {{
{state_bank_stores}
{leaf_loads}
{locals_decl}
{outputs_decl}
                    float keepalive __attribute__((unused));
                    asm volatile(
                        "{{\\n\\t"
                        ".reg .u32 keepalive_address;\\n\\t"
{marker_template}
                        "}}\\n\\t"
                        : {constraints}
                        : {input_constraints}
                        : "memory");

{derivative_lines}
                    const float weight = (stage == 0 || stage == 3) ? 1.0f : 2.0f;
{sum_lines}
                    if (stage != 3) {{
                        const float stage_h = stage == 2 ? h : half_h;
{stage_update_lines}
                    }}
                }}
{final_update_lines}
            }}

            valid = valid && {validity_expression};
            const int target = experiment * SSID_OBSERVATION_COUNT + observation;
            const int target_offset = SSID_STATE_COUNT * SSID_TRAJECTORY_COUNT;
            const int target_plane = SSID_TRAJECTORY_COUNT * SSID_OBSERVATION_COUNT;
{error_lines}
        }}
    }}

    mse_out[setting] = valid ? squared_error / (float)(SSID_STATE_COUNT * SSID_TRAJECTORY_COUNT * SSID_OBSERVATION_COUNT) : 3.402823466e+38F;
}}
'''
    return finalize_cuda_source(
        body,
        make_manifest(model, shape, KERNEL_NAME, "direct"),
    )


def generate_fedbatch_cuda(shape: KernelShape | None = None) -> str:
    """Compatibility wrapper for the initial fed-batch model."""

    from .fed_batch import FED_BATCH_MODEL, FED_BATCH_SHAPE

    return generate_cuda(FED_BATCH_MODEL, FED_BATCH_SHAPE if shape is None else shape)
