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

from dataclasses import dataclass
import re
from typing import Sequence

from .model import SystemModel
from .shape import KernelShape
from .toggle_ast import ToggleSystem


TOGGLE_KERNEL_NAME = "ssid_score_toggle_cuda"


@dataclass(frozen=True)
class ToggleCudaModule:
    source: str
    kernel_names: tuple[str, ...]
    systems_per_kernel: int


def _cuda_float(value: float) -> str:
    rendered = format(float(value), ".9g")
    if "." not in rendered and "e" not in rendered.lower():
        rendered += ".0"
    return rendered + "f"


def generate_toggle_cuda(
    model: SystemModel,
    shape: KernelShape,
    systems: Sequence[ToggleSystem],
    toggle_bit_count: int,
    kernel_name: str = TOGGLE_KERNEL_NAME,
) -> str:
    """Render packed CUDA systems whose leaf choices are low configuration bits."""

    if not systems:
        raise ValueError("toggle CUDA requires at least one system")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", kernel_name):
        raise ValueError("toggle CUDA kernel name must be a C identifier")
    if isinstance(toggle_bit_count, bool) or not isinstance(toggle_bit_count, int):
        raise TypeError("toggle-bit count must be an integer")
    if not 0 <= toggle_bit_count <= 30:
        raise ValueError("toggle-bit count must be in [0, 30]")
    if model.state_count != shape.state_count or model.ast_leaf_counts != shape.ast_leaf_counts:
        raise ValueError("model and toggle kernel shape do not match")
    for system in systems:
        system.validate(shape)
        if system.required_toggle_bits > toggle_bit_count:
            raise ValueError("a packed system uses toggle bits outside the configured mask")

    cases: list[str] = []
    for system_index, system in enumerate(systems):
        assignments = "\n".join(
            f"                            missing_output{site_index} = {program.render_cuda(shape)};"
            for site_index, program in enumerate(system.programs)
        )
        cases.append(
            f"                        case {system_index}u:\n"
            f"{assignments}\n"
            "                            break;"
        )
    rendered_cases = "\n".join(cases)
    constants = "\n".join(
        f"    const float constant{index} = constant_banks[constant_base + {index}ull];"
        for index in range(shape.constant_count)
    )
    outputs = "\n".join(
        f"                        float missing_output{index};"
        for index in range(shape.ast_count)
    )
    derivative_lines = "\n".join(
        f"                        const float derivative{index} = {expression.render()};"
        for index, expression in enumerate(model.derivatives)
    )
    state_loads = "\n".join(
        f"            state[{index}] = reference[{index} * SSID_TRAJECTORY_COUNT + experiment];"
        for index in range(shape.state_count)
    )
    sum_lines = "\n".join(
        f"                        sum_state[{index}] = fmaf(weight, derivative{index}, sum_state[{index}]);"
        for index in range(shape.state_count)
    )
    stage_lines = "\n".join(
        f"                            stage_state[{index}] = fmaf(stage_h, derivative{index}, base_state[{index}]);"
        for index in range(shape.state_count)
    )
    final_lines = "\n".join(
        f"                    state[{index}] = fmaf(sixth_h, sum_state[{index}], base_state[{index}]);"
        for index in range(shape.state_count)
    )
    errors = "\n".join(
        f"                const float error{index} = state[{index}] - reference[target_offset + {index} * target_plane + target];\n"
        f"                squared_error = fmaf(error{index}, error{index}, squared_error);"
        for index in range(shape.state_count)
    )
    finite = " && ".join(f"isfinite(state[{index}])" for index in range(shape.state_count))
    state_names = ", ".join(f"{index}={name}" for index, name in enumerate(model.state_names))
    missing_names = ", ".join(
        f"{index}={site.name}[{site.leaf_count}]"
        for index, site in enumerate(model.missing_sites)
    )
    mask = (1 << toggle_bit_count) - 1
    interval = _cuda_float(model.observation_interval)

    return f'''/* Generated by secant_system_id.toggle_template. Do not edit by hand.
 * prototype: CUDA-compiled packed systems with register-resident toggle leaves
 * model: {model.name}
 * states: {state_names}
 * missing sites: {missing_names}
 * packed systems: {len(systems)}
 * toggle bits: {toggle_bit_count}
 * configuration: low bits choose AST leaves; high bits choose a constant bank
 * shared memory: read-only trajectory reference only
 */
#define SSID_STATE_COUNT {shape.state_count}
#define SSID_CONSTANT_COUNT {shape.constant_count}
#define SSID_TRAJECTORY_COUNT {shape.trajectory_count}
#define SSID_OBSERVATION_COUNT {shape.observation_count}
#define SSID_REFERENCE_FLOAT_COUNT {shape.reference_float_count}
#define SSID_PACKED_SYSTEM_COUNT {len(systems)}
#define SSID_TOGGLE_BIT_COUNT {toggle_bit_count}
#define SSID_TOGGLE_MASK {mask}u

extern "C" __global__ void {kernel_name}(
    const float *__restrict__ constant_banks,
    unsigned int num_constant_banks,
    unsigned int configuration_count,
    const float *__restrict__ reference_data,
    unsigned int steps_per_observation,
    float *__restrict__ mse_out)
{{
    const unsigned int system_index = blockIdx.y;
    const unsigned long long available_configuration_count = (unsigned long long)num_constant_banks << SSID_TOGGLE_BIT_COUNT;
    if (system_index >= SSID_PACKED_SYSTEM_COUNT || configuration_count == 0u ||
        configuration_count > available_configuration_count || steps_per_observation == 0u) return;

    extern __shared__ float reference[];
    for (unsigned int index = threadIdx.x; index < SSID_REFERENCE_FLOAT_COUNT; index += blockDim.x) {{
        reference[index] = reference_data[index];
    }}
    __syncthreads();

    const unsigned long long configuration = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (configuration >= configuration_count) return;
    const unsigned int toggle_bits = (unsigned int)configuration & SSID_TOGGLE_MASK;
    const unsigned int constant_bank = (unsigned int)(configuration >> SSID_TOGGLE_BIT_COUNT);
    const unsigned long long constant_base = ((unsigned long long)system_index * num_constant_banks + constant_bank) * SSID_CONSTANT_COUNT;
{constants}

    const float h = {interval} / (float)steps_per_observation;
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
{outputs}
                    switch (system_index) {{
{rendered_cases}
                        default:
                            return;
                    }}

{derivative_lines}
                    const float weight = (stage == 0 || stage == 3) ? 1.0f : 2.0f;
{sum_lines}
                    if (stage != 3) {{
                        const float stage_h = stage == 2 ? h : half_h;
{stage_lines}
                    }}
                }}
{final_lines}
            }}

            valid = valid && {finite};
            const int target = experiment * SSID_OBSERVATION_COUNT + observation;
            const int target_offset = SSID_STATE_COUNT * SSID_TRAJECTORY_COUNT;
            const int target_plane = SSID_TRAJECTORY_COUNT * SSID_OBSERVATION_COUNT;
{errors}
        }}
    }}

    const unsigned long long output_index = (unsigned long long)system_index * configuration_count + configuration;
    mse_out[output_index] = valid && isfinite(squared_error)
        ? squared_error / (float)(SSID_STATE_COUNT * SSID_TRAJECTORY_COUNT * SSID_OBSERVATION_COUNT)
        : 3.402823466e+38F;
}}
'''


def generate_toggle_cuda_module(
    model: SystemModel,
    shape: KernelShape,
    systems: Sequence[ToggleSystem],
    systems_per_kernel: int,
    toggle_bit_count: int,
) -> ToggleCudaModule:
    """Render equal-sized independent entry points in one CUDA module."""

    if isinstance(systems_per_kernel, bool) or not isinstance(systems_per_kernel, int):
        raise TypeError("systems per toggle kernel must be an integer")
    if systems_per_kernel <= 0 or len(systems) % systems_per_kernel != 0:
        raise ValueError("systems must divide evenly into positive equal-sized kernels")
    names = tuple(
        f"{TOGGLE_KERNEL_NAME}_{index}"
        for index in range(len(systems) // systems_per_kernel)
    )
    sources = [
        generate_toggle_cuda(
            model,
            shape,
            systems[index * systems_per_kernel : (index + 1) * systems_per_kernel],
            toggle_bit_count,
            name,
        )
        for index, name in enumerate(names)
    ]
    return ToggleCudaModule("\n".join(sources), names, systems_per_kernel)
