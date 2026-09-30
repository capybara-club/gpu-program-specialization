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

import math

from .manifest import finalize_cuda_source, make_manifest, make_packed_module_manifest
from .model import SystemModel
from .shape import KernelShape, PackedDispatch, PackedKernelSpec
from .template import FIRST_MARKER_BITS


PACKED_KERNEL_NAME = "ssid_score_packed"
WINNER_REDUCTION_KERNEL_NAME = "ssid_reduce_winners"


def _cuda_float(value: float) -> str:
    rendered = format(float(value), ".9g")
    return rendered if any(marker in rendered for marker in (".", "e")) else rendered + ".0"


def _marker_assembly(
    shape: KernelShape,
    dispatch: PackedDispatch,
    label_prefix: str,
) -> tuple[str, str, str]:
    """Build one BRX/BRXU dispatch site followed by a compact code arena."""

    marked = [f"marked{index}" for index in range(shape.input_count)]
    outputs = [f"missing_output{index}" for index in range(shape.ast_count)]
    asm_output_count = shape.input_count + shape.ast_count + 1
    keepalive_operand = asm_output_count - 1
    input_operand_start = asm_output_count
    dispatch_operand = input_operand_start + shape.input_count
    targets = ", ".join(f"{label_prefix}_genome_{index}" for index in range(dispatch.genome_capacity))
    arena_capacity = shape.patch_capacity * dispatch.genome_capacity
    placeholder_instructions = 2 * dispatch.genome_capacity
    if arena_capacity < placeholder_instructions:
        raise ValueError(
            "the packed arena needs at least two instructions per target-table entry"
        )

    lines = ["brkpt;"]
    for index in range(shape.input_count):
        marker = FIRST_MARKER_BITS + index
        lines.append(
            f"add.rn.ftz.f32 %{index}, %{input_operand_start + index}, 0f{marker:08x};"
        )
    lines.append("brkpt;")
    for index in range(1, shape.input_count):
        lines.append(f"add.rn.ftz.f32 %0, %0, %{index};")

    lines.extend(
        (
            f"{label_prefix}_genome_targets: .branchtargets {targets};",
            f"brx.idx.uni %{dispatch_operand}, {label_prefix}_genome_targets;",
        )
    )
    for index in range(dispatch.genome_capacity):
        lines.extend((f"{label_prefix}_genome_{index}:", "brkpt;"))
        # PTXAS discards unreachable inline-PTX tail instructions.  Keeping the
        # reserve on the final reachable target preserves one contiguous arena;
        # specialization rewrites that target before the module can run.
        if index == dispatch.genome_capacity - 1:
            lines.extend(
                "brkpt;"
                for _ in range(arena_capacity - placeholder_instructions)
            )
        lines.append(f"bra.uni {label_prefix}_genome_continue;")
    lines.append(f"{label_prefix}_genome_continue:")

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
            f"add.rn.ftz.f32 %{keepalive_operand}, %{keepalive_operand}, "
            f"%{input_operand_start + index};"
        )
    lines.extend(
        (
            "mov.u32 keepalive_address, 0;",
            f"st.volatile.shared.f32 [keepalive_address], %{keepalive_operand};",
            "brkpt;",
        )
    )

    template = "\n".join(f'                            "{line}\\n\\t"' for line in lines)
    output_variables = marked + outputs + ["keepalive"]
    constraints = ", ".join(f'"=&f"({name})' for name in output_variables)
    inputs = ", ".join(
        [
            *(f'"f"(leaf{index})' for index in range(shape.input_count)),
            '"r"(dispatch_index)',
        ]
    )
    return template, constraints, inputs


def _render_packed_kernel(
    model: SystemModel,
    shape: KernelShape,
    dispatch: PackedDispatch,
    kernel_name: str,
    genome_base: int,
    label_prefix: str,
    module_kernel_count: int,
    include_preamble: bool,
    winner_output: bool,
    hashed_settings: bool,
    constant_mutation_scale: float,
    binding_keep_probability: float,
    relative_error_floor: float | None,
    reject_exact_zero_score: bool,
    warp_per_system: bool,
) -> str:
    if model.state_count != shape.state_count:
        raise ValueError("model state count does not match the kernel shape")
    if model.ast_leaf_counts != shape.ast_leaf_counts:
        raise ValueError("model missing sites do not match the kernel AST layout")

    marker_template, constraints, input_constraints = _marker_assembly(shape, dispatch, label_prefix)
    locals_decl = "\n".join(
        f"                        float marked{index} __attribute__((unused));"
        for index in range(shape.input_count)
    )
    outputs_decl = "\n".join(
        f"                        float missing_output{index};"
        for index in range(shape.ast_count)
    )
    if hashed_settings:
        scale = _cuda_float(constant_mutation_scale)
        keep = _cuda_float(binding_keep_probability)
        binding_loads = "\n".join(
            f"        const unsigned int binding{index}_incumbent = leaf_bindings[(unsigned long long)genome * SSID_INPUT_COUNT + {index}ull];\n"
            f"        const unsigned long long binding{index}_key = settings_leading_dimension ^ 0xa0761d6478bd642full ^ (random_generation << 48u) ^ (global_genome << 28u) ^ ({index}ull << 20u) ^ setting;\n"
            f"        const unsigned int binding{index}_raw = setting == 0ull || ssid_gp_hash_unit(binding{index}_key) < {keep}f ? binding{index}_incumbent : (unsigned int)(ssid_gp_mix64(binding{index}_key ^ 0xe7037ed1a0b428dbull) % SSID_BANK_SLOT_COUNT);\n"
            f"        const unsigned int binding{index} = binding{index}_raw < SSID_BANK_SLOT_COUNT ? binding{index}_raw : 0u;"
            for index in range(shape.input_count)
        )
        constant_stores = "\n".join(
            f"        const float constant{index}_incumbent = settings[(unsigned long long)genome * SSID_CONSTANT_COUNT + {index}ull];\n"
            f"        const unsigned long long constant{index}_key = settings_leading_dimension ^ (random_generation << 48u) ^ (global_genome << 28u) ^ ({index}ull << 20u) ^ setting;\n"
            f"        const float constant{index}_value = setting == 0ull ? constant{index}_incumbent : constant{index}_incumbent + (2.0f * ssid_gp_hash_unit(constant{index}_key) - 1.0f) * (fabsf(constant{index}_incumbent) + 1.0f) * {scale}f;\n"
            f"        bank[({shape.state_count}u + {index}u) * blockDim.x + threadIdx.x] = constant{index}_value;"
            for index in range(shape.constant_count)
        )
        settings_contract = "stateless hashed settings from per-genome incumbents"
        random_coordinates = '''    const unsigned long long random_generation = bindings_leading_dimension >> 32u;
    const unsigned long long population_base = bindings_leading_dimension & 0xffffffffull;'''
        global_genome_line = "        const unsigned long long global_genome = population_base + genome;"
        hash_helpers = '''__device__ __forceinline__ unsigned long long ssid_gp_mix64(unsigned long long value)
{
    value += 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27u)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31u);
}

__device__ __forceinline__ float ssid_gp_hash_unit(unsigned long long value)
{
    return (float)(ssid_gp_mix64(value) >> 40u) * (1.0f / 16777216.0f);
}

'''
    else:
        binding_loads = "\n".join(
            f"        const unsigned int binding{index}_raw = leaf_bindings[((unsigned long long)genome * SSID_INPUT_COUNT + {index}ull) * bindings_leading_dimension + setting];\n"
            f"        const unsigned int binding{index} = binding{index}_raw < SSID_BANK_SLOT_COUNT ? binding{index}_raw : 0u;"
            for index in range(shape.input_count)
        )
        constant_stores = "\n".join(
            f"        bank[({shape.state_count}u + {index}u) * blockDim.x + threadIdx.x] = settings[((unsigned long long)genome * SSID_CONSTANT_COUNT + {index}ull) * settings_leading_dimension + setting];"
            for index in range(shape.constant_count)
        )
        settings_contract = "materialized genome-major settings"
        random_coordinates = ""
        global_genome_line = ""
        hash_helpers = ""
    leaf_loads = "\n".join(
        f"                        const float leaf{index} = bank[binding{index} * blockDim.x + threadIdx.x];"
        for index in range(shape.input_count)
    )
    state_loads = "\n".join(
        f"            state[{index}] = reference[{index} * SSID_TRAJECTORY_COUNT + experiment];"
        for index in range(shape.state_count)
    )
    state_bank_stores = "\n".join(
        f"                        bank[{index}u * blockDim.x + threadIdx.x] = stage_state[{index}];"
        for index in range(shape.state_count)
    )
    derivative_lines = "\n".join(
        f"                        const float derivative{index} = {expression.render()};"
        for index, expression in enumerate(model.derivatives)
    )
    sum_lines = "\n".join(
        f"                        sum_state[{index}] = fmaf(weight, derivative{index}, sum_state[{index}]);"
        for index in range(shape.state_count)
    )
    stage_update_lines = "\n".join(
        f"                            stage_state[{index}] = fmaf(stage_h, derivative{index}, base_state[{index}]);"
        for index in range(shape.state_count)
    )
    final_update_lines = "\n".join(
        f"                    state[{index}] = fmaf(sixth_h, sum_state[{index}], base_state[{index}]);"
        for index in range(shape.state_count)
    )
    validity_expression = " && ".join(
        f"isfinite(state[{index}])" for index in range(shape.state_count)
    )
    if relative_error_floor is None:
        error_lines = "\n".join(
            f"                const float error{index} = state[{index}] - reference[target_offset + {index} * target_plane + target];\n"
            f"                squared_error = fmaf(error{index}, error{index}, squared_error);"
            for index in range(shape.state_count)
        )
        score_contract = "unweighted state-space MSE"
    else:
        if not math.isfinite(relative_error_floor) or relative_error_floor <= 0.0:
            raise ValueError("relative_error_floor must be finite and positive")
        floor = _cuda_float(relative_error_floor)
        error_lines = "\n".join(
            f"                const float target{index}_value = reference[target_offset + {index} * target_plane + target];\n"
            f"                const float error{index} = (state[{index}] - target{index}_value) / fmaxf(fabsf(target{index}_value), {floor}f);\n"
            f"                squared_error = fmaf(error{index}, error{index}, squared_error);"
            for index in range(shape.state_count)
        )
        score_contract = f"relative state-space MSE with denominator floor {floor}"
    state_names = ", ".join(
        f"{index}={name}" for index, name in enumerate(model.state_names)
    )
    missing_names = ", ".join(
        f"{index}={site.name}[{site.leaf_count}]"
        for index, site in enumerate(model.missing_sites)
    )
    observation_interval = format(model.observation_interval, ".9g")
    if "." not in observation_interval and "e" not in observation_interval.lower():
        observation_interval += ".0"

    score_validity = "valid && isfinite(squared_error)"
    if reject_exact_zero_score:
        score_validity += " && squared_error > 0.0f"
    if winner_output and warp_per_system:
        output_parameters = '''float *__restrict__ cta_best_score_out,
    unsigned int *__restrict__ cta_best_setting_out'''
        shared_scratch = ""
        setting_validation = "    const int active_setting = setting < num_settings;"
        computation_open = '''        float setting_score = 3.402823466e+38F;
        if (active_setting) {'''
        score_output = '''            setting_score = SSID_SCORE_VALIDITY ? squared_error / (float)(SSID_STATE_COUNT * SSID_TRAJECTORY_COUNT * SSID_OBSERVATION_COUNT) : 3.402823466e+38F;
        }

        unsigned int winning_setting = active_setting ? (unsigned int)setting : 0xffffffffu;
        const unsigned int active_mask = __activemask();
#pragma unroll
        for (int delta = 16; delta != 0; delta >>= 1) {
            const float other_score = __shfl_down_sync(active_mask, setting_score, delta);
            const unsigned int other_setting = __shfl_down_sync(active_mask, winning_setting, delta);
            if (other_score < setting_score || (other_score == setting_score && other_setting < winning_setting)) {
                setting_score = other_score;
                winning_setting = other_setting;
            }
        }
        if (lane == 0u) {
            const unsigned long long setting_tiles = (num_settings + 31u) / 32u;
            const unsigned long long output_index = (unsigned long long)genome * setting_tiles + blockIdx.x;
            cta_best_score_out[output_index] = setting_score;
            cta_best_setting_out[output_index] = winning_setting;
        }'''.replace("SSID_SCORE_VALIDITY", score_validity)
        output_description = "warp-local winning score and setting"
    elif winner_output:
        output_parameters = '''float *__restrict__ cta_best_score_out,
    unsigned int *__restrict__ cta_best_setting_out'''
        shared_scratch = '''    float *score_scratch = shared_storage + SSID_REFERENCE_FLOAT_COUNT + SSID_BANK_SLOT_COUNT * blockDim.x;
    unsigned int *setting_scratch = (unsigned int *)(score_scratch + blockDim.x);'''
        setting_validation = "    const int active_setting = setting < num_settings;"
        computation_open = '''        float setting_score = 3.402823466e+38F;
        if (active_setting) {'''
        score_output = '''            setting_score = SSID_SCORE_VALIDITY ? squared_error / (float)(SSID_STATE_COUNT * SSID_TRAJECTORY_COUNT * SSID_OBSERVATION_COUNT) : 3.402823466e+38F;
        }

        score_scratch[threadIdx.x] = setting_score;
        setting_scratch[threadIdx.x] = active_setting ? (unsigned int)setting : 0xffffffffu;
        __syncthreads();
        for (unsigned int stride = blockDim.x >> 1; stride != 0u; stride >>= 1) {
            if (threadIdx.x < stride) {
                const float other_score = score_scratch[threadIdx.x + stride];
                const unsigned int other_setting = setting_scratch[threadIdx.x + stride];
                const float own_score = score_scratch[threadIdx.x];
                const unsigned int own_setting = setting_scratch[threadIdx.x];
                if (other_score < own_score || (other_score == own_score && other_setting < own_setting)) {
                    score_scratch[threadIdx.x] = other_score;
                    setting_scratch[threadIdx.x] = other_setting;
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0u) {
            const unsigned long long setting_tiles = (num_settings + blockDim.x - 1u) / blockDim.x;
            const unsigned long long output_index = (unsigned long long)genome * setting_tiles + blockIdx.x;
            cta_best_score_out[output_index] = score_scratch[0];
            cta_best_setting_out[output_index] = setting_scratch[0];
        }
        __syncthreads();'''.replace("SSID_SCORE_VALIDITY", score_validity)
        output_description = "CTA-local winning score and setting"
    else:
        output_parameters = "float *__restrict__ mse_out"
        shared_scratch = ""
        setting_validation = '''    if (setting >= num_settings || steps_per_observation == 0u) return;'''
        computation_open = ""
        score_output = '''        const unsigned long long output_index = (unsigned long long)genome * num_settings + setting;
        mse_out[output_index] = SSID_SCORE_VALIDITY ? squared_error / (float)(SSID_STATE_COUNT * SSID_TRAJECTORY_COUNT * SSID_OBSERVATION_COUNT) : 3.402823466e+38F;'''.replace("SSID_SCORE_VALIDITY", score_validity)
        output_description = "one MSE per genome and setting"

    ownership_description = (
        "one system per warp, one setting per lane"
        if warp_per_system
        else "one or more serial systems per CTA, one setting per thread"
    )
    preamble = f'''/* Generated by secant_system_id.packed_template. Do not edit by hand.
 * model: {model.name}
 * states: {state_names}
 * missing sites: {missing_names}
 * module: {module_kernel_count} packed kernel entry points
 * dispatch per kernel: {dispatch.genome_capacity} packed genomes, {dispatch.genomes_per_cta} per CTA
 * ownership: {ownership_description}
 * output: {output_description}
 * settings: {settings_contract}
 * score: {score_contract}
 */
#define SSID_STATE_COUNT {shape.state_count}
#define SSID_CONSTANT_COUNT {shape.constant_count}
#define SSID_INPUT_COUNT {shape.input_count}
#define SSID_TRAJECTORY_COUNT {shape.trajectory_count}
#define SSID_OBSERVATION_COUNT {shape.observation_count}
#define SSID_REFERENCE_FLOAT_COUNT {shape.reference_float_count}
#define SSID_BANK_SLOT_COUNT {shape.bank_slot_count}
#define SSID_KERNEL_GENOME_CAPACITY {dispatch.genome_capacity}
#define SSID_GENOMES_PER_CTA {dispatch.genomes_per_cta}
#define SSID_MODULE_KERNEL_COUNT {module_kernel_count}
#define SSID_MODULE_GENOME_CAPACITY {module_kernel_count * dispatch.genome_capacity}

{hash_helpers}''' if include_preamble else f'''/* {kernel_name}: genomes {genome_base} through {genome_base + dispatch.genome_capacity - 1}. */

'''

    launch_guard = (
        f'''    if ((blockDim.x & 31u) != 0u || blockDim.x / 32u < SSID_GENOMES_PER_CTA || num_settings > 32u) return;'''
        if warp_per_system
        else ""
    )
    setting_coordinates = (
        '''    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int local_warp = threadIdx.x >> 5u;
    const unsigned long long setting = (unsigned long long)blockIdx.x * 32ull + lane;'''
        if warp_per_system
        else '''    const unsigned long long setting = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;'''
    )
    genome_coordinates = (
        f'''    if (local_warp >= SSID_GENOMES_PER_CTA) return;
    const unsigned int first_genome = {genome_base}u + blockIdx.y * SSID_GENOMES_PER_CTA + local_warp;
    const unsigned int genome_iterations = 1u;'''
        if warp_per_system
        else f'''    const unsigned int first_genome = {genome_base}u + blockIdx.y * SSID_GENOMES_PER_CTA;'''
    )
    genome_loop_limit = "genome_iterations" if warp_per_system else "SSID_GENOMES_PER_CTA"

    body = f'''{preamble}extern "C" __global__ void {kernel_name}(
    const float *__restrict__ settings,
    unsigned long long settings_leading_dimension,
    const unsigned int *__restrict__ leaf_bindings,
    unsigned long long bindings_leading_dimension,
    unsigned long long num_settings,
    unsigned int num_genomes,
    const float *__restrict__ reference_data,
    unsigned int steps_per_observation,
    {output_parameters})
{{
{launch_guard}
    extern __shared__ float shared_storage[];
    float *reference = shared_storage;
    volatile float *bank = shared_storage + SSID_REFERENCE_FLOAT_COUNT;
{shared_scratch}
    for (unsigned int index = threadIdx.x; index < SSID_REFERENCE_FLOAT_COUNT; index += blockDim.x) {{
        reference[index] = reference_data[index];
    }}
    __syncthreads();

{setting_coordinates}
{setting_validation}
{random_coordinates}

    const float h = {observation_interval}f / (float)steps_per_observation;
    const float half_h = 0.5f * h;
    const float sixth_h = h / 6.0f;
{genome_coordinates}

#pragma unroll 1
    for (unsigned int local_genome = 0; local_genome < {genome_loop_limit}; ++local_genome) {{
        const unsigned int genome = first_genome + local_genome;
        const unsigned int dispatch_index = genome - {genome_base}u;
        if (genome >= num_genomes || dispatch_index >= SSID_KERNEL_GENOME_CAPACITY) break;
{global_genome_line}

{computation_open}
{binding_loads}
{constant_stores}

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

{score_output}
    }}
}}
'''
    return body


def _render_winner_reduction_kernel() -> str:
    return f'''extern "C" __global__ void {WINNER_REDUCTION_KERNEL_NAME}(
    const float *__restrict__ cta_scores,
    const unsigned int *__restrict__ cta_settings,
    unsigned long long num_settings,
    unsigned int num_genomes,
    unsigned int setting_tiles,
    float *__restrict__ best_score_out,
    unsigned int *__restrict__ best_setting_out)
{{
    const unsigned int genome = blockIdx.x;
    if (genome >= num_genomes) return;
    extern __shared__ unsigned char reduction_storage[];
    float *score_scratch = (float *)reduction_storage;
    unsigned int *setting_scratch = (unsigned int *)(score_scratch + blockDim.x);
    float best_score = 3.402823466e+38F;
    unsigned int best_setting = 0xffffffffu;
    for (unsigned int tile = threadIdx.x; tile < setting_tiles; tile += blockDim.x) {{
        const unsigned long long index = (unsigned long long)genome * setting_tiles + tile;
        const float candidate_score = cta_scores[index];
        const unsigned int candidate_setting = cta_settings[index];
        if (candidate_setting < num_settings && isfinite(candidate_score) &&
            (candidate_score < best_score || (candidate_score == best_score && candidate_setting < best_setting))) {{
            best_score = candidate_score;
            best_setting = candidate_setting;
        }}
    }}
    score_scratch[threadIdx.x] = best_score;
    setting_scratch[threadIdx.x] = best_setting;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride != 0u; stride >>= 1) {{
        if (threadIdx.x < stride) {{
            const float other_score = score_scratch[threadIdx.x + stride];
            const unsigned int other_setting = setting_scratch[threadIdx.x + stride];
            const float own_score = score_scratch[threadIdx.x];
            const unsigned int own_setting = setting_scratch[threadIdx.x];
            if (other_score < own_score || (other_score == own_score && other_setting < own_setting)) {{
                score_scratch[threadIdx.x] = other_score;
                setting_scratch[threadIdx.x] = other_setting;
            }}
        }}
        __syncthreads();
    }}
    if (threadIdx.x == 0u) {{
        best_score_out[genome] = score_scratch[0];
        best_setting_out[genome] = setting_scratch[0] < num_settings ? setting_scratch[0] : 0u;
    }}
}}
'''


def generate_packed_cuda(
    model: SystemModel,
    shape: KernelShape,
    dispatch: PackedDispatch = PackedDispatch(),
    winner_output: bool = False,
    hashed_settings: bool = False,
    constant_mutation_scale: float = 0.5,
    binding_keep_probability: float = 0.5,
    relative_error_floor: float | None = None,
    reject_exact_zero_score: bool = False,
    warp_per_system: bool = False,
) -> str:
    """Render one packed, thread-owned trajectory kernel."""

    body = _render_packed_kernel(
        model,
        shape,
        dispatch,
        PACKED_KERNEL_NAME,
        0,
        "ssid",
        1,
        True,
        winner_output,
        hashed_settings,
        constant_mutation_scale,
        binding_keep_probability,
        relative_error_floor,
        reject_exact_zero_score,
        warp_per_system,
    )
    if winner_output:
        body += "\n" + _render_winner_reduction_kernel()
    return finalize_cuda_source(
        body,
        make_manifest(
            model,
            shape,
            PACKED_KERNEL_NAME,
            "packed",
            dispatch,
            "warp_per_system" if warp_per_system else "cta_serial",
        ),
    )


def generate_packed_cuda_module(
    model: SystemModel,
    shape: KernelShape,
    kernel_count: int,
    dispatch: PackedDispatch = PackedDispatch(),
    winner_output: bool = False,
    hashed_settings: bool = False,
    constant_mutation_scale: float = 0.5,
    binding_keep_probability: float = 0.5,
    relative_error_floor: float | None = None,
    reject_exact_zero_score: bool = False,
    warp_per_system: bool = False,
) -> str:
    """Render several independently specializable packed kernels in one module."""

    if isinstance(kernel_count, bool) or not isinstance(kernel_count, int) or kernel_count <= 0:
        raise ValueError("kernel_count must be a positive integer")
    specs = tuple(
        PackedKernelSpec(
            name=f"{PACKED_KERNEL_NAME}_{index}",
            genome_base=index * dispatch.genome_capacity,
            dispatch=dispatch,
        )
        for index in range(kernel_count)
    )
    bodies = [
        _render_packed_kernel(
            model,
            shape,
            dispatch,
            spec.name,
            spec.genome_base,
            f"ssid_k{index}",
            kernel_count,
            index == 0,
            winner_output,
            hashed_settings,
            constant_mutation_scale,
            binding_keep_probability,
            relative_error_floor,
            reject_exact_zero_score,
            warp_per_system,
        )
        for index, spec in enumerate(specs)
    ]
    if winner_output:
        bodies.append(_render_winner_reduction_kernel())
    return finalize_cuda_source(
        "\n".join(bodies),
        make_packed_module_manifest(
            model,
            shape,
            specs,
            "hashed_incumbent" if hashed_settings else "materialized",
            constant_mutation_scale if hashed_settings else None,
            binding_keep_probability if hashed_settings else None,
            "warp_per_system" if warp_per_system else "cta_serial",
        ),
    )
