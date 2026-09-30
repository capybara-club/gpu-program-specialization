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
"""CUDA source generator for thread- and subgroup-owned trajectory LM."""

from __future__ import annotations

from .lm_manifest import finalize_lm_cuda_source
from .lm_model import LM_KERNEL_NAME, LM_TWO_SITE_LAYOUT, LMKernelShape


LM_SITE_MARKER_BASE = 0x4FC40000
LM_SITE_MARKER_STRIDE = 0x100
LM_OUTPUT_MARKER_OFFSET = 0x80


def site_marker_bits(site_index: int) -> int:
    return LM_SITE_MARKER_BASE + site_index * LM_SITE_MARKER_STRIDE


def _ptx_line(value: str) -> str:
    return f'                            "{value}\\n\\t"'


def _output_site(
    shape: LMKernelShape,
    site_index: int,
    result_names: tuple[str, ...],
) -> str:
    input_count = shape.input_count
    output_count = len(result_names)
    if not 1 <= output_count <= shape.maximum_site_output_count:
        raise ValueError("LM output site exceeds its inline-assembly operand budget")
    output_operand_start = input_count
    keepalive_operand = output_operand_start + output_count
    input_operand_start = keepalive_operand + 1
    permutation_operand = input_operand_start + input_count
    marker = site_marker_bits(site_index)
    lines = ["{"]
    lines.append(".reg .pred odezza_lm_toggle_predicate;")
    lines.append(".reg .b32 odezza_lm_permutation_u32;")
    lines.append(".reg .b32 odezza_lm_toggle_salted;")
    lines.append(".reg .b32 odezza_lm_toggle_masked;")
    lines.append(".reg .u32 odezza_lm_keepalive_address;")
    lines.append("brkpt;")
    for index in range(input_count):
        lines.append(
            f"add.rn.ftz.f32 %{index}, %{input_operand_start + index}, 0f{marker + index:08x};"
        )
    lines.append("brkpt;")
    lines.append(
        f"mov.b32 odezza_lm_permutation_u32, %{permutation_operand};"
    )
    lines.append(
        f"add.u32 odezza_lm_toggle_salted, odezza_lm_permutation_u32, {site_index + 1};"
    )
    lines.append(
        "and.b32 odezza_lm_toggle_masked, odezza_lm_toggle_salted, 1;"
    )
    lines.append(
        "setp.ne.u32 odezza_lm_toggle_predicate, odezza_lm_toggle_masked, 0;"
    )
    lines.append(
        f"selp.f32 %{output_operand_start}, %0, %1, odezza_lm_toggle_predicate;"
    )
    lines.append(
        f"mov.b32 %{keepalive_operand}, %{output_operand_start};"
    )
    for index in range(output_count):
        source_operand = output_operand_start if index == 0 else 0
        lines.append(
            f"add.rn.ftz.f32 %{output_operand_start + index}, %{source_operand}, "
            f"0f{marker + LM_OUTPUT_MARKER_OFFSET + index:08x};"
        )
    for index in range(output_count):
        lines.append(
            f"add.rn.ftz.f32 %{keepalive_operand}, %{keepalive_operand}, %{output_operand_start + index};"
        )
    for index in range(input_count):
        lines.append(
            f"add.rn.ftz.f32 %{keepalive_operand}, %{keepalive_operand}, %{input_operand_start + index};"
        )
    for index in range(input_count):
        lines.append(
            f"add.rn.ftz.f32 %{keepalive_operand}, %{keepalive_operand}, %{index};"
        )
    lines.append(
        f"add.rn.ftz.f32 %{keepalive_operand}, %{keepalive_operand}, %{permutation_operand};"
    )
    lines.append("mov.u32 odezza_lm_keepalive_address, 0;")
    lines.append(
        f"st.volatile.shared.f32 [odezza_lm_keepalive_address], %{keepalive_operand};"
    )
    lines.extend("brkpt;" for _ in range(shape.site_patch_capacity))
    lines.append("}")
    assembly = "\n".join(_ptx_line(line) for line in lines)
    outputs = ", ".join(
        [
            *(f'"=&f"(lm_marked_{site_index}_{index})' for index in range(input_count)),
            *(f'"=&f"({result_name})' for result_name in result_names),
            f'"=&f"(lm_keepalive_{site_index})',
        ]
    )
    inputs = ", ".join(
        [
            *(f'"f"(lm_site_input_{index})' for index in range(input_count)),
            '"f"(lm_permutation_input)',
        ]
    )
    declarations = "\n".join(
        f"                        float lm_marked_{site_index}_{index} __attribute__((unused));"
        for index in range(input_count)
    )
    result_declarations = "\n".join(
        f"                        float {result_name};"
        for result_name in result_names
    )
    return f"""{declarations}
{result_declarations}
                        float lm_keepalive_{site_index} __attribute__((unused));
                        asm volatile(
{assembly}
                            : {outputs}
                            : {inputs}
                            : \"memory\");"""


def _readwrite_output_site(
    shape: LMKernelShape,
    site_index: int,
    input_names: tuple[str, ...],
    result_names: tuple[str, ...],
) -> str:
    input_count = len(input_names)
    output_count = len(result_names)
    operand_count = input_count + output_count + 2
    if operand_count > 30:
        raise ValueError("two-site LM site exceeds its inline-assembly operand budget")
    output_operand_start = input_count
    keepalive_operand = output_operand_start + output_count
    permutation_operand = keepalive_operand + 1
    marker = site_marker_bits(site_index)
    lines = ["{"]
    lines.append(".reg .pred odezza_lm_toggle_predicate;")
    lines.append(".reg .b32 odezza_lm_permutation_u32;")
    lines.append(".reg .b32 odezza_lm_toggle_salted;")
    lines.append(".reg .b32 odezza_lm_toggle_masked;")
    lines.append(".reg .u32 odezza_lm_keepalive_address;")
    lines.append("brkpt;")
    for index in range(input_count):
        lines.append(
            f"add.rn.ftz.f32 %{index}, %{index}, 0f{marker + index:08x};"
        )
    lines.append("brkpt;")
    lines.append(f"mov.b32 odezza_lm_permutation_u32, %{permutation_operand};")
    lines.append(
        f"add.u32 odezza_lm_toggle_salted, odezza_lm_permutation_u32, {site_index + 1};"
    )
    lines.append("and.b32 odezza_lm_toggle_masked, odezza_lm_toggle_salted, 1;")
    lines.append("setp.ne.u32 odezza_lm_toggle_predicate, odezza_lm_toggle_masked, 0;")
    lines.append(
        f"selp.f32 %{output_operand_start}, %0, %1, odezza_lm_toggle_predicate;"
    )
    lines.append(f"mov.b32 %{keepalive_operand}, %{output_operand_start};")
    for index in range(output_count):
        source_operand = output_operand_start if index == 0 else 0
        lines.append(
            f"add.rn.ftz.f32 %{output_operand_start + index}, %{source_operand}, "
            f"0f{marker + LM_OUTPUT_MARKER_OFFSET + index:08x};"
        )
    for index in range(output_count):
        lines.append(
            f"add.rn.ftz.f32 %{keepalive_operand}, %{keepalive_operand}, %{output_operand_start + index};"
        )
    for index in range(input_count):
        lines.append(
            f"add.rn.ftz.f32 %{keepalive_operand}, %{keepalive_operand}, %{index};"
        )
    lines.append(
        f"add.rn.ftz.f32 %{keepalive_operand}, %{keepalive_operand}, %{permutation_operand};"
    )
    lines.append("mov.u32 odezza_lm_keepalive_address, 0;")
    lines.append(
        f"st.volatile.shared.f32 [odezza_lm_keepalive_address], %{keepalive_operand};"
    )
    lines.extend("brkpt;" for _ in range(shape.site_patch_capacity))
    lines.append("}")
    assembly = "\n".join(_ptx_line(line) for line in lines)
    outputs = ", ".join(
        [
            *(f'"+f"({name})' for name in input_names),
            *(f'"=&f"({name})' for name in result_names),
            f'"=&f"(lm_keepalive_{site_index})',
        ]
    )
    inputs = ['"f"(lm_permutation_input)']
    declarations = "\n".join(f"                        float {name} = 0.0f;" for name in result_names)
    statement = f'''                        asm volatile(
{assembly}
                            : {outputs}
                            : {", ".join(inputs)}
                            : "memory");'''
    return f"""{declarations}
                        float lm_keepalive_{site_index} __attribute__((unused));
{statement}"""


def _site_inputs(shape: LMKernelShape) -> str:
    lines = [
        f"                        const float lm_site_input_{index} = stage_state[{index}];"
        for index in range(shape.state_count)
    ]
    lines.extend(
        f"                        const float lm_site_input_{shape.state_count + index} = evaluation_parameters[{index}];"
        for index in range(shape.optimized_constant_count)
    )
    return "\n".join(lines)


def _primal_sites(shape: LMKernelShape) -> str:
    blocks: list[str] = []
    for site_index, group in enumerate(
        shape.site_output_groups[: shape.primal_site_count]
    ):
        results = tuple(f"lm_primal_{rhs}" for rhs in group)
        blocks.append(_output_site(shape, site_index, results))
        blocks.extend(
            f"                        rhs[{rhs}] = {result};"
            for rhs, result in zip(group, results)
        )
    return "\n".join(blocks)


def _partial_sites(shape: LMKernelShape) -> str:
    blocks: list[str] = []
    for site_index, group in enumerate(
        shape.site_output_groups[shape.primal_site_count :],
        shape.primal_site_count,
    ):
        partial_indices = tuple(index - shape.state_count for index in group)
        results = tuple(
            f"lm_partial_{index // shape.state_count}_{index % shape.state_count}"
            for index in partial_indices
        )
        blocks.append(_output_site(shape, site_index, results))
        blocks.extend(
            f"                            local_partials[{partial_index}] = {result};"
            for partial_index, result in zip(partial_indices, results)
        )
    return "\n".join(blocks)


def _grouped_rk_stage(shape: LMKernelShape) -> str:
    return f"""{_site_inputs(shape)}
                            float rhs[ODEZZA_LM_STATE_COUNT] = {{}};
{_primal_sites(shape)}
                            float local_partials[ODEZZA_LM_STATE_COUNT * ODEZZA_LM_INPUT_COUNT];
                            if (statistics_phase) {{
{_partial_sites(shape)}
                            }}

                            const float weight = (rk_stage == 0 || rk_stage == 3) ? 1.0f : 2.0f;
#pragma unroll
                            for (int component = 0; component < ODEZZA_LM_STATE_COUNT; ++component) {{
                                sum_state[component] = fmaf(weight, rhs[component], sum_state[component]);
                            }}

                            if (statistics_phase) {{
#pragma unroll
                                for (int parameter = 0; parameter < ODEZZA_LM_PARAMETER_COUNT; ++parameter) {{
                                    float stage_derivative[ODEZZA_LM_STATE_COUNT];
#pragma unroll
                                    for (int output = 0; output < ODEZZA_LM_STATE_COUNT; ++output) {{
                                        float sensitivity_derivative =
                                            local_partials[(ODEZZA_LM_STATE_COUNT + parameter) * ODEZZA_LM_STATE_COUNT + output];
#pragma unroll
                                        for (int input_state = 0; input_state < ODEZZA_LM_STATE_COUNT; ++input_state) {{
                                            sensitivity_derivative = fmaf(
                                                local_partials[input_state * ODEZZA_LM_STATE_COUNT + output],
                                                stage_sensitivity[input_state * ODEZZA_LM_PARAMETER_COUNT + parameter],
                                                sensitivity_derivative
                                            );
                                        }}
                                        stage_derivative[output] = sensitivity_derivative;
                                    }}
#pragma unroll
                                    for (int output = 0; output < ODEZZA_LM_STATE_COUNT; ++output) {{
                                        const int sensitivity_index = output * ODEZZA_LM_PARAMETER_COUNT + parameter;
                                        sum_sensitivity[sensitivity_index] = fmaf(
                                            weight,
                                            stage_derivative[output],
                                            sum_sensitivity[sensitivity_index]
                                        );
                                        if (rk_stage != 3) {{
                                            const float stage_h = rk_stage == 2 ? h : half_h;
                                            stage_sensitivity[sensitivity_index] = fmaf(
                                                stage_h,
                                                stage_derivative[output],
                                                sensitivities[sensitivity_index]
                                            );
                                        }}
                                    }}
                                }}
                            }}"""


def _two_site_rk_stage(shape: LMKernelShape) -> str:
    inputs = tuple(
        [
            *(f"stage_state[{index}]" for index in range(shape.state_count)),
            *(
                f"evaluation_parameters[{index}]"
                for index in range(shape.optimized_constant_count)
            ),
        ]
    )
    primal_blocks: list[str] = []
    for site_index, group in enumerate(shape.site_output_groups[: shape.primal_site_count]):
        primal_results = tuple(f"lm_primal_{index}" for index in group)
        primal_blocks.append(
            _readwrite_output_site(
                shape,
                site_index,
                inputs,
                primal_results,
            )
        )
        for index, name in zip(group, primal_results):
            primal_blocks.append(f"                            rhs[{index}] = {name};")
    primal_code = "\n".join(primal_blocks)
    derivative_blocks: list[str] = []
    partial_groups = shape.site_output_groups[shape.primal_site_count :]
    group_offset = shape.primal_site_count
    group_index = 0
    for output in range(shape.state_count):
        stage_derivative = f"stage_derivative_{output}"
        derivative_blocks.append(
            f"                                float {stage_derivative}[ODEZZA_LM_LOCAL_PARAMETER_COUNT] = {{}};"
        )
        while group_index < len(partial_groups):
            group = partial_groups[group_index]
            group_outputs = tuple(
                (scalar_index - shape.state_count) % shape.state_count
                for scalar_index in group
            )
            if any(group_output != output for group_output in group_outputs):
                break
            site_index = group_offset + group_index
            names = tuple(
                f"lm_partial_{site_index}_{index}"
                for index in range(len(group))
            )
            derivative_blocks.append(
                _readwrite_output_site(
                    shape,
                    site_index,
                    inputs,
                    names,
                )
            )
            derivative_blocks.append("#pragma unroll")
            derivative_blocks.append(
                "                                for (int local_parameter = 0; local_parameter < ODEZZA_LM_LOCAL_PARAMETER_COUNT; ++local_parameter) {"
            )
            derivative_blocks.append(
                "                                    const int parameter = (int)lm_fit_lane + local_parameter * ODEZZA_LM_FIT_THREAD_COUNT;"
            )
            derivative_blocks.append(
                "                                    if (parameter >= ODEZZA_LM_PARAMETER_COUNT) continue;"
            )
            for scalar_index, name in zip(group, names):
                input_index = (scalar_index - shape.state_count) // shape.state_count
                value = name
                if input_index < shape.state_count:
                    derivative_blocks.append(
                        f"                                    {stage_derivative}[local_parameter] = fmaf("
                    )
                    derivative_blocks.append(
                        f"                                        {value},"
                    )
                    derivative_blocks.append(
                        f"                                        stage_sensitivity[{input_index} * ODEZZA_LM_LOCAL_PARAMETER_COUNT + local_parameter],"
                    )
                    derivative_blocks.append(
                        f"                                        {stage_derivative}[local_parameter]"
                    )
                    derivative_blocks.append("                                    );")
                else:
                    optimized_index = input_index - shape.state_count
                    derivative_blocks.append(
                        f"                                    if (parameter == {optimized_index}) {stage_derivative}[local_parameter] += {value};"
                    )
            derivative_blocks.append("                                }")
            group_index += 1
        derivative_blocks.append("#pragma unroll")
        derivative_blocks.append(
            "                                for (int local_parameter = 0; local_parameter < ODEZZA_LM_LOCAL_PARAMETER_COUNT; ++local_parameter) {"
        )
        derivative_blocks.append(
            "                                    const int parameter = (int)lm_fit_lane + local_parameter * ODEZZA_LM_FIT_THREAD_COUNT;"
        )
        derivative_blocks.append(
            "                                    if (parameter >= ODEZZA_LM_PARAMETER_COUNT) continue;"
        )
        derivative_blocks.append(
            f"                                    const int sensitivity_index = {output} * ODEZZA_LM_LOCAL_PARAMETER_COUNT + local_parameter;"
        )
        derivative_blocks.append(
            "                                    sum_sensitivity[sensitivity_index] = fmaf("
        )
        derivative_blocks.append("                                        weight,")
        derivative_blocks.append(
            f"                                        {stage_derivative}[local_parameter],"
        )
        derivative_blocks.append(
            "                                        sum_sensitivity[sensitivity_index]"
        )
        derivative_blocks.append("                                    );")
        derivative_blocks.append("                                    if (rk_stage != 3) {")
        derivative_blocks.append(
            "                                        const float stage_h = rk_stage == 2 ? h : half_h;"
        )
        derivative_blocks.append(
            "                                        stage_sensitivity[sensitivity_index] = fmaf("
        )
        derivative_blocks.append("                                            stage_h,")
        derivative_blocks.append(
            f"                                            {stage_derivative}[local_parameter],"
        )
        derivative_blocks.append(
            "                                            sensitivities[sensitivity_index]"
        )
        derivative_blocks.append("                                        );")
        derivative_blocks.append("                                    }")
        derivative_blocks.append("                                }")
    if group_index != len(partial_groups):
        raise ValueError("two-site derivative groups are not ordered by RHS output")
    derivative_code = "\n".join(derivative_blocks)
    return f"""                            float rhs[ODEZZA_LM_STATE_COUNT];
{primal_code}

                            const float weight = (rk_stage == 0 || rk_stage == 3) ? 1.0f : 2.0f;
#pragma unroll
                            for (int component = 0; component < ODEZZA_LM_STATE_COUNT; ++component) {{
                                sum_state[component] = fmaf(weight, rhs[component], sum_state[component]);
                            }}

                            if (statistics_phase) {{
{derivative_code}
                            }}"""


def _thread_owned_solve() -> str:
    return """                float factor[ODEZZA_LM_PARAMETER_COUNT * ODEZZA_LM_PARAMETER_COUNT];
                float solve_rhs[ODEZZA_LM_PARAMETER_COUNT];
                float delta[ODEZZA_LM_PARAMETER_COUNT];
#pragma unroll
                for (int row = 0; row < ODEZZA_LM_PARAMETER_COUNT; ++row) {
#pragma unroll
                    for (int column = 0; column < ODEZZA_LM_PARAMETER_COUNT; ++column) {
                        const int low = row < column ? row : column;
                        const int high = row < column ? column : row;
                        const int gram_index = low * ODEZZA_LM_PARAMETER_COUNT
                            - low * (low - 1) / 2 + high - low;
                        factor[row * ODEZZA_LM_PARAMETER_COUNT + column] = gram[gram_index];
                    }
                    solve_rhs[row] = -gradient[row];
                }

                int factor_ok = 1;
#pragma unroll
                for (int row = 0; row < ODEZZA_LM_PARAMETER_COUNT; ++row) {
#pragma unroll
                    for (int column = 0; column <= row; ++column) {
                        float sum = factor[row * ODEZZA_LM_PARAMETER_COUNT + column];
                        if (row == column) sum += lambda * fmaxf(fabsf(sum), 1.0e-8f);
#pragma unroll
                        for (int k = 0; k < column; ++k) {
                            sum = fmaf(
                                -factor[row * ODEZZA_LM_PARAMETER_COUNT + k],
                                factor[column * ODEZZA_LM_PARAMETER_COUNT + k],
                                sum
                            );
                        }
                        if (row == column) {
                            const int pivot_ok = factor_ok && isfinite(sum) && sum > 1.0e-12f;
                            factor[row * ODEZZA_LM_PARAMETER_COUNT + column] = pivot_ok ? sqrtf(sum) : 1.0f;
                            factor_ok = pivot_ok;
                        } else {
                            const float value = factor_ok
                                ? sum / factor[column * ODEZZA_LM_PARAMETER_COUNT + column]
                                : 0.0f;
                            factor[row * ODEZZA_LM_PARAMETER_COUNT + column] = value;
                            factor_ok = factor_ok && isfinite(value);
                        }
                    }
                }

#pragma unroll
                for (int row = 0; row < ODEZZA_LM_PARAMETER_COUNT; ++row) {
                    float value = solve_rhs[row];
#pragma unroll
                    for (int k = 0; k < row; ++k) {
                        value = fmaf(-factor[row * ODEZZA_LM_PARAMETER_COUNT + k], solve_rhs[k], value);
                    }
                    solve_rhs[row] = factor_ok
                        ? value / factor[row * ODEZZA_LM_PARAMETER_COUNT + row]
                        : 0.0f;
                    factor_ok = factor_ok && isfinite(solve_rhs[row]);
                }

#pragma unroll
                for (int reverse = 0; reverse < ODEZZA_LM_PARAMETER_COUNT; ++reverse) {
                    const int row = ODEZZA_LM_PARAMETER_COUNT - 1 - reverse;
                    float value = solve_rhs[row];
#pragma unroll
                    for (int k = row + 1; k < ODEZZA_LM_PARAMETER_COUNT; ++k) {
                        value = fmaf(-factor[k * ODEZZA_LM_PARAMETER_COUNT + row], delta[k], value);
                    }
                    delta[row] = factor_ok
                        ? value / factor[row * ODEZZA_LM_PARAMETER_COUNT + row]
                        : 0.0f;
                    factor_ok = factor_ok && isfinite(delta[row]);
                }

#pragma unroll
                for (int parameter = 0; parameter < ODEZZA_LM_PARAMETER_COUNT; ++parameter) {
                    proposal[parameter] = parameters[parameter] + delta[parameter];
                    factor_ok = factor_ok && isfinite(proposal[parameter]);
                }
                proposal_ready = factor_ok;"""


def _replicated_cooperative_solve(shape: LMKernelShape) -> str:
    parameter_count = shape.optimized_constant_count
    thread_count = shape.fit_thread_count
    lines = [
        "                float factor[ODEZZA_LM_PARAMETER_COUNT * ODEZZA_LM_PARAMETER_COUNT] = {};",
        "                float solve_rhs[ODEZZA_LM_PARAMETER_COUNT];",
        "                float delta[ODEZZA_LM_PARAMETER_COUNT];",
    ]
    for row in range(parameter_count):
        gradient_owner = row & (thread_count - 1)
        gradient_local = row // thread_count
        lines.extend(
            (
                f"                solve_rhs[{row}] = -__shfl_sync(",
                "                    lm_fit_mask,",
                f"                    local_optimizer[{gradient_local}],",
                f"                    {gradient_owner},",
                "                    ODEZZA_LM_FIT_THREAD_COUNT",
                "                );",
            )
        )
        for column in range(row + 1):
            gram_index = column * parameter_count - column * (column - 1) // 2 + row - column
            optimizer_index = parameter_count + gram_index
            gram_owner = optimizer_index & (thread_count - 1)
            gram_local = optimizer_index // thread_count
            lines.extend(
                (
                    f"                factor[{row * parameter_count + column}] = __shfl_sync(",
                    "                    lm_fit_mask,",
                    f"                    local_optimizer[{gram_local}],",
                    f"                    {gram_owner},",
                    "                    ODEZZA_LM_FIT_THREAD_COUNT",
                    "                );",
                )
            )
    lines.append(
        """
                int factor_ok = 1;
#pragma unroll
                for (int row = 0; row < ODEZZA_LM_PARAMETER_COUNT; ++row) {
#pragma unroll
                    for (int column = 0; column <= row; ++column) {
                        float sum = factor[row * ODEZZA_LM_PARAMETER_COUNT + column];
                        if (row == column) sum += lambda * fmaxf(fabsf(sum), 1.0e-8f);
#pragma unroll
                        for (int k = 0; k < column; ++k) {
                            sum = fmaf(
                                -factor[row * ODEZZA_LM_PARAMETER_COUNT + k],
                                factor[column * ODEZZA_LM_PARAMETER_COUNT + k],
                                sum
                            );
                        }
                        if (row == column) {
                            const int pivot_ok = factor_ok && isfinite(sum) && sum > 1.0e-12f;
                            factor[row * ODEZZA_LM_PARAMETER_COUNT + column] = pivot_ok ? sqrtf(sum) : 1.0f;
                            factor_ok = pivot_ok;
                        } else {
                            const float value = factor_ok
                                ? sum / factor[column * ODEZZA_LM_PARAMETER_COUNT + column]
                                : 0.0f;
                            factor[row * ODEZZA_LM_PARAMETER_COUNT + column] = value;
                            factor_ok = factor_ok && isfinite(value);
                        }
                    }
                }

#pragma unroll
                for (int row = 0; row < ODEZZA_LM_PARAMETER_COUNT; ++row) {
                    float value = solve_rhs[row];
#pragma unroll
                    for (int k = 0; k < row; ++k) {
                        value = fmaf(-factor[row * ODEZZA_LM_PARAMETER_COUNT + k], solve_rhs[k], value);
                    }
                    solve_rhs[row] = factor_ok
                        ? value / factor[row * ODEZZA_LM_PARAMETER_COUNT + row]
                        : 0.0f;
                    factor_ok = factor_ok && isfinite(solve_rhs[row]);
                }

#pragma unroll
                for (int reverse = 0; reverse < ODEZZA_LM_PARAMETER_COUNT; ++reverse) {
                    const int row = ODEZZA_LM_PARAMETER_COUNT - 1 - reverse;
                    float value = solve_rhs[row];
#pragma unroll
                    for (int k = row + 1; k < ODEZZA_LM_PARAMETER_COUNT; ++k) {
                        value = fmaf(-factor[k * ODEZZA_LM_PARAMETER_COUNT + row], delta[k], value);
                    }
                    delta[row] = factor_ok
                        ? value / factor[row * ODEZZA_LM_PARAMETER_COUNT + row]
                        : 0.0f;
                    factor_ok = factor_ok && isfinite(delta[row]);
                }

#pragma unroll
                for (int parameter = 0; parameter < ODEZZA_LM_PARAMETER_COUNT; ++parameter) {
                    proposal[parameter] = parameters[parameter] + delta[parameter];
                    factor_ok = factor_ok && isfinite(proposal[parameter]);
                }
                proposal_ready = factor_ok;"""
    )
    return "\n".join(lines)


def generate_lm_cuda(shape: LMKernelShape = LMKernelShape()) -> str:
    """Generate one-system resident trajectory LM with scalar SASS patch sites."""

    state_count = shape.state_count
    parameter_count = shape.optimized_constant_count
    gram_count = shape.gram_count
    two_site = shape.site_layout == LM_TWO_SITE_LAYOUT
    cooperative = shape.fit_thread_count > 1
    rk_stage = _two_site_rk_stage(shape) if two_site else _grouped_rk_stage(shape)
    thread_constraint = " || blockDim.x != 32u" if two_site or cooperative else ""
    if two_site:
        persistent_sensitivity = "            float sensitivities[ODEZZA_LM_STATE_COUNT * ODEZZA_LM_LOCAL_PARAMETER_COUNT];"
    else:
        persistent_sensitivity = "            float sensitivities[ODEZZA_LM_STATE_COUNT * ODEZZA_LM_PARAMETER_COUNT];"
    sensitivity_parameter_count = (
        "ODEZZA_LM_LOCAL_PARAMETER_COUNT"
        if cooperative
        else "ODEZZA_LM_PARAMETER_COUNT"
    )
    step_sensitivity = f"""                        float stage_sensitivity[ODEZZA_LM_STATE_COUNT * {sensitivity_parameter_count}];
                        float sum_sensitivity[ODEZZA_LM_STATE_COUNT * {sensitivity_parameter_count}];"""
    initialize_step_sensitivity = """                            stage_sensitivity[index] = sensitivities[index];
                            sum_sensitivity[index] = 0.0f;"""
    specialization = (
        "operand-bounded primal and derivative sites replicated across fit lanes"
        if two_site
        else "operand-bounded multi-output sites with cross-output CSE"
    )
    statistics_barrier = "                __syncwarp(lm_fit_mask);" if cooperative else ""
    if cooperative:
        optimizer_arrays = """        float local_optimizer[ODEZZA_LM_LOCAL_OPTIMIZER_COUNT];
        float proposal[ODEZZA_LM_PARAMETER_COUNT];
#pragma unroll
        for (int index = 0; index < ODEZZA_LM_LOCAL_OPTIMIZER_COUNT; ++index) local_optimizer[index] = 0.0f;"""
        statistics_accumulation = """                            float jacobian[ODEZZA_LM_PARAMETER_COUNT];
#pragma unroll
                            for (int parameter = 0; parameter < ODEZZA_LM_PARAMETER_COUNT; ++parameter) {
                                const int owner_lane = parameter & (ODEZZA_LM_FIT_THREAD_COUNT - 1);
                                const int local_parameter = parameter / ODEZZA_LM_FIT_THREAD_COUNT;
                                const float owned_jacobian = lm_fit_lane == (unsigned int)owner_lane
                                    ? sensitivities[component * ODEZZA_LM_LOCAL_PARAMETER_COUNT + local_parameter] * residual_weight
                                    : 0.0f;
                                jacobian[parameter] = __shfl_sync(
                                    lm_fit_mask,
                                    owned_jacobian,
                                    owner_lane,
                                    ODEZZA_LM_FIT_THREAD_COUNT
                                );
                            }
#pragma unroll
                            for (int row = 0; row < ODEZZA_LM_PARAMETER_COUNT; ++row) {
                                if (lm_fit_lane == (unsigned int)(row & (ODEZZA_LM_FIT_THREAD_COUNT - 1))) {
                                    const int local_gradient = row / ODEZZA_LM_FIT_THREAD_COUNT;
                                    local_optimizer[local_gradient] = fmaf(
                                        jacobian[row],
                                        residual,
                                        local_optimizer[local_gradient]
                                    );
                                }
#pragma unroll
                                for (int column = row; column < ODEZZA_LM_PARAMETER_COUNT; ++column) {
                                    const int gram_index = row * ODEZZA_LM_PARAMETER_COUNT
                                        - row * (row - 1) / 2 + column - row;
                                    const int optimizer_index = ODEZZA_LM_PARAMETER_COUNT + gram_index;
                                    if (lm_fit_lane == (unsigned int)(optimizer_index & (ODEZZA_LM_FIT_THREAD_COUNT - 1))) {
                                        const int local_gram = optimizer_index / ODEZZA_LM_FIT_THREAD_COUNT;
                                        local_optimizer[local_gram] = fmaf(
                                            jacobian[row],
                                            jacobian[column],
                                            local_optimizer[local_gram]
                                        );
                                    }
                                }
                            }"""
    else:
        optimizer_arrays = """        float gradient[ODEZZA_LM_PARAMETER_COUNT];
        float gram[ODEZZA_LM_GRAM_COUNT];
        float proposal[ODEZZA_LM_PARAMETER_COUNT];
#pragma unroll
        for (int parameter = 0; parameter < ODEZZA_LM_PARAMETER_COUNT; ++parameter) gradient[parameter] = 0.0f;
#pragma unroll
        for (int index = 0; index < ODEZZA_LM_GRAM_COUNT; ++index) gram[index] = 0.0f;"""
        statistics_accumulation = """#pragma unroll
                            for (int row = 0; row < ODEZZA_LM_PARAMETER_COUNT; ++row) {
                                const float jacobian_row =
                                    sensitivities[component * ODEZZA_LM_PARAMETER_COUNT + row] * residual_weight;
                                gradient[row] = fmaf(jacobian_row, residual, gradient[row]);
#pragma unroll
                                for (int column = row; column < ODEZZA_LM_PARAMETER_COUNT; ++column) {
                                    const int gram_index = row * ODEZZA_LM_PARAMETER_COUNT
                                        - row * (row - 1) / 2 + column - row;
                                    const float jacobian_column =
                                        sensitivities[component * ODEZZA_LM_PARAMETER_COUNT + column] * residual_weight;
                                    gram[gram_index] = fmaf(jacobian_row, jacobian_column, gram[gram_index]);
                                }
                            }"""
    solve = _replicated_cooperative_solve(shape) if cooperative else _thread_owned_solve()
    body = f'''/* Generated trajectory-LM CUDA template. Do not edit by hand.
 * ownership: {shape.fit_thread_count} thread(s) own one toggle permutation and one parameter start
 * integration: resident RK4 over runtime ragged, irregular trajectories
 * differentiation: specialized local RHS partials and CUDA sensitivity propagation
 * optimization: damped LM with in-kernel Cholesky retries and proposal evaluation
 * specialization: {specialization}
 */
#define ODEZZA_LM_STATE_COUNT {state_count}
#define ODEZZA_LM_PARAMETER_COUNT {parameter_count}
#define ODEZZA_LM_INPUT_COUNT {shape.input_count}
#define ODEZZA_LM_GRAM_COUNT {gram_count}
#define ODEZZA_LM_SITE_COUNT {shape.site_count}
#define ODEZZA_LM_FIT_THREAD_COUNT {shape.fit_thread_count}
#define ODEZZA_LM_LOCAL_PARAMETER_COUNT {shape.local_parameter_count}
#define ODEZZA_LM_LOCAL_OPTIMIZER_COUNT {shape.local_optimizer_count}
#define ODEZZA_LM_FITS_PER_WARP {shape.fits_per_warp}

extern "C" __global__ void {LM_KERNEL_NAME}(
    const float *__restrict__ starts,
    unsigned long long start_count,
    const unsigned int *__restrict__ trajectory_offsets,
    const float *__restrict__ trajectory_times,
    const float *__restrict__ reference_data,
    const float *__restrict__ reference_weights,
    unsigned int trajectory_count,
    unsigned int trajectory_point_count,
    unsigned int active_toggle_count,
    unsigned int steps_per_interval,
    unsigned int max_lm_iterations,
    unsigned int max_damping_attempts,
    float initial_damping,
    float *__restrict__ constants_out,
    float *__restrict__ initial_mse_out,
    float *__restrict__ mse_out,
    unsigned int *__restrict__ iterations_out,
    unsigned int *__restrict__ accepted_steps_out,
    unsigned int *__restrict__ factorization_attempts_out
) {{
    if (start_count == 0ull || trajectory_count == 0u || trajectory_point_count < trajectory_count ||
        active_toggle_count > 32u || steps_per_interval == 0u || max_lm_iterations == 0u ||
        max_damping_attempts == 0u || !isfinite(initial_damping) || initial_damping <= 0.0f{thread_constraint}) return;

    const unsigned long long permutation_count = 1ull << active_toggle_count;
    if (start_count > 0xffffffffffffffffull / permutation_count) return;
    const unsigned long long fit_count = start_count * permutation_count;
    const unsigned int lm_fit_lane = threadIdx.x & (ODEZZA_LM_FIT_THREAD_COUNT - 1u);
    const unsigned int lm_fit_base = threadIdx.x - lm_fit_lane;
    const unsigned int lm_fit_mask = ODEZZA_LM_FIT_THREAD_COUNT == 32
        ? 0xffffffffu
        : (unsigned int)(((1ull << ODEZZA_LM_FIT_THREAD_COUNT) - 1ull) << lm_fit_base);
    const unsigned long long global_thread = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long fit = global_thread / ODEZZA_LM_FIT_THREAD_COUNT;

    const unsigned int reference_float_count = ODEZZA_LM_STATE_COUNT * trajectory_point_count;
    extern __shared__ unsigned int shared_words[];
    float *reference = reinterpret_cast<float *>(shared_words);
    float *weights = reference + reference_float_count;
    float *step_sizes = weights + reference_float_count;
    unsigned int *offsets = reinterpret_cast<unsigned int *>(step_sizes + trajectory_point_count);
    for (unsigned int index = threadIdx.x; index < reference_float_count; index += blockDim.x) {{
        reference[index] = reference_data[index];
        weights[index] = reference_weights[index];
    }}
    for (unsigned int index = threadIdx.x; index < trajectory_point_count; index += blockDim.x) {{
        step_sizes[index] = index == 0u
            ? 0.0f
            : (trajectory_times[index] - trajectory_times[index - 1u]) / (float)steps_per_interval;
    }}
    for (unsigned int index = threadIdx.x; index <= trajectory_count; index += blockDim.x) {{
        offsets[index] = trajectory_offsets[index];
    }}
    __syncthreads();
    if (fit >= fit_count) return;

    const unsigned int permutation = (unsigned int)fit;
    const float lm_permutation_input = __uint_as_float(permutation);
    const unsigned long long start_index = fit >> active_toggle_count;
    if (lm_fit_lane == 0u) initial_mse_out[fit] = 3.402823466e+38f;

    float parameters[ODEZZA_LM_PARAMETER_COUNT];
#pragma unroll
    for (int parameter = 0; parameter < ODEZZA_LM_PARAMETER_COUNT; ++parameter) {{
        parameters[parameter] = starts[start_index * ODEZZA_LM_PARAMETER_COUNT + parameter];
    }}
    float lambda = fminf(fmaxf(initial_damping, 1.0e-8f), 1.0e8f);
    float current_sse = 3.402823466e+38f;
    unsigned int current_residual_count = 0u;
    unsigned int completed_iterations = 0u;
    unsigned int accepted_steps = 0u;
    unsigned int factorization_attempts = 0u;
    int optimizer_valid = 1;

    for (unsigned int lm_iteration = 0u;
         lm_iteration < max_lm_iterations && optimizer_valid;
         ++lm_iteration) {{
{optimizer_arrays}
        int accepted = 0;

        for (unsigned int phase = 0u; phase <= max_damping_attempts && !accepted; ++phase) {{
            const int statistics_phase = phase == 0u;
            int proposal_ready = statistics_phase;
            if (!statistics_phase) {{
                ++factorization_attempts;
{solve}
                if (!proposal_ready) {{
                    lambda = fminf(lambda * 10.0f, 1.0e8f);
                    continue;
                }}
            }}

            float evaluation_parameters[ODEZZA_LM_PARAMETER_COUNT];
#pragma unroll
            for (int parameter = 0; parameter < ODEZZA_LM_PARAMETER_COUNT; ++parameter) {{
                evaluation_parameters[parameter] = statistics_phase
                    ? parameters[parameter]
                    : proposal[parameter];
            }}

            float evaluated_sse = 0.0f;
            unsigned int evaluated_residual_count = 0u;
            unsigned int expected_start = 0u;
            int evaluated_valid = proposal_ready;
{persistent_sensitivity}
#pragma unroll
            for (int index = 0; index < ODEZZA_LM_STATE_COUNT * ODEZZA_LM_LOCAL_PARAMETER_COUNT; ++index) {{
                sensitivities[index] = 0.0f;
            }}

            for (unsigned int trajectory = 0u; trajectory < trajectory_count && evaluated_valid; ++trajectory) {{
                const unsigned int point_begin = offsets[trajectory];
                const unsigned int point_end = offsets[trajectory + 1u];
                if (point_begin != expected_start || point_end <= point_begin || point_end > trajectory_point_count) {{
                    evaluated_valid = 0;
                    break;
                }}
                expected_start = point_end;
                float state[ODEZZA_LM_STATE_COUNT];
#pragma unroll
                for (int component = 0; component < ODEZZA_LM_STATE_COUNT; ++component) {{
                    state[component] = reference[component * trajectory_point_count + point_begin];
                }}
#pragma unroll
                for (int index = 0; index < ODEZZA_LM_STATE_COUNT * ODEZZA_LM_LOCAL_PARAMETER_COUNT; ++index) {{
                    sensitivities[index] = 0.0f;
                }}

                for (unsigned int point = point_begin + 1u; point < point_end && evaluated_valid; ++point) {{
                    const float h = step_sizes[point];
                    evaluated_valid = evaluated_valid && isfinite(h) && h > 0.0f;
                    const float half_h = 0.5f * h;
                    const float sixth_h = h / 6.0f;

                    for (unsigned int step = 0u; step < steps_per_interval; ++step) {{
                        float base_state[ODEZZA_LM_STATE_COUNT];
                        float stage_state[ODEZZA_LM_STATE_COUNT];
                        float sum_state[ODEZZA_LM_STATE_COUNT];
{step_sensitivity}
#pragma unroll
                        for (int component = 0; component < ODEZZA_LM_STATE_COUNT; ++component) {{
                            base_state[component] = state[component];
                            stage_state[component] = state[component];
                            sum_state[component] = 0.0f;
                        }}
#pragma unroll
                        for (int index = 0; index < ODEZZA_LM_STATE_COUNT * ODEZZA_LM_LOCAL_PARAMETER_COUNT; ++index) {{
{initialize_step_sensitivity}
                        }}

                        for (int rk_stage = 0; rk_stage < 4; ++rk_stage) {{
{rk_stage}

                            if (rk_stage != 3) {{
                                const float stage_h = rk_stage == 2 ? h : half_h;
#pragma unroll
                                for (int component = 0; component < ODEZZA_LM_STATE_COUNT; ++component) {{
                                    stage_state[component] = fmaf(stage_h, rhs[component], base_state[component]);
                                }}
                            }}
                        }}

#pragma unroll
                        for (int component = 0; component < ODEZZA_LM_STATE_COUNT; ++component) {{
                            state[component] = fmaf(sixth_h, sum_state[component], base_state[component]);
                        }}
                        if (statistics_phase) {{
#pragma unroll
                            for (int index = 0; index < ODEZZA_LM_STATE_COUNT * ODEZZA_LM_LOCAL_PARAMETER_COUNT; ++index) {{
                                sensitivities[index] = fmaf(
                                    sixth_h,
                                    sum_sensitivity[index],
                                    sensitivities[index]
                                );
                            }}
                        }}
                    }}

#pragma unroll
                    for (int component = 0; component < ODEZZA_LM_STATE_COUNT; ++component) {{
                        evaluated_valid = evaluated_valid && isfinite(state[component]);
                        const float target = reference[component * trajectory_point_count + point];
                        const float residual_weight = weights[component * trajectory_point_count + point];
                        evaluated_valid = evaluated_valid && isfinite(residual_weight) && residual_weight >= 0.0f;
                        const float residual = (state[component] - target) * residual_weight;
                        evaluated_sse = fmaf(residual, residual, evaluated_sse);
                        ++evaluated_residual_count;
                        if (statistics_phase) {{
{statistics_accumulation}
                        }}
                    }}
                }}
            }}

            evaluated_valid = evaluated_valid && expected_start == trajectory_point_count
                && evaluated_residual_count != 0u && isfinite(evaluated_sse);
            if (!evaluated_valid) {{
                if (statistics_phase) {{
                    optimizer_valid = 0;
                    break;
                }}
                lambda = fminf(lambda * 10.0f, 1.0e8f);
                continue;
            }}
            if (statistics_phase) {{
                current_sse = evaluated_sse;
                current_residual_count = evaluated_residual_count;
                if (lm_iteration == 0u) {{
                    if (lm_fit_lane == 0u) initial_mse_out[fit] = current_sse / (float)current_residual_count;
                }}
{statistics_barrier}
                continue;
            }}
            if (evaluated_sse < current_sse) {{
                const float improvement = current_sse - evaluated_sse;
#pragma unroll
                for (int parameter = 0; parameter < ODEZZA_LM_PARAMETER_COUNT; ++parameter) {{
                    parameters[parameter] = proposal[parameter];
                }}
                current_sse = evaluated_sse;
                current_residual_count = evaluated_residual_count;
                lambda = fmaxf(lambda * 0.33333334f, 1.0e-8f);
                accepted = 1;
                ++accepted_steps;
                if (improvement <= 1.0e-7f * fmaxf(1.0f, current_sse)) {{
                    lm_iteration = max_lm_iterations - 1u;
                }}
            }} else {{
                lambda = fminf(lambda * 10.0f, 1.0e8f);
            }}
        }}

        ++completed_iterations;
        if (!accepted || current_sse <= 1.0e-12f) break;
    }}

    if (lm_fit_lane == 0u) {{
#pragma unroll
        for (int parameter = 0; parameter < ODEZZA_LM_PARAMETER_COUNT; ++parameter) {{
            constants_out[fit * ODEZZA_LM_PARAMETER_COUNT + parameter] = parameters[parameter];
        }}
        mse_out[fit] = optimizer_valid && current_residual_count != 0u
            ? current_sse / (float)current_residual_count
            : 3.402823466e+38f;
        iterations_out[fit] = completed_iterations;
        accepted_steps_out[fit] = accepted_steps;
        factorization_attempts_out[fit] = factorization_attempts;
    }}
}}
'''
    return finalize_lm_cuda_source(body, shape)
