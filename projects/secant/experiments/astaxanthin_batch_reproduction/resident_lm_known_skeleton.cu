/*
 * SPDX-FileCopyrightText: 2026 Charles Durham
 * SPDX-License-Identifier: MIT
 *
 * MIT License
 *
 * Copyright (c) 2026 Charles Durham
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
/*
 * Resident, thread-owned LM proof for the unblinded Riezzo et al. model.
 *
 * One kernel contains one known symbolic skeleton.  blockIdx.y selects a tile
 * of parameter starts and each active thread owns one complete LM trajectory.
 * The CTA cooperatively caches the complete immutable dataset in shared memory
 * once, then remains resident through integration, normal-equation assembly,
 * damping retries, the 6x6 solve, proposal evaluation, and acceptance.
 *
 * By default the two rate laws are a Secant materialize-shaped patch site.  A
 * CUBIN must be specialized before launch.  Defining SECANT_DIRECT_RATES=1
 * compiles the same known expressions directly and is useful as a numerical
 * reference for the resident optimizer.
 *
 * This proof specializes the two primal rate laws.  Their analytic local
 * derivatives below are deliberately coupled to the known, unblinded shape;
 * a general system-identification shape will extend the site ABI to emit the
 * primal values and their state/parameter partial derivatives together.
 */

#ifndef SECANT_DIRECT_RATES
#define SECANT_DIRECT_RATES 0
#endif

#define SECANT_BPT_16 \
    "brkpt;\n\t" "brkpt;\n\t" "brkpt;\n\t" "brkpt;\n\t" \
    "brkpt;\n\t" "brkpt;\n\t" "brkpt;\n\t" "brkpt;\n\t" \
    "brkpt;\n\t" "brkpt;\n\t" "brkpt;\n\t" "brkpt;\n\t" \
    "brkpt;\n\t" "brkpt;\n\t" "brkpt;\n\t" "brkpt;\n\t"

#define SECANT_BPT_128 \
    SECANT_BPT_16 SECANT_BPT_16 SECANT_BPT_16 SECANT_BPT_16 \
    SECANT_BPT_16 SECANT_BPT_16 SECANT_BPT_16 SECANT_BPT_16

extern "C" __global__
void secant_cubin_materialize_000(
    const float *__restrict__ starts,
    unsigned long long starts_leading_dimension,
    unsigned long long num_settings,
    const float *__restrict__ reference_data,
    unsigned int steps_per_observation,
    unsigned int max_lm_iterations,
    unsigned int max_damping_attempts,
    float initial_damping,
    float *__restrict__ constants_out,
    unsigned long long constants_leading_dimension,
    float *__restrict__ mse_out,
    unsigned int *__restrict__ iterations_out,
    unsigned int *__restrict__ accepted_steps_out)
{
    /*
     * reference_data layout:
     *   [0, 12)   initial state, state-major [state][experiment]
     *   [12, 156) targets, state-major [state][experiment][observation]
     */
    __shared__ float reference[156];
    for (unsigned int index = threadIdx.x; index < 156u; index += blockDim.x) {
        reference[index] = reference_data[index];
    }
    __syncthreads();

    const unsigned long long setting =
        (unsigned long long)blockIdx.y * blockDim.x + threadIdx.x;
    if (setting >= num_settings || steps_per_observation == 0u ||
        max_lm_iterations == 0u || max_damping_attempts == 0u) return;

    float parameters[6];
#pragma unroll
    for (int parameter = 0; parameter < 6; ++parameter) {
        parameters[parameter] =
            starts[(unsigned long long)parameter * starts_leading_dimension + setting];
    }
    /* Preserve the 12-input Secant marker ABI even though this proof fits six. */
    const float spare_parameter_0 =
        starts[6ull * starts_leading_dimension + setting];
    const float spare_parameter_1 =
        starts[7ull * starts_leading_dimension + setting];

    float lambda = fminf(fmaxf(initial_damping, 1.0e-8f), 1.0e8f);
    float current_sse = 3.402823466e+38F;
    unsigned int completed_iterations = 0u;
    unsigned int accepted_steps = 0u;
    int optimizer_valid = 1;

    const float h = 8.0f / (float)steps_per_observation;
    const float half_h = 0.5f * h;
    const float sixth_h = h / 6.0f;
    /* Normalize the differently scaled X, S1, S2, and P residual channels. */
    const float residual_scale[4] = {1.0f, 0.05f, 0.2f, 1.0f};

    const float mu_d = 0.0055f;
    const float Y_S1 = 2.58f;
    const float Y_S2 = 1.71f;
    const float beta = 0.21f;
    const float k_d = 0.0466f;

    for (unsigned int lm_iteration = 0u;
         lm_iteration < max_lm_iterations && optimizer_valid;
         ++lm_iteration) {
        float gradient[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float gram[21] = {
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
        };
        float proposal[6];
        int accepted = 0;

        /*
         * phase 0 evaluates the incumbent and its analytic Jacobian.  Later
         * phases solve at progressively larger damping and evaluate proposals.
         * Keeping every evaluation in this one lexical loop preserves one
         * Secant patch site in the CUBIN.
         */
        for (unsigned int phase = 0u;
             phase <= max_damping_attempts && !accepted;
             ++phase) {
            const int statistics_phase = phase == 0u;
            int proposal_ready = statistics_phase;

            if (!statistics_phase) {
                float factor[36];
                float rhs[6];
                float delta[6];

#pragma unroll
                for (int row = 0; row < 6; ++row) {
#pragma unroll
                    for (int column = 0; column < 6; ++column) {
                        const int lhs = row < column ? row : column;
                        const int rhs_column = row < column ? column : row;
                        const int gram_index =
                            lhs * 6 - lhs * (lhs - 1) / 2 + rhs_column - lhs;
                        factor[row * 6 + column] = gram[gram_index];
                    }
                    rhs[row] = -gradient[row];
                }

                int factor_ok = 1;
#pragma unroll
                for (int row = 0; row < 6; ++row) {
#pragma unroll
                    for (int column = 0; column <= row; ++column) {
                        float sum = factor[row * 6 + column];
                        if (row == column) {
                            sum += lambda * fmaxf(fabsf(sum), 1.0e-8f);
                        }
#pragma unroll
                        for (int k = 0; k < column; ++k) {
                            sum = fmaf(-factor[row * 6 + k],
                                       factor[column * 6 + k], sum);
                        }
                        if (row == column) {
                            const int pivot_ok = factor_ok &&
                                ((__float_as_uint(sum) & 0x7fffffffu) < 0x7f800000u) &&
                                sum > 1.0e-12f;
                            factor[row * 6 + column] = pivot_ok ? sqrtf(sum) : 1.0f;
                            factor_ok = pivot_ok;
                        } else {
                            const float value = factor_ok
                                ? sum / factor[column * 6 + column] : 0.0f;
                            factor[row * 6 + column] = value;
                            factor_ok = factor_ok &&
                                ((__float_as_uint(value) & 0x7fffffffu) < 0x7f800000u);
                        }
                    }
                }

#pragma unroll
                for (int row = 0; row < 6; ++row) {
                    float value = rhs[row];
#pragma unroll
                    for (int k = 0; k < row; ++k) {
                        value = fmaf(-factor[row * 6 + k], rhs[k], value);
                    }
                    rhs[row] = factor_ok ? value / factor[row * 6 + row] : 0.0f;
                    factor_ok = factor_ok &&
                        ((__float_as_uint(rhs[row]) & 0x7fffffffu) < 0x7f800000u);
                }

#pragma unroll
                for (int reverse = 0; reverse < 6; ++reverse) {
                    const int row = 5 - reverse;
                    float value = rhs[row];
#pragma unroll
                    for (int k = row + 1; k < 6; ++k) {
                        value = fmaf(-factor[k * 6 + row], delta[k], value);
                    }
                    delta[row] = factor_ok ? value / factor[row * 6 + row] : 0.0f;
                    factor_ok = factor_ok &&
                        ((__float_as_uint(delta[row]) & 0x7fffffffu) < 0x7f800000u);
                }

#pragma unroll
                for (int parameter = 0; parameter < 6; ++parameter) {
                    proposal[parameter] = parameters[parameter] + delta[parameter];
                    factor_ok = factor_ok &&
                        ((__float_as_uint(proposal[parameter]) & 0x7fffffffu) < 0x7f800000u);
                }
                proposal_ready = factor_ok;
                if (!proposal_ready) {
                    lambda = fminf(lambda * 10.0f, 1.0e8f);
                    continue;
                }
            }

            float evaluated_sse = 0.0f;
            int evaluated_valid = proposal_ready;
            float sensitivities[24];
#pragma unroll
            for (int index = 0; index < 24; ++index) sensitivities[index] = 0.0f;

#pragma unroll 1
            for (int experiment = 0; experiment < 3 && evaluated_valid; ++experiment) {
                float state[4];
#pragma unroll
                for (int component = 0; component < 4; ++component) {
                    state[component] = reference[component * 3 + experiment];
                }
#pragma unroll
                for (int index = 0; index < 24; ++index) sensitivities[index] = 0.0f;

#pragma unroll 1
                for (int observation = 0; observation < 12 && evaluated_valid; ++observation) {
#pragma unroll 1
                    for (unsigned int step = 0; step < steps_per_observation; ++step) {
                        float base_state[4];
                        float stage_state[4];
                        float sum_state[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                        float base_sensitivity[24];
                        float stage_sensitivity[24];
                        float sum_sensitivity[24];

#pragma unroll
                        for (int component = 0; component < 4; ++component) {
                            base_state[component] = state[component];
                            stage_state[component] = state[component];
                        }
#pragma unroll
                        for (int index = 0; index < 24; ++index) {
                            base_sensitivity[index] = sensitivities[index];
                            stage_sensitivity[index] = sensitivities[index];
                            sum_sensitivity[index] = 0.0f;
                        }

#pragma unroll 1
                        for (int stage = 0; stage < 4; ++stage) {
                            const float c0 = statistics_phase ? parameters[0] : proposal[0];
                            const float c1 = statistics_phase ? parameters[1] : proposal[1];
                            const float c2 = statistics_phase ? parameters[2] : proposal[2];
                            const float c3 = statistics_phase ? parameters[3] : proposal[3];
                            const float c4 = statistics_phase ? parameters[4] : proposal[4];
                            const float c5 = statistics_phase ? parameters[5] : proposal[5];
                            float specific_growth_1;
                            float specific_growth_2;

#if SECANT_DIRECT_RATES
                            specific_growth_1 = c0 * stage_state[1] /
                                ((stage_state[1] + c1 * stage_state[0]) *
                                 (1.0f + c2 * stage_state[2]));
                            specific_growth_2 = c3 * stage_state[2] /
                                ((stage_state[2] + c4 * stage_state[0]) *
                                 (1.0f + c5 * stage_state[1]));
#else
                            const float c6 = spare_parameter_0;
                            const float c7 = spare_parameter_1;
                            float marked0 __attribute__((unused));
                            float marked1 __attribute__((unused));
                            float marked2 __attribute__((unused));
                            float marked3 __attribute__((unused));
                            float marked4 __attribute__((unused));
                            float marked5 __attribute__((unused));
                            float marked6 __attribute__((unused));
                            float marked7 __attribute__((unused));
                            float marked8 __attribute__((unused));
                            float marked9 __attribute__((unused));
                            float marked10 __attribute__((unused));
                            float marked11 __attribute__((unused));
                            float keepalive __attribute__((unused));
                            asm volatile(
                                "{\n\t"
                                ".reg .u32 keepalive_address;\n\t"
                                "brkpt;\n\t"
                                "add.rn.ftz.f32 %0, %15, 0f7fc0ffee;\n\t"
                                "add.rn.ftz.f32 %1, %16, 0f7fc0ffef;\n\t"
                                "add.rn.ftz.f32 %2, %17, 0f7fc0fff0;\n\t"
                                "add.rn.ftz.f32 %3, %18, 0f7fc0fff1;\n\t"
                                "add.rn.ftz.f32 %4, %19, 0f7fc0fff2;\n\t"
                                "add.rn.ftz.f32 %5, %20, 0f7fc0fff3;\n\t"
                                "add.rn.ftz.f32 %6, %21, 0f7fc0fff4;\n\t"
                                "add.rn.ftz.f32 %7, %22, 0f7fc0fff5;\n\t"
                                "add.rn.ftz.f32 %8, %23, 0f7fc0fff6;\n\t"
                                "add.rn.ftz.f32 %9, %24, 0f7fc0fff7;\n\t"
                                "add.rn.ftz.f32 %10, %25, 0f7fc0fff8;\n\t"
                                "add.rn.ftz.f32 %11, %26, 0f7fc0fff9;\n\t"
                                "brkpt;\n\t"
                                "add.rn.ftz.f32 %0, %0, %1;\n\t"
                                "add.rn.ftz.f32 %0, %0, %2;\n\t"
                                "add.rn.ftz.f32 %0, %0, %3;\n\t"
                                "add.rn.ftz.f32 %0, %0, %4;\n\t"
                                "add.rn.ftz.f32 %0, %0, %5;\n\t"
                                "add.rn.ftz.f32 %0, %0, %6;\n\t"
                                "add.rn.ftz.f32 %0, %0, %7;\n\t"
                                "add.rn.ftz.f32 %0, %0, %8;\n\t"
                                "add.rn.ftz.f32 %0, %0, %9;\n\t"
                                "add.rn.ftz.f32 %0, %0, %10;\n\t"
                                "add.rn.ftz.f32 %0, %0, %11;\n\t"
                                "add.rn.ftz.f32 %12, %0, 0f7fc0fffa;\n\t"
                                "add.rn.ftz.f32 %13, %0, 0f7fc0fffb;\n\t"
                                "brkpt;\n\t"
                                "add.rn.ftz.f32 %14, %12, %13;\n\t"
                                "add.rn.ftz.f32 %14, %14, %15;\n\t"
                                "add.rn.ftz.f32 %14, %14, %16;\n\t"
                                "add.rn.ftz.f32 %14, %14, %17;\n\t"
                                "add.rn.ftz.f32 %14, %14, %18;\n\t"
                                "add.rn.ftz.f32 %14, %14, %19;\n\t"
                                "add.rn.ftz.f32 %14, %14, %20;\n\t"
                                "add.rn.ftz.f32 %14, %14, %21;\n\t"
                                "add.rn.ftz.f32 %14, %14, %22;\n\t"
                                "add.rn.ftz.f32 %14, %14, %23;\n\t"
                                "add.rn.ftz.f32 %14, %14, %24;\n\t"
                                "add.rn.ftz.f32 %14, %14, %25;\n\t"
                                "add.rn.ftz.f32 %14, %14, %26;\n\t"
                                "mov.u32 keepalive_address, 0;\n\t"
                                "st.volatile.shared.f32 [keepalive_address], %14;\n\t"
                                SECANT_BPT_128
                                "}\n\t"
                                : "=&f"(marked0), "=&f"(marked1), "=&f"(marked2),
                                  "=&f"(marked3), "=&f"(marked4), "=&f"(marked5),
                                  "=&f"(marked6), "=&f"(marked7), "=&f"(marked8),
                                  "=&f"(marked9), "=&f"(marked10), "=&f"(marked11),
                                  "=&f"(specific_growth_1), "=&f"(specific_growth_2),
                                  "=&f"(keepalive)
                                : "f"(stage_state[0]), "f"(stage_state[1]),
                                  "f"(stage_state[2]), "f"(stage_state[3]),
                                  "f"(c0), "f"(c1), "f"(c2), "f"(c3),
                                  "f"(c4), "f"(c5), "f"(c6), "f"(c7)
                                : "memory");
#endif

                            const float growth_1 = specific_growth_1 * stage_state[0];
                            const float growth_2 = specific_growth_2 * stage_state[0];
                            float derivative[4];
                            derivative[0] = growth_1 + growth_2 - mu_d * stage_state[0];
                            derivative[1] = -Y_S1 * growth_1;
                            derivative[2] = -Y_S2 * growth_2;
                            derivative[3] = beta * stage_state[0]
                                - k_d * stage_state[0] * stage_state[0];
                            const float weight = (stage == 0 || stage == 3) ? 1.0f : 2.0f;

#pragma unroll
                            for (int component = 0; component < 4; ++component) {
                                sum_state[component] = fmaf(weight, derivative[component],
                                                           sum_state[component]);
                            }

                            if (statistics_phase) {
                                const float denominator_1a = stage_state[1] + c1 * stage_state[0];
                                const float denominator_1b = 1.0f + c2 * stage_state[2];
                                const float denominator_2a = stage_state[2] + c4 * stage_state[0];
                                const float denominator_2b = 1.0f + c5 * stage_state[1];
                                const float inverse_1a = 1.0f / denominator_1a;
                                const float inverse_1b = 1.0f / denominator_1b;
                                const float inverse_2a = 1.0f / denominator_2a;
                                const float inverse_2b = 1.0f / denominator_2b;

                                const float dmu1_dx = -specific_growth_1 * c1 * inverse_1a;
                                const float dmu1_ds1 = c0 * c1 * stage_state[0] *
                                    inverse_1a * inverse_1a * inverse_1b;
                                const float dmu1_ds2 = -specific_growth_1 * c2 * inverse_1b;
                                const float dmu2_dx = -specific_growth_2 * c4 * inverse_2a;
                                const float dmu2_ds1 = -specific_growth_2 * c5 * inverse_2b;
                                const float dmu2_ds2 = c3 * c4 * stage_state[0] *
                                    inverse_2a * inverse_2a * inverse_2b;

                                const float dg1_dx = specific_growth_1 + stage_state[0] * dmu1_dx;
                                const float dg1_ds1 = stage_state[0] * dmu1_ds1;
                                const float dg1_ds2 = stage_state[0] * dmu1_ds2;
                                const float dg2_dx = specific_growth_2 + stage_state[0] * dmu2_dx;
                                const float dg2_ds1 = stage_state[0] * dmu2_ds1;
                                const float dg2_ds2 = stage_state[0] * dmu2_ds2;

                                const float a00 = dg1_dx + dg2_dx - mu_d;
                                const float a01 = dg1_ds1 + dg2_ds1;
                                const float a02 = dg1_ds2 + dg2_ds2;
                                const float a10 = -Y_S1 * dg1_dx;
                                const float a11 = -Y_S1 * dg1_ds1;
                                const float a12 = -Y_S1 * dg1_ds2;
                                const float a20 = -Y_S2 * dg2_dx;
                                const float a21 = -Y_S2 * dg2_ds1;
                                const float a22 = -Y_S2 * dg2_ds2;
                                const float a30 = beta - 2.0f * k_d * stage_state[0];

#pragma unroll
                                for (int parameter = 0; parameter < 6; ++parameter) {
                                    const float dmu1_dp = parameter == 0
                                        ? stage_state[1] * inverse_1a * inverse_1b
                                        : parameter == 1
                                            ? -specific_growth_1 * stage_state[0] * inverse_1a
                                            : parameter == 2
                                                ? -specific_growth_1 * stage_state[2] * inverse_1b
                                                : 0.0f;
                                    const float dmu2_dp = parameter == 3
                                        ? stage_state[2] * inverse_2a * inverse_2b
                                        : parameter == 4
                                            ? -specific_growth_2 * stage_state[0] * inverse_2a
                                            : parameter == 5
                                                ? -specific_growth_2 * stage_state[1] * inverse_2b
                                                : 0.0f;
                                    const float b_growth_1 = stage_state[0] * dmu1_dp;
                                    const float b_growth_2 = stage_state[0] * dmu2_dp;
                                    const float sx = stage_sensitivity[parameter];
                                    const float ss1 = stage_sensitivity[6 + parameter];
                                    const float ss2 = stage_sensitivity[12 + parameter];
                                    const float sensitivity_derivative_x =
                                        a00 * sx + a01 * ss1 + a02 * ss2 +
                                        b_growth_1 + b_growth_2;
                                    const float sensitivity_derivative_s1 =
                                        a10 * sx + a11 * ss1 + a12 * ss2 -
                                        Y_S1 * b_growth_1;
                                    const float sensitivity_derivative_s2 =
                                        a20 * sx + a21 * ss1 + a22 * ss2 -
                                        Y_S2 * b_growth_2;
                                    const float sensitivity_derivative_p = a30 * sx;

                                    sum_sensitivity[parameter] = fmaf(
                                        weight, sensitivity_derivative_x,
                                        sum_sensitivity[parameter]);
                                    sum_sensitivity[6 + parameter] = fmaf(
                                        weight, sensitivity_derivative_s1,
                                        sum_sensitivity[6 + parameter]);
                                    sum_sensitivity[12 + parameter] = fmaf(
                                        weight, sensitivity_derivative_s2,
                                        sum_sensitivity[12 + parameter]);
                                    sum_sensitivity[18 + parameter] = fmaf(
                                        weight, sensitivity_derivative_p,
                                        sum_sensitivity[18 + parameter]);

                                    if (stage != 3) {
                                        const float stage_h = stage == 2 ? h : half_h;
                                        stage_sensitivity[parameter] = fmaf(
                                            stage_h, sensitivity_derivative_x,
                                            base_sensitivity[parameter]);
                                        stage_sensitivity[6 + parameter] = fmaf(
                                            stage_h, sensitivity_derivative_s1,
                                            base_sensitivity[6 + parameter]);
                                        stage_sensitivity[12 + parameter] = fmaf(
                                            stage_h, sensitivity_derivative_s2,
                                            base_sensitivity[12 + parameter]);
                                        stage_sensitivity[18 + parameter] = fmaf(
                                            stage_h, sensitivity_derivative_p,
                                            base_sensitivity[18 + parameter]);
                                    }
                                }
                            }

                            if (stage != 3) {
                                const float stage_h = stage == 2 ? h : half_h;
#pragma unroll
                                for (int component = 0; component < 4; ++component) {
                                    stage_state[component] = fmaf(
                                        stage_h, derivative[component], base_state[component]);
                                }
                            }
                        }

#pragma unroll
                        for (int component = 0; component < 4; ++component) {
                            state[component] = fmaf(
                                sixth_h, sum_state[component], base_state[component]);
                        }
                        if (statistics_phase) {
#pragma unroll
                            for (int index = 0; index < 24; ++index) {
                                sensitivities[index] = fmaf(
                                    sixth_h, sum_sensitivity[index], base_sensitivity[index]);
                            }
                        }
                    }

#pragma unroll
                    for (int component = 0; component < 4; ++component) {
                        const unsigned int bits = __float_as_uint(state[component]) & 0x7fffffffu;
                        evaluated_valid = evaluated_valid && bits < 0x7f800000u;
                        const int target_index = 12 + component * 36 + experiment * 12 + observation;
                        const float residual =
                            (state[component] - reference[target_index]) * residual_scale[component];
                        evaluated_sse = fmaf(residual, residual, evaluated_sse);

                        if (statistics_phase) {
#pragma unroll
                            for (int row = 0; row < 6; ++row) {
                                const float jacobian_row =
                                    sensitivities[component * 6 + row] * residual_scale[component];
                                gradient[row] = fmaf(jacobian_row, residual, gradient[row]);
#pragma unroll
                                for (int column = row; column < 6; ++column) {
                                    const int gram_index =
                                        row * 6 - row * (row - 1) / 2 + column - row;
                                    const float jacobian_column =
                                        sensitivities[component * 6 + column] *
                                        residual_scale[component];
                                    gram[gram_index] = fmaf(
                                        jacobian_row, jacobian_column, gram[gram_index]);
                                }
                            }
                        }
                    }
                }
            }

            if (!evaluated_valid ||
                ((__float_as_uint(evaluated_sse) & 0x7fffffffu) >= 0x7f800000u)) {
                if (statistics_phase) {
                    optimizer_valid = 0;
                    break;
                }
                lambda = fminf(lambda * 10.0f, 1.0e8f);
                continue;
            }

            if (statistics_phase) {
                current_sse = evaluated_sse;
                continue;
            }

            if (evaluated_sse < current_sse) {
                const float improvement = current_sse - evaluated_sse;
#pragma unroll
                for (int parameter = 0; parameter < 6; ++parameter) {
                    parameters[parameter] = proposal[parameter];
                }
                current_sse = evaluated_sse;
                lambda = fmaxf(lambda * 0.33333334f, 1.0e-8f);
                accepted = 1;
                ++accepted_steps;
                if (improvement <= 1.0e-7f * fmaxf(1.0f, current_sse)) {
                    lm_iteration = max_lm_iterations - 1u;
                }
            } else {
                lambda = fminf(lambda * 10.0f, 1.0e8f);
            }
        }

        ++completed_iterations;
        if (!accepted || current_sse <= 1.0e-12f) break;
    }

#pragma unroll
    for (int parameter = 0; parameter < 6; ++parameter) {
        constants_out[(unsigned long long)parameter * constants_leading_dimension + setting] =
            parameters[parameter];
    }
    mse_out[setting] = optimizer_valid ? current_sse / 144.0f : 3.402823466e+38F;
    iterations_out[setting] = completed_iterations;
    accepted_steps_out[setting] = accepted_steps;
}

#undef SECANT_BPT_128
#undef SECANT_BPT_16
