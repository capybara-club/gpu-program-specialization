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
 * Dense-trajectory, thread-owned LM control for the fed-batch model.
 *
 * Each active thread owns one independent (leaf setting, parameter start) and
 * complete resident fit. The CTA caches all 16 aligned trajectories once, then
 * stays resident through RK4 sensitivity propagation, normal-equation
 * assembly, damping retries, the 6x6 solve, proposal evaluation, and
 * acceptance.
 *
 * The rate-law AST shape is fixed, while each setting dynamically binds its 16
 * leaves to a bank of four state values and eight constants. Constants 0..5 are
 * active LM variables; constants 6..7 are fixed within a setting. The default
 * build is the hand-written control. SSID_SPECIALIZED_GRADIENT replaces its
 * rate laws with a primal patch site and its local partials with a disjoint
 * statistics-only patch site, both derived from inspectable postorder tapes.
 */

extern "C" __global__
void secant_cubin_materialize_000(
    const float *__restrict__ starts,
    unsigned long long starts_leading_dimension,
    const unsigned int *__restrict__ leaf_bindings,
    unsigned long long bindings_leading_dimension,
    unsigned long long num_settings,
    unsigned int starts_per_setting,
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
     *   [0, 64)    initial state, state-major [state][trajectory]
     *   [64, 832)  targets, state-major [state][trajectory][observation]
     */
    __shared__ float reference[832];
    extern __shared__ float bank[];
    for (unsigned int index = threadIdx.x; index < 832u; index += blockDim.x) {
        reference[index] = reference_data[index];
    }
    __syncthreads();

    const unsigned long long fit =
        (unsigned long long)blockIdx.y * blockDim.x + threadIdx.x;
    const unsigned long long num_fits = num_settings * starts_per_setting;
    if (fit >= num_fits || steps_per_observation == 0u ||
        max_lm_iterations == 0u || max_damping_attempts == 0u ||
        starts_per_setting == 0u) return;
    const unsigned long long setting = fit / starts_per_setting;

    unsigned int bindings[16];
    int bindings_valid = 1;
#pragma unroll
    for (int leaf = 0; leaf < 16; ++leaf) {
        const unsigned int binding = leaf_bindings[
            (unsigned long long)leaf * bindings_leading_dimension + setting];
        bindings[leaf] = binding;
        bindings_valid = bindings_valid && binding < 12u;
    }
    if (!bindings_valid) {
#pragma unroll
        for (int parameter = 0; parameter < 6; ++parameter) {
            constants_out[
                (unsigned long long)parameter * constants_leading_dimension + fit] =
                starts[(unsigned long long)parameter * starts_leading_dimension + fit];
        }
        mse_out[fit] = 3.402823466e+38F;
        iterations_out[fit] = 0u;
        accepted_steps_out[fit] = 0u;
        return;
    }

    float parameters[6];
#pragma unroll
    for (int parameter = 0; parameter < 6; ++parameter) {
        parameters[parameter] =
            starts[(unsigned long long)parameter * starts_leading_dimension + fit];
    }
    const float fixed_parameter_6 =
        starts[6ull * starts_leading_dimension + fit];
    const float fixed_parameter_7 =
        starts[7ull * starts_leading_dimension + fit];

    float lambda = fminf(fmaxf(initial_damping, 1.0e-8f), 1.0e8f);
    float current_sse = 3.402823466e+38F;
    unsigned int completed_iterations = 0u;
    unsigned int accepted_steps = 0u;
    int optimizer_valid = 1;

    const float h = 8.0f / (float)steps_per_observation;
    const float half_h = 0.5f * h;
    const float sixth_h = h / 6.0f;
    /* Normalize the differently scaled X, S1, S2, and P residual channels. */

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

#pragma unroll
            for (int parameter = 0; parameter < 6; ++parameter) {
                bank[(4u + (unsigned int)parameter) * blockDim.x + threadIdx.x] =
                    statistics_phase ? parameters[parameter] : proposal[parameter];
            }
            bank[10u * blockDim.x + threadIdx.x] = fixed_parameter_6;
            bank[11u * blockDim.x + threadIdx.x] = fixed_parameter_7;

            float evaluated_sse = 0.0f;
            int evaluated_valid = proposal_ready;
            float sensitivities[24];
#pragma unroll
            for (int index = 0; index < 24; ++index) sensitivities[index] = 0.0f;

#pragma unroll 1
            for (int experiment = 0; experiment < 16 && evaluated_valid; ++experiment) {
                float state[4];
#pragma unroll
                for (int component = 0; component < 4; ++component) {
                    state[component] = reference[component * 16 + experiment];
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
                            bank[0u * blockDim.x + threadIdx.x] = stage_state[0];
                            bank[1u * blockDim.x + threadIdx.x] = stage_state[1];
                            bank[2u * blockDim.x + threadIdx.x] = stage_state[2];
                            bank[3u * blockDim.x + threadIdx.x] = stage_state[3];

                            float leaves[16];
#pragma unroll
                            for (int leaf = 0; leaf < 16; ++leaf) {
                                leaves[leaf] = bank[
                                    bindings[leaf] * blockDim.x + threadIdx.x];
                            }
#ifdef SSID_SPECIALIZED_GRADIENT
#include "fedbatch_primal_site.inc"
#else
                            const float denominator_1a = leaves[2] + leaves[3] * leaves[4];
                            const float denominator_1b = leaves[5] + leaves[6] * leaves[7];
                            const float denominator_2a = leaves[10] + leaves[11] * leaves[12];
                            const float denominator_2b = leaves[13] + leaves[14] * leaves[15];
                            const float inverse_denominator_1 = 1.0f /
                                (denominator_1a * denominator_1b);
                            const float inverse_denominator_2 = 1.0f /
                                (denominator_2a * denominator_2b);
                            const float specific_growth_1 =
                                leaves[0] * leaves[1] * inverse_denominator_1;
                            const float specific_growth_2 =
                                leaves[8] * leaves[9] * inverse_denominator_2;
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
#ifdef SSID_SPECIALIZED_GRADIENT
#include "fedbatch_partial_site.inc"
#else
                                float leaf_partials_1[8] = {
                                    leaves[1] * inverse_denominator_1,
                                    leaves[0] * inverse_denominator_1,
                                    -specific_growth_1 / denominator_1a,
                                    -specific_growth_1 * leaves[4] / denominator_1a,
                                    -specific_growth_1 * leaves[3] / denominator_1a,
                                    -specific_growth_1 / denominator_1b,
                                    -specific_growth_1 * leaves[7] / denominator_1b,
                                    -specific_growth_1 * leaves[6] / denominator_1b
                                };
                                float leaf_partials_2[8] = {
                                    leaves[9] * inverse_denominator_2,
                                    leaves[8] * inverse_denominator_2,
                                    -specific_growth_2 / denominator_2a,
                                    -specific_growth_2 * leaves[12] / denominator_2a,
                                    -specific_growth_2 * leaves[11] / denominator_2a,
                                    -specific_growth_2 / denominator_2b,
                                    -specific_growth_2 * leaves[15] / denominator_2b,
                                    -specific_growth_2 * leaves[14] / denominator_2b
                                };
#endif
                                float dmu1_state[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                                float dmu2_state[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                                float dmu1_parameter[6] = {
                                    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
                                };
                                float dmu2_parameter[6] = {
                                    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
                                };
#pragma unroll
                                for (int leaf = 0; leaf < 8; ++leaf) {
#pragma unroll
                                    for (int component = 0; component < 4; ++component) {
                                        dmu1_state[component] += bindings[leaf] == component
                                            ? leaf_partials_1[leaf] : 0.0f;
                                        dmu2_state[component] += bindings[8 + leaf] == component
                                            ? leaf_partials_2[leaf] : 0.0f;
                                    }
#pragma unroll
                                    for (int parameter = 0; parameter < 6; ++parameter) {
                                        dmu1_parameter[parameter] +=
                                            bindings[leaf] == 4u + (unsigned int)parameter
                                                ? leaf_partials_1[leaf] : 0.0f;
                                        dmu2_parameter[parameter] +=
                                            bindings[8 + leaf] == 4u + (unsigned int)parameter
                                                ? leaf_partials_2[leaf] : 0.0f;
                                    }
                                }

                                float dg1_state[4];
                                float dg2_state[4];
#pragma unroll
                                for (int component = 0; component < 4; ++component) {
                                    dg1_state[component] =
                                        stage_state[0] * dmu1_state[component] +
                                        (component == 0 ? specific_growth_1 : 0.0f);
                                    dg2_state[component] =
                                        stage_state[0] * dmu2_state[component] +
                                        (component == 0 ? specific_growth_2 : 0.0f);
                                }

#pragma unroll
                                for (int parameter = 0; parameter < 6; ++parameter) {
                                    const float b_growth_1 =
                                        stage_state[0] * dmu1_parameter[parameter];
                                    const float b_growth_2 =
                                        stage_state[0] * dmu2_parameter[parameter];
                                    float propagated_growth_1 = 0.0f;
                                    float propagated_growth_2 = 0.0f;
#pragma unroll
                                    for (int component = 0; component < 4; ++component) {
                                        const float sensitivity =
                                            stage_sensitivity[component * 6 + parameter];
                                        propagated_growth_1 = fmaf(
                                            dg1_state[component], sensitivity,
                                            propagated_growth_1);
                                        propagated_growth_2 = fmaf(
                                            dg2_state[component], sensitivity,
                                            propagated_growth_2);
                                    }
                                    const float sx = stage_sensitivity[parameter];
                                    const float total_growth_1 = propagated_growth_1 + b_growth_1;
                                    const float total_growth_2 = propagated_growth_2 + b_growth_2;
                                    const float sensitivity_derivative_x = total_growth_1 +
                                        total_growth_2 - mu_d * sx;
                                    const float sensitivity_derivative_s1 =
                                        -Y_S1 * total_growth_1;
                                    const float sensitivity_derivative_s2 =
                                        -Y_S2 * total_growth_2;
                                    const float sensitivity_derivative_p =
                                        (beta - 2.0f * k_d * stage_state[0]) * sx;

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
                        const int target_index = 64 + component * 192 + experiment * 12 + observation;
                        const float target_value = reference[target_index];
                        const float target_scale = 1.0f / fmaxf(fabsf(target_value), 1.0f);
                        const float residual = (state[component] - target_value) * target_scale;
                        evaluated_sse = fmaf(residual, residual, evaluated_sse);

                        if (statistics_phase) {
#pragma unroll
                            for (int row = 0; row < 6; ++row) {
                                const float jacobian_row =
                                    sensitivities[component * 6 + row] * target_scale;
                                gradient[row] = fmaf(jacobian_row, residual, gradient[row]);
#pragma unroll
                                for (int column = row; column < 6; ++column) {
                                    const int gram_index =
                                        row * 6 - row * (row - 1) / 2 + column - row;
                                    const float jacobian_column =
                                        sensitivities[component * 6 + column] * target_scale;
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
        constants_out[(unsigned long long)parameter * constants_leading_dimension + fit] =
            parameters[parameter];
    }
    mse_out[fit] = optimizer_valid ? current_sse / 768.0f : 3.402823466e+38F;
    iterations_out[fit] = completed_iterations;
    accepted_steps_out[fit] = accepted_steps;
}
