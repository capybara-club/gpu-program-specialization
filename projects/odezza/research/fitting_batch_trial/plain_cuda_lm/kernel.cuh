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
// Fork of scratch/cooperative_lm/cooperative_lm.cu. Plain compiled CUDA only.
#ifndef ODEZZA_FIT_THREAD_COUNT
#define ODEZZA_FIT_THREAD_COUNT 4
#endif

#ifndef ODEZZA_DISTRIBUTED_SOLVE
#define ODEZZA_DISTRIBUTED_SOLVE 1
#endif

#ifndef ODEZZA_LEADER_EVALUATION
#define ODEZZA_LEADER_EVALUATION 0
#endif

#include "model.cuh"
#define ODEZZA_STATE_COUNT MODEL_STATES
#define ODEZZA_PARAMETER_COUNT MODEL_PARAMETERS
#define ODEZZA_GRAM_COUNT (MODEL_PARAMETERS*(MODEL_PARAMETERS+1)/2)
#define ODEZZA_LOCAL_PARAMETER_COUNT ((ODEZZA_PARAMETER_COUNT + ODEZZA_FIT_THREAD_COUNT - 1) / ODEZZA_FIT_THREAD_COUNT)
#define ODEZZA_LOCAL_OPTIMIZER_COUNT ((ODEZZA_PARAMETER_COUNT + ODEZZA_GRAM_COUNT + ODEZZA_FIT_THREAD_COUNT - 1) / ODEZZA_FIT_THREAD_COUNT)
#define ODEZZA_LOCAL_FACTOR_COUNT ((ODEZZA_GRAM_COUNT + ODEZZA_FIT_THREAD_COUNT - 1) / ODEZZA_FIT_THREAD_COUNT)
#define ODEZZA_LOCAL_SOLVE_COUNT ((ODEZZA_PARAMETER_COUNT + ODEZZA_FIT_THREAD_COUNT - 1) / ODEZZA_FIT_THREAD_COUNT)

static_assert(
    ODEZZA_FIT_THREAD_COUNT == 1 ||
    ODEZZA_FIT_THREAD_COUNT == 2 ||
    ODEZZA_FIT_THREAD_COUNT == 4 ||
    ODEZZA_FIT_THREAD_COUNT == 8 ||
    ODEZZA_FIT_THREAD_COUNT == 16 ||
    ODEZZA_FIT_THREAD_COUNT == 32
);

extern "C" __global__ void odezza_trajectory_lm(
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
) {
    if (
        start_count == 0ull ||
        trajectory_count == 0u ||
        trajectory_point_count < trajectory_count ||
        active_toggle_count != 0u ||
        steps_per_interval == 0u ||
        max_lm_iterations == 0u ||
        max_damping_attempts == 0u ||
        !isfinite(initial_damping) ||
        initial_damping <= 0.0f ||
        blockDim.x != 32u
    ) return;

    const unsigned int fit_lane = threadIdx.x & (ODEZZA_FIT_THREAD_COUNT - 1u);
    const unsigned int fit_base = threadIdx.x - fit_lane;
    const unsigned int fit_mask = ODEZZA_FIT_THREAD_COUNT == 32
        ? 0xffffffffu
        : (unsigned int)(((1ull << ODEZZA_FIT_THREAD_COUNT) - 1ull) << fit_base);
    const unsigned long long global_thread = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long fit = global_thread / ODEZZA_FIT_THREAD_COUNT;

    const unsigned int reference_float_count = ODEZZA_STATE_COUNT * trajectory_point_count;
    extern __shared__ unsigned int shared_words[];
    float *reference = reinterpret_cast<float *>(shared_words);
    float *weights = reference + reference_float_count;
    float *step_sizes = weights + reference_float_count;
    unsigned int *offsets = reinterpret_cast<unsigned int *>(step_sizes + trajectory_point_count);
    for (unsigned int index = threadIdx.x; index < reference_float_count; index += blockDim.x) {
        reference[index] = reference_data[index];
        weights[index] = reference_weights[index];
    }
    for (unsigned int index = threadIdx.x; index < trajectory_point_count; index += blockDim.x) {
        step_sizes[index] = index == 0u
            ? 0.0f
            : (trajectory_times[index] - trajectory_times[index - 1u]) / (float)steps_per_interval;
    }
    for (unsigned int index = threadIdx.x; index <= trajectory_count; index += blockDim.x) {
        offsets[index] = trajectory_offsets[index];
    }
    __syncthreads();
    if (fit >= start_count) return;

    float parameters[ODEZZA_PARAMETER_COUNT];
#pragma unroll
    for (int parameter = 0; parameter < ODEZZA_PARAMETER_COUNT; ++parameter) {
        parameters[parameter] = starts[fit * ODEZZA_PARAMETER_COUNT + parameter];
    }

    if (fit_lane == 0u) initial_mse_out[fit] = 3.402823466e+38f;
    float lambda = fminf(fmaxf(initial_damping, 1.0e-8f), 1.0e8f);
    float current_sse = 3.402823466e+38f;
    unsigned int current_residual_count = 0u;
    unsigned int completed_iterations = 0u;
    unsigned int accepted_step_count = 0u;
    unsigned int factorization_attempt_count = 0u;
    int optimizer_valid = 1;

    for (
        unsigned int lm_iteration = 0u;
        lm_iteration < max_lm_iterations && optimizer_valid;
        ++lm_iteration
    ) {
        float local_optimizer[ODEZZA_LOCAL_OPTIMIZER_COUNT];
        float proposal[ODEZZA_PARAMETER_COUNT];
#pragma unroll
        for (int index = 0; index < ODEZZA_LOCAL_OPTIMIZER_COUNT; ++index) {
            local_optimizer[index] = 0.0f;
        }
        int accepted = 0;

        for (unsigned int phase = 0u; phase <= max_damping_attempts && !accepted; ++phase) {
            const int statistics_phase = phase == 0u;
            int proposal_ready = statistics_phase;
            if (!statistics_phase) {
                ++factorization_attempt_count;
#if ODEZZA_DISTRIBUTED_SOLVE
                float local_factor[ODEZZA_LOCAL_FACTOR_COUNT] = {};
                float local_solve[ODEZZA_LOCAL_SOLVE_COUNT] = {};

#pragma unroll
                for (int gram_index = 0; gram_index < ODEZZA_GRAM_COUNT; ++gram_index) {
                    const int source_index = ODEZZA_PARAMETER_COUNT + gram_index;
                    const int source_lane = source_index & (ODEZZA_FIT_THREAD_COUNT - 1);
                    const int source_local = source_index / ODEZZA_FIT_THREAD_COUNT;
                    const float value = __shfl_sync(
                        fit_mask,
                        local_optimizer[source_local],
                        source_lane,
                        ODEZZA_FIT_THREAD_COUNT
                    );
                    const int factor_lane = gram_index & (ODEZZA_FIT_THREAD_COUNT - 1);
                    if (fit_lane == (unsigned int)factor_lane) {
                        local_factor[gram_index / ODEZZA_FIT_THREAD_COUNT] = value;
                    }
                }
#pragma unroll
                for (int row = 0; row < ODEZZA_PARAMETER_COUNT; ++row) {
                    const int owner = row & (ODEZZA_FIT_THREAD_COUNT - 1);
                    const int local = row / ODEZZA_FIT_THREAD_COUNT;
                    const float value = -__shfl_sync(
                        fit_mask,
                        local_optimizer[local],
                        owner,
                        ODEZZA_FIT_THREAD_COUNT
                    );
                    if (fit_lane == (unsigned int)owner) local_solve[local] = value;
                }
                __syncwarp(fit_mask);

                int factor_ok = 1;
#pragma unroll
                for (int row = 0; row < ODEZZA_PARAMETER_COUNT; ++row) {
#pragma unroll
                    for (int column = 0; column <= row; ++column) {
                        const int gram_index = column * ODEZZA_PARAMETER_COUNT
                            - column * (column - 1) / 2 + row - column;
                        const int owner = gram_index & (ODEZZA_FIT_THREAD_COUNT - 1);
                        const int local = gram_index / ODEZZA_FIT_THREAD_COUNT;
                        float sum = __shfl_sync(
                            fit_mask,
                            local_factor[local],
                            owner,
                            ODEZZA_FIT_THREAD_COUNT
                        );
                        if (row == column) sum += lambda * fmaxf(fabsf(sum), 1.0e-8f);
#pragma unroll
                        for (int k = 0; k < column; ++k) {
                            const int row_index = k * ODEZZA_PARAMETER_COUNT
                                - k * (k - 1) / 2 + row - k;
                            const int column_index = k * ODEZZA_PARAMETER_COUNT
                                - k * (k - 1) / 2 + column - k;
                            const float row_value = __shfl_sync(
                                fit_mask,
                                local_factor[row_index / ODEZZA_FIT_THREAD_COUNT],
                                row_index & (ODEZZA_FIT_THREAD_COUNT - 1),
                                ODEZZA_FIT_THREAD_COUNT
                            );
                            const float column_value = __shfl_sync(
                                fit_mask,
                                local_factor[column_index / ODEZZA_FIT_THREAD_COUNT],
                                column_index & (ODEZZA_FIT_THREAD_COUNT - 1),
                                ODEZZA_FIT_THREAD_COUNT
                            );
                            sum = fmaf(-row_value, column_value, sum);
                        }
                        float factor_value;
                        if (row == column) {
                            const int pivot_ok = factor_ok && isfinite(sum) && sum > 1.0e-12f;
                            factor_value = pivot_ok ? sqrtf(sum) : 1.0f;
                            factor_ok = pivot_ok;
                        } else {
                            const int diagonal_index = column * ODEZZA_PARAMETER_COUNT
                                - column * (column - 1) / 2;
                            const float diagonal = __shfl_sync(
                                fit_mask,
                                local_factor[diagonal_index / ODEZZA_FIT_THREAD_COUNT],
                                diagonal_index & (ODEZZA_FIT_THREAD_COUNT - 1),
                                ODEZZA_FIT_THREAD_COUNT
                            );
                            factor_value = factor_ok ? sum / diagonal : 0.0f;
                            factor_ok = factor_ok && isfinite(factor_value);
                        }
                        if (fit_lane == (unsigned int)owner) local_factor[local] = factor_value;
                        __syncwarp(fit_mask);
                    }
                }

#pragma unroll
                for (int row = 0; row < ODEZZA_PARAMETER_COUNT; ++row) {
                    const int owner = row & (ODEZZA_FIT_THREAD_COUNT - 1);
                    const int local = row / ODEZZA_FIT_THREAD_COUNT;
                    float value = __shfl_sync(
                        fit_mask,
                        local_solve[local],
                        owner,
                        ODEZZA_FIT_THREAD_COUNT
                    );
#pragma unroll
                    for (int k = 0; k < row; ++k) {
                        const int factor_index = k * ODEZZA_PARAMETER_COUNT
                            - k * (k - 1) / 2 + row - k;
                        const float factor_value = __shfl_sync(
                            fit_mask,
                            local_factor[factor_index / ODEZZA_FIT_THREAD_COUNT],
                            factor_index & (ODEZZA_FIT_THREAD_COUNT - 1),
                            ODEZZA_FIT_THREAD_COUNT
                        );
                        const float solved = __shfl_sync(
                            fit_mask,
                            local_solve[k / ODEZZA_FIT_THREAD_COUNT],
                            k & (ODEZZA_FIT_THREAD_COUNT - 1),
                            ODEZZA_FIT_THREAD_COUNT
                        );
                        value = fmaf(-factor_value, solved, value);
                    }
                    const int diagonal_index = row * ODEZZA_PARAMETER_COUNT
                        - row * (row - 1) / 2;
                    const float diagonal = __shfl_sync(
                        fit_mask,
                        local_factor[diagonal_index / ODEZZA_FIT_THREAD_COUNT],
                        diagonal_index & (ODEZZA_FIT_THREAD_COUNT - 1),
                        ODEZZA_FIT_THREAD_COUNT
                    );
                    const float solved = factor_ok ? value / diagonal : 0.0f;
                    factor_ok = factor_ok && isfinite(solved);
                    if (fit_lane == (unsigned int)owner) local_solve[local] = solved;
                    __syncwarp(fit_mask);
                }

#pragma unroll
                for (int reverse = 0; reverse < ODEZZA_PARAMETER_COUNT; ++reverse) {
                    const int row = ODEZZA_PARAMETER_COUNT - 1 - reverse;
                    const int owner = row & (ODEZZA_FIT_THREAD_COUNT - 1);
                    const int local = row / ODEZZA_FIT_THREAD_COUNT;
                    float value = __shfl_sync(
                        fit_mask,
                        local_solve[local],
                        owner,
                        ODEZZA_FIT_THREAD_COUNT
                    );
#pragma unroll
                    for (int k = row + 1; k < ODEZZA_PARAMETER_COUNT; ++k) {
                        const int factor_index = row * ODEZZA_PARAMETER_COUNT
                            - row * (row - 1) / 2 + k - row;
                        const float factor_value = __shfl_sync(
                            fit_mask,
                            local_factor[factor_index / ODEZZA_FIT_THREAD_COUNT],
                            factor_index & (ODEZZA_FIT_THREAD_COUNT - 1),
                            ODEZZA_FIT_THREAD_COUNT
                        );
                        const float delta_value = __shfl_sync(
                            fit_mask,
                            local_solve[k / ODEZZA_FIT_THREAD_COUNT],
                            k & (ODEZZA_FIT_THREAD_COUNT - 1),
                            ODEZZA_FIT_THREAD_COUNT
                        );
                        value = fmaf(-factor_value, delta_value, value);
                    }
                    const int diagonal_index = row * ODEZZA_PARAMETER_COUNT
                        - row * (row - 1) / 2;
                    const float diagonal = __shfl_sync(
                        fit_mask,
                        local_factor[diagonal_index / ODEZZA_FIT_THREAD_COUNT],
                        diagonal_index & (ODEZZA_FIT_THREAD_COUNT - 1),
                        ODEZZA_FIT_THREAD_COUNT
                    );
                    const float delta_value = factor_ok ? value / diagonal : 0.0f;
                    factor_ok = factor_ok && isfinite(delta_value);
                    if (fit_lane == (unsigned int)owner) local_solve[local] = delta_value;
                    __syncwarp(fit_mask);
                }

#pragma unroll
                for (int parameter = 0; parameter < ODEZZA_PARAMETER_COUNT; ++parameter) {
                    const float delta_value = __shfl_sync(
                        fit_mask,
                        local_solve[parameter / ODEZZA_FIT_THREAD_COUNT],
                        parameter & (ODEZZA_FIT_THREAD_COUNT - 1),
                        ODEZZA_FIT_THREAD_COUNT
                    );
                    proposal[parameter] = fminf(1.5f, fmaxf(-1.5f, parameters[parameter] + delta_value));
                    factor_ok = factor_ok && isfinite(proposal[parameter]);
                }
                proposal_ready = factor_ok;
#else
                float factor[ODEZZA_PARAMETER_COUNT * ODEZZA_PARAMETER_COUNT] = {};
                float solve_rhs[ODEZZA_PARAMETER_COUNT];
                float delta[ODEZZA_PARAMETER_COUNT];

#pragma unroll
                for (int row = 0; row < ODEZZA_PARAMETER_COUNT; ++row) {
                    const int gradient_owner = row & (ODEZZA_FIT_THREAD_COUNT - 1);
                    const int gradient_local = row / ODEZZA_FIT_THREAD_COUNT;
                    solve_rhs[row] = -__shfl_sync(
                        fit_mask,
                        local_optimizer[gradient_local],
                        gradient_owner,
                        ODEZZA_FIT_THREAD_COUNT
                    );
#pragma unroll
                    for (int column = 0; column <= row; ++column) {
                        const int gram_index = column * ODEZZA_PARAMETER_COUNT
                            - column * (column - 1) / 2 + row - column;
                        const int optimizer_index = ODEZZA_PARAMETER_COUNT + gram_index;
                        const int gram_owner = optimizer_index & (ODEZZA_FIT_THREAD_COUNT - 1);
                        const int gram_local = optimizer_index / ODEZZA_FIT_THREAD_COUNT;
                        factor[row * ODEZZA_PARAMETER_COUNT + column] = __shfl_sync(
                            fit_mask,
                            local_optimizer[gram_local],
                            gram_owner,
                            ODEZZA_FIT_THREAD_COUNT
                        );
                    }
                }

                int factor_ok = 1;
#pragma unroll
                for (int row = 0; row < ODEZZA_PARAMETER_COUNT; ++row) {
#pragma unroll
                    for (int column = 0; column <= row; ++column) {
                        float sum = factor[row * ODEZZA_PARAMETER_COUNT + column];
                        if (row == column) sum += lambda * fmaxf(fabsf(sum), 1.0e-8f);
#pragma unroll
                        for (int k = 0; k < column; ++k) {
                            sum = fmaf(
                                -factor[row * ODEZZA_PARAMETER_COUNT + k],
                                factor[column * ODEZZA_PARAMETER_COUNT + k],
                                sum
                            );
                        }
                        if (row == column) {
                            const int pivot_ok = factor_ok && isfinite(sum) && sum > 1.0e-12f;
                            factor[row * ODEZZA_PARAMETER_COUNT + column] = pivot_ok ? sqrtf(sum) : 1.0f;
                            factor_ok = pivot_ok;
                        } else {
                            const float value = factor_ok
                                ? sum / factor[column * ODEZZA_PARAMETER_COUNT + column]
                                : 0.0f;
                            factor[row * ODEZZA_PARAMETER_COUNT + column] = value;
                            factor_ok = factor_ok && isfinite(value);
                        }
                    }
                }

#pragma unroll
                for (int row = 0; row < ODEZZA_PARAMETER_COUNT; ++row) {
                    float value = solve_rhs[row];
#pragma unroll
                    for (int k = 0; k < row; ++k) {
                        value = fmaf(-factor[row * ODEZZA_PARAMETER_COUNT + k], solve_rhs[k], value);
                    }
                    solve_rhs[row] = factor_ok
                        ? value / factor[row * ODEZZA_PARAMETER_COUNT + row]
                        : 0.0f;
                    factor_ok = factor_ok && isfinite(solve_rhs[row]);
                }

#pragma unroll
                for (int reverse = 0; reverse < ODEZZA_PARAMETER_COUNT; ++reverse) {
                    const int row = ODEZZA_PARAMETER_COUNT - 1 - reverse;
                    float value = solve_rhs[row];
#pragma unroll
                    for (int k = row + 1; k < ODEZZA_PARAMETER_COUNT; ++k) {
                        value = fmaf(-factor[k * ODEZZA_PARAMETER_COUNT + row], delta[k], value);
                    }
                    delta[row] = factor_ok
                        ? value / factor[row * ODEZZA_PARAMETER_COUNT + row]
                        : 0.0f;
                    factor_ok = factor_ok && isfinite(delta[row]);
                }

#pragma unroll
                for (int parameter = 0; parameter < ODEZZA_PARAMETER_COUNT; ++parameter) {
                    proposal[parameter] = fminf(1.5f, fmaxf(-1.5f, parameters[parameter] + delta[parameter]));
                    factor_ok = factor_ok && isfinite(proposal[parameter]);
                }
                proposal_ready = factor_ok;
#endif
                if (!proposal_ready) {
                    lambda = fminf(lambda * 10.0f, 1.0e8f);
                    continue;
                }
            }

            float evaluation_parameters[ODEZZA_PARAMETER_COUNT];
#pragma unroll
            for (int parameter = 0; parameter < ODEZZA_PARAMETER_COUNT; ++parameter) {
                evaluation_parameters[parameter] = statistics_phase
                    ? parameters[parameter]
                    : proposal[parameter];
            }

            float evaluated_sse = 0.0f;
            unsigned int evaluated_residual_count = 0u;
            unsigned int expected_start = 0u;
            int evaluated_valid = proposal_ready;
            float sensitivities[ODEZZA_STATE_COUNT * ODEZZA_LOCAL_PARAMETER_COUNT];
#pragma unroll
            for (int index = 0; index < ODEZZA_STATE_COUNT * ODEZZA_LOCAL_PARAMETER_COUNT; ++index) {
                sensitivities[index] = 0.0f;
            }

            for (unsigned int trajectory = 0u; trajectory < trajectory_count && evaluated_valid; ++trajectory) {
                const unsigned int point_begin = offsets[trajectory];
                const unsigned int point_end = offsets[trajectory + 1u];
                if (point_begin != expected_start || point_end <= point_begin || point_end > trajectory_point_count) {
                    evaluated_valid = 0;
                    break;
                }
                expected_start = point_end;
                float state[ODEZZA_STATE_COUNT];
#pragma unroll
                for (int component = 0; component < ODEZZA_STATE_COUNT; ++component) {
                    state[component] = reference[component * trajectory_point_count + point_begin];
                }
#pragma unroll
                for (int index = 0; index < ODEZZA_STATE_COUNT * ODEZZA_LOCAL_PARAMETER_COUNT; ++index) {
                    sensitivities[index] = 0.0f;
                }

                for (unsigned int point = point_begin + 1u; point < point_end && evaluated_valid; ++point) {
                    const float h = step_sizes[point];
                    evaluated_valid = evaluated_valid && isfinite(h) && h > 0.0f;
                    const float half_h = 0.5f * h;
                    const float sixth_h = h / 6.0f;

                    for (unsigned int step = 0u; step < steps_per_interval; ++step) {
                        float base_state[ODEZZA_STATE_COUNT];
                        float stage_state[ODEZZA_STATE_COUNT];
                        float sum_state[ODEZZA_STATE_COUNT];
                        float stage_sensitivity[ODEZZA_STATE_COUNT * ODEZZA_LOCAL_PARAMETER_COUNT];
                        float sum_sensitivity[ODEZZA_STATE_COUNT * ODEZZA_LOCAL_PARAMETER_COUNT];
#pragma unroll
                        for (int component = 0; component < ODEZZA_STATE_COUNT; ++component) {
                            base_state[component] = state[component];
                            stage_state[component] = state[component];
                            sum_state[component] = 0.0f;
                        }
#pragma unroll
                        for (int index = 0; index < ODEZZA_STATE_COUNT * ODEZZA_LOCAL_PARAMETER_COUNT; ++index) {
                            stage_sensitivity[index] = sensitivities[index];
                            sum_sensitivity[index] = 0.0f;
                        }

#pragma unroll
                        for (int rk_stage = 0; rk_stage < 4; ++rk_stage) {
                            float rhs[ODEZZA_STATE_COUNT];
                            float stage_derivative[ODEZZA_STATE_COUNT * ODEZZA_LOCAL_PARAMETER_COUNT];
// Each lane owns complete sensitivity columns. Directional forward AD
                            // evaluates the statically compiled RHS; no runtime AST or patch sites.
#pragma unroll
                            for (int lp = 0; lp < ODEZZA_LOCAL_PARAMETER_COUNT; ++lp) {
                                const int parameter = fit_lane + lp * ODEZZA_FIT_THREAD_COUNT;
                                Dual x[MODEL_STATES], p[MODEL_PARAMETERS], f[MODEL_STATES];
#pragma unroll
                                for (int i=0; i<MODEL_STATES; ++i)
                                    x[i] = Dual(stage_state[i], statistics_phase && parameter < MODEL_PARAMETERS
                                        ? stage_sensitivity[i*ODEZZA_LOCAL_PARAMETER_COUNT+lp] : 0.0f);
#pragma unroll
                                for (int j=0; j<MODEL_PARAMETERS; ++j)
                                    p[j] = Dual(evaluation_parameters[j], statistics_phase && parameter==j ? 1.0f : 0.0f);
                                model_rhs((int)(fit % MODEL_VARIANTS), x, p, f);
#pragma unroll
                                for (int i=0; i<MODEL_STATES; ++i) {
                                    // One shared primal prevents lane-specific arithmetic from
                                    // allowing subgroup control flow to disagree.
                                    if (lp==0) rhs[i] = __shfl_sync(fit_mask, f[i].v, 0, ODEZZA_FIT_THREAD_COUNT);
                                    stage_derivative[i*ODEZZA_LOCAL_PARAMETER_COUNT+lp] =
                                        statistics_phase && parameter < MODEL_PARAMETERS ? f[i].d : 0.0f;
                                }
                            }

                            const float stage_weight = rk_stage == 0 || rk_stage == 3 ? 1.0f : 2.0f;
#pragma unroll
                            for (int output = 0; output < ODEZZA_STATE_COUNT; ++output) {
                                sum_state[output] = fmaf(stage_weight, rhs[output], sum_state[output]);
#pragma unroll
                                for (int local_parameter = 0; local_parameter < ODEZZA_LOCAL_PARAMETER_COUNT; ++local_parameter) {
                                    const int sensitivity_index = output * ODEZZA_LOCAL_PARAMETER_COUNT
                                        + local_parameter;
                                    sum_sensitivity[sensitivity_index] = fmaf(
                                        stage_weight,
                                        stage_derivative[sensitivity_index],
                                        sum_sensitivity[sensitivity_index]
                                    );
                                    if (rk_stage != 3) {
                                        const float stage_h = rk_stage == 2 ? h : half_h;
                                        stage_sensitivity[sensitivity_index] = fmaf(
                                            stage_h,
                                            stage_derivative[sensitivity_index],
                                            sensitivities[sensitivity_index]
                                        );
                                    }
                                }
                            }
                            if (rk_stage != 3) {
                                const float stage_h = rk_stage == 2 ? h : half_h;
#pragma unroll
                                for (int component = 0; component < ODEZZA_STATE_COUNT; ++component) {
                                    stage_state[component] = fmaf(stage_h, rhs[component], base_state[component]);
                                }
                            }
                        }

#pragma unroll
                        for (int component = 0; component < ODEZZA_STATE_COUNT; ++component) {
                            state[component] = fmaf(sixth_h, sum_state[component], base_state[component]);
#pragma unroll
                            for (int local_parameter = 0; local_parameter < ODEZZA_LOCAL_PARAMETER_COUNT; ++local_parameter) {
                                const int sensitivity_index = component * ODEZZA_LOCAL_PARAMETER_COUNT
                                    + local_parameter;
                                if (statistics_phase) {
                                    sensitivities[sensitivity_index] = fmaf(
                                        sixth_h,
                                        sum_sensitivity[sensitivity_index],
                                        sensitivities[sensitivity_index]
                                    );
                                }
                            }
                        }
                    }

#pragma unroll
                    for (int component = 0; component < ODEZZA_STATE_COUNT; ++component) {
                        evaluated_valid = evaluated_valid && isfinite(state[component]);
                        const float target = reference[component * trajectory_point_count + point];
                        const float residual_weight = weights[component * trajectory_point_count + point];
                        evaluated_valid = evaluated_valid && isfinite(residual_weight) && residual_weight >= 0.0f;
                        const float residual = (state[component] - target) * residual_weight;
                        evaluated_sse = fmaf(residual, residual, evaluated_sse);
                        evaluated_residual_count += residual_weight > 0.0f;
                        if (statistics_phase) {
                            float jacobian[ODEZZA_PARAMETER_COUNT];
#pragma unroll
                            for (int parameter = 0; parameter < ODEZZA_PARAMETER_COUNT; ++parameter) {
                                const int owner = parameter & (ODEZZA_FIT_THREAD_COUNT - 1);
                                const int local_parameter = parameter / ODEZZA_FIT_THREAD_COUNT;
                                const float owned_jacobian = fit_lane == (unsigned int)owner
                                    ? sensitivities[component * ODEZZA_LOCAL_PARAMETER_COUNT + local_parameter]
                                        * residual_weight
                                    : 0.0f;
                                jacobian[parameter] = __shfl_sync(
                                    fit_mask,
                                    owned_jacobian,
                                    owner,
                                    ODEZZA_FIT_THREAD_COUNT
                                );
                            }
#pragma unroll
                            for (int row = 0; row < ODEZZA_PARAMETER_COUNT; ++row) {
                                if (fit_lane == (unsigned int)(row & (ODEZZA_FIT_THREAD_COUNT - 1))) {
                                    local_optimizer[row / ODEZZA_FIT_THREAD_COUNT] = fmaf(
                                        jacobian[row],
                                        residual,
                                        local_optimizer[row / ODEZZA_FIT_THREAD_COUNT]
                                    );
                                }
#pragma unroll
                                for (int column = row; column < ODEZZA_PARAMETER_COUNT; ++column) {
                                    const int gram_index = row * ODEZZA_PARAMETER_COUNT
                                        - row * (row - 1) / 2 + column - row;
                                    const int optimizer_index = ODEZZA_PARAMETER_COUNT + gram_index;
                                    if (fit_lane == (unsigned int)(optimizer_index & (ODEZZA_FIT_THREAD_COUNT - 1))) {
                                        local_optimizer[optimizer_index / ODEZZA_FIT_THREAD_COUNT] = fmaf(
                                            jacobian[row],
                                            jacobian[column],
                                            local_optimizer[optimizer_index / ODEZZA_FIT_THREAD_COUNT]
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }

            evaluated_valid = evaluated_valid
                && expected_start == trajectory_point_count
                && evaluated_residual_count != 0u
                && isfinite(evaluated_sse);
            if (!evaluated_valid) {
                if (statistics_phase) {
                    optimizer_valid = 0;
                    break;
                }
                lambda = fminf(lambda * 10.0f, 1.0e8f);
                continue;
            }
            if (statistics_phase) {
                current_sse = evaluated_sse;
                current_residual_count = evaluated_residual_count;
                if (lm_iteration == 0u && fit_lane == 0u) {
                    initial_mse_out[fit] = current_sse / (float)current_residual_count;
                }
                __syncwarp(fit_mask);
                continue;
            }
            if (evaluated_sse < current_sse) {
                const float improvement = current_sse - evaluated_sse;
#pragma unroll
                for (int parameter = 0; parameter < ODEZZA_PARAMETER_COUNT; ++parameter) {
                    parameters[parameter] = proposal[parameter];
                }
                current_sse = evaluated_sse;
                current_residual_count = evaluated_residual_count;
                lambda = fmaxf(lambda * 0.33333334f, 1.0e-8f);
                accepted = 1;
                ++accepted_step_count;
                if (improvement <= 1.0e-7f * fmaxf(1.0e-30f, current_sse)) {
                    lm_iteration = max_lm_iterations - 1u;
                }
            } else {
                lambda = fminf(lambda * 10.0f, 1.0e8f);
            }
        }

        ++completed_iterations;
        if (!accepted || current_sse <= 1.0e-13f * (float)current_residual_count) break;
    }

    if (fit_lane == 0u) {
#pragma unroll
        for (int parameter = 0; parameter < ODEZZA_PARAMETER_COUNT; ++parameter) {
            constants_out[fit * ODEZZA_PARAMETER_COUNT + parameter] = parameters[parameter];
        }
        mse_out[fit] = optimizer_valid && current_residual_count != 0u
            ? current_sse / (float)current_residual_count
            : 3.402823466e+38f;
        iterations_out[fit] = completed_iterations;
        accepted_steps_out[fit] = accepted_step_count;
        factorization_attempts_out[fit] = factorization_attempt_count;
    }
}
