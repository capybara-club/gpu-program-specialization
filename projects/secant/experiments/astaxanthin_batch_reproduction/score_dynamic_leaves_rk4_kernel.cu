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
 * Dynamic-leaf score-only CUDA kernel for the astaxanthin batch model.
 *
 * One thread owns one constant setting, eight state-leaf bindings, and one
 * experimental trajectory. The binding indices are loaded once and remain in
 * registers. At every RK4 stage the thread writes [X, S1, S2, P] to a
 * state-major shared-memory tile and dynamically gathers eight leaf values.
 *
 * For block sizes divisible by a warp, state planes are separated by a whole
 * number of 32-bank rotations. A warp can therefore use different state
 * indices in every lane without introducing shared-memory bank conflicts.
 */

#define ASTAXANTHIN_STATE_COUNT 4
#define ASTAXANTHIN_TRAJECTORY_COUNT 3
#define ASTAXANTHIN_OBSERVATION_COUNT 12
#define ASTAXANTHIN_TARGET_COUNT \
    (ASTAXANTHIN_STATE_COUNT * ASTAXANTHIN_TRAJECTORY_COUNT * \
     ASTAXANTHIN_OBSERVATION_COUNT)

extern "C" {
__constant__ float astaxanthin_initial_state[
    ASTAXANTHIN_STATE_COUNT * ASTAXANTHIN_TRAJECTORY_COUNT];
__constant__ float astaxanthin_targets[ASTAXANTHIN_TARGET_COUNT];
}

extern "C" __global__
void astaxanthin_score_dynamic_leaves_rk4(
    const float *__restrict__ settings,
    unsigned long long settings_leading_dimension,
    const unsigned int *__restrict__ leaf_bindings,
    unsigned long long bindings_leading_dimension,
    unsigned long long num_settings,
    unsigned int steps_per_observation,
    float *__restrict__ mse_out,
    float *__restrict__ planted_final_state_out)
{
    extern __shared__ float state_tile_storage[];
    volatile float *state_tile = state_tile_storage;
    const unsigned long long setting =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int trajectory = blockIdx.y;
    if (setting >= num_settings ||
        trajectory >= ASTAXANTHIN_TRAJECTORY_COUNT ||
        steps_per_observation == 0u) return;

    const float c0 = settings[0ull * settings_leading_dimension + setting];
    const float c1 = settings[1ull * settings_leading_dimension + setting];
    const float c2 = settings[2ull * settings_leading_dimension + setting];
    const float c3 = settings[3ull * settings_leading_dimension + setting];
    const float c4 = settings[4ull * settings_leading_dimension + setting];
    const float c5 = settings[5ull * settings_leading_dimension + setting];
    const unsigned int b0 =
        leaf_bindings[0ull * bindings_leading_dimension + setting] & 3u;
    const unsigned int b1 =
        leaf_bindings[1ull * bindings_leading_dimension + setting] & 3u;
    const unsigned int b2 =
        leaf_bindings[2ull * bindings_leading_dimension + setting] & 3u;
    const unsigned int b3 =
        leaf_bindings[3ull * bindings_leading_dimension + setting] & 3u;
    const unsigned int b4 =
        leaf_bindings[4ull * bindings_leading_dimension + setting] & 3u;
    const unsigned int b5 =
        leaf_bindings[5ull * bindings_leading_dimension + setting] & 3u;
    const unsigned int b6 =
        leaf_bindings[6ull * bindings_leading_dimension + setting] & 3u;
    const unsigned int b7 =
        leaf_bindings[7ull * bindings_leading_dimension + setting] & 3u;

    const float mu_d = 0.0055f;
    const float Y_S1 = 2.58f;
    const float Y_S2 = 1.71f;
    const float beta = 0.21f;
    const float k_d = 0.0466f;
    const float h = 8.0f / (float)steps_per_observation;
    const float half_h = 0.5f * h;
    const float sixth_h = h / 6.0f;

    float biomass = astaxanthin_initial_state[
        0 * ASTAXANTHIN_TRAJECTORY_COUNT + trajectory];
    float glucose = astaxanthin_initial_state[
        1 * ASTAXANTHIN_TRAJECTORY_COUNT + trajectory];
    float sucrose = astaxanthin_initial_state[
        2 * ASTAXANTHIN_TRAJECTORY_COUNT + trajectory];
    float product = astaxanthin_initial_state[
        3 * ASTAXANTHIN_TRAJECTORY_COUNT + trajectory];
    float squared_error = 0.0f;
    int valid = 1;

#pragma unroll 1
    for (int observation = 0;
         observation < ASTAXANTHIN_OBSERVATION_COUNT;
         ++observation) {
#pragma unroll 1
        for (unsigned int step = 0; step < steps_per_observation; ++step) {
            const float base_biomass = biomass;
            const float base_glucose = glucose;
            const float base_sucrose = sucrose;
            const float base_product = product;
            float stage_biomass = biomass;
            float stage_glucose = glucose;
            float stage_sucrose = sucrose;
            float stage_product = product;
            float sum_biomass = 0.0f;
            float sum_glucose = 0.0f;
            float sum_sucrose = 0.0f;
            float sum_product = 0.0f;

#pragma unroll 4
            for (int stage = 0; stage < 4; ++stage) {
                state_tile[0u * blockDim.x + threadIdx.x] = stage_biomass;
                state_tile[1u * blockDim.x + threadIdx.x] = stage_glucose;
                state_tile[2u * blockDim.x + threadIdx.x] = stage_sucrose;
                state_tile[3u * blockDim.x + threadIdx.x] = stage_product;

                const float leaf0 = state_tile[b0 * blockDim.x + threadIdx.x];
                const float leaf1 = state_tile[b1 * blockDim.x + threadIdx.x];
                const float leaf2 = state_tile[b2 * blockDim.x + threadIdx.x];
                const float leaf3 = state_tile[b3 * blockDim.x + threadIdx.x];
                const float leaf4 = state_tile[b4 * blockDim.x + threadIdx.x];
                const float leaf5 = state_tile[b5 * blockDim.x + threadIdx.x];
                const float leaf6 = state_tile[b6 * blockDim.x + threadIdx.x];
                const float leaf7 = state_tile[b7 * blockDim.x + threadIdx.x];
                const float specific_growth_1 =
                    c0 * leaf0 / ((leaf1 + c1 * leaf2) * (1.0f + c2 * leaf3));
                const float specific_growth_2 =
                    c3 * leaf4 / ((leaf5 + c4 * leaf6) * (1.0f + c5 * leaf7));
                const float glucose_growth = specific_growth_1 * stage_biomass;
                const float sucrose_growth = specific_growth_2 * stage_biomass;
                const float d_biomass =
                    glucose_growth + sucrose_growth - mu_d * stage_biomass;
                const float d_glucose = -Y_S1 * glucose_growth;
                const float d_sucrose = -Y_S2 * sucrose_growth;
                const float d_product =
                    beta * stage_biomass - k_d * stage_biomass * stage_biomass;
                const float weight = (stage == 0 || stage == 3) ? 1.0f : 2.0f;

                sum_biomass = fmaf(weight, d_biomass, sum_biomass);
                sum_glucose = fmaf(weight, d_glucose, sum_glucose);
                sum_sucrose = fmaf(weight, d_sucrose, sum_sucrose);
                sum_product = fmaf(weight, d_product, sum_product);

                if (stage != 3) {
                    const float stage_h = stage == 2 ? h : half_h;
                    stage_biomass = fmaf(stage_h, d_biomass, base_biomass);
                    stage_glucose = fmaf(stage_h, d_glucose, base_glucose);
                    stage_sucrose = fmaf(stage_h, d_sucrose, base_sucrose);
                    stage_product = fmaf(stage_h, d_product, base_product);
                }
            }

            biomass = fmaf(sixth_h, sum_biomass, base_biomass);
            glucose = fmaf(sixth_h, sum_glucose, base_glucose);
            sucrose = fmaf(sixth_h, sum_sucrose, base_sucrose);
            product = fmaf(sixth_h, sum_product, base_product);
        }

        valid = valid && isfinite(biomass) && isfinite(glucose) &&
                isfinite(sucrose) && isfinite(product);
        const int target_base =
            trajectory * ASTAXANTHIN_OBSERVATION_COUNT + observation;
        const float error_biomass = biomass - astaxanthin_targets[
            0 * ASTAXANTHIN_TRAJECTORY_COUNT * ASTAXANTHIN_OBSERVATION_COUNT +
            target_base];
        const float error_glucose = glucose - astaxanthin_targets[
            1 * ASTAXANTHIN_TRAJECTORY_COUNT * ASTAXANTHIN_OBSERVATION_COUNT +
            target_base];
        const float error_sucrose = sucrose - astaxanthin_targets[
            2 * ASTAXANTHIN_TRAJECTORY_COUNT * ASTAXANTHIN_OBSERVATION_COUNT +
            target_base];
        const float error_product = product - astaxanthin_targets[
            3 * ASTAXANTHIN_TRAJECTORY_COUNT * ASTAXANTHIN_OBSERVATION_COUNT +
            target_base];
        squared_error = fmaf(error_biomass, error_biomass, squared_error);
        squared_error = fmaf(error_glucose, error_glucose, squared_error);
        squared_error = fmaf(error_sucrose, error_sucrose, squared_error);
        squared_error = fmaf(error_product, error_product, squared_error);
    }

    mse_out[(unsigned long long)trajectory * num_settings + setting] =
        valid ? squared_error /
            (float)(ASTAXANTHIN_STATE_COUNT * ASTAXANTHIN_OBSERVATION_COUNT) :
            3.402823466e+38F;
    if (setting == 0ull) {
        planted_final_state_out[trajectory * ASTAXANTHIN_STATE_COUNT + 0] = biomass;
        planted_final_state_out[trajectory * ASTAXANTHIN_STATE_COUNT + 1] = glucose;
        planted_final_state_out[trajectory * ASTAXANTHIN_STATE_COUNT + 2] = sucrose;
        planted_final_state_out[trajectory * ASTAXANTHIN_STATE_COUNT + 3] = product;
    }
}

#undef ASTAXANTHIN_TARGET_COUNT
#undef ASTAXANTHIN_OBSERVATION_COUNT
#undef ASTAXANTHIN_TRAJECTORY_COUNT
#undef ASTAXANTHIN_STATE_COUNT
