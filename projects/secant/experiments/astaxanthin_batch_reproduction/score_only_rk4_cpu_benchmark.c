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
#define _POSIX_C_SOURCE 200112L

#include "astaxanthin_ground_truth.h"

#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    PARAMETER_COUNT = 6,
    STATE_COUNT = 4,
    TRAJECTORY_COUNT = 3,
    OBSERVATION_COUNT = 12,
    TARGET_COUNT = STATE_COUNT * TRAJECTORY_COUNT * OBSERVATION_COUNT
};

#if defined(ASTAXANTHIN_SCALAR)

typedef float lane_float;
typedef int lane_mask;
#define LANE_COUNT 1
#define SIMD_NAME "scalar"
static inline lane_float lane_load(const float *values) { return *values; }
static inline void lane_store(float *values, lane_float value) { *values = value; }
static inline lane_float lane_set(float value) { return value; }
static inline lane_float lane_add(lane_float a, lane_float b) { return a + b; }
static inline lane_float lane_sub(lane_float a, lane_float b) { return a - b; }
static inline lane_float lane_mul(lane_float a, lane_float b) { return a * b; }
static inline lane_float lane_div(lane_float a, lane_float b) { return a / b; }
static inline lane_float lane_fma(
    lane_float a, lane_float b, lane_float c) { return fmaf(a, b, c); }
static inline lane_mask lane_all_valid(void) { return 1; }
static inline lane_mask lane_and(lane_mask a, lane_mask b) { return a && b; }
static inline lane_mask lane_finite(lane_float value) { return isfinite(value); }
static inline lane_float lane_select_valid(
    lane_mask mask, lane_float value) { return mask ? value : 3.402823466e+38F; }
static inline float lane_first(lane_float value) { return value; }

#elif defined(__AVX512F__)

typedef __m512 lane_float;
typedef __mmask16 lane_mask;
#define LANE_COUNT 16
#define SIMD_NAME "avx512"
static inline lane_float lane_load(const float *values) { return _mm512_load_ps(values); }
static inline void lane_store(float *values, lane_float value) { _mm512_store_ps(values, value); }
static inline lane_float lane_set(float value) { return _mm512_set1_ps(value); }
static inline lane_float lane_add(lane_float a, lane_float b) { return _mm512_add_ps(a, b); }
static inline lane_float lane_sub(lane_float a, lane_float b) { return _mm512_sub_ps(a, b); }
static inline lane_float lane_mul(lane_float a, lane_float b) { return _mm512_mul_ps(a, b); }
static inline lane_float lane_div(lane_float a, lane_float b) { return a / b; }
static inline lane_float lane_fma(
    lane_float a, lane_float b, lane_float c) { return _mm512_fmadd_ps(a, b, c); }
static inline lane_mask lane_all_valid(void) { return (__mmask16)0xffffu; }
static inline lane_mask lane_and(lane_mask a, lane_mask b) { return a & b; }
static inline lane_mask lane_finite(lane_float value)
{
    const lane_float absolute = _mm512_andnot_ps(_mm512_set1_ps(-0.0f), value);
    return _mm512_cmp_ps_mask(
        absolute, _mm512_set1_ps(INFINITY), _CMP_LT_OQ);
}
static inline lane_float lane_select_valid(lane_mask mask, lane_float value)
{
    return _mm512_mask_blend_ps(
        mask, _mm512_set1_ps(3.402823466e+38F), value);
}
static inline float lane_first(lane_float value) { return _mm512_cvtss_f32(value); }

#elif defined(__AVX2__)

typedef __m256 lane_float;
typedef __m256 lane_mask;
#define LANE_COUNT 8
#define SIMD_NAME "avx2"
static inline lane_float lane_load(const float *values) { return _mm256_load_ps(values); }
static inline void lane_store(float *values, lane_float value) { _mm256_store_ps(values, value); }
static inline lane_float lane_set(float value) { return _mm256_set1_ps(value); }
static inline lane_float lane_add(lane_float a, lane_float b) { return _mm256_add_ps(a, b); }
static inline lane_float lane_sub(lane_float a, lane_float b) { return _mm256_sub_ps(a, b); }
static inline lane_float lane_mul(lane_float a, lane_float b) { return _mm256_mul_ps(a, b); }
static inline lane_float lane_div(lane_float a, lane_float b) { return a / b; }
static inline lane_float lane_fma(
    lane_float a, lane_float b, lane_float c) { return _mm256_fmadd_ps(a, b, c); }
static inline lane_mask lane_all_valid(void)
{
    return _mm256_castsi256_ps(_mm256_set1_epi32(-1));
}
static inline lane_mask lane_and(lane_mask a, lane_mask b) { return _mm256_and_ps(a, b); }
static inline lane_mask lane_finite(lane_float value)
{
    const lane_float absolute = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), value);
    return _mm256_cmp_ps(absolute, _mm256_set1_ps(INFINITY), _CMP_LT_OQ);
}
static inline lane_float lane_select_valid(lane_mask mask, lane_float value)
{
    return _mm256_blendv_ps(
        _mm256_set1_ps(3.402823466e+38F), value, mask);
}
static inline float lane_first(lane_float value) { return _mm_cvtss_f32(_mm256_castps256_ps128(value)); }

#else
#error "compile with ASTAXANTHIN_SCALAR or an AVX2/AVX-512 -march target"
#endif

static uint32_t next_random(uint32_t *state)
{
    uint32_t value = *state;
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    *state = value;
    return value;
}

static float uniform_random(uint32_t *state)
{
    return (float)(next_random(state) >> 8) * (1.0f / 16777216.0f);
}

static void build_reference(float *initial_state, float *targets)
{
    static const double observation_times[13] = {
        0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0,
        56.0, 64.0, 72.0, 80.0, 88.0, 96.0
    };
    static const double initial[STATE_COUNT][TRAJECTORY_COUNT] = {
        {0.1, 0.1, 0.2},
        {10.0, 20.0, 5.0},
        {5.0, 5.0, 2.5},
        {0.0, 0.0, 0.0}
    };

    for (int trajectory = 0; trajectory < TRAJECTORY_COUNT; ++trajectory) {
        for (int component = 0; component < STATE_COUNT; ++component) {
            initial_state[component * TRAJECTORY_COUNT + trajectory] =
                (float)initial[component][trajectory];
        }
        double state[STATE_COUNT][13];
        simulate_astaxanthin_ground_truth(
            observation_times, 13, 0.01,
            initial[0][trajectory], initial[1][trajectory],
            initial[2][trajectory], initial[3][trajectory],
            state[0], state[1], state[2], state[3]);
        for (int component = 0; component < STATE_COUNT; ++component) {
            for (int observation = 0; observation < OBSERVATION_COUNT; ++observation) {
                targets[component * TRAJECTORY_COUNT * OBSERVATION_COUNT +
                        trajectory * OBSERVATION_COUNT + observation] =
                    (float)state[component][observation + 1];
            }
        }
    }
}

static void score_trajectories(
    const float *restrict settings, unsigned long long num_settings,
    const float *restrict initial_state, const float *restrict targets,
    unsigned int steps_per_observation, float *restrict mse_out,
    float *restrict planted_final_state_out, unsigned int thread_count)
{
    const unsigned long long groups = num_settings / LANE_COUNT;
    const long long work_count = (long long)(TRAJECTORY_COUNT * groups);
    const lane_float mu_d = lane_set(0.0055f);
    const lane_float Y_S1 = lane_set(2.58f);
    const lane_float Y_S2 = lane_set(1.71f);
    const lane_float beta = lane_set(0.21f);
    const lane_float k_d = lane_set(0.0466f);
    const float scalar_h = 8.0f / (float)steps_per_observation;
    const lane_float h = lane_set(scalar_h);
    const lane_float half_h = lane_set(0.5f * scalar_h);
    const lane_float sixth_h = lane_set(scalar_h / 6.0f);

#pragma omp parallel for schedule(static) num_threads(thread_count)
    for (long long work = 0; work < work_count; ++work) {
        const int trajectory = (int)((unsigned long long)work / groups);
        const unsigned long long group = (unsigned long long)work % groups;
        const unsigned long long setting = group * LANE_COUNT;
        const lane_float mu_m1 = lane_load(settings + 0ull * num_settings + setting);
        const lane_float K_c1 = lane_load(settings + 1ull * num_settings + setting);
        const lane_float k_1 = lane_load(settings + 2ull * num_settings + setting);
        const lane_float mu_m2 = lane_load(settings + 3ull * num_settings + setting);
        const lane_float K_c2 = lane_load(settings + 4ull * num_settings + setting);
        const lane_float k_2 = lane_load(settings + 5ull * num_settings + setting);

        lane_float biomass = lane_set(initial_state[0 * TRAJECTORY_COUNT + trajectory]);
        lane_float glucose = lane_set(initial_state[1 * TRAJECTORY_COUNT + trajectory]);
        lane_float sucrose = lane_set(initial_state[2 * TRAJECTORY_COUNT + trajectory]);
        lane_float product = lane_set(initial_state[3 * TRAJECTORY_COUNT + trajectory]);
        lane_float squared_error = lane_set(0.0f);
        lane_mask valid = lane_all_valid();

        for (int observation = 0; observation < OBSERVATION_COUNT; ++observation) {
            for (unsigned int step = 0; step < steps_per_observation; ++step) {
                const lane_float base_biomass = biomass;
                const lane_float base_glucose = glucose;
                const lane_float base_sucrose = sucrose;
                const lane_float base_product = product;
                lane_float stage_biomass = biomass;
                lane_float stage_glucose = glucose;
                lane_float stage_sucrose = sucrose;
                lane_float sum_biomass = lane_set(0.0f);
                lane_float sum_glucose = lane_set(0.0f);
                lane_float sum_sucrose = lane_set(0.0f);
                lane_float sum_product = lane_set(0.0f);

#pragma GCC unroll 4
                for (int stage = 0; stage < 4; ++stage) {
                    const lane_float glucose_growth = lane_div(
                        lane_mul(lane_mul(mu_m1, stage_glucose), stage_biomass),
                        lane_mul(
                            lane_add(stage_glucose, lane_mul(K_c1, stage_biomass)),
                            lane_add(lane_set(1.0f), lane_mul(k_1, stage_sucrose))));
                    const lane_float sucrose_growth = lane_div(
                        lane_mul(lane_mul(mu_m2, stage_sucrose), stage_biomass),
                        lane_mul(
                            lane_add(stage_sucrose, lane_mul(K_c2, stage_biomass)),
                            lane_add(lane_set(1.0f), lane_mul(k_2, stage_glucose))));
                    const lane_float d_biomass = lane_sub(
                        lane_add(glucose_growth, sucrose_growth),
                        lane_mul(mu_d, stage_biomass));
                    const lane_float d_glucose = lane_mul(
                        lane_set(-1.0f), lane_mul(Y_S1, glucose_growth));
                    const lane_float d_sucrose = lane_mul(
                        lane_set(-1.0f), lane_mul(Y_S2, sucrose_growth));
                    const lane_float d_product = lane_sub(
                        lane_mul(beta, stage_biomass),
                        lane_mul(k_d, lane_mul(stage_biomass, stage_biomass)));
                    const lane_float weight =
                        lane_set((stage == 0 || stage == 3) ? 1.0f : 2.0f);

                    sum_biomass = lane_fma(weight, d_biomass, sum_biomass);
                    sum_glucose = lane_fma(weight, d_glucose, sum_glucose);
                    sum_sucrose = lane_fma(weight, d_sucrose, sum_sucrose);
                    sum_product = lane_fma(weight, d_product, sum_product);

                    if (stage != 3) {
                        const lane_float stage_h = stage == 2 ? h : half_h;
                        stage_biomass = lane_fma(stage_h, d_biomass, base_biomass);
                        stage_glucose = lane_fma(stage_h, d_glucose, base_glucose);
                        stage_sucrose = lane_fma(stage_h, d_sucrose, base_sucrose);
                    }
                }

                biomass = lane_fma(sixth_h, sum_biomass, base_biomass);
                glucose = lane_fma(sixth_h, sum_glucose, base_glucose);
                sucrose = lane_fma(sixth_h, sum_sucrose, base_sucrose);
                product = lane_fma(sixth_h, sum_product, base_product);
            }

            valid = lane_and(valid, lane_finite(biomass));
            valid = lane_and(valid, lane_finite(glucose));
            valid = lane_and(valid, lane_finite(sucrose));
            valid = lane_and(valid, lane_finite(product));
            const int target_base = trajectory * OBSERVATION_COUNT + observation;
            const lane_float error_biomass = lane_sub(
                biomass, lane_set(targets[0 * TRAJECTORY_COUNT * OBSERVATION_COUNT +
                                          target_base]));
            const lane_float error_glucose = lane_sub(
                glucose, lane_set(targets[1 * TRAJECTORY_COUNT * OBSERVATION_COUNT +
                                          target_base]));
            const lane_float error_sucrose = lane_sub(
                sucrose, lane_set(targets[2 * TRAJECTORY_COUNT * OBSERVATION_COUNT +
                                          target_base]));
            const lane_float error_product = lane_sub(
                product, lane_set(targets[3 * TRAJECTORY_COUNT * OBSERVATION_COUNT +
                                          target_base]));
            squared_error = lane_fma(error_biomass, error_biomass, squared_error);
            squared_error = lane_fma(error_glucose, error_glucose, squared_error);
            squared_error = lane_fma(error_sucrose, error_sucrose, squared_error);
            squared_error = lane_fma(error_product, error_product, squared_error);
        }

        lane_store(
            mse_out + (unsigned long long)trajectory * num_settings + setting,
            lane_select_valid(
                valid, lane_mul(squared_error, lane_set(1.0f / 48.0f))));
        if (group == 0ull) {
            planted_final_state_out[trajectory * STATE_COUNT + 0] = lane_first(biomass);
            planted_final_state_out[trajectory * STATE_COUNT + 1] = lane_first(glucose);
            planted_final_state_out[trajectory * STATE_COUNT + 2] = lane_first(sucrose);
            planted_final_state_out[trajectory * STATE_COUNT + 3] = lane_first(product);
        }
    }
}

static int verify_planted_setting(
    const float *mse, unsigned long long num_settings,
    const float *final_state, unsigned int steps_per_observation)
{
    static const double observation_times[2] = {0.0, 96.0};
    static const double initial[STATE_COUNT][TRAJECTORY_COUNT] = {
        {0.1, 0.1, 0.2},
        {10.0, 20.0, 5.0},
        {5.0, 5.0, 2.5},
        {0.0, 0.0, 0.0}
    };
    const double max_step = 8.0 / (double)steps_per_observation;
    double maximum_absolute_error = 0.0;
    double maximum_scaled_error = 0.0;
    int passed = 1;
    for (int trajectory = 0; trajectory < TRAJECTORY_COUNT; ++trajectory) {
        double state[STATE_COUNT][2];
        simulate_astaxanthin_ground_truth(
            observation_times, 2, max_step,
            initial[0][trajectory], initial[1][trajectory],
            initial[2][trajectory], initial[3][trajectory],
            state[0], state[1], state[2], state[3]);
        for (int component = 0; component < STATE_COUNT; ++component) {
            const double expected = state[component][1];
            const double actual = final_state[trajectory * STATE_COUNT + component];
            const double absolute_error = fabs(actual - expected);
            const double scaled_error = absolute_error / fmax(fabs(expected), 1.0);
            maximum_absolute_error = fmax(maximum_absolute_error, absolute_error);
            maximum_scaled_error = fmax(maximum_scaled_error, scaled_error);
        }
        passed = passed && isfinite(mse[(unsigned long long)trajectory * num_settings]);
    }
    passed = passed && maximum_scaled_error < 2.0e-4;
    printf("verification=%s max_final_abs_error=%.9g "
           "max_final_scaled_error=%.9g planted_mse=[%.9g,%.9g,%.9g]\n",
           passed ? "PASS" : "FAIL", maximum_absolute_error,
           maximum_scaled_error,
           mse[0], mse[num_settings], mse[2ull * num_settings]);
    return passed;
}

int main(int argc, char **argv)
{
    if (argc > 5) {
        fprintf(stderr,
                "usage: %s [settings] [threads] [steps_per_8h] [repeats]\n",
                argv[0]);
        return 2;
    }
    const unsigned long long num_settings =
        argc > 1 ? strtoull(argv[1], NULL, 10) : 262144ull;
    const unsigned int thread_count =
        argc > 2 ? (unsigned int)strtoul(argv[2], NULL, 10) : 12u;
    const unsigned int steps_per_observation =
        argc > 3 ? (unsigned int)strtoul(argv[3], NULL, 10) : 16u;
    const unsigned int repeats =
        argc > 4 ? (unsigned int)strtoul(argv[4], NULL, 10) : 3u;
    if (num_settings == 0ull || num_settings % LANE_COUNT != 0ull ||
        thread_count == 0u || steps_per_observation == 0u || repeats == 0u) {
        fprintf(stderr, "settings must be a positive multiple of %d; other dimensions must be positive\n",
                LANE_COUNT);
        return 2;
    }

    const size_t alignment = 64u;
    const size_t settings_bytes =
        PARAMETER_COUNT * (size_t)num_settings * sizeof(float);
    const size_t mse_bytes =
        TRAJECTORY_COUNT * (size_t)num_settings * sizeof(float);
    float *settings = NULL;
    float *mse = NULL;
    if (posix_memalign((void **)&settings, alignment, settings_bytes) != 0 ||
        posix_memalign((void **)&mse, alignment, mse_bytes) != 0) {
        fprintf(stderr, "aligned allocation failed\n");
        return 1;
    }

    uint32_t random_state = 0x6d2b79f5u;
    for (unsigned long long setting = 0; setting < num_settings; ++setting) {
        settings[0ull * num_settings + setting] =
            0.12f + 0.76f * uniform_random(&random_state);
        settings[1ull * num_settings + setting] =
            8.0f + 112.0f * uniform_random(&random_state);
        settings[2ull * num_settings + setting] =
            0.25f + 11.75f * uniform_random(&random_state);
        settings[3ull * num_settings + setting] =
            0.025f + 0.35f * uniform_random(&random_state);
        settings[4ull * num_settings + setting] =
            0.4f + 10.6f * uniform_random(&random_state);
        settings[5ull * num_settings + setting] =
            0.08f * uniform_random(&random_state);
    }
    const float truth[PARAMETER_COUNT] = {
        0.43f, 63.7f, 5.8f, 0.132f, 3.68f, 0.0f
    };
    for (int parameter = 0; parameter < PARAMETER_COUNT; ++parameter) {
        settings[(unsigned long long)parameter * num_settings] = truth[parameter];
    }

    float initial_state[STATE_COUNT * TRAJECTORY_COUNT];
    float targets[TARGET_COUNT];
    float planted_final_state[STATE_COUNT * TRAJECTORY_COUNT];
    build_reference(initial_state, targets);
    omp_set_dynamic(0);

    score_trajectories(
        settings, num_settings, initial_state, targets,
        steps_per_observation, mse, planted_final_state, thread_count);
    const double begin = omp_get_wtime();
    for (unsigned int repeat = 0; repeat < repeats; ++repeat) {
        score_trajectories(
            settings, num_settings, initial_state, targets,
            steps_per_observation, mse, planted_final_state, thread_count);
    }
    const double seconds = omp_get_wtime() - begin;

    double checksum = 0.0;
    for (size_t index = 0; index < mse_bytes / sizeof(float); ++index) {
        checksum += mse[index];
    }
    const double trajectory_count =
        (double)num_settings * TRAJECTORY_COUNT * repeats;
    const double trajectories_per_second = trajectory_count / seconds;
    const double rhs_per_trajectory =
        OBSERVATION_COUNT * (double)steps_per_observation * 4.0;
    printf("simd=%s lanes=%d omp_threads=%u settings=%llu "
           "trajectories_per_pass=%llu steps_per_8h=%u "
           "rk4_steps_per_trajectory=%u\n",
           SIMD_NAME, LANE_COUNT, thread_count, num_settings,
           num_settings * (unsigned long long)TRAJECTORY_COUNT,
           steps_per_observation, OBSERVATION_COUNT * steps_per_observation);
    printf("repeats=%u total_ms=%.6f per_pass_ms=%.6f "
           "trajectories_per_second=%.9g ns_per_trajectory=%.6f "
           "rhs_per_second=%.9g checksum=%.9g\n",
           repeats, seconds * 1.0e3, seconds * 1.0e3 / repeats,
           trajectories_per_second, 1.0e9 / trajectories_per_second,
           trajectories_per_second * rhs_per_trajectory, checksum);
    const int passed = verify_planted_setting(
        mse, num_settings, planted_final_state, steps_per_observation);
    free(settings);
    free(mse);
    return passed ? 0 : 3;
}
