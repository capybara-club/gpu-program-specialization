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
#include "astaxanthin_ground_truth.h"

#include <cuda.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call) do {                                                   \
    const CUresult cuda_result_ = (call);                                       \
    if (cuda_result_ != CUDA_SUCCESS) {                                         \
        const char *cuda_name_ = NULL;                                          \
        const char *cuda_message_ = NULL;                                       \
        (void)cuGetErrorName(cuda_result_, &cuda_name_);                        \
        (void)cuGetErrorString(cuda_result_, &cuda_message_);                   \
        fprintf(stderr, "%s failed: %s (%s)\n", #call,                       \
                cuda_name_ != NULL ? cuda_name_ : "unknown",                  \
                cuda_message_ != NULL ? cuda_message_ : "no description");    \
        exit(1);                                                               \
    }                                                                          \
} while (0)

enum {
    ASTAXANTHIN_PARAMETER_COUNT = 6,
    ASTAXANTHIN_STATE_COUNT = 4,
    ASTAXANTHIN_TRAJECTORY_COUNT = 3,
    ASTAXANTHIN_OBSERVATION_COUNT = 12,
    ASTAXANTHIN_TARGET_COUNT =
        ASTAXANTHIN_STATE_COUNT * ASTAXANTHIN_TRAJECTORY_COUNT *
        ASTAXANTHIN_OBSERVATION_COUNT
};

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
    static const double initial[ASTAXANTHIN_STATE_COUNT]
                               [ASTAXANTHIN_TRAJECTORY_COUNT] = {
        {0.1, 0.1, 0.2},
        {10.0, 20.0, 5.0},
        {5.0, 5.0, 2.5},
        {0.0, 0.0, 0.0}
    };

    for (int trajectory = 0;
         trajectory < ASTAXANTHIN_TRAJECTORY_COUNT;
         ++trajectory) {
        for (int component = 0; component < ASTAXANTHIN_STATE_COUNT; ++component) {
            initial_state[component * ASTAXANTHIN_TRAJECTORY_COUNT + trajectory] =
                (float)initial[component][trajectory];
        }

        double state[ASTAXANTHIN_STATE_COUNT][13];
        simulate_astaxanthin_ground_truth(
            observation_times, 13, 0.01,
            initial[0][trajectory], initial[1][trajectory],
            initial[2][trajectory], initial[3][trajectory],
            state[0], state[1], state[2], state[3]);
        for (int component = 0; component < ASTAXANTHIN_STATE_COUNT; ++component) {
            for (int observation = 0;
                 observation < ASTAXANTHIN_OBSERVATION_COUNT;
                 ++observation) {
                targets[
                    component * ASTAXANTHIN_TRAJECTORY_COUNT *
                        ASTAXANTHIN_OBSERVATION_COUNT +
                    trajectory * ASTAXANTHIN_OBSERVATION_COUNT + observation] =
                    (float)state[component][observation + 1];
            }
        }
    }
}

static int verify_planted_setting(
    const float *mse, const float *final_state,
    unsigned int steps_per_observation)
{
    static const double observation_times[2] = {0.0, 96.0};
    static const double initial[ASTAXANTHIN_STATE_COUNT]
                               [ASTAXANTHIN_TRAJECTORY_COUNT] = {
        {0.1, 0.1, 0.2},
        {10.0, 20.0, 5.0},
        {5.0, 5.0, 2.5},
        {0.0, 0.0, 0.0}
    };
    const double max_step = 8.0 / (double)steps_per_observation;
    double maximum_absolute_error = 0.0;
    double maximum_scaled_error = 0.0;
    int passed = 1;

    printf("planted_truth_verification:\n");
    printf("trajectory  mse_vs_0.01h_targets  max_final_abs_error  "
           "max_final_scaled_error\n");
    for (int trajectory = 0;
         trajectory < ASTAXANTHIN_TRAJECTORY_COUNT;
         ++trajectory) {
        double state[ASTAXANTHIN_STATE_COUNT][2];
        simulate_astaxanthin_ground_truth(
            observation_times, 2, max_step,
            initial[0][trajectory], initial[1][trajectory],
            initial[2][trajectory], initial[3][trajectory],
            state[0], state[1], state[2], state[3]);
        double trajectory_absolute_error = 0.0;
        double trajectory_scaled_error = 0.0;
        for (int component = 0; component < ASTAXANTHIN_STATE_COUNT; ++component) {
            const double expected = state[component][1];
            const double actual = final_state[
                trajectory * ASTAXANTHIN_STATE_COUNT + component];
            const double absolute_error = fabs(actual - expected);
            const double scaled_error = absolute_error / fmax(fabs(expected), 1.0);
            trajectory_absolute_error = fmax(
                trajectory_absolute_error, absolute_error);
            trajectory_scaled_error = fmax(
                trajectory_scaled_error, scaled_error);
        }
        maximum_absolute_error = fmax(
            maximum_absolute_error, trajectory_absolute_error);
        maximum_scaled_error = fmax(
            maximum_scaled_error, trajectory_scaled_error);
        passed = passed && isfinite(mse[trajectory]) &&
                 trajectory_scaled_error < 2.0e-4;
        printf("%10d  %20.9g  %19.9g  %22.9g\n",
               trajectory + 1, mse[trajectory], trajectory_absolute_error,
               trajectory_scaled_error);
    }
    printf("verification=%s max_final_abs_error=%.9g "
           "max_final_scaled_error=%.9g\n",
           passed ? "PASS" : "FAIL", maximum_absolute_error,
           maximum_scaled_error);
    return passed;
}

int main(int argc, char **argv)
{
    if (argc < 2 || argc > 6) {
        fprintf(stderr,
                "usage: %s kernel.cubin [settings] [threads] "
                "[steps_per_8h] [repeats]\n",
                argv[0]);
        return 2;
    }

    const unsigned long long num_settings =
        argc > 2 ? strtoull(argv[2], NULL, 10) : 65536ull;
    const unsigned int threads =
        argc > 3 ? (unsigned int)strtoul(argv[3], NULL, 10) : 128u;
    const unsigned int steps_per_observation =
        argc > 4 ? (unsigned int)strtoul(argv[4], NULL, 10) : 16u;
    const unsigned int repeats =
        argc > 5 ? (unsigned int)strtoul(argv[5], NULL, 10) : 5u;
    if (num_settings == 0ull || num_settings > 0xffffffffull ||
        threads == 0u || threads > 1024u ||
        steps_per_observation == 0u || repeats == 0u) {
        fprintf(stderr, "invalid benchmark dimensions\n");
        return 2;
    }

    const size_t settings_count =
        ASTAXANTHIN_PARAMETER_COUNT * (size_t)num_settings;
    float *settings = (float *)malloc(settings_count * sizeof(float));
    if (settings == NULL) {
        fprintf(stderr, "host settings allocation failed\n");
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
    const float truth[ASTAXANTHIN_PARAMETER_COUNT] = {
        0.43f, 63.7f, 5.8f, 0.132f, 3.68f, 0.0f
    };
    for (int parameter = 0; parameter < ASTAXANTHIN_PARAMETER_COUNT; ++parameter) {
        settings[(unsigned long long)parameter * num_settings] = truth[parameter];
    }

    float initial_state[
        ASTAXANTHIN_STATE_COUNT * ASTAXANTHIN_TRAJECTORY_COUNT];
    float targets[ASTAXANTHIN_TARGET_COUNT];
    build_reference(initial_state, targets);

    CUDA_CHECK(cuInit(0));
    CUdevice device;
    CUcontext context;
    CUmodule module;
    CUfunction function;
    CUDA_CHECK(cuDeviceGet(&device, 0));
    CUDA_CHECK(cuCtxCreate(&context, NULL, 0, device));
    CUDA_CHECK(cuModuleLoad(&module, argv[1]));
    int dynamic_leaves = 0;
    CUresult function_result = cuModuleGetFunction(
        &function, module, "astaxanthin_score_dynamic_leaves_rk4");
    if (function_result == CUDA_SUCCESS) {
        dynamic_leaves = 1;
    } else {
        CUDA_CHECK(cuModuleGetFunction(
            &function, module, "astaxanthin_score_only_rk4"));
    }

    CUdeviceptr device_initial_state;
    CUdeviceptr device_targets;
    size_t device_initial_state_bytes = 0u;
    size_t device_target_bytes = 0u;
    CUDA_CHECK(cuModuleGetGlobal(
        &device_initial_state, &device_initial_state_bytes,
        module, "astaxanthin_initial_state"));
    CUDA_CHECK(cuModuleGetGlobal(
        &device_targets, &device_target_bytes,
        module, "astaxanthin_targets"));
    if (device_initial_state_bytes != sizeof(initial_state) ||
        device_target_bytes != sizeof(targets)) {
        fprintf(stderr, "CUBIN constant-memory ABI mismatch\n");
        return 1;
    }
    CUDA_CHECK(cuMemcpyHtoD(
        device_initial_state, initial_state, sizeof(initial_state)));
    CUDA_CHECK(cuMemcpyHtoD(device_targets, targets, sizeof(targets)));

    CUdeviceptr device_settings;
    CUdeviceptr device_bindings = 0;
    CUdeviceptr device_mse;
    CUdeviceptr device_final_state;
    const size_t mse_count =
        ASTAXANTHIN_TRAJECTORY_COUNT * (size_t)num_settings;
    CUDA_CHECK(cuMemAlloc(
        &device_settings, settings_count * sizeof(float)));
    CUDA_CHECK(cuMemAlloc(&device_mse, mse_count * sizeof(float)));
    CUDA_CHECK(cuMemAlloc(
        &device_final_state,
        ASTAXANTHIN_STATE_COUNT * ASTAXANTHIN_TRAJECTORY_COUNT * sizeof(float)));
    CUDA_CHECK(cuMemcpyHtoD(
        device_settings, settings, settings_count * sizeof(float)));

    unsigned int *bindings = NULL;
    if (dynamic_leaves) {
        const size_t binding_count = 8u * (size_t)num_settings;
        bindings = (unsigned int *)malloc(binding_count * sizeof(unsigned int));
        if (bindings == NULL) {
            fprintf(stderr, "host binding allocation failed\n");
            return 1;
        }
        for (unsigned long long setting = 0; setting < num_settings; ++setting) {
            for (int leaf = 0; leaf < 8; ++leaf) {
                /* P is safe in numerator and inhibition leaves. Keep the two
                 * additive-denominator inputs on X/S1/S2 so their initial
                 * zero cannot manufacture a singular denominator. */
                const unsigned int leaf_value_count =
                    (leaf == 0 || leaf == 3 || leaf == 4 || leaf == 7) ? 4u : 3u;
                bindings[(unsigned long long)leaf * num_settings + setting] =
                    next_random(&random_state) % leaf_value_count;
            }
        }
        const unsigned int correct_bindings[8] = {
            1u, 1u, 0u, 2u,
            2u, 2u, 0u, 1u
        };
        for (int leaf = 0; leaf < 8; ++leaf) {
            bindings[(unsigned long long)leaf * num_settings] = correct_bindings[leaf];
        }
        CUDA_CHECK(cuMemAlloc(
            &device_bindings, binding_count * sizeof(unsigned int)));
        CUDA_CHECK(cuMemcpyHtoD(
            device_bindings, bindings,
            binding_count * sizeof(unsigned int)));
    }

    void *direct_arguments[] = {
        &device_settings,
        (void *)&num_settings,
        (void *)&num_settings,
        (void *)&steps_per_observation,
        &device_mse,
        &device_final_state
    };
    void *dynamic_arguments[] = {
        &device_settings,
        (void *)&num_settings,
        &device_bindings,
        (void *)&num_settings,
        (void *)&num_settings,
        (void *)&steps_per_observation,
        &device_mse,
        &device_final_state
    };
    void **arguments = dynamic_leaves ? dynamic_arguments : direct_arguments;
    const unsigned int dynamic_shared_bytes =
        dynamic_leaves ? ASTAXANTHIN_STATE_COUNT * threads * sizeof(float) : 0u;
    const unsigned int grid_x =
        (unsigned int)((num_settings + threads - 1u) / threads);
    for (int warmup = 0; warmup < 2; ++warmup) {
        CUDA_CHECK(cuLaunchKernel(
            function,
            grid_x, ASTAXANTHIN_TRAJECTORY_COUNT, 1u,
            threads, 1u, 1u,
            dynamic_shared_bytes, 0, arguments, NULL));
    }
    CUDA_CHECK(cuCtxSynchronize());

    CUevent begin;
    CUevent end;
    CUDA_CHECK(cuEventCreate(&begin, CU_EVENT_DEFAULT));
    CUDA_CHECK(cuEventCreate(&end, CU_EVENT_DEFAULT));
    CUDA_CHECK(cuEventRecord(begin, 0));
    for (unsigned int repeat = 0; repeat < repeats; ++repeat) {
        CUDA_CHECK(cuLaunchKernel(
            function,
            grid_x, ASTAXANTHIN_TRAJECTORY_COUNT, 1u,
            threads, 1u, 1u,
            dynamic_shared_bytes, 0, arguments, NULL));
    }
    CUDA_CHECK(cuEventRecord(end, 0));
    CUDA_CHECK(cuEventSynchronize(end));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cuEventElapsedTime(&elapsed_ms, begin, end));

    float planted_mse[ASTAXANTHIN_TRAJECTORY_COUNT];
    for (int trajectory = 0;
         trajectory < ASTAXANTHIN_TRAJECTORY_COUNT;
         ++trajectory) {
        CUDA_CHECK(cuMemcpyDtoH(
            &planted_mse[trajectory],
            device_mse + (CUdeviceptr)((size_t)trajectory * num_settings *
                                       sizeof(float)),
            sizeof(float)));
    }
    float planted_final_state[
        ASTAXANTHIN_STATE_COUNT * ASTAXANTHIN_TRAJECTORY_COUNT];
    CUDA_CHECK(cuMemcpyDtoH(
        planted_final_state, device_final_state, sizeof(planted_final_state)));

    int registers = 0;
    int static_shared_bytes = 0;
    int local_bytes = 0;
    int max_threads = 0;
    CUDA_CHECK(cuFuncGetAttribute(
        &registers, CU_FUNC_ATTRIBUTE_NUM_REGS, function));
    CUDA_CHECK(cuFuncGetAttribute(
        &static_shared_bytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, function));
    CUDA_CHECK(cuFuncGetAttribute(
        &local_bytes, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, function));
    CUDA_CHECK(cuFuncGetAttribute(
        &max_threads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, function));

    char device_name[256];
    int compute_capability_major = 0;
    int compute_capability_minor = 0;
    CUDA_CHECK(cuDeviceGetName(device_name, sizeof(device_name), device));
    CUDA_CHECK(cuDeviceGetAttribute(
        &compute_capability_major,
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    CUDA_CHECK(cuDeviceGetAttribute(
        &compute_capability_minor,
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));

    const double trajectories =
        (double)num_settings * ASTAXANTHIN_TRAJECTORY_COUNT * repeats;
    const double seconds = (double)elapsed_ms * 1.0e-3;
    const double trajectories_per_second = trajectories / seconds;
    const double rhs_per_trajectory =
        ASTAXANTHIN_OBSERVATION_COUNT * (double)steps_per_observation * 4.0;
    printf("kernel=%s device=%s cc=%d.%d settings=%llu trajectories_per_launch=%llu ",
           dynamic_leaves ? "dynamic_leaves_smem" : "direct",
           device_name, compute_capability_major, compute_capability_minor,
           num_settings,
           num_settings * (unsigned long long)ASTAXANTHIN_TRAJECTORY_COUNT);
    printf("threads=%u ctas=%u steps_per_8h=%u rk4_steps_per_trajectory=%u\n",
           threads, grid_x * ASTAXANTHIN_TRAJECTORY_COUNT,
           steps_per_observation,
           ASTAXANTHIN_OBSERVATION_COUNT * steps_per_observation);
    printf("registers_per_thread=%d static_shared_bytes=%d "
           "dynamic_shared_bytes=%u local_bytes_per_thread=%d "
           "max_threads_per_block=%d\n",
           registers, static_shared_bytes, dynamic_shared_bytes,
           local_bytes, max_threads);
    printf("repeats=%u total_ms=%.6f per_launch_ms=%.6f ",
           repeats, elapsed_ms, elapsed_ms / (float)repeats);
    printf("trajectories_per_second=%.9g ns_per_trajectory=%.6f "
           "rhs_per_second=%.9g\n",
           trajectories_per_second, 1.0e9 / trajectories_per_second,
           trajectories_per_second * rhs_per_trajectory);

    const int verification_passed = verify_planted_setting(
        planted_mse, planted_final_state, steps_per_observation);

    CUDA_CHECK(cuEventDestroy(begin));
    CUDA_CHECK(cuEventDestroy(end));
    CUDA_CHECK(cuMemFree(device_settings));
    if (dynamic_leaves) CUDA_CHECK(cuMemFree(device_bindings));
    CUDA_CHECK(cuMemFree(device_mse));
    CUDA_CHECK(cuMemFree(device_final_state));
    CUDA_CHECK(cuModuleUnload(module));
    CUDA_CHECK(cuCtxDestroy(context));
    free(settings);
    free(bindings);
    return verification_passed ? 0 : 3;
}

#undef CUDA_CHECK
