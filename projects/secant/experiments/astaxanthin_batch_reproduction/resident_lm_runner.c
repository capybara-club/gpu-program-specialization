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

#define CUDA_CHECK(call) do { \
    CUresult cuda_result_ = (call); \
    if (cuda_result_ != CUDA_SUCCESS) { \
        const char *cuda_name_ = NULL; \
        const char *cuda_message_ = NULL; \
        (void)cuGetErrorName(cuda_result_, &cuda_name_); \
        (void)cuGetErrorString(cuda_result_, &cuda_message_); \
        fprintf(stderr, "%s failed: %s (%s)\n", #call, \
                cuda_name_ != NULL ? cuda_name_ : "unknown", \
                cuda_message_ != NULL ? cuda_message_ : "no description"); \
        return 1; \
    } \
} while (0)

int main(int argc, char **argv)
{
    if (argc < 2 || argc > 8) {
        fprintf(stderr,
                "usage: %s specialized.cubin [settings] [threads] [steps/8h] "
                "[lm_iterations] [damping_attempts] [repeats]\n",
                argv[0]);
        return 2;
    }

    const unsigned int num_settings = argc > 2 ? (unsigned int)strtoul(argv[2], NULL, 10) : 128u;
    const unsigned int threads = argc > 3 ? (unsigned int)strtoul(argv[3], NULL, 10) : 64u;
    const unsigned int steps_per_observation =
        argc > 4 ? (unsigned int)strtoul(argv[4], NULL, 10) : 16u;
    const unsigned int max_lm_iterations =
        argc > 5 ? (unsigned int)strtoul(argv[5], NULL, 10) : 20u;
    const unsigned int max_damping_attempts =
        argc > 6 ? (unsigned int)strtoul(argv[6], NULL, 10) : 8u;
    const unsigned int repeats = argc > 7 ? (unsigned int)strtoul(argv[7], NULL, 10) : 3u;
    const float initial_damping = 1.0e-3f;

    if (num_settings == 0u || threads == 0u || threads > 1024u ||
        steps_per_observation == 0u || max_lm_iterations == 0u ||
        max_damping_attempts == 0u || repeats == 0u) {
        fprintf(stderr, "all dimensions must be positive and threads must be <= 1024\n");
        return 2;
    }

    const unsigned long long starts_leading_dimension = num_settings;
    const unsigned long long constants_leading_dimension = num_settings;
    float *starts = (float *)malloc((size_t)8u * num_settings * sizeof(float));
    float *constants = (float *)malloc((size_t)6u * num_settings * sizeof(float));
    float *mse = (float *)malloc((size_t)num_settings * sizeof(float));
    unsigned int *iterations =
        (unsigned int *)malloc((size_t)num_settings * sizeof(unsigned int));
    unsigned int *accepted_steps =
        (unsigned int *)malloc((size_t)num_settings * sizeof(unsigned int));
    if (starts == NULL || constants == NULL || mse == NULL ||
        iterations == NULL || accepted_steps == NULL) {
        fprintf(stderr, "host allocation failed\n");
        return 1;
    }

    uint32_t random_state = 0x6d2b79f5u;
    for (unsigned int setting = 0u; setting < num_settings; ++setting) {
        float random_values[6];
        for (int parameter = 0; parameter < 6; ++parameter) {
            random_state ^= random_state << 13;
            random_state ^= random_state >> 17;
            random_state ^= random_state << 5;
            random_values[parameter] =
                (float)(random_state >> 8) * (1.0f / 16777216.0f);
        }
        starts[0u * num_settings + setting] = 0.12f + 0.76f * random_values[0];
        starts[1u * num_settings + setting] = 8.0f + 112.0f * random_values[1];
        starts[2u * num_settings + setting] = 0.25f + 11.75f * random_values[2];
        starts[3u * num_settings + setting] = 0.025f + 0.35f * random_values[3];
        starts[4u * num_settings + setting] = 0.4f + 10.6f * random_values[4];
        starts[5u * num_settings + setting] = 0.08f * random_values[5];
        starts[6u * num_settings + setting] = 0.0f;
        starts[7u * num_settings + setting] = 0.0f;
    }

    const double observation_times[13] = {
        0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0,
        56.0, 64.0, 72.0, 80.0, 88.0, 96.0
    };
    const double initial_biomass[3] = {0.1, 0.1, 0.2};
    const double initial_glucose[3] = {10.0, 20.0, 5.0};
    const double initial_sucrose[3] = {5.0, 5.0, 2.5};
    const double initial_astaxanthin[3] = {0.0, 0.0, 0.0};
    float reference[156];

    for (int experiment = 0; experiment < 3; ++experiment) {
        reference[0 * 3 + experiment] = (float)initial_biomass[experiment];
        reference[1 * 3 + experiment] = (float)initial_glucose[experiment];
        reference[2 * 3 + experiment] = (float)initial_sucrose[experiment];
        reference[3 * 3 + experiment] = (float)initial_astaxanthin[experiment];

        double state[4][13];
        simulate_astaxanthin_ground_truth(
            observation_times, 13, 0.01,
            initial_biomass[experiment], initial_glucose[experiment],
            initial_sucrose[experiment], initial_astaxanthin[experiment],
            state[0], state[1], state[2], state[3]);
        for (int component = 0; component < 4; ++component) {
            for (int observation = 0; observation < 12; ++observation) {
                reference[12 + component * 36 + experiment * 12 + observation] =
                    (float)state[component][observation + 1];
            }
        }
    }

    CUDA_CHECK(cuInit(0));
    CUdevice device;
    CUcontext context;
    CUmodule module;
    CUfunction function;
    CUDA_CHECK(cuDeviceGet(&device, 0));
    CUDA_CHECK(cuCtxCreate(&context, NULL, 0, device));
    CUDA_CHECK(cuModuleLoad(&module, argv[1]));
    CUDA_CHECK(cuModuleGetFunction(&function, module, "secant_cubin_materialize_000"));

    CUdeviceptr device_starts;
    CUdeviceptr device_reference;
    CUdeviceptr device_constants;
    CUdeviceptr device_mse;
    CUdeviceptr device_iterations;
    CUdeviceptr device_accepted_steps;
    CUDA_CHECK(cuMemAlloc(&device_starts, (size_t)8u * num_settings * sizeof(float)));
    CUDA_CHECK(cuMemAlloc(&device_reference, sizeof(reference)));
    CUDA_CHECK(cuMemAlloc(&device_constants, (size_t)6u * num_settings * sizeof(float)));
    CUDA_CHECK(cuMemAlloc(&device_mse, (size_t)num_settings * sizeof(float)));
    CUDA_CHECK(cuMemAlloc(&device_iterations, (size_t)num_settings * sizeof(unsigned int)));
    CUDA_CHECK(cuMemAlloc(&device_accepted_steps, (size_t)num_settings * sizeof(unsigned int)));
    CUDA_CHECK(cuMemcpyHtoD(
        device_starts, starts, (size_t)8u * num_settings * sizeof(float)));
    CUDA_CHECK(cuMemcpyHtoD(device_reference, reference, sizeof(reference)));

    void *arguments[] = {
        &device_starts,
        (void *)&starts_leading_dimension,
        (void *)&starts_leading_dimension,
        &device_reference,
        (void *)&steps_per_observation,
        (void *)&max_lm_iterations,
        (void *)&max_damping_attempts,
        (void *)&initial_damping,
        &device_constants,
        (void *)&constants_leading_dimension,
        &device_mse,
        &device_iterations,
        &device_accepted_steps
    };
    unsigned long long kernel_num_settings = num_settings;
    arguments[2] = &kernel_num_settings;

    CUevent begin;
    CUevent end;
    CUDA_CHECK(cuEventCreate(&begin, CU_EVENT_DEFAULT));
    CUDA_CHECK(cuEventCreate(&end, CU_EVENT_DEFAULT));
    CUDA_CHECK(cuEventRecord(begin, 0));
    for (unsigned int repeat = 0u; repeat < repeats; ++repeat) {
        CUDA_CHECK(cuLaunchKernel(
            function,
            1u, (num_settings + threads - 1u) / threads, 1u,
            threads, 1u, 1u,
            0u, 0, arguments, NULL));
    }
    CUDA_CHECK(cuEventRecord(end, 0));
    CUDA_CHECK(cuEventSynchronize(end));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cuEventElapsedTime(&elapsed_ms, begin, end));

    CUDA_CHECK(cuMemcpyDtoH(
        constants, device_constants, (size_t)6u * num_settings * sizeof(float)));
    CUDA_CHECK(cuMemcpyDtoH(mse, device_mse, (size_t)num_settings * sizeof(float)));
    CUDA_CHECK(cuMemcpyDtoH(
        iterations, device_iterations, (size_t)num_settings * sizeof(unsigned int)));
    CUDA_CHECK(cuMemcpyDtoH(
        accepted_steps, device_accepted_steps,
        (size_t)num_settings * sizeof(unsigned int)));

    unsigned int best = 0u;
    unsigned int finite_count = 0u;
    unsigned int recovered_count = 0u;
    const float truth[6] = {0.43f, 63.7f, 5.8f, 0.132f, 3.68f, 0.0f};
    for (unsigned int setting = 0u; setting < num_settings; ++setting) {
        if (isfinite(mse[setting])) ++finite_count;
        if (mse[setting] < mse[best]) best = setting;
        float maximum_scaled_error = 0.0f;
        for (int parameter = 0; parameter < 6; ++parameter) {
            const float scale = parameter == 5 ? 0.01f : fmaxf(fabsf(truth[parameter]), 1.0e-6f);
            const float error = fabsf(constants[parameter * num_settings + setting] - truth[parameter]) / scale;
            if (error > maximum_scaled_error) maximum_scaled_error = error;
        }
        if (maximum_scaled_error <= 0.01f) ++recovered_count;
    }

    char device_name[256];
    CUDA_CHECK(cuDeviceGetName(device_name, sizeof(device_name), device));
    printf("device=%s settings=%u threads=%u ctas=%u steps_per_observation=%u "
           "lm_iterations=%u damping_attempts=%u\n",
           device_name, num_settings, threads,
           (num_settings + threads - 1u) / threads,
           steps_per_observation, max_lm_iterations, max_damping_attempts);
    printf("kernel_ms=%.6f per_launch_ms=%.6f finite=%u recovered_1pct=%u\n",
           elapsed_ms, elapsed_ms / (float)repeats, finite_count, recovered_count);
    printf("best_setting=%u weighted_mse=%.9g iterations=%u accepted=%u\n",
           best, mse[best], iterations[best], accepted_steps[best]);
    printf("parameter       start         fitted        truth          abs_error\n");
    const char *names[6] = {"mu_m1", "K_c1", "k_1", "mu_m2", "K_c2", "k_2"};
    for (int parameter = 0; parameter < 6; ++parameter) {
        const float start = starts[parameter * num_settings + best];
        const float fitted = constants[parameter * num_settings + best];
        printf("%-9s %12.7g  %12.7g  %12.7g  %12.7g\n",
               names[parameter], start, fitted, truth[parameter], fabsf(fitted - truth[parameter]));
    }

    float best_maximum_scaled_error = 0.0f;
    for (int parameter = 0; parameter < 6; ++parameter) {
        const float scale = parameter == 5 ? 0.01f : fmaxf(fabsf(truth[parameter]), 1.0e-6f);
        const float error = fabsf(constants[parameter * num_settings + best] - truth[parameter]) / scale;
        if (error > best_maximum_scaled_error) best_maximum_scaled_error = error;
    }
    const int recovery_passed = isfinite(mse[best]) && mse[best] < 1.0e-8f &&
        best_maximum_scaled_error < 0.01f;
    printf("recovery_test=%s maximum_scaled_parameter_error=%.9g\n",
           recovery_passed ? "PASS" : "FAIL", best_maximum_scaled_error);

    CUDA_CHECK(cuEventDestroy(begin));
    CUDA_CHECK(cuEventDestroy(end));
    CUDA_CHECK(cuMemFree(device_starts));
    CUDA_CHECK(cuMemFree(device_reference));
    CUDA_CHECK(cuMemFree(device_constants));
    CUDA_CHECK(cuMemFree(device_mse));
    CUDA_CHECK(cuMemFree(device_iterations));
    CUDA_CHECK(cuMemFree(device_accepted_steps));
    CUDA_CHECK(cuModuleUnload(module));
    CUDA_CHECK(cuCtxDestroy(context));
    free(starts);
    free(constants);
    free(mse);
    free(iterations);
    free(accepted_steps);
    return recovery_passed ? 0 : 3;
}

#undef CUDA_CHECK
