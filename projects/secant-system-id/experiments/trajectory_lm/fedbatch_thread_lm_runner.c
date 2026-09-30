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

#include <limits.h>
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
    if (argc < 2 || argc > 9) {
        fprintf(stderr,
                "usage: %s specialized.cubin [settings] [starts/setting] "
                "[threads] [steps/8h] [lm_iterations] [damping_attempts] "
                "[repeats]\n",
                argv[0]);
        return 2;
    }

    const unsigned int num_settings =
        argc > 2 ? (unsigned int)strtoul(argv[2], NULL, 10) : 4096u;
    const unsigned int starts_per_setting =
        argc > 3 ? (unsigned int)strtoul(argv[3], NULL, 10) : 4u;
    const unsigned int threads = argc > 4 ? (unsigned int)strtoul(argv[4], NULL, 10) : 128u;
    const unsigned int steps_per_observation =
        argc > 5 ? (unsigned int)strtoul(argv[5], NULL, 10) : 16u;
    const unsigned int max_lm_iterations =
        argc > 6 ? (unsigned int)strtoul(argv[6], NULL, 10) : 20u;
    const unsigned int max_damping_attempts =
        argc > 7 ? (unsigned int)strtoul(argv[7], NULL, 10) : 8u;
    const unsigned int repeats = argc > 8 ? (unsigned int)strtoul(argv[8], NULL, 10) : 3u;
    const float initial_damping = 1.0e-3f;
    const uint64_t num_fits_64 = (uint64_t)num_settings * starts_per_setting;

    if (num_settings == 0u || starts_per_setting == 0u ||
        num_fits_64 > UINT_MAX || threads == 0u || threads > 1024u ||
        steps_per_observation == 0u || max_lm_iterations == 0u ||
        max_damping_attempts == 0u || repeats == 0u) {
        fprintf(stderr, "dimensions must be positive, fits must fit in u32, and threads must be <= 1024\n");
        return 2;
    }
    const unsigned int num_fits = (unsigned int)num_fits_64;

    const unsigned long long starts_leading_dimension = num_fits;
    const unsigned long long bindings_leading_dimension = num_settings;
    const unsigned long long constants_leading_dimension = num_fits;
    float *starts = (float *)malloc((size_t)8u * num_fits * sizeof(float));
    unsigned int *bindings =
        (unsigned int *)malloc((size_t)16u * num_settings * sizeof(unsigned int));
    float *constants = (float *)malloc((size_t)6u * num_fits * sizeof(float));
    float *mse = (float *)malloc((size_t)num_fits * sizeof(float));
    unsigned int *iterations =
        (unsigned int *)malloc((size_t)num_fits * sizeof(unsigned int));
    unsigned int *accepted_steps =
        (unsigned int *)malloc((size_t)num_fits * sizeof(unsigned int));
    if (starts == NULL || bindings == NULL || constants == NULL || mse == NULL ||
        iterations == NULL || accepted_steps == NULL) {
        fprintf(stderr, "host allocation failed\n");
        return 1;
    }

    uint32_t random_state = 0x6d2b79f5u;
    for (unsigned int fit = 0u; fit < num_fits; ++fit) {
        float random_values[6];
        for (int parameter = 0; parameter < 6; ++parameter) {
            random_state ^= random_state << 13;
            random_state ^= random_state >> 17;
            random_state ^= random_state << 5;
            random_values[parameter] =
                (float)(random_state >> 8) * (1.0f / 16777216.0f);
        }
        starts[0u * num_fits + fit] = 0.12f + 0.76f * random_values[0];
        starts[1u * num_fits + fit] = 8.0f + 112.0f * random_values[1];
        starts[2u * num_fits + fit] = 0.25f + 11.75f * random_values[2];
        starts[3u * num_fits + fit] = 0.025f + 0.35f * random_values[3];
        starts[4u * num_fits + fit] = 0.4f + 10.6f * random_values[4];
        starts[5u * num_fits + fit] = 0.08f * random_values[5];
        starts[6u * num_fits + fit] = 1.0f;
        starts[7u * num_fits + fit] = 0.0f;
    }

    const unsigned int planted_bindings[16] = {
        4u, 1u, 1u, 5u, 0u, 10u, 6u, 2u,
        7u, 2u, 2u, 8u, 0u, 10u, 9u, 1u
    };
    uint32_t binding_random_state = 0x9e3779b9u;
    unsigned int exact_binding_settings = 0u;
    for (unsigned int setting = 0u; setting < num_settings; ++setting) {
        unsigned int exact = 1u;
        for (unsigned int leaf = 0u; leaf < 16u; ++leaf) {
            unsigned int binding = planted_bindings[leaf];
            if (setting != 0u) {
                binding_random_state ^= binding_random_state << 13;
                binding_random_state ^= binding_random_state >> 17;
                binding_random_state ^= binding_random_state << 5;
                const float keep =
                    (float)(binding_random_state >> 8) * (1.0f / 16777216.0f);
                if (keep >= 0.9f) {
                    binding_random_state ^= binding_random_state << 13;
                    binding_random_state ^= binding_random_state >> 17;
                    binding_random_state ^= binding_random_state << 5;
                    binding = binding_random_state % 12u;
                }
            }
            bindings[leaf * num_settings + setting] = binding;
            exact = exact && binding == planted_bindings[leaf];
        }
        exact_binding_settings += exact;
    }

    const double observation_times[13] = {
        0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0,
        56.0, 64.0, 72.0, 80.0, 88.0, 96.0
    };
    const double initial_biomass[16] = {
        0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08,
        0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24
    };
    const double initial_glucose[16] = {
        4.0, 4.0, 4.0, 4.0, 22.0, 22.0, 22.0, 22.0,
        4.0, 4.0, 4.0, 4.0, 22.0, 22.0, 22.0, 22.0
    };
    const double initial_sucrose[16] = {
        1.0, 1.0, 8.0, 8.0, 1.0, 1.0, 8.0, 8.0,
        1.0, 1.0, 8.0, 8.0, 1.0, 1.0, 8.0, 8.0
    };
    const double initial_astaxanthin[16] = {
        0.0, 4.0, 0.0, 4.0, 0.0, 4.0, 0.0, 4.0,
        0.0, 4.0, 0.0, 4.0, 0.0, 4.0, 0.0, 4.0
    };
    float reference[832];

    for (int experiment = 0; experiment < 16; ++experiment) {
        reference[0 * 16 + experiment] = (float)initial_biomass[experiment];
        reference[1 * 16 + experiment] = (float)initial_glucose[experiment];
        reference[2 * 16 + experiment] = (float)initial_sucrose[experiment];
        reference[3 * 16 + experiment] = (float)initial_astaxanthin[experiment];

        double state[4][13];
        simulate_astaxanthin_ground_truth(
            observation_times, 13, 0.01,
            initial_biomass[experiment], initial_glucose[experiment],
            initial_sucrose[experiment], initial_astaxanthin[experiment],
            state[0], state[1], state[2], state[3]);
        for (int component = 0; component < 4; ++component) {
            for (int observation = 0; observation < 12; ++observation) {
                reference[64 + component * 192 + experiment * 12 + observation] =
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
    CUdeviceptr device_bindings;
    CUdeviceptr device_reference;
    CUdeviceptr device_constants;
    CUdeviceptr device_mse;
    CUdeviceptr device_iterations;
    CUdeviceptr device_accepted_steps;
    CUDA_CHECK(cuMemAlloc(&device_starts, (size_t)8u * num_fits * sizeof(float)));
    CUDA_CHECK(cuMemAlloc(
        &device_bindings, (size_t)16u * num_settings * sizeof(unsigned int)));
    CUDA_CHECK(cuMemAlloc(&device_reference, sizeof(reference)));
    CUDA_CHECK(cuMemAlloc(&device_constants, (size_t)6u * num_fits * sizeof(float)));
    CUDA_CHECK(cuMemAlloc(&device_mse, (size_t)num_fits * sizeof(float)));
    CUDA_CHECK(cuMemAlloc(&device_iterations, (size_t)num_fits * sizeof(unsigned int)));
    CUDA_CHECK(cuMemAlloc(&device_accepted_steps, (size_t)num_fits * sizeof(unsigned int)));
    CUDA_CHECK(cuMemcpyHtoD(
        device_starts, starts, (size_t)8u * num_fits * sizeof(float)));
    CUDA_CHECK(cuMemcpyHtoD(
        device_bindings, bindings,
        (size_t)16u * num_settings * sizeof(unsigned int)));
    CUDA_CHECK(cuMemcpyHtoD(device_reference, reference, sizeof(reference)));

    void *arguments[] = {
        &device_starts,
        (void *)&starts_leading_dimension,
        &device_bindings,
        (void *)&bindings_leading_dimension,
        (void *)&bindings_leading_dimension,
        (void *)&starts_per_setting,
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
    arguments[4] = &kernel_num_settings;
    const unsigned int ctas = (num_fits + threads - 1u) / threads;
    const unsigned int dynamic_shared_bytes = 12u * threads * sizeof(float);

    CUevent begin;
    CUevent end;
    CUDA_CHECK(cuEventCreate(&begin, CU_EVENT_DEFAULT));
    CUDA_CHECK(cuEventCreate(&end, CU_EVENT_DEFAULT));
    CUDA_CHECK(cuEventRecord(begin, 0));
    for (unsigned int repeat = 0u; repeat < repeats; ++repeat) {
        CUDA_CHECK(cuLaunchKernel(
            function,
            1u, ctas, 1u,
            threads, 1u, 1u,
            dynamic_shared_bytes, 0, arguments, NULL));
    }
    CUDA_CHECK(cuEventRecord(end, 0));
    CUDA_CHECK(cuEventSynchronize(end));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cuEventElapsedTime(&elapsed_ms, begin, end));

    CUDA_CHECK(cuMemcpyDtoH(
        constants, device_constants, (size_t)6u * num_fits * sizeof(float)));
    CUDA_CHECK(cuMemcpyDtoH(mse, device_mse, (size_t)num_fits * sizeof(float)));
    CUDA_CHECK(cuMemcpyDtoH(
        iterations, device_iterations, (size_t)num_fits * sizeof(unsigned int)));
    CUDA_CHECK(cuMemcpyDtoH(
        accepted_steps, device_accepted_steps,
        (size_t)num_fits * sizeof(unsigned int)));

    unsigned int best = 0u;
    unsigned int best_exact = UINT_MAX;
    unsigned int best_planted = 0u;
    unsigned int finite_count = 0u;
    unsigned int low_loss_count = 0u;
    unsigned int recovered_exact_binding_count = 0u;
    uint64_t total_iterations = 0u;
    uint64_t total_accepted_steps = 0u;
    const float truth[6] = {0.43f, 63.7f, 5.8f, 0.132f, 3.68f, 0.0f};
    for (unsigned int fit = 0u; fit < num_fits; ++fit) {
        if (isfinite(mse[fit])) ++finite_count;
        if (isfinite(mse[fit]) && mse[fit] < 1.0e-8f) ++low_loss_count;
        total_iterations += iterations[fit];
        total_accepted_steps += accepted_steps[fit];
        if (mse[fit] < mse[best]) best = fit;
        if (fit < starts_per_setting && mse[fit] < mse[best_planted]) {
            best_planted = fit;
        }
        const unsigned int setting = fit / starts_per_setting;
        int exact_binding = 1;
        for (unsigned int leaf = 0u; leaf < 16u; ++leaf) {
            exact_binding = exact_binding &&
                bindings[leaf * num_settings + setting] == planted_bindings[leaf];
        }
        if (exact_binding &&
            (best_exact == UINT_MAX || mse[fit] < mse[best_exact])) {
            best_exact = fit;
        }
        float maximum_scaled_error = 0.0f;
        for (int parameter = 0; parameter < 6; ++parameter) {
            const float scale = parameter == 5 ? 0.01f : fmaxf(fabsf(truth[parameter]), 1.0e-6f);
            const float error = fabsf(constants[parameter * num_fits + fit] - truth[parameter]) / scale;
            if (error > maximum_scaled_error) maximum_scaled_error = error;
        }
        if (exact_binding && maximum_scaled_error <= 0.01f) {
            ++recovered_exact_binding_count;
        }
    }

    char device_name[256];
    CUDA_CHECK(cuDeviceGetName(device_name, sizeof(device_name), device));
    printf("device=%s settings=%u starts_per_setting=%u fits=%u threads=%u "
           "ctas=%u steps_per_observation=%u lm_iterations=%u "
           "damping_attempts=%u\n",
           device_name, num_settings, starts_per_setting, num_fits, threads, ctas,
           steps_per_observation, max_lm_iterations, max_damping_attempts);
    printf("kernel_ms=%.6f per_launch_ms=%.6f finite=%u low_loss=%u "
           "exact_binding_settings=%u recovered_exact_binding_1pct=%u\n",
           elapsed_ms, elapsed_ms / (float)repeats, finite_count, low_loss_count,
           exact_binding_settings, recovered_exact_binding_count);
    printf("fits_per_second=%.3f average_iterations=%.3f average_accepted_steps=%.3f\n",
           (double)num_fits * (double)repeats * 1000.0 / (double)elapsed_ms,
           (double)total_iterations / (double)num_fits,
           (double)total_accepted_steps / (double)num_fits);
    printf("best_fit=%u best_setting=%u best_start=%u relative_mse=%.9g "
           "iterations=%u accepted=%u\n",
           best, best / starts_per_setting, best % starts_per_setting,
           mse[best], iterations[best], accepted_steps[best]);
    printf("best_planted_fit=%u planted_start=%u relative_mse=%.9g "
           "best_exact_fit=%u exact_relative_mse=%.9g\n",
           best_planted, best_planted, mse[best_planted], best_exact,
           best_exact == UINT_MAX ? INFINITY : mse[best_exact]);
    printf("parameter       start         fitted        truth          abs_error\n");
    const char *names[6] = {"mu_m1", "K_c1", "k_1", "mu_m2", "K_c2", "k_2"};
    for (int parameter = 0; parameter < 6; ++parameter) {
        const float start = starts[parameter * num_fits + best_planted];
        const float fitted = constants[parameter * num_fits + best_planted];
        printf("%-9s %12.7g  %12.7g  %12.7g  %12.7g\n",
               names[parameter], start, fitted, truth[parameter], fabsf(fitted - truth[parameter]));
    }

    float best_maximum_scaled_error = 0.0f;
    for (int parameter = 0; parameter < 6; ++parameter) {
        const float scale = parameter == 5 ? 0.01f : fmaxf(fabsf(truth[parameter]), 1.0e-6f);
        const float error =
            fabsf(constants[parameter * num_fits + best_planted] - truth[parameter]) / scale;
        if (error > best_maximum_scaled_error) best_maximum_scaled_error = error;
    }
    const int recovery_passed = isfinite(mse[best_planted]) &&
        mse[best_planted] < 1.0e-8f &&
        best_maximum_scaled_error < 0.01f;
    printf("planted_setting_recovery_test=%s maximum_scaled_parameter_error=%.9g\n",
           recovery_passed ? "PASS" : "FAIL", best_maximum_scaled_error);

    CUDA_CHECK(cuEventDestroy(begin));
    CUDA_CHECK(cuEventDestroy(end));
    CUDA_CHECK(cuMemFree(device_starts));
    CUDA_CHECK(cuMemFree(device_bindings));
    CUDA_CHECK(cuMemFree(device_reference));
    CUDA_CHECK(cuMemFree(device_constants));
    CUDA_CHECK(cuMemFree(device_mse));
    CUDA_CHECK(cuMemFree(device_iterations));
    CUDA_CHECK(cuMemFree(device_accepted_steps));
    CUDA_CHECK(cuModuleUnload(module));
    CUDA_CHECK(cuCtxDestroy(context));
    free(starts);
    free(bindings);
    free(constants);
    free(mse);
    free(iterations);
    free(accepted_steps);
    return recovery_passed ? 0 : 3;
}

#undef CUDA_CHECK
