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
#define _POSIX_C_SOURCE 200809L

#include "o_odezza_internal.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define POINTS 5u
#define SYSTEMS 3u
#define BANKS 3u
#define MAX_STATES 8u
#define CHECK(condition) do { if (!(condition)) { fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #condition); return 1; } } while (0)

typedef struct Resources {
    CUdevice device;
    CUcontext context;
    OdezzaScoringPipeline *pipeline;
    CUdeviceptr offsets, times, reference, scores;
    void *workspace;
    size_t workspace_size;
    OdezzaScoringSystem systems[SYSTEMS];
    OdezzaScoringRhs rhs[SYSTEMS][MAX_STATES];
    uint8_t programs[SYSTEMS][6];
    OdezzaScoringLaunch launch;
} Resources;

static int score(Resources *r, const uint32_t offsets[3], const float times[POINTS], const float *reference,
                 size_t states, int invalid) {
    float scores[SYSTEMS * BANKS * 2u];
    OdezzaScoringRunReport report;
    size_t configurations = BANKS << r->launch.active_toggle_count;
    size_t count = SYSTEMS * configurations;
    size_t i;
    CHECK(cuMemcpyHtoD(r->offsets, offsets, 3u * sizeof(*offsets)) == CUDA_SUCCESS);
    CHECK(cuMemcpyHtoD(r->times, times, POINTS * sizeof(*times)) == CUDA_SUCCESS);
    CHECK(cuMemcpyHtoD(r->reference, reference, states * POINTS * sizeof(*reference)) == CUDA_SUCCESS);
    CHECK(cuMemsetD32(r->scores, 0x7fc00000u, count) == CUDA_SUCCESS);
    CHECK(odezza_scoring_pipeline_run(r->pipeline, r->systems, SYSTEMS, &r->launch, r->workspace, r->workspace_size, &report) == ODEZZA_SUCCESS);
    CHECK(report.system_count == SYSTEMS && report.configuration_count == count);
    CHECK(cuMemcpyDtoH(scores, r->scores, count * sizeof(*scores)) == CUDA_SUCCESS);
    for (i = 0u; i < count; ++i) {
        float slope_error = (float)(i / configurations) + (invalid == 3 ? (float)(i & 1u) : 0.0f);
        int penalized = invalid == 1 || (invalid == 2 && i / configurations == SYSTEMS - 1u);
        float expected = penalized ? FLT_MAX : 0.4375f * slope_error * slope_error;
        if (!isfinite(scores[i]) || fabsf(scores[i] - expected) > (penalized ? 0.0f : 2.0e-6f)) {
            fprintf(stderr, "score %zu: %.9g, expected %.9g\n", i, scores[i], expected);
            return 1;
        }
    }
    return 0;
}

static int reject_launch(Resources *r, OdezzaScoringLaunch launch, size_t count, OdezzaResult expected) {
    OdezzaScoringPipelineStats before, after;
    OdezzaScoringRunReport report;
    CHECK(o_scoring_pipeline_stats(r->pipeline, &before) == ODEZZA_SUCCESS);
    CHECK(odezza_scoring_pipeline_run(r->pipeline, r->systems, count, &launch, r->workspace, r->workspace_size, &report) == expected);
    CHECK(o_scoring_pipeline_stats(r->pipeline, &after) == ODEZZA_SUCCESS);
    CHECK(before.submitted_ticket_count == after.submitted_ticket_count);
    CHECK(report.module_count == 0u && report.configuration_count == 0u);
    return 0;
}

static int run(Resources *r, size_t states, uint32_t constants, uint32_t capacity) {
    CUdevice device;
    int major, minor;
    OdezzaScoringPipelineCreateInfo info = {0};
    uint32_t offsets[3] = {0u, 3u, POINTS};
    float times[POINTS] = {0.0f, 0.25f, 1.0f, 0.0f, 0.5f};
    float reference[MAX_STATES * POINTS];
    const float observations[POINTS] = {1.0f, 1.25f, 2.0f, 2.0f, 2.5f};
    size_t alignment, s, c;
    OdezzaResult result;
    CHECK(states > 0u && states <= MAX_STATES);
    CHECK(setenv("CUDA_MODULE_LOADING", "EAGER", 1) == 0);
    CHECK(cuInit(0) == CUDA_SUCCESS);
    CHECK(cuDeviceGet(&device, 0) == CUDA_SUCCESS);
    CHECK(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device) == CUDA_SUCCESS);
    CHECK(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device) == CUDA_SUCCESS);
    r->device = device;
    CHECK(cuDevicePrimaryCtxRetain(&r->context, device) == CUDA_SUCCESS);
    CHECK(cuCtxSetCurrent(r->context) == CUDA_SUCCESS);
    info.sm_version = (uint32_t)(major * 10 + minor);
    info.state_count = states;
    info.state_capacity = states;
    info.constant_capacity = constants;
    info.system_capacity = capacity;
    info.shared_patch_capacity = 64u;
    info.system_patch_capacity = 128u;
    info.worker_count = 1u;
    info.cubin_slots_per_worker = 2u;
    result = odezza_scoring_pipeline_create(&info, &r->pipeline);
    if (result != ODEZZA_SUCCESS) {
        char error[512];
        size_t bytes;
        if (r->pipeline && odezza_scoring_pipeline_write_error(r->pipeline, error, sizeof(error), &bytes) == ODEZZA_SUCCESS)
            fprintf(stderr, "create (%zu states, %u constants, %u systems): %d %s\n", states, constants, capacity, result, error);
        return 1;
    }
    CHECK(odezza_scoring_pipeline_workspace_requirements(r->pipeline, &r->workspace_size, &alignment) == ODEZZA_SUCCESS);
    r->workspace = malloc(r->workspace_size);
    CHECK(r->workspace != NULL && (uintptr_t)r->workspace % alignment == 0u);
    CHECK(cuMemAlloc(&r->offsets, sizeof(offsets)) == CUDA_SUCCESS);
    CHECK(cuMemAlloc(&r->times, sizeof(times)) == CUDA_SUCCESS);
    CHECK(cuMemAlloc(&r->reference, states * POINTS * sizeof(float)) == CUDA_SUCCESS);
    CHECK(cuMemAlloc(&r->scores, SYSTEMS * BANKS * 2u * sizeof(float)) == CUDA_SUCCESS);
    for (s = 0u; s < SYSTEMS; ++s) {
        float slope = (float)(s + 1u);
        uint32_t bits;
        memcpy(&bits, &slope, sizeof(bits));
        r->programs[s][0] = ODEZZA_AST_LITERAL_F32;
        for (c = 0u; c < 4u; ++c) r->programs[s][c + 1u] = (uint8_t)(bits >> (8u * c));
        r->programs[s][5] = ODEZZA_AST_RETURN_F32;
        for (c = 0u; c < states; ++c) {
            r->rhs[s][c].state_index = (uint8_t)c;
            r->rhs[s][c].program.bytes = r->programs[s];
            r->rhs[s][c].program.byte_count = sizeof(r->programs[s]);
        }
        r->systems[s].rhs = r->rhs[s];
        r->systems[s].rhs_count = states;
    }
    for (c = 0u; c < states; ++c) memcpy(reference + c * POINTS, observations, sizeof(observations));
    r->launch.constant_bank_count = BANKS;
    r->launch.trajectory_offsets_device = r->offsets;
    r->launch.trajectory_times_device = r->times;
    r->launch.reference_data_device = r->reference;
    r->launch.trajectory_count = 2u;
    r->launch.trajectory_point_count = POINTS;
    r->launch.steps_per_observation = 2u;
    r->launch.mse_output_device = r->scores;
    CHECK(score(r, offsets, times, reference, states, 0) == 0);

    offsets[0] = 1u;
    CHECK(score(r, offsets, times, reference, states, 1) == 0);
    offsets[0] = 0u; offsets[1] = 0u;
    CHECK(score(r, offsets, times, reference, states, 1) == 0);
    offsets[1] = 3u; offsets[2] = POINTS + 1u;
    CHECK(score(r, offsets, times, reference, states, 1) == 0);
    offsets[2] = POINTS - 1u;
    CHECK(score(r, offsets, times, reference, states, 1) == 0);
    offsets[2] = POINTS;
    times[1] = 0.0f;
    CHECK(score(r, offsets, times, reference, states, 1) == 0);
    times[1] = NAN;
    CHECK(score(r, offsets, times, reference, states, 1) == 0);
    times[1] = 0.25f; reference[1] = INFINITY;
    CHECK(score(r, offsets, times, reference, states, 1) == 0);
    reference[1] = observations[1];
    /* A singleton trajectory still has an initial observation to validate. */
    offsets[1] = 1u; times[3] = 1.25f; times[4] = 1.5f; reference[0] = NAN;
    CHECK(score(r, offsets, times, reference, states, 1) == 0);
    reference[0] = observations[0]; times[0] = NAN;
    CHECK(score(r, offsets, times, reference, states, 1) == 0);
    offsets[1] = 3u; times[0] = 0.0f; times[3] = 0.0f; times[4] = 0.5f;
    CHECK(score(r, offsets, times, reference, states, 0) == 0);
    times[1] = nextafterf(0.0f, 1.0f);
    CHECK(score(r, offsets, times, reference, states, 1) == 0);
    times[1] = 0.25f;
    {
        OdezzaScoringLaunch bad = r->launch;
        bad.mse_output_device = UINT64_MAX - 3u;
        CHECK(reject_launch(r, bad, SYSTEMS, ODEZZA_ERROR_OVERFLOW) == 0);
        bad = r->launch; bad.reference_data_device = UINT64_MAX - 3u;
        CHECK(reject_launch(r, bad, SYSTEMS, ODEZZA_ERROR_OVERFLOW) == 0);
        bad = r->launch; bad.trajectory_times_device += 1u;
        CHECK(reject_launch(r, bad, SYSTEMS, ODEZZA_ERROR_OVERFLOW) == 0);
        bad = r->launch; bad.trajectory_point_count = UINT32_MAX;
        CHECK(reject_launch(r, bad, SYSTEMS, ODEZZA_ERROR_OVERFLOW) == 0);
        bad = r->launch; bad.constant_bank_count = 128u; bad.active_toggle_count = 32u;
        CHECK(reject_launch(r, bad, SYSTEMS, ODEZZA_ERROR_UNSUPPORTED) == 0);
        bad = r->launch; bad.active_toggle_count = 33u;
        CHECK(reject_launch(r, bad, SYSTEMS, ODEZZA_ERROR_INVALID_ARGUMENT) == 0);
        CHECK(reject_launch(r, r->launch, SIZE_MAX / sizeof(*r->systems) + 1u, ODEZZA_ERROR_OVERFLOW) == 0);
    }
    /* A failed specialization must release slots and leave the handle reusable. */
    r->rhs[0][0].program.byte_count = 1u;
    CHECK(odezza_scoring_pipeline_run(r->pipeline, r->systems, SYSTEMS, &r->launch, r->workspace, r->workspace_size, NULL) == ODEZZA_ERROR_AST);
    r->rhs[0][0].program.byte_count = sizeof(r->programs[0]);
    CHECK(score(r, offsets, times, reference, states, 0) == 0);
    {
        static const uint8_t divide_zero[] = {ODEZZA_AST_LITERAL_F32, 0, 0, 128, 63,
            ODEZZA_AST_LITERAL_F32, 0, 0, 0, 0, ODEZZA_AST_DIV_F32, ODEZZA_AST_RETURN_F32};
        r->rhs[SYSTEMS - 1u][0].program.bytes = divide_zero;
        r->rhs[SYSTEMS - 1u][0].program.byte_count = sizeof(divide_zero);
        CHECK(score(r, offsets, times, reference, states, 2) == 0);
    }
    {
        uint8_t toggles[SYSTEMS][13];
        for (s = 0u; s < SYSTEMS; ++s) {
            float alternate = (float)(s + 2u);
            uint32_t bits;
            memcpy(&bits, &alternate, sizeof(bits));
            memcpy(toggles[s], r->programs[s], 5u);
            toggles[s][5] = ODEZZA_AST_LITERAL_F32;
            for (c = 0u; c < 4u; ++c) toggles[s][6u + c] = (uint8_t)(bits >> (8u * c));
            toggles[s][10] = ODEZZA_AST_TOGGLE2_F32;
            toggles[s][11] = 0u;
            toggles[s][12] = ODEZZA_AST_RETURN_F32;
            for (c = 0u; c < states; ++c) {
                r->rhs[s][c].program.bytes = toggles[s];
                r->rhs[s][c].program.byte_count = sizeof(toggles[s]);
            }
        }
        r->launch.active_toggle_count = 1u;
        CHECK(score(r, offsets, times, reference, states, 3) == 0);
    }
    printf("CUDA scoring: %zu states, %u constant capacity, %u system capacity; all scores and invalid data verified\n", states, constants, capacity);
    return 0;
}

int main(int argc, char **argv) {
    Resources r = {0};
    int status = run(&r, argc > 1 ? (size_t)strtoul(argv[1], NULL, 10) : 1u,
                     argc > 2 ? (uint32_t)strtoul(argv[2], NULL, 10) : 0u,
                     argc > 3 ? (uint32_t)strtoul(argv[3], NULL, 10) : 1u);
    if (r.pipeline != NULL && odezza_scoring_pipeline_destroy(r.pipeline) != ODEZZA_SUCCESS) status = 1;
    free(r.workspace);
    if (r.scores) (void)cuMemFree(r.scores);
    if (r.reference) (void)cuMemFree(r.reference);
    if (r.times) (void)cuMemFree(r.times);
    if (r.offsets) (void)cuMemFree(r.offsets);
    if (r.context) (void)cuDevicePrimaryCtxRelease(r.device);
    return status;
}
