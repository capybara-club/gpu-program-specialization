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

#include "odezza.h"
#include "o_odezza_internal.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define STATE_COUNT 2u
#define STATE_CAPACITY 4u
#define CONSTANT_COUNT 4u
#define CONSTANT_CAPACITY 8u
#define TOGGLE_BIT_COUNT 5u
#define TRAJECTORY_COUNT 3u
#define TRAJECTORY_POINT_COUNT (17u + 13u + 9u)
#define REFERENCE_FLOAT_COUNT (STATE_COUNT * TRAJECTORY_POINT_COUNT)
#define PROGRAM_CAPACITY 64u

static double monotonic_seconds(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
    return (double)value.tv_sec + (double)value.tv_nsec * 1.0e-9;
}

static int parse_u32(const char *text, uint32_t *value_ret) {
    char *end = NULL;
    unsigned long value;
    errno = 0;
    value = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value > UINT32_MAX) return 0;
    *value_ret = (uint32_t)value;
    return 1;
}

static int multiply_size(size_t lhs, size_t rhs, size_t *value_ret) {
    if (value_ret == NULL || (rhs != 0u && lhs > SIZE_MAX / rhs)) return 0;
    *value_ret = lhs * rhs;
    return 1;
}

static size_t write_candidate(unsigned char program[PROGRAM_CAPACITY], float bias, uint32_t variant, uint32_t toggle_test_bit) {
    uint32_t bits;
    size_t offset = 0u;
#define BYTE(value) program[offset++] = (uint8_t)(value)
#define STATE(index) \
    do { \
        BYTE(ODEZZA_AST_STATE_F32); \
        BYTE(index); \
    } while (0)
#define CONSTANT(index) \
    do { \
        BYTE(ODEZZA_AST_CONSTANT_F32); \
        BYTE(index); \
    } while (0)
#define LITERAL(value) \
    do { \
        float literal_value_ = (value); \
        memcpy(&bits, &literal_value_, sizeof(bits)); \
        BYTE(ODEZZA_AST_LITERAL_F32); \
        BYTE(bits); \
        BYTE(bits >> 8u); \
        BYTE(bits >> 16u); \
        BYTE(bits >> 24u); \
    } while (0)
    variant %= 8u;
    CONSTANT(2u);
    if (variant == 1u) {
        STATE(0u);
        LITERAL(100.0f);
        BYTE(ODEZZA_AST_TOGGLE2_F32);
        BYTE(toggle_test_bit);
    } else if (variant == 5u) {
        STATE(0u);
        STATE(1u);
        BYTE(ODEZZA_AST_ADD_F32);
    } else {
        STATE(0u);
    }
    BYTE(ODEZZA_AST_MUL_F32);
    if (variant == 2u) {
        LITERAL(bias);
        BYTE(ODEZZA_AST_ADD_F32);
    }
    STATE(1u);
    BYTE(ODEZZA_AST_MUL_F32);
    if (variant == 4u) {
        STATE(0u);
        BYTE(ODEZZA_AST_ABS_F32);
        LITERAL(1.0f);
        BYTE(ODEZZA_AST_ADD_F32);
        BYTE(ODEZZA_AST_DIV_F32);
    }
    CONSTANT(3u);
    STATE(1u);
    if (variant == 3u) {
        LITERAL(1.0f);
        BYTE(ODEZZA_AST_TOGGLE2_F32);
        BYTE(toggle_test_bit);
    }
    BYTE(ODEZZA_AST_MUL_F32);
    if (variant == 6u) {
        STATE(1u);
        BYTE(ODEZZA_AST_MUL_F32);
    }
    BYTE(ODEZZA_AST_SUB_F32);
    if (variant == 7u) {
        STATE(0u);
        BYTE(ODEZZA_AST_ABS_F32);
        LITERAL(bias);
        BYTE(ODEZZA_AST_MUL_F32);
        BYTE(ODEZZA_AST_ADD_F32);
    } else if (variant != 2u) {
        LITERAL(bias);
        BYTE(ODEZZA_AST_ADD_F32);
    }
    BYTE(ODEZZA_AST_RETURN_F32);
#undef LITERAL
#undef CONSTANT
#undef STATE
#undef BYTE
    return offset;
}

static size_t write_multi_toggle_candidate(unsigned char program[PROGRAM_CAPACITY]) {
    size_t offset = 0u;
    uint32_t zero = 0u;
    program[offset++] = ODEZZA_AST_LITERAL_F32;
    memcpy(program + offset, &zero, sizeof(zero));
    offset += sizeof(zero);
    program[offset++] = ODEZZA_AST_LITERAL_F32;
    memcpy(program + offset, &zero, sizeof(zero));
    offset += sizeof(zero);
    program[offset++] = ODEZZA_AST_TOGGLE2_F32;
    program[offset++] = 0u;
    program[offset++] = ODEZZA_AST_STATE_F32;
    program[offset++] = 0u;
    program[offset++] = ODEZZA_AST_STATE_F32;
    program[offset++] = 1u;
    program[offset++] = ODEZZA_AST_TOGGLE2_F32;
    program[offset++] = 1u;
    program[offset++] = ODEZZA_AST_ADD_F32;
    program[offset++] = ODEZZA_AST_RETURN_F32;
    return offset;
}

static void generate_reference(
    float reference[REFERENCE_FLOAT_COUNT],
    float times[TRAJECTORY_POINT_COUNT],
    uint32_t offsets[TRAJECTORY_COUNT + 1u],
    uint32_t steps
) {
    static const float initial[STATE_COUNT][TRAJECTORY_COUNT] = {{2.0f, 1.5f, 2.5f}, {1.0f, 0.75f, 1.25f}};
    static const float constants[CONSTANT_COUNT] = {1.1f, 0.4f, 0.1f, 0.4f};
    static const uint32_t point_counts[TRAJECTORY_COUNT] = {17u, 13u, 9u};
    uint32_t next_point = 0u;
    uint32_t trajectory;
    uint32_t state_index;
    memset(reference, 0, REFERENCE_FLOAT_COUNT * sizeof(float));
    for (trajectory = 0u; trajectory < TRAJECTORY_COUNT; ++trajectory) {
        uint32_t point_begin = next_point;
        uint32_t point_count = point_counts[trajectory];
        float state[STATE_COUNT] = {initial[0][trajectory], initial[1][trajectory]};
        uint32_t observation;
        float current_time = 0.0f;
        offsets[trajectory] = point_begin;
        times[point_begin] = 0.0f;
        for (state_index = 0u; state_index < STATE_COUNT; ++state_index) {
            reference[state_index * TRAJECTORY_POINT_COUNT + point_begin] = initial[state_index][trajectory];
        }
        for (observation = 0u; observation + 1u < point_count; ++observation) {
            float interval = 0.06f + 0.01f * (float)((observation + trajectory) % 5u);
            float h = interval / (float)steps;
            float half_h = 0.5f * h;
            float sixth_h = h / 6.0f;
            uint32_t point = point_begin + observation + 1u;
            uint32_t step;
            for (step = 0u; step < steps; ++step) {
                float base[STATE_COUNT] = {state[0], state[1]};
                float stage[STATE_COUNT] = {state[0], state[1]};
                float sum[STATE_COUNT] = {0.0f, 0.0f};
                uint32_t rk_stage;
                for (rk_stage = 0u; rk_stage < 4u; ++rk_stage) {
                    float rhs0 = fmaf(constants[0], stage[0], -constants[1] * stage[0] * stage[1]);
                    float rhs1 = fmaf(constants[2] * stage[0], stage[1], -constants[3] * stage[1]);
                    float weight = (rk_stage == 0u || rk_stage == 3u) ? 1.0f : 2.0f;
                    sum[0] = fmaf(weight, rhs0, sum[0]);
                    sum[1] = fmaf(weight, rhs1, sum[1]);
                    if (rk_stage != 3u) {
                        float stage_h = rk_stage == 2u ? h : half_h;
                        stage[0] = fmaf(stage_h, rhs0, base[0]);
                        stage[1] = fmaf(stage_h, rhs1, base[1]);
                    }
                }
                state[0] = fmaf(sixth_h, sum[0], base[0]);
                state[1] = fmaf(sixth_h, sum[1], base[1]);
            }
            current_time += interval;
            times[point] = current_time;
            reference[point] = state[0];
            reference[TRAJECTORY_POINT_COUNT + point] = state[1];
        }
        next_point += point_count;
    }
    offsets[TRAJECTORY_COUNT] = next_point;
}

static void print_pipeline_error(OdezzaScoringPipeline *pipeline, OdezzaResult result) {
    char error[512];
    size_t bytes = 0u;
    if (pipeline != NULL && odezza_scoring_pipeline_write_error(pipeline, error, sizeof(error), &bytes) == ODEZZA_SUCCESS && bytes != 0u) {
        fprintf(stderr, "Odezza failed with result %d: %s\n", (int)result, error);
    } else {
        fprintf(stderr, "Odezza failed with result %d\n", (int)result);
    }
}

static void usage(const char *program) {
    fprintf(stderr,
            "usage: %s [--submissions N] [--workers N] [--slots-per-worker N] "
            "[--queue-capacity N] [--loaded-modules N] [--streams N] [--systems N] "
            "[--state-capacity N] [--constant-capacity N] "
            "[--constant-banks N] [--toggle-bits N] [--toggle-test-bit N] [--steps-per-observation N] [--shared-patch N] "
            "[--system-patch N] [--runs N] [--validate-toggle 0|1] [--validate-runtime-widths 0|1] "
            "[--validate-multi-toggle 0|1] [--device N]\n",
            program);
}

typedef struct OBenchmarkResources {
    CUdevice device;
    CUcontext context;
    CUdeviceptr constants_device;
    CUdeviceptr trajectory_offsets_device;
    CUdeviceptr trajectory_times_device;
    CUdeviceptr reference_device;
    CUdeviceptr scores_device;
    OdezzaScoringPipeline *pipeline;
    OdezzaScoringSystem *systems;
    OdezzaScoringRhs *rhs;
    unsigned char *programs;
    float *constants;
    float *scores;
    void *workspace;
} OBenchmarkResources;

static int o_benchmark_run(int argc, char **argv, OBenchmarkResources *resources) {
    static const uint8_t fixed_program[] = {ODEZZA_AST_CONSTANT_F32,
                                            0u,
                                            ODEZZA_AST_STATE_F32,
                                            0u,
                                            ODEZZA_AST_MUL_F32,
                                            ODEZZA_AST_CONSTANT_F32,
                                            1u,
                                            ODEZZA_AST_STATE_F32,
                                            0u,
                                            ODEZZA_AST_MUL_F32,
                                            ODEZZA_AST_STATE_F32,
                                            1u,
                                            ODEZZA_AST_MUL_F32,
                                            ODEZZA_AST_SUB_F32,
                                            ODEZZA_AST_RETURN_F32};
    uint32_t submissions = 64u;
    uint32_t workers = 1u;
    uint32_t slots_per_worker = 2u;
    uint32_t queue_capacity = 256u;
    uint32_t loaded_modules = 2u;
    uint32_t streams = 2u;
    uint32_t systems_per_module = 2u;
    uint32_t state_capacity = STATE_CAPACITY;
    uint32_t constant_capacity = CONSTANT_CAPACITY;
    uint32_t toggle_bit_count = TOGGLE_BIT_COUNT;
    uint32_t toggle_test_bit = 0u;
    uint32_t constant_bank_count = 1u;
    uint32_t steps_per_observation = 4u;
    uint32_t shared_patch_capacity = 64u;
    uint32_t system_patch_capacity = 64u;
    uint32_t run_count = 1u;
    uint32_t validate_toggle = 0u;
    uint32_t validate_runtime_widths = 0u;
    uint32_t validate_multi_toggle = 0u;
    int runtime_widths_verified = 0;
    int multi_toggle_verified = 0;
    int constant_bank_pairing_verified = 0;
    uint32_t device_ordinal = 0u;
    uint32_t argument;
    CUdevice device = 0;
    CUcontext context = NULL;
    int compute_major = 0;
    int compute_minor = 0;
    CUdeviceptr constants_device = 0u;
    CUdeviceptr trajectory_offsets_device = 0u;
    CUdeviceptr trajectory_times_device = 0u;
    CUdeviceptr reference_device = 0u;
    CUdeviceptr scores_device = 0u;
    OdezzaScoringPipeline *pipeline = NULL;
    OdezzaScoringPipelineCreateInfo create_info;
    OdezzaScoringPipelineTuning tuning;
    OdezzaScoringLaunch launch;
    OdezzaScoringRunReport report;
    OdezzaScoringPipelineStats stats;
    OdezzaScoringRhs fixed_rhs;
    OdezzaScoringSystem *systems = NULL;
    OdezzaScoringRhs *rhs = NULL;
    unsigned char *programs = NULL;
    float *constants = NULL;
    float *scores = NULL;
    float reference[REFERENCE_FLOAT_COUNT];
    float trajectory_times[TRAJECTORY_POINT_COUNT];
    uint32_t trajectory_offsets[TRAJECTORY_COUNT + 1u];
    void *workspace = NULL;
    const void *retained_cubin = NULL;
    size_t retained_cubin_size = 0u;
    const OdezzaScoringCubinInspection *cubin_inspection = NULL;
    size_t total_systems;
    size_t constant_float_count;
    size_t score_count;
    size_t workspace_size;
    size_t workspace_alignment;
    uint64_t configurations;
    double create_started;
    double create_seconds;
    OdezzaResult result = ODEZZA_SUCCESS;
    int use_private_tuning = 0;

    for (argument = 1u; argument < (uint32_t)argc; argument += 2u) {
        uint32_t *target = NULL;
        if (strcmp(argv[argument], "--help") == 0) {
            usage(argv[0]);
            return 0;
        }
        if (argument + 1u >= (uint32_t)argc) {
            usage(argv[0]);
            return 2;
        }
        if (strcmp(argv[argument], "--submissions") == 0)
            target = &submissions;
        else if (strcmp(argv[argument], "--workers") == 0)
            target = &workers;
        else if (strcmp(argv[argument], "--slots-per-worker") == 0)
            target = &slots_per_worker;
        else if (strcmp(argv[argument], "--queue-capacity") == 0) {
            target = &queue_capacity;
            use_private_tuning = 1;
        } else if (strcmp(argv[argument], "--loaded-modules") == 0) {
            target = &loaded_modules;
            use_private_tuning = 1;
        } else if (strcmp(argv[argument], "--streams") == 0) {
            target = &streams;
            use_private_tuning = 1;
        } else if (strcmp(argv[argument], "--systems") == 0)
            target = &systems_per_module;
        else if (strcmp(argv[argument], "--state-capacity") == 0)
            target = &state_capacity;
        else if (strcmp(argv[argument], "--constant-capacity") == 0)
            target = &constant_capacity;
        else if (strcmp(argv[argument], "--constant-banks") == 0)
            target = &constant_bank_count;
        else if (strcmp(argv[argument], "--toggle-bits") == 0)
            target = &toggle_bit_count;
        else if (strcmp(argv[argument], "--toggle-test-bit") == 0)
            target = &toggle_test_bit;
        else if (strcmp(argv[argument], "--steps-per-observation") == 0)
            target = &steps_per_observation;
        else if (strcmp(argv[argument], "--shared-patch") == 0)
            target = &shared_patch_capacity;
        else if (strcmp(argv[argument], "--system-patch") == 0)
            target = &system_patch_capacity;
        else if (strcmp(argv[argument], "--runs") == 0)
            target = &run_count;
        else if (strcmp(argv[argument], "--validate-toggle") == 0)
            target = &validate_toggle;
        else if (strcmp(argv[argument], "--validate-runtime-widths") == 0)
            target = &validate_runtime_widths;
        else if (strcmp(argv[argument], "--validate-multi-toggle") == 0)
            target = &validate_multi_toggle;
        else if (strcmp(argv[argument], "--device") == 0)
            target = &device_ordinal;
        else {
            usage(argv[0]);
            return 2;
        }
        if (!parse_u32(argv[argument + 1u], target) ||
            (target != &validate_toggle && target != &validate_runtime_widths && target != &validate_multi_toggle && *target == 0u) ||
            validate_toggle > 1u || validate_runtime_widths > 1u || validate_multi_toggle > 1u
        ) {
            usage(argv[0]);
            return 2;
        }
    }
    if (state_capacity < STATE_COUNT || constant_capacity < CONSTANT_COUNT || toggle_bit_count > ODEZZA_MAX_TOGGLE_BITS ||
        toggle_test_bit >= toggle_bit_count
    ) {
        fprintf(stderr, "benchmark capacities are smaller than its active Lotka-Volterra shape\n");
        return 2;
    }
    if (!multiply_size(submissions, systems_per_module, &total_systems) || !multiply_size(total_systems, constant_bank_count, &constant_float_count) ||
        !multiply_size(constant_float_count, CONSTANT_COUNT, &constant_float_count)
    ) {
        fprintf(stderr, "benchmark shape overflows host sizes\n");
        return 1;
    }
    configurations = (uint64_t)constant_bank_count * (UINT64_C(1) << toggle_bit_count);
    if (configurations > SIZE_MAX || !multiply_size(total_systems, (size_t)configurations, &score_count)) {
        fprintf(stderr, "benchmark output shape overflows host sizes\n");
        return 1;
    }
    systems = (OdezzaScoringSystem *)calloc(total_systems, sizeof(*systems));
    rhs = (OdezzaScoringRhs *)calloc(total_systems, sizeof(*rhs));
    programs = (unsigned char *)calloc(total_systems, PROGRAM_CAPACITY);
    constants = (float *)malloc(constant_float_count * sizeof(*constants));
    scores = (float *)malloc(score_count * sizeof(*scores));
    resources->systems = systems;
    resources->rhs = rhs;
    resources->programs = programs;
    resources->constants = constants;
    resources->scores = scores;
    if (systems == NULL || rhs == NULL || programs == NULL || constants == NULL || scores == NULL) {
        fprintf(stderr, "benchmark host allocation failed\n");
        return 1;
    }
    for (size_t index = 0u; index < total_systems; ++index) {
        unsigned char *program = programs + index * PROGRAM_CAPACITY;
        float bias = index == 0u ? 0.0f : 1.0e-6f * (float)(index + 1u);
        rhs[index].state_index = 1u;
        rhs[index].program.bytes = program;
        rhs[index].program.byte_count = validate_multi_toggle && index == 2u
            ? write_multi_toggle_candidate(program)
            : write_candidate(program, bias, (uint32_t)index, toggle_test_bit);
        systems[index].rhs = &rhs[index];
        systems[index].rhs_count = 1u;
    }
    for (size_t index = 0u; index < total_systems * constant_bank_count; ++index) {
        float bias = 1.0e-5f * (float)(index % constant_bank_count);
        memcpy(
            constants + index * CONSTANT_COUNT,
            (float[CONSTANT_COUNT]){1.1f + bias, 0.4f, 0.1f, 0.4f},
            CONSTANT_COUNT * sizeof(float)
        );
    }
    generate_reference(reference, trajectory_times, trajectory_offsets, steps_per_observation);
    if (setenv("CUDA_MODULE_LOADING", "EAGER", 1) != 0 || cuInit(0u) != CUDA_SUCCESS || cuDeviceGet(&device, (int)device_ordinal) != CUDA_SUCCESS ||
        cuDeviceGetAttribute(&compute_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device) != CUDA_SUCCESS ||
        cuDeviceGetAttribute(&compute_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device) != CUDA_SUCCESS ||
        cuDevicePrimaryCtxRetain(&context, device) != CUDA_SUCCESS || cuCtxSetCurrent(context) != CUDA_SUCCESS
    ) {
        resources->device = device;
        resources->context = context;
        fprintf(stderr, "the benchmark context owner could not activate the CUDA device\n");
        return 1;
    }
    resources->device = device;
    resources->context = context;
    fixed_rhs.state_index = 0u;
    fixed_rhs.program.bytes = fixed_program;
    fixed_rhs.program.byte_count = sizeof(fixed_program);
    memset(&create_info, 0, sizeof(create_info));
    create_info.sm_version = (uint32_t)(compute_major * 10 + compute_minor);
    create_info.state_count = STATE_COUNT;
    create_info.state_capacity = state_capacity;
    create_info.constant_count = CONSTANT_COUNT;
    create_info.constant_capacity = constant_capacity;
    create_info.fixed_rhs = &fixed_rhs;
    create_info.fixed_rhs_count = 1u;
    create_info.system_capacity = systems_per_module;
    create_info.shared_patch_capacity = shared_patch_capacity;
    create_info.system_patch_capacity = system_patch_capacity;
    create_info.worker_count = workers;
    create_info.cubin_slots_per_worker = slots_per_worker;
    memset(&tuning, 0, sizeof(tuning));
    tuning.queue_capacity = queue_capacity;
    tuning.maximum_loaded_modules = loaded_modules;
    tuning.execution_stream_count = streams;
    create_started = monotonic_seconds();
    result =
        use_private_tuning ? o_scoring_pipeline_create_with_tuning(&create_info, &tuning, &pipeline) : odezza_scoring_pipeline_create(&create_info, &pipeline);
    resources->pipeline = pipeline;
    create_seconds = monotonic_seconds() - create_started;
    if (result != ODEZZA_SUCCESS) {
        print_pipeline_error(pipeline, result);
        return 1;
    }
    result = odezza_scoring_pipeline_workspace_requirements(pipeline, &workspace_size, &workspace_alignment);
    if (result != ODEZZA_SUCCESS) {
        print_pipeline_error(pipeline, result);
        return 1;
    }
    workspace = malloc(workspace_size);
    resources->workspace = workspace;
    if (workspace == NULL || ((uintptr_t)workspace % workspace_alignment) != 0u) {
        fprintf(stderr, "benchmark workspace allocation failed\n");
        return 1;
    }
    result = cuMemAlloc(&constants_device, constant_float_count * sizeof(*constants)) != CUDA_SUCCESS ||
                     cuMemAlloc(&trajectory_offsets_device, sizeof(trajectory_offsets)) != CUDA_SUCCESS ||
                     cuMemAlloc(&trajectory_times_device, sizeof(trajectory_times)) != CUDA_SUCCESS ||
                     cuMemAlloc(&reference_device, sizeof(reference)) != CUDA_SUCCESS ||
                     cuMemAlloc(&scores_device, score_count * sizeof(*scores)) != CUDA_SUCCESS ||
                     cuMemcpyHtoD(constants_device, constants, constant_float_count * sizeof(*constants)) != CUDA_SUCCESS ||
                     cuMemcpyHtoD(trajectory_offsets_device, trajectory_offsets, sizeof(trajectory_offsets)) != CUDA_SUCCESS ||
                     cuMemcpyHtoD(trajectory_times_device, trajectory_times, sizeof(trajectory_times)) != CUDA_SUCCESS ||
                     cuMemcpyHtoD(reference_device, reference, sizeof(reference)) != CUDA_SUCCESS
                 ? ODEZZA_ERROR_CUDA
                 : ODEZZA_SUCCESS;
    resources->constants_device = constants_device;
    resources->trajectory_offsets_device = trajectory_offsets_device;
    resources->trajectory_times_device = trajectory_times_device;
    resources->reference_device = reference_device;
    resources->scores_device = scores_device;
    if (result != ODEZZA_SUCCESS) {
        fprintf(stderr, "benchmark CUDA memory setup failed\n");
        return 1;
    }
    memset(&launch, 0, sizeof(launch));
    launch.constant_banks_device = constants_device;
    launch.constant_bank_count = constant_bank_count;
    launch.active_toggle_count = toggle_bit_count;
    launch.trajectory_offsets_device = trajectory_offsets_device;
    launch.trajectory_times_device = trajectory_times_device;
    launch.reference_data_device = reference_device;
    launch.trajectory_count = TRAJECTORY_COUNT;
    launch.trajectory_point_count = TRAJECTORY_POINT_COUNT;
    launch.steps_per_observation = steps_per_observation;
    launch.mse_output_device = scores_device;
    memset(&report, 0, sizeof(report));
    for (uint32_t run_index = 0u; run_index < run_count; ++run_index) {
        OdezzaScoringRunReport run_report;
        result = odezza_scoring_pipeline_run(
            pipeline,
            systems,
            total_systems,
            &launch,
            workspace,
            workspace_size,
            &run_report
        );
        if (result != ODEZZA_SUCCESS) {
            print_pipeline_error(pipeline, result);
            return 1;
        }
        if (SIZE_MAX - report.system_count < run_report.system_count || SIZE_MAX - report.module_count < run_report.module_count ||
            UINT64_MAX - report.configuration_count < run_report.configuration_count
        ) {
            fprintf(stderr, "aggregate benchmark report overflowed\n");
            return 1;
        }
        report.system_count += run_report.system_count;
        report.module_count += run_report.module_count;
        report.configuration_count += run_report.configuration_count;
        report.total_seconds += run_report.total_seconds;
    }
    if (cuMemcpyDtoH(scores, scores_device, score_count * sizeof(*scores)) != CUDA_SUCCESS) {
        fprintf(stderr, "benchmark score download failed\n");
        return 1;
    }
    if (!isfinite(scores[0]) || scores[0] > 1.0e-10f) {
        fprintf(stderr, "correct-system replay MSE %.9g exceeds tolerance\n", scores[0]);
        return 1;
    }
    if (validate_toggle) {
        float standard_false;
        float standard_true;
        if (total_systems < 2u || configurations < 2u) {
            fprintf(stderr, "specialized-toggle validation requires at least two systems and two configurations\n");
            return 1;
        }
        if (constant_bank_count > 1u) {
            size_t permutation_count = (size_t)UINT64_C(1) << toggle_bit_count;
            size_t permutation_index;
            for (permutation_index = 1u; permutation_index < permutation_count; ++permutation_index) {
                if (memcmp(&scores[0], &scores[permutation_index], sizeof(scores[0])) != 0 ||
                    memcmp(&scores[permutation_count], &scores[permutation_count + permutation_index], sizeof(scores[0])) != 0
                ) {
                    fprintf(stderr, "specialized permutation wrapping changed constants within one bank\n");
                    return 1;
                }
            }
            if (memcmp(&scores[0], &scores[permutation_count], sizeof(scores[0])) == 0) {
                fprintf(stderr, "specialized permutation wrapping did not advance to the next constant bank\n");
                return 1;
            }
            constant_bank_pairing_verified = 1;
        }
        standard_false = scores[configurations];
        standard_true = scores[configurations + (UINT64_C(1) << toggle_test_bit)];
        if (!isfinite(standard_false) || !isfinite(standard_true) || standard_true <= standard_false) {
            fprintf(
                stderr,
                "specialized toggle did not select the expected configuration alternatives: false %.9g (%a), true %.9g (%a)\n",
                standard_false,
                (double)standard_false,
                standard_true,
                standard_true
            );
            return 1;
        }
    }
    if (validate_multi_toggle) {
        size_t base = 2u * (size_t)configurations;
        if (total_systems < 3u || toggle_bit_count < 2u || configurations < 4u) {
            fprintf(stderr, "multi-toggle validation requires three systems and at least two toggle bits\n");
            return 1;
        }
        if (memcmp(scores + base, scores + base + 1u, sizeof(scores[0])) != 0 ||
            memcmp(scores + base + 2u, scores + base + 3u, sizeof(scores[0])) != 0 ||
            memcmp(scores + base, scores + base + 2u, sizeof(scores[0])) == 0
        ) {
            fprintf(stderr, "adjacent specialized toggle predicates selected stale configurations: %.9g %.9g %.9g %.9g\n",
                    scores[base], scores[base + 1u], scores[base + 2u], scores[base + 3u]);
            return 1;
        }
        multi_toggle_verified = 1;
    }
    if (o_scoring_pipeline_prespecialized_cubin(pipeline, &retained_cubin, &retained_cubin_size) != ODEZZA_SUCCESS || retained_cubin == NULL ||
        o_scoring_pipeline_cubin_inspection(pipeline, &cubin_inspection) != ODEZZA_SUCCESS || cubin_inspection == NULL ||
        o_scoring_pipeline_stats(pipeline, &stats) != ODEZZA_SUCCESS
    ) {
        fprintf(stderr, "internal benchmark diagnostics failed\n");
        return 1;
    }
    if (validate_runtime_widths) {
        static const uint32_t widths[] = {1u, 2u, 3u, 12u};
        OdezzaScoringRunReport sweep_report;
        size_t width_index;
        if (toggle_bit_count < 12u || toggle_test_bit != 0u) {
            fprintf(stderr, "runtime-width validation requires at least 12 active bits and toggle test bit zero\n");
            return 1;
        }
        for (width_index = 0u; width_index < sizeof(widths) / sizeof(widths[0]); ++width_index) {
            uint64_t expected = (uint64_t)total_systems * constant_bank_count * (UINT64_C(1) << widths[width_index]);
            launch.active_toggle_count = widths[width_index];
            result = odezza_scoring_pipeline_run(pipeline, systems, total_systems, &launch, workspace, workspace_size, &sweep_report);
            if (result != ODEZZA_SUCCESS || sweep_report.configuration_count != expected) {
                print_pipeline_error(pipeline, result);
                fprintf(stderr, "same-pipeline runtime toggle-width validation failed at %u bits\n", widths[width_index]);
                return 1;
            }
        }
        launch.active_toggle_count = toggle_bit_count;
        runtime_widths_verified = 1;
    }
    printf("{\n");
    printf("  \"schema\": \"odezza.public-bulk-pipeline-benchmark\",\n");
    printf("  \"context_contract\": \"caller keeps one context current for pipeline creation, every run, and destruction; the pipeline never manages context "
           "state\",\n");
    printf("  \"sm_version\": %u,\n", create_info.sm_version);
    printf("  \"cubin_bytes\": %zu,\n", retained_cubin_size);
    printf("  \"template_register_count\": %u,\n", cubin_inspection->register_count);
    printf("  \"submissions\": %u,\n", submissions);
    printf("  \"runs\": %u,\n", run_count);
    printf("  \"systems_per_module\": %u,\n", systems_per_module);
    printf("  \"state_capacity\": %u,\n", state_capacity);
    printf("  \"constant_capacity\": %u,\n", constant_capacity);
    printf("  \"toggle_bits\": %u,\n", toggle_bit_count);
    printf("  \"toggle_test_bit\": %u,\n", toggle_test_bit);
    printf("  \"shared_patch_capacity\": %u,\n", shared_patch_capacity);
    printf("  \"system_patch_capacity\": %u,\n", system_patch_capacity);
    printf("  \"total_systems\": %zu,\n", total_systems);
    printf("  \"configurations_per_system\": %llu,\n", (unsigned long long)configurations);
    printf("  \"toggle_permutations\": %llu,\n", (unsigned long long)(UINT64_C(1) << toggle_bit_count));
    printf("  \"constant_banks\": %u,\n", constant_bank_count);
    printf("  \"workspace_bytes\": %zu,\n", workspace_size);
    printf("  \"handle_create_seconds\": %.9g,\n", create_seconds);
    printf("  \"wall_seconds\": %.9g,\n", report.total_seconds);
    printf("  \"modules_per_second\": %.9g,\n", report.module_count / report.total_seconds);
    printf("  \"systems_per_second\": %.9g,\n", report.system_count / report.total_seconds);
    printf("  \"configurations_per_second\": %.9g,\n", report.configuration_count / report.total_seconds);
    printf("  \"correct_system_mse\": %.9g,\n", scores[0]);
    printf("  \"specialized_toggle_mapping_verified\": %s,\n", validate_toggle ? "true" : "false");
    printf("  \"same_pipeline_runtime_widths_verified\": %s,\n", runtime_widths_verified ? "true" : "false");
    printf("  \"multi_toggle_predicate_latency_verified\": %s,\n", multi_toggle_verified ? "true" : "false");
    printf("  \"constant_bank_pairing_verified\": %s,\n",
           constant_bank_count < 2u ? "null" : constant_bank_pairing_verified ? "true" : "false");
    printf("  \"mean_specialization_us\": %.9g,\n", 1.0e6 * stats.specialization_seconds / report.module_count);
    printf("  \"mean_module_load_us\": %.9g,\n", 1.0e6 * stats.module_load_seconds / report.module_count);
    printf("  \"mean_module_unload_us\": %.9g,\n", 1.0e6 * stats.module_unload_seconds / report.module_count);
    printf("  \"create_time_cuda_resource_us\": %.9g,\n", 1.0e6 * stats.cuda_resource_create_seconds);
    printf("  \"peak_inflight_modules\": %u,\n", stats.peak_inflight_module_count);
    printf("  \"timed_scope\": \"reset, C99 specialization, eager module load, lookup, launch and RK4 execution, event retirement, and module unload; "
           "streams/events already exist\"\n");
    printf("}\n");
    return 0;
}

int main(int argc, char **argv) {
    OBenchmarkResources resources;
    int exit_status;
    memset(&resources, 0, sizeof(resources));
    exit_status = o_benchmark_run(argc, argv, &resources);
    if (resources.pipeline != NULL) (void)odezza_scoring_pipeline_destroy(resources.pipeline);
    if (resources.context != NULL) {
        if (resources.scores_device != 0u) (void)cuMemFree(resources.scores_device);
        if (resources.reference_device != 0u) (void)cuMemFree(resources.reference_device);
        if (resources.trajectory_times_device != 0u) (void)cuMemFree(resources.trajectory_times_device);
        if (resources.trajectory_offsets_device != 0u) (void)cuMemFree(resources.trajectory_offsets_device);
        if (resources.constants_device != 0u) (void)cuMemFree(resources.constants_device);
        (void)cuCtxSetCurrent(NULL);
        (void)cuDevicePrimaryCtxRelease(resources.device);
    }
    if (resources.workspace != NULL) free(resources.workspace);
    if (resources.scores != NULL) free(resources.scores);
    if (resources.constants != NULL) free(resources.constants);
    if (resources.programs != NULL) free(resources.programs);
    if (resources.rhs != NULL) free(resources.rhs);
    if (resources.systems != NULL) free(resources.systems);
    return exit_status;
}
