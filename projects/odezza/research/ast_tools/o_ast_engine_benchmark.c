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

#include "o_ast_tools.h"
#include "odezza.h"
#include "o_odezza_internal.h"

#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define O_STATE_COUNT 4u
#define O_CONSTANT_COUNT 4u
#define O_TOGGLE_BIT_COUNT 5u
#define O_TRAJECTORY_COUNT 3u
#define O_OBSERVATION_COUNT 8u
#define O_TRAJECTORY_POINT_COUNT (O_TRAJECTORY_COUNT * (O_OBSERVATION_COUNT + 1u))
#define O_REFERENCE_FLOAT_COUNT (O_STATE_COUNT * O_TRAJECTORY_COUNT * (O_OBSERVATION_COUNT + 1u))
#define O_MAX_RUNS 32u

static const uint8_t o_fixed_rhs0[] = {
    O_AST_CONSTANT_F32, 0u,
    O_AST_STATE_F32, 0u,
    O_AST_MUL_F32,
    O_AST_RETURN_F32
};

static const uint8_t o_fixed_rhs1[] = {
    O_AST_CONSTANT_F32, 1u,
    O_AST_STATE_F32, 1u,
    O_AST_MUL_F32,
    O_AST_NEG_F32,
    O_AST_RETURN_F32
};

static const uint8_t o_fixed_rhs2[] = {
    O_AST_CONSTANT_F32, 2u,
    O_AST_STATE_F32, 0u,
    O_AST_MUL_F32,
    O_AST_CONSTANT_F32, 3u,
    O_AST_STATE_F32, 2u,
    O_AST_MUL_F32,
    O_AST_SUB_F32,
    O_AST_RETURN_F32
};

typedef struct OCorpusStorage {
    OdezzaScoringSystem *systems;
    OdezzaScoringRhs *rhs;
    uint8_t *programs;
    size_t system_capacity;
    size_t system_count;
    size_t rhs_per_system;
    size_t program_capacity;
    size_t program_bytes;
    size_t skip_record_count;
    size_t records_seen;
    size_t records_loaded;
} OCorpusStorage;

typedef struct OBenchmarkResources {
    CUdevice device;
    CUcontext context;
    CUdeviceptr constants_device;
    CUdeviceptr trajectory_offsets_device;
    CUdeviceptr trajectory_times_device;
    CUdeviceptr reference_device;
    CUdeviceptr scores_device;
    OdezzaScoringPipeline *pipeline;
    OCorpusStorage corpus;
    float *constants;
    float *scores;
    void *workspace;
} OBenchmarkResources;

typedef struct OLoadedKernelTiming {
    double minimum_seconds;
    double median_seconds;
    double mean_seconds;
    uint32_t register_count;
    uint32_t system_count;
    size_t body_instruction_count;
    size_t maximum_body_instruction_count;
} OLoadedKernelTiming;

static double o_monotonic_seconds(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
    return (double)value.tv_sec + (double)value.tv_nsec * 1.0e-9;
}

static int o_parse_u32(const char *text, uint32_t *value_ret) {
    char *end = NULL;
    unsigned long value;
    errno = 0;
    value = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value > UINT32_MAX) return 0;
    *value_ret = (uint32_t)value;
    return 1;
}

static int o_parse_size(const char *text, size_t *value_ret) {
    char *end = NULL;
    unsigned long long value;
    errno = 0;
    value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value > SIZE_MAX) return 0;
    *value_ret = (size_t)value;
    return 1;
}

static int o_multiply_size(size_t lhs, size_t rhs, size_t *value_ret) {
    if (rhs != 0u && lhs > SIZE_MAX / rhs) return 0;
    *value_ret = lhs * rhs;
    return 1;
}

static void o_usage(const char *program) {
    fprintf(
        stderr,
        "usage: %s --corpus FILE [--first-system N] [--system-count N] [--rhs-per-system 1..4] [--systems-per-module N] "
        "[--constant-banks N] [--workers N] [--slots-per-worker N] [--queue-capacity N] "
        "[--loaded-modules N] [--streams N] [--steps-per-observation N] [--runs N] "
        "[--spot-checks N] [--shared-patch N] [--system-patch N] [--device N]\n",
        program
    );
}

static void o_print_pipeline_error(OdezzaScoringPipeline *pipeline, OdezzaResult result) {
    char error[512];
    size_t bytes = 0u;
    if (pipeline != NULL &&
        odezza_scoring_pipeline_write_error(
            pipeline,
            error,
            sizeof(error),
            &bytes
        ) == ODEZZA_SUCCESS && bytes != 0u
    ) {
        fprintf(stderr, "Odezza failed with result %d: %s\n", (int)result, error);
    } else {
        fprintf(stderr, "Odezza failed with result %d\n", (int)result);
    }
}

static OAstToolResult o_copy_corpus_item(const OAstSpaceItem *item, void *context) {
    OCorpusStorage *storage = (OCorpusStorage *)context;
    OdezzaScoringRhs *rhs;
    size_t system_index;
    size_t rhs_index;
    if (storage->records_seen++ < storage->skip_record_count) return O_AST_TOOL_SUCCESS;
    if (storage->records_loaded == storage->system_capacity * storage->rhs_per_system) return O_AST_TOOL_SUCCESS;
    if (item->program.byte_count > storage->program_capacity - storage->program_bytes) return O_AST_TOOL_ERROR_CAPACITY;
    system_index = storage->records_loaded / storage->rhs_per_system;
    rhs_index = storage->records_loaded % storage->rhs_per_system;
    rhs = storage->rhs + storage->records_loaded;
    memcpy(storage->programs + storage->program_bytes, item->program.bytes, item->program.byte_count);
    rhs->state_index = (uint8_t)(O_STATE_COUNT - storage->rhs_per_system + rhs_index);
    rhs->program.bytes = storage->programs + storage->program_bytes;
    rhs->program.byte_count = item->program.byte_count;
    if (rhs_index == 0u) {
        storage->systems[system_index].rhs = rhs;
        storage->systems[system_index].rhs_count = storage->rhs_per_system;
    }
    storage->program_bytes += item->program.byte_count;
    ++storage->records_loaded;
    if (rhs_index + 1u == storage->rhs_per_system) ++storage->system_count;
    return O_AST_TOOL_SUCCESS;
}

static int o_load_corpus(
    const char *path,
    size_t first_system,
    size_t requested_system_count,
    size_t rhs_per_system,
    OCorpusStorage *storage,
    OAstFileReport *file_report_ret
) {
    OAstFileReport file_report;
    size_t program_capacity;
    size_t corpus_system_count;
    size_t selected_record_count;
    OAstToolResult result = o_ast_file_read(path, NULL, NULL, &file_report);
    if (result != O_AST_TOOL_SUCCESS) {
        fprintf(stderr, "corpus verification failed: %s\n", o_ast_tool_result_string(result));
        return 0;
    }
    if (rhs_per_system == 0u || rhs_per_system > O_STATE_COUNT || file_report.record_count % rhs_per_system != 0u) {
        fprintf(stderr, "corpus AST count is not divisible by the requested RHS count\n");
        return 0;
    }
    corpus_system_count = (size_t)file_report.record_count / rhs_per_system;
    if (first_system > corpus_system_count || requested_system_count == 0u ||
        requested_system_count > corpus_system_count - first_system
    ) {
        requested_system_count = corpus_system_count - first_system;
    }
    if (requested_system_count == 0u ||
        !o_multiply_size(requested_system_count, rhs_per_system, &selected_record_count) ||
        !o_multiply_size(selected_record_count, file_report.maximum_program_bytes, &program_capacity)
    ) {
        fprintf(stderr, "corpus selection is empty or overflows host sizes\n");
        return 0;
    }
    storage->systems = (OdezzaScoringSystem *)calloc(requested_system_count, sizeof(*storage->systems));
    storage->rhs = (OdezzaScoringRhs *)calloc(selected_record_count, sizeof(*storage->rhs));
    storage->programs = (uint8_t *)malloc(program_capacity);
    storage->system_capacity = requested_system_count;
    storage->rhs_per_system = rhs_per_system;
    storage->program_capacity = program_capacity;
    storage->skip_record_count = first_system * rhs_per_system;
    if (storage->systems == NULL || storage->rhs == NULL || storage->programs == NULL) {
        fprintf(stderr, "corpus storage allocation failed\n");
        return 0;
    }
    result = o_ast_file_read(path, o_copy_corpus_item, storage, NULL);
    if (result != O_AST_TOOL_SUCCESS || storage->system_count != requested_system_count) {
        fprintf(stderr, "corpus load failed: %s\n", o_ast_tool_result_string(result));
        return 0;
    }
    *file_report_ret = file_report;
    return 1;
}

static void o_reference_rhs(const float state[O_STATE_COUNT], const float constants[O_CONSTANT_COUNT], float rhs[O_STATE_COUNT]) {
    rhs[0] = constants[0] * state[0];
    rhs[1] = -(constants[1] * state[1]);
    rhs[2] = constants[2] * state[0] - constants[3] * state[2];
    rhs[3] = state[0] - state[3];
}

static void o_generate_reference(float reference[O_REFERENCE_FLOAT_COUNT], uint32_t steps_per_observation) {
    static const float initial[O_STATE_COUNT][O_TRAJECTORY_COUNT] = {
        {0.75f, 1.0f, 1.25f},
        {1.25f, 1.0f, 0.75f},
        {0.5f, 0.8f, 1.1f},
        {0.9f, 1.2f, 1.5f}
    };
    static const float constants[O_CONSTANT_COUNT] = {0.05f, 0.04f, 0.03f, 0.02f};
    const float h = 0.05f / (float)steps_per_observation;
    const float half_h = 0.5f * h;
    const float sixth_h = h / 6.0f;
    const size_t target_offset = O_STATE_COUNT * O_TRAJECTORY_COUNT;
    const size_t target_plane = O_TRAJECTORY_COUNT * O_OBSERVATION_COUNT;
    uint32_t state_index;
    uint32_t trajectory;
    memset(reference, 0, O_REFERENCE_FLOAT_COUNT * sizeof(*reference));
    for (state_index = 0u; state_index < O_STATE_COUNT; ++state_index) {
        for (trajectory = 0u; trajectory < O_TRAJECTORY_COUNT; ++trajectory) {
            reference[state_index * O_TRAJECTORY_COUNT + trajectory] = initial[state_index][trajectory];
        }
    }
    for (trajectory = 0u; trajectory < O_TRAJECTORY_COUNT; ++trajectory) {
        float state[O_STATE_COUNT];
        uint32_t observation;
        for (state_index = 0u; state_index < O_STATE_COUNT; ++state_index) state[state_index] = initial[state_index][trajectory];
        for (observation = 0u; observation < O_OBSERVATION_COUNT; ++observation) {
            uint32_t step;
            for (step = 0u; step < steps_per_observation; ++step) {
                float base[O_STATE_COUNT];
                float stage[O_STATE_COUNT];
                float sum[O_STATE_COUNT] = {0.0f, 0.0f, 0.0f, 0.0f};
                uint32_t rk_stage;
                memcpy(base, state, sizeof(base));
                memcpy(stage, state, sizeof(stage));
                for (rk_stage = 0u; rk_stage < 4u; ++rk_stage) {
                    float rhs[O_STATE_COUNT];
                    float weight = (rk_stage == 0u || rk_stage == 3u) ? 1.0f : 2.0f;
                    o_reference_rhs(stage, constants, rhs);
                    for (state_index = 0u; state_index < O_STATE_COUNT; ++state_index) {
                        sum[state_index] = fmaf(weight, rhs[state_index], sum[state_index]);
                    }
                    if (rk_stage != 3u) {
                        float stage_h = rk_stage == 2u ? h : half_h;
                        for (state_index = 0u; state_index < O_STATE_COUNT; ++state_index) {
                            stage[state_index] = fmaf(stage_h, rhs[state_index], base[state_index]);
                        }
                    }
                }
                for (state_index = 0u; state_index < O_STATE_COUNT; ++state_index) {
                    state[state_index] = fmaf(sixth_h, sum[state_index], base[state_index]);
                }
            }
            for (state_index = 0u; state_index < O_STATE_COUNT; ++state_index) {
                reference[target_offset + state_index * target_plane + trajectory * O_OBSERVATION_COUNT + observation] = state[state_index];
            }
        }
    }
}

static int o_evaluate_rhs(
    const OAstProgramView fixed[3],
    size_t fixed_count,
    const OdezzaScoringSystem *system,
    const float state[O_STATE_COUNT],
    const float constants[O_CONSTANT_COUNT],
    uint32_t permutation,
    float rhs[O_STATE_COUNT]
) {
    size_t index;
    for (index = 0u; index < fixed_count; ++index) {
        if (o_ast_cpu_evaluate(
                fixed[index],
                state,
                O_STATE_COUNT,
                constants,
                O_CONSTANT_COUNT,
                O_TOGGLE_BIT_COUNT,
                permutation,
                rhs + index
            ) != O_AST_TOOL_SUCCESS
        ) {
            return 0;
        }
    }
    for (index = 0u; index < system->rhs_count; ++index) {
        OAstProgramView program = {
            system->rhs[index].program.bytes,
            system->rhs[index].program.byte_count
        };
        if (o_ast_cpu_evaluate(
                program,
                state,
                O_STATE_COUNT,
                constants,
                O_CONSTANT_COUNT,
                O_TOGGLE_BIT_COUNT,
                permutation,
                rhs + system->rhs[index].state_index
            ) != O_AST_TOOL_SUCCESS
        ) {
            return 0;
        }
    }
    return 1;
}

static float o_cpu_score(
    const OdezzaScoringSystem *system,
    size_t fixed_count,
    const float constants[O_CONSTANT_COUNT],
    uint32_t permutation,
    const float reference[O_REFERENCE_FLOAT_COUNT],
    uint32_t steps_per_observation
) {
    static const OAstProgramView fixed[3] = {
        {o_fixed_rhs0, sizeof(o_fixed_rhs0)},
        {o_fixed_rhs1, sizeof(o_fixed_rhs1)},
        {o_fixed_rhs2, sizeof(o_fixed_rhs2)}
    };
    const float h = 0.05f / (float)steps_per_observation;
    const float half_h = 0.5f * h;
    const float sixth_h = h / 6.0f;
    const size_t target_offset = O_STATE_COUNT * O_TRAJECTORY_COUNT;
    const size_t target_plane = O_TRAJECTORY_COUNT * O_OBSERVATION_COUNT;
    float squared_error = 0.0f;
    int valid = 1;
    uint32_t trajectory;
    for (trajectory = 0u; trajectory < O_TRAJECTORY_COUNT; ++trajectory) {
        float state[O_STATE_COUNT];
        uint32_t state_index;
        uint32_t observation;
        for (state_index = 0u; state_index < O_STATE_COUNT; ++state_index) {
            state[state_index] = reference[state_index * O_TRAJECTORY_COUNT + trajectory];
        }
        for (observation = 0u; observation < O_OBSERVATION_COUNT; ++observation) {
            uint32_t step;
            for (step = 0u; step < steps_per_observation; ++step) {
                float base[O_STATE_COUNT];
                float stage[O_STATE_COUNT];
                float sum[O_STATE_COUNT] = {0.0f, 0.0f, 0.0f, 0.0f};
                uint32_t rk_stage;
                memcpy(base, state, sizeof(base));
                memcpy(stage, state, sizeof(stage));
                for (rk_stage = 0u; rk_stage < 4u; ++rk_stage) {
                    float rhs[O_STATE_COUNT];
                    float weight = (rk_stage == 0u || rk_stage == 3u) ? 1.0f : 2.0f;
                    if (!o_evaluate_rhs(fixed, fixed_count, system, stage, constants, permutation, rhs)) return FLT_MAX;
                    for (state_index = 0u; state_index < O_STATE_COUNT; ++state_index) {
                        sum[state_index] = fmaf(weight, rhs[state_index], sum[state_index]);
                    }
                    if (rk_stage != 3u) {
                        float stage_h = rk_stage == 2u ? h : half_h;
                        for (state_index = 0u; state_index < O_STATE_COUNT; ++state_index) {
                            stage[state_index] = fmaf(stage_h, rhs[state_index], base[state_index]);
                        }
                    }
                }
                for (state_index = 0u; state_index < O_STATE_COUNT; ++state_index) {
                    state[state_index] = fmaf(sixth_h, sum[state_index], base[state_index]);
                }
            }
            for (state_index = 0u; state_index < O_STATE_COUNT; ++state_index) valid = valid && isfinite(state[state_index]);
            for (state_index = 0u; state_index < O_STATE_COUNT; ++state_index) {
                size_t target = target_offset + state_index * target_plane + trajectory * O_OBSERVATION_COUNT + observation;
                float error = state[state_index] - reference[target];
                squared_error = fmaf(error, error, squared_error);
            }
        }
    }
    return valid && isfinite(squared_error)
        ? squared_error / (float)(O_STATE_COUNT * O_TRAJECTORY_COUNT * O_OBSERVATION_COUNT)
        : FLT_MAX;
}

static int o_program_uses_approximate_device_math(OAstProgramView program) {
    size_t offset = 0u;
    while (offset < program.byte_count) {
        uint8_t opcode = program.bytes[offset++];
        switch ((OAstToolOpcode)opcode) {
        case O_AST_STATE_F32:
        case O_AST_CONSTANT_F32:
        case O_AST_TOGGLE2_F32:
            offset += 1u;
            break;
        case O_AST_LITERAL_F32:
            offset += 4u;
            break;
        case O_AST_TOGGLE4_F32:
            offset += 2u;
            break;
        case O_AST_DIV_F32:
        case O_AST_SQRT_F32:
        case O_AST_RCP_F32:
        case O_AST_SIN_F32:
        case O_AST_COS_F32:
        case O_AST_EX2_F32:
        case O_AST_LG2_F32:
        case O_AST_RSQRT_F32:
        case O_AST_TANH_F32:
        case O_AST_EXP_F32:
        case O_AST_LOG_F32:
            return 1;
        default:
            break;
        }
    }
    return 0;
}

static int o_system_uses_approximate_device_math(const OdezzaScoringSystem *system) {
    size_t index;
    for (index = 0u; index < system->rhs_count; ++index) {
        OAstProgramView program = {
            system->rhs[index].program.bytes,
            system->rhs[index].program.byte_count
        };
        if (o_program_uses_approximate_device_math(program)) return 1;
    }
    return 0;
}

static int o_spot_verify(
    const OCorpusStorage *corpus,
    const float *constants,
    uint32_t constant_bank_count,
    const float reference[O_REFERENCE_FLOAT_COUNT],
    uint32_t steps_per_observation,
    const float *gpu_scores,
    uint32_t requested_checks,
    uint32_t *checked_ret,
    uint32_t *invalid_ret,
    uint32_t *approximate_skipped_ret,
    float *maximum_relative_error_ret,
    double *seconds_ret
) {
    uint64_t configurations = (uint64_t)constant_bank_count << O_TOGGLE_BIT_COUNT;
    uint64_t candidate_count = (uint64_t)corpus->system_count * configurations;
    size_t fixed_count = O_STATE_COUNT - corpus->rhs_per_system;
    uint32_t checked = 0u;
    uint32_t invalid = 0u;
    uint32_t approximate_skipped = 0u;
    float maximum_relative_error = 0.0f;
    double started = o_monotonic_seconds();
    uint64_t candidate_index;
    for (candidate_index = 0u; candidate_index < candidate_count && checked < requested_checks; ++candidate_index) {
        size_t system_index = candidate_index == 0u
            ? 0u
            : (size_t)((candidate_index * UINT64_C(2654435761)) % corpus->system_count);
        uint64_t configuration = candidate_index == 0u
            ? 0u
            : (candidate_index * UINT64_C(2246822519)) % configurations;
        uint32_t permutation = (uint32_t)(configuration & ((UINT64_C(1) << O_TOGGLE_BIT_COUNT) - 1u));
        uint32_t constant_bank = (uint32_t)(configuration >> O_TOGGLE_BIT_COUNT);
        const float *constant_values = constants + ((system_index * constant_bank_count + constant_bank) * O_CONSTANT_COUNT);
        const OdezzaScoringSystem *system = corpus->systems + system_index;
        if (o_system_uses_approximate_device_math(system)) {
            ++approximate_skipped;
            continue;
        }
        float cpu = o_cpu_score(
            system,
            fixed_count,
            constant_values,
            permutation,
            reference,
            steps_per_observation
        );
        float gpu = gpu_scores[system_index * configurations + configuration];
        if (cpu == FLT_MAX || gpu == FLT_MAX) {
            if (cpu != gpu) {
                fprintf(stderr, "CPU/GPU invalid mismatch at system %zu configuration %llu: cpu=%.9g gpu=%.9g\n",
                        system_index, (unsigned long long)configuration, cpu, gpu);
                return 0;
            }
            ++invalid;
        } else {
            float absolute_error = fabsf(cpu - gpu);
            float relative_error = absolute_error / fmaxf(fmaxf(fabsf(cpu), fabsf(gpu)), 1.0e-12f);
            float tolerance = 2.0e-5f + 2.0e-4f * fabsf(cpu);
            if (maximum_relative_error < relative_error) maximum_relative_error = relative_error;
            if (!isfinite(cpu) || !isfinite(gpu) || absolute_error > tolerance) {
                uint64_t nearest_cpu_configuration = 0u;
                uint64_t nearest_gpu_configuration = 0u;
                float nearest_cpu_error = FLT_MAX;
                float nearest_gpu_error = FLT_MAX;
                fprintf(stderr, "CPU/GPU score mismatch at system %zu configuration %llu: cpu=%.9g gpu=%.9g abs=%.9g tolerance=%.9g\n",
                        system_index, (unsigned long long)configuration, cpu, gpu, absolute_error, tolerance);
                fprintf(stderr, "programs=");
                for (size_t rhs_index = 0u; rhs_index < system->rhs_count; ++rhs_index) {
                    if (rhs_index != 0u) fputc(',', stderr);
                    for (size_t byte = 0u; byte < system->rhs[rhs_index].program.byte_count; ++byte) {
                        fprintf(stderr, "%02x", system->rhs[rhs_index].program.bytes[byte]);
                    }
                }
                fprintf(stderr, " permutation=%u constant_bank=%u\n", permutation, constant_bank);
                for (uint64_t probe = 0u; probe < configurations; ++probe) {
                    uint32_t probe_permutation = (uint32_t)(probe & ((UINT64_C(1) << O_TOGGLE_BIT_COUNT) - 1u));
                    uint32_t probe_bank = (uint32_t)(probe >> O_TOGGLE_BIT_COUNT);
                    const float *probe_constants = constants + ((system_index * constant_bank_count + probe_bank) * O_CONSTANT_COUNT);
                    float probe_cpu = o_cpu_score(system, fixed_count, probe_constants, probe_permutation, reference, steps_per_observation);
                    float cpu_error = fabsf(probe_cpu - gpu);
                    float gpu_error = fabsf(gpu_scores[system_index * configurations + probe] - cpu);
                    if (cpu_error < nearest_cpu_error) {
                        nearest_cpu_error = cpu_error;
                        nearest_cpu_configuration = probe;
                    }
                    if (gpu_error < nearest_gpu_error) {
                        nearest_gpu_error = gpu_error;
                        nearest_gpu_configuration = probe;
                    }
                }
                fprintf(stderr, "nearest_cpu_configuration=%llu error=%.9g nearest_gpu_configuration=%llu error=%.9g\n",
                        (unsigned long long)nearest_cpu_configuration, nearest_cpu_error,
                        (unsigned long long)nearest_gpu_configuration, nearest_gpu_error);
                if (corpus->system_count == 1u) {
                    uint32_t mismatch_count = 0u;
                    fprintf(stderr, "mismatching configurations:");
                    for (uint64_t probe = 0u; probe < configurations; ++probe) {
                        uint32_t probe_permutation = (uint32_t)(probe & ((UINT64_C(1) << O_TOGGLE_BIT_COUNT) - 1u));
                        uint32_t probe_bank = (uint32_t)(probe >> O_TOGGLE_BIT_COUNT);
                        const float *probe_constants = constants + probe_bank * O_CONSTANT_COUNT;
                        float probe_cpu = o_cpu_score(system, fixed_count, probe_constants, probe_permutation, reference, steps_per_observation);
                        float probe_gpu = gpu_scores[probe];
                        float probe_tolerance = 2.0e-5f + 2.0e-4f * fabsf(probe_cpu);
                        if (!isfinite(probe_cpu) || !isfinite(probe_gpu) || fabsf(probe_cpu - probe_gpu) > probe_tolerance) {
                            if (mismatch_count < 64u) fprintf(stderr, " %llu", (unsigned long long)probe);
                            ++mismatch_count;
                        }
                    }
                    fprintf(stderr, " (total %u/%llu)\n", mismatch_count, (unsigned long long)configurations);
                }
                return 0;
            }
        }
        ++checked;
    }
    if (checked != requested_checks && checked != candidate_count) {
        fprintf(stderr, "not enough exact-operation candidates for CPU spot verification\n");
        return 0;
    }
    *checked_ret = checked;
    *invalid_ret = invalid;
    *approximate_skipped_ret = approximate_skipped;
    *maximum_relative_error_ret = maximum_relative_error;
    *seconds_ret = o_monotonic_seconds() - started;
    return 1;
}

static int o_compare_double(const void *lhs, const void *rhs) {
    double left = *(const double *)lhs;
    double right = *(const double *)rhs;
    return left < right ? -1 : left > right;
}

static int o_measure_loaded_kernel(
    OBenchmarkResources *resources,
    const OdezzaScoringLaunch *launch,
    uint32_t run_count,
    OLoadedKernelTiming *timing_ret
) {
    const OdezzaScoringCubinInspection *inspection = NULL;
    const void *prespecialized_cubin = NULL;
    OdezzaScoringPrespecialization prespecialization;
    OdezzaScoringSpecializationReport specialization_report;
    unsigned char *specialized_cubin = NULL;
    void *specialization_workspace = NULL;
    size_t cubin_size = 0u;
    size_t specialization_workspace_size = 0u;
    CUmodule module = NULL;
    CUfunction function = NULL;
    CUstream stream = NULL;
    CUevent started = NULL;
    CUevent completed = NULL;
    CUresult cuda_result;
    uint64_t configuration_count;
    uint64_t constant_banks_device = launch->constant_banks_device;
    uint32_t constant_bank_count = launch->constant_bank_count;
    uint64_t trajectory_offsets_device = launch->trajectory_offsets_device;
    uint64_t trajectory_times_device = launch->trajectory_times_device;
    uint64_t reference_data_device = launch->reference_data_device;
    uint32_t trajectory_count = launch->trajectory_count;
    uint32_t trajectory_point_count = launch->trajectory_point_count;
    uint32_t active_state_count = O_STATE_COUNT;
    uint32_t active_constant_count = O_CONSTANT_COUNT;
    uint32_t active_toggle_count = launch->active_toggle_count;
    uint32_t active_system_count;
    uint32_t steps_per_observation = launch->steps_per_observation;
    uint64_t mse_output_device = launch->mse_output_device;
    uint32_t grid_x;
    uint32_t shared_memory_bytes;
    void *arguments[13];
    double seconds[O_MAX_RUNS];
    double mean_seconds = 0.0;
    int success = 0;

    if (run_count == 0u || run_count > O_MAX_RUNS || resources->corpus.system_count > UINT32_MAX) return 0;
    if (o_scoring_pipeline_cubin_inspection(resources->pipeline, &inspection) != ODEZZA_SUCCESS ||
        o_scoring_pipeline_prespecialized_cubin(resources->pipeline, &prespecialized_cubin, &cubin_size) != ODEZZA_SUCCESS ||
        o_scoring_pipeline_prespecialization(resources->pipeline, &prespecialization) != ODEZZA_SUCCESS ||
        odezza_scoring_specialization_workspace_size(inspection, &specialization_workspace_size) != ODEZZA_SUCCESS
    ) {
        fprintf(stderr, "loaded-kernel metadata query failed\n");
        return 0;
    }
    active_system_count = resources->corpus.system_count < inspection->system_capacity
        ? (uint32_t)resources->corpus.system_count
        : (uint32_t)inspection->system_capacity;
    specialized_cubin = (unsigned char *)malloc(cubin_size);
    specialization_workspace = malloc(specialization_workspace_size == 0u ? 1u : specialization_workspace_size);
    if (specialized_cubin == NULL || specialization_workspace == NULL) {
        fprintf(stderr, "loaded-kernel host allocation failed\n");
    } else {
        memcpy(specialized_cubin, prespecialized_cubin, cubin_size);
        if (odezza_specialize_scoring_cubin_systems(
                specialized_cubin,
                cubin_size,
                inspection,
                &prespecialization,
                active_toggle_count,
                resources->corpus.systems,
                active_system_count,
                specialization_workspace,
                specialization_workspace_size,
                &specialization_report
            ) != ODEZZA_SUCCESS) {
            fprintf(stderr, "loaded-kernel specialization failed\n");
        } else if (cuModuleLoadData(&module, specialized_cubin) != CUDA_SUCCESS ||
            cuModuleGetFunction(&function, module, "odezza_scoring") != CUDA_SUCCESS ||
            cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS ||
            cuEventCreate(&started, CU_EVENT_DEFAULT) != CUDA_SUCCESS ||
            cuEventCreate(&completed, CU_EVENT_DEFAULT) != CUDA_SUCCESS
        ) {
            fprintf(stderr, "loaded-kernel CUDA setup failed\n");
        } else {
            configuration_count = (uint64_t)constant_bank_count << active_toggle_count;
            grid_x = (uint32_t)((configuration_count + 127u) / 128u);
            shared_memory_bytes = (uint32_t)(
                (O_STATE_COUNT * trajectory_point_count + trajectory_point_count + trajectory_count + 1u) * sizeof(uint32_t)
            );
            arguments[0] = &constant_banks_device;
            arguments[1] = &constant_bank_count;
            arguments[2] = &trajectory_offsets_device;
            arguments[3] = &trajectory_times_device;
            arguments[4] = &reference_data_device;
            arguments[5] = &trajectory_count;
            arguments[6] = &trajectory_point_count;
            arguments[7] = &active_state_count;
            arguments[8] = &active_constant_count;
            arguments[9] = &active_toggle_count;
            arguments[10] = &active_system_count;
            arguments[11] = &steps_per_observation;
            arguments[12] = &mse_output_device;
            cuda_result = cuLaunchKernel(
                function,
                grid_x,
                active_system_count,
                1u,
                128u,
                1u,
                1u,
                shared_memory_bytes,
                stream,
                arguments,
                NULL
            );
            if (cuda_result == CUDA_SUCCESS) cuda_result = cuStreamSynchronize(stream);
            for (uint32_t run = 0u; cuda_result == CUDA_SUCCESS && run < run_count; ++run) {
                float elapsed_ms = 0.0f;
                cuda_result = cuEventRecord(started, stream);
                if (cuda_result == CUDA_SUCCESS) {
                    cuda_result = cuLaunchKernel(
                        function,
                        grid_x,
                        active_system_count,
                        1u,
                        128u,
                        1u,
                        1u,
                        shared_memory_bytes,
                        stream,
                        arguments,
                        NULL
                    );
                }
                if (cuda_result == CUDA_SUCCESS) cuda_result = cuEventRecord(completed, stream);
                if (cuda_result == CUDA_SUCCESS) cuda_result = cuEventSynchronize(completed);
                if (cuda_result == CUDA_SUCCESS) cuda_result = cuEventElapsedTime(&elapsed_ms, started, completed);
                seconds[run] = (double)elapsed_ms * 1.0e-3;
                mean_seconds += seconds[run];
            }
            if (cuda_result != CUDA_SUCCESS) {
                fprintf(stderr, "loaded-kernel timed launch failed with CUDA result %d\n", (int)cuda_result);
            } else {
                qsort(seconds, run_count, sizeof(seconds[0]), o_compare_double);
                timing_ret->minimum_seconds = seconds[0];
                timing_ret->median_seconds = run_count & 1u
                    ? seconds[run_count / 2u]
                    : 0.5 * (seconds[run_count / 2u - 1u] + seconds[run_count / 2u]);
                timing_ret->mean_seconds = mean_seconds / run_count;
                timing_ret->register_count = specialization_report.register_count;
                timing_ret->system_count = active_system_count;
                timing_ret->body_instruction_count = specialization_report.body_instruction_count;
                timing_ret->maximum_body_instruction_count = specialization_report.maximum_body_instruction_count;
                success = 1;
            }
        }
    }
    if (completed != NULL) (void)cuEventDestroy(completed);
    if (started != NULL) (void)cuEventDestroy(started);
    if (stream != NULL) (void)cuStreamDestroy(stream);
    if (module != NULL) (void)cuModuleUnload(module);
    if (specialization_workspace != NULL) free(specialization_workspace);
    if (specialized_cubin != NULL) free(specialized_cubin);
    return success;
}

static int o_benchmark_run(int argc, char **argv, OBenchmarkResources *resources) {
    const char *corpus_path = NULL;
    size_t first_system = 0u;
    size_t requested_system_count = 65536u;
    uint32_t rhs_per_system = 1u;
    uint32_t systems_per_module = 4u;
    uint32_t constant_bank_count = 1u;
    uint32_t worker_count = 1u;
    uint32_t slots_per_worker = 2u;
    uint32_t queue_capacity = 256u;
    uint32_t loaded_modules = 2u;
    uint32_t streams = 2u;
    uint32_t steps_per_observation = 2u;
    uint32_t run_count = 3u;
    uint32_t spot_checks = 64u;
    uint32_t shared_patch_capacity = 128u;
    uint32_t system_patch_capacity = 128u;
    uint32_t device_ordinal = 0u;
    OAstFileReport corpus_report;
    OdezzaScoringRhs fixed_rhs[3];
    OdezzaScoringPipelineCreateInfo create_info;
    OdezzaScoringPipelineTuning tuning;
    OdezzaScoringLaunch launch;
    OdezzaScoringPipelineStats stats;
    OLoadedKernelTiming loaded_timing;
    const OdezzaScoringCubinInspection *inspection = NULL;
    const void *retained_cubin = NULL;
    size_t retained_cubin_size = 0u;
    float reference[O_REFERENCE_FLOAT_COUNT];
    float packed_reference[O_REFERENCE_FLOAT_COUNT];
    float trajectory_times[O_TRAJECTORY_POINT_COUNT];
    uint32_t trajectory_offsets[O_TRAJECTORY_COUNT + 1u];
    double run_seconds[O_MAX_RUNS];
    size_t constant_float_count;
    size_t score_count;
    size_t workspace_size;
    size_t workspace_alignment;
    size_t module_count;
    uint64_t configurations;
    CUdevice device = 0;
    CUcontext context = NULL;
    int compute_major = 0;
    int compute_minor = 0;
    uint32_t argument;
    uint32_t run_index;
    uint32_t checked;
    uint32_t invalid;
    uint32_t approximate_skipped;
    float maximum_relative_error;
    double create_seconds;
    double cpu_check_seconds;
    double minimum_seconds;
    double mean_seconds = 0.0;
    double median_seconds;
    OdezzaResult result;

    for (argument = 1u; argument < (uint32_t)argc; ++argument) {
        const char *option = argv[argument];
        uint32_t *u32_target = NULL;
        if (strcmp(option, "--help") == 0) {
            o_usage(argv[0]);
            return 0;
        }
        if (argument + 1u >= (uint32_t)argc) {
            o_usage(argv[0]);
            return 2;
        }
        ++argument;
        if (strcmp(option, "--corpus") == 0) {
            corpus_path = argv[argument];
            continue;
        }
        if (strcmp(option, "--system-count") == 0) {
            if (!o_parse_size(argv[argument], &requested_system_count)) return 2;
            continue;
        }
        if (strcmp(option, "--first-system") == 0) {
            if (!o_parse_size(argv[argument], &first_system)) return 2;
            continue;
        }
        if (strcmp(option, "--rhs-per-system") == 0) u32_target = &rhs_per_system;
        else if (strcmp(option, "--systems-per-module") == 0) u32_target = &systems_per_module;
        else if (strcmp(option, "--constant-banks") == 0) u32_target = &constant_bank_count;
        else if (strcmp(option, "--workers") == 0) u32_target = &worker_count;
        else if (strcmp(option, "--slots-per-worker") == 0) u32_target = &slots_per_worker;
        else if (strcmp(option, "--queue-capacity") == 0) u32_target = &queue_capacity;
        else if (strcmp(option, "--loaded-modules") == 0) u32_target = &loaded_modules;
        else if (strcmp(option, "--streams") == 0) u32_target = &streams;
        else if (strcmp(option, "--steps-per-observation") == 0) u32_target = &steps_per_observation;
        else if (strcmp(option, "--runs") == 0) u32_target = &run_count;
        else if (strcmp(option, "--spot-checks") == 0) u32_target = &spot_checks;
        else if (strcmp(option, "--shared-patch") == 0) u32_target = &shared_patch_capacity;
        else if (strcmp(option, "--system-patch") == 0) u32_target = &system_patch_capacity;
        else if (strcmp(option, "--device") == 0) u32_target = &device_ordinal;
        else {
            o_usage(argv[0]);
            return 2;
        }
        if (!o_parse_u32(argv[argument], u32_target) || *u32_target == 0u) return 2;
    }
    if (corpus_path == NULL || run_count > O_MAX_RUNS || requested_system_count == 0u || rhs_per_system > O_STATE_COUNT) {
        o_usage(argv[0]);
        return 2;
    }
    if (!o_load_corpus(corpus_path, first_system, requested_system_count, rhs_per_system, &resources->corpus, &corpus_report)) return 1;
    configurations = (uint64_t)constant_bank_count << O_TOGGLE_BIT_COUNT;
    if (configurations > SIZE_MAX ||
        !o_multiply_size(resources->corpus.system_count, constant_bank_count, &constant_float_count) ||
        !o_multiply_size(constant_float_count, O_CONSTANT_COUNT, &constant_float_count) ||
        !o_multiply_size(resources->corpus.system_count, (size_t)configurations, &score_count)
    ) {
        fprintf(stderr, "benchmark dimensions overflow host sizes\n");
        return 1;
    }
    resources->constants = (float *)malloc(constant_float_count * sizeof(*resources->constants));
    resources->scores = (float *)malloc(score_count * sizeof(*resources->scores));
    if (resources->constants == NULL || resources->scores == NULL) {
        fprintf(stderr, "benchmark host allocation failed\n");
        return 1;
    }
    for (size_t system_index = 0u; system_index < resources->corpus.system_count; ++system_index) {
        for (uint32_t bank = 0u; bank < constant_bank_count; ++bank) {
            float *values = resources->constants + ((system_index * constant_bank_count + bank) * O_CONSTANT_COUNT);
            float perturbation = 1.0e-5f * (float)(system_index % 17u) + 1.0e-4f * (float)bank;
            values[0] = 0.05f + perturbation;
            values[1] = 0.04f + 0.5f * perturbation;
            values[2] = 0.03f - 0.25f * perturbation;
            values[3] = 0.02f + 0.75f * perturbation;
        }
    }
    o_generate_reference(reference, steps_per_observation);
    for (uint32_t trajectory = 0u; trajectory < O_TRAJECTORY_COUNT; ++trajectory) {
        uint32_t point_begin = trajectory * (O_OBSERVATION_COUNT + 1u);
        trajectory_offsets[trajectory] = point_begin;
        trajectory_times[point_begin] = 0.0f;
        for (uint32_t state_index = 0u; state_index < O_STATE_COUNT; ++state_index) {
            packed_reference[state_index * O_TRAJECTORY_POINT_COUNT + point_begin] =
                reference[state_index * O_TRAJECTORY_COUNT + trajectory];
        }
        for (uint32_t observation = 0u; observation < O_OBSERVATION_COUNT; ++observation) {
            uint32_t point = point_begin + observation + 1u;
            trajectory_times[point] = 0.05f * (float)(observation + 1u);
            for (uint32_t state_index = 0u; state_index < O_STATE_COUNT; ++state_index) {
                packed_reference[state_index * O_TRAJECTORY_POINT_COUNT + point] =
                    reference[O_STATE_COUNT * O_TRAJECTORY_COUNT +
                              state_index * O_TRAJECTORY_COUNT * O_OBSERVATION_COUNT +
                              trajectory * O_OBSERVATION_COUNT + observation];
            }
        }
    }
    trajectory_offsets[O_TRAJECTORY_COUNT] = O_TRAJECTORY_POINT_COUNT;
    if (setenv("CUDA_MODULE_LOADING", "EAGER", 1) != 0 || cuInit(0u) != CUDA_SUCCESS ||
        cuDeviceGet(&device, (int)device_ordinal) != CUDA_SUCCESS ||
        cuDeviceGetAttribute(&compute_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device) != CUDA_SUCCESS ||
        cuDeviceGetAttribute(&compute_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device) != CUDA_SUCCESS ||
        cuDevicePrimaryCtxRetain(&context, device) != CUDA_SUCCESS || cuCtxSetCurrent(context) != CUDA_SUCCESS
    ) {
        resources->device = device;
        resources->context = context;
        fprintf(stderr, "benchmark context owner could not activate the CUDA device\n");
        return 1;
    }
    resources->device = device;
    resources->context = context;
    fixed_rhs[0] = (OdezzaScoringRhs){0u, {o_fixed_rhs0, sizeof(o_fixed_rhs0)}};
    fixed_rhs[1] = (OdezzaScoringRhs){1u, {o_fixed_rhs1, sizeof(o_fixed_rhs1)}};
    fixed_rhs[2] = (OdezzaScoringRhs){2u, {o_fixed_rhs2, sizeof(o_fixed_rhs2)}};
    memset(&create_info, 0, sizeof(create_info));
    create_info.sm_version = (uint32_t)(compute_major * 10 + compute_minor);
    create_info.state_count = O_STATE_COUNT;
    create_info.state_capacity = O_STATE_COUNT;
    create_info.constant_count = O_CONSTANT_COUNT;
    create_info.constant_capacity = O_CONSTANT_COUNT;
    create_info.fixed_rhs = fixed_rhs;
    create_info.fixed_rhs_count = O_STATE_COUNT - rhs_per_system;
    create_info.system_capacity = systems_per_module;
    create_info.shared_patch_capacity = shared_patch_capacity;
    create_info.system_patch_capacity = system_patch_capacity;
    create_info.worker_count = worker_count;
    create_info.cubin_slots_per_worker = slots_per_worker;
    memset(&tuning, 0, sizeof(tuning));
    tuning.queue_capacity = queue_capacity;
    tuning.maximum_loaded_modules = loaded_modules;
    tuning.execution_stream_count = streams;
    {
        double started = o_monotonic_seconds();
        result = o_scoring_pipeline_create_with_tuning(&create_info, &tuning, &resources->pipeline);
        create_seconds = o_monotonic_seconds() - started;
    }
    if (result != ODEZZA_SUCCESS) {
        o_print_pipeline_error(resources->pipeline, result);
        return 1;
    }
    result = odezza_scoring_pipeline_workspace_requirements(
        resources->pipeline,
        &workspace_size,
        &workspace_alignment
    );
    if (result != ODEZZA_SUCCESS) {
        o_print_pipeline_error(resources->pipeline, result);
        return 1;
    }
    resources->workspace = malloc(workspace_size);
    if (resources->workspace == NULL || (uintptr_t)resources->workspace % workspace_alignment != 0u) {
        fprintf(stderr, "benchmark workspace allocation failed\n");
        return 1;
    }
    if (cuMemAlloc(&resources->constants_device, constant_float_count * sizeof(*resources->constants)) != CUDA_SUCCESS ||
        cuMemAlloc(&resources->trajectory_offsets_device, sizeof(trajectory_offsets)) != CUDA_SUCCESS ||
        cuMemAlloc(&resources->trajectory_times_device, sizeof(trajectory_times)) != CUDA_SUCCESS ||
        cuMemAlloc(&resources->reference_device, sizeof(reference)) != CUDA_SUCCESS ||
        cuMemAlloc(&resources->scores_device, score_count * sizeof(*resources->scores)) != CUDA_SUCCESS ||
        cuMemcpyHtoD(
            resources->constants_device,
            resources->constants,
            constant_float_count * sizeof(*resources->constants)
        ) != CUDA_SUCCESS ||
        cuMemcpyHtoD(resources->trajectory_offsets_device, trajectory_offsets, sizeof(trajectory_offsets)) != CUDA_SUCCESS ||
        cuMemcpyHtoD(resources->trajectory_times_device, trajectory_times, sizeof(trajectory_times)) != CUDA_SUCCESS ||
        cuMemcpyHtoD(resources->reference_device, packed_reference, sizeof(packed_reference)) != CUDA_SUCCESS
    ) {
        fprintf(stderr, "benchmark CUDA memory setup failed\n");
        return 1;
    }
    memset(&launch, 0, sizeof(launch));
    launch.constant_banks_device = resources->constants_device;
    launch.constant_bank_count = constant_bank_count;
    launch.active_toggle_count = O_TOGGLE_BIT_COUNT;
    launch.trajectory_offsets_device = resources->trajectory_offsets_device;
    launch.trajectory_times_device = resources->trajectory_times_device;
    launch.reference_data_device = resources->reference_device;
    launch.trajectory_count = O_TRAJECTORY_COUNT;
    launch.trajectory_point_count = O_TRAJECTORY_POINT_COUNT;
    launch.steps_per_observation = steps_per_observation;
    launch.mse_output_device = resources->scores_device;
    for (run_index = 0u; run_index < run_count; ++run_index) {
        OdezzaScoringRunReport run_report;
        result = odezza_scoring_pipeline_run(
            resources->pipeline,
            resources->corpus.systems,
            resources->corpus.system_count,
            &launch,
            resources->workspace,
            workspace_size,
            &run_report
        );
        if (result != ODEZZA_SUCCESS) {
            o_print_pipeline_error(resources->pipeline, result);
            return 1;
        }
        if (run_report.system_count != resources->corpus.system_count ||
            run_report.configuration_count != resources->corpus.system_count * configurations
        ) {
            fprintf(stderr, "pipeline report does not match requested work\n");
            return 1;
        }
        run_seconds[run_index] = run_report.total_seconds;
        mean_seconds += run_report.total_seconds;
    }
    mean_seconds /= run_count;
    qsort(run_seconds, run_count, sizeof(run_seconds[0]), o_compare_double);
    minimum_seconds = run_seconds[0];
    median_seconds = run_count & 1u
        ? run_seconds[run_count / 2u]
        : 0.5 * (run_seconds[run_count / 2u - 1u] + run_seconds[run_count / 2u]);
    module_count = (resources->corpus.system_count + systems_per_module - 1u) / systems_per_module;
    if (cuMemcpyDtoH(resources->scores, resources->scores_device, score_count * sizeof(*resources->scores)) != CUDA_SUCCESS) {
        fprintf(stderr, "benchmark score download failed\n");
        return 1;
    }
        if (!o_spot_verify(
            &resources->corpus,
            resources->constants,
            constant_bank_count,
            reference,
            steps_per_observation,
            resources->scores,
            spot_checks,
            &checked,
            &invalid,
            &approximate_skipped,
            &maximum_relative_error,
            &cpu_check_seconds
        )
    ) {
        return 1;
    }
    if (!o_measure_loaded_kernel(resources, &launch, run_count, &loaded_timing)) return 1;
    if (o_scoring_pipeline_stats(resources->pipeline, &stats) != ODEZZA_SUCCESS ||
        o_scoring_pipeline_cubin_inspection(resources->pipeline, &inspection) != ODEZZA_SUCCESS ||
        o_scoring_pipeline_prespecialized_cubin(
            resources->pipeline,
            &retained_cubin,
            &retained_cubin_size
        ) != ODEZZA_SUCCESS
    ) {
        fprintf(stderr, "pipeline diagnostics failed\n");
        return 1;
    }
    printf("{\n");
    printf("  \"schema\": \"odezza.ast-corpus-engine-benchmark-v2\",\n");
    printf("  \"corpus_path\": \"%s\",\n", corpus_path);
    printf("  \"corpus_ast_count\": %llu,\n", (unsigned long long)corpus_report.record_count);
    printf("  \"corpus_system_count\": %llu,\n", (unsigned long long)(corpus_report.record_count / rhs_per_system));
    printf("  \"selected_system_count\": %zu,\n", resources->corpus.system_count);
    printf("  \"rhs_per_system\": %u,\n", rhs_per_system);
    printf("  \"selected_ast_count\": %zu,\n", resources->corpus.system_count * rhs_per_system);
    printf("  \"selected_program_bytes\": %zu,\n", resources->corpus.program_bytes);
    printf("  \"mean_program_bytes_per_rhs\": %.9g,\n",
           resources->corpus.program_bytes / (double)(resources->corpus.system_count * rhs_per_system));
    printf("  \"maximum_program_bytes_per_rhs\": %u,\n", corpus_report.maximum_program_bytes);
    printf("  \"sm_version\": %u,\n", create_info.sm_version);
    printf("  \"systems_per_module\": %u,\n", systems_per_module);
    printf("  \"constant_bank_count\": %u,\n", constant_bank_count);
    printf("  \"toggle_permutations\": %u,\n", 1u << O_TOGGLE_BIT_COUNT);
    printf("  \"configurations_per_system\": %llu,\n", (unsigned long long)configurations);
    printf("  \"workers\": %u,\n", worker_count);
    printf("  \"slots_per_worker\": %u,\n", slots_per_worker);
    printf("  \"loaded_module_limit\": %u,\n", loaded_modules);
    printf("  \"execution_streams\": %u,\n", streams);
    printf("  \"steps_per_observation\": %u,\n", steps_per_observation);
    printf("  \"trajectory_count\": %u,\n", O_TRAJECTORY_COUNT);
    printf("  \"observation_count\": %u,\n", O_OBSERVATION_COUNT);
    printf("  \"runs\": %u,\n", run_count);
    printf("  \"handle_create_seconds\": %.9g,\n", create_seconds);
    printf("  \"minimum_engine_seconds\": %.9g,\n", minimum_seconds);
    printf("  \"median_engine_seconds\": %.9g,\n", median_seconds);
    printf("  \"mean_engine_seconds\": %.9g,\n", mean_seconds);
    printf("  \"systems_per_second\": %.9g,\n", resources->corpus.system_count / median_seconds);
    printf("  \"rhs_asts_per_second\": %.9g,\n", resources->corpus.system_count * rhs_per_system / median_seconds);
    printf("  \"configurations_per_second\": %.9g,\n", resources->corpus.system_count * (double)configurations / median_seconds);
    printf("  \"modules_per_second\": %.9g,\n", module_count / median_seconds);
    printf("  \"cubin_bytes\": %zu,\n", retained_cubin_size);
    printf("  \"template_register_count\": %u,\n", inspection->register_count);
    printf("  \"specialized_register_count\": %u,\n", loaded_timing.register_count);
    printf("  \"minimum_loaded_kernel_seconds\": %.9g,\n", loaded_timing.minimum_seconds);
    printf("  \"median_loaded_kernel_seconds\": %.9g,\n", loaded_timing.median_seconds);
    printf("  \"mean_loaded_kernel_seconds\": %.9g,\n", loaded_timing.mean_seconds);
    printf(
        "  \"loaded_kernel_configurations_per_second\": %.9g,\n",
        loaded_timing.system_count * (double)configurations / loaded_timing.median_seconds
    );
    printf("  \"loaded_kernel_system_count\": %u,\n", loaded_timing.system_count);
    printf("  \"loaded_module_body_instructions\": %zu,\n", loaded_timing.body_instruction_count);
    printf("  \"mean_body_instructions_per_system\": %.9g,\n",
           loaded_timing.body_instruction_count / (double)loaded_timing.system_count);
    printf("  \"maximum_body_instructions_per_system\": %zu,\n", loaded_timing.maximum_body_instruction_count);
    printf("  \"workspace_bytes\": %zu,\n", workspace_size);
    printf("  \"mean_specialization_us\": %.9g,\n", 1.0e6 * stats.specialization_seconds / stats.completed_ticket_count);
    printf("  \"mean_module_load_us\": %.9g,\n", 1.0e6 * stats.module_load_seconds / stats.completed_ticket_count);
    printf("  \"mean_module_unload_us\": %.9g,\n", 1.0e6 * stats.module_unload_seconds / stats.completed_ticket_count);
    printf("  \"peak_inflight_modules\": %u,\n", stats.peak_inflight_module_count);
    printf("  \"specialization_failures\": %llu,\n", (unsigned long long)stats.specialization_failure_count);
    printf("  \"cuda_failures\": %llu,\n", (unsigned long long)stats.cuda_failure_count);
    printf("  \"cpu_exact_math_spot_checks\": %u,\n", checked);
    printf("  \"cpu_exact_math_mutual_invalid_scores\": %u,\n", invalid);
    printf("  \"cpu_approximate_math_candidates_skipped\": %u,\n", approximate_skipped);
    printf("  \"cpu_exact_math_maximum_relative_error\": %.9g,\n", maximum_relative_error);
    printf("  \"cpu_check_seconds\": %.9g,\n", cpu_check_seconds);
    printf("  \"cpu_verification_scope\": \"complete RK4 score replay for ASTs without device-approximate math\",\n");
    printf(
        "  \"timed_scope\": \"C99 specialization, eager module load, lookup, RK4 execution, retirement, and unload; "
        "corpus parse, allocation, upload, download, and CPU replay excluded\"\n"
    );
    printf("}\n");
    return 0;
}

static void o_resources_destroy(OBenchmarkResources *resources) {
    if (resources->pipeline != NULL) (void)odezza_scoring_pipeline_destroy(resources->pipeline);
    if (resources->context != NULL) {
        if (resources->scores_device != 0u) (void)cuMemFree(resources->scores_device);
        if (resources->reference_device != 0u) (void)cuMemFree(resources->reference_device);
        if (resources->trajectory_times_device != 0u) (void)cuMemFree(resources->trajectory_times_device);
        if (resources->trajectory_offsets_device != 0u) (void)cuMemFree(resources->trajectory_offsets_device);
        if (resources->constants_device != 0u) (void)cuMemFree(resources->constants_device);
        (void)cuCtxSetCurrent(NULL);
        (void)cuDevicePrimaryCtxRelease(resources->device);
    }
    if (resources->workspace != NULL) free(resources->workspace);
    if (resources->scores != NULL) free(resources->scores);
    if (resources->constants != NULL) free(resources->constants);
    if (resources->corpus.programs != NULL) free(resources->corpus.programs);
    if (resources->corpus.rhs != NULL) free(resources->corpus.rhs);
    if (resources->corpus.systems != NULL) free(resources->corpus.systems);
}

int main(int argc, char **argv) {
    OBenchmarkResources resources;
    int status;
    memset(&resources, 0, sizeof(resources));
    status = o_benchmark_run(argc, argv, &resources);
    o_resources_destroy(&resources);
    return status;
}
