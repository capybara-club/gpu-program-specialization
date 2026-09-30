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

#include "bench_ast.h"
#include "secant_cpu.h"

#ifndef SECANT_BENCH_HAS_CUDA
#define SECANT_BENCH_HAS_CUDA 0
#endif
#ifndef SECANT_BENCH_HAS_PTX
#define SECANT_BENCH_HAS_PTX 0
#endif
#ifndef SECANT_BENCH_HAS_HIP
#define SECANT_BENCH_HAS_HIP 0
#endif
#ifndef SECANT_BENCH_HAS_CUBIN
#define SECANT_BENCH_HAS_CUBIN 0
#endif
#ifndef SECANT_BENCH_HAS_HSACO
#define SECANT_BENCH_HAS_HSACO 0
#endif

#if SECANT_BENCH_HAS_CUDA
#include "secant_cuda.h"
#endif
#if SECANT_BENCH_HAS_PTX
#include "secant_ptx.h"
#endif
#if SECANT_BENCH_HAS_HIP
#include "secant_hip.h"
#endif
#if SECANT_BENCH_HAS_CUBIN
#define SECANT_BENCH_CUBIN_TEMPLATE_DYNAMIC_ONLY
#include "bench_cubin_template.h"
#undef SECANT_BENCH_CUBIN_TEMPLATE_DYNAMIC_ONLY
#include "secant_cubin.h"
#endif
#if SECANT_BENCH_HAS_HSACO
#define SECANT_BENCH_HSACO_TEMPLATE_DYNAMIC_ONLY
#include "bench_hsaco_template.h"
#undef SECANT_BENCH_HSACO_TEMPLATE_DYNAMIC_ONLY
#include "secant_hsaco.h"
#endif

#if SECANT_BENCH_HAS_CUDA || SECANT_BENCH_HAS_PTX || \
    SECANT_BENCH_HAS_CUBIN
#include "cuda_runtime.h"
#include <cuda.h>
#endif
#if SECANT_BENCH_HAS_HIP || SECANT_BENCH_HAS_HSACO
#include "hip_runtime.h"
#include <hip/hip_runtime_api.h>
#endif

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BENCH_COMPILE_SCRATCH_BYTES (64u * 1024u * 1024u)
#define BENCH_ERROR_BYTES 16384u

typedef enum BenchmarkBackend {
    BENCH_BACKEND_CUDA = 0,
    BENCH_BACKEND_PTX = 1,
    BENCH_BACKEND_HIP = 2,
    BENCH_BACKEND_CUBIN = 3,
    BENCH_BACKEND_HSACO = 4
} BenchmarkBackend;

typedef struct BenchmarkOptions {
    BenchmarkBackend backend;
    SecantBenchAstMode ast_mode;
    size_t num_kernels;
    size_t asts_per_kernel;
    size_t num_input_columns;
    size_t num_input_constants;
    size_t num_targets;
    size_t num_settings;
    size_t num_rows;
    size_t tile_rows;
    size_t threads_per_block;
    size_t num_streams;
    size_t patch_instructions_per_ast;
    SecantSSEReductionMode reduction_mode;
    size_t warmups;
    size_t iterations;
    uint32_t seed;
    uint32_t source_sm;
    uint32_t target_sm;
    uint32_t opt_level;
    const char* hip_architecture;
    int device_ordinal;
} BenchmarkOptions;

typedef struct BenchmarkArtifact {
    const void* binary;
    size_t binary_size;
    double template_prepare_seconds;
#if SECANT_BENCH_HAS_CUDA
    SecantCUDACompiled cuda_compiled;
#endif
#if SECANT_BENCH_HAS_PTX
    SecantPTXHandle ptx_handle;
    SecantPTXCompiled ptx_compiled;
#endif
#if SECANT_BENCH_HAS_HIP
    SecantHIPCompiled hip_compiled;
#endif
#if SECANT_BENCH_HAS_CUBIN
    SecantCubinPlan* cubin_plan;
    unsigned char* cubin_template;
    unsigned char* cubin_binary;
    void* cubin_plan_storage;
#endif
#if SECANT_BENCH_HAS_HSACO
    SecantHsacoPlan* hsaco_plan;
    unsigned char* hsaco_template;
    unsigned char* hsaco_binary;
    void* hsaco_plan_storage;
#endif
} BenchmarkArtifact;

typedef struct BenchmarkDevice {
    BenchmarkBackend backend;
    void* module;
    void** functions;
    void** streams;
    void** done_events;
    void* start_event;
    void* stop_event;
    void* reduction_done_event;
    void* reducer_function;
    void* input;
    void* constants;
    void* targets;
    void* output;
    void* workspace;
    size_t workspace_num_elements;
    uint32_t target_sm;
    uint32_t gfx_arch;
    char hip_architecture[128];
    int registers;
    int static_shared_bytes;
    int local_bytes;
    int active_blocks;
#if SECANT_BENCH_HAS_CUDA || SECANT_BENCH_HAS_PTX || \
    SECANT_BENCH_HAS_CUBIN
    CUdevice cuda_device;
    CUcontext cuda_context;
    CUcontext cuda_previous_context;
    int cuda_context_retained;
#endif
} BenchmarkDevice;

static const char*
benchmark_backend_name(BenchmarkBackend backend) {
    switch (backend) {
        case BENCH_BACKEND_CUDA: return "cuda";
        case BENCH_BACKEND_PTX: return "ptx";
        case BENCH_BACKEND_HIP: return "hip";
        case BENCH_BACKEND_CUBIN: return "cubin";
        case BENCH_BACKEND_HSACO: return "hsaco";
        default: return "unknown";
    }
}

static const char*
benchmark_ast_mode_name(SecantBenchAstMode mode) {
    switch (mode) {
        case SECANT_BENCH_AST_MODE_SIMPLE: return "simple";
        case SECANT_BENCH_AST_MODE_ALU: return "alu";
        case SECANT_BENCH_AST_MODE_MUFU: return "mufu";
        default: return "unknown";
    }
}

static const char*
benchmark_reduction_mode_name(SecantSSEReductionMode mode) {
    return mode == SECANT_SSE_REDUCTION_MODE_WORKSPACE
        ? "workspace"
        : "atomic";
}

static int
benchmark_backend_available(BenchmarkBackend backend) {
    switch (backend) {
        case BENCH_BACKEND_CUDA: return SECANT_BENCH_HAS_CUDA;
        case BENCH_BACKEND_PTX: return SECANT_BENCH_HAS_PTX;
        case BENCH_BACKEND_HIP: return SECANT_BENCH_HAS_HIP;
        case BENCH_BACKEND_CUBIN: return SECANT_BENCH_HAS_CUBIN;
        case BENCH_BACKEND_HSACO: return SECANT_BENCH_HAS_HSACO;
        default: return 0;
    }
}

static int
benchmark_backend_is_nvidia(BenchmarkBackend backend) {
    return backend == BENCH_BACKEND_CUDA ||
        backend == BENCH_BACKEND_PTX ||
        backend == BENCH_BACKEND_CUBIN;
}

static void
benchmark_options_default(BenchmarkOptions* options) {
    memset(options, 0, sizeof(*options));
#if SECANT_BENCH_HAS_CUBIN
    options->backend = BENCH_BACKEND_CUBIN;
#elif SECANT_BENCH_HAS_CUDA
    options->backend = BENCH_BACKEND_CUDA;
#elif SECANT_BENCH_HAS_HSACO
    options->backend = BENCH_BACKEND_HSACO;
#else
    options->backend = BENCH_BACKEND_HIP;
#endif
    options->ast_mode = SECANT_BENCH_AST_MODE_ALU;
    options->num_kernels = 1u;
    options->asts_per_kernel = 24u;
    options->num_input_columns = 4u;
    options->num_input_constants = 4u;
    options->num_targets = 2u;
    options->num_settings = 256u;
    options->num_rows = 131072u;
    options->tile_rows = 0u;
    options->threads_per_block = 256u;
    options->num_streams = 1u;
    options->patch_instructions_per_ast = 64u;
    options->reduction_mode = SECANT_SSE_REDUCTION_MODE_ATOMIC;
    options->warmups = 0u;
    options->iterations = 50u;
    options->seed = 1u;
    options->source_sm = 80u;
    options->target_sm = 0u;
    options->opt_level = 1u;
    options->device_ordinal = 0;
}

static void
benchmark_usage(const char* program) {
    fprintf(
        stderr,
        "usage: %s [options]\n"
        "  --backend cuda|ptx|hip|cubin|hsaco\n"
        "  --ast-mode simple|alu|mufu\n"
        "  --kernels N --asts-per-kernel N\n"
        "  --columns N --constants N --targets N --settings N\n"
        "  --rows N --tile-rows N --threads N --streams N\n"
        "  --reduction atomic|workspace\n"
        "  --patch-instructions-per-ast N\n"
        "  --warmups N --iterations N --seed N\n"
        "  --source-sm NN --target-sm NN --opt-level 0|1\n"
        "  --hip-architecture gfxNNNN --device N\n"
        "\n"
        "default tile rows: NVIDIA=64, HIP=1024, HSACO=256\n"
        "default warmups: NVIDIA=3, AMD=100\n",
        program);
}

static int
benchmark_parse_size(const char* text, size_t* value_ret) {
    char* end = NULL;
    unsigned long long value;

    errno = 0;
    value = strtoull(text, &end, 0);
    if (errno != 0 || text == end || *end != '\0' ||
        value == 0u || value > SIZE_MAX) {
        return 0;
    }
    *value_ret = (size_t)value;
    return 1;
}

static int
benchmark_parse_u32(const char* text, uint32_t* value_ret) {
    char* end = NULL;
    unsigned long value;

    errno = 0;
    value = strtoul(text, &end, 0);
    if (errno != 0 || text == end || *end != '\0' ||
        value > UINT32_MAX) {
        return 0;
    }
    *value_ret = (uint32_t)value;
    return 1;
}

static int
benchmark_parse_int(const char* text, int* value_ret) {
    char* end = NULL;
    long value;

    errno = 0;
    value = strtol(text, &end, 0);
    if (errno != 0 || text == end || *end != '\0' ||
        value < 0 || value > INT32_MAX) {
        return 0;
    }
    *value_ret = (int)value;
    return 1;
}

static int
benchmark_parse_options(
    int argc,
    char** argv,
    BenchmarkOptions* options
) {
    int arg_idx;

    for (arg_idx = 1; arg_idx < argc; ++arg_idx) {
        const char* name = argv[arg_idx];
        const char* value;

        if (strcmp(name, "--help") == 0) {
            return -1;
        }
        if (arg_idx + 1 >= argc) {
            return 0;
        }
        value = argv[++arg_idx];
        if (strcmp(name, "--backend") == 0) {
            if (strcmp(value, "cuda") == 0) {
                options->backend = BENCH_BACKEND_CUDA;
            } else if (strcmp(value, "ptx") == 0) {
                options->backend = BENCH_BACKEND_PTX;
            } else if (strcmp(value, "hip") == 0) {
                options->backend = BENCH_BACKEND_HIP;
            } else if (strcmp(value, "cubin") == 0) {
                options->backend = BENCH_BACKEND_CUBIN;
            } else if (strcmp(value, "hsaco") == 0) {
                options->backend = BENCH_BACKEND_HSACO;
            } else {
                return 0;
            }
        } else if (strcmp(name, "--ast-mode") == 0) {
            if (strcmp(value, "simple") == 0) {
                options->ast_mode = SECANT_BENCH_AST_MODE_SIMPLE;
            } else if (strcmp(value, "alu") == 0) {
                options->ast_mode = SECANT_BENCH_AST_MODE_ALU;
            } else if (strcmp(value, "mufu") == 0) {
                options->ast_mode = SECANT_BENCH_AST_MODE_MUFU;
            } else {
                return 0;
            }
        } else if (strcmp(name, "--kernels") == 0) {
            if (!benchmark_parse_size(value, &options->num_kernels)) {
                return 0;
            }
        } else if (strcmp(name, "--asts-per-kernel") == 0) {
            if (!benchmark_parse_size(value, &options->asts_per_kernel)) {
                return 0;
            }
        } else if (strcmp(name, "--columns") == 0) {
            if (!benchmark_parse_size(
                    value,
                    &options->num_input_columns)) {
                return 0;
            }
        } else if (strcmp(name, "--constants") == 0) {
            if (!benchmark_parse_size(
                    value,
                    &options->num_input_constants)) {
                return 0;
            }
        } else if (strcmp(name, "--targets") == 0) {
            if (!benchmark_parse_size(value, &options->num_targets)) {
                return 0;
            }
        } else if (strcmp(name, "--settings") == 0) {
            if (!benchmark_parse_size(value, &options->num_settings)) {
                return 0;
            }
        } else if (strcmp(name, "--rows") == 0) {
            if (!benchmark_parse_size(value, &options->num_rows)) {
                return 0;
            }
        } else if (strcmp(name, "--tile-rows") == 0) {
            if (!benchmark_parse_size(value, &options->tile_rows)) {
                return 0;
            }
        } else if (strcmp(name, "--threads") == 0) {
            if (!benchmark_parse_size(
                    value,
                    &options->threads_per_block)) {
                return 0;
            }
        } else if (strcmp(name, "--streams") == 0) {
            if (!benchmark_parse_size(value, &options->num_streams)) {
                return 0;
            }
        } else if (strcmp(name, "--reduction") == 0) {
            if (strcmp(value, "atomic") == 0) {
                options->reduction_mode =
                    SECANT_SSE_REDUCTION_MODE_ATOMIC;
            } else if (strcmp(value, "workspace") == 0) {
                options->reduction_mode =
                    SECANT_SSE_REDUCTION_MODE_WORKSPACE;
            } else {
                return 0;
            }
        } else if (strcmp(name, "--patch-instructions-per-ast") == 0) {
            if (!benchmark_parse_size(
                    value,
                    &options->patch_instructions_per_ast)) {
                return 0;
            }
        } else if (strcmp(name, "--warmups") == 0) {
            if (!benchmark_parse_size(value, &options->warmups)) {
                return 0;
            }
        } else if (strcmp(name, "--iterations") == 0) {
            if (!benchmark_parse_size(value, &options->iterations)) {
                return 0;
            }
        } else if (strcmp(name, "--seed") == 0) {
            if (!benchmark_parse_u32(value, &options->seed)) {
                return 0;
            }
        } else if (strcmp(name, "--source-sm") == 0) {
            if (!benchmark_parse_u32(value, &options->source_sm)) {
                return 0;
            }
        } else if (strcmp(name, "--target-sm") == 0) {
            if (!benchmark_parse_u32(value, &options->target_sm)) {
                return 0;
            }
        } else if (strcmp(name, "--opt-level") == 0) {
            if (!benchmark_parse_u32(value, &options->opt_level)) {
                return 0;
            }
        } else if (strcmp(name, "--hip-architecture") == 0) {
            options->hip_architecture = value;
        } else if (strcmp(name, "--device") == 0) {
            if (!benchmark_parse_int(value, &options->device_ordinal)) {
                return 0;
            }
        } else {
            return 0;
        }
    }
    return 1;
}

static int
benchmark_checked_mul(
    size_t lhs,
    size_t rhs,
    size_t* result_ret
) {
    if (lhs != 0u && rhs > SIZE_MAX / lhs) {
        return 0;
    }
    *result_ret = lhs * rhs;
    return 1;
}

static int
benchmark_options_valid(
    const BenchmarkOptions* options,
    size_t* num_inputs_ret,
    size_t* num_asts_ret,
    size_t* output_per_kernel_ret
) {
    size_t num_inputs;
    size_t num_asts;
    size_t output_series;

    if (!benchmark_backend_available(options->backend) ||
        options->num_input_columns > 32u ||
        options->num_input_constants > 32u ||
        options->threads_per_block > 1024u ||
        options->num_streams > options->num_kernels ||
        options->opt_level > 1u ||
        options->reduction_mode >=
            SECANT_SSE_REDUCTION_MODE_NUM_ENUMS ||
        options->num_input_constants >
            SIZE_MAX - options->num_input_columns ||
        options->asts_per_kernel >
            SIZE_MAX / options->patch_instructions_per_ast) {
        return 0;
    }
    num_inputs =
        options->num_input_columns + options->num_input_constants;
    if ((options->ast_mode == SECANT_BENCH_AST_MODE_ALU &&
         num_inputs < 8u) ||
        !benchmark_checked_mul(
            options->num_kernels,
            options->asts_per_kernel,
            &num_asts) ||
        !benchmark_checked_mul(
            options->asts_per_kernel,
            options->num_targets,
            &output_series) ||
        !benchmark_checked_mul(
            output_series,
            options->num_settings,
            output_per_kernel_ret)) {
        return 0;
    }
    *num_inputs_ret = num_inputs;
    *num_asts_ret = num_asts;
    return 1;
}

static double
benchmark_wall_seconds(void) {
    struct timespec timestamp;

    if (clock_gettime(CLOCK_MONOTONIC, &timestamp) != 0) {
        return -1.0;
    }
    return (double)timestamp.tv_sec +
        (double)timestamp.tv_nsec * 1.0e-9;
}

static float
benchmark_value(size_t outer, size_t inner, uint32_t seed) {
    const uint32_t hash = secant_bench_ast_hash32(
        (uint32_t)outer * 0x9e3779b9u ^
        (uint32_t)inner * 0x85ebca6bu ^
        seed * 0xc2b2ae35u);

    return ((float)(hash & 0xffffu) / 65535.0f) * 1.5f - 0.75f;
}

#if SECANT_BENCH_HAS_HIP || SECANT_BENCH_HAS_HSACO
static int
benchmark_gfx_arch_parse(const char* architecture, uint32_t* gfx_arch_ret) {
    const char* digits;
    uint32_t value = 0u;
    size_t digit_count = 0u;

    if (architecture == NULL || gfx_arch_ret == NULL) {
        return 0;
    }
    digits = strncmp(architecture, "gfx", 3u) == 0
        ? architecture + 3u
        : architecture;
    while (digits[digit_count] >= '0' &&
           digits[digit_count] <= '9') {
        const uint32_t digit =
            (uint32_t)(digits[digit_count] - '0');

        if (value > (UINT32_MAX - digit) / 10u) {
            return 0;
        }
        value = value * 10u + digit;
        digit_count += 1u;
    }
    if (digit_count == 0u) {
        return 0;
    }
    *gfx_arch_ret = value;
    return 1;
}
#endif

static int
benchmark_device_create(
    const BenchmarkOptions* options,
    BenchmarkDevice* device
) {
    size_t stream_idx;

    memset(device, 0, sizeof(*device));
    device->backend = options->backend;
    device->functions = (void**)calloc(
        options->num_kernels,
        sizeof(*device->functions));
    device->streams = (void**)calloc(
        options->num_streams,
        sizeof(*device->streams));
    device->done_events = (void**)calloc(
        options->num_streams,
        sizeof(*device->done_events));
    if (device->functions == NULL || device->streams == NULL ||
        device->done_events == NULL) {
        return 0;
    }
    if (benchmark_backend_is_nvidia(options->backend)) {
#if SECANT_BENCH_HAS_CUDA || SECANT_BENCH_HAS_PTX || \
    SECANT_BENCH_HAS_CUBIN
        int major;
        int minor;
        CUevent start;
        CUevent stop;
        CUevent reduction_done;

        if (cuInit(0u) != CUDA_SUCCESS ||
            cuDeviceGet(
                &device->cuda_device,
                options->device_ordinal) != CUDA_SUCCESS ||
            cuDevicePrimaryCtxRetain(
                &device->cuda_context,
                device->cuda_device) != CUDA_SUCCESS) {
            return 0;
        }
        device->cuda_context_retained = 1;
        if (cuCtxGetCurrent(
                &device->cuda_previous_context) != CUDA_SUCCESS ||
            cuCtxSetCurrent(device->cuda_context) != CUDA_SUCCESS ||
            cuDeviceGetAttribute(
                &major,
                CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                device->cuda_device) != CUDA_SUCCESS ||
            cuDeviceGetAttribute(
                &minor,
                CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                device->cuda_device) != CUDA_SUCCESS ||
            cuEventCreate(&start, CU_EVENT_DEFAULT) != CUDA_SUCCESS ||
            cuEventCreate(&stop, CU_EVENT_DEFAULT) != CUDA_SUCCESS ||
            cuEventCreate(
                &reduction_done,
                CU_EVENT_DISABLE_TIMING) != CUDA_SUCCESS) {
            return 0;
        }
        device->target_sm = options->target_sm != 0u
            ? options->target_sm
            : (uint32_t)(major * 10 + minor);
        device->start_event = (void*)start;
        device->stop_event = (void*)stop;
        device->reduction_done_event = (void*)reduction_done;
        for (stream_idx = 0u;
             stream_idx < options->num_streams;
             ++stream_idx) {
            CUstream stream;
            CUevent done;

            if (cuStreamCreate(
                    &stream,
                    CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS ||
                cuEventCreate(
                    &done,
                    CU_EVENT_DISABLE_TIMING) != CUDA_SUCCESS) {
                return 0;
            }
            device->streams[stream_idx] = (void*)stream;
            device->done_events[stream_idx] = (void*)done;
        }
        return 1;
#else
        return 0;
#endif
    }
#if SECANT_BENCH_HAS_HIP || SECANT_BENCH_HAS_HSACO
    {
        hipDeviceProp_t properties;
        hipEvent_t start;
        hipEvent_t stop;
        hipEvent_t reduction_done;
        const char* architecture;

        if (hipSetDevice(options->device_ordinal) != hipSuccess ||
            hipGetDeviceProperties(
                &properties,
                options->device_ordinal) != hipSuccess ||
            hipEventCreate(&start) != hipSuccess ||
            hipEventCreate(&stop) != hipSuccess ||
            hipEventCreateWithFlags(
                &reduction_done,
                hipEventDisableTiming) != hipSuccess) {
            return 0;
        }
        architecture = options->hip_architecture != NULL
            ? options->hip_architecture
            : properties.gcnArchName;
        if (snprintf(
                device->hip_architecture,
                sizeof(device->hip_architecture),
                "%s",
                architecture) < 0 ||
            !benchmark_gfx_arch_parse(
                device->hip_architecture,
                &device->gfx_arch)) {
            return 0;
        }
        device->start_event = (void*)start;
        device->stop_event = (void*)stop;
        device->reduction_done_event = (void*)reduction_done;
        for (stream_idx = 0u;
             stream_idx < options->num_streams;
             ++stream_idx) {
            hipStream_t stream;
            hipEvent_t done;

            if (hipStreamCreateWithFlags(
                    &stream,
                    hipStreamNonBlocking) != hipSuccess ||
                hipEventCreateWithFlags(
                    &done,
                    hipEventDisableTiming) != hipSuccess) {
                return 0;
            }
            device->streams[stream_idx] = (void*)stream;
            device->done_events[stream_idx] = (void*)done;
        }
        return 1;
    }
#else
    return 0;
#endif
}

static void
benchmark_device_destroy(
    const BenchmarkOptions* options,
    BenchmarkDevice* device
) {
    size_t stream_idx;

    if (benchmark_backend_is_nvidia(device->backend)) {
#if SECANT_BENCH_HAS_CUDA || SECANT_BENCH_HAS_PTX || \
    SECANT_BENCH_HAS_CUBIN
        if (device->module != NULL) {
            (void)cuModuleUnload((CUmodule)device->module);
        }
        if (device->output != NULL) {
            (void)cuMemFree((CUdeviceptr)(uintptr_t)device->output);
        }
        if (device->workspace != NULL) {
            (void)cuMemFree((CUdeviceptr)(uintptr_t)device->workspace);
        }
        if (device->targets != NULL) {
            (void)cuMemFree((CUdeviceptr)(uintptr_t)device->targets);
        }
        if (device->constants != NULL) {
            (void)cuMemFree((CUdeviceptr)(uintptr_t)device->constants);
        }
        if (device->input != NULL) {
            (void)cuMemFree((CUdeviceptr)(uintptr_t)device->input);
        }
        for (stream_idx = 0u;
             stream_idx < options->num_streams;
             ++stream_idx) {
            if (device->done_events != NULL &&
                device->done_events[stream_idx] != NULL) {
                (void)cuEventDestroy(
                    (CUevent)device->done_events[stream_idx]);
            }
            if (device->streams != NULL &&
                device->streams[stream_idx] != NULL) {
                (void)cuStreamDestroy(
                    (CUstream)device->streams[stream_idx]);
            }
        }
        if (device->stop_event != NULL) {
            (void)cuEventDestroy((CUevent)device->stop_event);
        }
        if (device->reduction_done_event != NULL) {
            (void)cuEventDestroy(
                (CUevent)device->reduction_done_event);
        }
        if (device->start_event != NULL) {
            (void)cuEventDestroy((CUevent)device->start_event);
        }
        if (device->cuda_context_retained) {
            (void)cuCtxSetCurrent(device->cuda_previous_context);
            (void)cuDevicePrimaryCtxRelease(device->cuda_device);
        }
#endif
    } else {
#if SECANT_BENCH_HAS_HIP || SECANT_BENCH_HAS_HSACO
        if (device->module != NULL) {
            (void)hipModuleUnload((hipModule_t)device->module);
        }
        if (device->output != NULL) {
            (void)hipFree(device->output);
        }
        if (device->workspace != NULL) {
            (void)hipFree(device->workspace);
        }
        if (device->targets != NULL) {
            (void)hipFree(device->targets);
        }
        if (device->constants != NULL) {
            (void)hipFree(device->constants);
        }
        if (device->input != NULL) {
            (void)hipFree(device->input);
        }
        for (stream_idx = 0u;
             stream_idx < options->num_streams;
             ++stream_idx) {
            if (device->done_events != NULL &&
                device->done_events[stream_idx] != NULL) {
                (void)hipEventDestroy(
                    (hipEvent_t)device->done_events[stream_idx]);
            }
            if (device->streams != NULL &&
                device->streams[stream_idx] != NULL) {
                (void)hipStreamDestroy(
                    (hipStream_t)device->streams[stream_idx]);
            }
        }
        if (device->stop_event != NULL) {
            (void)hipEventDestroy((hipEvent_t)device->stop_event);
        }
        if (device->reduction_done_event != NULL) {
            (void)hipEventDestroy(
                (hipEvent_t)device->reduction_done_event);
        }
        if (device->start_event != NULL) {
            (void)hipEventDestroy((hipEvent_t)device->start_event);
        }
#endif
    }
    free(device->done_events);
    free(device->streams);
    free(device->functions);
}

static int
benchmark_artifact_template_prepare(
    const BenchmarkOptions* options,
    const BenchmarkDevice* device,
    BenchmarkArtifact* artifact,
    char* error,
    size_t error_size
) {
    const double start = benchmark_wall_seconds();

    memset(artifact, 0, sizeof(*artifact));
    switch (options->backend) {
#if SECANT_BENCH_HAS_PTX
        case BENCH_BACKEND_PTX:
            {
                static const char* const nvrtc_options[] = {
                    "--no-cache",
                    "--split-compile=1"
                };
                const SecantPTXResult result =
                    secant_ptx_dynamic_constant_sse_create(
                        options->num_kernels,
                        options->asts_per_kernel,
                        options->num_input_columns,
                        options->num_input_constants,
                        options->num_targets,
                        options->tile_rows,
                        options->threads_per_block,
                        options->reduction_mode,
                        options->source_sm / 10u,
                        options->source_sm % 10u,
                        nvrtc_options,
                        sizeof(nvrtc_options) /
                            sizeof(nvrtc_options[0]),
                        false,
                        error,
                        error_size,
                        NULL,
                        &artifact->ptx_handle);

                if (result != SECANT_PTX_SUCCESS) {
                    snprintf(
                        error,
                        error_size,
                        "%s",
                        secant_ptx_result_to_string(result));
                    return 0;
                }
                break;
            }
#endif
#if SECANT_BENCH_HAS_CUBIN
        case BENCH_BACKEND_CUBIN:
            {
                const size_t patch_capacity =
                    options->asts_per_kernel *
                    options->patch_instructions_per_ast;
                const SecantCubinDynamicConstantSSERecipe recipe = {
                    options->num_kernels,
                    options->asts_per_kernel,
                    options->num_input_columns,
                    options->num_input_constants,
                    options->num_targets,
                    options->tile_rows,
                    options->threads_per_block,
                    patch_capacity,
                    options->reduction_mode
                };
                size_t plan_size = 0u;
                SecantResult result;

                if (!secant_bench_cubin_dynamic_constant_sse_template_compile(
                        options->num_kernels,
                        options->asts_per_kernel,
                        options->num_input_columns,
                        options->num_input_constants,
                        options->num_targets,
                        options->tile_rows,
                        options->threads_per_block,
                        options->reduction_mode,
                        patch_capacity,
                        device->target_sm,
                        error,
                        error_size,
                        &artifact->cubin_template,
                        &artifact->binary_size)) {
                    return 0;
                }
                result = secant_cubin_dynamic_constant_sse_inspect(
                    &recipe,
                    artifact->cubin_template,
                    artifact->binary_size,
                    NULL,
                    0u,
                    &plan_size,
                    &artifact->cubin_plan);
                if (result != SECANT_SUCCESS || plan_size == 0u) {
                    snprintf(
                        error,
                        error_size,
                        "CUBIN inspect measure failed: %s",
                        secant_result_to_string(result));
                    return 0;
                }
                artifact->cubin_plan_storage = malloc(plan_size);
                artifact->cubin_binary =
                    (unsigned char*)malloc(artifact->binary_size);
                if (artifact->cubin_plan_storage == NULL ||
                    artifact->cubin_binary == NULL) {
                    snprintf(error, error_size, "CUBIN allocation failed");
                    return 0;
                }
                result = secant_cubin_dynamic_constant_sse_inspect(
                    &recipe,
                    artifact->cubin_template,
                    artifact->binary_size,
                    artifact->cubin_plan_storage,
                    plan_size,
                    &plan_size,
                    &artifact->cubin_plan);
                if (result != SECANT_SUCCESS) {
                    snprintf(
                        error,
                        error_size,
                        "CUBIN inspect failed: %s",
                        secant_result_to_string(result));
                    return 0;
                }
                memcpy(
                    artifact->cubin_binary,
                    artifact->cubin_template,
                    artifact->binary_size);
                artifact->binary = artifact->cubin_binary;
                break;
            }
#endif
#if SECANT_BENCH_HAS_HSACO
        case BENCH_BACKEND_HSACO:
            {
                const size_t patch_capacity =
                    options->asts_per_kernel *
                    options->patch_instructions_per_ast;
                const SecantHsacoDynamicConstantSSERecipe recipe = {
                    options->num_kernels,
                    options->asts_per_kernel,
                    options->num_input_columns,
                    options->num_input_constants,
                    options->num_targets,
                    options->tile_rows,
                    options->threads_per_block,
                    patch_capacity,
                    options->reduction_mode
                };
                size_t plan_size = 0u;
                SecantResult result;

                if (!secant_bench_hsaco_dynamic_constant_sse_template_compile(
                        options->num_kernels,
                        options->asts_per_kernel,
                        options->num_input_columns,
                        options->num_input_constants,
                        options->num_targets,
                        options->tile_rows,
                        options->threads_per_block,
                        options->reduction_mode,
                        patch_capacity,
                        options->opt_level,
                        device->gfx_arch,
                        error,
                        error_size,
                        &artifact->hsaco_template,
                        &artifact->binary_size)) {
                    return 0;
                }
                result = secant_hsaco_dynamic_constant_sse_inspect(
                    &recipe,
                    artifact->hsaco_template,
                    artifact->binary_size,
                    NULL,
                    0u,
                    &plan_size,
                    &artifact->hsaco_plan);
                if (result != SECANT_SUCCESS || plan_size == 0u) {
                    snprintf(
                        error,
                        error_size,
                        "HSACO inspect measure failed: %s",
                        secant_result_to_string(result));
                    return 0;
                }
                artifact->hsaco_plan_storage = malloc(plan_size);
                artifact->hsaco_binary =
                    (unsigned char*)malloc(artifact->binary_size);
                if (artifact->hsaco_plan_storage == NULL ||
                    artifact->hsaco_binary == NULL) {
                    snprintf(error, error_size, "HSACO allocation failed");
                    return 0;
                }
                result = secant_hsaco_dynamic_constant_sse_inspect(
                    &recipe,
                    artifact->hsaco_template,
                    artifact->binary_size,
                    artifact->hsaco_plan_storage,
                    plan_size,
                    &plan_size,
                    &artifact->hsaco_plan);
                if (result != SECANT_SUCCESS) {
                    snprintf(
                        error,
                        error_size,
                        "HSACO inspect failed: %s",
                        secant_result_to_string(result));
                    return 0;
                }
                memcpy(
                    artifact->hsaco_binary,
                    artifact->hsaco_template,
                    artifact->binary_size);
                artifact->binary = artifact->hsaco_binary;
                break;
            }
#endif
        case BENCH_BACKEND_CUDA:
        case BENCH_BACKEND_HIP:
            break;
        default:
            snprintf(error, error_size, "backend unavailable");
            return 0;
    }
    artifact->template_prepare_seconds =
        benchmark_wall_seconds() - start;
    return artifact->template_prepare_seconds >= 0.0;
}

static int
benchmark_artifact_compile(
    const BenchmarkOptions* options,
    const BenchmarkDevice* device,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    const SecantAstInstruction* const* asts,
    void* compile_scratch,
    size_t compile_scratch_size,
    BenchmarkArtifact* artifact,
    char* error,
    size_t error_size,
    double* seconds_ret
) {
    double start = benchmark_wall_seconds();

    switch (options->backend) {
#if SECANT_BENCH_HAS_CUDA
        case BENCH_BACKEND_CUDA:
            {
                const char* compile_options[] = {
                    "--no-cache",
                    "--split-compile=1",
                    NULL
                };
                char opt_level[32];
                SecantCUDAResult result;

                snprintf(
                    opt_level,
                    sizeof(opt_level),
                    "--ptxas-options=-O%u",
                    options->opt_level);
                compile_options[2] = opt_level;
                result = secant_cuda_dynamic_constant_sse_compile(
                    options->num_kernels,
                    options->asts_per_kernel,
                    options->num_input_columns,
                    options->num_input_constants,
                    options->num_targets,
                    options->tile_rows,
                    options->threads_per_block,
                    options->reduction_mode,
                    routines,
                    num_routines,
                    routine_names,
                    asts,
                    device->target_sm / 10u,
                    device->target_sm % 10u,
                    compile_options,
                    sizeof(compile_options) /
                        sizeof(compile_options[0]),
                    false,
                    compile_scratch,
                    compile_scratch_size,
                    error,
                    error_size,
                    NULL,
                    &artifact->cuda_compiled);
                if (result != SECANT_CUDA_SUCCESS ||
                    secant_cuda_compiled_binary_get(
                        artifact->cuda_compiled,
                        &artifact->binary,
                        &artifact->binary_size) != SECANT_CUDA_SUCCESS) {
                    snprintf(
                        error,
                        error_size,
                        "%s",
                        secant_cuda_result_to_string(result));
                    return 0;
                }
                break;
            }
#endif
#if SECANT_BENCH_HAS_PTX
        case BENCH_BACKEND_PTX:
            {
                const char* compile_options[] = {
                    NULL,
                    "--split-compile=1"
                };
                char opt_level[32];
                SecantPTXResult result;

                snprintf(
                    opt_level,
                    sizeof(opt_level),
                    "--opt-level=%u",
                    options->opt_level);
                compile_options[0] = opt_level;
                result = secant_ptx_compile(
                    artifact->ptx_handle,
                    routines,
                    num_routines,
                    asts,
                    device->target_sm / 10u,
                    device->target_sm % 10u,
                    compile_options,
                    sizeof(compile_options) /
                        sizeof(compile_options[0]),
                    false,
                    compile_scratch,
                    compile_scratch_size,
                    error,
                    error_size,
                    NULL,
                    &artifact->ptx_compiled);
                if (result != SECANT_PTX_SUCCESS ||
                    secant_ptx_compiled_binary_get(
                        artifact->ptx_compiled,
                        &artifact->binary,
                        &artifact->binary_size) != SECANT_PTX_SUCCESS) {
                    snprintf(
                        error,
                        error_size,
                        "%s",
                        secant_ptx_result_to_string(result));
                    return 0;
                }
                break;
            }
#endif
#if SECANT_BENCH_HAS_HIP
        case BENCH_BACKEND_HIP:
            {
                const char* compile_options[] = { NULL };
                char opt_level[16];
                SecantHIPResult result;

                snprintf(
                    opt_level,
                    sizeof(opt_level),
                    "-O%u",
                    options->opt_level);
                compile_options[0] = opt_level;
                result = secant_hip_dynamic_constant_sse_compile(
                    options->num_kernels,
                    options->asts_per_kernel,
                    options->num_input_columns,
                    options->num_input_constants,
                    options->num_targets,
                    options->tile_rows,
                    options->threads_per_block,
                    options->reduction_mode,
                    routines,
                    num_routines,
                    routine_names,
                    asts,
                    device->hip_architecture,
                    compile_options,
                    sizeof(compile_options) /
                        sizeof(compile_options[0]),
                    false,
                    compile_scratch,
                    compile_scratch_size,
                    error,
                    error_size,
                    NULL,
                    &artifact->hip_compiled);
                if (result != SECANT_HIP_SUCCESS ||
                    secant_hip_compiled_binary_get(
                        artifact->hip_compiled,
                        &artifact->binary,
                        &artifact->binary_size) != SECANT_HIP_SUCCESS) {
                    snprintf(
                        error,
                        error_size,
                        "%s",
                        secant_hip_result_to_string(result));
                    return 0;
                }
                break;
            }
#endif
#if SECANT_BENCH_HAS_CUBIN
        case BENCH_BACKEND_CUBIN:
            {
                const SecantResult result = secant_cubin_specialize_into(
                    artifact->cubin_plan,
                    routines,
                    num_routines,
                    asts,
                    options->num_kernels * options->asts_per_kernel,
                    artifact->cubin_binary,
                    artifact->binary_size);

                if (result != SECANT_SUCCESS) {
                    snprintf(
                        error,
                        error_size,
                        "%s",
                        secant_result_to_string(result));
                    return 0;
                }
                break;
            }
#endif
#if SECANT_BENCH_HAS_HSACO
        case BENCH_BACKEND_HSACO:
            {
                const SecantResult result = secant_hsaco_specialize_into(
                    artifact->hsaco_plan,
                    routines,
                    num_routines,
                    asts,
                    options->num_kernels * options->asts_per_kernel,
                    artifact->hsaco_binary,
                    artifact->binary_size);

                if (result != SECANT_SUCCESS) {
                    snprintf(
                        error,
                        error_size,
                        "%s",
                        secant_result_to_string(result));
                    return 0;
                }
                break;
            }
#endif
        default:
            snprintf(error, error_size, "backend unavailable");
            return 0;
    }
    *seconds_ret = benchmark_wall_seconds() - start;
    return *seconds_ret >= 0.0;
}

static void
benchmark_artifact_destroy(BenchmarkArtifact* artifact) {
#if SECANT_BENCH_HAS_CUDA
    if (artifact->cuda_compiled != NULL) {
        (void)secant_cuda_compiled_destroy(artifact->cuda_compiled);
    }
#endif
#if SECANT_BENCH_HAS_PTX
    if (artifact->ptx_compiled != NULL) {
        (void)secant_ptx_compiled_destroy(artifact->ptx_compiled);
    }
    if (artifact->ptx_handle != NULL) {
        (void)secant_ptx_handle_destroy(artifact->ptx_handle);
    }
#endif
#if SECANT_BENCH_HAS_HIP
    if (artifact->hip_compiled != NULL) {
        (void)secant_hip_compiled_destroy(artifact->hip_compiled);
    }
#endif
#if SECANT_BENCH_HAS_CUBIN
    free(artifact->cubin_plan_storage);
    free(artifact->cubin_binary);
    free(artifact->cubin_template);
#endif
#if SECANT_BENCH_HAS_HSACO
    free(artifact->hsaco_plan_storage);
    free(artifact->hsaco_binary);
    free(artifact->hsaco_template);
#endif
}

static int
benchmark_module_load(
    const BenchmarkOptions* options,
    const BenchmarkArtifact* artifact,
    BenchmarkDevice* device,
    double* seconds_ret
) {
    const double start = benchmark_wall_seconds();

    if (benchmark_backend_is_nvidia(options->backend)) {
#if SECANT_BENCH_HAS_CUDA || SECANT_BENCH_HAS_PTX || \
    SECANT_BENCH_HAS_CUBIN
        const char* reducer_name =
            options->backend == BENCH_BACKEND_CUBIN
                ? "secant_cubin_dynamic_constant_sse_reduce"
                : "secant_dynamic_constant_sse_reduce";
        CUfunction reducer = NULL;

        if (secant_test_cuda_module_load(
                artifact->binary,
                artifact->binary_size,
                options->num_kernels,
                &device->module,
                device->functions) != SECANT_CUDA_SUCCESS) {
            return 0;
        }
        if (options->reduction_mode ==
                SECANT_SSE_REDUCTION_MODE_WORKSPACE &&
            cuModuleGetFunction(
                &reducer,
                (CUmodule)device->module,
                reducer_name) != CUDA_SUCCESS) {
            return 0;
        }
        device->reducer_function = (void*)reducer;
#else
        return 0;
#endif
    } else {
#if SECANT_BENCH_HAS_HIP || SECANT_BENCH_HAS_HSACO
        const char* prefix = options->backend == BENCH_BACKEND_HIP
            ? "secant_dynamic_constant_sse"
            : "secant_hsaco_dynamic_constant_sse";
        hipFunction_t reducer = NULL;

        if (secant_test_hip_module_load(
                artifact->binary,
                artifact->binary_size,
                prefix,
                options->num_kernels,
                &device->module,
                device->functions) != SECANT_HIP_SUCCESS) {
            return 0;
        }
        if (options->reduction_mode ==
                SECANT_SSE_REDUCTION_MODE_WORKSPACE &&
            hipModuleGetFunction(
                &reducer,
                (hipModule_t)device->module,
                options->backend == BENCH_BACKEND_HSACO
                    ? "secant_hsaco_dynamic_constant_sse_reduce"
                    : "secant_dynamic_constant_sse_reduce") != hipSuccess) {
            return 0;
        }
        device->reducer_function = (void*)reducer;
#else
        return 0;
#endif
    }
    *seconds_ret = benchmark_wall_seconds() - start;
    return *seconds_ret >= 0.0;
}

static int
benchmark_device_buffers_create(
    const BenchmarkOptions* options,
    const float* input,
    const float* constants,
    const float* targets,
    size_t input_bytes,
    size_t constants_bytes,
    size_t target_bytes,
    size_t output_bytes,
    size_t workspace_bytes,
    BenchmarkDevice* device
) {
    if (benchmark_backend_is_nvidia(options->backend)) {
#if SECANT_BENCH_HAS_CUDA || SECANT_BENCH_HAS_PTX || \
    SECANT_BENCH_HAS_CUBIN
        CUdeviceptr input_device;
        CUdeviceptr constants_device;
        CUdeviceptr targets_device;
        CUdeviceptr output_device;
        CUdeviceptr workspace_device;

        if (cuMemAlloc(&input_device, input_bytes) != CUDA_SUCCESS) {
            return 0;
        }
        device->input = (void*)(uintptr_t)input_device;
        if (cuMemAlloc(
                &constants_device,
                constants_bytes) != CUDA_SUCCESS) {
            return 0;
        }
        device->constants = (void*)(uintptr_t)constants_device;
        if (cuMemAlloc(&targets_device, target_bytes) != CUDA_SUCCESS) {
            return 0;
        }
        device->targets = (void*)(uintptr_t)targets_device;
        if (cuMemAlloc(&output_device, output_bytes) != CUDA_SUCCESS) {
            return 0;
        }
        device->output = (void*)(uintptr_t)output_device;
        if (workspace_bytes != 0u) {
            if (cuMemAlloc(
                    &workspace_device,
                    workspace_bytes) != CUDA_SUCCESS) {
                return 0;
            }
            device->workspace = (void*)(uintptr_t)workspace_device;
            device->workspace_num_elements =
                workspace_bytes / sizeof(float);
        }
        if (cuMemcpyHtoD(
                input_device,
                input,
                input_bytes) != CUDA_SUCCESS ||
            cuMemcpyHtoD(
                constants_device,
                constants,
                constants_bytes) != CUDA_SUCCESS ||
            cuMemcpyHtoD(
                targets_device,
                targets,
                target_bytes) != CUDA_SUCCESS) {
            return 0;
        }
        return 1;
#else
        return 0;
#endif
    }
#if SECANT_BENCH_HAS_HIP || SECANT_BENCH_HAS_HSACO
    if (hipMalloc(&device->input, input_bytes) != hipSuccess ||
        hipMalloc(&device->constants, constants_bytes) != hipSuccess ||
        hipMalloc(&device->targets, target_bytes) != hipSuccess ||
        hipMalloc(&device->output, output_bytes) != hipSuccess ||
        (workspace_bytes != 0u &&
         hipMalloc(&device->workspace, workspace_bytes) != hipSuccess) ||
        hipMemcpy(
            device->input,
            input,
            input_bytes,
            hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(
            device->constants,
            constants,
            constants_bytes,
            hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(
            device->targets,
            targets,
            target_bytes,
            hipMemcpyHostToDevice) != hipSuccess) {
        return 0;
    }
    device->workspace_num_elements =
        workspace_bytes / sizeof(float);
    return 1;
#else
    return 0;
#endif
}

static int
benchmark_output_clear(
    const BenchmarkOptions* options,
    BenchmarkDevice* device,
    size_t output_bytes,
    size_t workspace_bytes
) {
    if (benchmark_backend_is_nvidia(options->backend)) {
#if SECANT_BENCH_HAS_CUDA || SECANT_BENCH_HAS_PTX || \
    SECANT_BENCH_HAS_CUBIN
        return cuMemsetD8Async(
                (CUdeviceptr)(uintptr_t)device->output,
                0u,
                output_bytes,
                (CUstream)device->streams[0]) == CUDA_SUCCESS &&
            (workspace_bytes == 0u ||
             cuMemsetD8Async(
                (CUdeviceptr)(uintptr_t)device->workspace,
                0u,
                workspace_bytes,
                (CUstream)device->streams[0]) == CUDA_SUCCESS);
#else
        return 0;
#endif
    }
#if SECANT_BENCH_HAS_HIP || SECANT_BENCH_HAS_HSACO
    return hipMemsetAsync(
            device->output,
            0,
            output_bytes,
            (hipStream_t)device->streams[0]) == hipSuccess &&
        (workspace_bytes == 0u ||
         hipMemsetAsync(
            device->workspace,
            0,
            workspace_bytes,
            (hipStream_t)device->streams[0]) == hipSuccess);
#else
    return 0;
#endif
}

static int
benchmark_launch(
    const BenchmarkOptions* options,
    const BenchmarkDevice* device,
    size_t kernel_idx,
    size_t num_rows,
    size_t num_settings
) {
    const size_t output_elements =
        options->asts_per_kernel *
        options->num_targets *
        options->num_settings;
    const int workspace_mode =
        options->reduction_mode ==
            SECANT_SSE_REDUCTION_MODE_WORKSPACE;
    const size_t stream_idx = kernel_idx % options->num_streams;
    float* const output = workspace_mode
        ? (float*)device->workspace
        : (float*)((uintptr_t)device->output +
            kernel_idx * output_elements * sizeof(float));
    const size_t output_capacity = workspace_mode
        ? device->workspace_num_elements
        : output_elements;

    if (benchmark_backend_is_nvidia(options->backend)) {
#if SECANT_BENCH_HAS_CUDA || SECANT_BENCH_HAS_PTX || \
    SECANT_BENCH_HAS_CUBIN
        return secant_test_cuda_run_dynamic_constant_sse(
            device->functions[kernel_idx],
            options->num_input_columns,
            options->num_input_constants,
            options->asts_per_kernel,
            options->num_targets,
            options->tile_rows,
            options->threads_per_block,
            (const float*)device->input,
            options->num_input_columns * options->num_rows,
            options->num_rows,
            (const float*)device->constants,
            options->num_input_constants * options->num_settings,
            options->num_settings,
            num_settings,
            (const float*)device->targets,
            options->num_targets * options->num_rows,
            options->num_rows,
            num_rows,
            device->streams[stream_idx],
            output,
            output_capacity,
            options->num_settings) == SECANT_CUDA_SUCCESS;
#else
        return 0;
#endif
    }
#if SECANT_BENCH_HAS_HIP || SECANT_BENCH_HAS_HSACO
    return secant_test_hip_run_dynamic_constant_sse(
        device->functions[kernel_idx],
        options->num_input_columns,
        options->num_input_constants,
        options->asts_per_kernel,
        options->num_targets,
        options->tile_rows,
        options->threads_per_block,
        (const float*)device->input,
        options->num_input_columns * options->num_rows,
        options->num_rows,
        (const float*)device->constants,
        options->num_input_constants * options->num_settings,
        options->num_settings,
        num_settings,
        (const float*)device->targets,
        options->num_targets * options->num_rows,
        options->num_rows,
        num_rows,
        device->streams[stream_idx],
        output,
        output_capacity,
        options->num_settings) == SECANT_HIP_SUCCESS;
#else
    return 0;
#endif
}

static int
benchmark_device_synchronize(const BenchmarkOptions* options) {
    if (benchmark_backend_is_nvidia(options->backend)) {
#if SECANT_BENCH_HAS_CUDA || SECANT_BENCH_HAS_PTX || \
    SECANT_BENCH_HAS_CUBIN
        return cuCtxSynchronize() == CUDA_SUCCESS;
#else
        return 0;
#endif
    }
#if SECANT_BENCH_HAS_HIP || SECANT_BENCH_HAS_HSACO
    return hipDeviceSynchronize() == hipSuccess;
#else
    return 0;
#endif
}

static int
benchmark_reduce(
    const BenchmarkOptions* options,
    const BenchmarkDevice* device,
    size_t num_rows,
    size_t num_settings
) {
    const size_t num_tiles =
        (num_rows + options->tile_rows - 1u) /
        options->tile_rows;
    const size_t output_leading_dimension =
        options->num_settings;
    size_t num_results;
    unsigned int grid_x;

    if (options->reduction_mode !=
            SECANT_SSE_REDUCTION_MODE_WORKSPACE) {
        return 1;
    }
    if (device->reducer_function == NULL ||
        !benchmark_checked_mul(
            options->num_kernels,
            options->asts_per_kernel,
            &num_results) ||
        !benchmark_checked_mul(
            num_results,
            options->num_targets,
            &num_results) ||
        !benchmark_checked_mul(
            num_results,
            num_settings,
            &num_results) ||
        num_results >
            (size_t)UINT_MAX * 256u) {
        return 0;
    }
    grid_x = (unsigned int)((num_results + 255u) / 256u);
    if (benchmark_backend_is_nvidia(options->backend)) {
#if SECANT_BENCH_HAS_CUDA || SECANT_BENCH_HAS_PTX || \
    SECANT_BENCH_HAS_CUBIN
        const CUdeviceptr workspace =
            (CUdeviceptr)(uintptr_t)device->workspace;
        const CUdeviceptr output =
            (CUdeviceptr)(uintptr_t)device->output;
        void* args[5] = {
            (void*)&workspace,
            (void*)&num_tiles,
            (void*)&num_settings,
            (void*)&output,
            (void*)&output_leading_dimension
        };

        return cuLaunchKernel(
            (CUfunction)device->reducer_function,
            grid_x, 1u, 1u,
            256u, 1u, 1u,
            0u,
            (CUstream)device->streams[0],
            args,
            NULL) == CUDA_SUCCESS;
#else
        return 0;
#endif
    }
#if SECANT_BENCH_HAS_HIP || SECANT_BENCH_HAS_HSACO
    {
        const float* workspace = (const float*)device->workspace;
        float* output = (float*)device->output;
        void* args[5] = {
            (void*)&workspace,
            (void*)&num_tiles,
            (void*)&num_settings,
            (void*)&output,
            (void*)&output_leading_dimension
        };

        return hipModuleLaunchKernel(
            (hipFunction_t)device->reducer_function,
            grid_x, 1u, 1u,
            256u, 1u, 1u,
            0u,
            (hipStream_t)device->streams[0],
            args,
            NULL) == hipSuccess;
    }
#else
    return 0;
#endif
}

static int
benchmark_output_copy(
    const BenchmarkOptions* options,
    const BenchmarkDevice* device,
    float* output,
    size_t output_bytes
) {
    if (benchmark_backend_is_nvidia(options->backend)) {
#if SECANT_BENCH_HAS_CUDA || SECANT_BENCH_HAS_PTX || \
    SECANT_BENCH_HAS_CUBIN
        return cuMemcpyDtoH(
            output,
            (CUdeviceptr)(uintptr_t)device->output,
            output_bytes) == CUDA_SUCCESS;
#else
        return 0;
#endif
    }
#if SECANT_BENCH_HAS_HIP || SECANT_BENCH_HAS_HSACO
    return hipMemcpy(
        output,
        device->output,
        output_bytes,
        hipMemcpyDeviceToHost) == hipSuccess;
#else
    return 0;
#endif
}

static int
benchmark_verify(
    const BenchmarkOptions* options,
    size_t num_inputs,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const SecantAstInstruction* const* asts,
    const float* input,
    const float* constants,
    const float* targets,
    size_t output_bytes,
    size_t workspace_bytes,
    BenchmarkDevice* device
) {
    const size_t check_rows =
        options->num_rows < 257u ? options->num_rows : 257u;
    const size_t check_settings =
        options->num_settings < 4u ? options->num_settings : 4u;
    const size_t output_series =
        options->asts_per_kernel * options->num_targets;
    const size_t output_elements =
        output_series * options->num_settings;
    const float relative_tolerance =
        options->ast_mode == SECANT_BENCH_AST_MODE_MUFU
            ? 5.0e-2f
            : 1.0e-3f;
    float* combined = (float*)malloc(
        num_inputs * check_rows * sizeof(float));
    float* setting_expected =
        (float*)malloc(output_series * sizeof(float));
    float* expected =
        (float*)calloc(output_elements, sizeof(float));
    float* actual = (float*)malloc(output_elements * sizeof(float));
    size_t setting;
    int passed = 1;

    if (combined == NULL || setting_expected == NULL ||
        expected == NULL || actual == NULL) {
        passed = 0;
    }
    for (setting = 0u; setting < check_settings && passed; ++setting) {
        size_t input_idx;
        size_t output_idx;

        for (input_idx = 0u;
             input_idx < options->num_input_columns;
             ++input_idx) {
            memcpy(
                combined + input_idx * check_rows,
                input + input_idx * options->num_rows,
                check_rows * sizeof(float));
        }
        for (input_idx = 0u;
             input_idx < options->num_input_constants;
             ++input_idx) {
            const float value =
                constants[input_idx * options->num_settings + setting];
            size_t row;

            for (row = 0u; row < check_rows; ++row) {
                combined[
                    (options->num_input_columns + input_idx) *
                        check_rows +
                    row] = value;
            }
        }
        memset(
            setting_expected,
            0,
            output_series * sizeof(float));
        if (secant_cpu_run_static_column_sse(
                num_inputs,
                options->num_targets,
                routines,
                num_routines,
                asts,
                options->asts_per_kernel,
                combined,
                num_inputs * check_rows,
                check_rows,
                targets,
                options->num_targets * options->num_rows,
                options->num_rows,
                check_rows,
                setting_expected,
                output_series,
                options->num_targets) != SECANT_SUCCESS) {
            passed = 0;
        }
        for (output_idx = 0u;
             output_idx < output_series;
             ++output_idx) {
            expected[output_idx * options->num_settings + setting] =
                setting_expected[output_idx];
        }
    }
    if (passed &&
        (!benchmark_output_clear(
             options,
             device,
             output_bytes,
             workspace_bytes) ||
         !benchmark_launch(
             options,
             device,
             0u,
             check_rows,
             check_settings) ||
         !benchmark_reduce(
             options,
             device,
             check_rows,
             check_settings) ||
         !benchmark_device_synchronize(options) ||
         !benchmark_output_copy(
             options,
             device,
             actual,
             output_elements * sizeof(float)))) {
        passed = 0;
    }
    for (setting = 0u; setting < check_settings && passed; ++setting) {
        size_t output_idx;

        for (output_idx = 0u;
             output_idx < output_series;
             ++output_idx) {
            const size_t idx =
                output_idx * options->num_settings + setting;
            const float tolerance =
                relative_tolerance * (1.0f + fabsf(expected[idx]));

            if (!isfinite(actual[idx]) ||
                !isfinite(expected[idx]) ||
                fabsf(actual[idx] - expected[idx]) > tolerance) {
                fprintf(
                    stderr,
                    "verify mismatch setting=%zu output=%zu "
                    "actual=%.9g expected=%.9g tolerance=%.9g\n",
                    setting,
                    output_idx,
                    actual[idx],
                    expected[idx],
                    tolerance);
                passed = 0;
                break;
            }
        }
    }
    free(actual);
    free(expected);
    free(setting_expected);
    free(combined);
    return passed;
}

static int
benchmark_reduce_after_evaluations(
    const BenchmarkOptions* options,
    const BenchmarkDevice* device,
    size_t num_rows,
    size_t num_settings
) {
    size_t stream_idx;

    if (options->reduction_mode !=
            SECANT_SSE_REDUCTION_MODE_WORKSPACE) {
        return 1;
    }
    if (benchmark_backend_is_nvidia(options->backend)) {
#if SECANT_BENCH_HAS_CUDA || SECANT_BENCH_HAS_PTX || \
    SECANT_BENCH_HAS_CUBIN
        for (stream_idx = 0u;
             stream_idx < options->num_streams;
             ++stream_idx) {
            if (cuEventRecord(
                    (CUevent)device->done_events[stream_idx],
                    (CUstream)device->streams[stream_idx]) !=
                CUDA_SUCCESS) {
                return 0;
            }
        }
        for (stream_idx = 1u;
             stream_idx < options->num_streams;
             ++stream_idx) {
            if (cuStreamWaitEvent(
                    (CUstream)device->streams[0],
                    (CUevent)device->done_events[stream_idx],
                    0u) != CUDA_SUCCESS) {
                return 0;
            }
        }
        if (!benchmark_reduce(
                options,
                device,
                num_rows,
                num_settings) ||
            cuEventRecord(
                (CUevent)device->reduction_done_event,
                (CUstream)device->streams[0]) != CUDA_SUCCESS) {
            return 0;
        }
        for (stream_idx = 1u;
             stream_idx < options->num_streams;
             ++stream_idx) {
            if (cuStreamWaitEvent(
                    (CUstream)device->streams[stream_idx],
                    (CUevent)device->reduction_done_event,
                    0u) != CUDA_SUCCESS) {
                return 0;
            }
        }
        return 1;
#else
        return 0;
#endif
    }
#if SECANT_BENCH_HAS_HIP || SECANT_BENCH_HAS_HSACO
    for (stream_idx = 0u;
         stream_idx < options->num_streams;
         ++stream_idx) {
        if (hipEventRecord(
                (hipEvent_t)device->done_events[stream_idx],
                (hipStream_t)device->streams[stream_idx]) != hipSuccess) {
            return 0;
        }
    }
    for (stream_idx = 1u;
         stream_idx < options->num_streams;
         ++stream_idx) {
        if (hipStreamWaitEvent(
                (hipStream_t)device->streams[0],
                (hipEvent_t)device->done_events[stream_idx],
                0u) != hipSuccess) {
            return 0;
        }
    }
    if (!benchmark_reduce(
            options,
            device,
            num_rows,
            num_settings) ||
        hipEventRecord(
            (hipEvent_t)device->reduction_done_event,
            (hipStream_t)device->streams[0]) != hipSuccess) {
        return 0;
    }
    for (stream_idx = 1u;
         stream_idx < options->num_streams;
         ++stream_idx) {
        if (hipStreamWaitEvent(
                (hipStream_t)device->streams[stream_idx],
                (hipEvent_t)device->reduction_done_event,
                0u) != hipSuccess) {
            return 0;
        }
    }
    return 1;
#else
    return 0;
#endif
}

static int
benchmark_time_cuda(
    const BenchmarkOptions* options,
    BenchmarkDevice* device,
    size_t output_bytes,
    size_t workspace_bytes,
    double* seconds_ret
) {
#if SECANT_BENCH_HAS_CUDA || SECANT_BENCH_HAS_PTX || \
    SECANT_BENCH_HAS_CUBIN
    float milliseconds = 0.0f;
    size_t iteration;
    size_t kernel_idx;
    size_t stream_idx;

    for (iteration = 0u; iteration < options->warmups; ++iteration) {
        for (kernel_idx = 0u;
             kernel_idx < options->num_kernels;
             ++kernel_idx) {
            if (!benchmark_launch(
                    options,
                    device,
                    kernel_idx,
                    options->num_rows,
                    options->num_settings)) {
                return 0;
            }
        }
        if (!benchmark_reduce_after_evaluations(
                options,
                device,
                options->num_rows,
                options->num_settings)) {
            return 0;
        }
    }
    if (!benchmark_device_synchronize(options) ||
        !benchmark_output_clear(
            options,
            device,
            output_bytes,
            workspace_bytes) ||
        cuEventRecord(
            (CUevent)device->start_event,
            (CUstream)device->streams[0]) != CUDA_SUCCESS) {
        return 0;
    }
    for (stream_idx = 1u;
         stream_idx < options->num_streams;
         ++stream_idx) {
        if (cuStreamWaitEvent(
                (CUstream)device->streams[stream_idx],
                (CUevent)device->start_event,
                0u) != CUDA_SUCCESS) {
            return 0;
        }
    }
    for (iteration = 0u;
         iteration < options->iterations;
         ++iteration) {
        for (kernel_idx = 0u;
             kernel_idx < options->num_kernels;
             ++kernel_idx) {
            if (!benchmark_launch(
                    options,
                    device,
                    kernel_idx,
                    options->num_rows,
                    options->num_settings)) {
                return 0;
            }
        }
        if (!benchmark_reduce_after_evaluations(
                options,
                device,
                options->num_rows,
                options->num_settings)) {
            return 0;
        }
    }
    for (stream_idx = 0u;
         stream_idx < options->num_streams;
         ++stream_idx) {
        if (cuEventRecord(
                (CUevent)device->done_events[stream_idx],
                (CUstream)device->streams[stream_idx]) != CUDA_SUCCESS) {
            return 0;
        }
    }
    for (stream_idx = 1u;
         stream_idx < options->num_streams;
         ++stream_idx) {
        if (cuStreamWaitEvent(
                (CUstream)device->streams[0],
                (CUevent)device->done_events[stream_idx],
                0u) != CUDA_SUCCESS) {
            return 0;
        }
    }
    if (cuEventRecord(
            (CUevent)device->stop_event,
            (CUstream)device->streams[0]) != CUDA_SUCCESS ||
        cuEventSynchronize(
            (CUevent)device->stop_event) != CUDA_SUCCESS ||
        cuEventElapsedTime(
            &milliseconds,
            (CUevent)device->start_event,
            (CUevent)device->stop_event) != CUDA_SUCCESS ||
        milliseconds <= 0.0f) {
        return 0;
    }
    *seconds_ret = (double)milliseconds * 1.0e-3;
    return 1;
#else
    (void)options;
    (void)device;
    (void)output_bytes;
    (void)workspace_bytes;
    (void)seconds_ret;
    return 0;
#endif
}

static int
benchmark_time_hip(
    const BenchmarkOptions* options,
    BenchmarkDevice* device,
    size_t output_bytes,
    size_t workspace_bytes,
    double* seconds_ret
) {
#if SECANT_BENCH_HAS_HIP || SECANT_BENCH_HAS_HSACO
    float milliseconds = 0.0f;
    size_t iteration;
    size_t kernel_idx;
    size_t stream_idx;

    for (iteration = 0u; iteration < options->warmups; ++iteration) {
        for (kernel_idx = 0u;
             kernel_idx < options->num_kernels;
             ++kernel_idx) {
            if (!benchmark_launch(
                    options,
                    device,
                    kernel_idx,
                    options->num_rows,
                    options->num_settings)) {
                return 0;
            }
        }
        if (!benchmark_reduce_after_evaluations(
                options,
                device,
                options->num_rows,
                options->num_settings)) {
            return 0;
        }
    }
    if (!benchmark_device_synchronize(options) ||
        !benchmark_output_clear(
            options,
            device,
            output_bytes,
            workspace_bytes) ||
        hipEventRecord(
            (hipEvent_t)device->start_event,
            (hipStream_t)device->streams[0]) != hipSuccess) {
        return 0;
    }
    for (stream_idx = 1u;
         stream_idx < options->num_streams;
         ++stream_idx) {
        if (hipStreamWaitEvent(
                (hipStream_t)device->streams[stream_idx],
                (hipEvent_t)device->start_event,
                0u) != hipSuccess) {
            return 0;
        }
    }
    for (iteration = 0u;
         iteration < options->iterations;
         ++iteration) {
        for (kernel_idx = 0u;
             kernel_idx < options->num_kernels;
             ++kernel_idx) {
            if (!benchmark_launch(
                    options,
                    device,
                    kernel_idx,
                    options->num_rows,
                    options->num_settings)) {
                return 0;
            }
        }
        if (!benchmark_reduce_after_evaluations(
                options,
                device,
                options->num_rows,
                options->num_settings)) {
            return 0;
        }
    }
    for (stream_idx = 0u;
         stream_idx < options->num_streams;
         ++stream_idx) {
        if (hipEventRecord(
                (hipEvent_t)device->done_events[stream_idx],
                (hipStream_t)device->streams[stream_idx]) != hipSuccess) {
            return 0;
        }
    }
    for (stream_idx = 1u;
         stream_idx < options->num_streams;
         ++stream_idx) {
        if (hipStreamWaitEvent(
                (hipStream_t)device->streams[0],
                (hipEvent_t)device->done_events[stream_idx],
                0u) != hipSuccess) {
            return 0;
        }
    }
    if (hipEventRecord(
            (hipEvent_t)device->stop_event,
            (hipStream_t)device->streams[0]) != hipSuccess ||
        hipEventSynchronize(
            (hipEvent_t)device->stop_event) != hipSuccess ||
        hipEventElapsedTime(
            &milliseconds,
            (hipEvent_t)device->start_event,
            (hipEvent_t)device->stop_event) != hipSuccess ||
        milliseconds <= 0.0f) {
        return 0;
    }
    *seconds_ret = (double)milliseconds * 1.0e-3;
    return 1;
#else
    (void)options;
    (void)device;
    (void)output_bytes;
    (void)workspace_bytes;
    (void)seconds_ret;
    return 0;
#endif
}

static int
benchmark_time(
    const BenchmarkOptions* options,
    BenchmarkDevice* device,
    size_t output_bytes,
    size_t workspace_bytes,
    double* seconds_ret
) {
    return benchmark_backend_is_nvidia(options->backend)
        ? benchmark_time_cuda(
            options,
            device,
            output_bytes,
            workspace_bytes,
            seconds_ret)
        : benchmark_time_hip(
            options,
            device,
            output_bytes,
            workspace_bytes,
            seconds_ret);
}

static void
benchmark_resource_query(
    const BenchmarkOptions* options,
    BenchmarkDevice* device
) {
    if (benchmark_backend_is_nvidia(options->backend)) {
#if SECANT_BENCH_HAS_CUDA || SECANT_BENCH_HAS_PTX || \
    SECANT_BENCH_HAS_CUBIN
        (void)cuFuncGetAttribute(
            &device->registers,
            CU_FUNC_ATTRIBUTE_NUM_REGS,
            (CUfunction)device->functions[0]);
        (void)cuFuncGetAttribute(
            &device->static_shared_bytes,
            CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
            (CUfunction)device->functions[0]);
        (void)cuFuncGetAttribute(
            &device->local_bytes,
            CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
            (CUfunction)device->functions[0]);
        (void)cuOccupancyMaxActiveBlocksPerMultiprocessor(
            &device->active_blocks,
            (CUfunction)device->functions[0],
            (int)options->threads_per_block,
            0u);
#endif
    } else {
#if SECANT_BENCH_HAS_HIP || SECANT_BENCH_HAS_HSACO
        (void)hipFuncGetAttribute(
            &device->registers,
            HIP_FUNC_ATTRIBUTE_NUM_REGS,
            (hipFunction_t)device->functions[0]);
        (void)hipFuncGetAttribute(
            &device->static_shared_bytes,
            HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
            (hipFunction_t)device->functions[0]);
        (void)hipFuncGetAttribute(
            &device->local_bytes,
            HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
            (hipFunction_t)device->functions[0]);
        (void)hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &device->active_blocks,
            (hipFunction_t)device->functions[0],
            (int)options->threads_per_block,
            0u);
#endif
    }
}

int
main(int argc, char** argv) {
    BenchmarkOptions options;
    BenchmarkArtifact artifact;
    BenchmarkDevice device;
    SecantAstInstruction* programs = NULL;
    const SecantAstInstruction** asts = NULL;
    const SecantAstInstruction* const* routines = NULL;
    const char* const* routine_names = NULL;
    float* input = NULL;
    float* constants = NULL;
    float* targets = NULL;
    unsigned char* compile_scratch = NULL;
    char error[BENCH_ERROR_BYTES] = { 0 };
    size_t program_bytes;
    size_t pointer_bytes;
    size_t num_inputs;
    size_t num_asts;
    size_t output_per_kernel;
    size_t output_elements;
    size_t input_elements;
    size_t constants_elements;
    size_t target_elements;
    size_t input_bytes;
    size_t constants_bytes;
    size_t target_bytes;
    size_t output_bytes;
    size_t num_tiles;
    size_t workspace_elements = 0u;
    size_t workspace_bytes = 0u;
    size_t num_routines = 0u;
    size_t idx;
    double compile_seconds = 0.0;
    double load_seconds = 0.0;
    double run_seconds = 0.0;
    double row_evals;
    double atomic_updates;
    int parse_result;
    int success = 1;

    memset(&artifact, 0, sizeof(artifact));
    memset(&device, 0, sizeof(device));
    benchmark_options_default(&options);
    parse_result = benchmark_parse_options(argc, argv, &options);
    if (parse_result < 0) {
        benchmark_usage(argv[0]);
        return 0;
    }
    if (options.tile_rows == 0u) {
        if (options.backend == BENCH_BACKEND_HIP) {
            options.tile_rows = 1024u;
        } else if (options.backend == BENCH_BACKEND_HSACO) {
            options.tile_rows = 256u;
        } else {
            options.tile_rows = 64u;
        }
    }
    if (options.warmups == 0u) {
        options.warmups =
            options.backend == BENCH_BACKEND_HIP ||
            options.backend == BENCH_BACKEND_HSACO
                ? 100u
                : 3u;
    }
    num_tiles =
        options.num_rows / options.tile_rows +
        (options.num_rows % options.tile_rows != 0u);
    if (parse_result == 0 ||
        !benchmark_options_valid(
            &options,
            &num_inputs,
            &num_asts,
            &output_per_kernel) ||
        !secant_bench_ast_storage_sizes(
            1u,
            options.num_kernels,
            options.asts_per_kernel,
            &program_bytes,
            &pointer_bytes) ||
        !benchmark_checked_mul(
            options.num_kernels,
            output_per_kernel,
            &output_elements) ||
        !benchmark_checked_mul(
            options.num_input_columns,
            options.num_rows,
            &input_elements) ||
        !benchmark_checked_mul(
            options.num_input_constants,
            options.num_settings,
            &constants_elements) ||
        !benchmark_checked_mul(
            options.num_targets,
            options.num_rows,
            &target_elements) ||
        !benchmark_checked_mul(
            input_elements,
            sizeof(float),
            &input_bytes) ||
        !benchmark_checked_mul(
            constants_elements,
            sizeof(float),
            &constants_bytes) ||
        !benchmark_checked_mul(
            target_elements,
            sizeof(float),
            &target_bytes) ||
        !benchmark_checked_mul(
            output_elements,
            sizeof(float),
            &output_bytes) ||
        (options.reduction_mode ==
             SECANT_SSE_REDUCTION_MODE_WORKSPACE &&
         (!benchmark_checked_mul(
              output_elements,
              num_tiles,
              &workspace_elements) ||
          !benchmark_checked_mul(
              workspace_elements,
              sizeof(float),
              &workspace_bytes)))) {
        benchmark_usage(argv[0]);
        return 1;
    }
    programs = (SecantAstInstruction*)malloc(program_bytes);
    asts = (const SecantAstInstruction**)malloc(pointer_bytes);
    input = (float*)malloc(input_bytes);
    constants = (float*)malloc(constants_bytes);
    targets = (float*)malloc(target_bytes);
    compile_scratch = (unsigned char*)malloc(
        BENCH_COMPILE_SCRATCH_BYTES);
    if (programs == NULL || asts == NULL || input == NULL ||
        constants == NULL || targets == NULL ||
        compile_scratch == NULL) {
        success = 0;
    }
    if (success) {
        secant_bench_ast_fill(
            1u,
            options.num_kernels,
            options.asts_per_kernel,
            num_inputs,
            options.seed,
            options.ast_mode,
            programs,
            asts);
        secant_bench_ast_get_routines(
            options.ast_mode,
            &routines,
            &num_routines,
            &routine_names);
        for (idx = 0u; idx < input_elements; ++idx) {
            input[idx] = benchmark_value(
                idx / options.num_rows,
                idx % options.num_rows,
                options.seed);
        }
        for (idx = 0u; idx < constants_elements; ++idx) {
            constants[idx] = benchmark_value(
                idx / options.num_settings,
                idx % options.num_settings,
                options.seed + 17u);
        }
        for (idx = 0u; idx < target_elements; ++idx) {
            targets[idx] = benchmark_value(
                idx / options.num_rows,
                idx % options.num_rows,
                options.seed + 31u);
        }
        success = benchmark_device_create(&options, &device);
    }
    if (success) {
        success = benchmark_artifact_template_prepare(
            &options,
            &device,
            &artifact,
            error,
            sizeof(error));
    }
    if (success) {
        success = benchmark_artifact_compile(
            &options,
            &device,
            routines,
            num_routines,
            routine_names,
            asts,
            compile_scratch,
            BENCH_COMPILE_SCRATCH_BYTES,
            &artifact,
            error,
            sizeof(error),
            &compile_seconds);
    }
    if (success) {
        success = benchmark_module_load(
            &options,
            &artifact,
            &device,
            &load_seconds);
    }
    if (success) {
        success = benchmark_device_buffers_create(
            &options,
            input,
            constants,
            targets,
            input_bytes,
            constants_bytes,
            target_bytes,
            output_bytes,
            workspace_bytes,
            &device);
    }
    if (success) {
        success = benchmark_verify(
            &options,
            num_inputs,
            routines,
            num_routines,
            asts,
            input,
            constants,
            targets,
            output_bytes,
            workspace_bytes,
            &device);
        if (!success) {
            snprintf(error, sizeof(error), "CPU verification failed");
        }
    }
    if (success) {
        benchmark_resource_query(&options, &device);
        success = benchmark_time(
            &options,
            &device,
            output_bytes,
            workspace_bytes,
            &run_seconds);
        if (!success) {
            snprintf(error, sizeof(error), "runtime timing failed");
        }
    }
    if (success) {
        row_evals =
            (double)options.iterations *
            (double)num_asts *
            (double)options.num_settings *
            (double)options.num_rows;
        atomic_updates =
            options.reduction_mode ==
                SECANT_SSE_REDUCTION_MODE_ATOMIC
            ?
            (double)options.iterations *
            (double)num_asts *
            (double)options.num_targets *
            (double)options.num_settings *
            (double)num_tiles
            : 0.0;
        printf(
            "backend=%s shape=dynamic_constant_sse ast_mode=%s "
            "reduction=%s "
            "kernels=%zu asts_per_kernel=%zu asts=%zu "
            "columns=%zu constants=%zu targets=%zu settings=%zu "
            "rows=%zu tile_rows=%zu threads=%zu streams=%zu "
            "iterations=%zu template_prepare_seconds=%.6f "
            "compile_seconds=%.6f compile_asts_per_second=%.3f "
            "module_load_seconds=%.6f runtime_seconds=%.6f "
            "row_evals=%.0f row_evals_per_second=%.3e "
            "workspace_bytes=%zu tile_partials=%.0f "
            "atomic_updates=%.0f atomic_updates_per_second=%.3e "
            "registers=%d static_shared_bytes=%d local_bytes=%d "
            "active_blocks_per_sm=%d verify=pass\n",
            benchmark_backend_name(options.backend),
            benchmark_ast_mode_name(options.ast_mode),
            benchmark_reduction_mode_name(options.reduction_mode),
            options.num_kernels,
            options.asts_per_kernel,
            num_asts,
            options.num_input_columns,
            options.num_input_constants,
            options.num_targets,
            options.num_settings,
            options.num_rows,
            options.tile_rows,
            options.threads_per_block,
            options.num_streams,
            options.iterations,
            artifact.template_prepare_seconds,
            compile_seconds,
            (double)num_asts / compile_seconds,
            load_seconds,
            run_seconds,
            row_evals,
            row_evals / run_seconds,
            workspace_bytes,
            (double)options.iterations *
                (double)output_elements *
                (double)num_tiles,
            atomic_updates,
            atomic_updates / run_seconds,
            device.registers,
            device.static_shared_bytes,
            device.local_bytes,
            device.active_blocks);
    } else {
        fprintf(
            stderr,
            "dynamic_constant_sse backend=%s failed: %s\n",
            benchmark_backend_name(options.backend),
            error[0] != '\0' ? error : "unknown error");
    }
    benchmark_device_destroy(&options, &device);
    benchmark_artifact_destroy(&artifact);
    free(compile_scratch);
    free(targets);
    free(constants);
    free(input);
    free(asts);
    free(programs);
    return success ? 0 : 1;
}
