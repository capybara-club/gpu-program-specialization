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

#include "bench_common.h"
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
#ifndef SECANT_BENCH_HAS_HSACO
#define SECANT_BENCH_HAS_HSACO 0
#endif
#ifndef SECANT_BENCH_HAS_CUBIN
#define SECANT_BENCH_HAS_CUBIN 0
#endif
#ifndef SECANT_BENCH_HAS_EMBEDDED_CUBIN
#define SECANT_BENCH_HAS_EMBEDDED_CUBIN 0
#endif
#ifndef SECANT_BENCH_HAS_OPENMP
#define SECANT_BENCH_HAS_OPENMP 0
#endif
#ifndef SECANT_BENCH_PIPELINE
#define SECANT_BENCH_PIPELINE 0
#endif

#if SECANT_BENCH_HAS_CUDA || SECANT_BENCH_HAS_PTX || SECANT_BENCH_HAS_CUBIN
#include "secant_cuda.h"
#include "cuda_runtime.h"
#include <cuda.h>
#endif
#if SECANT_BENCH_HAS_PTX
#include "secant_ptx.h"
#endif
#if SECANT_BENCH_HAS_CUBIN
#include "bench_cubin_template.h"
#include "secant_cubin.h"
#endif
#if SECANT_BENCH_PIPELINE && SECANT_BENCH_HAS_CUDA
#include "secant_cuda_runner.h"
#endif
#if SECANT_BENCH_PIPELINE && SECANT_BENCH_HAS_PTX
#include "secant_ptx_runner.h"
#endif
#if SECANT_BENCH_PIPELINE && SECANT_BENCH_HAS_CUBIN
#include "secant_cubin_runner.h"
#endif
#if SECANT_BENCH_PIPELINE && SECANT_BENCH_HAS_HIP
#include "secant_hip_runner.h"
#endif
#if SECANT_BENCH_PIPELINE && SECANT_BENCH_HAS_HSACO
#include "secant_hsaco_runner.h"
#endif
#if SECANT_BENCH_HAS_EMBEDDED_CUBIN
#include "bench_cubin_embedded.h"
#endif
#if SECANT_BENCH_HAS_HIP || SECANT_BENCH_HAS_HSACO
#include "secant_hip.h"
#include "hip_runtime.h"
#include <hip/hip_runtime_api.h>
#endif
#if SECANT_BENCH_HAS_HSACO
#include "bench_hsaco_template.h"
#include "secant_hsaco.h"
#endif
#if SECANT_BENCH_HAS_OPENMP
#include <omp.h>
#endif
#if SECANT_BENCH_PIPELINE
#include "secant_runner.h"
#endif

typedef struct SecantCompileArtifact {
#if SECANT_BENCH_HAS_CUDA
    SecantCUDACompiled cuda;
#endif
#if SECANT_BENCH_HAS_PTX
    SecantPTXCompiled ptx;
#endif
#if SECANT_BENCH_HAS_HIP
    SecantHIPCompiled hip;
#endif
#if SECANT_BENCH_HAS_CUBIN
    unsigned char* cubin;
    size_t cubin_size;
#endif
#if SECANT_BENCH_HAS_HSACO
    unsigned char* hsaco;
    size_t hsaco_size;
#endif
} SecantCompileArtifact;

typedef struct SecantCompileContext {
    SecantBenchOptions options;
    SecantAstInstruction* programs;
    const SecantAstInstruction** asts;
    const SecantAstInstruction* const* routines;
    const char* const* routine_names;
    size_t num_routines;
    unsigned char* worker_scratch;
    SecantCompileArtifact* artifacts;
#if SECANT_BENCH_HAS_PTX
    SecantPTXHandle ptx_handle;
#endif
#if SECANT_BENCH_HAS_HIP || SECANT_BENCH_HAS_HSACO
    char hip_architecture[64];
#endif
#if SECANT_BENCH_HAS_CUBIN
    SecantCubinPlan* cubin_plan;
    void* cubin_plan_storage;
    const unsigned char* cubin_template;
    size_t cubin_size;
    int cubin_template_owned;
#endif
#if SECANT_BENCH_HAS_HSACO
    SecantHsacoPlan* hsaco_plan;
    void* hsaco_plan_storage;
    unsigned char* hsaco_template;
    size_t hsaco_size;
    uint32_t gfx_arch;
#endif
} SecantCompileContext;

#if !SECANT_BENCH_PIPELINE
static double
secant_compile_seconds(void) {
    struct timespec time;

    if (clock_gettime(CLOCK_MONOTONIC, &time) != 0) {
        return 0.0;
    }
    return (double)time.tv_sec + (double)time.tv_nsec * 1.0e-9;
}
#endif

#if !SECANT_BENCH_PIPELINE
static void
secant_compile_usage(const char* program) {
    fprintf(
        stderr,
        "usage: %s [options]\n"
        "  --backend cuda|ptx|hip|cubin|hsaco\n"
        "  --shape materialize|sse\n"
        "  --ast-mode simple|alu|mufu\n"
        "  --warmups N --iterations N --workers N\n"
        "  --kernels N --asts-per-kernel N --inputs N --targets N\n"
        "  --tile-rows N --threads N --patch-instructions-per-ast N\n"
        "  --compile-scratch-bytes N\n"
        "  --check-rows N --check-modules N\n"
        "  --source-sm NN --target-sm NN --opt-level 0|1\n"
        "  --hip-arch gfxNNNN --device N\n",
        program);
}
#endif

static int
secant_compile_backend_available(SecantBenchBackend backend) {
    switch (backend) {
        case SECANT_BENCH_BACKEND_CUDA: return SECANT_BENCH_HAS_CUDA;
        case SECANT_BENCH_BACKEND_PTX: return SECANT_BENCH_HAS_PTX;
        case SECANT_BENCH_BACKEND_HIP: return SECANT_BENCH_HAS_HIP;
        case SECANT_BENCH_BACKEND_CUBIN: return SECANT_BENCH_HAS_CUBIN;
        case SECANT_BENCH_BACKEND_HSACO: return SECANT_BENCH_HAS_HSACO;
        default: return 0;
    }
}

static size_t
secant_compile_worker_idx(void) {
    size_t worker_idx = 0u;

#if SECANT_BENCH_HAS_OPENMP
    if (omp_in_parallel()) {
        worker_idx = (size_t)omp_get_thread_num();
    }
#endif
    return worker_idx;
}

static void*
secant_compile_worker_scratch(SecantCompileContext* context) {
    return context->worker_scratch +
        secant_compile_worker_idx() * context->options.compile_scratch_size;
}

static const SecantAstInstruction* const*
secant_compile_module_asts(
    const SecantCompileContext* context,
    size_t module_idx
) {
    const size_t asts_per_module =
        context->options.num_kernels * context->options.asts_per_kernel;

    return context->asts + module_idx * asts_per_module;
}

#if SECANT_BENCH_HAS_CUBIN
static int
secant_compile_prepare_cubin_artifact(
    SecantCompileContext* context,
    SecantCompileArtifact* artifact
) {
    if (artifact->cubin != NULL) {
        return artifact->cubin_size == context->cubin_size;
    }
    artifact->cubin = (unsigned char*)malloc(context->cubin_size);
    if (artifact->cubin == NULL) {
        return 0;
    }
    memcpy(
        artifact->cubin,
        context->cubin_template,
        context->cubin_size);
    artifact->cubin_size = context->cubin_size;
    return 1;
}
#endif

#if SECANT_BENCH_HAS_HSACO
static int
secant_compile_prepare_hsaco_artifact(
    SecantCompileContext* context,
    SecantCompileArtifact* artifact
) {
    if (artifact->hsaco != NULL) {
        return artifact->hsaco_size == context->hsaco_size;
    }
    artifact->hsaco = (unsigned char*)malloc(context->hsaco_size);
    if (artifact->hsaco == NULL) {
        return 0;
    }
    memcpy(
        artifact->hsaco,
        context->hsaco_template,
        context->hsaco_size);
    artifact->hsaco_size = context->hsaco_size;
    return 1;
}
#endif

static int
secant_compile_one(
    SecantCompileContext* context,
    size_t module_idx,
    SecantCompileArtifact* artifact,
    char* error,
    size_t error_size
) {
    const SecantBenchOptions* options = &context->options;
    const SecantAstInstruction* const* asts =
        secant_compile_module_asts(context, module_idx);
    void* scratch =
        options->backend == SECANT_BENCH_BACKEND_CUBIN ||
        options->backend == SECANT_BENCH_BACKEND_HSACO
        ? NULL
        : secant_compile_worker_scratch(context);

    switch (options->backend) {
#if SECANT_BENCH_HAS_CUDA
        case SECANT_BENCH_BACKEND_CUDA:
            {
                const char* compile_options[] = {
                    "--no-cache",
                    "--split-compile=1",
                    NULL
                };
                char opt_level[32];
                const uint32_t major = options->target_sm / 10u;
                const uint32_t minor = options->target_sm % 10u;
                SecantCUDAResult result;

                snprintf(
                    opt_level,
                    sizeof(opt_level),
                    "--ptxas-options=-O%u",
                    options->opt_level);
                compile_options[2] = opt_level;
                if (options->shape == SECANT_BENCH_SHAPE_MATERIALIZE) {
                    result = secant_cuda_materialize_compile(
                        options->num_kernels,
                        options->asts_per_kernel,
                        options->num_inputs,
                        context->routines,
                        context->num_routines,
                        context->routine_names,
                        asts,
                        major,
                        minor,
                        compile_options,
                        sizeof(compile_options) / sizeof(compile_options[0]),
                        false,
                        scratch,
                        options->compile_scratch_size,
                        NULL,
                        0u,
                        NULL,
                        &artifact->cuda);
                } else {
                    result = secant_cuda_sse_compile(
                        options->num_kernels,
                        options->asts_per_kernel,
                        options->num_inputs,
                        options->num_targets,
                        options->tile_rows,
                        options->threads_per_block,
                        SECANT_SSE_REDUCTION_MODE_ATOMIC,
                        context->routines,
                        context->num_routines,
                        context->routine_names,
                        asts,
                        major,
                        minor,
                        compile_options,
                        sizeof(compile_options) / sizeof(compile_options[0]),
                        false,
                        scratch,
                        options->compile_scratch_size,
                        NULL,
                        0u,
                        NULL,
                        &artifact->cuda);
                }
                if (result != SECANT_CUDA_SUCCESS) {
                    snprintf(error, error_size, "%s", secant_cuda_result_to_string(result));
                    return 0;
                }
                return 1;
            }
#endif
#if SECANT_BENCH_HAS_PTX
        case SECANT_BENCH_BACKEND_PTX:
            {
                const char* compile_options[] = {
                    NULL,
                    "--split-compile=1"
                };
                char opt_level[32];
                const uint32_t major = options->target_sm / 10u;
                const uint32_t minor = options->target_sm % 10u;
                SecantPTXResult result;

                snprintf(
                    opt_level,
                    sizeof(opt_level),
                    "--opt-level=%u",
                    options->opt_level);
                compile_options[0] = opt_level;
                result = secant_ptx_compile(
                    context->ptx_handle,
                    context->routines,
                    context->num_routines,
                    asts,
                    major,
                    minor,
                    compile_options,
                    sizeof(compile_options) / sizeof(compile_options[0]),
                    false,
                    scratch,
                    options->compile_scratch_size,
                    NULL,
                    0u,
                    NULL,
                    &artifact->ptx);

                if (result != SECANT_PTX_SUCCESS) {
                    snprintf(error, error_size, "%s", secant_ptx_result_to_string(result));
                    return 0;
                }
                return 1;
            }
#endif
#if SECANT_BENCH_HAS_HIP
        case SECANT_BENCH_BACKEND_HIP:
            {
                const char* compile_options[] = { NULL };
                char opt_level[16];
                SecantHIPResult result;

                snprintf(opt_level, sizeof(opt_level), "-O%u", options->opt_level);
                compile_options[0] = opt_level;
                if (options->shape == SECANT_BENCH_SHAPE_MATERIALIZE) {
                    result = secant_hip_materialize_compile(
                        options->num_kernels,
                        options->asts_per_kernel,
                        options->num_inputs,
                        context->routines,
                        context->num_routines,
                        context->routine_names,
                        asts,
                        options->hip_architecture,
                        compile_options,
                        sizeof(compile_options) / sizeof(compile_options[0]),
                        false,
                        scratch,
                        options->compile_scratch_size,
                        NULL,
                        0u,
                        NULL,
                        &artifact->hip);
                } else {
                    result = secant_hip_sse_compile(
                        options->num_kernels,
                        options->asts_per_kernel,
                        options->num_inputs,
                        options->num_targets,
                        options->tile_rows,
                        options->threads_per_block,
                        SECANT_SSE_REDUCTION_MODE_ATOMIC,
                        context->routines,
                        context->num_routines,
                        context->routine_names,
                        asts,
                        options->hip_architecture,
                        compile_options,
                        sizeof(compile_options) / sizeof(compile_options[0]),
                        false,
                        scratch,
                        options->compile_scratch_size,
                        NULL,
                        0u,
                        NULL,
                        &artifact->hip);
                }
                if (result != SECANT_HIP_SUCCESS) {
                    snprintf(error, error_size, "%s", secant_hip_result_to_string(result));
                    return 0;
                }
                return 1;
            }
#endif
#if SECANT_BENCH_HAS_CUBIN
        case SECANT_BENCH_BACKEND_CUBIN:
            {
                SecantResult result;

                if (artifact->cubin == NULL ||
                    artifact->cubin_size != context->cubin_size) {
                    snprintf(error, error_size, "CUBIN artifact is not prepared");
                    return 0;
                }
                result = secant_cubin_specialize_into(
                    context->cubin_plan,
                    context->routines,
                    context->num_routines,
                    asts,
                    options->num_kernels * options->asts_per_kernel,
                    artifact->cubin,
                    artifact->cubin_size);
                if (result != SECANT_SUCCESS) {
                    snprintf(
                        error,
                        error_size,
                        "SecantResult(%d)",
                        (int)result);
                    return 0;
                }
                return 1;
            }
#endif
#if SECANT_BENCH_HAS_HSACO
        case SECANT_BENCH_BACKEND_HSACO:
            {
                SecantResult result;

                if (artifact->hsaco == NULL ||
                    artifact->hsaco_size != context->hsaco_size) {
                    snprintf(
                        error,
                        error_size,
                        "HSACO artifact is not prepared");
                    return 0;
                }
                result = secant_hsaco_specialize_into(
                    context->hsaco_plan,
                    context->routines,
                    context->num_routines,
                    asts,
                    options->num_kernels *
                        options->asts_per_kernel,
                    artifact->hsaco,
                    artifact->hsaco_size);
                if (result != SECANT_SUCCESS) {
                    snprintf(
                        error,
                        error_size,
                        "%s",
                        secant_result_to_string(result));
                    return 0;
                }
                return 1;
            }
#endif
        default:
            snprintf(error, error_size, "backend unavailable");
            return 0;
    }
}

static void
secant_compile_artifact_destroy(
    SecantBenchBackend backend,
    SecantCompileArtifact* artifact
) {
    switch (backend) {
#if SECANT_BENCH_HAS_CUDA
        case SECANT_BENCH_BACKEND_CUDA:
            if (artifact->cuda != NULL) {
                (void)secant_cuda_compiled_destroy(artifact->cuda);
                artifact->cuda = NULL;
            }
            break;
#endif
#if SECANT_BENCH_HAS_HSACO
        case SECANT_BENCH_BACKEND_HSACO:
            free(artifact->hsaco);
            artifact->hsaco = NULL;
            artifact->hsaco_size = 0u;
            break;
#endif
#if SECANT_BENCH_HAS_PTX
        case SECANT_BENCH_BACKEND_PTX:
            if (artifact->ptx != NULL) {
                (void)secant_ptx_compiled_destroy(artifact->ptx);
                artifact->ptx = NULL;
            }
            break;
#endif
#if SECANT_BENCH_HAS_HIP
        case SECANT_BENCH_BACKEND_HIP:
            if (artifact->hip != NULL) {
                (void)secant_hip_compiled_destroy(artifact->hip);
                artifact->hip = NULL;
            }
            break;
#endif
#if SECANT_BENCH_HAS_CUBIN
        case SECANT_BENCH_BACKEND_CUBIN:
            free(artifact->cubin);
            artifact->cubin = NULL;
            artifact->cubin_size = 0u;
            break;
#endif
        default:
            break;
    }
}

#if SECANT_BENCH_HAS_CUDA || SECANT_BENCH_HAS_PTX || SECANT_BENCH_HAS_CUBIN
static int
secant_compile_verify_cuda_module(
    SecantCompileContext* context,
    size_t module_idx,
    const float* host_input,
    const float* host_targets,
    float* expected,
    float* actual
) {
    const SecantBenchOptions* options = &context->options;
    const size_t rows = options->check_rows;
    const size_t output_count = options->shape == SECANT_BENCH_SHAPE_MATERIALIZE
        ? options->asts_per_kernel * rows
        : options->asts_per_kernel * options->num_targets;
    const size_t output_bytes = output_count * sizeof(float);
    CUdevice device;
    CUcontext cuda_context = NULL;
    CUdeviceptr device_input = 0u;
    CUdeviceptr device_targets = 0u;
    CUdeviceptr device_output = 0u;
    CUstream stream = NULL;
    const void* binary = NULL;
    size_t binary_size = 0u;
    void* module = NULL;
    void** functions = NULL;
    size_t kernel_idx;
    int passed = 0;
    int failed = 0;

    do {
        if (cuInit(0u) != CUDA_SUCCESS ||
            cuDeviceGet(&device, options->device_ordinal) != CUDA_SUCCESS ||
            cuDevicePrimaryCtxRetain(&cuda_context, device) != CUDA_SUCCESS ||
            cuCtxSetCurrent(cuda_context) != CUDA_SUCCESS ||
            cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS ||
            cuMemAlloc(&device_input, options->num_inputs * rows * sizeof(float)) != CUDA_SUCCESS ||
            cuMemcpyHtoD(device_input, host_input, options->num_inputs * rows * sizeof(float)) != CUDA_SUCCESS ||
            cuMemAlloc(&device_output, output_bytes) != CUDA_SUCCESS) {
            break;
        }
        functions = (void**)calloc(
            options->num_kernels,
            sizeof(*functions));
        if (functions == NULL) {
            break;
        }
        if (options->shape == SECANT_BENCH_SHAPE_SSE &&
            (cuMemAlloc(&device_targets, options->num_targets * rows * sizeof(float)) != CUDA_SUCCESS ||
             cuMemcpyHtoD(device_targets, host_targets, options->num_targets * rows * sizeof(float)) != CUDA_SUCCESS)) {
            break;
        }
#if SECANT_BENCH_HAS_CUDA
        if (options->backend == SECANT_BENCH_BACKEND_CUDA &&
            secant_cuda_compiled_binary_get(
                context->artifacts[module_idx].cuda,
                &binary,
                &binary_size) != SECANT_CUDA_SUCCESS) {
            break;
        }
#endif
#if SECANT_BENCH_HAS_PTX
        if (options->backend == SECANT_BENCH_BACKEND_PTX &&
            secant_ptx_compiled_binary_get(
                context->artifacts[module_idx].ptx,
                &binary,
                &binary_size) != SECANT_PTX_SUCCESS) {
            break;
        }
#endif
#if SECANT_BENCH_HAS_CUBIN
        if (options->backend == SECANT_BENCH_BACKEND_CUBIN) {
            binary = context->artifacts[module_idx].cubin;
            binary_size = context->artifacts[module_idx].cubin_size;
        }
#endif
        if (secant_test_cuda_module_load(
                binary,
                binary_size,
                options->num_kernels,
                &module,
                functions) != SECANT_CUDA_SUCCESS) {
            break;
        }

        for (kernel_idx = 0u; kernel_idx < options->num_kernels; ++kernel_idx) {
            const size_t ast_offset =
                ((options->warmups + module_idx) * options->num_kernels + kernel_idx) *
                options->asts_per_kernel;
            float max_abs_error = 0.0f;
            size_t worst_idx = 0u;

            memset(expected, 0, output_bytes);
            if (cuMemsetD8(device_output, 0u, output_bytes) != CUDA_SUCCESS) {
                failed = 1;
                break;
            }
            if (options->shape == SECANT_BENCH_SHAPE_MATERIALIZE) {
                if (secant_cpu_run_static_column_materialize(
                    options->num_inputs,
                    context->routines,
                    context->num_routines,
                    context->asts + ast_offset,
                    options->asts_per_kernel,
                    host_input,
                    options->num_inputs * rows,
                    rows,
                    rows,
                    expected,
                    output_count,
                    rows) != SECANT_SUCCESS) {
                    failed = 1;
                    break;
                }
                if (secant_test_cuda_run_static_column_materialize(
                        functions[kernel_idx],
                        options->num_inputs,
                        options->asts_per_kernel,
                        (const float*)(uintptr_t)device_input,
                        options->num_inputs * rows,
                        rows,
                        rows,
                        stream,
                        (float*)(uintptr_t)device_output,
                        output_count,
                        rows) != SECANT_CUDA_SUCCESS ||
                    cuStreamSynchronize(stream) != CUDA_SUCCESS) {
                    failed = 1;
                    break;
                }
            } else {
                if (secant_cpu_run_static_column_sse(
                    options->num_inputs,
                    options->num_targets,
                    context->routines,
                    context->num_routines,
                    context->asts + ast_offset,
                    options->asts_per_kernel,
                    host_input,
                    options->num_inputs * rows,
                    rows,
                    host_targets,
                    options->num_targets * rows,
                    rows,
                    rows,
                    expected,
                    output_count,
                    options->num_targets) != SECANT_SUCCESS) {
                    failed = 1;
                    break;
                }
                if (secant_test_cuda_run_static_column_sse(
                        functions[kernel_idx],
                        options->num_inputs,
                        options->asts_per_kernel,
                        options->num_targets,
                        options->tile_rows,
                        options->threads_per_block,
                        (const float*)(uintptr_t)device_input,
                        options->num_inputs * rows,
                        rows,
                        (const float*)(uintptr_t)device_targets,
                        options->num_targets * rows,
                        rows,
                        rows,
                        stream,
                        (float*)(uintptr_t)device_output,
                        output_count,
                        options->num_targets) != SECANT_CUDA_SUCCESS ||
                    cuStreamSynchronize(stream) != CUDA_SUCCESS) {
                    failed = 1;
                    break;
                }
            }
            if (cuMemcpyDtoH(actual, device_output, output_bytes) != CUDA_SUCCESS ||
                !secant_bench_compare(
                    options->ast_mode,
                    expected,
                    actual,
                    output_count,
                    &max_abs_error,
                    &worst_idx)) {
                fprintf(
                    stderr,
                    "verification failed: module=%zu kernel=%zu worst_idx=%zu "
                    "expected=%.9g actual=%.9g max_abs_error=%g\n",
                    module_idx,
                    kernel_idx,
                    worst_idx,
                    (double)expected[worst_idx],
                    (double)actual[worst_idx],
                    (double)max_abs_error);
                failed = 1;
                break;
            }
        }
        if (!failed) {
            passed = 1;
        }
    } while (0);

    if (module != NULL) {
        (void)cuModuleUnload((CUmodule)module);
    }
    free(functions);
    if (device_output != 0u) {
        (void)cuMemFree(device_output);
    }
    if (device_targets != 0u) {
        (void)cuMemFree(device_targets);
    }
    if (device_input != 0u) {
        (void)cuMemFree(device_input);
    }
    if (stream != NULL) {
        (void)cuStreamDestroy(stream);
    }
    if (cuda_context != NULL) {
        (void)cuDevicePrimaryCtxRelease(device);
    }
    return passed;
}
#endif

#if SECANT_BENCH_HAS_HIP
static int
secant_compile_verify_hip_module(
    SecantCompileContext* context,
    size_t module_idx,
    const float* host_input,
    const float* host_targets,
    float* expected,
    float* actual
) {
    const SecantBenchOptions* options = &context->options;
    const size_t rows = options->check_rows;
    const size_t output_count = options->shape == SECANT_BENCH_SHAPE_MATERIALIZE
        ? options->asts_per_kernel * rows
        : options->asts_per_kernel * options->num_targets;
    const size_t output_bytes = output_count * sizeof(float);
    float* device_input = NULL;
    float* device_targets = NULL;
    float* device_output = NULL;
    hipStream_t stream = NULL;
    const void* binary = NULL;
    size_t binary_size = 0u;
    void* module = NULL;
    void** functions = NULL;
    size_t kernel_idx;
    int passed = 0;
    int failed = 0;

    do {
        if (hipSetDevice(options->device_ordinal) != hipSuccess ||
            hipStreamCreateWithFlags(&stream, hipStreamNonBlocking) != hipSuccess ||
            hipMalloc((void**)&device_input, options->num_inputs * rows * sizeof(float)) != hipSuccess ||
            hipMemcpy(device_input, host_input, options->num_inputs * rows * sizeof(float), hipMemcpyHostToDevice) != hipSuccess ||
            hipMalloc((void**)&device_output, output_bytes) != hipSuccess) {
            break;
        }
        functions = (void**)calloc(
            options->num_kernels,
            sizeof(*functions));
        if (functions == NULL ||
            secant_hip_compiled_binary_get(
                context->artifacts[module_idx].hip,
                &binary,
                &binary_size) != SECANT_HIP_SUCCESS ||
            secant_test_hip_module_load(
                binary,
                binary_size,
                options->shape == SECANT_BENCH_SHAPE_MATERIALIZE
                    ? "secant_static_column_materialize"
                    : "secant_static_column_sse",
                options->num_kernels,
                &module,
                functions) != SECANT_HIP_SUCCESS) {
            break;
        }
        if (options->shape == SECANT_BENCH_SHAPE_SSE &&
            (hipMalloc((void**)&device_targets, options->num_targets * rows * sizeof(float)) != hipSuccess ||
             hipMemcpy(device_targets, host_targets, options->num_targets * rows * sizeof(float), hipMemcpyHostToDevice) != hipSuccess)) {
            break;
        }

        for (kernel_idx = 0u; kernel_idx < options->num_kernels; ++kernel_idx) {
            const size_t ast_offset =
                ((options->warmups + module_idx) * options->num_kernels + kernel_idx) *
                options->asts_per_kernel;
            float max_abs_error = 0.0f;
            size_t worst_idx = 0u;

            memset(expected, 0, output_bytes);
            if (hipMemset(device_output, 0, output_bytes) != hipSuccess) {
                failed = 1;
                break;
            }
            if (options->shape == SECANT_BENCH_SHAPE_MATERIALIZE) {
                if (secant_cpu_run_static_column_materialize(
                    options->num_inputs, context->routines, context->num_routines,
                    context->asts + ast_offset, options->asts_per_kernel,
                    host_input, options->num_inputs * rows, rows, rows,
                    expected, output_count, rows) != SECANT_SUCCESS ||
                    secant_test_hip_run_static_column_materialize(
                        functions[kernel_idx],
                        options->num_inputs,
                        options->asts_per_kernel,
                        device_input,
                        options->num_inputs * rows,
                        rows,
                        rows,
                        stream,
                        device_output,
                        output_count,
                        rows) != SECANT_HIP_SUCCESS) {
                    failed = 1;
                    break;
                }
            } else {
                if (secant_cpu_run_static_column_sse(
                    options->num_inputs, options->num_targets,
                    context->routines, context->num_routines,
                    context->asts + ast_offset, options->asts_per_kernel,
                    host_input, options->num_inputs * rows, rows,
                    host_targets, options->num_targets * rows, rows, rows,
                    expected, output_count, options->num_targets) != SECANT_SUCCESS ||
                secant_test_hip_run_static_column_sse(
                    functions[kernel_idx],
                    options->num_inputs,
                    options->asts_per_kernel,
                    options->num_targets,
                    options->tile_rows,
                    options->threads_per_block,
                    device_input,
                    options->num_inputs * rows,
                    rows,
                    device_targets,
                    options->num_targets * rows,
                    rows,
                    rows,
                    stream,
                    device_output,
                    output_count,
                    options->num_targets) != SECANT_HIP_SUCCESS) {
                    failed = 1;
                    break;
                }
            }
            if (hipStreamSynchronize(stream) != hipSuccess ||
                hipMemcpy(actual, device_output, output_bytes, hipMemcpyDeviceToHost) != hipSuccess ||
                !secant_bench_compare(
                    options->ast_mode,
                    expected,
                    actual,
                    output_count,
                    &max_abs_error,
                    &worst_idx)) {
                fprintf(
                    stderr,
                    "verification failed: module=%zu kernel=%zu worst_idx=%zu "
                    "expected=%.9g actual=%.9g max_abs_error=%g\n",
                    module_idx,
                    kernel_idx,
                    worst_idx,
                    (double)expected[worst_idx],
                    (double)actual[worst_idx],
                    (double)max_abs_error);
                failed = 1;
                break;
            }
        }
        if (!failed) {
            passed = 1;
        }
    } while (0);

    if (module != NULL) {
        (void)hipModuleUnload((hipModule_t)module);
    }
    free(functions);
    if (device_output != NULL) {
        (void)hipFree(device_output);
    }
    if (device_targets != NULL) {
        (void)hipFree(device_targets);
    }
    if (device_input != NULL) {
        (void)hipFree(device_input);
    }
    if (stream != NULL) {
        (void)hipStreamDestroy(stream);
    }
    return passed;
}
#endif

#if SECANT_BENCH_HAS_HSACO
static int
secant_compile_verify_hsaco_module(
    SecantCompileContext* context,
    size_t module_idx,
    const float* host_input,
    const float* host_targets,
    float* expected,
    float* actual
) {
    const SecantBenchOptions* options = &context->options;
    const size_t rows = options->check_rows;
    const size_t output_count =
        options->shape == SECANT_BENCH_SHAPE_MATERIALIZE
        ? options->asts_per_kernel * rows
        : options->asts_per_kernel * options->num_targets;
    const size_t output_bytes = output_count * sizeof(float);
    const SecantAstInstruction* const* module_asts =
        secant_compile_module_asts(
            context,
            options->warmups + module_idx);
    float* device_input = NULL;
    float* device_targets = NULL;
    float* device_output = NULL;
    hipStream_t stream = NULL;
    void* module = NULL;
    void** functions = NULL;
    size_t kernel_idx;
    int passed = 0;
    int failed = 0;

    do {
        if (hipSetDevice(options->device_ordinal) != hipSuccess ||
            hipStreamCreateWithFlags(
                &stream,
                hipStreamNonBlocking) != hipSuccess ||
            hipMalloc(
                (void**)&device_input,
                options->num_inputs * rows * sizeof(float)) !=
                hipSuccess ||
            hipMemcpy(
                device_input,
                host_input,
                options->num_inputs * rows * sizeof(float),
                hipMemcpyHostToDevice) != hipSuccess ||
            hipMalloc(
                (void**)&device_output,
                output_bytes) != hipSuccess) {
            break;
        }
        if (options->shape == SECANT_BENCH_SHAPE_SSE &&
            (hipMalloc(
                (void**)&device_targets,
                options->num_targets * rows * sizeof(float)) !=
                hipSuccess ||
             hipMemcpy(
                device_targets,
                host_targets,
                options->num_targets * rows * sizeof(float),
                hipMemcpyHostToDevice) != hipSuccess)) {
            break;
        }
        functions = (void**)calloc(
            options->num_kernels,
            sizeof(*functions));
        if (functions == NULL ||
            secant_test_hip_module_load(
                context->artifacts[module_idx].hsaco,
                context->artifacts[module_idx].hsaco_size,
                options->shape == SECANT_BENCH_SHAPE_MATERIALIZE
                    ? "secant_hsaco_materialize"
                    : "secant_hsaco_sse",
                options->num_kernels,
                &module,
                functions) != SECANT_HIP_SUCCESS) {
            break;
        }
        for (kernel_idx = 0u;
             kernel_idx < options->num_kernels;
             ++kernel_idx) {
            const size_t ast_offset =
                kernel_idx * options->asts_per_kernel;
            float max_abs_error = 0.0f;
            size_t worst_idx = 0u;
            SecantHIPResult run_result;

            memset(expected, 0, output_bytes);
            if (hipMemset(
                    device_output,
                    0,
                    output_bytes) != hipSuccess) {
                failed = 1;
                break;
            }
            if (options->shape ==
                SECANT_BENCH_SHAPE_MATERIALIZE) {
                if (secant_cpu_run_static_column_materialize(
                        options->num_inputs,
                        context->routines,
                        context->num_routines,
                        module_asts + ast_offset,
                        options->asts_per_kernel,
                        host_input,
                        options->num_inputs * rows,
                        rows,
                        rows,
                        expected,
                        output_count,
                        rows) != SECANT_SUCCESS) {
                    failed = 1;
                    break;
                }
                run_result = secant_test_hip_run_static_column_materialize(
                    functions[kernel_idx],
                    options->num_inputs,
                    options->asts_per_kernel,
                    device_input,
                    options->num_inputs * rows,
                    rows,
                    rows,
                    stream,
                    device_output,
                    output_count,
                    rows);
            } else {
                if (secant_cpu_run_static_column_sse(
                        options->num_inputs,
                        options->num_targets,
                        context->routines,
                        context->num_routines,
                        module_asts + ast_offset,
                        options->asts_per_kernel,
                        host_input,
                        options->num_inputs * rows,
                        rows,
                        host_targets,
                        options->num_targets * rows,
                        rows,
                        rows,
                        expected,
                        output_count,
                        options->num_targets) !=
                    SECANT_SUCCESS) {
                    failed = 1;
                    break;
                }
                run_result = secant_test_hip_run_static_column_sse(
                    functions[kernel_idx],
                    options->num_inputs,
                    options->asts_per_kernel,
                    options->num_targets,
                    options->tile_rows,
                    options->threads_per_block,
                    device_input,
                    options->num_inputs * rows,
                    rows,
                    device_targets,
                    options->num_targets * rows,
                    rows,
                    rows,
                    stream,
                    device_output,
                    output_count,
                    options->num_targets);
            }
            if (run_result != SECANT_HIP_SUCCESS ||
                hipStreamSynchronize(stream) != hipSuccess ||
                hipMemcpy(
                    actual,
                    device_output,
                    output_bytes,
                    hipMemcpyDeviceToHost) != hipSuccess ||
                !secant_bench_compare(
                    options->ast_mode,
                    expected,
                    actual,
                    output_count,
                    &max_abs_error,
                    &worst_idx)) {
                fprintf(
                    stderr,
                    "verification failed: module=%zu kernel=%zu "
                    "worst_idx=%zu expected=%.9g actual=%.9g "
                    "max_abs_error=%g\n",
                    module_idx,
                    kernel_idx,
                    worst_idx,
                    (double)expected[worst_idx],
                    (double)actual[worst_idx],
                    (double)max_abs_error);
                failed = 1;
                break;
            }
        }
        if (!failed) {
            passed = 1;
        }
    } while (0);

    if (module != NULL) {
        (void)hipModuleUnload((hipModule_t)module);
    }
    free(functions);
    if (device_output != NULL) {
        (void)hipFree(device_output);
    }
    if (device_targets != NULL) {
        (void)hipFree(device_targets);
    }
    if (device_input != NULL) {
        (void)hipFree(device_input);
    }
    if (stream != NULL) {
        (void)hipStreamDestroy(stream);
    }
    return passed;
}
#endif

static int
secant_compile_verify(SecantCompileContext* context) {
    const SecantBenchOptions* options = &context->options;
    const size_t rows = options->check_rows;
    const size_t output_count = options->shape == SECANT_BENCH_SHAPE_MATERIALIZE
        ? options->asts_per_kernel * rows
        : options->asts_per_kernel * options->num_targets;
    const size_t check_modules =
        options->check_modules < options->iterations
            ? options->check_modules
            : options->iterations;
    float* host_input;
    float* host_targets;
    float* expected;
    float* actual;
    size_t module_idx;
    int passed = 1;

    if (check_modules == 0u) {
        return 1;
    }
    host_input = (float*)malloc(options->num_inputs * rows * sizeof(float));
    host_targets = (float*)malloc(options->num_targets * rows * sizeof(float));
    expected = (float*)malloc(output_count * sizeof(float));
    actual = (float*)malloc(output_count * sizeof(float));
    if (host_input == NULL || host_targets == NULL || expected == NULL || actual == NULL) {
        free(actual);
        free(expected);
        free(host_targets);
        free(host_input);
        return 0;
    }
    secant_bench_fill_input(
        host_input,
        options->num_inputs,
        rows,
        options->seed);
    secant_bench_fill_targets(
        host_targets,
        options->num_targets,
        rows,
        options->seed);

    for (module_idx = 0u; module_idx < check_modules; ++module_idx) {
        switch (options->backend) {
#if SECANT_BENCH_HAS_CUDA || SECANT_BENCH_HAS_PTX || SECANT_BENCH_HAS_CUBIN
            case SECANT_BENCH_BACKEND_CUDA:
            case SECANT_BENCH_BACKEND_PTX:
            case SECANT_BENCH_BACKEND_CUBIN:
                passed = secant_compile_verify_cuda_module(
                    context, module_idx, host_input, host_targets, expected, actual);
                break;
#endif
#if SECANT_BENCH_HAS_HIP
            case SECANT_BENCH_BACKEND_HIP:
                passed = secant_compile_verify_hip_module(
                    context, module_idx, host_input, host_targets, expected, actual);
                break;
#endif
#if SECANT_BENCH_HAS_HSACO
            case SECANT_BENCH_BACKEND_HSACO:
                passed = secant_compile_verify_hsaco_module(
                    context,
                    module_idx,
                    host_input,
                    host_targets,
                    expected,
                    actual);
                break;
#endif
            default:
                passed = 0;
                break;
        }
        if (!passed) {
            break;
        }
    }

    free(actual);
    free(expected);
    free(host_targets);
    free(host_input);
    return passed;
}

static int
secant_compile_prepare(SecantCompileContext* context) {
    SecantBenchOptions* options = &context->options;
    const size_t num_modules = options->warmups + options->iterations;
    size_t scratch_workers = options->workers;
    size_t program_bytes;
    size_t pointer_bytes;

#if SECANT_BENCH_PIPELINE
    if (options->backend == SECANT_BENCH_BACKEND_CUDA ||
        options->backend == SECANT_BENCH_BACKEND_PTX ||
        options->backend == SECANT_BENCH_BACKEND_HIP) {
        scratch_workers = 1u;
    }
#endif
#if SECANT_BENCH_HAS_HIP || SECANT_BENCH_HAS_HSACO
    if ((options->backend == SECANT_BENCH_BACKEND_HIP ||
         options->backend == SECANT_BENCH_BACKEND_HSACO) &&
        options->hip_architecture == NULL) {
        hipDeviceProp_t properties;
        const char* feature_suffix;
        size_t architecture_size;

        if (hipSetDevice(options->device_ordinal) != hipSuccess ||
            hipGetDeviceProperties(&properties, options->device_ordinal) != hipSuccess) {
            return 0;
        }
        feature_suffix = strchr(properties.gcnArchName, ':');
        architecture_size = feature_suffix != NULL
            ? (size_t)(feature_suffix - properties.gcnArchName)
            : strlen(properties.gcnArchName);
        if (architecture_size == 0u ||
            architecture_size >= sizeof(context->hip_architecture)) {
            return 0;
        }
        memcpy(
            context->hip_architecture,
            properties.gcnArchName,
            architecture_size);
        context->hip_architecture[architecture_size] = '\0';
        options->hip_architecture = context->hip_architecture;
    }
#if SECANT_BENCH_HAS_HSACO
    if (options->backend == SECANT_BENCH_BACKEND_HSACO) {
        char* end = NULL;
        unsigned long gfx_arch;

        if (options->hip_architecture == NULL ||
            strncmp(options->hip_architecture, "gfx", 3u) != 0) {
            return 0;
        }
        gfx_arch = strtoul(
            options->hip_architecture + 3u,
            &end,
            10);
        if (end == options->hip_architecture + 3u ||
            *end != '\0' || gfx_arch > UINT32_MAX) {
            return 0;
        }
        context->gfx_arch = (uint32_t)gfx_arch;
    }
#endif
#endif
    if (!secant_bench_ast_storage_sizes(
            num_modules,
            options->num_kernels,
            options->asts_per_kernel,
            &program_bytes,
            &pointer_bytes) ||
        (options->backend != SECANT_BENCH_BACKEND_CUBIN &&
         options->backend != SECANT_BENCH_BACKEND_HSACO &&
         scratch_workers > SIZE_MAX / options->compile_scratch_size)) {
        return 0;
    }
    context->programs = (SecantAstInstruction*)malloc(program_bytes);
    context->asts = (const SecantAstInstruction**)malloc(pointer_bytes);
    if (options->backend != SECANT_BENCH_BACKEND_CUBIN &&
        options->backend != SECANT_BENCH_BACKEND_HSACO) {
        context->worker_scratch = (unsigned char*)malloc(
            scratch_workers * options->compile_scratch_size);
    }
    context->artifacts = (SecantCompileArtifact*)calloc(
        options->iterations, sizeof(*context->artifacts));
    if (context->programs == NULL || context->asts == NULL ||
        (options->backend != SECANT_BENCH_BACKEND_CUBIN &&
         options->backend != SECANT_BENCH_BACKEND_HSACO &&
         context->worker_scratch == NULL) ||
        context->artifacts == NULL) {
        return 0;
    }
    secant_bench_ast_fill(
        num_modules,
        options->num_kernels,
        options->asts_per_kernel,
        options->num_inputs,
        options->seed,
        options->ast_mode,
        context->programs,
        context->asts);
    secant_bench_ast_get_routines(
        options->ast_mode,
        &context->routines,
        &context->num_routines,
        &context->routine_names);

#if SECANT_BENCH_HAS_PTX
    if (options->backend == SECANT_BENCH_BACKEND_PTX) {
        static const char* const nvrtc_options[] = {
            "--no-cache",
            "--split-compile=1"
        };
        const uint32_t major = options->source_sm / 10u;
        const uint32_t minor = options->source_sm % 10u;
        SecantPTXResult result;

        if (options->shape == SECANT_BENCH_SHAPE_MATERIALIZE) {
            result = secant_ptx_materialize_create(
                options->num_kernels,
                options->asts_per_kernel,
                options->num_inputs,
                major,
                minor,
                nvrtc_options,
                sizeof(nvrtc_options) / sizeof(nvrtc_options[0]),
                false,
                NULL,
                0u,
                NULL,
                &context->ptx_handle);
        } else {
            result = secant_ptx_sse_create(
                options->num_kernels,
                options->asts_per_kernel,
                options->num_inputs,
                options->num_targets,
                options->tile_rows,
                options->threads_per_block,
                SECANT_SSE_REDUCTION_MODE_ATOMIC,
                major,
                minor,
                nvrtc_options,
                sizeof(nvrtc_options) / sizeof(nvrtc_options[0]),
                false,
                NULL,
                0u,
                NULL,
                &context->ptx_handle);
        }
        if (result != SECANT_PTX_SUCCESS) {
            fprintf(stderr, "PTX setup failed: %s\n", secant_ptx_result_to_string(result));
            return 0;
        }
    }
#endif
#if SECANT_BENCH_HAS_CUBIN
    if (options->backend == SECANT_BENCH_BACKEND_CUBIN) {
        char template_error[16384];
        SecantCubinMaterializeRecipe materialize_recipe;
        SecantCubinSSERecipe sse_recipe;
        size_t patch_capacity_instructions;
#if !SECANT_BENCH_PIPELINE
        size_t worker_idx;
#endif

        if (options->asts_per_kernel >
            SIZE_MAX / options->patch_instructions_per_ast) {
            return 0;
        }
        patch_capacity_instructions =
            options->asts_per_kernel *
            options->patch_instructions_per_ast;
        materialize_recipe.num_kernels = options->num_kernels;
        materialize_recipe.asts_per_kernel = options->asts_per_kernel;
        materialize_recipe.num_inputs = options->num_inputs;
        materialize_recipe.patch_capacity_instructions =
            patch_capacity_instructions;
        sse_recipe.num_kernels = options->num_kernels;
        sse_recipe.asts_per_kernel = options->asts_per_kernel;
        sse_recipe.num_inputs = options->num_inputs;
        sse_recipe.num_targets = options->num_targets;
        sse_recipe.tile_rows = options->tile_rows;
        sse_recipe.threads_per_block = options->threads_per_block;
        sse_recipe.patch_capacity_instructions =
            patch_capacity_instructions;
        sse_recipe.reduction_mode =
            SECANT_SSE_REDUCTION_MODE_ATOMIC;
#if SECANT_BENCH_HAS_EMBEDDED_CUBIN
        context->cubin_template =
            secant_bench_cubin_embedded_template(
                options->shape,
                options->num_kernels,
                options->asts_per_kernel,
                options->num_inputs,
                options->num_targets,
                options->tile_rows,
                options->threads_per_block,
                SECANT_SSE_REDUCTION_MODE_ATOMIC,
                patch_capacity_instructions,
                options->target_sm,
                &context->cubin_size);
#endif
        if (context->cubin_template == NULL) {
            unsigned char* generated_template = NULL;

            if (!secant_bench_cubin_template_compile(
                    options->shape,
                    options->num_kernels,
                    options->asts_per_kernel,
                    options->num_inputs,
                    options->num_targets,
                    options->tile_rows,
                    options->threads_per_block,
                    SECANT_SSE_REDUCTION_MODE_ATOMIC,
                    patch_capacity_instructions,
                    options->target_sm,
                    template_error,
                    sizeof(template_error),
                    &generated_template,
                    &context->cubin_size)) {
                fprintf(
                    stderr,
                    "CUBIN template compilation failed: %s\n",
                    template_error);
                return 0;
            }
            context->cubin_template = generated_template;
            context->cubin_template_owned = 1;
        }
        {
            size_t workspace_size = 0u;
            SecantResult result;

            if (options->shape == SECANT_BENCH_SHAPE_MATERIALIZE) {
                result = secant_cubin_materialize_inspect(
                    &materialize_recipe,
                    context->cubin_template,
                    context->cubin_size,
                    NULL,
                    0u,
                    &workspace_size,
                    &context->cubin_plan);
            } else {
                result = secant_cubin_sse_inspect(
                    &sse_recipe,
                    context->cubin_template,
                    context->cubin_size,
                    NULL,
                    0u,
                    &workspace_size,
                    &context->cubin_plan);
            }
            if (result == SECANT_SUCCESS) {
                context->cubin_plan_storage = malloc(workspace_size);
                if (context->cubin_plan_storage == NULL) {
                    return 0;
                }
                if (options->shape == SECANT_BENCH_SHAPE_MATERIALIZE) {
                    result = secant_cubin_materialize_inspect(
                        &materialize_recipe,
                        context->cubin_template,
                        context->cubin_size,
                        context->cubin_plan_storage,
                        workspace_size,
                        &workspace_size,
                        &context->cubin_plan);
                } else {
                    result = secant_cubin_sse_inspect(
                        &sse_recipe,
                        context->cubin_template,
                        context->cubin_size,
                        context->cubin_plan_storage,
                        workspace_size,
                        &workspace_size,
                        &context->cubin_plan);
                }
            }
            if (result != SECANT_SUCCESS) {
                fprintf(
                    stderr,
                    "CUBIN setup failed: SecantResult(%d)\n",
                    (int)result);
                return 0;
            }
        }
        if (!secant_compile_prepare_cubin_artifact(
                context,
                context->artifacts)) {
            fprintf(stderr, "CUBIN artifact preparation failed\n");
            return 0;
        }
#if !SECANT_BENCH_PIPELINE
        for (worker_idx = 1u; worker_idx < options->iterations; ++worker_idx) {
            if (!secant_compile_prepare_cubin_artifact(
                    context,
                    context->artifacts + worker_idx)) {
                fprintf(stderr, "CUBIN artifact preparation failed\n");
                return 0;
            }
        }
#endif
    }
#endif
#if SECANT_BENCH_HAS_HSACO
    if (options->backend == SECANT_BENCH_BACKEND_HSACO) {
        char template_error[16384];
        SecantHsacoMaterializeRecipe materialize_recipe;
        SecantHsacoSSERecipe sse_recipe;
        size_t patch_capacity_instructions;
#if !SECANT_BENCH_PIPELINE
        size_t worker_idx;
#endif

        if (options->asts_per_kernel >
            SIZE_MAX / options->patch_instructions_per_ast) {
            return 0;
        }
        patch_capacity_instructions =
            options->asts_per_kernel *
            options->patch_instructions_per_ast;
        materialize_recipe.num_kernels = options->num_kernels;
        materialize_recipe.asts_per_kernel = options->asts_per_kernel;
        materialize_recipe.num_inputs = options->num_inputs;
        materialize_recipe.patch_capacity_instructions =
            patch_capacity_instructions;
        sse_recipe.num_kernels = options->num_kernels;
        sse_recipe.asts_per_kernel = options->asts_per_kernel;
        sse_recipe.num_inputs = options->num_inputs;
        sse_recipe.num_targets = options->num_targets;
        sse_recipe.tile_rows = options->tile_rows;
        sse_recipe.threads_per_block = options->threads_per_block;
        sse_recipe.patch_capacity_instructions =
            patch_capacity_instructions;
        sse_recipe.reduction_mode =
            SECANT_SSE_REDUCTION_MODE_ATOMIC;
        if (!secant_bench_hsaco_template_compile(
                options->shape,
                options->num_kernels,
                options->asts_per_kernel,
                options->num_inputs,
                options->num_targets,
                options->tile_rows,
                options->threads_per_block,
                SECANT_SSE_REDUCTION_MODE_ATOMIC,
                patch_capacity_instructions,
                options->opt_level,
                context->gfx_arch,
                template_error,
                sizeof(template_error),
                &context->hsaco_template,
                &context->hsaco_size)) {
            fprintf(
                stderr,
                "HSACO template compilation failed: %s\n",
                template_error);
            return 0;
        }
        {
            SecantResult result;
            size_t workspace_size = 0u;

            if (options->shape == SECANT_BENCH_SHAPE_MATERIALIZE) {
                result = secant_hsaco_materialize_inspect(
                    &materialize_recipe,
                    context->hsaco_template,
                    context->hsaco_size,
                    NULL,
                    0u,
                    &workspace_size,
                    &context->hsaco_plan);
            } else {
                result = secant_hsaco_sse_inspect(
                    &sse_recipe,
                    context->hsaco_template,
                    context->hsaco_size,
                    NULL,
                    0u,
                    &workspace_size,
                    &context->hsaco_plan);
            }
            if (result == SECANT_SUCCESS) {
                context->hsaco_plan_storage = malloc(workspace_size);
                if (context->hsaco_plan_storage == NULL) {
                    return 0;
                }
                if (options->shape ==
                    SECANT_BENCH_SHAPE_MATERIALIZE) {
                    result = secant_hsaco_materialize_inspect(
                        &materialize_recipe,
                        context->hsaco_template,
                        context->hsaco_size,
                        context->hsaco_plan_storage,
                        workspace_size,
                        &workspace_size,
                        &context->hsaco_plan);
                } else {
                    result = secant_hsaco_sse_inspect(
                        &sse_recipe,
                        context->hsaco_template,
                        context->hsaco_size,
                        context->hsaco_plan_storage,
                        workspace_size,
                        &workspace_size,
                        &context->hsaco_plan);
                }
            }
            if (result != SECANT_SUCCESS) {
                fprintf(
                    stderr,
                    "HSACO setup failed: SecantResult(%d)\n",
                    (int)result);
                return 0;
            }
        }
        if (!secant_compile_prepare_hsaco_artifact(
                context,
                context->artifacts)) {
            fprintf(
                stderr,
                "HSACO artifact preparation failed\n");
            return 0;
        }
#if !SECANT_BENCH_PIPELINE
        for (worker_idx = 1u; worker_idx < options->iterations; ++worker_idx) {
            if (!secant_compile_prepare_hsaco_artifact(
                    context,
                    context->artifacts + worker_idx)) {
                fprintf(
                    stderr,
                    "HSACO artifact preparation failed\n");
                return 0;
            }
        }
#endif
    }
#endif
    return 1;
}

static void
secant_compile_cleanup(SecantCompileContext* context) {
    size_t idx;

    if (context->artifacts != NULL) {
        for (idx = 0u; idx < context->options.iterations; ++idx) {
            secant_compile_artifact_destroy(context->options.backend, context->artifacts + idx);
        }
    }
#if SECANT_BENCH_HAS_PTX
    if (context->ptx_handle != NULL) {
        (void)secant_ptx_handle_destroy(context->ptx_handle);
    }
#endif
#if SECANT_BENCH_HAS_HSACO
    free(context->hsaco_plan_storage);
    free(context->hsaco_template);
#endif
#if SECANT_BENCH_HAS_CUBIN
    free(context->cubin_plan_storage);
    if (context->cubin_template_owned) {
        free((void*)context->cubin_template);
    }
#endif
    free(context->artifacts);
    free(context->worker_scratch);
    free(context->asts);
    free(context->programs);
}

#if !SECANT_BENCH_PIPELINE
static int
secant_compile_run(SecantCompileContext* context) {
    const SecantBenchOptions* options = &context->options;
    const size_t asts_per_module = options->num_kernels * options->asts_per_kernel;
    size_t warmup_idx;
    int failed = 0;
    char error[256] = { 0 };
    double begin;
    double seconds;

    for (warmup_idx = 0u; warmup_idx < options->warmups; ++warmup_idx) {
        SecantCompileArtifact artifact;

        memset(&artifact, 0, sizeof(artifact));
#if SECANT_BENCH_HAS_CUBIN
        if (options->backend == SECANT_BENCH_BACKEND_CUBIN &&
            !secant_compile_prepare_cubin_artifact(context, &artifact)) {
            fprintf(stderr, "CUBIN warmup artifact preparation failed\n");
            return 0;
        }
#endif
#if SECANT_BENCH_HAS_HSACO
        if (options->backend == SECANT_BENCH_BACKEND_HSACO &&
            !secant_compile_prepare_hsaco_artifact(
                context,
                &artifact)) {
            fprintf(
                stderr,
                "HSACO warmup artifact preparation failed\n");
            return 0;
        }
#endif
        if (!secant_compile_one(context, warmup_idx, &artifact, error, sizeof(error))) {
            fprintf(stderr, "warmup failed: %s\n", error);
            return 0;
        }
        secant_compile_artifact_destroy(options->backend, &artifact);
    }

    begin = secant_compile_seconds();
#if SECANT_BENCH_HAS_OPENMP
#pragma omp parallel for num_threads((int)options->workers) schedule(static)
#endif
    for (size_t idx = 0u; idx < options->iterations; ++idx) {
        char local_error[256] = { 0 };

        if (!secant_compile_one(
                context,
                options->warmups + idx,
                context->artifacts + idx,
                local_error,
                sizeof(local_error))) {
#if SECANT_BENCH_HAS_OPENMP
#pragma omp critical
#endif
            {
                if (error[0] == '\0') {
                    snprintf(error, sizeof(error), "%s", local_error);
                }
                failed = 1;
            }
        }
    }
    seconds = secant_compile_seconds() - begin;
    if (failed) {
        fprintf(stderr, "timed compile failed: %s\n", error);
        return 0;
    }

    printf(
        "compile backend=%s shape=%s ast_mode=%s seed=%u opt_level=%u workers=%zu modules=%zu "
        "kernels_per_module=%zu asts_per_kernel=%zu asts=%zu seconds=%.6f "
        "asts_per_second=%.3f us_per_ast=%.6f corpus=%s corpus_hash=%s "
        "expanded_corpus_hash=%016llx\n",
        secant_bench_backend_name(options->backend),
        secant_bench_shape_name(options->shape),
        secant_bench_ast_mode_name(options->ast_mode),
        options->seed,
        options->opt_level,
        options->workers,
        options->iterations,
        options->num_kernels,
        options->asts_per_kernel,
        options->iterations * asts_per_module,
        seconds,
        (double)(options->iterations * asts_per_module) / seconds,
        seconds * 1.0e6 / (double)(options->iterations * asts_per_module),
        secant_bench_ast_corpus_name(options->ast_mode),
        secant_bench_ast_corpus_definition_hash(options->ast_mode),
        (unsigned long long)secant_bench_ast_corpus_hash(
            context->programs,
            (options->warmups + options->iterations) * asts_per_module));

    if (!secant_compile_verify(context)) {
        fprintf(stderr, "correctness verification failed\n");
        return 0;
    }
    printf(
        "verify backend=%s modules=%zu rows=%zu status=pass\n",
        secant_bench_backend_name(options->backend),
        options->check_modules < options->iterations
            ? options->check_modules
            : options->iterations,
        options->check_rows);
    return 1;
}
#endif

#if SECANT_BENCH_PIPELINE
#if SECANT_BENCH_HAS_CUDA || SECANT_BENCH_HAS_PTX || SECANT_BENCH_HAS_CUBIN
typedef struct SecantPipelineNvidiaData {
    float* host_input;
    float* host_targets;
    CUdevice device;
    CUcontext context;
    CUcontext previous_context;
    CUdeviceptr input;
    CUdeviceptr targets;
    CUdeviceptr output;
    size_t input_count;
    size_t target_count;
    size_t output_count;
    int context_retained;
} SecantPipelineNvidiaData;

static void
secant_pipeline_nvidia_data_cleanup(
    SecantPipelineNvidiaData* data
) {
    if (data->output != 0u) {
        (void)cuMemFree(data->output);
    }
    if (data->targets != 0u) {
        (void)cuMemFree(data->targets);
    }
    if (data->input != 0u) {
        (void)cuMemFree(data->input);
    }
    if (data->context_retained) {
        (void)cuCtxSetCurrent(data->previous_context);
        (void)cuDevicePrimaryCtxRelease(data->device);
    }
    free(data->host_targets);
    free(data->host_input);
    memset(data, 0, sizeof(*data));
}

static int
secant_pipeline_nvidia_data_setup(
    const SecantCompileContext* context,
    size_t total_asts,
    SecantPipelineNvidiaData* data
) {
    const SecantBenchOptions* options = &context->options;
    size_t input_bytes;
    size_t module_asts;
    size_t target_bytes = 0u;
    size_t output_bytes;

    memset(data, 0, sizeof(*data));
    if (options->num_inputs > SIZE_MAX / options->run_rows ||
        total_asts == 0u ||
        options->num_kernels >
            SIZE_MAX / options->asts_per_kernel) {
        return 0;
    }
    module_asts =
        options->num_kernels * options->asts_per_kernel;
    data->input_count = options->num_inputs * options->run_rows;
    if (options->shape == SECANT_BENCH_SHAPE_MATERIALIZE) {
        if (module_asts > SIZE_MAX / options->run_rows) {
            return 0;
        }
        data->output_count = module_asts * options->run_rows;
    } else {
        if (options->num_targets > SIZE_MAX / options->run_rows ||
            total_asts > SIZE_MAX / options->num_targets) {
            return 0;
        }
        data->target_count =
            options->num_targets * options->run_rows;
        data->output_count =
            total_asts * options->num_targets;
    }
    if (data->input_count > SIZE_MAX / sizeof(float) ||
        data->target_count > SIZE_MAX / sizeof(float) ||
        data->output_count > SIZE_MAX / sizeof(float)) {
        return 0;
    }
    input_bytes = data->input_count * sizeof(float);
    target_bytes = data->target_count * sizeof(float);
    output_bytes = data->output_count * sizeof(float);
    data->host_input = (float*)malloc(input_bytes);
    if (options->shape == SECANT_BENCH_SHAPE_SSE) {
        data->host_targets = (float*)malloc(target_bytes);
    }
    if (data->host_input == NULL ||
        (options->shape == SECANT_BENCH_SHAPE_SSE &&
         data->host_targets == NULL)) {
        return 0;
    }
    secant_bench_fill_input(
        data->host_input,
        options->num_inputs,
        options->run_rows,
        options->seed);
    if (options->shape == SECANT_BENCH_SHAPE_SSE) {
        secant_bench_fill_targets(
            data->host_targets,
            options->num_targets,
            options->run_rows,
            options->seed);
    }
    if (cuInit(0u) != CUDA_SUCCESS ||
        cuDeviceGet(
            &data->device,
            options->device_ordinal) != CUDA_SUCCESS ||
        cuDevicePrimaryCtxRetain(
            &data->context,
            data->device) != CUDA_SUCCESS) {
        return 0;
    }
    data->context_retained = 1;
    if (cuCtxGetCurrent(&data->previous_context) != CUDA_SUCCESS ||
        cuCtxSetCurrent(data->context) != CUDA_SUCCESS ||
        cuMemAlloc(&data->input, input_bytes) != CUDA_SUCCESS ||
        cuMemcpyHtoD(
            data->input,
            data->host_input,
            input_bytes) != CUDA_SUCCESS ||
        cuMemAlloc(&data->output, output_bytes) != CUDA_SUCCESS) {
        return 0;
    }
    if (options->shape == SECANT_BENCH_SHAPE_SSE &&
        (cuMemAlloc(&data->targets, target_bytes) != CUDA_SUCCESS ||
         cuMemcpyHtoD(
            data->targets,
            data->host_targets,
            target_bytes) != CUDA_SUCCESS)) {
        return 0;
    }
    return 1;
}

static SecantResult
secant_pipeline_nvidia_run_all(
    SecantCompileContext* context,
    const SecantPipelineNvidiaData* data,
    size_t total_asts,
    SecantRunnerStats* stats
) {
    const SecantBenchOptions* options = &context->options;
    const uint32_t major = options->target_sm / 10u;
    const uint32_t minor = options->target_sm % 10u;

    switch (options->backend) {
#if SECANT_BENCH_HAS_CUDA
        case SECANT_BENCH_BACKEND_CUDA: {
            const char* compile_options[] = {
                "--no-cache",
                "--split-compile=1",
                NULL
            };
            char opt_level[32];
            SecantCUDARunner runner = NULL;
            SecantResult result;

            snprintf(
                opt_level,
                sizeof(opt_level),
                "--ptxas-options=-O%u",
                options->opt_level);
            compile_options[2] = opt_level;
            if (options->shape ==
                SECANT_BENCH_SHAPE_MATERIALIZE) {
                result = secant_cuda_materialize_runner_create(
                    options->num_kernels,
                    options->asts_per_kernel,
                    options->num_inputs,
                    major,
                    minor,
                    compile_options,
                    sizeof(compile_options) /
                        sizeof(compile_options[0]),
                    options->compile_scratch_size,
                    options->workers,
                    options->num_streams,
                    &runner);
            } else {
                result = secant_cuda_sse_runner_create(
                    options->num_kernels,
                    options->asts_per_kernel,
                    options->num_inputs,
                    options->num_targets,
                    options->tile_rows,
                    options->threads_per_block,
                    major,
                    minor,
                    compile_options,
                    sizeof(compile_options) /
                        sizeof(compile_options[0]),
                    options->compile_scratch_size,
                    options->workers,
                    options->num_streams,
                    &runner);
            }
            if (result == SECANT_SUCCESS &&
                options->shape ==
                    SECANT_BENCH_SHAPE_MATERIALIZE) {
                result = secant_cuda_materialize_runner_run_all(
                    runner,
                    context->routines,
                    context->num_routines,
                    context->routine_names,
                    context->asts,
                    total_asts,
                    (uintptr_t)data->input,
                    data->input_count,
                    options->run_rows,
                    options->run_rows,
                    (uintptr_t)data->output,
                    data->output_count,
                    options->run_rows,
                    0u,
                    stats);
            } else if (result == SECANT_SUCCESS) {
                result = secant_cuda_sse_runner_run_all(
                    runner,
                    context->routines,
                    context->num_routines,
                    context->routine_names,
                    context->asts,
                    total_asts,
                    (uintptr_t)data->input,
                    data->input_count,
                    options->run_rows,
                    (uintptr_t)data->targets,
                    data->target_count,
                    options->run_rows,
                    options->run_rows,
                    (uintptr_t)data->output,
                    data->output_count,
                    options->num_targets,
                    stats);
            }
            if (runner != NULL) {
                const SecantResult destroy_result =
                    secant_cuda_runner_destroy(runner);

                if (result == SECANT_SUCCESS) {
                    result = destroy_result;
                }
            }
            return result;
        }
#endif
#if SECANT_BENCH_HAS_PTX
        case SECANT_BENCH_BACKEND_PTX: {
            const char* compile_options[] = {
                NULL,
                "--split-compile=1"
            };
            char opt_level[32];
            SecantPTXRunner runner = NULL;
            SecantResult result;

            snprintf(
                opt_level,
                sizeof(opt_level),
                "--opt-level=%u",
                options->opt_level);
            compile_options[0] = opt_level;
            if (options->shape ==
                SECANT_BENCH_SHAPE_MATERIALIZE) {
                result = secant_ptx_materialize_runner_create(
                    context->ptx_handle,
                    major,
                    minor,
                    compile_options,
                    sizeof(compile_options) /
                        sizeof(compile_options[0]),
                    options->compile_scratch_size,
                    options->workers,
                    options->num_streams,
                    &runner);
            } else {
                result = secant_ptx_sse_runner_create(
                    context->ptx_handle,
                    major,
                    minor,
                    compile_options,
                    sizeof(compile_options) /
                        sizeof(compile_options[0]),
                    options->compile_scratch_size,
                    options->workers,
                    options->num_streams,
                    &runner);
            }
            if (result == SECANT_SUCCESS &&
                options->shape ==
                    SECANT_BENCH_SHAPE_MATERIALIZE) {
                result = secant_ptx_materialize_runner_run_all(
                    runner,
                    context->routines,
                    context->num_routines,
                    context->asts,
                    total_asts,
                    (uintptr_t)data->input,
                    data->input_count,
                    options->run_rows,
                    options->run_rows,
                    (uintptr_t)data->output,
                    data->output_count,
                    options->run_rows,
                    0u,
                    stats);
            } else if (result == SECANT_SUCCESS) {
                result = secant_ptx_sse_runner_run_all(
                    runner,
                    context->routines,
                    context->num_routines,
                    context->asts,
                    total_asts,
                    (uintptr_t)data->input,
                    data->input_count,
                    options->run_rows,
                    (uintptr_t)data->targets,
                    data->target_count,
                    options->run_rows,
                    options->run_rows,
                    (uintptr_t)data->output,
                    data->output_count,
                    options->num_targets,
                    stats);
            }
            if (runner != NULL) {
                const SecantResult destroy_result =
                    secant_ptx_runner_destroy(runner);

                if (result == SECANT_SUCCESS) {
                    result = destroy_result;
                }
            }
            return result;
        }
#endif
#if SECANT_BENCH_HAS_CUBIN
        case SECANT_BENCH_BACKEND_CUBIN: {
            SecantCubinRunner runner = NULL;
            SecantResult result;

            if (options->shape ==
                SECANT_BENCH_SHAPE_MATERIALIZE) {
                result = secant_cubin_materialize_runner_create(
                    context->cubin_plan,
                    context->cubin_template,
                    context->cubin_size,
                    options->workers,
                    options->num_streams,
                    &runner);
            } else {
                result = secant_cubin_sse_runner_create(
                    context->cubin_plan,
                    context->cubin_template,
                    context->cubin_size,
                    options->workers,
                    options->num_streams,
                    &runner);
            }
            if (result == SECANT_SUCCESS &&
                options->shape ==
                    SECANT_BENCH_SHAPE_MATERIALIZE) {
                result = secant_cubin_materialize_runner_run_all(
                    runner,
                    context->routines,
                    context->num_routines,
                    context->asts,
                    total_asts,
                    (uintptr_t)data->input,
                    data->input_count,
                    options->run_rows,
                    options->run_rows,
                    (uintptr_t)data->output,
                    data->output_count,
                    options->run_rows,
                    0u,
                    stats);
            } else if (result == SECANT_SUCCESS) {
                result = secant_cubin_sse_runner_run_all(
                    runner,
                    context->routines,
                    context->num_routines,
                    context->asts,
                    total_asts,
                    (uintptr_t)data->input,
                    data->input_count,
                    options->run_rows,
                    (uintptr_t)data->targets,
                    data->target_count,
                    options->run_rows,
                    options->run_rows,
                    (uintptr_t)data->output,
                    data->output_count,
                    options->num_targets,
                    stats);
            }
            if (runner != NULL) {
                const SecantResult destroy_result =
                    secant_cubin_runner_destroy(runner);

                if (result == SECANT_SUCCESS) {
                    result = destroy_result;
                }
            }
            return result;
        }
#endif
        default:
            return SECANT_ERROR_BACKEND_UNAVAILABLE;
    }
}

static int
secant_pipeline_run_nvidia(
    SecantCompileContext* context,
    size_t total_asts,
    SecantRunnerStats* stats
) {
    SecantPipelineNvidiaData data;
    SecantResult result;

    memset(&data, 0, sizeof(data));
    if (context->options.run_iterations != 1u) {
        fprintf(
            stderr,
            "NVIDIA bulk runners require --run-iterations 1\n");
        return 0;
    }
    if (!secant_pipeline_nvidia_data_setup(
            context,
            total_asts,
            &data)) {
        secant_pipeline_nvidia_data_cleanup(&data);
        return 0;
    }
    result = secant_pipeline_nvidia_run_all(
        context,
        &data,
        total_asts,
        stats);
    secant_pipeline_nvidia_data_cleanup(&data);
    if (result != SECANT_SUCCESS) {
        fprintf(
            stderr,
            "NVIDIA runner failed: %s\n",
            secant_result_to_string(result));
        return 0;
    }
    return 1;
}
#endif

#if SECANT_BENCH_HAS_HIP || SECANT_BENCH_HAS_HSACO
typedef struct SecantPipelineAmdData {
    float* host_input;
    float* host_targets;
    float* input;
    float* targets;
    float* output;
    size_t input_count;
    size_t target_count;
    size_t output_count;
} SecantPipelineAmdData;

static void
secant_pipeline_amd_data_cleanup(
    SecantPipelineAmdData* data
) {
    if (data->output != NULL) {
        (void)hipFree(data->output);
    }
    if (data->targets != NULL) {
        (void)hipFree(data->targets);
    }
    if (data->input != NULL) {
        (void)hipFree(data->input);
    }
    free(data->host_targets);
    free(data->host_input);
    memset(data, 0, sizeof(*data));
}

static int
secant_pipeline_amd_data_setup(
    const SecantCompileContext* context,
    size_t total_asts,
    SecantPipelineAmdData* data
) {
    const SecantBenchOptions* options = &context->options;
    size_t module_asts;
    size_t input_bytes;
    size_t target_bytes;
    size_t output_bytes;

    memset(data, 0, sizeof(*data));
    if (options->num_inputs > SIZE_MAX / options->run_rows ||
        options->num_kernels > SIZE_MAX / options->asts_per_kernel) {
        return 0;
    }
    module_asts =
        options->num_kernels * options->asts_per_kernel;
    data->input_count = options->num_inputs * options->run_rows;
    if (options->shape == SECANT_BENCH_SHAPE_MATERIALIZE) {
        if (module_asts > SIZE_MAX / options->run_rows) {
            return 0;
        }
        data->output_count = module_asts * options->run_rows;
    } else {
        if (options->num_targets > SIZE_MAX / options->run_rows ||
            total_asts > SIZE_MAX / options->num_targets) {
            return 0;
        }
        data->target_count =
            options->num_targets * options->run_rows;
        data->output_count =
            total_asts * options->num_targets;
    }
    if (data->input_count > SIZE_MAX / sizeof(float) ||
        data->target_count > SIZE_MAX / sizeof(float) ||
        data->output_count > SIZE_MAX / sizeof(float)) {
        return 0;
    }
    input_bytes = data->input_count * sizeof(float);
    target_bytes = data->target_count * sizeof(float);
    output_bytes = data->output_count * sizeof(float);
    data->host_input = (float*)malloc(input_bytes);
    if (options->shape == SECANT_BENCH_SHAPE_SSE) {
        data->host_targets = (float*)malloc(target_bytes);
    }
    if (data->host_input == NULL ||
        (options->shape == SECANT_BENCH_SHAPE_SSE &&
         data->host_targets == NULL)) {
        return 0;
    }
    secant_bench_fill_input(
        data->host_input,
        options->num_inputs,
        options->run_rows,
        options->seed);
    if (options->shape == SECANT_BENCH_SHAPE_SSE) {
        secant_bench_fill_targets(
            data->host_targets,
            options->num_targets,
            options->run_rows,
            options->seed);
    }
    if (hipSetDevice(options->device_ordinal) != hipSuccess ||
        hipMalloc((void**)&data->input, input_bytes) != hipSuccess ||
        hipMemcpy(
            data->input,
            data->host_input,
            input_bytes,
            hipMemcpyHostToDevice) != hipSuccess ||
        hipMalloc((void**)&data->output, output_bytes) != hipSuccess) {
        return 0;
    }
    if (options->shape == SECANT_BENCH_SHAPE_SSE &&
        (hipMalloc(
            (void**)&data->targets,
            target_bytes) != hipSuccess ||
         hipMemcpy(
            data->targets,
            data->host_targets,
            target_bytes,
            hipMemcpyHostToDevice) != hipSuccess)) {
        return 0;
    }
    return 1;
}

static SecantResult
secant_pipeline_amd_run_all(
    SecantCompileContext* context,
    const SecantPipelineAmdData* data,
    size_t total_asts,
    SecantRunnerStats* stats
) {
    const SecantBenchOptions* options = &context->options;

    switch (options->backend) {
#if SECANT_BENCH_HAS_HIP
        case SECANT_BENCH_BACKEND_HIP: {
            const char* compile_options[] = { NULL };
            char opt_level[16];
            SecantHIPRunner runner = NULL;
            SecantResult result;

            snprintf(
                opt_level,
                sizeof(opt_level),
                "-O%u",
                options->opt_level);
            compile_options[0] = opt_level;
            if (options->shape ==
                SECANT_BENCH_SHAPE_MATERIALIZE) {
                result = secant_hip_materialize_runner_create(
                    options->num_kernels,
                    options->asts_per_kernel,
                    options->num_inputs,
                    options->hip_architecture,
                    compile_options,
                    sizeof(compile_options) /
                        sizeof(compile_options[0]),
                    options->compile_scratch_size,
                    options->workers,
                    options->num_streams,
                    &runner);
            } else {
                result = secant_hip_sse_runner_create(
                    options->num_kernels,
                    options->asts_per_kernel,
                    options->num_inputs,
                    options->num_targets,
                    options->tile_rows,
                    options->threads_per_block,
                    options->hip_architecture,
                    compile_options,
                    sizeof(compile_options) /
                        sizeof(compile_options[0]),
                    options->compile_scratch_size,
                    options->workers,
                    options->num_streams,
                    &runner);
            }
            if (result == SECANT_SUCCESS &&
                options->shape ==
                    SECANT_BENCH_SHAPE_MATERIALIZE) {
                result = secant_hip_materialize_runner_run_all(
                    runner,
                    context->routines,
                    context->num_routines,
                    context->routine_names,
                    context->asts,
                    total_asts,
                    (uintptr_t)data->input,
                    data->input_count,
                    options->run_rows,
                    options->run_rows,
                    (uintptr_t)data->output,
                    data->output_count,
                    options->run_rows,
                    0u,
                    stats);
            } else if (result == SECANT_SUCCESS) {
                result = secant_hip_sse_runner_run_all(
                    runner,
                    context->routines,
                    context->num_routines,
                    context->routine_names,
                    context->asts,
                    total_asts,
                    (uintptr_t)data->input,
                    data->input_count,
                    options->run_rows,
                    (uintptr_t)data->targets,
                    data->target_count,
                    options->run_rows,
                    options->run_rows,
                    (uintptr_t)data->output,
                    data->output_count,
                    options->num_targets,
                    stats);
            }
            if (runner != NULL) {
                const SecantResult destroy_result =
                    secant_hip_runner_destroy(runner);

                if (result == SECANT_SUCCESS) {
                    result = destroy_result;
                }
            }
            return result;
        }
#endif
#if SECANT_BENCH_HAS_HSACO
        case SECANT_BENCH_BACKEND_HSACO: {
            SecantHsacoRunner runner = NULL;
            SecantResult result;

            if (options->shape ==
                SECANT_BENCH_SHAPE_MATERIALIZE) {
                result = secant_hsaco_materialize_runner_create(
                    context->hsaco_plan,
                    context->hsaco_template,
                    context->hsaco_size,
                    options->workers,
                    options->num_streams,
                    &runner);
            } else {
                result = secant_hsaco_sse_runner_create(
                    context->hsaco_plan,
                    context->hsaco_template,
                    context->hsaco_size,
                    options->workers,
                    options->num_streams,
                    &runner);
            }
            if (result == SECANT_SUCCESS &&
                options->shape ==
                    SECANT_BENCH_SHAPE_MATERIALIZE) {
                result = secant_hsaco_materialize_runner_run_all(
                    runner,
                    context->routines,
                    context->num_routines,
                    context->asts,
                    total_asts,
                    (uintptr_t)data->input,
                    data->input_count,
                    options->run_rows,
                    options->run_rows,
                    (uintptr_t)data->output,
                    data->output_count,
                    options->run_rows,
                    0u,
                    stats);
            } else if (result == SECANT_SUCCESS) {
                result = secant_hsaco_sse_runner_run_all(
                    runner,
                    context->routines,
                    context->num_routines,
                    context->asts,
                    total_asts,
                    (uintptr_t)data->input,
                    data->input_count,
                    options->run_rows,
                    (uintptr_t)data->targets,
                    data->target_count,
                    options->run_rows,
                    options->run_rows,
                    (uintptr_t)data->output,
                    data->output_count,
                    options->num_targets,
                    stats);
            }
            if (runner != NULL) {
                const SecantResult destroy_result =
                    secant_hsaco_runner_destroy(runner);

                if (result == SECANT_SUCCESS) {
                    result = destroy_result;
                }
            }
            return result;
        }
#endif
        default:
            return SECANT_ERROR_BACKEND_UNAVAILABLE;
    }
}

static int
secant_pipeline_run_amd(
    SecantCompileContext* context,
    size_t total_asts,
    SecantRunnerStats* stats
) {
    SecantPipelineAmdData data;
    SecantResult result;

    memset(&data, 0, sizeof(data));
    if (context->options.run_iterations != 1u) {
        fprintf(
            stderr,
            "AMD bulk runners require --run-iterations 1\n");
        return 0;
    }
    if (!secant_pipeline_amd_data_setup(
            context,
            total_asts,
            &data)) {
        secant_pipeline_amd_data_cleanup(&data);
        return 0;
    }
    result = secant_pipeline_amd_run_all(
        context,
        &data,
        total_asts,
        stats);
    secant_pipeline_amd_data_cleanup(&data);
    if (result != SECANT_SUCCESS) {
        fprintf(
            stderr,
            "AMD runner failed: %s\n",
            secant_result_to_string(result));
        return 0;
    }
    return 1;
}
#endif

typedef struct SecantPipelineMetrics {
    double compile_window_seconds;
    double compile_critical_seconds;
    double compile_work_seconds;
    double module_load_seconds;
    double completion_wait_seconds;
    double module_unload_seconds;
    double runtime_seconds;
    double pipeline_seconds;
    size_t modules_loaded;
} SecantPipelineMetrics;

static void
secant_pipeline_usage(const char* program) {
    fprintf(
        stderr,
        "usage: %s [options]\n"
        "  --backend cuda|ptx|cubin|hip|hsaco\n"
        "  --shape materialize|sse\n"
        "  --ast-mode alu|mufu\n"
        "  --modules N --workers N\n"
        "  --kernels N --asts-per-kernel N --inputs N --targets N\n"
        "  --run-rows N --run-iterations N --streams N\n"
        "  --tile-rows N --threads N --patch-instructions-per-ast N\n"
        "  --compile-scratch-bytes N --check-rows N\n"
        "  --source-sm NN --target-sm NN|0(native) --opt-level 0|1 --device N\n",
        program);
}

static int
secant_pipeline_target_sm_resolve(SecantBenchOptions* options) {
#if SECANT_BENCH_HAS_CUDA || SECANT_BENCH_HAS_PTX || SECANT_BENCH_HAS_CUBIN
    CUdevice device;
    int major;
    int minor;
#endif

    if (options->target_sm != 0u) {
        return 1;
    }
    if (options->backend == SECANT_BENCH_BACKEND_HIP ||
        options->backend == SECANT_BENCH_BACKEND_HSACO) {
        options->target_sm = 10u;
        return 1;
    }
#if SECANT_BENCH_HAS_CUDA || SECANT_BENCH_HAS_PTX || SECANT_BENCH_HAS_CUBIN
    if (cuInit(0u) != CUDA_SUCCESS ||
        cuDeviceGet(&device, options->device_ordinal) != CUDA_SUCCESS ||
        cuDeviceGetAttribute(
            &major,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            device) != CUDA_SUCCESS ||
        cuDeviceGetAttribute(
            &minor,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            device) != CUDA_SUCCESS ||
        major < 0 || minor < 0 || major > 999 ||
        minor > 9) {
        return 0;
    }
    options->target_sm = (uint32_t)(major * 10 + minor);
    return 1;
#else
    return 0;
#endif
}

static int
secant_pipeline_backend_uses_eager_cuda(
    SecantBenchBackend backend
) {
    return backend == SECANT_BENCH_BACKEND_CUDA ||
        backend == SECANT_BENCH_BACKEND_PTX ||
        backend == SECANT_BENCH_BACKEND_CUBIN;
}

static int
secant_pipeline_run(SecantCompileContext* context) {
    const SecantBenchOptions* options = &context->options;
    size_t asts_per_module;
    size_t total_asts;
    double row_evals;
    SecantPipelineMetrics metrics;
    SecantRunnerStats runner_stats;
    int passed = 0;

    memset(&metrics, 0, sizeof(metrics));
    memset(&runner_stats, 0, sizeof(runner_stats));
    if (options->num_kernels > SIZE_MAX / options->asts_per_kernel) {
        return 0;
    }
    asts_per_module =
        options->num_kernels * options->asts_per_kernel;
    if (options->iterations > SIZE_MAX / asts_per_module) {
        return 0;
    }
    total_asts = options->iterations * asts_per_module;
    row_evals =
        (double)total_asts *
        (double)options->run_rows *
        (double)options->run_iterations;
    if (secant_pipeline_backend_uses_eager_cuda(
            options->backend)) {
#if SECANT_BENCH_HAS_CUDA || SECANT_BENCH_HAS_PTX || SECANT_BENCH_HAS_CUBIN
        char error[256] = { 0 };

        if (secant_pipeline_run_nvidia(
                context,
                total_asts,
                &runner_stats)) {
            metrics.compile_window_seconds =
                runner_stats.compile_window_seconds;
            metrics.compile_critical_seconds =
                runner_stats.compile_critical_seconds;
            metrics.compile_work_seconds =
                runner_stats.compile_work_seconds;
            metrics.module_load_seconds =
                runner_stats.module_load_seconds;
            metrics.completion_wait_seconds =
                runner_stats.completion_wait_seconds;
            metrics.module_unload_seconds =
                runner_stats.module_unload_seconds;
            metrics.runtime_seconds =
                runner_stats.runtime_seconds;
            metrics.pipeline_seconds =
                runner_stats.total_seconds;
            metrics.modules_loaded =
                runner_stats.modules_loaded;
            passed = secant_compile_one(
                context,
                0u,
                context->artifacts,
                error,
                sizeof(error));
            if (!passed) {
                fprintf(
                    stderr,
                    "verification artifact compile failed: %s\n",
                    error);
            }
        }
#endif
    } else {
#if SECANT_BENCH_HAS_HIP || SECANT_BENCH_HAS_HSACO
        char error[256] = { 0 };

        if (secant_pipeline_run_amd(
                context,
                total_asts,
                &runner_stats)) {
            metrics.compile_window_seconds =
                runner_stats.compile_window_seconds;
            metrics.compile_critical_seconds =
                runner_stats.compile_critical_seconds;
            metrics.compile_work_seconds =
                runner_stats.compile_work_seconds;
            metrics.module_load_seconds =
                runner_stats.module_load_seconds;
            metrics.completion_wait_seconds =
                runner_stats.completion_wait_seconds;
            metrics.module_unload_seconds =
                runner_stats.module_unload_seconds;
            metrics.runtime_seconds =
                runner_stats.runtime_seconds;
            metrics.pipeline_seconds =
                runner_stats.total_seconds;
            metrics.modules_loaded =
                runner_stats.modules_loaded;
            passed = secant_compile_one(
                context,
                0u,
                context->artifacts,
                error,
                sizeof(error));
            if (!passed) {
                fprintf(
                    stderr,
                    "verification artifact compile failed: %s\n",
                    error);
            }
        }
#endif
    }
    if (!passed) {
        return 0;
    }
    if (!secant_compile_verify(context)) {
        fprintf(stderr, "correctness verification failed\n");
        return 0;
    }
    printf(
        "verify backend=%s modules=1 rows=%zu status=pass\n",
        secant_bench_backend_name(options->backend),
        options->check_rows);
    printf(
        "pipeline backend=%s shape=%s ast_mode=%s seed=%u opt_level=%u "
        "workers=%zu modules=%zu kernels_per_module=%zu asts_per_kernel=%zu "
        "asts_per_module=%zu asts=%zu rows=%zu run_iterations=%zu streams=%zu "
        "tile_rows=%zu threads=%zu compile_window_seconds=%.9f "
        "compile_critical_seconds=%.9f compile_work_seconds=%.9f "
        "module_load_seconds=%.9f "
        "completion_wait_seconds=%.9f module_unload_seconds=%.9f "
        "runtime_seconds=%.9f pipeline_seconds=%.9f "
        "compile_asts_per_second=%.3f row_evals=%.0f "
        "runtime_row_evals_per_second=%.3e "
        "pipeline_row_evals_per_second=%.3e modules_loaded=%zu "
        "gpu_completion=%s "
        "module_transition=%s cuda_module_loading=%s "
        "completion_order=compile_ready_queue device_inputs_resident=1 "
        "timed_transfers=0 cold_template_prepare_timed=0 "
        "template_copy_timed=0 template_source=%s "
        "corpus=%s corpus_hash=%s "
        "expanded_corpus_hash=%016llx\n",
        secant_bench_backend_name(options->backend),
        secant_bench_shape_name(options->shape),
        secant_bench_ast_mode_name(options->ast_mode),
        options->seed,
        options->opt_level,
        options->workers,
        options->iterations,
        options->num_kernels,
        options->asts_per_kernel,
        asts_per_module,
        total_asts,
        options->run_rows,
        options->run_iterations,
        options->num_streams,
        options->tile_rows,
        options->threads_per_block,
        metrics.compile_window_seconds,
        metrics.compile_critical_seconds,
        metrics.compile_work_seconds,
        metrics.module_load_seconds,
        metrics.completion_wait_seconds,
        metrics.module_unload_seconds,
        metrics.runtime_seconds,
        metrics.pipeline_seconds,
        (double)total_asts / metrics.compile_critical_seconds,
        row_evals,
        row_evals / metrics.runtime_seconds,
        row_evals / metrics.pipeline_seconds,
        metrics.modules_loaded,
        options->backend == SECANT_BENCH_BACKEND_HIP ||
            options->backend == SECANT_BENCH_BACKEND_HSACO
            ? "hip_events"
            : "cuda_events",
        secant_pipeline_backend_uses_eager_cuda(
            options->backend)
            ? "eager_load_overlap_wait_unload"
            : "event_wait_unload_load",
        secant_pipeline_backend_uses_eager_cuda(
            options->backend)
            ? "eager"
            : "not_applicable",
#if SECANT_BENCH_HAS_CUBIN
        options->backend == SECANT_BENCH_BACKEND_CUBIN
            ? (context->cubin_template_owned
                ? "runtime_nvrtc"
                : "embedded_cubin")
            : "backend_create",
#else
        "backend_create",
#endif
        secant_bench_ast_corpus_name(options->ast_mode),
        secant_bench_ast_corpus_definition_hash(options->ast_mode),
        (unsigned long long)secant_bench_ast_corpus_hash(
            context->programs,
            total_asts));
    return 1;
}
#endif

int
main(int argc, char** argv) {
    SecantCompileContext context;
    int arg_idx;
    int result = 1;

    memset(&context, 0, sizeof(context));
    secant_bench_options_default(&context.options);
#if SECANT_BENCH_PIPELINE
    context.options.backend = SECANT_BENCH_BACKEND_CUBIN;
    context.options.shape = SECANT_BENCH_SHAPE_SSE;
    context.options.ast_mode = SECANT_BENCH_AST_MODE_ALU;
    context.options.warmups = 0u;
    context.options.iterations = 48u;
    context.options.workers = 24u;
    context.options.num_kernels = 64u;
    context.options.asts_per_kernel = 128u;
    context.options.tile_rows = 8192u;
    context.options.num_streams = 8u;
    context.options.run_rows = 131072u;
    context.options.run_iterations = 1u;
    context.options.check_modules = 1u;
    context.options.target_sm = 0u;
#endif
    for (arg_idx = 1; arg_idx < argc; ++arg_idx) {
        if (strcmp(argv[arg_idx], "--help") == 0) {
#if SECANT_BENCH_PIPELINE
            secant_pipeline_usage(argv[0]);
#else
            secant_compile_usage(argv[0]);
#endif
            return 0;
        }
        if (!secant_bench_parse_common_option(argc, argv, &arg_idx, &context.options)) {
#if SECANT_BENCH_PIPELINE
            secant_pipeline_usage(argv[0]);
#else
            secant_compile_usage(argv[0]);
#endif
            return 1;
        }
    }
#if SECANT_BENCH_PIPELINE && \
    (SECANT_BENCH_HAS_CUDA || \
     SECANT_BENCH_HAS_PTX || \
     SECANT_BENCH_HAS_CUBIN)
    if (secant_pipeline_backend_uses_eager_cuda(
            context.options.backend) &&
        setenv("CUDA_MODULE_LOADING", "EAGER", 1) != 0) {
        fprintf(
            stderr,
            "failed to enable CUDA eager module loading\n");
        return 1;
    }
#endif
    if (
#if SECANT_BENCH_PIPELINE
        !secant_pipeline_target_sm_resolve(&context.options) ||
#endif
        !secant_bench_options_valid(&context.options) ||
        !secant_compile_backend_available(context.options.backend) ||
        (context.options.backend == SECANT_BENCH_BACKEND_CUBIN &&
         context.options.opt_level != 1u) ||
#if SECANT_BENCH_PIPELINE
        context.options.warmups != 0u ||
        context.options.check_modules != 1u ||
        !SECANT_BENCH_HAS_OPENMP ||
#endif
        (!SECANT_BENCH_HAS_OPENMP && context.options.workers != 1u)) {
        fprintf(stderr, "invalid or unavailable benchmark configuration\n");
        return 1;
    }
    if (!secant_compile_prepare(&context)) {
        fprintf(stderr, "benchmark setup failed\n");
    } else if (
#if SECANT_BENCH_PIPELINE
        secant_pipeline_run(&context)
#else
        secant_compile_run(&context)
#endif
    ) {
        result = 0;
    }
    secant_compile_cleanup(&context);
    return result;
}
