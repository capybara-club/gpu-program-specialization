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
#include "secant.h"

#ifndef SECANT_BENCH_HAS_CUDA
#define SECANT_BENCH_HAS_CUDA 0
#endif
#ifndef SECANT_BENCH_HAS_PTX
#define SECANT_BENCH_HAS_PTX 0
#endif
#ifndef SECANT_BENCH_HAS_CUBIN
#define SECANT_BENCH_HAS_CUBIN 0
#endif
#ifndef SECANT_BENCH_END_TO_END
#define SECANT_BENCH_END_TO_END 0
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
#endif

typedef struct SecantRuntimeContext {
    SecantBenchOptions options;
    SecantAstInstruction* programs;
    const SecantAstInstruction** asts;
    const SecantAstInstruction* const* routines;
    const char* const* routine_names;
    size_t num_routines;
    unsigned char* compile_scratch;
    float* host_input;
    float* host_targets;
    float* expected;
    float* actual;
    size_t output_count_per_kernel;
    size_t output_count_total;
    double compile_seconds;
    double module_load_seconds;
#if SECANT_BENCH_HAS_CUDA
    SecantCUDACompiled cuda_compiled;
#endif
#if SECANT_BENCH_HAS_PTX
    SecantPTXHandle ptx_handle;
    SecantPTXCompiled ptx_compiled;
#endif
#if SECANT_BENCH_HAS_CUBIN
    SecantCubinPlan* cubin_handle;
    void* cubin_workspace;
    unsigned char* cubin_template;
    unsigned char* cubin_binary;
    size_t cubin_size;
#endif
#if SECANT_BENCH_HAS_CUDA || SECANT_BENCH_HAS_PTX || SECANT_BENCH_HAS_CUBIN
    CUdevice cuda_device;
    CUcontext cuda_context;
    CUcontext cuda_previous_context;
    void* cuda_module;
    void** cuda_functions;
    CUstream* cuda_streams;
    CUevent* cuda_done;
    CUevent cuda_start;
    CUevent cuda_stop;
    CUdeviceptr cuda_input;
    CUdeviceptr cuda_targets;
    CUdeviceptr cuda_output;
    int cuda_context_retained;
#endif
} SecantRuntimeContext;

static double
secant_runtime_wall_seconds(void) {
    struct timespec timestamp;

    if (clock_gettime(CLOCK_MONOTONIC, &timestamp) != 0) {
        return -1.0;
    }
    return (double)timestamp.tv_sec + (double)timestamp.tv_nsec * 1.0e-9;
}

static void
secant_runtime_usage(const char* program) {
    fprintf(
        stderr,
        "usage: %s [options]\n"
        "  --backend cuda|ptx|cubin\n"
        "  --shape materialize|sse\n"
        "  --ast-mode simple|alu|mufu\n"
        "  --warmups N --run-iterations N --run-rows N\n"
        "  --kernels N --asts-per-kernel N --inputs N --targets N\n"
        "  --tile-rows N --threads N --streams N --patch-instructions-per-ast N\n"
        "  --compile-scratch-bytes N\n"
        "  --check-rows N\n"
        "  --source-sm NN --target-sm NN --opt-level 0|1 --device N\n",
        program);
}

static int
secant_runtime_backend_available(SecantBenchBackend backend) {
    switch (backend) {
        case SECANT_BENCH_BACKEND_CUDA: return SECANT_BENCH_HAS_CUDA;
        case SECANT_BENCH_BACKEND_PTX: return SECANT_BENCH_HAS_PTX;
        case SECANT_BENCH_BACKEND_CUBIN: return SECANT_BENCH_HAS_CUBIN;
        default: return 0;
    }
}

static int
secant_runtime_output_count_per_kernel(
    const SecantRuntimeContext* context,
    size_t* output_count_ret
) {
    const SecantBenchOptions* options = &context->options;
    const size_t count = options->asts_per_kernel;
    const size_t stride = options->shape == SECANT_BENCH_SHAPE_MATERIALIZE
        ? options->run_rows
        : options->num_targets;

    if (count == 0u || stride == 0u || count > SIZE_MAX / stride) {
        return 0;
    }
    *output_count_ret = count * stride;
    return 1;
}

static int
secant_runtime_compile_cuda(SecantRuntimeContext* context) {
#if SECANT_BENCH_HAS_CUDA
    const char* compile_options[] = {
        "--no-cache",
        "--split-compile=1",
        NULL
    };
    const SecantBenchOptions* options = &context->options;
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
            context->asts,
            major,
            minor,
            compile_options,
            sizeof(compile_options) / sizeof(compile_options[0]),
            false,
            context->compile_scratch,
            options->compile_scratch_size,
            NULL,
            0u,
            NULL,
            &context->cuda_compiled);
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
            context->asts,
            major,
            minor,
            compile_options,
            sizeof(compile_options) / sizeof(compile_options[0]),
            false,
            context->compile_scratch,
            options->compile_scratch_size,
            NULL,
            0u,
            NULL,
            &context->cuda_compiled);
    }
    if (result != SECANT_CUDA_SUCCESS) {
        fprintf(stderr, "CUDA compile failed: %s\n", secant_cuda_result_to_string(result));
        return 0;
    }
    return 1;
#else
    (void)context;
    return 0;
#endif
}

#if SECANT_BENCH_HAS_PTX
static int
secant_runtime_prepare_ptx(SecantRuntimeContext* context) {
    static const char* const nvrtc_options[] = {
        "--no-cache",
        "--split-compile=1"
    };
    const SecantBenchOptions* options = &context->options;
    const uint32_t source_major = options->source_sm / 10u;
    const uint32_t source_minor = options->source_sm % 10u;
    SecantPTXResult result;

    if (options->shape == SECANT_BENCH_SHAPE_MATERIALIZE) {
        result = secant_ptx_materialize_create(
            options->num_kernels,
            options->asts_per_kernel,
            options->num_inputs,
            source_major,
            source_minor,
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
            source_major,
            source_minor,
            nvrtc_options,
            sizeof(nvrtc_options) / sizeof(nvrtc_options[0]),
            false,
            NULL,
            0u,
            NULL,
            &context->ptx_handle);
    }
    if (result != SECANT_PTX_SUCCESS) {
        fprintf(stderr, "PTX template preparation failed: %s\n", secant_ptx_result_to_string(result));
        return 0;
    }
    return 1;
}
#endif

static int
secant_runtime_compile_ptx(SecantRuntimeContext* context) {
#if SECANT_BENCH_HAS_PTX
    const char* compile_options[] = {
        NULL,
        "--split-compile=1"
    };
    const SecantBenchOptions* options = &context->options;
    char opt_level[32];
    const uint32_t target_major = options->target_sm / 10u;
    const uint32_t target_minor = options->target_sm % 10u;
    SecantPTXResult result;

    snprintf(opt_level, sizeof(opt_level), "--opt-level=%u", options->opt_level);
    compile_options[0] = opt_level;
    result = secant_ptx_compile(
        context->ptx_handle,
        context->routines,
        context->num_routines,
        context->asts,
        target_major,
        target_minor,
        compile_options,
        sizeof(compile_options) / sizeof(compile_options[0]),
        false,
        context->compile_scratch,
        options->compile_scratch_size,
        NULL,
        0u,
        NULL,
        &context->ptx_compiled);
    if (result != SECANT_PTX_SUCCESS) {
        fprintf(stderr, "PTX compile failed: %s\n", secant_ptx_result_to_string(result));
        return 0;
    }
    return 1;
#else
    (void)context;
    return 0;
#endif
}

#if SECANT_BENCH_HAS_CUBIN
static int
secant_runtime_prepare_cubin(SecantRuntimeContext* context) {
    const SecantBenchOptions* options = &context->options;
    SecantCubinMaterializeRecipe materialize_recipe;
    SecantCubinSSERecipe sse_recipe;
    char error[16384];
    size_t patch_capacity_instructions;
    size_t workspace_size = 0u;
    SecantResult result;

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
            error,
            sizeof(error),
            &context->cubin_template,
            &context->cubin_size)) {
        fprintf(stderr, "CUBIN template compilation failed: %s\n", error);
        return 0;
    }
    if (options->shape == SECANT_BENCH_SHAPE_MATERIALIZE) {
        result = secant_cubin_materialize_inspect(
            &materialize_recipe,
            context->cubin_template,
            context->cubin_size,
            NULL,
            0u,
            &workspace_size,
            &context->cubin_handle);
    } else {
        result = secant_cubin_sse_inspect(
            &sse_recipe,
            context->cubin_template,
            context->cubin_size,
            NULL,
            0u,
            &workspace_size,
            &context->cubin_handle);
    }
    if (result == SECANT_SUCCESS) {
        context->cubin_workspace = malloc(workspace_size);
        if (context->cubin_workspace == NULL) {
            return 0;
        }
        if (options->shape == SECANT_BENCH_SHAPE_MATERIALIZE) {
            result = secant_cubin_materialize_inspect(
                &materialize_recipe,
                context->cubin_template,
                context->cubin_size,
                context->cubin_workspace,
                workspace_size,
                &workspace_size,
                &context->cubin_handle);
        } else {
            result = secant_cubin_sse_inspect(
                &sse_recipe,
                context->cubin_template,
                context->cubin_size,
                context->cubin_workspace,
                workspace_size,
                &workspace_size,
                &context->cubin_handle);
        }
    }
    if (result != SECANT_SUCCESS) {
        fprintf(
            stderr,
            "CUBIN template inspection failed: SecantResult(%d)\n",
            (int)result);
        return 0;
    }
    context->cubin_binary = (unsigned char*)malloc(context->cubin_size);
    if (context->cubin_binary == NULL) {
        return 0;
    }
    memcpy(context->cubin_binary, context->cubin_template, context->cubin_size);
    return 1;
}

static int
secant_runtime_compile_cubin(SecantRuntimeContext* context) {
    const SecantBenchOptions* options = &context->options;
    SecantResult result;

    result = secant_cubin_specialize_into(
        context->cubin_handle,
        context->routines,
        context->num_routines,
        context->asts,
        options->num_kernels * options->asts_per_kernel,
        context->cubin_binary,
        context->cubin_size);
    if (result != SECANT_SUCCESS) {
        fprintf(
            stderr,
            "CUBIN specialization failed: SecantResult(%d)\n",
            (int)result);
        return 0;
    }
    return 1;
}
#endif


static int
secant_runtime_backend_prepare(SecantRuntimeContext* context) {
    switch (context->options.backend) {
        case SECANT_BENCH_BACKEND_CUDA:
            return 1;
#if SECANT_BENCH_HAS_PTX
        case SECANT_BENCH_BACKEND_PTX:
            return secant_runtime_prepare_ptx(context);
#endif
#if SECANT_BENCH_HAS_CUBIN
        case SECANT_BENCH_BACKEND_CUBIN:
            return secant_runtime_prepare_cubin(context);
#endif
        default:
            return 0;
    }
}

static int
secant_runtime_compile(SecantRuntimeContext* context) {
    switch (context->options.backend) {
        case SECANT_BENCH_BACKEND_CUDA: return secant_runtime_compile_cuda(context);
        case SECANT_BENCH_BACKEND_PTX: return secant_runtime_compile_ptx(context);
#if SECANT_BENCH_HAS_CUBIN
        case SECANT_BENCH_BACKEND_CUBIN: return secant_runtime_compile_cubin(context);
#endif
        default: return 0;
    }
}

#if SECANT_BENCH_HAS_CUDA || SECANT_BENCH_HAS_PTX || SECANT_BENCH_HAS_CUBIN
static int
secant_runtime_cuda_setup(SecantRuntimeContext* context) {
    const SecantBenchOptions* options = &context->options;
    const size_t input_bytes = options->num_inputs * options->run_rows * sizeof(float);
    const size_t target_bytes = options->num_targets * options->run_rows * sizeof(float);
    const size_t output_bytes = context->output_count_total * sizeof(float);
    const void* binary = NULL;
    size_t binary_size = 0u;
    size_t stream_idx;
    double load_start;

    if (cuInit(0u) != CUDA_SUCCESS ||
        cuDeviceGet(&context->cuda_device, options->device_ordinal) != CUDA_SUCCESS ||
        cuDevicePrimaryCtxRetain(
            &context->cuda_context,
            context->cuda_device) != CUDA_SUCCESS) {
        return 0;
    }
    context->cuda_context_retained = 1;
    context->cuda_streams = (CUstream*)calloc(
        options->num_streams,
        sizeof(*context->cuda_streams));
    context->cuda_done = (CUevent*)calloc(
        options->num_streams,
        sizeof(*context->cuda_done));
    if (context->cuda_streams == NULL || context->cuda_done == NULL) {
        return 0;
    }
    if (cuCtxGetCurrent(&context->cuda_previous_context) != CUDA_SUCCESS ||
        cuCtxSetCurrent(context->cuda_context) != CUDA_SUCCESS ||
        cuEventCreate(&context->cuda_start, CU_EVENT_DEFAULT) != CUDA_SUCCESS ||
        cuEventCreate(&context->cuda_stop, CU_EVENT_DEFAULT) != CUDA_SUCCESS ||
        cuMemAlloc(&context->cuda_input, input_bytes) != CUDA_SUCCESS ||
        cuMemcpyHtoD(context->cuda_input, context->host_input, input_bytes) != CUDA_SUCCESS ||
        cuMemAlloc(&context->cuda_output, output_bytes) != CUDA_SUCCESS) {
        return 0;
    }
    for (stream_idx = 0u; stream_idx < options->num_streams; ++stream_idx) {
        if (cuStreamCreate(
                &context->cuda_streams[stream_idx],
                CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS ||
            cuEventCreate(
                &context->cuda_done[stream_idx],
                CU_EVENT_DISABLE_TIMING) != CUDA_SUCCESS) {
            return 0;
        }
    }
    if (options->shape == SECANT_BENCH_SHAPE_SSE &&
        (cuMemAlloc(&context->cuda_targets, target_bytes) != CUDA_SUCCESS ||
         cuMemcpyHtoD(
            context->cuda_targets,
            context->host_targets,
            target_bytes) != CUDA_SUCCESS)) {
        return 0;
    }
    context->cuda_functions = (void**)calloc(
        options->num_kernels,
        sizeof(*context->cuda_functions));
    if (context->cuda_functions == NULL) {
        return 0;
    }

#if SECANT_BENCH_HAS_CUDA
    if (options->backend == SECANT_BENCH_BACKEND_CUDA) {
        if (secant_cuda_compiled_binary_get(
            context->cuda_compiled,
            &binary,
            &binary_size) != SECANT_CUDA_SUCCESS) {
            return 0;
        }
    } else
#endif
#if SECANT_BENCH_HAS_PTX
    if (options->backend == SECANT_BENCH_BACKEND_PTX) {
        if (secant_ptx_compiled_binary_get(
            context->ptx_compiled,
            &binary,
            &binary_size) != SECANT_PTX_SUCCESS) {
            return 0;
        }
    } else
#endif
#if SECANT_BENCH_HAS_CUBIN
    if (options->backend == SECANT_BENCH_BACKEND_CUBIN) {
        binary = context->cubin_binary;
        binary_size = context->cubin_size;
    } else
#endif
    {
        return 0;
    }
    load_start = secant_runtime_wall_seconds();
    if (load_start < 0.0) {
        return 0;
    }
    if (secant_test_cuda_module_load(
            binary,
            binary_size,
            options->num_kernels,
            &context->cuda_module,
            context->cuda_functions) != SECANT_CUDA_SUCCESS) {
        fprintf(stderr, "CUDA module load failed\n");
        return 0;
    }
    context->module_load_seconds = secant_runtime_wall_seconds() - load_start;
    if (context->module_load_seconds < 0.0) {
        return 0;
    }
    return 1;
}

static int
secant_runtime_cuda_launch(
    SecantRuntimeContext* context,
    size_t kernel_idx,
    size_t num_rows,
    size_t stream_idx
) {
    const SecantBenchOptions* options = &context->options;
    const size_t output_count = context->output_count_per_kernel;
    const size_t output_offset = kernel_idx * output_count * sizeof(float);
    const CUdeviceptr output = context->cuda_output + output_offset;
    const CUstream stream = context->cuda_streams[stream_idx];

    if (options->shape == SECANT_BENCH_SHAPE_MATERIALIZE) {
        return secant_test_cuda_run_static_column_materialize(
            context->cuda_functions[kernel_idx],
            options->num_inputs,
            options->asts_per_kernel,
            (const float*)(uintptr_t)context->cuda_input,
            options->num_inputs * options->run_rows,
            options->run_rows,
            num_rows,
            stream,
            (float*)(uintptr_t)output,
            output_count,
            options->run_rows) == SECANT_CUDA_SUCCESS;
    }
    return secant_test_cuda_run_static_column_sse(
        context->cuda_functions[kernel_idx],
        options->num_inputs,
        options->asts_per_kernel,
        options->num_targets,
        options->tile_rows,
        options->threads_per_block,
        (const float*)(uintptr_t)context->cuda_input,
        options->num_inputs * options->run_rows,
        options->run_rows,
        (const float*)(uintptr_t)context->cuda_targets,
        options->num_targets * options->run_rows,
        options->run_rows,
        num_rows,
        stream,
        (float*)(uintptr_t)output,
        output_count,
        options->num_targets) == SECANT_CUDA_SUCCESS;
}
#endif


static int
secant_runtime_launch(
    SecantRuntimeContext* context,
    size_t kernel_idx,
    size_t num_rows,
    size_t stream_idx
) {
    switch (context->options.backend) {
#if SECANT_BENCH_HAS_CUDA || SECANT_BENCH_HAS_PTX || SECANT_BENCH_HAS_CUBIN
        case SECANT_BENCH_BACKEND_CUDA:
        case SECANT_BENCH_BACKEND_PTX:
        case SECANT_BENCH_BACKEND_CUBIN:
            return secant_runtime_cuda_launch(
                context,
                kernel_idx,
                num_rows,
                stream_idx);
#endif
        default:
            return 0;
    }
}

static int
secant_runtime_clear_output(SecantRuntimeContext* context) {
    const size_t output_bytes = context->output_count_total * sizeof(float);

    switch (context->options.backend) {
#if SECANT_BENCH_HAS_CUDA || SECANT_BENCH_HAS_PTX || SECANT_BENCH_HAS_CUBIN
        case SECANT_BENCH_BACKEND_CUDA:
        case SECANT_BENCH_BACKEND_PTX:
        case SECANT_BENCH_BACKEND_CUBIN:
            return cuMemsetD8Async(
                context->cuda_output,
                0u,
                output_bytes,
                context->cuda_streams[0]) == CUDA_SUCCESS;
#endif
        default:
            return 0;
    }
}

static int
secant_runtime_synchronize(SecantRuntimeContext* context) {
    size_t stream_idx;

    switch (context->options.backend) {
#if SECANT_BENCH_HAS_CUDA || SECANT_BENCH_HAS_PTX || SECANT_BENCH_HAS_CUBIN
        case SECANT_BENCH_BACKEND_CUDA:
        case SECANT_BENCH_BACKEND_PTX:
        case SECANT_BENCH_BACKEND_CUBIN:
            for (stream_idx = 0u;
                 stream_idx < context->options.num_streams;
                 ++stream_idx) {
                if (cuStreamSynchronize(
                        context->cuda_streams[stream_idx]) != CUDA_SUCCESS) {
                    return 0;
                }
            }
            return 1;
#endif
        default:
            return 0;
    }
}

static int
secant_runtime_copy_output(SecantRuntimeContext* context, size_t kernel_idx) {
    const size_t output_bytes = context->output_count_per_kernel * sizeof(float);

    switch (context->options.backend) {
#if SECANT_BENCH_HAS_CUDA || SECANT_BENCH_HAS_PTX || SECANT_BENCH_HAS_CUBIN
        case SECANT_BENCH_BACKEND_CUDA:
        case SECANT_BENCH_BACKEND_PTX:
        case SECANT_BENCH_BACKEND_CUBIN:
            return cuMemcpyDtoH(
                context->actual,
                context->cuda_output + kernel_idx * output_bytes,
                output_bytes) == CUDA_SUCCESS;
#endif
        default:
            return 0;
    }
}

static int
secant_runtime_compare_materialize(
    const SecantRuntimeContext* context,
    float* max_abs_error_ret,
    size_t* worst_idx_ret
) {
    const SecantBenchOptions* options = &context->options;
    float max_abs_error = 0.0f;
    size_t worst_idx = 0u;
    size_t ast_idx;

    for (ast_idx = 0u; ast_idx < options->asts_per_kernel; ++ast_idx) {
        float ast_error;
        size_t ast_worst;

        if (!secant_bench_compare(
                options->ast_mode,
                context->expected + ast_idx * options->run_rows,
                context->actual + ast_idx * options->run_rows,
                options->check_rows,
                &ast_error,
                &ast_worst)) {
            *max_abs_error_ret = ast_error;
            *worst_idx_ret = ast_idx * options->run_rows + ast_worst;
            return 0;
        }
        if (ast_error > max_abs_error) {
            max_abs_error = ast_error;
            worst_idx = ast_idx * options->run_rows + ast_worst;
        }
    }
    *max_abs_error_ret = max_abs_error;
    *worst_idx_ret = worst_idx;
    return 1;
}

static int
secant_runtime_verify(SecantRuntimeContext* context) {
    const SecantBenchOptions* options = &context->options;
    const size_t output_count = context->output_count_per_kernel;
    size_t kernel_idx;

    for (kernel_idx = 0u; kernel_idx < options->num_kernels; ++kernel_idx) {
        const SecantAstInstruction* const* asts =
            context->asts + kernel_idx * options->asts_per_kernel;
        float max_abs_error = 0.0f;
        size_t worst_idx = 0u;
        int close;

        memset(context->expected, 0, output_count * sizeof(float));
        if (!secant_runtime_clear_output(context)) {
            return 0;
        }
        if (options->shape == SECANT_BENCH_SHAPE_MATERIALIZE) {
            if (secant_cpu_run_static_column_materialize(
                    options->num_inputs,
                    context->routines,
                    context->num_routines,
                    asts,
                    options->asts_per_kernel,
                    context->host_input,
                    options->num_inputs * options->run_rows,
                    options->run_rows,
                    options->check_rows,
                    context->expected,
                    output_count,
                    options->run_rows) != SECANT_SUCCESS) {
                return 0;
            }
        } else if (secant_cpu_run_static_column_sse(
                options->num_inputs,
                options->num_targets,
                context->routines,
                context->num_routines,
                asts,
                options->asts_per_kernel,
                context->host_input,
                options->num_inputs * options->run_rows,
                options->run_rows,
                context->host_targets,
                options->num_targets * options->run_rows,
                options->run_rows,
                options->check_rows,
                context->expected,
                output_count,
                options->num_targets) != SECANT_SUCCESS) {
            return 0;
        }
        if (!secant_runtime_launch(
                context,
                kernel_idx,
                options->check_rows,
                0u)) {
            return 0;
        }
        if (!secant_runtime_synchronize(context) ||
            !secant_runtime_copy_output(context, kernel_idx)) {
            return 0;
        }
        close = options->shape == SECANT_BENCH_SHAPE_MATERIALIZE
            ? secant_runtime_compare_materialize(context, &max_abs_error, &worst_idx)
            : secant_bench_compare(
                options->ast_mode,
                context->expected,
                context->actual,
                output_count,
                &max_abs_error,
                &worst_idx);
        if (!close) {
            fprintf(
                stderr,
                "verification failed: kernel=%zu worst_idx=%zu max_abs_error=%g\n",
                kernel_idx,
                worst_idx,
                (double)max_abs_error);
            return 0;
        }
    }
    return 1;
}

static int
secant_runtime_time_cuda(SecantRuntimeContext* context, double* seconds_ret) {
#if SECANT_BENCH_HAS_CUDA || SECANT_BENCH_HAS_PTX || SECANT_BENCH_HAS_CUBIN
    const SecantBenchOptions* options = &context->options;
    float milliseconds;
    size_t iteration;
    size_t kernel_idx;
    size_t stream_idx;

    if (cuEventRecord(
            context->cuda_start,
            context->cuda_streams[0]) != CUDA_SUCCESS) {
        return 0;
    }
    for (stream_idx = 1u; stream_idx < options->num_streams; ++stream_idx) {
        if (cuStreamWaitEvent(
                context->cuda_streams[stream_idx],
                context->cuda_start,
                0u) != CUDA_SUCCESS) {
            return 0;
        }
    }
    for (iteration = 0u; iteration < options->run_iterations; ++iteration) {
        for (kernel_idx = 0u; kernel_idx < options->num_kernels; ++kernel_idx) {
            stream_idx = kernel_idx % options->num_streams;
            if (!secant_runtime_cuda_launch(
                    context,
                    kernel_idx,
                    options->run_rows,
                    stream_idx)) {
                return 0;
            }
        }
    }
    for (stream_idx = 0u; stream_idx < options->num_streams; ++stream_idx) {
        if (cuEventRecord(
                context->cuda_done[stream_idx],
                context->cuda_streams[stream_idx]) != CUDA_SUCCESS) {
            return 0;
        }
    }
    for (stream_idx = 1u; stream_idx < options->num_streams; ++stream_idx) {
        if (cuStreamWaitEvent(
                context->cuda_streams[0],
                context->cuda_done[stream_idx],
                0u) != CUDA_SUCCESS) {
            return 0;
        }
    }
    if (cuEventRecord(context->cuda_stop, context->cuda_streams[0]) != CUDA_SUCCESS ||
        cuEventSynchronize(context->cuda_stop) != CUDA_SUCCESS ||
        cuEventElapsedTime(
            &milliseconds,
            context->cuda_start,
            context->cuda_stop) != CUDA_SUCCESS) {
        return 0;
    }
    *seconds_ret = (double)milliseconds * 1.0e-3;
    return 1;
#else
    (void)context;
    (void)seconds_ret;
    return 0;
#endif
}

static int
secant_runtime_time(SecantRuntimeContext* context, double* seconds_ret) {
    const SecantBenchOptions* options = &context->options;
    const double row_evals =
        (double)options->run_iterations *
        (double)options->num_kernels *
        (double)options->asts_per_kernel *
        (double)options->run_rows;
    size_t warmup;
    size_t kernel_idx;
    double seconds = 0.0;
    int passed;

    for (warmup = 0u; warmup < options->warmups; ++warmup) {
        for (kernel_idx = 0u; kernel_idx < options->num_kernels; ++kernel_idx) {
            if (!secant_runtime_launch(
                    context,
                    kernel_idx,
                    options->run_rows,
                    kernel_idx % options->num_streams)) {
                return 0;
            }
        }
    }
    if (!secant_runtime_synchronize(context) || !secant_runtime_clear_output(context)) {
        return 0;
    }
    switch (options->backend) {
        case SECANT_BENCH_BACKEND_CUDA:
        case SECANT_BENCH_BACKEND_PTX:
        case SECANT_BENCH_BACKEND_CUBIN:
            passed = secant_runtime_time_cuda(context, &seconds);
            break;
        default:
            passed = 0;
            break;
    }
    if (!passed || seconds <= 0.0) {
        return 0;
    }
    if (seconds_ret != NULL) {
        *seconds_ret = seconds;
    }
    printf(
        "runtime backend=%s shape=%s ast_mode=%s seed=%u opt_level=%u kernels=%zu "
        "asts_per_kernel=%zu tile_rows=%zu threads=%zu streams=%zu rows=%zu "
        "device_inputs_resident=1 timed_transfers=0 "
        "iterations=%zu timed_batches=%zu kernel_launches=%.0f seconds=%.6f "
        "row_evals=%.0f row_evals_per_second=%.3e corpus=%s corpus_hash=%s "
        "expanded_corpus_hash=%016llx\n",
        secant_bench_backend_name(options->backend),
        secant_bench_shape_name(options->shape),
        secant_bench_ast_mode_name(options->ast_mode),
        options->seed,
        options->opt_level,
        options->num_kernels,
        options->asts_per_kernel,
        options->tile_rows,
        options->threads_per_block,
        options->num_streams,
        options->run_rows,
        options->run_iterations,
        options->run_iterations,
        (double)options->run_iterations * (double)options->num_kernels,
        seconds,
        row_evals,
        row_evals / seconds,
        secant_bench_ast_corpus_name(options->ast_mode),
        secant_bench_ast_corpus_definition_hash(options->ast_mode),
        (unsigned long long)secant_bench_ast_corpus_hash(
            context->programs,
            options->num_kernels * options->asts_per_kernel));
    return 1;
}

static int
secant_runtime_prepare(SecantRuntimeContext* context) {
    SecantBenchOptions* options = &context->options;
    size_t output_count;
    size_t program_bytes;
    size_t pointer_bytes;
    double compile_start;

    if (options->check_rows > options->run_rows ||
        !secant_bench_ast_storage_sizes(
            1u,
            options->num_kernels,
            options->asts_per_kernel,
            &program_bytes,
            &pointer_bytes)) {
        return 0;
    }
    if (!secant_runtime_output_count_per_kernel(context, &output_count) ||
        options->num_kernels > SIZE_MAX / output_count ||
        output_count * options->num_kernels > SIZE_MAX / sizeof(float)) {
        return 0;
    }
    context->output_count_per_kernel = output_count;
    context->output_count_total = output_count * options->num_kernels;
    context->programs = (SecantAstInstruction*)malloc(program_bytes);
    context->asts = (const SecantAstInstruction**)malloc(pointer_bytes);
    if (options->backend != SECANT_BENCH_BACKEND_CUBIN) {
        context->compile_scratch =
            (unsigned char*)malloc(options->compile_scratch_size);
    }
    context->host_input = (float*)malloc(
        options->num_inputs * options->run_rows * sizeof(float));
    context->host_targets = (float*)malloc(
        options->num_targets * options->run_rows * sizeof(float));
    context->expected = (float*)malloc(output_count * sizeof(float));
    context->actual = (float*)malloc(output_count * sizeof(float));
    if (context->programs == NULL || context->asts == NULL ||
        (options->backend != SECANT_BENCH_BACKEND_CUBIN &&
         context->compile_scratch == NULL) ||
        context->host_input == NULL ||
        context->host_targets == NULL || context->expected == NULL ||
        context->actual == NULL) {
        return 0;
    }
    secant_bench_ast_fill(
        1u,
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
    secant_bench_fill_input(
        context->host_input,
        options->num_inputs,
        options->run_rows,
        options->seed);
    secant_bench_fill_targets(
        context->host_targets,
        options->num_targets,
        options->run_rows,
        options->seed);
    if (!secant_runtime_backend_prepare(context)) {
        return 0;
    }
    compile_start = secant_runtime_wall_seconds();
    if (compile_start < 0.0) {
        return 0;
    }
    if (!secant_runtime_compile(context)) {
        return 0;
    }
    context->compile_seconds = secant_runtime_wall_seconds() - compile_start;
    if (context->compile_seconds < 0.0) {
        return 0;
    }

    switch (options->backend) {
#if SECANT_BENCH_HAS_CUDA || SECANT_BENCH_HAS_PTX || SECANT_BENCH_HAS_CUBIN
        case SECANT_BENCH_BACKEND_CUDA:
        case SECANT_BENCH_BACKEND_PTX:
        case SECANT_BENCH_BACKEND_CUBIN:
            return secant_runtime_cuda_setup(context);
#endif
        default:
            return 0;
    }
}

static void
secant_runtime_cleanup(SecantRuntimeContext* context) {
    size_t stream_idx;

#if SECANT_BENCH_HAS_CUDA || SECANT_BENCH_HAS_PTX || SECANT_BENCH_HAS_CUBIN
    if (context->cuda_module != NULL) {
        (void)cuModuleUnload((CUmodule)context->cuda_module);
    }
    free(context->cuda_functions);
    if (context->cuda_output != 0u) {
        (void)cuMemFree(context->cuda_output);
    }
    if (context->cuda_targets != 0u) {
        (void)cuMemFree(context->cuda_targets);
    }
    if (context->cuda_input != 0u) {
        (void)cuMemFree(context->cuda_input);
    }
    if (context->cuda_stop != NULL) {
        (void)cuEventDestroy(context->cuda_stop);
    }
    if (context->cuda_start != NULL) {
        (void)cuEventDestroy(context->cuda_start);
    }
    if (context->cuda_done != NULL) {
        for (stream_idx = 0u;
             stream_idx < context->options.num_streams;
             ++stream_idx) {
            if (context->cuda_done[stream_idx] != NULL) {
                (void)cuEventDestroy(context->cuda_done[stream_idx]);
            }
        }
    }
    if (context->cuda_streams != NULL) {
        for (stream_idx = 0u;
             stream_idx < context->options.num_streams;
             ++stream_idx) {
            if (context->cuda_streams[stream_idx] != NULL) {
                (void)cuStreamDestroy(context->cuda_streams[stream_idx]);
            }
        }
    }
    free(context->cuda_done);
    free(context->cuda_streams);
    if (context->cuda_context_retained) {
        (void)cuCtxSetCurrent(context->cuda_previous_context);
        (void)cuDevicePrimaryCtxRelease(context->cuda_device);
    }
#endif
#if SECANT_BENCH_HAS_PTX
    if (context->ptx_compiled != NULL) {
        (void)secant_ptx_compiled_destroy(context->ptx_compiled);
    }
    if (context->ptx_handle != NULL) {
        (void)secant_ptx_handle_destroy(context->ptx_handle);
    }
#endif
#if SECANT_BENCH_HAS_CUDA
    if (context->cuda_compiled != NULL) {
        (void)secant_cuda_compiled_destroy(context->cuda_compiled);
    }
#endif
#if SECANT_BENCH_HAS_CUBIN
    free(context->cubin_workspace);
    free(context->cubin_binary);
    free(context->cubin_template);
#endif
    free(context->actual);
    free(context->expected);
    free(context->host_targets);
    free(context->host_input);
    free(context->compile_scratch);
    free(context->asts);
    free(context->programs);
}

#if SECANT_BENCH_END_TO_END
static void
secant_runtime_print_end_to_end(
    const SecantRuntimeContext* context,
    double run_seconds
) {
    const SecantBenchOptions* options = &context->options;
    const double asts =
        (double)options->num_kernels *
        (double)options->asts_per_kernel;
    const double row_evals =
        (double)options->run_iterations *
        asts *
        (double)options->run_rows;
    const double pipeline_seconds =
        context->compile_seconds +
        context->module_load_seconds +
        run_seconds;

    printf(
        "end_to_end backend=%s shape=%s ast_mode=%s seed=%u opt_level=%u "
        "kernels=%zu asts_per_kernel=%zu asts=%.0f tile_rows=%zu threads=%zu "
        "streams=%zu rows=%zu iterations=%zu compile_seconds=%.9f "
        "load_seconds=%.9f run_seconds=%.9f pipeline_seconds=%.9f "
        "compile_asts_per_second=%.3f load_asts_per_second=%.3f "
        "row_evals=%.0f runtime_row_evals_per_second=%.3e "
        "pipeline_row_evals_per_second=%.3e device_inputs_resident=1 "
        "timed_transfers=0 cold_template_prepare_timed=0 template_copy_timed=0 "
        "corpus=%s corpus_hash=%s expanded_corpus_hash=%016llx\n",
        secant_bench_backend_name(options->backend),
        secant_bench_shape_name(options->shape),
        secant_bench_ast_mode_name(options->ast_mode),
        options->seed,
        options->opt_level,
        options->num_kernels,
        options->asts_per_kernel,
        asts,
        options->tile_rows,
        options->threads_per_block,
        options->num_streams,
        options->run_rows,
        options->run_iterations,
        context->compile_seconds,
        context->module_load_seconds,
        run_seconds,
        pipeline_seconds,
        asts / context->compile_seconds,
        asts / context->module_load_seconds,
        row_evals,
        row_evals / run_seconds,
        row_evals / pipeline_seconds,
        secant_bench_ast_corpus_name(options->ast_mode),
        secant_bench_ast_corpus_definition_hash(options->ast_mode),
        (unsigned long long)secant_bench_ast_corpus_hash(
            context->programs,
            options->num_kernels * options->asts_per_kernel));
}
#endif

int
main(int argc, char** argv) {
    SecantRuntimeContext context;
    double run_seconds = 0.0;
    int arg_idx;
    int result = 1;

    memset(&context, 0, sizeof(context));
    secant_bench_options_default(&context.options);
#if SECANT_BENCH_END_TO_END
    context.options.shape = SECANT_BENCH_SHAPE_SSE;
    context.options.warmups = 0u;
    context.options.run_iterations = 1u;
    context.options.num_kernels = 8u;
    context.options.asts_per_kernel = 32u;
    context.options.tile_rows = 1024u;
#endif
    for (arg_idx = 1; arg_idx < argc; ++arg_idx) {
        if (strcmp(argv[arg_idx], "--help") == 0) {
            secant_runtime_usage(argv[0]);
            return 0;
        }
        if (!secant_bench_parse_common_option(argc, argv, &arg_idx, &context.options)) {
            secant_runtime_usage(argv[0]);
            return 1;
        }
    }
    if (!secant_bench_options_valid(&context.options) ||
        !secant_runtime_backend_available(context.options.backend) ||
        (context.options.backend == SECANT_BENCH_BACKEND_CUBIN &&
         context.options.opt_level != 1u)) {
        fprintf(stderr, "invalid or unavailable benchmark configuration\n");
        return 1;
    }
    if (!secant_runtime_prepare(&context)) {
        fprintf(stderr, "runtime setup failed\n");
#if SECANT_BENCH_END_TO_END
    } else if (!secant_runtime_time(&context, &run_seconds)) {
        fprintf(stderr, "runtime timing failed\n");
    } else if (!secant_runtime_verify(&context)) {
        fprintf(stderr, "correctness verification failed\n");
    } else {
        printf(
            "verify backend=%s rows=%zu kernels=%zu status=pass\n",
            secant_bench_backend_name(context.options.backend),
            context.options.check_rows,
            context.options.num_kernels);
        secant_runtime_print_end_to_end(&context, run_seconds);
        result = 0;
    }
#else
    } else if (!secant_runtime_verify(&context)) {
        fprintf(stderr, "correctness verification failed\n");
    } else {
        printf(
            "verify backend=%s rows=%zu kernels=%zu status=pass\n",
            secant_bench_backend_name(context.options.backend),
            context.options.check_rows,
            context.options.num_kernels);
        if (secant_runtime_time(&context, &run_seconds)) {
            result = 0;
        }
    }
#endif
    secant_runtime_cleanup(&context);
    return result;
}
