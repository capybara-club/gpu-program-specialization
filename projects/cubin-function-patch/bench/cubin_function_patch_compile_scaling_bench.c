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
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Charles Durham
 * SPDX-License-Identifier: MIT
 */

#define CUBIN_FUNCTION_PATCH_IMPLEMENTATION
#include "cubin_function_patch.h"

#include <cuda.h>
#include <math.h>
#include <nvJitLink.h>
#include <nvPTXCompiler.h>
#include <nvrtc.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

enum {
    BENCH_REPLACEMENT_OPS = 12,
    BENCH_RESERVED_OPS = 96,
    BENCH_KERNEL_CALLS = 32,
    BENCH_SOURCE_CAPACITY = 65536
};

typedef struct {
    void* data;
    size_t bytes;
} BenchBlob;

typedef enum {
    BENCH_MODE_FULL_COMPILE = 0,
    BENCH_MODE_FULL_PTX_INJECT = 1,
    BENCH_MODE_PARTIAL_NVJITLINK = 2,
    BENCH_MODE_PARTIAL_PATCH = 3,
    BENCH_MODE_NVJITLINK_ONLY = 4,
    BENCH_MODE_PATCH_ONLY = 5
} BenchMode;

typedef struct {
    double nvrtc_ms;
    double nvptx_ms;
    double inject_ms;
    double nvjitlink_ms;
    double patch_ms;
    size_t modules;
} BenchStats;

typedef struct {
    unsigned int sm_major;
    unsigned int sm_minor;
    BenchBlob caller_rdc;
    BenchBlob reserved_rdc;
    BenchBlob template_cubin;
    BenchBlob ptx_inject_template_ptx;
    size_t ptx_inject_func_begin;
    size_t ptx_inject_func_end;
    BenchBlob* prepared_eval_rdcs;
    size_t num_prepared_eval_rdcs;
    CubinFunctionPatchHandle* patch_handle;
    void* patch_handle_memory;
    size_t patch_output_bytes;
} BenchContext;

typedef struct {
    const BenchContext* context;
    BenchMode mode;
    size_t begin_module;
    size_t end_module;
    BenchStats stats;
    int ok;
} BenchWorker;

typedef struct {
    double wall_ms;
    BenchStats stats;
    int ok;
} BenchRunResult;

typedef struct {
    size_t modules;
    unsigned int repeats;
    unsigned int warmup;
    unsigned int workers[32];
    size_t num_worker_counts;
    unsigned int sm_major;
    unsigned int sm_minor;
    int sm_forced;
    int skip_correctness;
    int skip_final_sanity;
} BenchConfig;

static void
bench_free_blob(
    BenchBlob* blob
) {
    if (blob != NULL) {
        free(blob->data);
        blob->data = NULL;
        blob->bytes = 0u;
    }
}

static double
bench_now_ms(
    void
) {
    struct timespec ts;
    (void)clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec * 1.0e-6;
}

static int
bench_appendf(
    char* dst,
    size_t capacity,
    size_t* length,
    const char* fmt,
    ...
) {
    va_list args;
    int written;
    size_t remain;
    if (dst == NULL || length == NULL || fmt == NULL || *length >= capacity) return 0;
    remain = capacity - *length;
    va_start(args, fmt);
    written = vsnprintf(dst + *length, remain, fmt, args);
    va_end(args);
    if (written < 0 || (size_t)written >= remain) return 0;
    *length += (size_t)written;
    return 1;
}

static float
bench_op_scale(
    size_t variant,
    size_t op
) {
    unsigned int bits = (unsigned int)((variant * 1103515245u + op * 12345u + 17u) & 1023u);
    return 0.99925f + (float)bits * 0.000001f;
}

static float
bench_op_bias(
    size_t variant,
    size_t op
) {
    unsigned int bits = (unsigned int)((variant * 1664525u + op * 1013904223u + 13u) & 255u);
    return ((float)bits - 127.0f) * 0.0007f;
}

static float
bench_eval_cpu(
    size_t variant,
    float x
) {
    size_t i;
    float y = x;
    for (i = 0u; i < BENCH_REPLACEMENT_OPS; ++i) {
        y = y * bench_op_scale(variant, i) + bench_op_bias(variant, i);
    }
    return y;
}

static uint32_t
bench_f32_bits(
    float value
) {
    union {
        float f;
        uint32_t u;
    } bits;
    bits.f = value;
    return bits.u;
}

static float
bench_kernel_cpu(
    size_t variant,
    float x
) {
    size_t i;
    float acc = 0.0f;
    for (i = 0u; i < BENCH_KERNEL_CALLS; ++i) {
        float shifted = x + ((float)i - 15.5f) * 0.003125f;
        float y = bench_eval_cpu(variant, shifted);
        acc += y * (1.0f + (float)i * 0.0009765625f) + shifted * 0.000244140625f;
    }
    return acc;
}

static const char*
bench_line_begin(
    const char* text,
    const char* at
) {
    const char* p = at;
    while (p > text && p[-1] != '\n') p--;
    return p;
}

static int
bench_find_ptx_function_span(
    const char* ptx,
    const char* symbol,
    size_t* begin_out,
    size_t* end_out
) {
    const char* name;
    const char* begin;
    const char* body;
    const char* p;
    int depth = 0;
    if (ptx == NULL || symbol == NULL || begin_out == NULL || end_out == NULL) return 0;
    name = strstr(ptx, symbol);
    while (name != NULL) {
        const char* line = bench_line_begin(ptx, name);
        if (strstr(line, ".func") != NULL) break;
        name = strstr(name + 1, symbol);
    }
    if (name == NULL) return 0;
    begin = bench_line_begin(ptx, name);
    while (begin > ptx) {
        const char* prev = bench_line_begin(ptx, begin - 1);
        if (strstr(prev, ".visible") != NULL || strstr(prev, ".func") != NULL) {
            begin = prev;
        }
        break;
    }
    body = strchr(name, '{');
    if (body == NULL) return 0;
    for (p = body; *p != '\0'; ++p) {
        if (*p == '{') depth++;
        if (*p == '}') {
            depth--;
            if (depth == 0) {
                p++;
                if (*p == '\r') p++;
                if (*p == '\n') p++;
                *begin_out = (size_t)(begin - ptx);
                *end_out = (size_t)(p - ptx);
                return 1;
            }
        }
    }
    return 0;
}

static int
bench_make_eval_ptx_function(
    char* source,
    size_t source_capacity,
    size_t variant
) {
    size_t len = 0u;
    size_t i;
    if (!bench_appendf(source, source_capacity, &len,
        ".visible .func (.param .b32 func_retval0) bench_eval(\n"
        "    .param .b32 bench_eval_param_0\n"
        ")\n"
        "{\n"
        "    .reg .f32 %%f<2>;\n"
        "    ld.param.f32 %%f1, [bench_eval_param_0];\n")) return 0;
    for (i = 0u; i < BENCH_REPLACEMENT_OPS; ++i) {
        if (!bench_appendf(source, source_capacity, &len,
            "    mul.rn.ftz.f32 %%f1, %%f1, 0f%08x;\n"
            "    add.rn.ftz.f32 %%f1, %%f1, 0f%08x;\n",
            (unsigned int)bench_f32_bits(bench_op_scale(variant, i)),
            (unsigned int)bench_f32_bits(bench_op_bias(variant, i)))) return 0;
    }
    return bench_appendf(source, source_capacity, &len,
        "    st.param.f32 [func_retval0+0], %%f1;\n"
        "    ret;\n"
        "}\n");
}

static int
bench_render_injected_ptx(
    const BenchContext* context,
    size_t variant,
    BenchBlob* ptx_out
) {
    char function_ptx[BENCH_SOURCE_CAPACITY];
    size_t function_bytes;
    size_t template_bytes;
    size_t prefix_bytes;
    size_t suffix_bytes;
    size_t output_bytes;
    char* output;

    if (context == NULL || ptx_out == NULL || context->ptx_inject_template_ptx.data == NULL) return 0;
    ptx_out->data = NULL;
    ptx_out->bytes = 0u;
    if (!bench_make_eval_ptx_function(function_ptx, sizeof(function_ptx), variant)) return 0;
    function_bytes = strlen(function_ptx);
    template_bytes = strlen((const char*)context->ptx_inject_template_ptx.data);
    if (context->ptx_inject_func_begin > context->ptx_inject_func_end ||
        context->ptx_inject_func_end > template_bytes) return 0;
    prefix_bytes = context->ptx_inject_func_begin;
    suffix_bytes = template_bytes - context->ptx_inject_func_end;
    output_bytes = prefix_bytes + function_bytes + suffix_bytes;
    output = (char*)malloc(output_bytes + 1u);
    if (output == NULL) return 0;
    memcpy(output, context->ptx_inject_template_ptx.data, prefix_bytes);
    memcpy(output + prefix_bytes, function_ptx, function_bytes);
    memcpy(output + prefix_bytes + function_bytes,
        (const char*)context->ptx_inject_template_ptx.data + context->ptx_inject_func_end,
        suffix_bytes);
    output[output_bytes] = '\0';
    ptx_out->data = output;
    ptx_out->bytes = output_bytes + 1u;
    return 1;
}

static int
bench_append_kernel_source(
    char* source,
    size_t source_capacity,
    size_t* len
) {
    size_t i;
    if (!bench_appendf(source, source_capacity, len,
        "extern \"C\" __global__\n"
        "void\n"
        "bench_kernel(const float* input, float* output, unsigned long long n) {\n"
        "    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n"
        "    if (idx < n) {\n"
        "        float x = input[idx];\n"
        "        float acc = 0.0f;\n")) return 0;
    for (i = 0u; i < BENCH_KERNEL_CALLS; ++i) {
        double shift = ((double)i - 15.5) * 0.003125;
        double scale = 1.0 + (double)i * 0.0009765625;
        if (!bench_appendf(source, source_capacity, len,
            "        { float z = x + %.17g; float y = bench_eval(z); acc += y * %.17g + z * 0.000244140625f; }\n",
            shift,
            scale)) return 0;
    }
    return bench_appendf(source, source_capacity, len,
        "        output[idx] = acc;\n"
        "    }\n"
        "}\n");
}

static int
bench_make_eval_source(
    char* source,
    size_t source_capacity,
    size_t variant,
    int reserved,
    int include_kernel
) {
    size_t len = 0u;
    size_t i;
    size_t ops = reserved ? BENCH_RESERVED_OPS : BENCH_REPLACEMENT_OPS;
    if (!bench_appendf(source, source_capacity, &len,
        "extern \"C\" __device__ __noinline__\n"
        "float\n"
        "bench_eval(float x) {\n"
        "    float y = x;\n")) return 0;
    for (i = 0u; i < ops; ++i) {
        float scale = reserved ? bench_op_scale(0u, i) : bench_op_scale(variant, i);
        float bias = reserved ? bench_op_bias(0u, i) : bench_op_bias(variant, i);
        if (!bench_appendf(source, source_capacity, &len,
            "    y = y * %.9ef + %.9ef;\n",
            (double)scale,
            (double)bias)) return 0;
    }
    if (!bench_appendf(source, source_capacity, &len, "    return y;\n}\n")) return 0;
    if (include_kernel && !bench_append_kernel_source(source, source_capacity, &len)) return 0;
    return 1;
}

static int
bench_make_caller_source(
    char* source,
    size_t source_capacity
) {
    size_t len = 0u;
    return bench_appendf(source, source_capacity, &len,
        "extern \"C\" __device__ __noinline__ float bench_eval(float x);\n") &&
        bench_append_kernel_source(source, source_capacity, &len);
}

static int
bench_check_cuda(
    CUresult result,
    const char* expr
) {
    const char* name = NULL;
    const char* text = NULL;
    if (result == CUDA_SUCCESS) return 1;
    (void)cuGetErrorName(result, &name);
    (void)cuGetErrorString(result, &text);
    fprintf(stderr, "%s failed: %s: %s\n", expr, name != NULL ? name : "CUDA_ERROR", text != NULL ? text : "");
    return 0;
}

#define BENCH_CHECK_CUDA(EXPR) bench_check_cuda((EXPR), #EXPR)

static void
bench_print_nvrtc_log(
    nvrtcProgram program
) {
    size_t log_size = 0u;
    char* log = NULL;
    if (nvrtcGetProgramLogSize(program, &log_size) != NVRTC_SUCCESS || log_size <= 1u) return;
    log = (char*)malloc(log_size);
    if (log == NULL) return;
    if (nvrtcGetProgramLog(program, log) == NVRTC_SUCCESS && log[0] != '\0') fprintf(stderr, "%s\n", log);
    free(log);
}

static void
bench_print_nvptx_log(
    nvPTXCompilerHandle compiler,
    int error_log
) {
    size_t log_size = 0u;
    char* log = NULL;
    if (error_log) {
        if (nvPTXCompilerGetErrorLogSize(compiler, &log_size) != NVPTXCOMPILE_SUCCESS) return;
    } else {
        if (nvPTXCompilerGetInfoLogSize(compiler, &log_size) != NVPTXCOMPILE_SUCCESS) return;
    }
    if (log_size <= 1u) return;
    log = (char*)malloc(log_size);
    if (log == NULL) return;
    if (error_log) {
        (void)nvPTXCompilerGetErrorLog(compiler, log);
    } else {
        (void)nvPTXCompilerGetInfoLog(compiler, log);
    }
    if (log[0] != '\0') fprintf(stderr, "%s\n", log);
    free(log);
}

static int
bench_compile_cuda_to_ptx(
    const char* source,
    const char* name,
    unsigned int sm_major,
    unsigned int sm_minor,
    int rdc,
    BenchBlob* ptx_out
) {
    nvrtcProgram program = NULL;
    nvrtcResult result;
    const char* options[5];
    char arch_option[64];
    size_t num_options = 0u;
    size_t ptx_size = 0u;
    char* ptx = NULL;

    if (source == NULL || name == NULL || ptx_out == NULL) return 0;
    ptx_out->data = NULL;
    ptx_out->bytes = 0u;
    snprintf(arch_option, sizeof(arch_option), "--gpu-architecture=compute_%u%u", sm_major, sm_minor);
    options[num_options++] = arch_option;
    options[num_options++] = "--std=c++17";
    options[num_options++] = "--use_fast_math";
    if (rdc) {
        options[num_options++] = "--relocatable-device-code=true";
        options[num_options++] = "--device-c";
    }

    result = nvrtcCreateProgram(&program, source, name, 0, NULL, NULL);
    if (result != NVRTC_SUCCESS) return 0;
    result = nvrtcCompileProgram(program, (int)num_options, options);
    if (result != NVRTC_SUCCESS) {
        bench_print_nvrtc_log(program);
        (void)nvrtcDestroyProgram(&program);
        return 0;
    }
    if (nvrtcGetPTXSize(program, &ptx_size) != NVRTC_SUCCESS || ptx_size == 0u) {
        (void)nvrtcDestroyProgram(&program);
        return 0;
    }
    ptx = (char*)malloc(ptx_size + 1u);
    if (ptx == NULL) {
        (void)nvrtcDestroyProgram(&program);
        return 0;
    }
    if (nvrtcGetPTX(program, ptx) != NVRTC_SUCCESS) {
        free(ptx);
        (void)nvrtcDestroyProgram(&program);
        return 0;
    }
    (void)nvrtcDestroyProgram(&program);
    ptx[ptx_size] = '\0';
    ptx_out->data = ptx;
    ptx_out->bytes = ptx_size;
    return 1;
}

static int
bench_compile_ptx(
    const char* ptx,
    size_t ptx_bytes,
    unsigned int sm_major,
    unsigned int sm_minor,
    int compile_only,
    BenchBlob* image_out
) {
    nvPTXCompilerHandle compiler = NULL;
    nvPTXCompileResult result;
    const char* options[4];
    char arch_option[64];
    size_t num_options = 0u;
    size_t image_size = 0u;
    void* image = NULL;

    if (ptx == NULL || ptx_bytes == 0u || image_out == NULL) return 0;
    image_out->data = NULL;
    image_out->bytes = 0u;
    snprintf(arch_option, sizeof(arch_option), "--gpu-name=sm_%u%u", sm_major, sm_minor);
    options[num_options++] = arch_option;
    if (compile_only) options[num_options++] = "--compile-only";
    options[num_options++] = "--opt-level=0";
    options[num_options++] = "--allow-expensive-optimizations=false";

    result = nvPTXCompilerCreate(&compiler, ptx_bytes, ptx);
    if (result != NVPTXCOMPILE_SUCCESS) return 0;
    result = nvPTXCompilerCompile(compiler, (int)num_options, options);
    if (result != NVPTXCOMPILE_SUCCESS) {
        bench_print_nvptx_log(compiler, 1);
        (void)nvPTXCompilerDestroy(&compiler);
        return 0;
    }
    if (nvPTXCompilerGetCompiledProgramSize(compiler, &image_size) != NVPTXCOMPILE_SUCCESS || image_size == 0u) {
        (void)nvPTXCompilerDestroy(&compiler);
        return 0;
    }
    image = malloc(image_size);
    if (image == NULL) {
        (void)nvPTXCompilerDestroy(&compiler);
        return 0;
    }
    if (nvPTXCompilerGetCompiledProgram(compiler, image) != NVPTXCOMPILE_SUCCESS) {
        free(image);
        (void)nvPTXCompilerDestroy(&compiler);
        return 0;
    }
    (void)nvPTXCompilerDestroy(&compiler);
    image_out->data = image;
    image_out->bytes = image_size;
    return 1;
}

static int
bench_compile_source_to_rdc(
    const char* source,
    const char* name,
    unsigned int sm_major,
    unsigned int sm_minor,
    BenchBlob* object_out,
    BenchStats* stats
) {
    BenchBlob ptx = {0};
    double t0;
    double t1;
    int ok = 0;
    t0 = bench_now_ms();
    if (!bench_compile_cuda_to_ptx(source, name, sm_major, sm_minor, 1, &ptx)) return 0;
    t1 = bench_now_ms();
    if (stats != NULL) stats->nvrtc_ms += t1 - t0;
    t0 = bench_now_ms();
    if (!bench_compile_ptx((const char*)ptx.data, ptx.bytes, sm_major, sm_minor, 1, object_out)) goto done;
    t1 = bench_now_ms();
    if (stats != NULL) stats->nvptx_ms += t1 - t0;
    ok = 1;

done:
    bench_free_blob(&ptx);
    return ok;
}

static int
bench_compile_source_to_final_cubin(
    const char* source,
    const char* name,
    unsigned int sm_major,
    unsigned int sm_minor,
    BenchBlob* cubin_out,
    BenchStats* stats
) {
    BenchBlob ptx = {0};
    double t0;
    double t1;
    int ok = 0;
    t0 = bench_now_ms();
    if (!bench_compile_cuda_to_ptx(source, name, sm_major, sm_minor, 0, &ptx)) return 0;
    t1 = bench_now_ms();
    if (stats != NULL) stats->nvrtc_ms += t1 - t0;
    t0 = bench_now_ms();
    if (!bench_compile_ptx((const char*)ptx.data, ptx.bytes, sm_major, sm_minor, 0, cubin_out)) goto done;
    t1 = bench_now_ms();
    if (stats != NULL) stats->nvptx_ms += t1 - t0;
    ok = 1;

done:
    bench_free_blob(&ptx);
    return ok;
}

static int
bench_link_objects(
    const BenchBlob* caller_object,
    const BenchBlob* site_object,
    unsigned int sm_major,
    unsigned int sm_minor,
    BenchBlob* cubin_out
) {
    nvJitLinkHandle link = NULL;
    nvJitLinkResult result;
    const char* options[3];
    char arch_option[64];
    size_t cubin_size = 0u;
    void* cubin = NULL;

    if (caller_object == NULL || site_object == NULL || cubin_out == NULL ||
        caller_object->data == NULL || site_object->data == NULL) return 0;
    cubin_out->data = NULL;
    cubin_out->bytes = 0u;
    snprintf(arch_option, sizeof(arch_option), "-arch=sm_%u%u", sm_major, sm_minor);
    options[0] = arch_option;
    options[1] = "-O0";
    options[2] = "-no-cache";

    result = nvJitLinkCreate(&link, 3, options);
    if (result == NVJITLINK_SUCCESS) {
        result = nvJitLinkAddData(link, NVJITLINK_INPUT_CUBIN, caller_object->data, caller_object->bytes, "caller_object");
    }
    if (result == NVJITLINK_SUCCESS) {
        result = nvJitLinkAddData(link, NVJITLINK_INPUT_CUBIN, site_object->data, site_object->bytes, "site_object");
    }
    if (result == NVJITLINK_SUCCESS) result = nvJitLinkComplete(link);
    if (result == NVJITLINK_SUCCESS) result = nvJitLinkGetLinkedCubinSize(link, &cubin_size);
    if (result == NVJITLINK_SUCCESS && cubin_size != 0u) {
        cubin = malloc(cubin_size);
        if (cubin == NULL) result = NVJITLINK_ERROR_INTERNAL;
    }
    if (result == NVJITLINK_SUCCESS) result = nvJitLinkGetLinkedCubin(link, cubin);
    if (link != NULL) (void)nvJitLinkDestroy(&link);
    if (result != NVJITLINK_SUCCESS || cubin == NULL || cubin_size == 0u) {
        free(cubin);
        fprintf(stderr, "nvJitLink failed: %d\n", (int)result);
        return 0;
    }
    cubin_out->data = cubin;
    cubin_out->bytes = cubin_size;
    return 1;
}

static int
bench_compile_full_module(
    const BenchContext* context,
    size_t variant,
    BenchBlob* cubin_out,
    BenchStats* stats
) {
    char source[BENCH_SOURCE_CAPACITY];
    if (!bench_make_eval_source(source, sizeof(source), variant, 0, 1)) return 0;
    return bench_compile_source_to_final_cubin(
        source,
        "bench_full.cu",
        context->sm_major,
        context->sm_minor,
        cubin_out,
        stats);
}

static int
bench_compile_ptx_inject_module(
    const BenchContext* context,
    size_t variant,
    BenchBlob* cubin_out,
    BenchStats* stats
) {
    BenchBlob ptx = {0};
    double t0;
    double t1;
    int ok = 0;
    t0 = bench_now_ms();
    if (!bench_render_injected_ptx(context, variant, &ptx)) return 0;
    t1 = bench_now_ms();
    if (stats != NULL) stats->inject_ms += t1 - t0;
    t0 = bench_now_ms();
    if (!bench_compile_ptx((const char*)ptx.data, ptx.bytes, context->sm_major, context->sm_minor, 0, cubin_out)) goto done;
    t1 = bench_now_ms();
    if (stats != NULL) stats->nvptx_ms += t1 - t0;
    ok = 1;

done:
    bench_free_blob(&ptx);
    return ok;
}

static int
bench_compile_eval_rdc(
    const BenchContext* context,
    size_t variant,
    BenchBlob* rdc_out,
    BenchStats* stats
) {
    char source[BENCH_SOURCE_CAPACITY];
    if (!bench_make_eval_source(source, sizeof(source), variant, 0, 0)) return 0;
    return bench_compile_source_to_rdc(
        source,
        "bench_eval.cu",
        context->sm_major,
        context->sm_minor,
        rdc_out,
        stats);
}

static void*
bench_worker_main(
    void* user
) {
    BenchWorker* worker = (BenchWorker*)user;
    size_t i;
    worker->ok = 1;
    for (i = worker->begin_module; i < worker->end_module; ++i) {
        BenchBlob cubin = {0};
        BenchBlob eval_rdc = {0};
        double t0;
        double t1;

        if (worker->mode == BENCH_MODE_FULL_COMPILE) {
            if (!bench_compile_full_module(worker->context, i, &cubin, &worker->stats)) {
                worker->ok = 0;
                break;
            }
        } else if (worker->mode == BENCH_MODE_FULL_PTX_INJECT) {
            if (!bench_compile_ptx_inject_module(worker->context, i, &cubin, &worker->stats)) {
                worker->ok = 0;
                break;
            }
        } else {
            const BenchBlob* eval_source = &eval_rdc;
            if (worker->mode == BENCH_MODE_NVJITLINK_ONLY || worker->mode == BENCH_MODE_PATCH_ONLY) {
                if (i >= worker->context->num_prepared_eval_rdcs) {
                    worker->ok = 0;
                    break;
                }
                eval_source = &worker->context->prepared_eval_rdcs[i];
            } else {
                if (!bench_compile_eval_rdc(worker->context, i, &eval_rdc, &worker->stats)) {
                    worker->ok = 0;
                    break;
                }
            }
            if (worker->mode == BENCH_MODE_PARTIAL_NVJITLINK || worker->mode == BENCH_MODE_NVJITLINK_ONLY) {
                t0 = bench_now_ms();
                if (!bench_link_objects(
                    &worker->context->caller_rdc,
                    eval_source,
                    worker->context->sm_major,
                    worker->context->sm_minor,
                    &cubin)) {
                    worker->ok = 0;
                    bench_free_blob(&eval_rdc);
                    break;
                }
                t1 = bench_now_ms();
                worker->stats.nvjitlink_ms += t1 - t0;
            } else {
                void* output = malloc(worker->context->patch_output_bytes);
                size_t output_bytes = 0u;
                CubinFunctionPatchReport report;
                CubinFunctionPatchResult patch_result;
                if (output == NULL) {
                    worker->ok = 0;
                    bench_free_blob(&eval_rdc);
                    break;
                }
                t0 = bench_now_ms();
                patch_result = cubin_function_patch_apply_one_ex(
                    worker->context->patch_handle,
                    "bench_eval",
                    eval_source->data,
                    eval_source->bytes,
                    CUBIN_FUNCTION_PATCH_TAIL_LEAVE_UNCHANGED,
                    output,
                    worker->context->patch_output_bytes,
                    &output_bytes,
                    &report);
                t1 = bench_now_ms();
                worker->stats.patch_ms += t1 - t0;
                if (patch_result != CUBIN_FUNCTION_PATCH_SUCCESS || output_bytes == 0u) {
                    fprintf(stderr, "patch failed: %s\n", cubin_function_patch_result_to_string(patch_result));
                    free(output);
                    worker->ok = 0;
                    bench_free_blob(&eval_rdc);
                    break;
                }
                cubin.data = output;
                cubin.bytes = output_bytes;
            }
        }
        worker->stats.modules++;
        bench_free_blob(&cubin);
        bench_free_blob(&eval_rdc);
    }
    return NULL;
}

static int
bench_run_parallel(
    const BenchContext* context,
    BenchMode mode,
    size_t modules,
    unsigned int worker_count,
    BenchRunResult* result_out
) {
    pthread_t* threads = NULL;
    BenchWorker* workers = NULL;
    size_t i;
    double t0;
    double t1;
    int ok = 1;

    if (context == NULL || modules == 0u || worker_count == 0u || result_out == NULL) return 0;
    memset(result_out, 0, sizeof(*result_out));
    threads = (pthread_t*)calloc(worker_count, sizeof(*threads));
    workers = (BenchWorker*)calloc(worker_count, sizeof(*workers));
    if (threads == NULL || workers == NULL) {
        free(threads);
        free(workers);
        return 0;
    }

    t0 = bench_now_ms();
    for (i = 0u; i < worker_count; ++i) {
        size_t begin = (modules * i) / worker_count;
        size_t end = (modules * (i + 1u)) / worker_count;
        workers[i].context = context;
        workers[i].mode = mode;
        workers[i].begin_module = begin;
        workers[i].end_module = end;
        workers[i].ok = 1;
        if (pthread_create(&threads[i], NULL, bench_worker_main, &workers[i]) != 0) {
            ok = 0;
            worker_count = (unsigned int)i;
            break;
        }
    }
    for (i = 0u; i < worker_count; ++i) {
        if (pthread_join(threads[i], NULL) != 0) ok = 0;
    }
    t1 = bench_now_ms();

    result_out->wall_ms = t1 - t0;
    result_out->ok = ok;
    for (i = 0u; i < worker_count; ++i) {
        if (!workers[i].ok) result_out->ok = 0;
        result_out->stats.nvrtc_ms += workers[i].stats.nvrtc_ms;
        result_out->stats.nvptx_ms += workers[i].stats.nvptx_ms;
        result_out->stats.inject_ms += workers[i].stats.inject_ms;
        result_out->stats.nvjitlink_ms += workers[i].stats.nvjitlink_ms;
        result_out->stats.patch_ms += workers[i].stats.patch_ms;
        result_out->stats.modules += workers[i].stats.modules;
    }
    free(workers);
    free(threads);
    return result_out->ok;
}

static int
bench_setup_context(
    BenchContext* context
) {
    char source[BENCH_SOURCE_CAPACITY];
    const char* symbols[1] = {"bench_eval"};
    size_t handle_bytes = 0u;
    CubinFunctionPatchResult patch_result;
    BenchStats ignored = {0};
    int ok = 0;

    if (context == NULL) return 0;
    if (!bench_make_caller_source(source, sizeof(source))) return 0;
    if (!bench_compile_source_to_rdc(source, "bench_caller.cu", context->sm_major, context->sm_minor, &context->caller_rdc, &ignored)) return 0;
    if (!bench_make_eval_source(source, sizeof(source), 0u, 1, 1)) return 0;
    if (!bench_compile_cuda_to_ptx(source, "bench_ptx_inject_template.cu", context->sm_major, context->sm_minor, 0, &context->ptx_inject_template_ptx)) return 0;
    if (!bench_find_ptx_function_span(
        (const char*)context->ptx_inject_template_ptx.data,
        "bench_eval",
        &context->ptx_inject_func_begin,
        &context->ptx_inject_func_end)) {
        fprintf(stderr, "failed to locate bench_eval in template PTX\n");
        return 0;
    }
    if (!bench_make_eval_source(source, sizeof(source), 0u, 1, 0)) return 0;
    if (!bench_compile_source_to_rdc(source, "bench_reserved.cu", context->sm_major, context->sm_minor, &context->reserved_rdc, &ignored)) return 0;
    if (!bench_link_objects(&context->caller_rdc, &context->reserved_rdc, context->sm_major, context->sm_minor, &context->template_cubin)) return 0;

    patch_result = cubin_function_patch_handle_size(symbols, 1u, &handle_bytes);
    if (patch_result != CUBIN_FUNCTION_PATCH_SUCCESS || handle_bytes == 0u) return 0;
    context->patch_handle_memory = malloc(handle_bytes);
    if (context->patch_handle_memory == NULL) return 0;
    patch_result = cubin_function_patch_create(
        context->template_cubin.data,
        context->template_cubin.bytes,
        symbols,
        1u,
        context->patch_handle_memory,
        handle_bytes,
        &context->patch_handle);
    if (patch_result != CUBIN_FUNCTION_PATCH_SUCCESS) {
        fprintf(stderr, "patch handle create failed: %s\n", cubin_function_patch_result_to_string(patch_result));
        return 0;
    }
    patch_result = cubin_function_patch_output_size(context->patch_handle, &context->patch_output_bytes);
    if (patch_result != CUBIN_FUNCTION_PATCH_SUCCESS || context->patch_output_bytes == 0u) return 0;
    ok = 1;
    return ok;
}

static int
bench_prepare_eval_rdcs(
    BenchContext* context,
    size_t modules
) {
    size_t i;
    double t0;
    double t1;
    BenchStats stats = {0};
    if (context == NULL || modules == 0u) return 0;
    context->prepared_eval_rdcs = (BenchBlob*)calloc(modules, sizeof(context->prepared_eval_rdcs[0]));
    if (context->prepared_eval_rdcs == NULL) return 0;
    context->num_prepared_eval_rdcs = modules;
    t0 = bench_now_ms();
    for (i = 0u; i < modules; ++i) {
        if (!bench_compile_eval_rdc(context, i, &context->prepared_eval_rdcs[i], &stats)) return 0;
    }
    t1 = bench_now_ms();
    printf("prepared_eval_rdcs: modules=%zu wall_ms=%.3f nvrtc_sum_ms=%.3f nvptx_sum_ms=%.3f\n",
        modules,
        t1 - t0,
        stats.nvrtc_ms,
        stats.nvptx_ms);
    return 1;
}

static void
bench_destroy_context(
    BenchContext* context
) {
    if (context != NULL) {
        size_t i;
        if (context->patch_handle != NULL) cubin_function_patch_destroy(context->patch_handle);
        free(context->patch_handle_memory);
        for (i = 0u; i < context->num_prepared_eval_rdcs; ++i) {
            bench_free_blob(&context->prepared_eval_rdcs[i]);
        }
        free(context->prepared_eval_rdcs);
        bench_free_blob(&context->ptx_inject_template_ptx);
        bench_free_blob(&context->template_cubin);
        bench_free_blob(&context->reserved_rdc);
        bench_free_blob(&context->caller_rdc);
        memset(context, 0, sizeof(*context));
    }
}

static int
bench_run_kernel(
    const void* cubin,
    const float* input,
    float* output,
    size_t count
) {
    CUmodule module = NULL;
    CUfunction kernel = NULL;
    CUdeviceptr d_input = 0;
    CUdeviceptr d_output = 0;
    unsigned long long n_arg = (unsigned long long)count;
    unsigned int block = 128u;
    unsigned int grid = (unsigned int)((count + block - 1u) / block);
    void* args[3];
    int ok = 0;

    if (cubin == NULL || input == NULL || output == NULL || count == 0u) return 0;
    if (!BENCH_CHECK_CUDA(cuModuleLoadData(&module, cubin))) goto done;
    if (!BENCH_CHECK_CUDA(cuModuleGetFunction(&kernel, module, "bench_kernel"))) goto done;
    if (!BENCH_CHECK_CUDA(cuMemAlloc(&d_input, count * sizeof(float)))) goto done;
    if (!BENCH_CHECK_CUDA(cuMemAlloc(&d_output, count * sizeof(float)))) goto done;
    if (!BENCH_CHECK_CUDA(cuMemcpyHtoD(d_input, input, count * sizeof(float)))) goto done;
    args[0] = &d_input;
    args[1] = &d_output;
    args[2] = &n_arg;
    if (!BENCH_CHECK_CUDA(cuLaunchKernel(kernel, grid, 1u, 1u, block, 1u, 1u, 0u, NULL, args, NULL))) goto done;
    if (!BENCH_CHECK_CUDA(cuCtxSynchronize())) goto done;
    if (!BENCH_CHECK_CUDA(cuMemcpyDtoH(output, d_output, count * sizeof(float)))) goto done;
    ok = 1;

done:
    if (d_output != 0) (void)cuMemFree(d_output);
    if (d_input != 0) (void)cuMemFree(d_input);
    if (module != NULL) (void)cuModuleUnload(module);
    return ok;
}

static int
bench_check_output(
    const char* label,
    const float* output,
    size_t variant,
    const float* input,
    size_t count
) {
    size_t i;
    float max_abs = 0.0f;
    for (i = 0u; i < count; ++i) {
        float expected = bench_kernel_cpu(variant, input[i]);
        float diff = fabsf(output[i] - expected);
        if (diff > max_abs) max_abs = diff;
    }
    if (max_abs > 1.0e-3f) {
        fprintf(stderr, "%s max_abs=%g\n", label, (double)max_abs);
        return 0;
    }
    return 1;
}

static const char*
bench_mode_name(
    BenchMode mode
);

static int
bench_correctness(
    const BenchContext* context
) {
    enum { COUNT = 257 };
    float input[COUNT];
    float full_output[COUNT];
    float jitlink_output[COUNT];
    float patch_output[COUNT];
    BenchBlob full_cubin = {0};
    BenchBlob ptx_inject_cubin = {0};
    BenchBlob eval_rdc = {0};
    BenchBlob jitlink_cubin = {0};
    void* patch_cubin = NULL;
    size_t patch_cubin_bytes = 0u;
    CubinFunctionPatchReport report;
    CubinFunctionPatchResult patch_result;
    BenchStats stats = {0};
    size_t i;
    size_t variant = 7u;
    int ok = 0;

    for (i = 0u; i < COUNT; ++i) input[i] = ((float)i - 120.0f) * 0.013f;
    if (!bench_compile_full_module(context, variant, &full_cubin, &stats)) goto done;
    if (!bench_compile_ptx_inject_module(context, variant, &ptx_inject_cubin, &stats)) goto done;
    if (!bench_compile_eval_rdc(context, variant, &eval_rdc, &stats)) goto done;
    if (!bench_link_objects(&context->caller_rdc, &eval_rdc, context->sm_major, context->sm_minor, &jitlink_cubin)) goto done;
    patch_cubin = malloc(context->patch_output_bytes);
    if (patch_cubin == NULL) goto done;
    patch_result = cubin_function_patch_apply_one_ex(
        context->patch_handle,
        "bench_eval",
        eval_rdc.data,
        eval_rdc.bytes,
        CUBIN_FUNCTION_PATCH_TAIL_LEAVE_UNCHANGED,
        patch_cubin,
        context->patch_output_bytes,
        &patch_cubin_bytes,
        &report);
    if (patch_result != CUBIN_FUNCTION_PATCH_SUCCESS) {
        fprintf(stderr, "correctness patch failed: %s\n", cubin_function_patch_result_to_string(patch_result));
        goto done;
    }
    if (!bench_run_kernel(full_cubin.data, input, full_output, COUNT)) goto done;
    if (!bench_run_kernel(ptx_inject_cubin.data, input, jitlink_output, COUNT)) goto done;
    if (!bench_check_output("full_ptx_inject", jitlink_output, variant, input, COUNT)) goto done;
    if (!bench_run_kernel(jitlink_cubin.data, input, jitlink_output, COUNT)) goto done;
    if (!bench_run_kernel(patch_cubin, input, patch_output, COUNT)) goto done;
    if (!bench_check_output("full_compile", full_output, variant, input, COUNT)) goto done;
    if (!bench_check_output("partial_nvjitlink", jitlink_output, variant, input, COUNT)) goto done;
    if (!bench_check_output("partial_patch", patch_output, variant, input, COUNT)) goto done;
    printf("correctness: ok reserved_bytes=%zu replacement_bytes=%zu output_cubin_bytes=%zu\n",
        report.reserved_size,
        report.replacement_size,
        patch_cubin_bytes);
    ok = 1;

done:
    free(patch_cubin);
    bench_free_blob(&jitlink_cubin);
    bench_free_blob(&eval_rdc);
    bench_free_blob(&ptx_inject_cubin);
    bench_free_blob(&full_cubin);
    return ok;
}

static int
bench_compile_mode_cubin(
    const BenchContext* context,
    BenchMode mode,
    size_t variant,
    BenchBlob* cubin_out
) {
    BenchBlob eval_rdc = {0};
    BenchStats stats = {0};
    CubinFunctionPatchResult patch_result;
    size_t output_bytes = 0u;
    CubinFunctionPatchReport report;
    int ok = 0;

    if (context == NULL || cubin_out == NULL) return 0;
    cubin_out->data = NULL;
    cubin_out->bytes = 0u;
    if (mode == BENCH_MODE_FULL_COMPILE) {
        return bench_compile_full_module(context, variant, cubin_out, &stats);
    }
    if (mode == BENCH_MODE_FULL_PTX_INJECT) {
        return bench_compile_ptx_inject_module(context, variant, cubin_out, &stats);
    }
    if (variant < context->num_prepared_eval_rdcs &&
        (mode == BENCH_MODE_NVJITLINK_ONLY || mode == BENCH_MODE_PATCH_ONLY)) {
        eval_rdc = context->prepared_eval_rdcs[variant];
    } else {
        if (!bench_compile_eval_rdc(context, variant, &eval_rdc, &stats)) return 0;
    }
    if (mode == BENCH_MODE_PARTIAL_NVJITLINK || mode == BENCH_MODE_NVJITLINK_ONLY) {
        ok = bench_link_objects(&context->caller_rdc, &eval_rdc, context->sm_major, context->sm_minor, cubin_out);
    } else if (mode == BENCH_MODE_PARTIAL_PATCH || mode == BENCH_MODE_PATCH_ONLY) {
        cubin_out->data = malloc(context->patch_output_bytes);
        if (cubin_out->data != NULL) {
            patch_result = cubin_function_patch_apply_one_ex(
                context->patch_handle,
                "bench_eval",
                eval_rdc.data,
                eval_rdc.bytes,
                CUBIN_FUNCTION_PATCH_TAIL_LEAVE_UNCHANGED,
                cubin_out->data,
                context->patch_output_bytes,
                &output_bytes,
                &report);
            if (patch_result == CUBIN_FUNCTION_PATCH_SUCCESS && output_bytes != 0u) {
                cubin_out->bytes = output_bytes;
                ok = 1;
            } else {
                fprintf(stderr, "sanity patch failed: %s\n", cubin_function_patch_result_to_string(patch_result));
            }
        }
    }
    if (!(variant < context->num_prepared_eval_rdcs &&
        (mode == BENCH_MODE_NVJITLINK_ONLY || mode == BENCH_MODE_PATCH_ONLY))) {
        bench_free_blob(&eval_rdc);
    }
    if (!ok) bench_free_blob(cubin_out);
    return ok;
}

static int
bench_final_sanity_check(
    const BenchContext* context,
    size_t modules
) {
    enum { COUNT = 257 };
    BenchMode modes[] = {
        BENCH_MODE_FULL_COMPILE,
        BENCH_MODE_FULL_PTX_INJECT,
        BENCH_MODE_PARTIAL_NVJITLINK,
        BENCH_MODE_PARTIAL_PATCH,
        BENCH_MODE_NVJITLINK_ONLY,
        BENCH_MODE_PATCH_ONLY
    };
    float input[COUNT];
    float output[COUNT];
    size_t variants[3];
    size_t num_variants = 0u;
    size_t mi;
    size_t vi;
    int ok = 1;

    if (context == NULL || modules == 0u) return 0;
    variants[num_variants++] = 0u;
    if (modules > 2u) variants[num_variants++] = modules / 2u;
    if (modules > 1u) variants[num_variants++] = modules - 1u;
    for (vi = 0u; vi < COUNT; ++vi) input[vi] = ((float)vi - 120.0f) * 0.013f;
    for (mi = 0u; mi < sizeof(modes) / sizeof(modes[0]); ++mi) {
        for (vi = 0u; vi < num_variants; ++vi) {
            BenchBlob cubin = {0};
            size_t variant = variants[vi];
            if (!bench_compile_mode_cubin(context, modes[mi], variant, &cubin)) {
                ok = 0;
                continue;
            }
            if (!bench_run_kernel(cubin.data, input, output, COUNT) ||
                !bench_check_output(bench_mode_name(modes[mi]), output, variant, input, COUNT)) {
                ok = 0;
            }
            bench_free_blob(&cubin);
        }
    }
    if (ok) printf("final_sanity: ok variants_checked=%zu modes_checked=%zu\n", num_variants, sizeof(modes) / sizeof(modes[0]));
    return ok;
}

static const char*
bench_mode_name(
    BenchMode mode
) {
    switch (mode) {
        case BENCH_MODE_FULL_COMPILE: return "full_compile";
        case BENCH_MODE_FULL_PTX_INJECT: return "full_ptx_inject";
        case BENCH_MODE_PARTIAL_NVJITLINK: return "partial_nvjitlink";
        case BENCH_MODE_PARTIAL_PATCH: return "partial_patch";
        case BENCH_MODE_NVJITLINK_ONLY: return "nvjitlink_only";
        case BENCH_MODE_PATCH_ONLY: return "patch_only";
        default: return "unknown";
    }
}

static int
bench_run_mode(
    const BenchContext* context,
    BenchMode mode,
    const BenchConfig* config
) {
    size_t worker_idx;
    for (worker_idx = 0u; worker_idx < config->num_worker_counts; ++worker_idx) {
        unsigned int workers = config->workers[worker_idx];
        unsigned int run;
        double sum = 0.0;
        double sum_sq = 0.0;
        BenchRunResult best;
        memset(&best, 0, sizeof(best));
        best.wall_ms = 1.0e100;
        for (run = 0u; run < config->warmup + config->repeats; ++run) {
            BenchRunResult result;
            if (!bench_run_parallel(context, mode, config->modules, workers, &result)) return 0;
            if (run < config->warmup) continue;
            sum += result.wall_ms;
            sum_sq += result.wall_ms * result.wall_ms;
            if (result.wall_ms < best.wall_ms) best = result;
        }
        {
            double repeat_count = (double)config->repeats;
            double mean = sum / repeat_count;
            double variance = sum_sq / repeat_count - mean * mean;
            double stddev = variance > 0.0 ? sqrt(variance) : 0.0;
            double modules_per_sec = best.wall_ms > 0.0 ? (double)config->modules * 1000.0 / best.wall_ms : 0.0;
            printf("%-20s %8u %12.3f %12.3f %12.3f %14.3f %14.3f %14.3f %14.3f %14.3f %14.3f\n",
                bench_mode_name(mode),
                workers,
                best.wall_ms,
                mean,
                stddev,
                modules_per_sec,
                best.stats.nvrtc_ms,
                best.stats.nvptx_ms,
                best.stats.inject_ms,
                best.stats.nvjitlink_ms,
                best.stats.patch_ms);
        }
    }
    return 1;
}

static int
bench_parse_worker_list(
    const char* text,
    BenchConfig* config
) {
    char* copy;
    char* token;
    char* save = NULL;
    if (text == NULL || config == NULL) return 0;
    copy = (char*)malloc(strlen(text) + 1u);
    if (copy == NULL) return 0;
    strcpy(copy, text);
    config->num_worker_counts = 0u;
    token = strtok_r(copy, ",", &save);
    while (token != NULL && config->num_worker_counts < 32u) {
        long value = strtol(token, NULL, 10);
        if (value <= 0) {
            free(copy);
            return 0;
        }
        config->workers[config->num_worker_counts++] = (unsigned int)value;
        token = strtok_r(NULL, ",", &save);
    }
    free(copy);
    return config->num_worker_counts != 0u;
}

static void
bench_default_workers(
    BenchConfig* config
) {
    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    unsigned int candidates[] = {1u, 2u, 4u, 8u, 12u, 16u, 24u, 32u, 64u};
    size_t i;
    config->num_worker_counts = 0u;
    if (ncpu <= 0) ncpu = 1;
    for (i = 0u; i < sizeof(candidates) / sizeof(candidates[0]); ++i) {
        if ((long)candidates[i] <= ncpu) config->workers[config->num_worker_counts++] = candidates[i];
    }
    if (config->num_worker_counts == 0u) {
        config->workers[0] = 1u;
        config->num_worker_counts = 1u;
    }
}

static int
bench_parse_sm(
    const char* text,
    unsigned int* major_out,
    unsigned int* minor_out
) {
    unsigned int value;
    if (text == NULL || major_out == NULL || minor_out == NULL) return 0;
    if (strncmp(text, "sm_", 3u) == 0) text += 3u;
    value = (unsigned int)strtoul(text, NULL, 10);
    if (value < 10u) return 0;
    *major_out = value / 10u;
    *minor_out = value % 10u;
    return 1;
}

static int
bench_detect_sm(
    unsigned int* major_out,
    unsigned int* minor_out
) {
    CUdevice device;
    int major = 0;
    int minor = 0;
    if (major_out == NULL || minor_out == NULL) return 0;
    if (!BENCH_CHECK_CUDA(cuInit(0))) return 0;
    if (!BENCH_CHECK_CUDA(cuDeviceGet(&device, 0))) return 0;
    if (!BENCH_CHECK_CUDA(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device))) return 0;
    if (!BENCH_CHECK_CUDA(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device))) return 0;
    *major_out = (unsigned int)major;
    *minor_out = (unsigned int)minor;
    return 1;
}

static void
bench_print_usage(
    const char* argv0
) {
    fprintf(stderr,
        "usage: %s [--modules N] [--workers 1,2,4,8] [--repeats N] [--warmup N] [--sm sm_120] [--skip-correctness] [--skip-final-sanity]\n",
        argv0);
}

static int
bench_parse_args(
    int argc,
    char** argv,
    BenchConfig* config
) {
    int i;
    if (config == NULL) return 0;
    memset(config, 0, sizeof(*config));
    config->modules = 240u;
    config->repeats = 1u;
    config->warmup = 0u;
    bench_default_workers(config);
    for (i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--modules") == 0 && i + 1 < argc) {
            config->modules = (size_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--workers") == 0 && i + 1 < argc) {
            if (!bench_parse_worker_list(argv[++i], config)) return 0;
        } else if (strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) {
            config->repeats = (unsigned int)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            config->warmup = (unsigned int)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--sm") == 0 && i + 1 < argc) {
            if (!bench_parse_sm(argv[++i], &config->sm_major, &config->sm_minor)) return 0;
            config->sm_forced = 1;
        } else if (strcmp(argv[i], "--skip-correctness") == 0) {
            config->skip_correctness = 1;
        } else if (strcmp(argv[i], "--skip-final-sanity") == 0) {
            config->skip_final_sanity = 1;
        } else {
            bench_print_usage(argv[0]);
            return 0;
        }
    }
    if (config->modules == 0u || config->repeats == 0u) return 0;
    return 1;
}

int
main(
    int argc,
    char** argv
) {
    BenchConfig config;
    BenchContext context;
    CUdevice device;
    CUcontext cu_context = NULL;
    int ok = 0;

    memset(&context, 0, sizeof(context));
    if (!bench_parse_args(argc, argv, &config)) return 2;
    if (!bench_detect_sm(&context.sm_major, &context.sm_minor)) return 1;
    if (config.sm_forced) {
        context.sm_major = config.sm_major;
        context.sm_minor = config.sm_minor;
    }
    if (!BENCH_CHECK_CUDA(cuDeviceGet(&device, 0))) return 1;
    if (!BENCH_CHECK_CUDA(cuDevicePrimaryCtxRetain(&cu_context, device))) return 1;
    if (!BENCH_CHECK_CUDA(cuCtxSetCurrent(cu_context))) goto done;

    printf("cubin_function_patch_compile_scaling_bench modules=%zu sm_%u%u repeats=%u warmup=%u\n",
        config.modules,
        context.sm_major,
        context.sm_minor,
        config.repeats,
        config.warmup);
    if (!bench_setup_context(&context)) goto done;
    if (!config.skip_correctness && !bench_correctness(&context)) goto done;
    if (!bench_prepare_eval_rdcs(&context, config.modules)) goto done;

    printf("%-20s %8s %12s %12s %12s %14s %14s %14s %14s %14s %14s\n",
        "mode",
        "workers",
        "best_ms",
        "mean_ms",
        "stddev_ms",
        "modules/s",
        "nvrtc_sum_ms",
        "nvptx_sum_ms",
        "inject_ms",
        "nvjitlink_ms",
        "patch_ms");
    if (!bench_run_mode(&context, BENCH_MODE_FULL_COMPILE, &config)) goto done;
    if (!bench_run_mode(&context, BENCH_MODE_FULL_PTX_INJECT, &config)) goto done;
    if (!bench_run_mode(&context, BENCH_MODE_PARTIAL_NVJITLINK, &config)) goto done;
    if (!bench_run_mode(&context, BENCH_MODE_PARTIAL_PATCH, &config)) goto done;
    if (!bench_run_mode(&context, BENCH_MODE_NVJITLINK_ONLY, &config)) goto done;
    if (!bench_run_mode(&context, BENCH_MODE_PATCH_ONLY, &config)) goto done;
    if (!config.skip_final_sanity && !bench_final_sanity_check(&context, config.modules)) goto done;
    ok = 1;

done:
    bench_destroy_context(&context);
    if (cu_context != NULL) {
        (void)cuCtxSetCurrent(NULL);
        (void)cuDevicePrimaryCtxRelease(device);
    }
    return ok ? 0 : 1;
}
