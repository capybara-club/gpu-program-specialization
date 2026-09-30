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
#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <nvPTXCompiler.h>
#include <nvrtc.h>

#ifndef PTX_INJECT_MAX_UNIQUE_INJECTS
#define PTX_INJECT_MAX_UNIQUE_INJECTS 16384
#endif
#define PTX_INJECT_IMPLEMENTATION
#include "ptx_inject.h"

#define STACK_PTX_IMPLEMENTATION
#include "stack_ptx.h"

#include "stack_ptx_example_descriptions.h"
#include "ptx_inject_emitter.h"

#ifdef _OPENMP
#include <omp.h>
#endif

static StackPtxInstruction stack_ptx_make_input(StackPtxIdx idx) {
    StackPtxInstruction v = stack_ptx_encode_input(idx);
    return v;
}

static StackPtxInstruction stack_ptx_make_constant_f32(float value) {
    StackPtxInstruction v = stack_ptx_encode_constant_f32(value);
    return v;
}

static StackPtxInstruction stack_ptx_make_return(void) {
    StackPtxInstruction v = stack_ptx_encode_return;
    return v;
}

typedef struct {
    size_t gene_length;
    size_t execution_limit;
    size_t cores;
    uint64_t seed;
    int verbose;
    unsigned int sm_ptx_major;
    unsigned int sm_ptx_minor;
    int sm_ptx_set;
    unsigned int sm_cubin_major;
    unsigned int sm_cubin_minor;
    int sm_cubin_set;
    size_t num_kernels;
    size_t groups_per_kernel;
    size_t tile_size;
    size_t embed_dims;
    size_t input_dims;
    char input_type[8];
    size_t modules;
    int modules_set;
    size_t workspace_bytes;
    int workspace_set;
    const char* dump_cu_path;
    const char* dump_ptx_path;
    const char* dump_module_ptx_path;
    const char* dump_module_cu_path;
    size_t dump_module_idx;
    int dump_module_idx_set;
} BenchConfig;

typedef struct {
    StackPtxRegister* registers;
    size_t num_registers;
    size_t* requests;
    size_t num_requests;
    size_t input_count;
    size_t output_count;
    size_t num_injects;
} BenchLayout;

typedef struct {
    const char** program_stubs;
    const char** inject_stubs;
    unsigned char* arena_base;
    size_t arena_capacity;
} ThreadScratch;

typedef struct {
    unsigned char* base;
    size_t capacity;
    size_t offset;
} Arena;

typedef struct {
    char* ptx;
    size_t ptx_bytes;
    char* cuda;
    size_t cuda_bytes;
} ModuleOutput;

typedef struct {
    size_t programs_per_module;
    BenchLayout layout;
    size_t* reverse_indices;
    StackPtxCompilerInfo compiler_info;
    const StackPtxStackInfo* stack_info;
    size_t stack_ptx_workspace_size;
    size_t execution_limit;
    size_t workspace_bytes;
    size_t num_injects;
} BenchContext;

static const char* kKernelNameFormat = "kernel_%06zu";

static const StackPtxInstruction kF32OpPool[] = {
    stack_ptx_encode_ptx_instruction_add_ftz_f32,
    stack_ptx_encode_ptx_instruction_sub_ftz_f32,
    stack_ptx_encode_ptx_instruction_mul_ftz_f32,
    stack_ptx_encode_ptx_instruction_fma_rn_ftz_f32,
    stack_ptx_encode_ptx_instruction_div_approx_ftz_f32,
    stack_ptx_encode_ptx_instruction_abs_ftz_f32,
    stack_ptx_encode_ptx_instruction_neg_ftz_f32,
    stack_ptx_encode_ptx_instruction_min_ftz_f32,
    stack_ptx_encode_ptx_instruction_max_ftz_f32,
    stack_ptx_encode_ptx_instruction_rcp_approx_ftz_f32,
    stack_ptx_encode_ptx_instruction_sqrt_approx_ftz_f32,
    stack_ptx_encode_ptx_instruction_rsqrt_approx_ftz_f32,
    stack_ptx_encode_ptx_instruction_sin_approx_ftz_f32,
    stack_ptx_encode_ptx_instruction_cos_approx_ftz_f32,
    stack_ptx_encode_ptx_instruction_lg2_approx_ftz_f32,
    stack_ptx_encode_ptx_instruction_ex2_approx_ftz_f32,
    stack_ptx_encode_ptx_instruction_tanh_approx_f32,
    stack_ptx_encode_ptx_instruction_copysign_f32
};

enum {
    LABEL_WIDTH = 28,
    NUM_WIDTH = 12
};

static int g_label_width = LABEL_WIDTH;

static void print_kv_str(const char* label, const char* value) {
    printf("%-*s %*s\n", g_label_width, label, NUM_WIDTH, value ? value : "");
}

static void print_kv_size(const char* label, size_t value) {
    printf("%-*s %*zu\n", g_label_width, label, NUM_WIDTH, value);
}

static void print_kv_double(const char* label, double value, int precision) {
    printf("%-*s %*.*f\n", g_label_width, label, NUM_WIDTH, precision, value);
}

static void init_label_width(void) {
    static const char* labels[] = {
        "Kernel source bytes:",
        "Kernel PTX bytes:",
        "Dumped PTX CU:",
        "Dumped PTX:",
        "Dumped module PTX:",
        "Dumped module CU:",
        "Dumped module index:",
        "PTX version:",
        "nvPTXCompiler API:",
        "SM (PTX):",
        "SM (cubin):",
        "Modules:",
        "Programs per module:",
        "Total programs:",
        "Kernels:",
        "Groups per kernel:",
        "Tile size:",
        "Embed dims:",
        "Input dims:",
        "Input type:",
        "PTX instructions per program:",
        "Program execution limit:",
        "Workspace bytes/thread:",
        "Threads:",
        "Wall time (ms):",
        "Modules/sec:",
        "Modules/sec/thread:",
        "Full compile (us/prog):",
        "Full compile (us/prog/thread):",
        "Full compile (progs/sec):",
        "Full compile (progs/sec/thread):"
    };
    size_t max_len = 0;
    for (size_t i = 0; i < sizeof(labels) / sizeof(labels[0]); ++i) {
        size_t len = strlen(labels[i]);
        if (len > max_len) {
            max_len = len;
        }
    }
    if ((int)max_len > g_label_width) {
        g_label_width = (int)max_len;
    }
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static void print_compile_stats(
    const char* title,
    double wall_ms,
    size_t modules,
    size_t programs,
    size_t threads
) {
    if (title && title[0]) {
        printf("\n%s\n", title);
    }
    const double wall_s = wall_ms / 1000.0;
    const double modules_per_sec = wall_s > 0.0 ? (double)modules / wall_s : 0.0;
    const double modules_per_sec_thread = threads > 0 ? modules_per_sec / (double)threads : 0.0;
    const double us_per_prog = programs > 0 ? (wall_ms * 1000.0) / (double)programs : 0.0;
    const double us_per_prog_thread = us_per_prog * (double)threads;
    const double progs_per_sec = wall_s > 0.0 ? (double)programs / wall_s : 0.0;
    const double progs_per_sec_thread = threads > 0 ? progs_per_sec / (double)threads : 0.0;

    print_kv_double("Wall time (ms):", wall_ms, 3);
    print_kv_double("Modules/sec:", modules_per_sec, 2);
    print_kv_double("Modules/sec/thread:", modules_per_sec_thread, 2);
    print_kv_double("Full compile (us/prog):", us_per_prog, 3);
    print_kv_double("Full compile (us/prog/thread):", us_per_prog_thread, 3);
    print_kv_double("Full compile (progs/sec):", progs_per_sec, 2);
    print_kv_double("Full compile (progs/sec/thread):", progs_per_sec_thread, 2);
}

static int parse_size_t_arg(const char* s, size_t* out) {
    if (!s || !out) {
        return 0;
    }
    while (*s == ' ' || *s == '\t' || *s == '\n') {
        ++s;
    }
    errno = 0;
    char* end = NULL;
    unsigned long long v = strtoull(s, &end, 10);
    if (errno != 0 || end == s) {
        return 0;
    }
    while (*end == ' ' || *end == '\t' || *end == '\n') {
        ++end;
    }
    if (*end != '\0') {
        return 0;
    }
    if (v > (unsigned long long)SIZE_MAX) {
        return 0;
    }
    *out = (size_t)v;
    return 1;
}

static int parse_u64_arg(const char* s, uint64_t* out) {
    if (!s || !out) {
        return 0;
    }
    while (*s == ' ' || *s == '\t' || *s == '\n') {
        ++s;
    }
    errno = 0;
    char* end = NULL;
    unsigned long long v = strtoull(s, &end, 10);
    if (errno != 0 || end == s) {
        return 0;
    }
    while (*end == ' ' || *end == '\t' || *end == '\n') {
        ++end;
    }
    if (*end != '\0') {
        return 0;
    }
    *out = (uint64_t)v;
    return 1;
}

static int parse_sm(const char* s, unsigned int* major_out, unsigned int* minor_out) {
    if (!s || !s[0] || !major_out || !minor_out) {
        return 0;
    }
    while (*s == ' ' || *s == '\t' || *s == '\n') {
        ++s;
    }
    if ((s[0] == 's' || s[0] == 'S') && (s[1] == 'm' || s[1] == 'M') && s[2] == '_') {
        s += 3;
    }

    const char* start = s;
    const char* dot = NULL;
    const char* p = s;
    while (*p) {
        if (*p == '.') {
            dot = p;
            break;
        }
        if (!isdigit((unsigned char)*p)) {
            return 0;
        }
        ++p;
    }

    const char* end = dot ? dot : p;
    if (end == start) {
        return 0;
    }

    unsigned long major = 0;
    for (const char* cur = start; cur < end; ++cur) {
        major = major * 10 + (unsigned long)(*cur - '0');
    }

    unsigned long minor = 0;
    if (dot) {
        const char* minor_start = dot + 1;
        if (*minor_start == '\0') {
            return 0;
        }
        for (const char* cur = minor_start; *cur; ++cur) {
            if (!isdigit((unsigned char)*cur)) {
                return 0;
            }
            minor = minor * 10 + (unsigned long)(*cur - '0');
        }
    } else {
        size_t len = (size_t)(end - start);
        if (len == 1) {
            minor = 0;
        } else {
            minor = (unsigned long)(end[-1] - '0');
            major = major / 10;
        }
    }

    if (major == 0 || minor > 9) {
        return 0;
    }
    *major_out = (unsigned int)major;
    *minor_out = (unsigned int)minor;
    return 1;
}

static int parse_input_type(const char* s, char out[8]) {
    if (!s || !out) {
        return 0;
    }
    char tmp[8];
    size_t len = strlen(s);
    if (len == 0 || len >= sizeof(tmp)) {
        return 0;
    }
    for (size_t i = 0; i < len; ++i) {
        tmp[i] = (char)toupper((unsigned char)s[i]);
    }
    tmp[len] = '\0';
    if (strcmp(tmp, "F32") == 0 || strcmp(tmp, "U32") == 0) {
        strcpy(out, tmp);
        return 1;
    }
    return 0;
}

static int parse_ptx_version(const char* ptx, unsigned int* major_out, unsigned int* minor_out) {
    if (!ptx || !major_out || !minor_out) {
        return 0;
    }

    const char* p = ptx;
    while (*p) {
        while (*p == ' ' || *p == '\t') {
            ++p;
        }
        if (*p == '.' && strncmp(p, ".version", 8) == 0 && isspace((unsigned char)p[8])) {
            const char* q = p + 8;
            while (*q == ' ' || *q == '\t') {
                ++q;
            }
            if (!isdigit((unsigned char)*q)) {
                return 0;
            }
            unsigned int major = 0;
            while (isdigit((unsigned char)*q)) {
                major = major * 10u + (unsigned int)(*q - '0');
                ++q;
            }
            unsigned int minor = 0;
            if (*q == '.') {
                ++q;
                if (!isdigit((unsigned char)*q)) {
                    return 0;
                }
                while (isdigit((unsigned char)*q)) {
                    minor = minor * 10u + (unsigned int)(*q - '0');
                    ++q;
                }
            }
            *major_out = major;
            *minor_out = minor;
            return 1;
        }
        while (*p && *p != '\n') {
            ++p;
        }
        if (*p == '\n') {
            ++p;
        }
    }
    return 0;
}

static void get_sm_fallback(unsigned int* major_out, unsigned int* minor_out) {
    const char* env = getenv("STACK_PTX_COMPILER_SM");
    if (parse_sm(env, major_out, minor_out)) {
        return;
    }
    env = getenv("STACK_PTX_NNG_SM");
    if (parse_sm(env, major_out, minor_out)) {
        return;
    }
    *major_out = 8;
    *minor_out = 0;
}

static uint64_t rng_splitmix64(uint64_t* state) {
    uint64_t x = (*state += 0x9e3779b97f4a7c15ull);
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

static uint64_t rng_next_u64(uint64_t* state) {
    return rng_splitmix64(state);
}

static size_t rng_uniform_index(uint64_t* state, size_t upper_bound) {
    if (upper_bound == 0) {
        return 0;
    }
    const uint64_t limit = UINT64_MAX - (UINT64_MAX % upper_bound);
    uint64_t value = rng_next_u64(state);
    while (value > limit) {
        value = rng_next_u64(state);
    }
    return (size_t)(value % upper_bound);
}

static float rng_uniform_f32_range(uint64_t* state, float min_value, float max_value) {
    const uint64_t v = rng_next_u64(state);
    const double t = (double)(v >> 11) * (1.0 / 9007199254740992.0);
    return (float)(min_value + (max_value - min_value) * t);
}

static StackPtxInstruction random_instruction(uint64_t* state, int allow_inputs, size_t input_count) {
    const uint64_t roll = rng_next_u64(state);
    const uint64_t bucket = roll % 10;
    if (allow_inputs && bucket < 2 && input_count > 0) {
        const size_t idx = rng_uniform_index(state, input_count);
        return stack_ptx_make_input((StackPtxIdx)idx);
    }
    if (bucket < 4) {
        const float v = rng_uniform_f32_range(state, -10.0f, 10.0f);
        return stack_ptx_make_constant_f32(v);
    }
    const size_t op_idx = rng_uniform_index(state, STACK_PTX_ARRAY_NUM_ELEMS(kF32OpPool));
    return kF32OpPool[op_idx];
}

static void generate_programs(
    StackPtxInstruction* programs,
    size_t num_programs,
    size_t gene_length,
    size_t input_count,
    int input_stack_is_f32,
    uint64_t seed
) {
    uint64_t state = seed ? seed : 0x1d2e3f4a5b6c7d8eull;
    (void)rng_next_u64(&state);

    for (size_t i = 0; i < num_programs; ++i) {
        StackPtxInstruction* prog = programs + i * gene_length;
        size_t pos = 0;
        if (input_count > 0 && pos + 1 < gene_length) {
            prog[pos++] = stack_ptx_make_input(0);
            if (!input_stack_is_f32 && pos + 1 < gene_length) {
                prog[pos++] = stack_ptx_encode_ptx_instruction_cvt_rn_f32_u32;
            }
        }
        if (input_count > 1 && pos + 1 < gene_length) {
            prog[pos++] = stack_ptx_make_input(1);
            if (!input_stack_is_f32 && pos + 1 < gene_length) {
                prog[pos++] = stack_ptx_encode_ptx_instruction_cvt_rn_f32_u32;
            }
        }
        while (pos + 1 < gene_length) {
            prog[pos++] = random_instruction(&state, input_stack_is_f32, input_count);
        }
        prog[gene_length - 1] = stack_ptx_make_return();
    }
}

static const char* nvptx_result_to_string(nvPTXCompileResult result) {
    switch (result) {
        case NVPTXCOMPILE_SUCCESS:
            return "NVPTXCOMPILE_SUCCESS";
#ifdef NVPTXCOMPILE_PARSE_ONLY_SUCCESS
        case NVPTXCOMPILE_PARSE_ONLY_SUCCESS:
            return "NVPTXCOMPILE_PARSE_ONLY_SUCCESS";
#endif
        case NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE:
            return "NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE";
        case NVPTXCOMPILE_ERROR_INVALID_INPUT:
            return "NVPTXCOMPILE_ERROR_INVALID_INPUT";
        case NVPTXCOMPILE_ERROR_COMPILATION_FAILURE:
            return "NVPTXCOMPILE_ERROR_COMPILATION_FAILURE";
        case NVPTXCOMPILE_ERROR_INTERNAL:
            return "NVPTXCOMPILE_ERROR_INTERNAL";
        case NVPTXCOMPILE_ERROR_OUT_OF_MEMORY:
            return "NVPTXCOMPILE_ERROR_OUT_OF_MEMORY";
        case NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE:
            return "NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE";
        case NVPTXCOMPILE_ERROR_UNSUPPORTED_PTX_VERSION:
            return "NVPTXCOMPILE_ERROR_UNSUPPORTED_PTX_VERSION";
        case NVPTXCOMPILE_ERROR_UNSUPPORTED_DEVSIDE_SYNC:
            return "NVPTXCOMPILE_ERROR_UNSUPPORTED_DEVSIDE_SYNC";
        case NVPTXCOMPILE_ERROR_CANCELLED:
            return "NVPTXCOMPILE_ERROR_CANCELLED";
        default:
            break;
    }
    return "NVPTXCOMPILE_ERROR_UNKNOWN";
}

static void nvptx_print_log(nvPTXCompilerHandle compiler, int print_info) {
    size_t log_size = 0;
    if (print_info) {
        if (nvPTXCompilerGetInfoLogSize(compiler, &log_size) == NVPTXCOMPILE_SUCCESS && log_size > 1) {
            char* log_buf = (char*)malloc(log_size);
            if (log_buf && nvPTXCompilerGetInfoLog(compiler, log_buf) == NVPTXCOMPILE_SUCCESS) {
                fprintf(stderr, "nvPTXCompiler info log:\n%s\n", log_buf);
            }
            free(log_buf);
        }
    } else {
        if (nvPTXCompilerGetErrorLogSize(compiler, &log_size) == NVPTXCOMPILE_SUCCESS && log_size > 1) {
            char* log_buf = (char*)malloc(log_size);
            if (log_buf && nvPTXCompilerGetErrorLog(compiler, log_buf) == NVPTXCOMPILE_SUCCESS) {
                fprintf(stderr, "nvPTXCompiler error log:\n%s\n", log_buf);
            }
            free(log_buf);
        }
    }
}

static char* nvrtc_compile_ptx(
    const char* source,
    unsigned int sm_major,
    unsigned int sm_minor,
    int verbose,
    size_t* out_bytes
) {
    nvrtcProgram program = NULL;
    nvrtcResult rc = nvrtcCreateProgram(&program, source, "ptx_inject_bench.cu", 0, NULL, NULL);
    if (rc != NVRTC_SUCCESS) {
        fprintf(stderr, "nvrtcCreateProgram failed: %s\n", nvrtcGetErrorString(rc));
        return NULL;
    }

    char arch_option[64];
    snprintf(arch_option, sizeof(arch_option), "--gpu-architecture=compute_%u%u", sm_major, sm_minor);
    const char* options[] = { "--std=c++14", "-default-device", arch_option };
    rc = nvrtcCompileProgram(program, 3, options);

    size_t log_size = 0;
    if (nvrtcGetProgramLogSize(program, &log_size) == NVRTC_SUCCESS && log_size > 1) {
        if (verbose || rc != NVRTC_SUCCESS) {
            char* log_buf = (char*)malloc(log_size);
            if (log_buf && nvrtcGetProgramLog(program, log_buf) == NVRTC_SUCCESS) {
                fprintf(stderr, "NVRTC log:\n%s\n", log_buf);
            }
            free(log_buf);
        }
    }

    if (rc != NVRTC_SUCCESS) {
        fprintf(stderr, "nvrtcCompileProgram failed: %s\n", nvrtcGetErrorString(rc));
        nvrtcDestroyProgram(&program);
        return NULL;
    }

    size_t ptx_size = 0;
    rc = nvrtcGetPTXSize(program, &ptx_size);
    if (rc != NVRTC_SUCCESS) {
        fprintf(stderr, "nvrtcGetPTXSize failed: %s\n", nvrtcGetErrorString(rc));
        nvrtcDestroyProgram(&program);
        return NULL;
    }

    char* ptx = (char*)malloc(ptx_size + 1);
    if (!ptx) {
        fprintf(stderr, "failed to allocate PTX buffer (%zu bytes)\n", ptx_size + 1);
        nvrtcDestroyProgram(&program);
        return NULL;
    }

    rc = nvrtcGetPTX(program, ptx);
    if (rc != NVRTC_SUCCESS) {
        fprintf(stderr, "nvrtcGetPTX failed: %s\n", nvrtcGetErrorString(rc));
        free(ptx);
        nvrtcDestroyProgram(&program);
        return NULL;
    }

    nvrtcDestroyProgram(&program);
    ptx[ptx_size] = '\0';
    if (out_bytes) {
        *out_bytes = ptx_size;
    }
    return ptx;
}

static int nvrtc_compile_cubin(
    const char* source,
    unsigned int sm_major,
    unsigned int sm_minor,
    int verbose,
    void** out_cubin,
    size_t* out_bytes
) {
    if (!source || !out_cubin || !out_bytes) {
        return 0;
    }
    *out_cubin = NULL;
    *out_bytes = 0;

    nvrtcProgram program = NULL;
    nvrtcResult rc = nvrtcCreateProgram(&program, source, "inline_ptx_bench.cu", 0, NULL, NULL);
    if (rc != NVRTC_SUCCESS) {
        fprintf(stderr, "nvrtcCreateProgram failed: %s\n", nvrtcGetErrorString(rc));
        return 0;
    }

    char arch_option[64];
    snprintf(arch_option, sizeof(arch_option), "--gpu-architecture=sm_%u%u", sm_major, sm_minor);
    const char* options[] = { "--std=c++14", "-default-device", arch_option };
    rc = nvrtcCompileProgram(program, 3, options);

    size_t log_size = 0;
    if (nvrtcGetProgramLogSize(program, &log_size) == NVRTC_SUCCESS && log_size > 1) {
        if (verbose || rc != NVRTC_SUCCESS) {
            char* log_buf = (char*)malloc(log_size);
            if (log_buf && nvrtcGetProgramLog(program, log_buf) == NVRTC_SUCCESS) {
                fprintf(stderr, "NVRTC log:\n%s\n", log_buf);
            }
            free(log_buf);
        }
    }

    if (rc != NVRTC_SUCCESS) {
        fprintf(stderr, "nvrtcCompileProgram failed: %s\n", nvrtcGetErrorString(rc));
        nvrtcDestroyProgram(&program);
        return 0;
    }

    size_t cubin_size = 0;
    rc = nvrtcGetCUBINSize(program, &cubin_size);
    if (rc != NVRTC_SUCCESS) {
        fprintf(stderr, "nvrtcGetCUBINSize failed: %s\n", nvrtcGetErrorString(rc));
        nvrtcDestroyProgram(&program);
        return 0;
    }

    void* cubin = malloc(cubin_size);
    if (!cubin) {
        fprintf(stderr, "failed to allocate cubin buffer (%zu bytes)\n", cubin_size);
        nvrtcDestroyProgram(&program);
        return 0;
    }

    rc = nvrtcGetCUBIN(program, cubin);
    if (rc != NVRTC_SUCCESS) {
        fprintf(stderr, "nvrtcGetCUBIN failed: %s\n", nvrtcGetErrorString(rc));
        free(cubin);
        nvrtcDestroyProgram(&program);
        return 0;
    }

    nvrtcDestroyProgram(&program);
    *out_cubin = cubin;
    *out_bytes = cubin_size;
    return 1;
}

static int nvptx_compile_cubin(
    const char* ptx,
    size_t ptx_bytes,
    unsigned int sm_major,
    unsigned int sm_minor,
    int verbose,
    void** out_cubin,
    size_t* out_bytes
) {
    if (!ptx || !out_cubin || !out_bytes) {
        return 0;
    }
    *out_cubin = NULL;
    *out_bytes = 0;

    nvPTXCompilerHandle compiler = NULL;
    nvPTXCompileResult rc = nvPTXCompilerCreate(&compiler, ptx_bytes, ptx);
    if (rc != NVPTXCOMPILE_SUCCESS) {
        fprintf(stderr, "nvPTXCompiler create failed: %s\n", nvptx_result_to_string(rc));
        if (compiler) {
            nvptx_print_log(compiler, 0);
            nvptx_print_log(compiler, 1);
            nvPTXCompilerDestroy(&compiler);
        }
        return 0;
    }

    char arch_option[64];
    snprintf(arch_option, sizeof(arch_option), "--gpu-name=sm_%u%u", sm_major, sm_minor);
    const char* options[2];
    options[0] = arch_option;
    options[1] = verbose ? "--verbose" : NULL;
    int num_options = verbose ? 2 : 1;

    rc = nvPTXCompilerCompile(compiler, num_options, options);
    if (rc != NVPTXCOMPILE_SUCCESS) {
        fprintf(stderr, "nvPTXCompiler compile failed: %s\n", nvptx_result_to_string(rc));
        nvptx_print_log(compiler, 0);
        nvptx_print_log(compiler, 1);
        nvPTXCompilerDestroy(&compiler);
        return 0;
    }
    if (verbose) {
        nvptx_print_log(compiler, 1);
    }

    size_t cubin_size = 0;
    rc = nvPTXCompilerGetCompiledProgramSize(compiler, &cubin_size);
    if (rc != NVPTXCOMPILE_SUCCESS) {
        fprintf(stderr, "nvPTXCompiler get size failed: %s\n", nvptx_result_to_string(rc));
        nvptx_print_log(compiler, 0);
        nvptx_print_log(compiler, 1);
        nvPTXCompilerDestroy(&compiler);
        return 0;
    }

    void* cubin = malloc(cubin_size);
    if (!cubin) {
        fprintf(stderr, "failed to allocate cubin buffer (%zu bytes)\n", cubin_size);
        nvPTXCompilerDestroy(&compiler);
        return 0;
    }

    rc = nvPTXCompilerGetCompiledProgram(compiler, cubin);
    if (rc != NVPTXCOMPILE_SUCCESS) {
        fprintf(stderr, "nvPTXCompiler get program failed: %s\n", nvptx_result_to_string(rc));
        nvptx_print_log(compiler, 0);
        nvptx_print_log(compiler, 1);
        free(cubin);
        nvPTXCompilerDestroy(&compiler);
        return 0;
    }

    nvPTXCompilerDestroy(&compiler);
    *out_cubin = cubin;
    *out_bytes = cubin_size;
    return 1;
}

static int write_file(const char* path, const char* data, size_t size) {
    if (!path || !data) {
        return 0;
    }
    FILE* fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "failed to open %s for writing\n", path);
        return 0;
    }
    size_t written = fwrite(data, 1, size, fp);
    if (written != size) {
        fprintf(stderr, "failed to write %zu bytes to %s\n", size, path);
        fclose(fp);
        return 0;
    }
    if (fclose(fp) != 0) {
        fprintf(stderr, "failed to close %s\n", path);
        return 0;
    }
    return 1;
}

static void trim_trailing_whitespace(char* s) {
    if (!s) {
        return;
    }
    size_t len = strlen(s);
    while (len > 0 && isspace((unsigned char)s[len - 1])) {
        s[len - 1] = '\0';
        len -= 1;
    }
}

static int read_first_line(const char* path, char* buffer, size_t buffer_size) {
    if (!path || !buffer || buffer_size == 0) {
        return 0;
    }
    FILE* fp = fopen(path, "r");
    if (!fp) {
        return 0;
    }
    if (!fgets(buffer, (int)buffer_size, fp)) {
        fclose(fp);
        return 0;
    }
    fclose(fp);
    trim_trailing_whitespace(buffer);
    return 1;
}

static int read_u64_from_file(const char* path, unsigned long long* out) {
    if (!out) {
        return 0;
    }
    char buffer[64];
    if (!read_first_line(path, buffer, sizeof(buffer))) {
        return 0;
    }
    errno = 0;
    char* end = NULL;
    unsigned long long value = strtoull(buffer, &end, 10);
    if (errno != 0 || end == buffer) {
        return 0;
    }
    *out = value;
    return 1;
}

typedef struct {
    int physical_id;
    int core_id;
} CpuCorePair;

static int parse_int_field(const char* line, int* out) {
    if (!line || !out) {
        return 0;
    }
    const char* colon = strchr(line, ':');
    if (!colon) {
        return 0;
    }
    errno = 0;
    char* end = NULL;
    long value = strtol(colon + 1, &end, 10);
    if (errno != 0 || end == colon + 1) {
        return 0;
    }
    *out = (int)value;
    return 1;
}

static int parse_cpu_mhz_field(const char* line, double* out) {
    if (!line || !out) {
        return 0;
    }
    const char* colon = strchr(line, ':');
    if (!colon) {
        return 0;
    }
    errno = 0;
    char* end = NULL;
    double value = strtod(colon + 1, &end);
    if (errno != 0 || end == colon + 1) {
        return 0;
    }
    *out = value;
    return 1;
}

static int read_cpu_info(char* model, size_t model_size, size_t* cores_out, double* mhz_out, int* mhz_set_out) {
    if (model && model_size > 0) {
        model[0] = '\0';
    }
    if (cores_out) {
        *cores_out = 0;
    }
    if (mhz_out) {
        *mhz_out = 0.0;
    }
    if (mhz_set_out) {
        *mhz_set_out = 0;
    }

    FILE* fp = fopen("/proc/cpuinfo", "r");
    if (!fp) {
        return 0;
    }

    char line[512];
    size_t cpu_count = 0;
    size_t pair_count = 0;
    size_t pair_capacity = 0;
    CpuCorePair* pairs = NULL;
    int cur_physical = -1;
    int cur_core = -1;
    while (fgets(line, sizeof(line), fp)) {
        if (strncmp(line, "processor", 9) == 0) {
            cpu_count += 1;
        } else if (strncmp(line, "model name", 10) == 0) {
            if (model && model_size > 0 && model[0] == '\0') {
                const char* colon = strchr(line, ':');
                if (colon) {
                    const char* name = colon + 1;
                    while (*name == ' ' || *name == '\t') {
                        ++name;
                    }
                    strncpy(model, name, model_size - 1);
                    model[model_size - 1] = '\0';
                    trim_trailing_whitespace(model);
                }
            }
        } else if (strncmp(line, "cpu MHz", 7) == 0) {
            double mhz_value = 0.0;
            if (parse_cpu_mhz_field(line, &mhz_value)) {
                if (mhz_out) {
                    *mhz_out = mhz_value;
                }
                if (mhz_set_out) {
                    *mhz_set_out = 1;
                }
            }
        } else if (strncmp(line, "physical id", 11) == 0) {
            parse_int_field(line, &cur_physical);
        } else if (strncmp(line, "core id", 7) == 0) {
            parse_int_field(line, &cur_core);
        } else if (line[0] == '\n' || line[0] == '\r') {
            if (cur_physical >= 0 && cur_core >= 0) {
                if (pair_count == pair_capacity) {
                    size_t new_capacity = pair_capacity ? pair_capacity * 2 : 16;
                    CpuCorePair* next = (CpuCorePair*)realloc(pairs, new_capacity * sizeof(*pairs));
                    if (!next) {
                        break;
                    }
                    pairs = next;
                    pair_capacity = new_capacity;
                }
                pairs[pair_count].physical_id = cur_physical;
                pairs[pair_count].core_id = cur_core;
                pair_count += 1;
            }
            cur_physical = -1;
            cur_core = -1;
        }
    }
    fclose(fp);

    if (cores_out) {
        if (pair_count > 0) {
            *cores_out = pair_count;
        } else {
            *cores_out = cpu_count;
        }
    }

    free(pairs);
    return 1;
}

static int build_cpu_label(char* out, size_t out_size) {
    if (!out || out_size == 0) {
        return 0;
    }
    out[0] = '\0';

    char model[256];
    size_t cores = 0;
    double mhz = 0.0;
    int mhz_set = 0;
    read_cpu_info(model, sizeof(model), &cores, &mhz, &mhz_set);

    unsigned long long khz = 0;
    double ghz = 0.0;
    int has_freq = 0;
    if (read_u64_from_file("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq", &khz) ||
        read_u64_from_file("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq", &khz)) {
        ghz = (double)khz / 1000000.0;
        has_freq = 1;
    } else if (mhz_set) {
        ghz = mhz / 1000.0;
        has_freq = 1;
    }
    if (model[0] == '\0' && cores == 0 && !has_freq) {
        return 0;
    }
    if (has_freq) {
        if (model[0] != '\0' && cores > 0) {
            snprintf(out, out_size, "%s (%zu) @ %.2f GHz", model, cores, ghz);
        } else if (model[0] != '\0') {
            snprintf(out, out_size, "%s @ %.2f GHz", model, ghz);
        } else if (cores > 0) {
            snprintf(out, out_size, "%zu cores @ %.2f GHz", cores, ghz);
        } else {
            snprintf(out, out_size, "%.2f GHz", ghz);
        }
    } else {
        if (model[0] != '\0' && cores > 0) {
            snprintf(out, out_size, "%s (%zu)", model, cores);
        } else if (model[0] != '\0') {
            snprintf(out, out_size, "%s", model);
        } else {
            snprintf(out, out_size, "%zu cores", cores);
        }
    }
    return 1;
}

static size_t default_workspace_bytes(
    size_t stack_ptx_workspace,
    size_t gene_length,
    size_t programs_per_module
) {
    const size_t bytes_per_instruction = 64;
    size_t stub_estimate = programs_per_module;
    if (gene_length != 0 && stub_estimate <= SIZE_MAX / gene_length) {
        stub_estimate *= gene_length;
    } else {
        stub_estimate = SIZE_MAX;
    }
    if (stub_estimate <= SIZE_MAX / bytes_per_instruction) {
        stub_estimate *= bytes_per_instruction;
    } else {
        stub_estimate = SIZE_MAX;
    }

    size_t total = stack_ptx_workspace;
    if (stub_estimate < SIZE_MAX - total) {
        total += stub_estimate;
    } else {
        total = SIZE_MAX;
    }
    const size_t overhead = 4u * 1024u * 1024u;
    if (total < SIZE_MAX - overhead) {
        total += overhead;
    }
    const size_t min_bytes = 8u * 1024u * 1024u;
    if (total < min_bytes) {
        total = min_bytes;
    }
    return total;
}

static int stack_idx_from_data_type(
    const StackPtxStackInfo* info,
    const char* data_type,
    size_t* out_idx
) {
    if (!info || !data_type || !out_idx) {
        return 0;
    }
    char lowered[32];
    size_t len = strlen(data_type);
    if (len == 0 || len >= sizeof(lowered)) {
        return 0;
    }
    for (size_t i = 0; i < len; ++i) {
        lowered[i] = (char)tolower((unsigned char)data_type[i]);
    }
    lowered[len] = '\0';
    for (size_t i = 0; i < info->num_stacks; ++i) {
        const char* prefix = info->stack_literal_prefixes[i];
        if (!prefix) {
            continue;
        }
        if (strcmp(prefix, lowered) == 0) {
            *out_idx = i;
            return 1;
        }
    }
    return 0;
}

static int init_layout(PtxInjectHandle ptx_inject, BenchLayout* layout) {
    if (!ptx_inject || !layout) {
        return 0;
    }
    size_t num_injects = 0;
    if (ptx_inject_num_injects(ptx_inject, &num_injects) != PTX_INJECT_SUCCESS || num_injects == 0) {
        fprintf(stderr, "no inject sites found in kernel PTX\n");
        return 0;
    }
    layout->num_injects = num_injects;

    const char* inject_name = "func_0";
    size_t inject_idx = 0;
    size_t num_args = 0;
    if (ptx_inject_inject_info_by_name(
            ptx_inject,
            inject_name,
            &inject_idx,
            &num_args,
            NULL
        ) != PTX_INJECT_SUCCESS) {
        fprintf(stderr, "failed to lookup inject site %s\n", inject_name);
        return 0;
    }

    size_t input_count = 0;
    size_t output_count = 0;
    for (size_t arg_idx = 0; arg_idx < num_args; ++arg_idx) {
        PtxInjectMutType mut_type = PTX_INJECT_MUT_TYPE_IN;
        if (ptx_inject_variable_info_by_index(
                ptx_inject,
                inject_idx,
                arg_idx,
                NULL,
                NULL,
                &mut_type,
                NULL,
                NULL
            ) != PTX_INJECT_SUCCESS) {
            fprintf(stderr, "failed to query inject arg %zu\n", arg_idx);
            return 0;
        }
        if (mut_type == PTX_INJECT_MUT_TYPE_IN) {
            input_count += 1;
        } else if (mut_type == PTX_INJECT_MUT_TYPE_OUT || mut_type == PTX_INJECT_MUT_TYPE_MOD) {
            output_count += 1;
        }
    }
    if (input_count == 0 || output_count == 0) {
        fprintf(stderr, "invalid inject arg classification: inputs=%zu outputs=%zu\n", input_count, output_count);
        return 0;
    }

    layout->input_count = input_count;
    layout->output_count = output_count;
    layout->num_registers = input_count + output_count;
    layout->num_requests = output_count;

    layout->registers = (StackPtxRegister*)calloc(layout->num_registers, sizeof(*layout->registers));
    layout->requests = (size_t*)calloc(layout->num_requests, sizeof(*layout->requests));
    if (!layout->registers || !layout->requests) {
        fprintf(stderr, "failed to allocate register layout\n");
        return 0;
    }

    size_t input_written = 0;
    size_t output_written = 0;
    for (size_t arg_idx = 0; arg_idx < num_args; ++arg_idx) {
        const char* reg_name = NULL;
        const char* data_type = NULL;
        PtxInjectMutType mut_type = PTX_INJECT_MUT_TYPE_IN;
        if (ptx_inject_variable_info_by_index(
                ptx_inject,
                inject_idx,
                arg_idx,
                NULL,
                &reg_name,
                &mut_type,
                NULL,
                &data_type
            ) != PTX_INJECT_SUCCESS) {
            fprintf(stderr, "failed to query inject arg %zu\n", arg_idx);
            return 0;
        }
        size_t stack_idx = 0;
        if (!stack_idx_from_data_type(&stack_ptx_stack_info, data_type, &stack_idx)) {
            fprintf(stderr, "unsupported data type: %s\n", data_type ? data_type : "<null>");
            return 0;
        }
        if (mut_type == PTX_INJECT_MUT_TYPE_IN) {
            layout->registers[input_written].name = reg_name;
            layout->registers[input_written].stack_idx = stack_idx;
            input_written += 1;
        } else if (mut_type == PTX_INJECT_MUT_TYPE_OUT || mut_type == PTX_INJECT_MUT_TYPE_MOD) {
            const size_t reg_idx = input_count + output_written;
            layout->registers[reg_idx].name = reg_name;
            layout->registers[reg_idx].stack_idx = stack_idx;
            layout->requests[output_written] = reg_idx;
            output_written += 1;
        }
    }

    if (input_written != input_count || output_written != output_count) {
        fprintf(stderr, "failed to map inject registers\n");
        return 0;
    }
    return 1;
}

static void free_layout(BenchLayout* layout) {
    if (!layout) {
        return;
    }
    free(layout->registers);
    free(layout->requests);
    memset(layout, 0, sizeof(*layout));
}

static int parse_inject_site_index(const char* name, size_t* out_idx) {
    const char* prefix = "func_";
    const size_t prefix_len = strlen(prefix);
    if (!name || strncmp(name, prefix, prefix_len) != 0) {
        return 0;
    }
    return parse_size_t_arg(name + prefix_len, out_idx);
}

static int build_reverse_indices(
    PtxInjectHandle ptx_inject,
    size_t programs_per_module,
    size_t** out_map
) {
    if (!ptx_inject || !out_map) {
        return 0;
    }

    size_t num_injects = 0;
    if (ptx_inject_num_injects(ptx_inject, &num_injects) != PTX_INJECT_SUCCESS || num_injects == 0) {
        fprintf(stderr, "failed to query inject count\n");
        return 0;
    }

    size_t* map = (size_t*)calloc(num_injects, sizeof(*map));
    if (!map) {
        fprintf(stderr, "failed to allocate reverse map\n");
        return 0;
    }

    unsigned char* seen = (unsigned char*)calloc(programs_per_module, sizeof(*seen));
    if (!seen) {
        fprintf(stderr, "failed to allocate seen array\n");
        free(map);
        return 0;
    }

    for (size_t inject_idx = 0; inject_idx < num_injects; ++inject_idx) {
        const char* inject_name = NULL;
        if (ptx_inject_inject_info_by_index(
                ptx_inject,
                inject_idx,
                &inject_name,
                NULL,
                NULL
            ) != PTX_INJECT_SUCCESS) {
            fprintf(stderr, "failed to query inject index %zu\n", inject_idx);
            free(seen);
            free(map);
            return 0;
        }
        size_t program_idx = 0;
        if (!parse_inject_site_index(inject_name, &program_idx)) {
            fprintf(stderr, "failed to parse inject name: %s\n", inject_name ? inject_name : "<null>");
            free(seen);
            free(map);
            return 0;
        }
        if (program_idx >= programs_per_module) {
            fprintf(stderr, "inject index out of range: %s\n", inject_name ? inject_name : "<null>");
            free(seen);
            free(map);
            return 0;
        }
        if (seen[program_idx]) {
            fprintf(stderr, "duplicate inject index: %s\n", inject_name ? inject_name : "<null>");
            free(seen);
            free(map);
            return 0;
        }
        seen[program_idx] = 1;
        map[inject_idx] = program_idx;
    }

    free(seen);
    *out_map = map;
    return 1;
}

static int arena_alloc(Arena* arena, size_t bytes, size_t alignment, void** out_ptr) {
    if (!arena || !out_ptr || alignment == 0) {
        return 0;
    }
    size_t aligned_offset = arena->offset;
    size_t mod = aligned_offset % alignment;
    if (mod) {
        aligned_offset += alignment - mod;
    }
    if (aligned_offset > arena->capacity || bytes > arena->capacity - aligned_offset) {
        return 0;
    }
    *out_ptr = arena->base + aligned_offset;
    arena->offset = aligned_offset + bytes;
    return 1;
}

static int build_module_outputs(
    const BenchContext* ctx,
    const BenchConfig* cfg,
    PtxInjectHandle inject,
    const StackPtxInstruction* population,
    size_t gene_length,
    size_t module_idx,
    ThreadScratch* scratch,
    Arena* arena,
    ModuleOutput* out
) {
    if (!ctx || !cfg || !inject || !population || !scratch || !arena || !out) {
        return 0;
    }
    if (module_idx >= SIZE_MAX / ctx->programs_per_module) {
        return 0;
    }

    arena->offset = 0;

    void* stack_ptx_workspace_ptr = NULL;
    if (!arena_alloc(arena, ctx->stack_ptx_workspace_size, 64, &stack_ptx_workspace_ptr)) {
        return 0;
    }

    const size_t module_offset = module_idx * ctx->programs_per_module;
    for (size_t prog_idx = 0; prog_idx < ctx->programs_per_module; ++prog_idx) {
        const size_t population_idx = module_offset + prog_idx;
        const size_t instruction_idx = population_idx * gene_length;
        const StackPtxInstruction* instructions = &population[instruction_idx];

        size_t required = 0;
        StackPtxResult stack_ptx_result = stack_ptx_compile(
                &ctx->compiler_info,
                ctx->stack_info,
                instructions,
                ctx->layout.registers,
                ctx->layout.num_registers,
                NULL,
                0,
                ctx->layout.requests,
                ctx->layout.num_requests,
                ctx->execution_limit,
                stack_ptx_workspace_ptr,
                ctx->stack_ptx_workspace_size,
                NULL,
                0,
                &required
            );
        if (stack_ptx_result != STACK_PTX_SUCCESS) {
            fprintf(stderr, "stack_ptx_compile (size) failed: %s\n",
                stack_ptx_result_to_string(stack_ptx_result));
            return 0;
        }

        const size_t capacity = required + 1;
        char* stub_ptr = NULL;
        if (!arena_alloc(arena, capacity, 1, (void**)&stub_ptr)) {
            return 0;
        }

        stack_ptx_result = stack_ptx_compile(
                &ctx->compiler_info,
                ctx->stack_info,
                instructions,
                ctx->layout.registers,
                ctx->layout.num_registers,
                NULL,
                0,
                ctx->layout.requests,
                ctx->layout.num_requests,
                ctx->execution_limit,
                stack_ptx_workspace_ptr,
                ctx->stack_ptx_workspace_size,
                stub_ptr,
                capacity,
                &required
            );
        if (stack_ptx_result != STACK_PTX_SUCCESS) {
            fprintf(stderr, "stack_ptx_compile (render) failed: %s\n",
                stack_ptx_result_to_string(stack_ptx_result));
            return 0;
        }
        if (required < capacity) {
            stub_ptr[required] = '\0';
        }
        scratch->program_stubs[prog_idx] = stub_ptr;
    }

    for (size_t i = 0; i < ctx->num_injects; ++i) {
        const size_t program_idx = ctx->reverse_indices[i];
        if (program_idx >= ctx->programs_per_module) {
            return 0;
        }
        scratch->inject_stubs[i] = scratch->program_stubs[program_idx];
    }

    size_t rendered_required = 0;
    PtxInjectResult ptx_inject_result = ptx_inject_render_ptx(
            inject,
            scratch->inject_stubs,
            ctx->num_injects,
            NULL,
            0,
            &rendered_required
        );
    if (ptx_inject_result != PTX_INJECT_SUCCESS) {
        fprintf(stderr, "ptx_inject_render_ptx (size) failed: %s\n",
            ptx_inject_result_to_string(ptx_inject_result));
        return 0;
    }

    char* rendered_ptx_ptr = (char*)malloc(rendered_required + 1);
    if (!rendered_ptx_ptr) {
        fprintf(stderr, "failed to allocate rendered PTX (%zu bytes)\n", rendered_required + 1);
        return 0;
    }

    ptx_inject_result = ptx_inject_render_ptx(
            inject,
            scratch->inject_stubs,
            ctx->num_injects,
            rendered_ptx_ptr,
            rendered_required + 1,
            &rendered_required
        );
    if (ptx_inject_result != PTX_INJECT_SUCCESS) {
        fprintf(stderr, "ptx_inject_render_ptx (render) failed: %s\n",
            ptx_inject_result_to_string(ptx_inject_result));
        free(rendered_ptx_ptr);
        return 0;
    }
    rendered_ptx_ptr[rendered_required] = '\0';

    size_t cuda_required = 0;
    PtxInjectEmitterResult cuda_emit_result = ptx_inject_emit_cuda(
        (int64_t)cfg->num_kernels,
        (int64_t)cfg->groups_per_kernel,
        (int64_t)cfg->tile_size,
        (int64_t)cfg->embed_dims,
        kKernelNameFormat,
        cfg->input_type,
        (int64_t)cfg->input_dims,
        0,
        scratch->program_stubs,
        ctx->programs_per_module,
        NULL,
        0,
        &cuda_required
    );
    if (cuda_emit_result != PTX_INJECT_EMITTER_SUCCESS) {
        free(rendered_ptx_ptr);
        return 0;
    }

    char* rendered_cuda_ptr = (char*)malloc(cuda_required + 1);
    if (!rendered_cuda_ptr) {
        fprintf(stderr, "failed to allocate rendered CUDA (%zu bytes)\n", cuda_required + 1);
        free(rendered_ptx_ptr);
        return 0;
    }

    size_t cuda_written = 0;
    cuda_emit_result = ptx_inject_emit_cuda(
        (int64_t)cfg->num_kernels,
        (int64_t)cfg->groups_per_kernel,
        (int64_t)cfg->tile_size,
        (int64_t)cfg->embed_dims,
        kKernelNameFormat,
        cfg->input_type,
        (int64_t)cfg->input_dims,
        0,
        scratch->program_stubs,
        ctx->programs_per_module,
        rendered_cuda_ptr,
        cuda_required + 1,
        &cuda_written
    );
    if (cuda_emit_result != PTX_INJECT_EMITTER_SUCCESS) {
        free(rendered_cuda_ptr);
        free(rendered_ptx_ptr);
        return 0;
    }
    rendered_cuda_ptr[cuda_written] = '\0';

    out->ptx = rendered_ptx_ptr;
    out->ptx_bytes = rendered_required;
    out->cuda = rendered_cuda_ptr;
    out->cuda_bytes = cuda_written;
    return 1;
}

static void print_usage(const char* argv0) {
    const char* name = argv0 ? argv0 : "mm_ptx_compile_bench";
    fprintf(stderr,
        "usage: %s [options]\n"
        "  --modules N           Total modules to compile (default: 128)\n"
        "  --kernels N           Kernels per module (default: 16)\n"
        "  --groups-per-kernel N Inject sites per kernel (default: 128)\n"
        "  --tile-size N         Kernel tile size (default: 256)\n"
        "  --embed-dims N        Output dims per inject (default: 1)\n"
        "  --input-dims N        Input dims per inject (default: 1)\n"
        "  --input-type STR      Input type: F32 or U32 (default: F32)\n"
        "  --ptx-instructions-per-program N  Instructions per program (default: 32, includes return)\n"
        "  --program-execution-limit N       Stack-ptx execution limit (default: 100)\n"
        "  --workspace-bytes N   Per-thread workspace bytes (default: auto)\n"
        "  --cores N             OpenMP threads (default: OpenMP runtime)\n"
        "  --seed N              RNG seed (default: 1)\n"
        "  --sm SM               GPU SM for both stages (e.g. 80, sm_80, 8.0)\n"
        "  --sm-ptx SM           GPU SM for CUDA->PTX (NVRTC)\n"
        "  --sm-cubin SM         GPU SM for PTX->cubin (nvPTXCompiler)\n"
        "  --dump-ptx-cu PATH    Write generated CUDA source for PTX path to PATH\n"
        "  --dump-cu PATH        Alias for --dump-ptx-cu\n"
        "  --dump-ptx PATH       Write generated PTX to PATH\n"
        "  --dump-module-ptx PATH   Write injected PTX for one module to PATH\n"
        "  --dump-module-cu PATH    Write inline CUDA for one module to PATH\n"
        "  --dump-module-index N    Module index to dump (default: 0)\n"
        "  --verbose             Print NVRTC info logs\n"
        "  -h, --help            Show this help\n",
        name
    );
}

int main(int argc, char** argv) {
    BenchConfig cfg = {
        .gene_length = 32,
        .execution_limit = 100,
        .cores = 0,
        .seed = 1,
        .verbose = 0,
        .sm_ptx_major = 0,
        .sm_ptx_minor = 0,
        .sm_ptx_set = 0,
        .sm_cubin_major = 0,
        .sm_cubin_minor = 0,
        .sm_cubin_set = 0,
        .num_kernels = 16,
        .groups_per_kernel = 128,
        .tile_size = 256,
        .embed_dims = 1,
        .input_dims = 1,
        .input_type = "F32",
        .modules = 0,
        .modules_set = 0,
        .workspace_bytes = 0,
        .workspace_set = 0,
        .dump_cu_path = NULL,
        .dump_ptx_path = NULL,
        .dump_module_ptx_path = NULL,
        .dump_module_cu_path = NULL,
        .dump_module_idx = 0,
        .dump_module_idx_set = 0
    };

    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
        if (strcmp(arg, "--modules") == 0) {
            if (i + 1 >= argc || !parse_size_t_arg(argv[i + 1], &cfg.modules)) {
                fprintf(stderr, "invalid --modules value\n");
                return 1;
            }
            cfg.modules_set = 1;
            i += 1;
            continue;
        }
        if (strcmp(arg, "--kernels") == 0 || strcmp(arg, "--num-kernels") == 0) {
            if (i + 1 >= argc || !parse_size_t_arg(argv[i + 1], &cfg.num_kernels)) {
                fprintf(stderr, "invalid --kernels value\n");
                return 1;
            }
            i += 1;
            continue;
        }
        if (strcmp(arg, "--groups-per-kernel") == 0) {
            if (i + 1 >= argc || !parse_size_t_arg(argv[i + 1], &cfg.groups_per_kernel)) {
                fprintf(stderr, "invalid --groups-per-kernel value\n");
                return 1;
            }
            i += 1;
            continue;
        }
        if (strcmp(arg, "--tile-size") == 0) {
            if (i + 1 >= argc || !parse_size_t_arg(argv[i + 1], &cfg.tile_size)) {
                fprintf(stderr, "invalid --tile-size value\n");
                return 1;
            }
            i += 1;
            continue;
        }
        if (strcmp(arg, "--embed-dims") == 0) {
            if (i + 1 >= argc || !parse_size_t_arg(argv[i + 1], &cfg.embed_dims)) {
                fprintf(stderr, "invalid --embed-dims value\n");
                return 1;
            }
            i += 1;
            continue;
        }
        if (strcmp(arg, "--input-dims") == 0) {
            if (i + 1 >= argc || !parse_size_t_arg(argv[i + 1], &cfg.input_dims)) {
                fprintf(stderr, "invalid --input-dims value\n");
                return 1;
            }
            i += 1;
            continue;
        }
        if (strcmp(arg, "--input-type") == 0) {
            if (i + 1 >= argc || !parse_input_type(argv[i + 1], cfg.input_type)) {
                fprintf(stderr, "invalid --input-type value\n");
                return 1;
            }
            i += 1;
            continue;
        }
        if (strcmp(arg, "--ptx-instructions-per-program") == 0 || strcmp(arg, "--gene-length") == 0) {
            if (i + 1 >= argc || !parse_size_t_arg(argv[i + 1], &cfg.gene_length)) {
                fprintf(stderr, "invalid --ptx-instructions-per-program value\n");
                return 1;
            }
            i += 1;
            continue;
        }
        if (strcmp(arg, "--program-execution-limit") == 0 || strcmp(arg, "--execution-limit") == 0) {
            if (i + 1 >= argc || !parse_size_t_arg(argv[i + 1], &cfg.execution_limit)) {
                fprintf(stderr, "invalid --program-execution-limit value\n");
                return 1;
            }
            i += 1;
            continue;
        }
        if (strcmp(arg, "--workspace-bytes") == 0) {
            if (i + 1 >= argc || !parse_size_t_arg(argv[i + 1], &cfg.workspace_bytes)) {
                fprintf(stderr, "invalid --workspace-bytes value\n");
                return 1;
            }
            cfg.workspace_set = 1;
            i += 1;
            continue;
        }
        if (strcmp(arg, "--cores") == 0 || strcmp(arg, "--omp-threads") == 0) {
            if (i + 1 >= argc || !parse_size_t_arg(argv[i + 1], &cfg.cores)) {
                fprintf(stderr, "invalid --cores value\n");
                return 1;
            }
            i += 1;
            continue;
        }
        if (strcmp(arg, "--seed") == 0) {
            if (i + 1 >= argc || !parse_u64_arg(argv[i + 1], &cfg.seed)) {
                fprintf(stderr, "invalid --seed value\n");
                return 1;
            }
            i += 1;
            continue;
        }
        if (strcmp(arg, "--sm") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "--sm requires an argument\n");
                return 1;
            }
            if (!parse_sm(argv[i + 1], &cfg.sm_ptx_major, &cfg.sm_ptx_minor)) {
                fprintf(stderr, "invalid --sm value\n");
                return 1;
            }
            cfg.sm_cubin_major = cfg.sm_ptx_major;
            cfg.sm_cubin_minor = cfg.sm_ptx_minor;
            cfg.sm_ptx_set = 1;
            cfg.sm_cubin_set = 1;
            i += 1;
            continue;
        }
        if (strcmp(arg, "--sm-ptx") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "--sm-ptx requires an argument\n");
                return 1;
            }
            if (!parse_sm(argv[i + 1], &cfg.sm_ptx_major, &cfg.sm_ptx_minor)) {
                fprintf(stderr, "invalid --sm-ptx value\n");
                return 1;
            }
            cfg.sm_ptx_set = 1;
            i += 1;
            continue;
        }
        if (strcmp(arg, "--sm-cubin") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "--sm-cubin requires an argument\n");
                return 1;
            }
            if (!parse_sm(argv[i + 1], &cfg.sm_cubin_major, &cfg.sm_cubin_minor)) {
                fprintf(stderr, "invalid --sm-cubin value\n");
                return 1;
            }
            cfg.sm_cubin_set = 1;
            i += 1;
            continue;
        }
        if (strcmp(arg, "--dump-ptx-cu") == 0 || strcmp(arg, "--dump-cu") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "--dump-ptx-cu/--dump-cu requires a path\n");
                return 1;
            }
            cfg.dump_cu_path = argv[i + 1];
            i += 1;
            continue;
        }
        if (strcmp(arg, "--dump-ptx") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "--dump-ptx requires a path\n");
                return 1;
            }
            cfg.dump_ptx_path = argv[i + 1];
            i += 1;
            continue;
        }
        if (strcmp(arg, "--dump-module-ptx") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "--dump-module-ptx requires a path\n");
                return 1;
            }
            cfg.dump_module_ptx_path = argv[i + 1];
            i += 1;
            continue;
        }
        if (strcmp(arg, "--dump-module-cu") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "--dump-module-cu requires a path\n");
                return 1;
            }
            cfg.dump_module_cu_path = argv[i + 1];
            i += 1;
            continue;
        }
        if (strcmp(arg, "--dump-module-index") == 0) {
            if (i + 1 >= argc || !parse_size_t_arg(argv[i + 1], &cfg.dump_module_idx)) {
                fprintf(stderr, "invalid --dump-module-index value\n");
                return 1;
            }
            cfg.dump_module_idx_set = 1;
            i += 1;
            continue;
        }
        if (strcmp(arg, "--verbose") == 0) {
            cfg.verbose = 1;
            continue;
        }

        fprintf(stderr, "unknown argument: %s\n", arg);
        print_usage(argv[0]);
        return 1;
    }

    if (!cfg.sm_ptx_set || !cfg.sm_cubin_set) {
        unsigned int fallback_major = 0;
        unsigned int fallback_minor = 0;
        get_sm_fallback(&fallback_major, &fallback_minor);
        if (!cfg.sm_ptx_set) {
            cfg.sm_ptx_major = fallback_major;
            cfg.sm_ptx_minor = fallback_minor;
        }
        if (!cfg.sm_cubin_set) {
            cfg.sm_cubin_major = fallback_major;
            cfg.sm_cubin_minor = fallback_minor;
        }
    }

    if (cfg.gene_length < 2) {
        fprintf(stderr, "--ptx-instructions-per-program must be >= 2\n");
        return 1;
    }
    if (cfg.num_kernels == 0) {
        fprintf(stderr, "--kernels must be > 0\n");
        return 1;
    }
    if (cfg.groups_per_kernel == 0) {
        fprintf(stderr, "--groups-per-kernel must be > 0\n");
        return 1;
    }
    if (cfg.tile_size == 0) {
        fprintf(stderr, "--tile-size must be > 0\n");
        return 1;
    }
    if (cfg.embed_dims == 0) {
        fprintf(stderr, "--embed-dims must be > 0\n");
        return 1;
    }
    if (cfg.input_dims == 0) {
        fprintf(stderr, "--input-dims must be > 0\n");
        return 1;
    }
    if (cfg.modules_set && cfg.modules == 0) {
        fprintf(stderr, "--modules must be > 0\n");
        return 1;
    }

    size_t programs_per_module = cfg.num_kernels * cfg.groups_per_kernel;
    if (cfg.num_kernels != 0 && programs_per_module / cfg.num_kernels != cfg.groups_per_kernel) {
        fprintf(stderr, "programs per module overflow\n");
        return 1;
    }

#ifdef _OPENMP
    if (cfg.cores > 0) {
        omp_set_num_threads((int)cfg.cores);
    }
#endif

    size_t threads = 1;
#ifdef _OPENMP
    int max_threads = omp_get_max_threads();
    if (max_threads > 0) {
        threads = (size_t)max_threads;
    }
#endif

    if (!cfg.modules_set) {
        cfg.modules = 128;
    }
    if (cfg.modules == 0) {
        fprintf(stderr, "--modules must be > 0\n");
        return 1;
    }

    if (cfg.dump_module_idx_set && !cfg.dump_module_ptx_path && !cfg.dump_module_cu_path) {
        fprintf(stderr, "--dump-module-index requires --dump-module-ptx or --dump-module-cu\n");
        return 1;
    }

    size_t kernel_source_size = 0;
    PtxInjectEmitterResult emit_rc = ptx_inject_emit_cuda(
        (int64_t)cfg.num_kernels,
        (int64_t)cfg.groups_per_kernel,
        (int64_t)cfg.tile_size,
        (int64_t)cfg.embed_dims,
        kKernelNameFormat,
        cfg.input_type,
        (int64_t)cfg.input_dims,
        1,
        NULL,
        0,
        NULL,
        0,
        &kernel_source_size
    );
    if (emit_rc != PTX_INJECT_EMITTER_SUCCESS) {
        fprintf(stderr, "kernel emitter failed (size): %d\n", (int)emit_rc);
        return 1;
    }

    char* kernel_source = (char*)malloc(kernel_source_size + 1);
    if (!kernel_source) {
        fprintf(stderr, "failed to allocate kernel source (%zu bytes)\n", kernel_source_size + 1);
        return 1;
    }
    size_t written = 0;
    emit_rc = ptx_inject_emit_cuda(
        (int64_t)cfg.num_kernels,
        (int64_t)cfg.groups_per_kernel,
        (int64_t)cfg.tile_size,
        (int64_t)cfg.embed_dims,
        kKernelNameFormat,
        cfg.input_type,
        (int64_t)cfg.input_dims,
        1,
        NULL,
        0,
        kernel_source,
        kernel_source_size + 1,
        &written
    );
    if (emit_rc != PTX_INJECT_EMITTER_SUCCESS) {
        fprintf(stderr, "kernel emitter failed (render): %d\n", (int)emit_rc);
        free(kernel_source);
        return 1;
    }
    if (written > kernel_source_size) {
        written = kernel_source_size;
    }
    kernel_source[written] = '\0';

    if (cfg.dump_cu_path) {
        if (!write_file(cfg.dump_cu_path, kernel_source, written)) {
            free(kernel_source);
            return 1;
        }
    }

    size_t kernel_ptx_size = 0;
    char* kernel_ptx = nvrtc_compile_ptx(
        kernel_source,
        cfg.sm_ptx_major,
        cfg.sm_ptx_minor,
        cfg.verbose,
        &kernel_ptx_size
    );
    free(kernel_source);
    if (!kernel_ptx) {
        return 1;
    }

    init_label_width();
    char cpu_label[256];
    if (build_cpu_label(cpu_label, sizeof(cpu_label))) {
        printf("CPU: %s\n", cpu_label);
    }
    if (cfg.verbose) {
        print_kv_size("Kernel source bytes:", kernel_source_size);
        print_kv_size("Kernel PTX bytes:", kernel_ptx_size);
    }
    unsigned int ptx_major = 0;
    unsigned int ptx_minor = 0;
    char version_buf[32];
    if (parse_ptx_version(kernel_ptx, &ptx_major, &ptx_minor)) {
        snprintf(version_buf, sizeof(version_buf), "%u.%u", ptx_major, ptx_minor);
        print_kv_str("PTX version:", version_buf);
    } else {
        print_kv_str("PTX version:", "unknown");
    }
    unsigned int nvptx_major = 0;
    unsigned int nvptx_minor = 0;
    if (nvPTXCompilerGetVersion(&nvptx_major, &nvptx_minor) == NVPTXCOMPILE_SUCCESS) {
        snprintf(version_buf, sizeof(version_buf), "%u.%u", nvptx_major, nvptx_minor);
        print_kv_str("nvPTXCompiler API:", version_buf);
    } else {
        print_kv_str("nvPTXCompiler API:", "unknown");
    }
    snprintf(version_buf, sizeof(version_buf), "%u.%u", cfg.sm_ptx_major, cfg.sm_ptx_minor);
    print_kv_str("SM (PTX):", version_buf);
    snprintf(version_buf, sizeof(version_buf), "%u.%u", cfg.sm_cubin_major, cfg.sm_cubin_minor);
    print_kv_str("SM (cubin):", version_buf);
    if (cfg.dump_cu_path) {
        print_kv_str("Dumped PTX CU:", cfg.dump_cu_path);
    }
    if (cfg.dump_ptx_path) {
        if (!write_file(cfg.dump_ptx_path, kernel_ptx, kernel_ptx_size)) {
            free(kernel_ptx);
            return 1;
        }
        print_kv_str("Dumped PTX:", cfg.dump_ptx_path);
    }

    PtxInjectHandle ptx_inject = NULL;
    if (ptx_inject_create(&ptx_inject, kernel_ptx) != PTX_INJECT_SUCCESS) {
        fprintf(stderr, "ptx_inject_create failed\n");
        free(kernel_ptx);
        return 1;
    }

    BenchContext ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.programs_per_module = programs_per_module;
    ctx.stack_info = &stack_ptx_stack_info;
    ctx.compiler_info.max_ast_size = 100;
    ctx.compiler_info.max_ast_to_visit_stack_depth = 20;
    ctx.compiler_info.stack_size = 128;
    ctx.compiler_info.max_frame_depth = 4;
    ctx.compiler_info.store_size = 16;
    ctx.execution_limit = cfg.execution_limit;

    const size_t min_ast = cfg.gene_length * 2;
    if (ctx.compiler_info.max_ast_size < min_ast) {
        ctx.compiler_info.max_ast_size = min_ast;
    }
    if (ctx.compiler_info.max_ast_to_visit_stack_depth < cfg.gene_length) {
        ctx.compiler_info.max_ast_to_visit_stack_depth = cfg.gene_length;
    }
    if (ctx.compiler_info.stack_size < cfg.gene_length) {
        ctx.compiler_info.stack_size = cfg.gene_length;
    }

    if (stack_ptx_compile_workspace_size(&ctx.compiler_info, ctx.stack_info, &ctx.stack_ptx_workspace_size) !=
        STACK_PTX_SUCCESS) {
        fprintf(stderr, "stack_ptx_compile_workspace_size failed\n");
        ptx_inject_destroy(ptx_inject);
        free(kernel_ptx);
        return 1;
    }

    if (!init_layout(ptx_inject, &ctx.layout)) {
        ptx_inject_destroy(ptx_inject);
        free(kernel_ptx);
        return 1;
    }

    if (!build_reverse_indices(ptx_inject, ctx.programs_per_module, &ctx.reverse_indices)) {
        ptx_inject_destroy(ptx_inject);
        free(kernel_ptx);
        free_layout(&ctx.layout);
        return 1;
    }
    ctx.num_injects = ctx.layout.num_injects;

    if (ctx.num_injects != ctx.programs_per_module) {
        fprintf(stderr, "inject sites (%zu) != programs per module (%zu)\n",
            ctx.num_injects, ctx.programs_per_module);
        ptx_inject_destroy(ptx_inject);
        free(kernel_ptx);
        free_layout(&ctx.layout);
        free(ctx.reverse_indices);
        return 1;
    }

    if (!cfg.workspace_set) {
        cfg.workspace_bytes = default_workspace_bytes(ctx.stack_ptx_workspace_size, cfg.gene_length, programs_per_module);
    }
    if (cfg.workspace_bytes < ctx.stack_ptx_workspace_size) {
        fprintf(stderr, "workspace bytes too small (need at least %zu)\n", ctx.stack_ptx_workspace_size);
        ptx_inject_destroy(ptx_inject);
        free(kernel_ptx);
        free_layout(&ctx.layout);
        free(ctx.reverse_indices);
        return 1;
    }
    ctx.workspace_bytes = cfg.workspace_bytes;

    print_kv_size("Modules:", cfg.modules);
    print_kv_size("Programs per module:", programs_per_module);
    print_kv_size("Total programs:", cfg.modules * programs_per_module);
    print_kv_size("Kernels:", cfg.num_kernels);
    print_kv_size("Groups per kernel:", cfg.groups_per_kernel);
    print_kv_size("Tile size:", cfg.tile_size);
    print_kv_size("Embed dims:", cfg.embed_dims);
    print_kv_size("Input dims:", cfg.input_dims);
    print_kv_str("Input type:", cfg.input_type);
    print_kv_size("PTX instructions per program:", cfg.gene_length);
    print_kv_size("Program execution limit:", cfg.execution_limit);
    print_kv_size("Workspace bytes/thread:", cfg.workspace_bytes);
    print_kv_size("Threads:", threads);

    if (cfg.dump_module_idx_set) {
        print_kv_size("Dumped module index:", cfg.dump_module_idx);
    }

    size_t total_programs = cfg.modules * programs_per_module;
    if (cfg.modules != 0 && total_programs / cfg.modules != programs_per_module) {
        fprintf(stderr, "total programs overflow\n");
        ptx_inject_destroy(ptx_inject);
        free(kernel_ptx);
        free_layout(&ctx.layout);
        free(ctx.reverse_indices);
        return 1;
    }
    if (cfg.gene_length > 0 && total_programs > SIZE_MAX / cfg.gene_length) {
        fprintf(stderr, "population size overflow\n");
        ptx_inject_destroy(ptx_inject);
        free(kernel_ptx);
        free_layout(&ctx.layout);
        free(ctx.reverse_indices);
        return 1;
    }
    const size_t population_instructions = total_programs * cfg.gene_length;

    StackPtxInstruction* population = (StackPtxInstruction*)malloc(
        population_instructions * sizeof(StackPtxInstruction)
    );
    if (!population) {
        fprintf(stderr, "failed to allocate population\n");
        ptx_inject_destroy(ptx_inject);
        free(kernel_ptx);
        free_layout(&ctx.layout);
        free(ctx.reverse_indices);
        return 1;
    }

    int input_stack_is_f32 = 1;
    if (ctx.layout.input_count > 0) {
        size_t stack_idx = ctx.layout.registers[0].stack_idx;
        if (stack_idx == STACK_PTX_STACK_TYPE_F32) {
            input_stack_is_f32 = 1;
        } else if (stack_idx == STACK_PTX_STACK_TYPE_U32) {
            input_stack_is_f32 = 0;
        } else {
            fprintf(stderr, "unsupported input stack index: %zu\n", stack_idx);
            free(population);
            ptx_inject_destroy(ptx_inject);
            free(kernel_ptx);
            free_layout(&ctx.layout);
            free(ctx.reverse_indices);
            return 1;
        }
    }

    generate_programs(population, total_programs, cfg.gene_length, ctx.layout.input_count, input_stack_is_f32, cfg.seed);

    ModuleOutput* module_outputs = (ModuleOutput*)calloc(cfg.modules, sizeof(*module_outputs));
    if (!module_outputs) {
        fprintf(stderr, "failed to allocate module outputs\n");
        free(population);
        ptx_inject_destroy(ptx_inject);
        free(kernel_ptx);
        free_layout(&ctx.layout);
        free(ctx.reverse_indices);
        return 1;
    }

    int build_ok = 1;
#pragma omp parallel
    {
        ThreadScratch scratch = { 0 };
        scratch.arena_capacity = ctx.workspace_bytes;
        scratch.arena_base = (unsigned char*)malloc(ctx.workspace_bytes);
        scratch.program_stubs = (const char**)malloc(ctx.programs_per_module * sizeof(*scratch.program_stubs));
        scratch.inject_stubs = (const char**)malloc(ctx.num_injects * sizeof(*scratch.inject_stubs));
        if (!scratch.arena_base || !scratch.program_stubs || !scratch.inject_stubs) {
#pragma omp atomic write
            build_ok = 0;
        }

        Arena arena = { scratch.arena_base, scratch.arena_capacity, 0 };

#pragma omp for schedule(static)
        for (size_t module_idx = 0; module_idx < cfg.modules; ++module_idx) {
            ModuleOutput out = { 0 };
            if (!build_ok) {
                continue;
            }
            if (!build_module_outputs(&ctx, &cfg, ptx_inject, population, cfg.gene_length, module_idx, &scratch, &arena, &out)) {
#pragma omp atomic write
                build_ok = 0;
                continue;
            }
            module_outputs[module_idx] = out;
        }

        free(scratch.inject_stubs);
        free(scratch.program_stubs);
        free(scratch.arena_base);
    }

    ptx_inject_destroy(ptx_inject);
    free(kernel_ptx);

    if (!build_ok) {
        fprintf(stderr, "failed to build module sources\n");
        for (size_t i = 0; i < cfg.modules; ++i) {
            free(module_outputs[i].ptx);
            free(module_outputs[i].cuda);
        }
        free(module_outputs);
        free(population);
        free_layout(&ctx.layout);
        free(ctx.reverse_indices);
        return 1;
    }

    size_t dump_idx = cfg.dump_module_idx_set ? cfg.dump_module_idx : 0;
    if (dump_idx >= cfg.modules) {
        fprintf(stderr, "dump module index out of range\n");
        for (size_t i = 0; i < cfg.modules; ++i) {
            free(module_outputs[i].ptx);
            free(module_outputs[i].cuda);
        }
        free(module_outputs);
        free(population);
        free_layout(&ctx.layout);
        free(ctx.reverse_indices);
        return 1;
    }
    if (cfg.dump_module_ptx_path) {
        if (!write_file(cfg.dump_module_ptx_path, module_outputs[dump_idx].ptx, module_outputs[dump_idx].ptx_bytes)) {
            for (size_t i = 0; i < cfg.modules; ++i) {
                free(module_outputs[i].ptx);
                free(module_outputs[i].cuda);
            }
            free(module_outputs);
            free(population);
            free_layout(&ctx.layout);
            free(ctx.reverse_indices);
            return 1;
        }
        print_kv_str("Dumped module PTX:", cfg.dump_module_ptx_path);
    }
    if (cfg.dump_module_cu_path) {
        if (!write_file(cfg.dump_module_cu_path, module_outputs[dump_idx].cuda, module_outputs[dump_idx].cuda_bytes)) {
            for (size_t i = 0; i < cfg.modules; ++i) {
                free(module_outputs[i].ptx);
                free(module_outputs[i].cuda);
            }
            free(module_outputs);
            free(population);
            free_layout(&ctx.layout);
            free(ctx.reverse_indices);
            return 1;
        }
        print_kv_str("Dumped module CU:", cfg.dump_module_cu_path);
    }

    double ptx_start = now_ms();
    int ptx_ok = 1;
#pragma omp parallel for schedule(static)
    for (size_t module_idx = 0; module_idx < cfg.modules; ++module_idx) {
        if (!ptx_ok) {
            continue;
        }
        void* cubin = NULL;
        size_t cubin_size = 0;
        if (!nvptx_compile_cubin(
                module_outputs[module_idx].ptx,
                module_outputs[module_idx].ptx_bytes,
                cfg.sm_cubin_major,
                cfg.sm_cubin_minor,
                cfg.verbose,
                &cubin,
                &cubin_size
            )) {
#pragma omp atomic write
            ptx_ok = 0;
            continue;
        }
        free(cubin);
    }
    double ptx_end = now_ms();

    if (!ptx_ok) {
        fprintf(stderr, "ptx_inject compile failed\n");
        for (size_t i = 0; i < cfg.modules; ++i) {
            free(module_outputs[i].ptx);
            free(module_outputs[i].cuda);
        }
        free(module_outputs);
        free(population);
        free_layout(&ctx.layout);
        free(ctx.reverse_indices);
        return 1;
    }

    double cuda_start = now_ms();
    int cuda_ok = 1;
#pragma omp parallel for schedule(static)
    for (size_t module_idx = 0; module_idx < cfg.modules; ++module_idx) {
        if (!cuda_ok) {
            continue;
        }
        void* cubin = NULL;
        size_t cubin_size = 0;
        if (!nvrtc_compile_cubin(
                module_outputs[module_idx].cuda,
                cfg.sm_cubin_major,
                cfg.sm_cubin_minor,
                cfg.verbose,
                &cubin,
                &cubin_size
            )) {
#pragma omp atomic write
            cuda_ok = 0;
            continue;
        }
        free(cubin);
    }
    double cuda_end = now_ms();

    if (!cuda_ok) {
        fprintf(stderr, "cuda inline compile failed\n");
        for (size_t i = 0; i < cfg.modules; ++i) {
            free(module_outputs[i].ptx);
            free(module_outputs[i].cuda);
        }
        free(module_outputs);
        free(population);
        free_layout(&ctx.layout);
        free(ctx.reverse_indices);
        return 1;
    }

    double ptx_ms = ptx_end - ptx_start;
    double cuda_ms = cuda_end - cuda_start;
    double baseline_ms = cuda_ms > 0.0 ? cuda_ms : 1.0;

    double ptx_speedup = baseline_ms / (ptx_ms > 0.0 ? ptx_ms : baseline_ms);

    printf("\nResults (threads=%zu, compile only, speedup vs cuda_inline):\n", threads);
    printf("  cuda_inline     total_ms=%8.2f  us_per_module=%10.2f  1.00x\n",
        cuda_ms,
        (cuda_ms * 1000.0) / (double)cfg.modules);
    printf("  ptx_inject      total_ms=%8.2f  us_per_module=%10.2f  %5.2fx\n",
        ptx_ms,
        (ptx_ms * 1000.0) / (double)cfg.modules,
        ptx_speedup);

    print_compile_stats("CUDA inline:", cuda_ms, cfg.modules, total_programs, threads);
    print_compile_stats("PTX inject:", ptx_ms, cfg.modules, total_programs, threads);

    for (size_t i = 0; i < cfg.modules; ++i) {
        free(module_outputs[i].ptx);
        free(module_outputs[i].cuda);
    }
    free(module_outputs);
    free(population);
    free_layout(&ctx.layout);
    free(ctx.reverse_indices);
    return 0;
}
