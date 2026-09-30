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
#include "stack_ptx_inject_kernel_gen.h"

#include <inttypes.h>
#include <limits.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#if defined(STACK_PTX_KERNEL_GEN_DEBUG) || defined(ELITE_NLE_GEN_KERNEL_DEBUG)
#include <assert.h>
#include <stdlib.h>
#endif

typedef enum {
    PTX_ARG_IN,
    PTX_ARG_OUT,
    PTX_ARG_MOD
} PtxArgKind;

typedef struct {
    PtxArgKind kind;
    const char* type_name;
    const char* name;
    const char* expr;
} PtxArg;

typedef struct {
    const char* type_name;
    const char* reg_suffix;
    const char* mov_postfix;
    const char* constraint;
    const char* bind_kind;
} PtxTypeInfo;

typedef struct {
    char* buffer;
    size_t buffer_size;
    size_t offset;
    StackPtxKernelGenResult status;
} GenBuffer;

#define STACK_PTX_KERNEL_GEN_MAX_ARGS 1024u
#define STACK_PTX_KERNEL_GEN_MAX_EMBED_DIMS (STACK_PTX_KERNEL_GEN_MAX_ARGS - 1u)
#define STACK_PTX_KERNEL_GEN_NAME_STRIDE 32u

#if defined(STACK_PTX_KERNEL_GEN_DEBUG) || defined(ELITE_NLE_GEN_KERNEL_DEBUG)
static const char* stack_ptx_kernel_gen_result_to_string(StackPtxKernelGenResult result) {
    switch (result) {
        case STACK_PTX_KERNEL_GEN_SUCCESS:
            return "STACK_PTX_KERNEL_GEN_SUCCESS";
        case STACK_PTX_KERNEL_GEN_ERROR_INVALID_VALUE:
            return "STACK_PTX_KERNEL_GEN_ERROR_INVALID_VALUE";
        case STACK_PTX_KERNEL_GEN_ERROR_INSUFFICIENT_BUFFER:
            return "STACK_PTX_KERNEL_GEN_ERROR_INSUFFICIENT_BUFFER";
        case STACK_PTX_KERNEL_GEN_ERROR_OUT_OF_MEMORY:
            return "STACK_PTX_KERNEL_GEN_ERROR_OUT_OF_MEMORY";
        case STACK_PTX_KERNEL_GEN_ERROR_FORMAT:
            return "STACK_PTX_KERNEL_GEN_ERROR_FORMAT";
    }
    return "STACK_PTX_KERNEL_GEN_ERROR_UNKNOWN";
}
#endif

#if defined(STACK_PTX_KERNEL_GEN_DEBUG) || defined(ELITE_NLE_GEN_KERNEL_DEBUG)
#define _STACK_PTX_KERNEL_GEN_ERROR(ans)                                                        \
    do {                                                                                        \
        StackPtxKernelGenResult _result = (ans);                                                \
        const char* error_name = stack_ptx_kernel_gen_result_to_string(_result);                \
        fprintf(stderr, "STACK_PTX_KERNEL_GEN_ERROR: %s \n  %s %d\n",                            \
                error_name, __FILE__, __LINE__);                                                \
        assert(0);                                                                              \
        exit(1);                                                                                \
    } while (0)

#define _STACK_PTX_KERNEL_GEN_CHECK_RET(ans)                                                     \
    do {                                                                                        \
        StackPtxKernelGenResult _result = (ans);                                                \
        if (_result != STACK_PTX_KERNEL_GEN_SUCCESS) {                                          \
            const char* error_name = stack_ptx_kernel_gen_result_to_string(_result);            \
            fprintf(stderr, "STACK_PTX_KERNEL_GEN_CHECK: %s \n  %s %d\n",                        \
                    error_name, __FILE__, __LINE__);                                            \
            assert(0);                                                                          \
            exit(1);                                                                            \
            return _result;                                                                     \
        }                                                                                       \
    } while (0)
#else
#define _STACK_PTX_KERNEL_GEN_ERROR(ans)                                                        \
    do {                                                                                        \
        StackPtxKernelGenResult _result = (ans);                                                \
        return _result;                                                                         \
    } while (0)

#define _STACK_PTX_KERNEL_GEN_CHECK_RET(ans)                                                     \
    do {                                                                                        \
        StackPtxKernelGenResult _result = (ans);                                                \
        if (_result != STACK_PTX_KERNEL_GEN_SUCCESS) return _result;                            \
    } while (0)
#endif

static StackPtxKernelGenResult gen_write(GenBuffer* gb, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);

    char* out = NULL;
    size_t remaining = 0;
    if (gb->buffer && gb->status == STACK_PTX_KERNEL_GEN_SUCCESS) {
        if (gb->offset < gb->buffer_size) {
            out = gb->buffer + gb->offset;
            remaining = gb->buffer_size - gb->offset;
        }
    }

    int needed_int = vsnprintf(out, remaining, fmt, args);
    va_end(args);

    if (needed_int < 0) {
        return STACK_PTX_KERNEL_GEN_ERROR_FORMAT;
    }

    size_t needed = (size_t)needed_int;
    if (SIZE_MAX - gb->offset < needed) {
        return STACK_PTX_KERNEL_GEN_ERROR_INVALID_VALUE;
    }

    if (gb->buffer && gb->status == STACK_PTX_KERNEL_GEN_SUCCESS &&
        gb->offset + needed > gb->buffer_size) {
        gb->status = STACK_PTX_KERNEL_GEN_ERROR_INSUFFICIENT_BUFFER;
    }

    gb->offset += needed;
    return STACK_PTX_KERNEL_GEN_SUCCESS;
}

static const PtxTypeInfo kPtxTypeInfo[] = {
    { "F16",   "b16", "b16", "h", "U16" },
    { "F16X2", "b32", "b32", "r", "U32" },
    { "S32",   "s32", "s32", "r", "ID" },
    { "U32",   "u32", "u32", "r", "ID" },
    { "F32",   "f32", "f32", "f", "ID" },
    { "B32",   "b32", "b32", "r", "ID" }
};

static const size_t kPtxTypeInfoCount = sizeof(kPtxTypeInfo) / sizeof(kPtxTypeInfo[0]);
static const char* kDefaultKernelNameFormat = "kernel_%06zu";

static const PtxTypeInfo* ptx_type_info(const char* type_name) {
    for (size_t i = 0; i < kPtxTypeInfoCount; ++i) {
        if (strcmp(kPtxTypeInfo[i].type_name, type_name) == 0) {
            return &kPtxTypeInfo[i];
        }
    }
    return NULL;
}

static const char* ptx_type_c_name(const char* type_name) {
    if (!type_name) {
        return NULL;
    }
    if (strcmp(type_name, "U32") == 0 || strcmp(type_name, "B32") == 0) {
        return "uint32_t";
    }
    if (strcmp(type_name, "S32") == 0) {
        return "int";
    }
    if (strcmp(type_name, "F32") == 0) {
        return "float";
    }
    return NULL;
}

static char kind_char(PtxArgKind kind) {
    switch (kind) {
        case PTX_ARG_MOD: return 'm';
        case PTX_ARG_OUT: return 'o';
        case PTX_ARG_IN: return 'i';
    }
    return '?';
}

static StackPtxKernelGenResult build_bind_expr(
    const char* bind_kind,
    const char* expr,
    char* out,
    size_t out_size
) {
    if (strcmp(bind_kind, "ID") == 0) {
        return snprintf(out, out_size, "%s", expr) >= 0
            ? STACK_PTX_KERNEL_GEN_SUCCESS
            : STACK_PTX_KERNEL_GEN_ERROR_FORMAT;
    }
    if (strcmp(bind_kind, "U16") == 0) {
        return snprintf(out, out_size, "(*reinterpret_cast<unsigned short*>(& (%s) ))", expr) >= 0
            ? STACK_PTX_KERNEL_GEN_SUCCESS
            : STACK_PTX_KERNEL_GEN_ERROR_FORMAT;
    }
    if (strcmp(bind_kind, "U32") == 0) {
        return snprintf(out, out_size, "(*reinterpret_cast<unsigned int  *>(& (%s) ))", expr) >= 0
            ? STACK_PTX_KERNEL_GEN_SUCCESS
            : STACK_PTX_KERNEL_GEN_ERROR_FORMAT;
    }
    return STACK_PTX_KERNEL_GEN_ERROR_FORMAT;
}

static StackPtxKernelGenResult emit_ptx_inject_asm(
    GenBuffer* gb,
    const char* site_name,
    const PtxArg* args,
    size_t num_args,
    size_t mod_count,
    size_t out_count,
    size_t in_count,
    const char* indent
) {
    if (!args || num_args == 0) {
        _STACK_PTX_KERNEL_GEN_ERROR(STACK_PTX_KERNEL_GEN_ERROR_INVALID_VALUE);
    }
    if (num_args > STACK_PTX_KERNEL_GEN_MAX_ARGS) {
        _STACK_PTX_KERNEL_GEN_ERROR(STACK_PTX_KERNEL_GEN_ERROR_INVALID_VALUE);
    }
    if (mod_count + out_count + in_count != num_args) {
        _STACK_PTX_KERNEL_GEN_ERROR(STACK_PTX_KERNEL_GEN_ERROR_INVALID_VALUE);
    }

    _STACK_PTX_KERNEL_GEN_CHECK_RET(gen_write(gb, "%sasm (\n", indent));
    _STACK_PTX_KERNEL_GEN_CHECK_RET(gen_write(gb, "%s    \"{\\n\\t\"\n", indent));

    for (size_t i = 0; i < num_args; ++i) {
        const PtxArg* arg = &args[i];
        const PtxTypeInfo* info = ptx_type_info(arg->type_name);
        if (!info) {
            _STACK_PTX_KERNEL_GEN_ERROR(STACK_PTX_KERNEL_GEN_ERROR_INVALID_VALUE);
        }
        _STACK_PTX_KERNEL_GEN_CHECK_RET(
            gen_write(gb, "%s    \".reg .%s %%%%_x%zu;\\n\\t\"\n", indent, info->reg_suffix, i)
        );
    }

    for (size_t i = 0; i < mod_count; ++i) {
        const PtxArg* arg = &args[i];
        const PtxTypeInfo* info = ptx_type_info(arg->type_name);
        _STACK_PTX_KERNEL_GEN_CHECK_RET(
            gen_write(gb, "%s    \"mov.%s %%%%_x%zu, %%%zu;\\n\\t\"\n", indent, info->mov_postfix, i, i)
        );
    }
    for (size_t i = 0; i < in_count; ++i) {
        size_t arg_idx = mod_count + out_count + i;
        const PtxArg* arg = &args[arg_idx];
        const PtxTypeInfo* info = ptx_type_info(arg->type_name);
        _STACK_PTX_KERNEL_GEN_CHECK_RET(
            gen_write(gb, "%s    \"mov.%s %%%%_x%zu, %%%zu;\\n\\t\"\n", indent, info->mov_postfix, arg_idx, arg_idx)
        );
    }

    _STACK_PTX_KERNEL_GEN_CHECK_RET(
        gen_write(gb, "%s    \"// PTX_INJECT_START %s\\n\\t\"\n", indent, site_name)
    );

    for (size_t i = 0; i < num_args; ++i) {
        const PtxArg* arg = &args[i];
        const PtxTypeInfo* info = ptx_type_info(arg->type_name);
        _STACK_PTX_KERNEL_GEN_CHECK_RET(
            gen_write(
                gb,
                "%s    \"// _x%zu %c %s %s %s\\n\\t\"\n",
                indent,
                i,
                kind_char(arg->kind),
                info->reg_suffix,
                arg->type_name,
                arg->name
            )
        );
    }
    _STACK_PTX_KERNEL_GEN_CHECK_RET(gen_write(gb, "%s    \"// PTX_INJECT_END\\n\\t\"\n", indent));

    for (size_t i = 0; i < mod_count; ++i) {
        const PtxArg* arg = &args[i];
        const PtxTypeInfo* info = ptx_type_info(arg->type_name);
        _STACK_PTX_KERNEL_GEN_CHECK_RET(
            gen_write(gb, "%s    \"mov.%s %%%zu, %%%%_x%zu;\\n\\t\"\n", indent, info->mov_postfix, i, i)
        );
    }
    for (size_t i = 0; i < out_count; ++i) {
        size_t arg_idx = mod_count + i;
        const PtxArg* arg = &args[arg_idx];
        const PtxTypeInfo* info = ptx_type_info(arg->type_name);
        _STACK_PTX_KERNEL_GEN_CHECK_RET(
            gen_write(gb, "%s    \"mov.%s %%%zu, %%%%_x%zu;\\n\\t\"\n", indent, info->mov_postfix, arg_idx, arg_idx)
        );
    }

    _STACK_PTX_KERNEL_GEN_CHECK_RET(gen_write(gb, "%s    \"}\"\n", indent));

    if (mod_count + out_count > 0) {
        bool first = true;
        _STACK_PTX_KERNEL_GEN_CHECK_RET(gen_write(gb, "%s    : ", indent));
        for (size_t i = 0; i < mod_count; ++i) {
            const PtxArg* arg = &args[i];
            const PtxTypeInfo* info = ptx_type_info(arg->type_name);
            char expr_buf[256];
            _STACK_PTX_KERNEL_GEN_CHECK_RET(
                build_bind_expr(info->bind_kind, arg->expr, expr_buf, sizeof(expr_buf))
            );
            if (!first) {
                _STACK_PTX_KERNEL_GEN_CHECK_RET(gen_write(gb, ", "));
            }
            first = false;
            _STACK_PTX_KERNEL_GEN_CHECK_RET(gen_write(gb, "\"+%s\"(%s)", info->constraint, expr_buf));
        }
        for (size_t i = 0; i < out_count; ++i) {
            const PtxArg* arg = &args[mod_count + i];
            const PtxTypeInfo* info = ptx_type_info(arg->type_name);
            char expr_buf[256];
            _STACK_PTX_KERNEL_GEN_CHECK_RET(
                build_bind_expr(info->bind_kind, arg->expr, expr_buf, sizeof(expr_buf))
            );
            if (!first) {
                _STACK_PTX_KERNEL_GEN_CHECK_RET(gen_write(gb, ", "));
            }
            first = false;
            _STACK_PTX_KERNEL_GEN_CHECK_RET(gen_write(gb, "\"=%s\"(%s)", info->constraint, expr_buf));
        }
        _STACK_PTX_KERNEL_GEN_CHECK_RET(gen_write(gb, "\n"));
    }

    if (in_count > 0) {
        bool first = true;
        if (mod_count + out_count > 0) {
            _STACK_PTX_KERNEL_GEN_CHECK_RET(gen_write(gb, "%s    : ", indent));
        } else {
            _STACK_PTX_KERNEL_GEN_CHECK_RET(gen_write(gb, "%s    : : ", indent));
        }
        for (size_t i = 0; i < in_count; ++i) {
            const PtxArg* arg = &args[mod_count + out_count + i];
            const PtxTypeInfo* info = ptx_type_info(arg->type_name);
            char expr_buf[256];
            _STACK_PTX_KERNEL_GEN_CHECK_RET(
                build_bind_expr(info->bind_kind, arg->expr, expr_buf, sizeof(expr_buf))
            );
            if (!first) {
                _STACK_PTX_KERNEL_GEN_CHECK_RET(gen_write(gb, ", "));
            }
            first = false;
            _STACK_PTX_KERNEL_GEN_CHECK_RET(gen_write(gb, "\"%s\"(%s)", info->constraint, expr_buf));
        }
        _STACK_PTX_KERNEL_GEN_CHECK_RET(gen_write(gb, "\n"));
    }

    _STACK_PTX_KERNEL_GEN_CHECK_RET(gen_write(gb, "%s);\n", indent));
    return STACK_PTX_KERNEL_GEN_SUCCESS;
}

static StackPtxKernelGenResult emit_header(
    GenBuffer* gb,
    int64_t num_kernels,
    int64_t groups_per_kernel,
    int64_t tile_size,
    int64_t embed_dims,
    int64_t input_dims
) {
    int64_t indivs_per_kernel = groups_per_kernel;
    int64_t total_indivs = num_kernels * indivs_per_kernel;

    return gen_write(
        gb,
        "// Auto-generated\n"
        "// Configuration:\n"
        "//   NUM_KERNELS        = %" PRId64 "\n"
        "//   GROUPS_PER_KERNEL  = %" PRId64 "   // cases for blockIdx.x\n"
        "//   INDIVS_PER_KERNEL  = GROUPS_PER_KERNEL = %" PRId64 "\n"
        "//   TOTAL_INDIVS       = NUM_KERNELS * INDIVS_PER_KERNEL = %" PRId64 "\n"
        "//   TILE_SIZE          = %" PRId64 "\n"
        "//   EMBED_DIMS         = %" PRId64 "\n"
        "//   INPUT_DIMS         = %" PRId64 "\n"
        "\n"
        "typedef long long int64_t;\n"
        "typedef unsigned int uint32_t;\n"
        "typedef unsigned long long uint64_t;\n"
        "\n"
        "#ifndef TILE_SIZE\n"
        "#define TILE_SIZE         %" PRId64 "\n"
        "#endif\n"
        "\n"
        "#ifndef GROUPS_PER_KERNEL\n"
        "#define GROUPS_PER_KERNEL %" PRId64 "\n"
        "#endif\n"
        "\n"
        "#ifndef NUM_KERNELS\n"
        "#define NUM_KERNELS       %" PRId64 "\n"
        "#endif\n"
        "\n",
        num_kernels,
        groups_per_kernel,
        indivs_per_kernel,
        total_indivs,
        tile_size,
        embed_dims,
        input_dims,
        tile_size,
        groups_per_kernel,
        num_kernels
    );
}

static StackPtxKernelGenResult emit_case_block(
    GenBuffer* gb,
    int64_t group,
    int64_t global_base,
    const PtxArg* inject_args,
    size_t num_args,
    size_t mod_count,
    size_t out_count,
    size_t in_count
) {
    _STACK_PTX_KERNEL_GEN_CHECK_RET(gen_write(gb, "        case %" PRId64 ": {\n", group));

    int64_t local_idx = group;
    int64_t global_indiv = global_base + local_idx;
    char site_name[64];
    snprintf(site_name, sizeof(site_name), "func_%" PRId64, global_indiv);
    _STACK_PTX_KERNEL_GEN_CHECK_RET(
        gen_write(gb, "            local_idx = %" PRId64 ";\n", local_idx)
    );
    _STACK_PTX_KERNEL_GEN_CHECK_RET(
        emit_ptx_inject_asm(
            gb,
            site_name,
            inject_args,
            num_args,
            mod_count,
            out_count,
            in_count,
            "            "
        )
    );

    _STACK_PTX_KERNEL_GEN_CHECK_RET(gen_write(gb, "            break;\n"));
    _STACK_PTX_KERNEL_GEN_CHECK_RET(gen_write(gb, "        }\n"));
    return STACK_PTX_KERNEL_GEN_SUCCESS;
}

static StackPtxKernelGenResult emit_kernel(
    GenBuffer* gb,
    int64_t kernel_index,
    int64_t groups_per_kernel,
    int64_t embed_dims,
    int64_t input_dims,
    const char* input_type_name,
    const char* kernel_name_format
) {
    if (kernel_index < 0) {
        _STACK_PTX_KERNEL_GEN_ERROR(STACK_PTX_KERNEL_GEN_ERROR_INVALID_VALUE);
    }

    char kernel_name[128];
    snprintf(kernel_name, sizeof(kernel_name), kernel_name_format, (size_t)kernel_index);

    const char* resolved_input_type = (input_type_name && input_type_name[0])
        ? input_type_name
        : "U32";
    const PtxTypeInfo* input_info = ptx_type_info(resolved_input_type);
    const char* input_c_type = ptx_type_c_name(resolved_input_type);
    if (!input_info || !input_c_type) {
        _STACK_PTX_KERNEL_GEN_ERROR(STACK_PTX_KERNEL_GEN_ERROR_INVALID_VALUE);
    }
    if (input_dims < 1) {
        _STACK_PTX_KERNEL_GEN_ERROR(STACK_PTX_KERNEL_GEN_ERROR_INVALID_VALUE);
    }

    _STACK_PTX_KERNEL_GEN_CHECK_RET(gen_write(
            gb,
            "extern \"C\"\n"
            "__global__\n"
            "void %s(\n"
            "    const int64_t  num_train,\n"
            "    const %s* __restrict__ data,\n"
            "    int64_t        ld_input,\n"
            "    float* __restrict__ embed,\n"
            "    int64_t        ld_embed,\n"
            "    int64_t        batch_stride\n"
            ") {\n"
            "    const int group = (int)blockIdx.x;\n"
            "    const int64_t tile_start = (int64_t)blockIdx.y * (int64_t)TILE_SIZE;\n"
            "    const int64_t tile_end   = (tile_start + (int64_t)TILE_SIZE < num_train)\n"
            "                               ? (tile_start + (int64_t)TILE_SIZE)\n"
            "                               : num_train;\n"
            "\n"
            "    for (int64_t i = tile_start + (int64_t)threadIdx.x; i < tile_end; i += (int64_t)blockDim.x) {\n",
            kernel_name,
            input_c_type
        ));

    int64_t indivs_per_kernel = groups_per_kernel;
    int64_t global_base = kernel_index * indivs_per_kernel;

    if (embed_dims < 1) {
        _STACK_PTX_KERNEL_GEN_ERROR(STACK_PTX_KERNEL_GEN_ERROR_INVALID_VALUE);
    }
    if (embed_dims > (int64_t)STACK_PTX_KERNEL_GEN_MAX_EMBED_DIMS) {
        _STACK_PTX_KERNEL_GEN_ERROR(STACK_PTX_KERNEL_GEN_ERROR_INVALID_VALUE);
    }
    if (input_dims > (int64_t)STACK_PTX_KERNEL_GEN_MAX_ARGS) {
        _STACK_PTX_KERNEL_GEN_ERROR(STACK_PTX_KERNEL_GEN_ERROR_INVALID_VALUE);
    }

    size_t embed_dims_size = (size_t)embed_dims;
    size_t mod_count = 0;
    size_t out_count = embed_dims_size;
    size_t in_count = (size_t)input_dims;
    size_t num_args = mod_count + out_count + in_count;
    size_t output_count = mod_count + out_count;
    if (num_args > STACK_PTX_KERNEL_GEN_MAX_ARGS) {
        _STACK_PTX_KERNEL_GEN_ERROR(STACK_PTX_KERNEL_GEN_ERROR_INVALID_VALUE);
    }
    PtxArg inject_args[STACK_PTX_KERNEL_GEN_MAX_ARGS];
    char output_names[STACK_PTX_KERNEL_GEN_MAX_EMBED_DIMS][STACK_PTX_KERNEL_GEN_NAME_STRIDE];
    char input_names[STACK_PTX_KERNEL_GEN_MAX_ARGS][STACK_PTX_KERNEL_GEN_NAME_STRIDE];

    if (input_dims == 1) {
        (void)snprintf(input_names[0], STACK_PTX_KERNEL_GEN_NAME_STRIDE, "x");
    } else {
        for (int64_t d = 0; d < input_dims; ++d) {
            (void)snprintf(input_names[d], STACK_PTX_KERNEL_GEN_NAME_STRIDE, "x%" PRId64, d);
        }
    }

    for (size_t d = 0; d < embed_dims_size; ++d) {
        char* name = output_names[d];
        (void)snprintf(name, STACK_PTX_KERNEL_GEN_NAME_STRIDE, "y%zu", d);
        inject_args[d].kind = PTX_ARG_OUT;
        inject_args[d].type_name = "F32";
        inject_args[d].name = name;
        inject_args[d].expr = name;
    }
    for (size_t d = 0; d < in_count; ++d) {
        size_t arg_idx = mod_count + out_count + d;
        inject_args[arg_idx].kind = PTX_ARG_IN;
        inject_args[arg_idx].type_name = resolved_input_type;
        inject_args[arg_idx].name = input_names[d];
        inject_args[arg_idx].expr = input_names[d];
    }

    for (size_t d = 0; d < in_count; ++d) {
        _STACK_PTX_KERNEL_GEN_CHECK_RET(gen_write(
                gb,
                "        %s %s = data[(uint64_t)%zu * (uint64_t)ld_input + (uint64_t)i];\n",
                input_c_type,
                input_names[d],
                d
            ));
    }
    _STACK_PTX_KERNEL_GEN_CHECK_RET(gen_write(gb, "\n"));
    for (size_t i = 0; i < output_count; ++i) {
        _STACK_PTX_KERNEL_GEN_CHECK_RET(gen_write(
                gb,
                "        float %s;\n",
                inject_args[i].name
            ));
    }
    _STACK_PTX_KERNEL_GEN_CHECK_RET(gen_write(gb, "        int64_t local_idx = -1;\n"));
    _STACK_PTX_KERNEL_GEN_CHECK_RET(gen_write(gb, "\n"));
    _STACK_PTX_KERNEL_GEN_CHECK_RET(gen_write(gb, "        switch (group) {\n"));

    for (int64_t group = 0; group < groups_per_kernel; ++group) {
        _STACK_PTX_KERNEL_GEN_CHECK_RET(
            emit_case_block(
                gb,
                group,
                global_base,
                inject_args,
                num_args,
                mod_count,
                out_count,
                in_count
            )
        );
        if (group + 1 < groups_per_kernel) {
            _STACK_PTX_KERNEL_GEN_CHECK_RET(gen_write(gb, "\n"));
        }
    }

    _STACK_PTX_KERNEL_GEN_CHECK_RET(gen_write(gb, "\n"));
    _STACK_PTX_KERNEL_GEN_CHECK_RET(gen_write(
            gb,
            "        default:\n"
            "            break;\n"
            "        } // switch(group)\n"
            "        if (local_idx >= 0) {\n"
        ));
    for (size_t i = 0; i < output_count; ++i) {
        _STACK_PTX_KERNEL_GEN_CHECK_RET(gen_write(
                gb,
                "            embed[(uint64_t)local_idx * (uint64_t)batch_stride + "
                "(uint64_t)%zu * (uint64_t)ld_embed + (uint64_t)i] = %s;\n",
                i,
                inject_args[i].name
            ));
    }
    _STACK_PTX_KERNEL_GEN_CHECK_RET(gen_write(
            gb,
            "        }\n"
            "    } // for i\n"
            "} // kernel\n"
        ));

    return STACK_PTX_KERNEL_GEN_SUCCESS;
}

static StackPtxKernelGenResult gen_kernel_emit(
    GenBuffer* gb,
    int64_t num_kernels,
    int64_t groups_per_kernel,
    int64_t tile_size,
    int64_t embed_dims,
    const char* kernel_name_format,
    const char* input_type_name,
    int64_t input_dims
) {
    _STACK_PTX_KERNEL_GEN_CHECK_RET(
        emit_header(gb, num_kernels, groups_per_kernel, tile_size, embed_dims, input_dims)
    );

    for (int64_t k = 0; k < num_kernels; ++k) {
        _STACK_PTX_KERNEL_GEN_CHECK_RET(gen_write(gb, "\n"));
        _STACK_PTX_KERNEL_GEN_CHECK_RET(
            emit_kernel(
                gb,
                k,
                groups_per_kernel,
                embed_dims,
                input_dims,
                input_type_name,
                kernel_name_format
            )
        );
    }

    return STACK_PTX_KERNEL_GEN_SUCCESS;
}

StackPtxKernelGenResult
elite_nle_stack_ptx_inject_kernel_gen(
    int64_t num_kernels,
    int64_t groups_per_kernel,
    int64_t tile_size,
    int64_t embed_dims,
    const char* kernel_name_format,
    const char* input_type_name,
    int64_t input_dims,
    void* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_ret
) {
    if (!buffer_bytes_written_ret) {
        _STACK_PTX_KERNEL_GEN_ERROR(STACK_PTX_KERNEL_GEN_ERROR_INVALID_VALUE);
    }

    GenBuffer gb = {
        .buffer = (char*)buffer,
        .buffer_size = buffer ? buffer_size : 0,
        .offset = 0,
        .status = STACK_PTX_KERNEL_GEN_SUCCESS
    };

    const char* name_format = (kernel_name_format && kernel_name_format[0])
        ? kernel_name_format
        : kDefaultKernelNameFormat;

    StackPtxKernelGenResult result = gen_kernel_emit(
        &gb,
        num_kernels,
        groups_per_kernel,
        tile_size,
        embed_dims,
        name_format,
        input_type_name,
        input_dims
    );

    *buffer_bytes_written_ret = gb.offset;

    if (result != STACK_PTX_KERNEL_GEN_SUCCESS) {
        return result;
    }

    return gb.status;
}
