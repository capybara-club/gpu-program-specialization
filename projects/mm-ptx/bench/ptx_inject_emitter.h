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
#ifndef MM_PTX_PTX_INJECT_EMITTER_H
#define MM_PTX_PTX_INJECT_EMITTER_H

#include <ctype.h>
#include <inttypes.h>
#include <limits.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

typedef enum {
    PTX_INJECT_EMITTER_SUCCESS = 0,
    PTX_INJECT_EMITTER_ERROR_INVALID_VALUE = 1,
    PTX_INJECT_EMITTER_ERROR_INSUFFICIENT_BUFFER = 2,
    PTX_INJECT_EMITTER_ERROR_OUT_OF_MEMORY = 3,
    PTX_INJECT_EMITTER_ERROR_FORMAT = 4
} PtxInjectEmitterResult;

typedef enum {
    PTX_INJECT_EMITTER_ARG_IN,
    PTX_INJECT_EMITTER_ARG_OUT,
    PTX_INJECT_EMITTER_ARG_MOD
} PtxInjectEmitterArgKind;

typedef struct {
    PtxInjectEmitterArgKind kind;
    const char* type_name;
    const char* name;
    const char* expr;
} PtxInjectEmitterArg;

typedef struct {
    const char* type_name;
    const char* reg_suffix;
    const char* mov_postfix;
    const char* constraint;
    const char* bind_kind;
} PtxInjectEmitterTypeInfo;

typedef struct {
    char* buffer;
    size_t buffer_size;
    size_t offset;
    PtxInjectEmitterResult status;
} PtxInjectEmitterBuffer;

#define PTX_INJECT_EMITTER_MAX_ARGS 1024u
#define PTX_INJECT_EMITTER_MAX_EMBED_DIMS (PTX_INJECT_EMITTER_MAX_ARGS - 1u)
#define PTX_INJECT_EMITTER_NAME_STRIDE 32u

static inline PtxInjectEmitterResult ptx_inject_emitter_write(
    PtxInjectEmitterBuffer* gb,
    const char* fmt,
    ...
) {
    va_list args;
    va_start(args, fmt);

    char* out = NULL;
    size_t remaining = 0;
    if (gb->buffer && gb->status == PTX_INJECT_EMITTER_SUCCESS) {
        if (gb->offset < gb->buffer_size) {
            out = gb->buffer + gb->offset;
            remaining = gb->buffer_size - gb->offset;
        }
    }

    int needed_int = vsnprintf(out, remaining, fmt, args);
    va_end(args);

    if (needed_int < 0) {
        return PTX_INJECT_EMITTER_ERROR_FORMAT;
    }

    size_t needed = (size_t)needed_int;
    if (SIZE_MAX - gb->offset < needed) {
        return PTX_INJECT_EMITTER_ERROR_INVALID_VALUE;
    }

    if (gb->buffer && gb->status == PTX_INJECT_EMITTER_SUCCESS &&
        gb->offset + needed > gb->buffer_size) {
        gb->status = PTX_INJECT_EMITTER_ERROR_INSUFFICIENT_BUFFER;
    }

    gb->offset += needed;
    return PTX_INJECT_EMITTER_SUCCESS;
}

static inline const PtxInjectEmitterTypeInfo* ptx_inject_emitter_type_info(
    const char* type_name
) {
    static const PtxInjectEmitterTypeInfo kTypeInfo[] = {
        { "F16",   "b16", "b16", "h", "U16" },
        { "F16X2", "b32", "b32", "r", "U32" },
        { "S32",   "s32", "s32", "r", "ID" },
        { "U32",   "u32", "u32", "r", "ID" },
        { "F32",   "f32", "f32", "f", "ID" },
        { "B32",   "b32", "b32", "r", "ID" }
    };
    static const size_t kTypeInfoCount = sizeof(kTypeInfo) / sizeof(kTypeInfo[0]);

    if (!type_name) {
        return NULL;
    }
    for (size_t i = 0; i < kTypeInfoCount; ++i) {
        if (strcmp(kTypeInfo[i].type_name, type_name) == 0) {
            return &kTypeInfo[i];
        }
    }
    return NULL;
}

static inline const char* ptx_inject_emitter_c_type(const char* type_name) {
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

static inline char ptx_inject_emitter_kind_char(PtxInjectEmitterArgKind kind) {
    switch (kind) {
        case PTX_INJECT_EMITTER_ARG_MOD: return 'm';
        case PTX_INJECT_EMITTER_ARG_OUT: return 'o';
        case PTX_INJECT_EMITTER_ARG_IN: return 'i';
    }
    return '?';
}

static inline PtxInjectEmitterResult ptx_inject_emitter_bind_expr(
    const char* bind_kind,
    const char* expr,
    char* out,
    size_t out_size
) {
    if (strcmp(bind_kind, "ID") == 0) {
        return snprintf(out, out_size, "%s", expr) >= 0
            ? PTX_INJECT_EMITTER_SUCCESS
            : PTX_INJECT_EMITTER_ERROR_FORMAT;
    }
    if (strcmp(bind_kind, "U16") == 0) {
        return snprintf(out, out_size, "(*reinterpret_cast<unsigned short*>(& (%s) ))", expr) >= 0
            ? PTX_INJECT_EMITTER_SUCCESS
            : PTX_INJECT_EMITTER_ERROR_FORMAT;
    }
    if (strcmp(bind_kind, "U32") == 0) {
        return snprintf(out, out_size, "(*reinterpret_cast<unsigned int  *>(& (%s) ))", expr) >= 0
            ? PTX_INJECT_EMITTER_SUCCESS
            : PTX_INJECT_EMITTER_ERROR_FORMAT;
    }
    return PTX_INJECT_EMITTER_ERROR_FORMAT;
}

static inline PtxInjectEmitterResult ptx_inject_emitter_emit_escaped_line(
    PtxInjectEmitterBuffer* gb,
    const char* line,
    size_t line_len,
    const char* indent
) {
    if (ptx_inject_emitter_write(gb, "%s    \"", indent) != PTX_INJECT_EMITTER_SUCCESS) {
        return gb->status;
    }
    for (size_t i = 0; i < line_len; ++i) {
        char c = line[i];
        if (c == '%') {
            if (ptx_inject_emitter_write(gb, "%%%%") != PTX_INJECT_EMITTER_SUCCESS) {
                return gb->status;
            }
        } else if (c == '\\') {
            if (ptx_inject_emitter_write(gb, "\\\\") != PTX_INJECT_EMITTER_SUCCESS) {
                return gb->status;
            }
        } else if (c == '\"') {
            if (ptx_inject_emitter_write(gb, "\\\"") != PTX_INJECT_EMITTER_SUCCESS) {
                return gb->status;
            }
        } else {
            if (ptx_inject_emitter_write(gb, "%c", c) != PTX_INJECT_EMITTER_SUCCESS) {
                return gb->status;
            }
        }
    }
    if (ptx_inject_emitter_write(gb, "\\n\\t\"\n") != PTX_INJECT_EMITTER_SUCCESS) {
        return gb->status;
    }
    return PTX_INJECT_EMITTER_SUCCESS;
}

static inline int ptx_inject_emitter_line_is_blank(const char* line, size_t line_len) {
    for (size_t i = 0; i < line_len; ++i) {
        if (!isspace((unsigned char)line[i])) {
            return 0;
        }
    }
    return 1;
}

static inline int ptx_inject_emitter_line_is_token(
    const char* line,
    size_t line_len,
    char token
) {
    size_t i = 0;
    while (i < line_len && isspace((unsigned char)line[i])) {
        i += 1;
    }
    if (i >= line_len || line[i] != token) {
        return 0;
    }
    i += 1;
    while (i < line_len && isspace((unsigned char)line[i])) {
        i += 1;
    }
    return i == line_len;
}

static inline PtxInjectEmitterResult ptx_inject_emitter_emit_stub(
    PtxInjectEmitterBuffer* gb,
    const char* stub,
    const char* indent
) {
    if (!stub || !indent) {
        return PTX_INJECT_EMITTER_ERROR_INVALID_VALUE;
    }

    const char* stub_end = stub + strlen(stub);
    const char* first_nonblank = NULL;
    const char* last_nonblank = NULL;
    size_t first_len = 0;
    size_t last_len = 0;

    const char* scan = stub;
    while (*scan) {
        const char* line_end = strchr(scan, '\n');
        size_t line_len = line_end ? (size_t)(line_end - scan) : strlen(scan);
        if (!ptx_inject_emitter_line_is_blank(scan, line_len)) {
            if (!first_nonblank) {
                first_nonblank = scan;
                first_len = line_len;
            }
            last_nonblank = scan;
            last_len = line_len;
        }
        if (!line_end) {
            break;
        }
        scan = line_end + 1;
    }

    int drop_first = first_nonblank && ptx_inject_emitter_line_is_token(first_nonblank, first_len, '{');
    int drop_last = last_nonblank && ptx_inject_emitter_line_is_token(last_nonblank, last_len, '}');

    const char* cursor = stub;
    while (*cursor) {
        const char* line_end = strchr(cursor, '\n');
        size_t line_len = line_end ? (size_t)(line_end - cursor) : strlen(cursor);
        if (!((drop_first && cursor == first_nonblank) || (drop_last && cursor == last_nonblank))) {
            if (ptx_inject_emitter_emit_escaped_line(gb, cursor, line_len, indent) !=
                PTX_INJECT_EMITTER_SUCCESS) {
                return gb->status;
            }
        }
        if (!line_end) {
            break;
        }
        cursor = line_end + 1;
    }

    return PTX_INJECT_EMITTER_SUCCESS;
}

static inline PtxInjectEmitterResult ptx_inject_emitter_emit_asm(
    PtxInjectEmitterBuffer* gb,
    const char* site_name,
    const PtxInjectEmitterArg* args,
    size_t num_args,
    size_t mod_count,
    size_t out_count,
    size_t in_count,
    int emit_ptx_inject,
    const char* stub,
    const char* indent
) {
    if (!args || num_args == 0) {
        return PTX_INJECT_EMITTER_ERROR_INVALID_VALUE;
    }
    if (num_args > PTX_INJECT_EMITTER_MAX_ARGS) {
        return PTX_INJECT_EMITTER_ERROR_INVALID_VALUE;
    }
    if (mod_count + out_count + in_count != num_args) {
        return PTX_INJECT_EMITTER_ERROR_INVALID_VALUE;
    }
    if (!emit_ptx_inject && !stub) {
        return PTX_INJECT_EMITTER_ERROR_INVALID_VALUE;
    }

    if (ptx_inject_emitter_write(gb, "%sasm (\n", indent) != PTX_INJECT_EMITTER_SUCCESS) {
        return gb->status;
    }
    if (ptx_inject_emitter_write(gb, "%s    \"{\\n\\t\"\n", indent) != PTX_INJECT_EMITTER_SUCCESS) {
        return gb->status;
    }

    for (size_t i = 0; i < num_args; ++i) {
        const PtxInjectEmitterArg* arg = &args[i];
        const PtxInjectEmitterTypeInfo* info = ptx_inject_emitter_type_info(arg->type_name);
        if (!info) {
            return PTX_INJECT_EMITTER_ERROR_INVALID_VALUE;
        }
        if (ptx_inject_emitter_write(
                gb,
                "%s    \".reg .%s %%%%_x%zu;\\n\\t\"\n",
                indent,
                info->reg_suffix,
                i
            ) != PTX_INJECT_EMITTER_SUCCESS) {
            return gb->status;
        }
    }

    for (size_t i = 0; i < mod_count; ++i) {
        const PtxInjectEmitterArg* arg = &args[i];
        const PtxInjectEmitterTypeInfo* info = ptx_inject_emitter_type_info(arg->type_name);
        if (ptx_inject_emitter_write(
                gb,
                "%s    \"mov.%s %%%%_x%zu, %%%zu;\\n\\t\"\n",
                indent,
                info->mov_postfix,
                i,
                i
            ) != PTX_INJECT_EMITTER_SUCCESS) {
            return gb->status;
        }
    }
    for (size_t i = 0; i < in_count; ++i) {
        size_t arg_idx = mod_count + out_count + i;
        const PtxInjectEmitterArg* arg = &args[arg_idx];
        const PtxInjectEmitterTypeInfo* info = ptx_inject_emitter_type_info(arg->type_name);
        if (ptx_inject_emitter_write(
                gb,
                "%s    \"mov.%s %%%%_x%zu, %%%zu;\\n\\t\"\n",
                indent,
                info->mov_postfix,
                arg_idx,
                arg_idx
            ) != PTX_INJECT_EMITTER_SUCCESS) {
            return gb->status;
        }
    }

    if (ptx_inject_emitter_write(
            gb,
            "%s    \"// PTX_INJECT_START %s\\n\\t\"\n",
            indent,
            site_name
        ) != PTX_INJECT_EMITTER_SUCCESS) {
        return gb->status;
    }

    if (emit_ptx_inject) {
        for (size_t i = 0; i < num_args; ++i) {
            const PtxInjectEmitterArg* arg = &args[i];
            const PtxInjectEmitterTypeInfo* info = ptx_inject_emitter_type_info(arg->type_name);
            if (ptx_inject_emitter_write(
                    gb,
                    "%s    \"// _x%zu %c %s %s %s\\n\\t\"\n",
                    indent,
                    i,
                    ptx_inject_emitter_kind_char(arg->kind),
                    info->reg_suffix,
                    arg->type_name,
                    arg->name
                ) != PTX_INJECT_EMITTER_SUCCESS) {
                return gb->status;
            }
        }
    } else {
        if (ptx_inject_emitter_emit_stub(gb, stub, indent) != PTX_INJECT_EMITTER_SUCCESS) {
            return gb->status;
        }
    }
    if (ptx_inject_emitter_write(gb, "%s    \"// PTX_INJECT_END\\n\\t\"\n", indent) !=
        PTX_INJECT_EMITTER_SUCCESS) {
        return gb->status;
    }

    for (size_t i = 0; i < mod_count; ++i) {
        const PtxInjectEmitterArg* arg = &args[i];
        const PtxInjectEmitterTypeInfo* info = ptx_inject_emitter_type_info(arg->type_name);
        if (ptx_inject_emitter_write(
                gb,
                "%s    \"mov.%s %%%zu, %%%%_x%zu;\\n\\t\"\n",
                indent,
                info->mov_postfix,
                i,
                i
            ) != PTX_INJECT_EMITTER_SUCCESS) {
            return gb->status;
        }
    }
    for (size_t i = 0; i < out_count; ++i) {
        size_t arg_idx = mod_count + i;
        const PtxInjectEmitterArg* arg = &args[arg_idx];
        const PtxInjectEmitterTypeInfo* info = ptx_inject_emitter_type_info(arg->type_name);
        if (ptx_inject_emitter_write(
                gb,
                "%s    \"mov.%s %%%zu, %%%%_x%zu;\\n\\t\"\n",
                indent,
                info->mov_postfix,
                arg_idx,
                arg_idx
            ) != PTX_INJECT_EMITTER_SUCCESS) {
            return gb->status;
        }
    }

    if (ptx_inject_emitter_write(gb, "%s    \"}\"\n", indent) != PTX_INJECT_EMITTER_SUCCESS) {
        return gb->status;
    }

    if (mod_count + out_count > 0) {
        bool first = true;
        if (ptx_inject_emitter_write(gb, "%s    : ", indent) != PTX_INJECT_EMITTER_SUCCESS) {
            return gb->status;
        }
        for (size_t i = 0; i < mod_count; ++i) {
            const PtxInjectEmitterArg* arg = &args[i];
            const PtxInjectEmitterTypeInfo* info = ptx_inject_emitter_type_info(arg->type_name);
            char expr_buf[256];
            if (ptx_inject_emitter_bind_expr(info->bind_kind, arg->expr, expr_buf, sizeof(expr_buf)) !=
                PTX_INJECT_EMITTER_SUCCESS) {
                return PTX_INJECT_EMITTER_ERROR_INVALID_VALUE;
            }
            if (!first) {
                if (ptx_inject_emitter_write(gb, ", ") != PTX_INJECT_EMITTER_SUCCESS) {
                    return gb->status;
                }
            }
            first = false;
            if (ptx_inject_emitter_write(gb, "\"+%s\"(%s)", info->constraint, expr_buf) !=
                PTX_INJECT_EMITTER_SUCCESS) {
                return gb->status;
            }
        }
        for (size_t i = 0; i < out_count; ++i) {
            const PtxInjectEmitterArg* arg = &args[mod_count + i];
            const PtxInjectEmitterTypeInfo* info = ptx_inject_emitter_type_info(arg->type_name);
            char expr_buf[256];
            if (ptx_inject_emitter_bind_expr(info->bind_kind, arg->expr, expr_buf, sizeof(expr_buf)) !=
                PTX_INJECT_EMITTER_SUCCESS) {
                return PTX_INJECT_EMITTER_ERROR_INVALID_VALUE;
            }
            if (!first) {
                if (ptx_inject_emitter_write(gb, ", ") != PTX_INJECT_EMITTER_SUCCESS) {
                    return gb->status;
                }
            }
            first = false;
            if (ptx_inject_emitter_write(gb, "\"=%s\"(%s)", info->constraint, expr_buf) !=
                PTX_INJECT_EMITTER_SUCCESS) {
                return gb->status;
            }
        }
        if (ptx_inject_emitter_write(gb, "\n") != PTX_INJECT_EMITTER_SUCCESS) {
            return gb->status;
        }
    }

    if (in_count > 0) {
        bool first = true;
        if (mod_count + out_count > 0) {
            if (ptx_inject_emitter_write(gb, "%s    : ", indent) != PTX_INJECT_EMITTER_SUCCESS) {
                return gb->status;
            }
        } else {
            if (ptx_inject_emitter_write(gb, "%s    : : ", indent) != PTX_INJECT_EMITTER_SUCCESS) {
                return gb->status;
            }
        }
        for (size_t i = 0; i < in_count; ++i) {
            const PtxInjectEmitterArg* arg = &args[mod_count + out_count + i];
            const PtxInjectEmitterTypeInfo* info = ptx_inject_emitter_type_info(arg->type_name);
            char expr_buf[256];
            if (ptx_inject_emitter_bind_expr(info->bind_kind, arg->expr, expr_buf, sizeof(expr_buf)) !=
                PTX_INJECT_EMITTER_SUCCESS) {
                return PTX_INJECT_EMITTER_ERROR_INVALID_VALUE;
            }
            if (!first) {
                if (ptx_inject_emitter_write(gb, ", ") != PTX_INJECT_EMITTER_SUCCESS) {
                    return gb->status;
                }
            }
            first = false;
            if (ptx_inject_emitter_write(gb, "\"%s\"(%s)", info->constraint, expr_buf) !=
                PTX_INJECT_EMITTER_SUCCESS) {
                return gb->status;
            }
        }
        if (ptx_inject_emitter_write(gb, "\n") != PTX_INJECT_EMITTER_SUCCESS) {
            return gb->status;
        }
    }

    if (ptx_inject_emitter_write(gb, "%s);\n", indent) != PTX_INJECT_EMITTER_SUCCESS) {
        return gb->status;
    }

    return PTX_INJECT_EMITTER_SUCCESS;
}

static inline PtxInjectEmitterResult ptx_inject_emitter_emit_header(
    PtxInjectEmitterBuffer* gb,
    int64_t num_kernels,
    int64_t groups_per_kernel,
    int64_t tile_size,
    int64_t embed_dims,
    int64_t input_dims
) {
    int64_t indivs_per_kernel = groups_per_kernel;
    int64_t total_indivs = num_kernels * indivs_per_kernel;

    return ptx_inject_emitter_write(
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

static inline PtxInjectEmitterResult ptx_inject_emitter_emit_case(
    PtxInjectEmitterBuffer* gb,
    int64_t group,
    int64_t global_base,
    const PtxInjectEmitterArg* inject_args,
    size_t num_args,
    size_t mod_count,
    size_t out_count,
    size_t in_count,
    int emit_ptx_inject,
    const char* stub
) {
    if (ptx_inject_emitter_write(gb, "        case %" PRId64 ": {\n", group) !=
        PTX_INJECT_EMITTER_SUCCESS) {
        return gb->status;
    }

    int64_t local_idx = group;
    int64_t global_indiv = global_base + local_idx;
    char site_name[64];
    snprintf(site_name, sizeof(site_name), "func_%" PRId64, global_indiv);
    if (ptx_inject_emitter_write(gb, "            local_idx = %" PRId64 ";\n", local_idx) !=
        PTX_INJECT_EMITTER_SUCCESS) {
        return gb->status;
    }
    if (ptx_inject_emitter_emit_asm(
            gb,
            site_name,
            inject_args,
            num_args,
            mod_count,
            out_count,
            in_count,
            emit_ptx_inject,
            stub,
            "            "
        ) != PTX_INJECT_EMITTER_SUCCESS) {
        return gb->status;
    }

    if (ptx_inject_emitter_write(gb, "            break;\n") != PTX_INJECT_EMITTER_SUCCESS) {
        return gb->status;
    }
    if (ptx_inject_emitter_write(gb, "        }\n") != PTX_INJECT_EMITTER_SUCCESS) {
        return gb->status;
    }
    return PTX_INJECT_EMITTER_SUCCESS;
}

static inline PtxInjectEmitterResult ptx_inject_emitter_emit_kernel(
    PtxInjectEmitterBuffer* gb,
    int64_t kernel_index,
    int64_t groups_per_kernel,
    int64_t embed_dims,
    int64_t input_dims,
    const char* input_type_name,
    const char* kernel_name_format,
    int emit_ptx_inject,
    const char* const* stubs,
    size_t num_stubs
) {
    if (kernel_index < 0) {
        return PTX_INJECT_EMITTER_ERROR_INVALID_VALUE;
    }

    char kernel_name[128];
    snprintf(kernel_name, sizeof(kernel_name), kernel_name_format, (size_t)kernel_index);

    const char* resolved_input_type = (input_type_name && input_type_name[0])
        ? input_type_name
        : "U32";
    const PtxInjectEmitterTypeInfo* input_info = ptx_inject_emitter_type_info(resolved_input_type);
    const char* input_c_type = ptx_inject_emitter_c_type(resolved_input_type);
    if (!input_info || !input_c_type) {
        return PTX_INJECT_EMITTER_ERROR_INVALID_VALUE;
    }
    if (input_dims < 1) {
        return PTX_INJECT_EMITTER_ERROR_INVALID_VALUE;
    }

    if (ptx_inject_emitter_write(
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
            "    for (int64_t i = tile_start + (int64_t)threadIdx.x; i < tile_end; "
            "i += (int64_t)blockDim.x) {\n",
            kernel_name,
            input_c_type
        ) != PTX_INJECT_EMITTER_SUCCESS) {
        return gb->status;
    }

    int64_t indivs_per_kernel = groups_per_kernel;
    int64_t global_base = kernel_index * indivs_per_kernel;

    if (embed_dims < 1) {
        return PTX_INJECT_EMITTER_ERROR_INVALID_VALUE;
    }
    if (embed_dims > (int64_t)PTX_INJECT_EMITTER_MAX_EMBED_DIMS) {
        return PTX_INJECT_EMITTER_ERROR_INVALID_VALUE;
    }
    if (input_dims > (int64_t)PTX_INJECT_EMITTER_MAX_ARGS) {
        return PTX_INJECT_EMITTER_ERROR_INVALID_VALUE;
    }

    size_t embed_dims_size = (size_t)embed_dims;
    size_t mod_count = 0;
    size_t out_count = embed_dims_size;
    size_t in_count = (size_t)input_dims;
    size_t num_args = mod_count + out_count + in_count;
    size_t output_count = mod_count + out_count;
    if (num_args > PTX_INJECT_EMITTER_MAX_ARGS) {
        return PTX_INJECT_EMITTER_ERROR_INVALID_VALUE;
    }
    PtxInjectEmitterArg inject_args[PTX_INJECT_EMITTER_MAX_ARGS];
    char output_names[PTX_INJECT_EMITTER_MAX_EMBED_DIMS][PTX_INJECT_EMITTER_NAME_STRIDE];
    char input_names[PTX_INJECT_EMITTER_MAX_ARGS][PTX_INJECT_EMITTER_NAME_STRIDE];

    if (input_dims == 1) {
        (void)snprintf(input_names[0], PTX_INJECT_EMITTER_NAME_STRIDE, "x");
    } else {
        for (int64_t d = 0; d < input_dims; ++d) {
            (void)snprintf(input_names[d], PTX_INJECT_EMITTER_NAME_STRIDE, "x%" PRId64, d);
        }
    }

    for (size_t d = 0; d < embed_dims_size; ++d) {
        char* name = output_names[d];
        (void)snprintf(name, PTX_INJECT_EMITTER_NAME_STRIDE, "y%zu", d);
        inject_args[d].kind = PTX_INJECT_EMITTER_ARG_OUT;
        inject_args[d].type_name = "F32";
        inject_args[d].name = name;
        inject_args[d].expr = name;
    }
    for (size_t d = 0; d < in_count; ++d) {
        size_t arg_idx = mod_count + out_count + d;
        inject_args[arg_idx].kind = PTX_INJECT_EMITTER_ARG_IN;
        inject_args[arg_idx].type_name = resolved_input_type;
        inject_args[arg_idx].name = input_names[d];
        inject_args[arg_idx].expr = input_names[d];
    }

    for (size_t d = 0; d < in_count; ++d) {
        if (ptx_inject_emitter_write(
                gb,
                "        %s %s = data[(uint64_t)%zu * (uint64_t)ld_input + (uint64_t)i];\n",
                input_c_type,
                input_names[d],
                d
            ) != PTX_INJECT_EMITTER_SUCCESS) {
            return gb->status;
        }
    }
    if (ptx_inject_emitter_write(gb, "\n") != PTX_INJECT_EMITTER_SUCCESS) {
        return gb->status;
    }
    for (size_t i = 0; i < output_count; ++i) {
        if (ptx_inject_emitter_write(gb, "        float %s;\n", inject_args[i].name) !=
            PTX_INJECT_EMITTER_SUCCESS) {
            return gb->status;
        }
    }
    if (ptx_inject_emitter_write(gb, "        int64_t local_idx = -1;\n\n") !=
        PTX_INJECT_EMITTER_SUCCESS) {
        return gb->status;
    }
    if (ptx_inject_emitter_write(gb, "        switch (group) {\n") != PTX_INJECT_EMITTER_SUCCESS) {
        return gb->status;
    }

    for (int64_t group = 0; group < groups_per_kernel; ++group) {
        const char* stub = NULL;
        if (!emit_ptx_inject) {
            if (!stubs) {
                return PTX_INJECT_EMITTER_ERROR_INVALID_VALUE;
            }
            size_t stub_idx = (size_t)(global_base + group);
            if (stub_idx >= num_stubs) {
                return PTX_INJECT_EMITTER_ERROR_INVALID_VALUE;
            }
            stub = stubs[stub_idx];
            if (!stub) {
                return PTX_INJECT_EMITTER_ERROR_INVALID_VALUE;
            }
        }
        if (ptx_inject_emitter_emit_case(
                gb,
                group,
                global_base,
                inject_args,
                num_args,
                mod_count,
                out_count,
                in_count,
                emit_ptx_inject,
                stub
            ) != PTX_INJECT_EMITTER_SUCCESS) {
            return gb->status;
        }
        if (group + 1 < groups_per_kernel) {
            if (ptx_inject_emitter_write(gb, "\n") != PTX_INJECT_EMITTER_SUCCESS) {
                return gb->status;
            }
        }
    }

    if (ptx_inject_emitter_write(
            gb,
            "\n"
            "        default:\n"
            "            break;\n"
            "        } // switch(group)\n"
            "        if (local_idx >= 0) {\n"
        ) != PTX_INJECT_EMITTER_SUCCESS) {
        return gb->status;
    }
    for (size_t i = 0; i < output_count; ++i) {
        if (ptx_inject_emitter_write(
                gb,
                "            embed[(uint64_t)local_idx * (uint64_t)batch_stride + "
                "(uint64_t)%zu * (uint64_t)ld_embed + (uint64_t)i] = %s;\n",
                i,
                inject_args[i].name
            ) != PTX_INJECT_EMITTER_SUCCESS) {
            return gb->status;
        }
    }
    if (ptx_inject_emitter_write(
            gb,
            "        }\n"
            "    } // for i\n"
            "} // kernel\n"
        ) != PTX_INJECT_EMITTER_SUCCESS) {
        return gb->status;
    }

    return PTX_INJECT_EMITTER_SUCCESS;
}

static inline PtxInjectEmitterResult ptx_inject_emit_cuda(
    int64_t num_kernels,
    int64_t groups_per_kernel,
    int64_t tile_size,
    int64_t embed_dims,
    const char* kernel_name_format,
    const char* input_type_name,
    int64_t input_dims,
    int emit_ptx_inject,
    const char* const* stubs,
    size_t num_stubs,
    void* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_ret
) {
    if (!buffer_bytes_written_ret) {
        return PTX_INJECT_EMITTER_ERROR_INVALID_VALUE;
    }
    if (!emit_ptx_inject && (!stubs || num_stubs == 0)) {
        return PTX_INJECT_EMITTER_ERROR_INVALID_VALUE;
    }

    PtxInjectEmitterBuffer gb = {
        .buffer = (char*)buffer,
        .buffer_size = buffer ? buffer_size : 0,
        .offset = 0,
        .status = PTX_INJECT_EMITTER_SUCCESS
    };

    const char* name_format = (kernel_name_format && kernel_name_format[0])
        ? kernel_name_format
        : "kernel_%06zu";

    PtxInjectEmitterResult result = ptx_inject_emitter_emit_header(
        &gb,
        num_kernels,
        groups_per_kernel,
        tile_size,
        embed_dims,
        input_dims
    );
    if (result != PTX_INJECT_EMITTER_SUCCESS) {
        return result;
    }

    for (int64_t k = 0; k < num_kernels; ++k) {
        if (ptx_inject_emitter_write(&gb, "\n") != PTX_INJECT_EMITTER_SUCCESS) {
            return gb.status;
        }
        result = ptx_inject_emitter_emit_kernel(
            &gb,
            k,
            groups_per_kernel,
            embed_dims,
            input_dims,
            input_type_name,
            name_format,
            emit_ptx_inject,
            stubs,
            num_stubs
        );
        if (result != PTX_INJECT_EMITTER_SUCCESS) {
            return result;
        }
    }

    *buffer_bytes_written_ret = gb.offset;
    if (gb.status != PTX_INJECT_EMITTER_SUCCESS) {
        return gb.status;
    }
    return PTX_INJECT_EMITTER_SUCCESS;
}

#endif
