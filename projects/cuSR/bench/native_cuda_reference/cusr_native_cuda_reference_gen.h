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
#ifndef CUSR_NATIVE_CUDA_REFERENCE_GEN_H_INCLUDED
#define CUSR_NATIVE_CUDA_REFERENCE_GEN_H_INCLUDED

#include <stddef.h>

#define CUSR_NATIVE_CUDA_REFERENCE_EXPR_OFFSET_MINMAX 0u
#define CUSR_NATIVE_CUDA_REFERENCE_EXPR_MIXED_REDUCE 1u
#define CUSR_NATIVE_CUDA_REFERENCE_EXPR_DEPTH3_ALU 2u
#define CUSR_NATIVE_CUDA_REFERENCE_EXPR_DEPTH3_MUFU 3u

#ifdef __cplusplus
#define CUSR_NATIVE_CUDA_REFERENCE_GEN_PUBLIC_DEF extern "C"
#else
#define CUSR_NATIVE_CUDA_REFERENCE_GEN_PUBLIC_DEF
#endif

typedef enum CusrNativeCudaReferenceGenResult {
    CUSR_NATIVE_CUDA_REFERENCE_GEN_SUCCESS = 0,
    CUSR_NATIVE_CUDA_REFERENCE_GEN_ERROR_INVALID_VALUE = 1,
    CUSR_NATIVE_CUDA_REFERENCE_GEN_ERROR_OVERFLOW = 2,
    CUSR_NATIVE_CUDA_REFERENCE_GEN_ERROR_INSUFFICIENT_BUFFER = 3,
    CUSR_NATIVE_CUDA_REFERENCE_GEN_ERROR_FORMAT = 4
} CusrNativeCudaReferenceGenResult;

CUSR_NATIVE_CUDA_REFERENCE_GEN_PUBLIC_DEF const char*
cusr_native_cuda_reference_gen_result_to_string(CusrNativeCudaReferenceGenResult result);

CUSR_NATIVE_CUDA_REFERENCE_GEN_PUBLIC_DEF CusrNativeCudaReferenceGenResult
cusr_native_cuda_reference_gen_cuda_size(
    size_t kernels_per_file,
    size_t asts_per_kernel,
    size_t tile_rows,
    size_t threads_per_cta,
    unsigned expr_mode,
    size_t* size_ret);

CUSR_NATIVE_CUDA_REFERENCE_GEN_PUBLIC_DEF CusrNativeCudaReferenceGenResult
cusr_native_cuda_reference_gen_cuda(
    char* buffer,
    size_t buffer_size,
    size_t kernels_per_file,
    size_t asts_per_kernel,
    size_t tile_rows,
    size_t threads_per_cta,
    unsigned expr_mode,
    size_t* size_ret);

#endif /* CUSR_NATIVE_CUDA_REFERENCE_GEN_H_INCLUDED */

#ifdef CUSR_NATIVE_CUDA_REFERENCE_GEN_IMPLEMENTATION
#ifndef CUSR_NATIVE_CUDA_REFERENCE_GEN_IMPLEMENTATION_ONCE
#define CUSR_NATIVE_CUDA_REFERENCE_GEN_IMPLEMENTATION_ONCE

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>

#define _CUSR_NATIVE_CUDA_REFERENCE_GEN_ERROR_RET(ans) \
    do { \
        CusrNativeCudaReferenceGenResult cusr_native_cuda_reference_gen_result = (ans); \
        return cusr_native_cuda_reference_gen_result; \
    } while (0)

#define _CUSR_NATIVE_CUDA_REFERENCE_GEN_CHECK_RET(ans) \
    do { \
        CusrNativeCudaReferenceGenResult cusr_native_cuda_reference_gen_check_ret = (ans); \
        if (cusr_native_cuda_reference_gen_check_ret != CUSR_NATIVE_CUDA_REFERENCE_GEN_SUCCESS) { \
            _CUSR_NATIVE_CUDA_REFERENCE_GEN_ERROR_RET(cusr_native_cuda_reference_gen_check_ret); \
        } \
    } while (0)

typedef struct CusrNativeCudaReferenceGenWriter {
    char* buffer;
    size_t buffer_size;
    size_t size;
    CusrNativeCudaReferenceGenResult result;
} CusrNativeCudaReferenceGenWriter;

static int
cusr_native_cuda_reference_gen_checked_mul(size_t a, size_t b, size_t* out)
{
    if (a != 0u && b > SIZE_MAX / a) {
        return 0;
    }

    *out = a * b;
    return 1;
}

static uint32_t
cusr_native_cuda_reference_gen_hash32(uint32_t x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

static uint32_t
cusr_native_cuda_reference_gen_reduce_leaf(uint32_t global_ast_idx, uint32_t term)
{
    const uint32_t leaf_count = 8u;
    const uint32_t offset = global_ast_idx % leaf_count;
    return (term + offset) % leaf_count;
}

static uint32_t
cusr_native_cuda_reference_gen_reduce_op(uint32_t global_ast_idx, uint32_t term)
{
    return cusr_native_cuda_reference_gen_hash32(global_ast_idx * 0xc2b2ae35u + term * 0x27d4eb2fu) & 3u;
}

static uint32_t
cusr_native_cuda_reference_gen_depth3_unary_op(uint32_t global_ast_idx, uint32_t term)
{
    return cusr_native_cuda_reference_gen_hash32(global_ast_idx * 0x9e3779b9u + term * 0x85ebca6bu) & 3u;
}

static CusrNativeCudaReferenceGenResult
cusr_native_cuda_reference_gen_write(CusrNativeCudaReferenceGenWriter* writer, const char* fmt, ...)
{
    va_list args;
    va_list args_copy;
    int bytes;

    if (writer->result != CUSR_NATIVE_CUDA_REFERENCE_GEN_SUCCESS) {
        _CUSR_NATIVE_CUDA_REFERENCE_GEN_ERROR_RET(writer->result);
    }

    va_start(args, fmt);
    va_copy(args_copy, args);
    bytes = vsnprintf(NULL, 0u, fmt, args_copy);
    va_end(args_copy);

    if (bytes < 0) {
        va_end(args);
        writer->result = CUSR_NATIVE_CUDA_REFERENCE_GEN_ERROR_FORMAT;
        _CUSR_NATIVE_CUDA_REFERENCE_GEN_ERROR_RET(writer->result);
    }

    if ((size_t)bytes > SIZE_MAX - writer->size) {
        va_end(args);
        writer->result = CUSR_NATIVE_CUDA_REFERENCE_GEN_ERROR_OVERFLOW;
        _CUSR_NATIVE_CUDA_REFERENCE_GEN_ERROR_RET(writer->result);
    }

    if (writer->buffer != NULL) {
        if (writer->size >= writer->buffer_size ||
            (size_t)bytes >= writer->buffer_size - writer->size) {
            writer->result = CUSR_NATIVE_CUDA_REFERENCE_GEN_ERROR_INSUFFICIENT_BUFFER;
        } else {
            (void)vsnprintf(writer->buffer + writer->size, writer->buffer_size - writer->size, fmt, args);
        }
    }

    va_end(args);
    writer->size += (size_t)bytes;
    _CUSR_NATIVE_CUDA_REFERENCE_GEN_ERROR_RET(writer->result);
}

static CusrNativeCudaReferenceGenResult
cusr_native_cuda_reference_gen_emit_prelude(CusrNativeCudaReferenceGenWriter* writer, size_t tile_rows, size_t threads_per_cta)
{
    _CUSR_NATIVE_CUDA_REFERENCE_GEN_CHECK_RET(cusr_native_cuda_reference_gen_write(writer,
        "#if defined(__CUDACC_RTC__)\n"
        "#ifndef CUSR_RTC_TYPES_DEFINED\n"
        "#define CUSR_RTC_TYPES_DEFINED\n"
        "typedef unsigned char uint8_t;\n"
        "typedef unsigned int uint32_t;\n"
        "#endif\n"
        "#else\n"
        "#include <stddef.h>\n"
        "#include <stdint.h>\n"
        "#endif\n\n"
        "#define CUSR_NATIVE_CUDA_REFERENCE_TILE_ROWS %zuu\n"
        "#define CUSR_NATIVE_CUDA_REFERENCE_THREADS %zuu\n\n"
        "static __device__ __forceinline__ uint32_t cusr_native_cuda_reference_leaf_is_column(uint8_t mask, uint32_t leaf) {\n"
        "    return (((uint32_t)mask) >> leaf) & 1u;\n"
        "}\n\n"
        "static __device__ __forceinline__ uint32_t cusr_native_cuda_reference_shared_stride(uint32_t num_columns) {\n"
        "    return num_columns | 1u;\n"
        "}\n\n"
        "static __device__ __forceinline__ uint32_t cusr_native_cuda_reference_load_tile_f32(\n"
        "    const float* __restrict__ X,\n"
        "    const float* __restrict__ target,\n"
        "    size_t num_rows,\n"
        "    uint32_t num_columns,\n"
        "    size_t leading_dim,\n"
        "    float* x_tile,\n"
        "    float* target_tile,\n"
        "    uint32_t shared_stride\n"
        ") {\n"
        "    const size_t row_base = (size_t)blockIdx.x * CUSR_NATIVE_CUDA_REFERENCE_TILE_ROWS;\n"
        "    #pragma unroll\n"
        "    for (uint32_t tile_row = threadIdx.x; tile_row < CUSR_NATIVE_CUDA_REFERENCE_TILE_ROWS; tile_row += CUSR_NATIVE_CUDA_REFERENCE_THREADS) {\n"
        "        const size_t row = row_base + tile_row;\n"
        "        const bool row_valid = row < num_rows;\n"
        "        target_tile[tile_row] = row_valid ? target[row] : 0.0f;\n"
        "        #pragma unroll 1\n"
        "        for (uint32_t column = 0u; column < num_columns; ++column) {\n"
        "            x_tile[(size_t)tile_row * shared_stride + column] = row_valid ? X[(size_t)column * leading_dim + row] : 0.0f;\n"
        "        }\n"
        "    }\n"
        "    return row_base + CUSR_NATIVE_CUDA_REFERENCE_TILE_ROWS <= num_rows\n"
        "        ? CUSR_NATIVE_CUDA_REFERENCE_TILE_ROWS\n"
        "        : (row_base < num_rows ? (uint32_t)(num_rows - row_base) : 0u);\n"
        "}\n\n"
        "static __device__ __forceinline__ void cusr_native_cuda_reference_predicated_shared_load8_f32(\n"
        "    float* value0,\n"
        "    float* value1,\n"
        "    float* value2,\n"
        "    float* value3,\n"
        "    float* value4,\n"
        "    float* value5,\n"
        "    float* value6,\n"
        "    float* value7,\n"
        "    uint32_t address0,\n"
        "    uint32_t address1,\n"
        "    uint32_t address2,\n"
        "    uint32_t address3,\n"
        "    uint32_t address4,\n"
        "    uint32_t address5,\n"
        "    uint32_t address6,\n"
        "    uint32_t address7,\n"
        "    uint32_t predicate0,\n"
        "    uint32_t predicate1,\n"
        "    uint32_t predicate2,\n"
        "    uint32_t predicate3,\n"
        "    uint32_t predicate4,\n"
        "    uint32_t predicate5,\n"
        "    uint32_t predicate6,\n"
        "    uint32_t predicate7) {\n"
        "    asm volatile(\n"
        "        \"{\\n\\t\"\n"
        "        \".reg .pred %%%%q0, %%%%q1, %%%%q2, %%%%q3, %%%%q4, %%%%q5, %%%%q6, %%%%q7;\\n\\t\"\n"
        "        \"setp.ne.u32 %%%%q0, %%16, 0;\\n\\t\"\n"
        "        \"setp.ne.u32 %%%%q1, %%17, 0;\\n\\t\"\n"
        "        \"setp.ne.u32 %%%%q2, %%18, 0;\\n\\t\"\n"
        "        \"setp.ne.u32 %%%%q3, %%19, 0;\\n\\t\"\n"
        "        \"setp.ne.u32 %%%%q4, %%20, 0;\\n\\t\"\n"
        "        \"setp.ne.u32 %%%%q5, %%21, 0;\\n\\t\"\n"
        "        \"setp.ne.u32 %%%%q6, %%22, 0;\\n\\t\"\n"
        "        \"setp.ne.u32 %%%%q7, %%23, 0;\\n\\t\"\n"
        "        \"@%%%%q0 ld.shared.f32 %%0, [%%8];\\n\\t\"\n"
        "        \"@%%%%q1 ld.shared.f32 %%1, [%%9];\\n\\t\"\n"
        "        \"@%%%%q2 ld.shared.f32 %%2, [%%10];\\n\\t\"\n"
        "        \"@%%%%q3 ld.shared.f32 %%3, [%%11];\\n\\t\"\n"
        "        \"@%%%%q4 ld.shared.f32 %%4, [%%12];\\n\\t\"\n"
        "        \"@%%%%q5 ld.shared.f32 %%5, [%%13];\\n\\t\"\n"
        "        \"@%%%%q6 ld.shared.f32 %%6, [%%14];\\n\\t\"\n"
        "        \"@%%%%q7 ld.shared.f32 %%7, [%%15];\\n\\t\"\n"
        "        \"}\\n\"\n"
        "        : \"+f\"(*value0), \"+f\"(*value1), \"+f\"(*value2), \"+f\"(*value3), \"+f\"(*value4), \"+f\"(*value5), \"+f\"(*value6), \"+f\"(*value7)\n"
        "        : \"r\"(address0), \"r\"(address1), \"r\"(address2), \"r\"(address3), \"r\"(address4), \"r\"(address5), \"r\"(address6), \"r\"(address7),\n"
        "          \"r\"(predicate0), \"r\"(predicate1), \"r\"(predicate2), \"r\"(predicate3), \"r\"(predicate4), \"r\"(predicate5), \"r\"(predicate6), \"r\"(predicate7)\n"
        "        : \"memory\");\n"
        "}\n\n",
        tile_rows,
        threads_per_cta));

    _CUSR_NATIVE_CUDA_REFERENCE_GEN_ERROR_RET(CUSR_NATIVE_CUDA_REFERENCE_GEN_SUCCESS);
}

static CusrNativeCudaReferenceGenResult
cusr_native_cuda_reference_gen_emit_mixed_reduce_expr_body(CusrNativeCudaReferenceGenWriter* writer, uint32_t global_ast_idx)
{
    uint32_t term;
    const uint32_t first_leaf = cusr_native_cuda_reference_gen_reduce_leaf(global_ast_idx, 0u);

    _CUSR_NATIVE_CUDA_REFERENCE_GEN_CHECK_RET(cusr_native_cuda_reference_gen_write(writer,
        "    float output = value%u;\n",
        (unsigned)first_leaf));

    for (term = 1u; term < 8u; ++term) {
        const uint32_t leaf = cusr_native_cuda_reference_gen_reduce_leaf(global_ast_idx, term);
        const uint32_t op = cusr_native_cuda_reference_gen_reduce_op(global_ast_idx, term);

        switch (op) {
            case 0u:
                _CUSR_NATIVE_CUDA_REFERENCE_GEN_CHECK_RET(cusr_native_cuda_reference_gen_write(writer,
                    "    output = output + value%u;\n",
                    (unsigned)leaf));
                break;
            case 1u:
                _CUSR_NATIVE_CUDA_REFERENCE_GEN_CHECK_RET(cusr_native_cuda_reference_gen_write(writer,
                    "    output = output * value%u;\n",
                    (unsigned)leaf));
                break;
            case 2u:
                _CUSR_NATIVE_CUDA_REFERENCE_GEN_CHECK_RET(cusr_native_cuda_reference_gen_write(writer,
                    "    output = fminf(output, value%u);\n",
                    (unsigned)leaf));
                break;
            default:
                _CUSR_NATIVE_CUDA_REFERENCE_GEN_CHECK_RET(cusr_native_cuda_reference_gen_write(writer,
                    "    output = fmaxf(output, value%u);\n",
                    (unsigned)leaf));
                break;
        }
    }

    _CUSR_NATIVE_CUDA_REFERENCE_GEN_CHECK_RET(cusr_native_cuda_reference_gen_write(writer,
        "    return output;\n"));

    _CUSR_NATIVE_CUDA_REFERENCE_GEN_ERROR_RET(CUSR_NATIVE_CUDA_REFERENCE_GEN_SUCCESS);
}

static CusrNativeCudaReferenceGenResult
cusr_native_cuda_reference_gen_emit_depth3_binary(
    CusrNativeCudaReferenceGenWriter* writer,
    const char* dst,
    const char* lhs,
    const char* rhs,
    uint32_t op)
{
    switch (op & 3u) {
        case 0u:
            _CUSR_NATIVE_CUDA_REFERENCE_GEN_CHECK_RET(cusr_native_cuda_reference_gen_write(
                writer,
                "    const float %s = %s + %s;\n",
                dst,
                lhs,
                rhs));
            break;
        case 1u:
            _CUSR_NATIVE_CUDA_REFERENCE_GEN_CHECK_RET(cusr_native_cuda_reference_gen_write(
                writer,
                "    const float %s = %s * %s;\n",
                dst,
                lhs,
                rhs));
            break;
        case 2u:
            _CUSR_NATIVE_CUDA_REFERENCE_GEN_CHECK_RET(cusr_native_cuda_reference_gen_write(
                writer,
                "    const float %s = fminf(%s, %s);\n",
                dst,
                lhs,
                rhs));
            break;
        default:
            _CUSR_NATIVE_CUDA_REFERENCE_GEN_CHECK_RET(cusr_native_cuda_reference_gen_write(
                writer,
                "    const float %s = fmaxf(%s, %s);\n",
                dst,
                lhs,
                rhs));
            break;
    }

    _CUSR_NATIVE_CUDA_REFERENCE_GEN_ERROR_RET(CUSR_NATIVE_CUDA_REFERENCE_GEN_SUCCESS);
}

static CusrNativeCudaReferenceGenResult
cusr_native_cuda_reference_gen_emit_depth3_leaf(
    CusrNativeCudaReferenceGenWriter* writer,
    uint32_t global_ast_idx,
    uint32_t term,
    int use_mufu)
{
    const uint32_t leaf = cusr_native_cuda_reference_gen_reduce_leaf(global_ast_idx, term);

    if (!use_mufu) {
        _CUSR_NATIVE_CUDA_REFERENCE_GEN_CHECK_RET(cusr_native_cuda_reference_gen_write(
            writer,
            "    const float leaf%u = value%u;\n",
            (unsigned)term,
            (unsigned)leaf));
        _CUSR_NATIVE_CUDA_REFERENCE_GEN_ERROR_RET(CUSR_NATIVE_CUDA_REFERENCE_GEN_SUCCESS);
    }

    switch (cusr_native_cuda_reference_gen_depth3_unary_op(global_ast_idx, term)) {
        case 0u:
            _CUSR_NATIVE_CUDA_REFERENCE_GEN_CHECK_RET(cusr_native_cuda_reference_gen_write(
                writer,
                "    const float leaf%u = sinf(value%u);\n",
                (unsigned)term,
                (unsigned)leaf));
            break;
        case 1u:
            _CUSR_NATIVE_CUDA_REFERENCE_GEN_CHECK_RET(cusr_native_cuda_reference_gen_write(
                writer,
                "    const float leaf%u = cosf(value%u);\n",
                (unsigned)term,
                (unsigned)leaf));
            break;
        case 2u:
            _CUSR_NATIVE_CUDA_REFERENCE_GEN_CHECK_RET(cusr_native_cuda_reference_gen_write(
                writer,
                "    const float leaf%u = exp2f(value%u * 0.125f);\n",
                (unsigned)term,
                (unsigned)leaf));
            break;
        default:
            _CUSR_NATIVE_CUDA_REFERENCE_GEN_CHECK_RET(cusr_native_cuda_reference_gen_write(
                writer,
                "    const float leaf%u = rsqrtf(fabsf(value%u) + 0.25f);\n",
                (unsigned)term,
                (unsigned)leaf));
            break;
    }

    _CUSR_NATIVE_CUDA_REFERENCE_GEN_ERROR_RET(CUSR_NATIVE_CUDA_REFERENCE_GEN_SUCCESS);
}

static CusrNativeCudaReferenceGenResult
cusr_native_cuda_reference_gen_emit_depth3_expr_body(
    CusrNativeCudaReferenceGenWriter* writer,
    uint32_t global_ast_idx,
    int use_mufu)
{
    uint32_t term;
    uint32_t node = 0u;

    for (term = 0u; term < 8u; ++term) {
        _CUSR_NATIVE_CUDA_REFERENCE_GEN_CHECK_RET(cusr_native_cuda_reference_gen_emit_depth3_leaf(
            writer,
            global_ast_idx,
            term,
            use_mufu));
    }

    _CUSR_NATIVE_CUDA_REFERENCE_GEN_CHECK_RET(cusr_native_cuda_reference_gen_emit_depth3_binary(writer, "node0", "leaf0", "leaf1", cusr_native_cuda_reference_gen_reduce_op(global_ast_idx, node++)));
    _CUSR_NATIVE_CUDA_REFERENCE_GEN_CHECK_RET(cusr_native_cuda_reference_gen_emit_depth3_binary(writer, "node1", "leaf2", "leaf3", cusr_native_cuda_reference_gen_reduce_op(global_ast_idx, node++)));
    _CUSR_NATIVE_CUDA_REFERENCE_GEN_CHECK_RET(cusr_native_cuda_reference_gen_emit_depth3_binary(writer, "node2", "node0", "node1", cusr_native_cuda_reference_gen_reduce_op(global_ast_idx, node++)));
    _CUSR_NATIVE_CUDA_REFERENCE_GEN_CHECK_RET(cusr_native_cuda_reference_gen_emit_depth3_binary(writer, "node3", "leaf4", "leaf5", cusr_native_cuda_reference_gen_reduce_op(global_ast_idx, node++)));
    _CUSR_NATIVE_CUDA_REFERENCE_GEN_CHECK_RET(cusr_native_cuda_reference_gen_emit_depth3_binary(writer, "node4", "leaf6", "leaf7", cusr_native_cuda_reference_gen_reduce_op(global_ast_idx, node++)));
    _CUSR_NATIVE_CUDA_REFERENCE_GEN_CHECK_RET(cusr_native_cuda_reference_gen_emit_depth3_binary(writer, "node5", "node3", "node4", cusr_native_cuda_reference_gen_reduce_op(global_ast_idx, node++)));
    _CUSR_NATIVE_CUDA_REFERENCE_GEN_CHECK_RET(cusr_native_cuda_reference_gen_emit_depth3_binary(writer, "output", "node2", "node5", cusr_native_cuda_reference_gen_reduce_op(global_ast_idx, node++)));
    _CUSR_NATIVE_CUDA_REFERENCE_GEN_CHECK_RET(cusr_native_cuda_reference_gen_write(writer, "    return output;\n"));

    _CUSR_NATIVE_CUDA_REFERENCE_GEN_ERROR_RET(CUSR_NATIVE_CUDA_REFERENCE_GEN_SUCCESS);
}

static CusrNativeCudaReferenceGenResult
cusr_native_cuda_reference_gen_emit_expr(
    CusrNativeCudaReferenceGenWriter* writer,
    size_t kernel_idx,
    size_t ast_idx,
    size_t asts_per_kernel,
    unsigned expr_mode)
{
    const int packed = asts_per_kernel != 1u;
    const size_t global_ast_idx = kernel_idx * asts_per_kernel + ast_idx;
    const double ast_offset = (double)global_ast_idx;

    if (packed) {
        _CUSR_NATIVE_CUDA_REFERENCE_GEN_CHECK_RET(cusr_native_cuda_reference_gen_write(writer,
            "static __device__ __forceinline__ float cusr_native_cuda_reference_expr_%03u_%03u(\n",
            (unsigned)kernel_idx,
            (unsigned)ast_idx));
    } else {
        _CUSR_NATIVE_CUDA_REFERENCE_GEN_CHECK_RET(cusr_native_cuda_reference_gen_write(writer,
            "static __device__ __forceinline__ float cusr_native_cuda_reference_expr_%03u(\n",
            (unsigned)kernel_idx));
    }

    _CUSR_NATIVE_CUDA_REFERENCE_GEN_CHECK_RET(cusr_native_cuda_reference_gen_write(writer,
        "    float value0,\n"
        "    float value1,\n"
        "    float value2,\n"
        "    float value3,\n"
        "    float value4,\n"
        "    float value5,\n"
        "    float value6,\n"
        "    float value7\n"
        ") {\n"));

    if (expr_mode == CUSR_NATIVE_CUDA_REFERENCE_EXPR_MIXED_REDUCE) {
        _CUSR_NATIVE_CUDA_REFERENCE_GEN_CHECK_RET(cusr_native_cuda_reference_gen_emit_mixed_reduce_expr_body(writer, (uint32_t)global_ast_idx));
        _CUSR_NATIVE_CUDA_REFERENCE_GEN_CHECK_RET(cusr_native_cuda_reference_gen_write(writer, "}\n\n"));
        _CUSR_NATIVE_CUDA_REFERENCE_GEN_ERROR_RET(CUSR_NATIVE_CUDA_REFERENCE_GEN_SUCCESS);
    }

    if (expr_mode == CUSR_NATIVE_CUDA_REFERENCE_EXPR_DEPTH3_ALU ||
        expr_mode == CUSR_NATIVE_CUDA_REFERENCE_EXPR_DEPTH3_MUFU) {
        _CUSR_NATIVE_CUDA_REFERENCE_GEN_CHECK_RET(cusr_native_cuda_reference_gen_emit_depth3_expr_body(
            writer,
            (uint32_t)global_ast_idx,
            expr_mode == CUSR_NATIVE_CUDA_REFERENCE_EXPR_DEPTH3_MUFU));
        _CUSR_NATIVE_CUDA_REFERENCE_GEN_CHECK_RET(cusr_native_cuda_reference_gen_write(writer, "}\n\n"));
        _CUSR_NATIVE_CUDA_REFERENCE_GEN_ERROR_RET(CUSR_NATIVE_CUDA_REFERENCE_GEN_SUCCESS);
    }

    _CUSR_NATIVE_CUDA_REFERENCE_GEN_CHECK_RET(cusr_native_cuda_reference_gen_write(writer,
        "    const float leaf0 = value0 + %.1ff;\n"
        "    const float leaf1 = value1 + %.1ff;\n"
        "    const float leaf2 = value2 + %.1ff;\n"
        "    const float leaf3 = value3 + %.1ff;\n"
        "    const float leaf4 = value4 + %.1ff;\n"
        "    const float leaf5 = value5 + %.1ff;\n"
        "    const float leaf6 = value6 + %.1ff;\n"
        "    const float leaf7 = value7 + %.1ff;\n"
        "    const float reduce0 = fminf(leaf0, leaf1);\n"
        "    const float reduce1 = fmaxf(leaf2, leaf3);\n"
        "    const float reduce2 = fminf(leaf4, leaf5);\n"
        "    const float reduce3 = fmaxf(leaf6, leaf7);\n"
        "    const float reduce4 = fminf(reduce0, reduce1);\n"
        "    const float reduce5 = fmaxf(reduce2, reduce3);\n"
        "    return fminf(reduce4, reduce5);\n"
        "}\n\n",
        ast_offset,
        ast_offset,
        ast_offset,
        ast_offset,
        ast_offset,
        ast_offset,
        ast_offset,
        ast_offset));

    _CUSR_NATIVE_CUDA_REFERENCE_GEN_ERROR_RET(CUSR_NATIVE_CUDA_REFERENCE_GEN_SUCCESS);
}

static CusrNativeCudaReferenceGenResult
cusr_native_cuda_reference_gen_emit_ast_eval(CusrNativeCudaReferenceGenWriter* writer, size_t kernel_idx, size_t ast_idx, size_t asts_per_kernel)
{
    const int packed = asts_per_kernel != 1u;

    if (packed) {
        _CUSR_NATIVE_CUDA_REFERENCE_GEN_CHECK_RET(cusr_native_cuda_reference_gen_write(writer,
            "            const float ast_output_%03u = cusr_native_cuda_reference_expr_%03u_%03u(\n"
            "                value0,\n"
            "                value1,\n"
            "                value2,\n"
            "                value3,\n"
            "                value4,\n"
            "                value5,\n"
            "                value6,\n"
            "                value7);\n",
            (unsigned)ast_idx,
            (unsigned)kernel_idx,
            (unsigned)ast_idx));
    } else {
        _CUSR_NATIVE_CUDA_REFERENCE_GEN_CHECK_RET(cusr_native_cuda_reference_gen_write(writer,
            "            const float ast_output_%03u = cusr_native_cuda_reference_expr_%03u(\n"
            "                value0,\n"
            "                value1,\n"
            "                value2,\n"
            "                value3,\n"
            "                value4,\n"
            "                value5,\n"
            "                value6,\n"
            "                value7);\n",
            (unsigned)ast_idx,
            (unsigned)kernel_idx));
    }

    _CUSR_NATIVE_CUDA_REFERENCE_GEN_CHECK_RET(cusr_native_cuda_reference_gen_write(writer,
        "            const float error_%03u = ast_output_%03u - target;\n"
        "            sse_%03u = error_%03u * error_%03u + sse_%03u;\n",
        (unsigned)ast_idx,
        (unsigned)ast_idx,
        (unsigned)ast_idx,
        (unsigned)ast_idx,
        (unsigned)ast_idx,
        (unsigned)ast_idx));

    _CUSR_NATIVE_CUDA_REFERENCE_GEN_ERROR_RET(CUSR_NATIVE_CUDA_REFERENCE_GEN_SUCCESS);
}

static CusrNativeCudaReferenceGenResult
cusr_native_cuda_reference_gen_emit_score(CusrNativeCudaReferenceGenWriter* writer, size_t kernel_idx, size_t asts_per_kernel)
{
    size_t ast_idx;

    _CUSR_NATIVE_CUDA_REFERENCE_GEN_CHECK_RET(cusr_native_cuda_reference_gen_write(writer,
        "static __device__ __forceinline__ void cusr_native_cuda_reference_score_asts_%03u(\n"
        "    const float* x_tile,\n"
        "    const float* target_tile,\n"
        "    uint32_t row_limit,\n"
        "    uint32_t shared_stride,\n"
        "    const uint8_t* __restrict__ leaf_masks,\n"
        "    const uint32_t* __restrict__ leaf_words,\n"
        "    size_t leaf_words_stride,\n"
        "    uint32_t num_settings,\n"
        "    float* __restrict__ output_sse\n"
        ") {\n"
        "    for (uint32_t setting_base = 0u; setting_base < num_settings; setting_base += CUSR_NATIVE_CUDA_REFERENCE_THREADS) {\n"
        "        const uint32_t setting = setting_base + threadIdx.x;\n"
        "        if (setting >= num_settings) {\n"
        "            continue;\n"
        "        }\n"
        "        const uint8_t mask = leaf_masks[setting];\n"
        "        const uint32_t* const words = leaf_words + (size_t)setting * leaf_words_stride;\n"
        "        const uint32_t word0 = words[0u];\n"
        "        const uint32_t word1 = words[1u];\n"
        "        const uint32_t word2 = words[2u];\n"
        "        const uint32_t word3 = words[3u];\n"
        "        const uint32_t word4 = words[4u];\n"
        "        const uint32_t word5 = words[5u];\n"
        "        const uint32_t word6 = words[6u];\n"
        "        const uint32_t word7 = words[7u];\n"
        "        const uint32_t is_column0 = cusr_native_cuda_reference_leaf_is_column(mask, 0u);\n"
        "        const uint32_t is_column1 = cusr_native_cuda_reference_leaf_is_column(mask, 1u);\n"
        "        const uint32_t is_column2 = cusr_native_cuda_reference_leaf_is_column(mask, 2u);\n"
        "        const uint32_t is_column3 = cusr_native_cuda_reference_leaf_is_column(mask, 3u);\n"
        "        const uint32_t is_column4 = cusr_native_cuda_reference_leaf_is_column(mask, 4u);\n"
        "        const uint32_t is_column5 = cusr_native_cuda_reference_leaf_is_column(mask, 5u);\n"
        "        const uint32_t is_column6 = cusr_native_cuda_reference_leaf_is_column(mask, 6u);\n"
        "        const uint32_t is_column7 = cusr_native_cuda_reference_leaf_is_column(mask, 7u);\n"
        "        const uint32_t x_tile_address = (uint32_t)__cvta_generic_to_shared((const void*)x_tile);\n"
        "        const uint32_t shared_stride_bytes = shared_stride * sizeof(float);\n"
        "        const uint32_t base_address0 = x_tile_address + word0 * sizeof(float);\n"
        "        const uint32_t base_address1 = x_tile_address + word1 * sizeof(float);\n"
        "        const uint32_t base_address2 = x_tile_address + word2 * sizeof(float);\n"
        "        const uint32_t base_address3 = x_tile_address + word3 * sizeof(float);\n"
        "        const uint32_t base_address4 = x_tile_address + word4 * sizeof(float);\n"
        "        const uint32_t base_address5 = x_tile_address + word5 * sizeof(float);\n"
        "        const uint32_t base_address6 = x_tile_address + word6 * sizeof(float);\n"
        "        const uint32_t base_address7 = x_tile_address + word7 * sizeof(float);\n"
        "        float value0 = __uint_as_float(word0);\n"
        "        float value1 = __uint_as_float(word1);\n"
        "        float value2 = __uint_as_float(word2);\n"
        "        float value3 = __uint_as_float(word3);\n"
        "        float value4 = __uint_as_float(word4);\n"
        "        float value5 = __uint_as_float(word5);\n"
        "        float value6 = __uint_as_float(word6);\n"
        "        float value7 = __uint_as_float(word7);\n"
        "        uint32_t address0 = base_address0;\n"
        "        uint32_t address1 = base_address1;\n"
        "        uint32_t address2 = base_address2;\n"
        "        uint32_t address3 = base_address3;\n"
        "        uint32_t address4 = base_address4;\n"
        "        uint32_t address5 = base_address5;\n"
        "        uint32_t address6 = base_address6;\n"
        "        uint32_t address7 = base_address7;\n",
        (unsigned)kernel_idx));

    for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
        _CUSR_NATIVE_CUDA_REFERENCE_GEN_CHECK_RET(cusr_native_cuda_reference_gen_write(writer,
            "        float sse_%03u = 0.0f;\n",
            (unsigned)ast_idx));
    }

    _CUSR_NATIVE_CUDA_REFERENCE_GEN_CHECK_RET(cusr_native_cuda_reference_gen_write(writer,
        "        #pragma unroll 1\n"
        "        for (uint32_t eval_row = 0u; eval_row < row_limit; ++eval_row) {\n"
        "            cusr_native_cuda_reference_predicated_shared_load8_f32(\n"
        "                &value0,\n"
        "                &value1,\n"
        "                &value2,\n"
        "                &value3,\n"
        "                &value4,\n"
        "                &value5,\n"
        "                &value6,\n"
        "                &value7,\n"
        "                address0,\n"
        "                address1,\n"
        "                address2,\n"
        "                address3,\n"
        "                address4,\n"
        "                address5,\n"
        "                address6,\n"
        "                address7,\n"
        "                is_column0,\n"
        "                is_column1,\n"
        "                is_column2,\n"
        "                is_column3,\n"
        "                is_column4,\n"
        "                is_column5,\n"
        "                is_column6,\n"
        "                is_column7);\n"
        "            const float target = target_tile[eval_row];\n"));

    for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
        _CUSR_NATIVE_CUDA_REFERENCE_GEN_CHECK_RET(cusr_native_cuda_reference_gen_emit_ast_eval(writer, kernel_idx, ast_idx, asts_per_kernel));
    }

    _CUSR_NATIVE_CUDA_REFERENCE_GEN_CHECK_RET(cusr_native_cuda_reference_gen_write(writer,
        "            address0 += shared_stride_bytes;\n"
        "            address1 += shared_stride_bytes;\n"
        "            address2 += shared_stride_bytes;\n"
        "            address3 += shared_stride_bytes;\n"
        "            address4 += shared_stride_bytes;\n"
        "            address5 += shared_stride_bytes;\n"
        "            address6 += shared_stride_bytes;\n"
        "            address7 += shared_stride_bytes;\n"
        "        }\n"));

    for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
        _CUSR_NATIVE_CUDA_REFERENCE_GEN_CHECK_RET(cusr_native_cuda_reference_gen_write(writer,
            "        atomicAdd(output_sse + (size_t)%uu * num_settings + setting, sse_%03u);\n",
            (unsigned)ast_idx,
            (unsigned)ast_idx));
    }

    _CUSR_NATIVE_CUDA_REFERENCE_GEN_CHECK_RET(cusr_native_cuda_reference_gen_write(writer,
        "    }\n"
        "}\n\n"));

    _CUSR_NATIVE_CUDA_REFERENCE_GEN_ERROR_RET(CUSR_NATIVE_CUDA_REFERENCE_GEN_SUCCESS);
}

static CusrNativeCudaReferenceGenResult
cusr_native_cuda_reference_gen_emit_kernel(CusrNativeCudaReferenceGenWriter* writer, size_t kernel_idx)
{
    _CUSR_NATIVE_CUDA_REFERENCE_GEN_CHECK_RET(cusr_native_cuda_reference_gen_write(writer,
        "extern \"C\" __global__\n"
        "void cusr_native_cuda_reference_f32_%03u(\n"
        "    const float* __restrict__ X,\n"
        "    const float* __restrict__ target,\n"
        "    size_t num_rows,\n"
        "    uint32_t num_columns,\n"
        "    size_t leading_dim,\n"
        "    const uint8_t* __restrict__ leaf_masks,\n"
        "    const uint32_t* __restrict__ leaf_words,\n"
        "    size_t leaf_words_stride,\n"
        "    uint32_t num_settings,\n"
        "    float* __restrict__ output_sse\n"
        ") {\n"
        "    extern __shared__ float shared[];\n"
        "    const uint32_t shared_stride = cusr_native_cuda_reference_shared_stride(num_columns);\n"
        "    float* const x_tile = shared;\n"
        "    float* const target_tile = shared + (size_t)CUSR_NATIVE_CUDA_REFERENCE_TILE_ROWS * shared_stride;\n"
        "    const uint32_t row_limit = cusr_native_cuda_reference_load_tile_f32(\n"
        "        X,\n"
        "        target,\n"
        "        num_rows,\n"
        "        num_columns,\n"
        "        leading_dim,\n"
        "        x_tile,\n"
        "        target_tile,\n"
        "        shared_stride);\n"
        "    __syncthreads();\n"
        "    cusr_native_cuda_reference_score_asts_%03u(\n"
        "        x_tile,\n"
        "        target_tile,\n"
        "        row_limit,\n"
        "        shared_stride,\n"
        "        leaf_masks,\n"
        "        leaf_words,\n"
        "        leaf_words_stride,\n"
        "        num_settings,\n"
        "        output_sse);\n"
        "}\n\n",
        (unsigned)kernel_idx,
        (unsigned)kernel_idx));

    _CUSR_NATIVE_CUDA_REFERENCE_GEN_ERROR_RET(CUSR_NATIVE_CUDA_REFERENCE_GEN_SUCCESS);
}

static CusrNativeCudaReferenceGenResult
cusr_native_cuda_reference_gen_cuda_impl(
    CusrNativeCudaReferenceGenWriter* writer,
    size_t kernels_per_file,
    size_t asts_per_kernel,
    size_t tile_rows,
    size_t threads_per_cta,
    unsigned expr_mode)
{
    size_t kernel_idx;
    size_t total_asts;

    if (kernels_per_file == 0u || asts_per_kernel == 0u ||
        (tile_rows != 64u && tile_rows != 128u && tile_rows != 256u) ||
        (threads_per_cta != 64u && threads_per_cta != 128u && threads_per_cta != 256u) ||
        (expr_mode != CUSR_NATIVE_CUDA_REFERENCE_EXPR_OFFSET_MINMAX &&
            expr_mode != CUSR_NATIVE_CUDA_REFERENCE_EXPR_MIXED_REDUCE &&
            expr_mode != CUSR_NATIVE_CUDA_REFERENCE_EXPR_DEPTH3_ALU &&
            expr_mode != CUSR_NATIVE_CUDA_REFERENCE_EXPR_DEPTH3_MUFU)) {
        _CUSR_NATIVE_CUDA_REFERENCE_GEN_ERROR_RET(CUSR_NATIVE_CUDA_REFERENCE_GEN_ERROR_INVALID_VALUE);
    }

    if (!cusr_native_cuda_reference_gen_checked_mul(kernels_per_file, asts_per_kernel, &total_asts)) {
        _CUSR_NATIVE_CUDA_REFERENCE_GEN_ERROR_RET(CUSR_NATIVE_CUDA_REFERENCE_GEN_ERROR_OVERFLOW);
    }

    (void)total_asts;
    _CUSR_NATIVE_CUDA_REFERENCE_GEN_CHECK_RET(cusr_native_cuda_reference_gen_emit_prelude(writer, tile_rows, threads_per_cta));

    for (kernel_idx = 0u; kernel_idx < kernels_per_file; ++kernel_idx) {
        size_t ast_idx;

        for (ast_idx = 0u; ast_idx < asts_per_kernel; ++ast_idx) {
            _CUSR_NATIVE_CUDA_REFERENCE_GEN_CHECK_RET(cusr_native_cuda_reference_gen_emit_expr(writer, kernel_idx, ast_idx, asts_per_kernel, expr_mode));
        }

        _CUSR_NATIVE_CUDA_REFERENCE_GEN_CHECK_RET(cusr_native_cuda_reference_gen_emit_score(writer, kernel_idx, asts_per_kernel));
        _CUSR_NATIVE_CUDA_REFERENCE_GEN_CHECK_RET(cusr_native_cuda_reference_gen_emit_kernel(writer, kernel_idx));
    }

    _CUSR_NATIVE_CUDA_REFERENCE_GEN_ERROR_RET(CUSR_NATIVE_CUDA_REFERENCE_GEN_SUCCESS);
}

CUSR_NATIVE_CUDA_REFERENCE_GEN_PUBLIC_DEF const char*
cusr_native_cuda_reference_gen_result_to_string(CusrNativeCudaReferenceGenResult result)
{
    switch (result) {
        case CUSR_NATIVE_CUDA_REFERENCE_GEN_SUCCESS:
            return "CUSR_NATIVE_CUDA_REFERENCE_GEN_SUCCESS";
        case CUSR_NATIVE_CUDA_REFERENCE_GEN_ERROR_INVALID_VALUE:
            return "CUSR_NATIVE_CUDA_REFERENCE_GEN_ERROR_INVALID_VALUE";
        case CUSR_NATIVE_CUDA_REFERENCE_GEN_ERROR_OVERFLOW:
            return "CUSR_NATIVE_CUDA_REFERENCE_GEN_ERROR_OVERFLOW";
        case CUSR_NATIVE_CUDA_REFERENCE_GEN_ERROR_INSUFFICIENT_BUFFER:
            return "CUSR_NATIVE_CUDA_REFERENCE_GEN_ERROR_INSUFFICIENT_BUFFER";
        case CUSR_NATIVE_CUDA_REFERENCE_GEN_ERROR_FORMAT:
            return "CUSR_NATIVE_CUDA_REFERENCE_GEN_ERROR_FORMAT";
    }

    return "CUSR_NATIVE_CUDA_REFERENCE_GEN_ERROR_UNKNOWN";
}

CUSR_NATIVE_CUDA_REFERENCE_GEN_PUBLIC_DEF CusrNativeCudaReferenceGenResult
cusr_native_cuda_reference_gen_cuda_size(
    size_t kernels_per_file,
    size_t asts_per_kernel,
    size_t tile_rows,
    size_t threads_per_cta,
    unsigned expr_mode,
    size_t* size_ret)
{
    CusrNativeCudaReferenceGenWriter writer;
    CusrNativeCudaReferenceGenResult result;

    if (size_ret == NULL) {
        _CUSR_NATIVE_CUDA_REFERENCE_GEN_ERROR_RET(CUSR_NATIVE_CUDA_REFERENCE_GEN_ERROR_INVALID_VALUE);
    }

    writer.buffer = NULL;
    writer.buffer_size = 0u;
    writer.size = 0u;
    writer.result = CUSR_NATIVE_CUDA_REFERENCE_GEN_SUCCESS;

    result = cusr_native_cuda_reference_gen_cuda_impl(
        &writer,
        kernels_per_file,
        asts_per_kernel,
        tile_rows,
        threads_per_cta,
        expr_mode);
    if (result != CUSR_NATIVE_CUDA_REFERENCE_GEN_SUCCESS) {
        _CUSR_NATIVE_CUDA_REFERENCE_GEN_ERROR_RET(result);
    }

    if (writer.size == SIZE_MAX) {
        _CUSR_NATIVE_CUDA_REFERENCE_GEN_ERROR_RET(CUSR_NATIVE_CUDA_REFERENCE_GEN_ERROR_OVERFLOW);
    }

    *size_ret = writer.size + 1u;
    _CUSR_NATIVE_CUDA_REFERENCE_GEN_ERROR_RET(CUSR_NATIVE_CUDA_REFERENCE_GEN_SUCCESS);
}

CUSR_NATIVE_CUDA_REFERENCE_GEN_PUBLIC_DEF CusrNativeCudaReferenceGenResult
cusr_native_cuda_reference_gen_cuda(
    char* buffer,
    size_t buffer_size,
    size_t kernels_per_file,
    size_t asts_per_kernel,
    size_t tile_rows,
    size_t threads_per_cta,
    unsigned expr_mode,
    size_t* size_ret)
{
    CusrNativeCudaReferenceGenWriter writer;
    CusrNativeCudaReferenceGenResult result;
    size_t required_size = 0u;

    if (buffer == NULL || size_ret == NULL) {
        _CUSR_NATIVE_CUDA_REFERENCE_GEN_ERROR_RET(CUSR_NATIVE_CUDA_REFERENCE_GEN_ERROR_INVALID_VALUE);
    }

    _CUSR_NATIVE_CUDA_REFERENCE_GEN_CHECK_RET(cusr_native_cuda_reference_gen_cuda_size(
        kernels_per_file,
        asts_per_kernel,
        tile_rows,
        threads_per_cta,
        expr_mode,
        &required_size));

    if (buffer_size < required_size) {
        *size_ret = required_size;
        _CUSR_NATIVE_CUDA_REFERENCE_GEN_ERROR_RET(CUSR_NATIVE_CUDA_REFERENCE_GEN_ERROR_INSUFFICIENT_BUFFER);
    }

    writer.buffer = buffer;
    writer.buffer_size = buffer_size;
    writer.size = 0u;
    writer.result = CUSR_NATIVE_CUDA_REFERENCE_GEN_SUCCESS;

    result = cusr_native_cuda_reference_gen_cuda_impl(
        &writer,
        kernels_per_file,
        asts_per_kernel,
        tile_rows,
        threads_per_cta,
        expr_mode);
    if (result != CUSR_NATIVE_CUDA_REFERENCE_GEN_SUCCESS) {
        *size_ret = required_size;
        _CUSR_NATIVE_CUDA_REFERENCE_GEN_ERROR_RET(result);
    }

    buffer[writer.size] = '\0';
    *size_ret = required_size;
    _CUSR_NATIVE_CUDA_REFERENCE_GEN_ERROR_RET(CUSR_NATIVE_CUDA_REFERENCE_GEN_SUCCESS);
}

#undef _CUSR_NATIVE_CUDA_REFERENCE_GEN_CHECK_RET
#undef _CUSR_NATIVE_CUDA_REFERENCE_GEN_ERROR_RET

#endif /* CUSR_NATIVE_CUDA_REFERENCE_GEN_IMPLEMENTATION_ONCE */
#endif /* CUSR_NATIVE_CUDA_REFERENCE_GEN_IMPLEMENTATION */
