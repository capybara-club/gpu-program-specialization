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
#ifndef CUSR_TILE_STATIC_EVAL_TEMPLATE_CUH_INCLUDED
#define CUSR_TILE_STATIC_EVAL_TEMPLATE_CUH_INCLUDED

#if defined(__CUDACC_RTC__)
#ifndef CUSR_RTC_TYPES_DEFINED
#define CUSR_RTC_TYPES_DEFINED
typedef unsigned char uint8_t;
typedef unsigned int uint32_t;
#endif
#else
#include <stddef.h>
#include <stdint.h>
#endif

#define CUSR_TILE_STATIC_EVAL_TEMPLATE_FIRST_MARKER_BITS 0x7fc0ffeeu
#define CUSR_TILE_STATIC_EVAL_TEMPLATE_MARKER_STRIDE 128u
#define CUSR_TILE_STATIC_EVAL_TEMPLATE_OUTPUT_MARKER_OFFSET 16u
#define CUSR_TILE_STATIC_EVAL_TEMPLATE_PAD_INSTRUCTIONS_PER_AST 64u

template <uint32_t NumBrkpts>
static __device__ __forceinline__ void
cusr_tile_static_eval_template_emit_brkpt_pad(void)
{
    #pragma unroll
    for (uint32_t i = 0u; i < NumBrkpts; ++i) {
        asm volatile("brkpt;" ::: "memory");
    }
}

template <uint32_t AstCapacity>
struct CusrTileStaticEvalTemplateValues;

template <>
struct CusrTileStaticEvalTemplateValues<8u> {
    float value_000;
    float value_001;
    float value_002;
    float value_003;
    float value_004;
    float value_005;
    float value_006;
    float value_007;
};

template <>
struct CusrTileStaticEvalTemplateValues<16u> {
    float value_000;
    float value_001;
    float value_002;
    float value_003;
    float value_004;
    float value_005;
    float value_006;
    float value_007;
    float value_008;
    float value_009;
    float value_010;
    float value_011;
    float value_012;
    float value_013;
    float value_014;
    float value_015;
};

template <>
struct CusrTileStaticEvalTemplateValues<32u> {
    float value_000;
    float value_001;
    float value_002;
    float value_003;
    float value_004;
    float value_005;
    float value_006;
    float value_007;
    float value_008;
    float value_009;
    float value_010;
    float value_011;
    float value_012;
    float value_013;
    float value_014;
    float value_015;
    float value_016;
    float value_017;
    float value_018;
    float value_019;
    float value_020;
    float value_021;
    float value_022;
    float value_023;
    float value_024;
    float value_025;
    float value_026;
    float value_027;
    float value_028;
    float value_029;
    float value_030;
    float value_031;
};

static __device__ __forceinline__ uint32_t
cusr_tile_static_eval_template_leaf_is_column(uint8_t mask, uint32_t leaf)
{
    return (((uint32_t)mask) >> leaf) & 1u;
}

static __device__ __forceinline__ uint32_t
cusr_tile_static_eval_template_shared_stride(uint32_t num_columns)
{
    return num_columns | 1u;
}

template <uint32_t TileRows, uint32_t ThreadsPerCta>
static __device__ __forceinline__ uint32_t
cusr_tile_static_eval_template_load_tile_f32(
    const float* __restrict__ X,
    size_t num_rows,
    uint32_t num_columns,
    size_t leading_dim,
    float* x_tile,
    uint32_t shared_stride)
{
    const size_t row_base = (size_t)blockIdx.x * TileRows;

    #pragma unroll
    for (uint32_t tile_row = threadIdx.x; tile_row < TileRows; tile_row += ThreadsPerCta) {
        const size_t row = row_base + tile_row;
        const bool row_valid = row < num_rows;

        #pragma unroll 1
        for (uint32_t column = 0u; column < num_columns; ++column) {
            x_tile[(size_t)tile_row * shared_stride + column] = row_valid ? X[(size_t)column * leading_dim + row] : 0.0f;
        }
    }

    return row_base + TileRows <= num_rows ? TileRows : (row_base < num_rows ? (uint32_t)(num_rows - row_base) : 0u);
}

static __device__ __forceinline__ void
cusr_tile_static_eval_template_predicated_shared_load8_f32(
    float* value0, float* value1, float* value2, float* value3,
    float* value4, float* value5, float* value6, float* value7,
    uint32_t address0, uint32_t address1, uint32_t address2, uint32_t address3,
    uint32_t address4, uint32_t address5, uint32_t address6, uint32_t address7,
    uint32_t predicate0, uint32_t predicate1, uint32_t predicate2, uint32_t predicate3,
    uint32_t predicate4, uint32_t predicate5, uint32_t predicate6, uint32_t predicate7)
{
    asm volatile(
        "{\n\t"
        ".reg .pred %%q0, %%q1, %%q2, %%q3, %%q4, %%q5, %%q6, %%q7;\n\t"
        "setp.ne.u32 %%q0, %16, 0;\n\t"
        "setp.ne.u32 %%q1, %17, 0;\n\t"
        "setp.ne.u32 %%q2, %18, 0;\n\t"
        "setp.ne.u32 %%q3, %19, 0;\n\t"
        "setp.ne.u32 %%q4, %20, 0;\n\t"
        "setp.ne.u32 %%q5, %21, 0;\n\t"
        "setp.ne.u32 %%q6, %22, 0;\n\t"
        "setp.ne.u32 %%q7, %23, 0;\n\t"
        "@%%q0 ld.shared.f32 %0, [%8];\n\t"
        "@%%q1 ld.shared.f32 %1, [%9];\n\t"
        "@%%q2 ld.shared.f32 %2, [%10];\n\t"
        "@%%q3 ld.shared.f32 %3, [%11];\n\t"
        "@%%q4 ld.shared.f32 %4, [%12];\n\t"
        "@%%q5 ld.shared.f32 %5, [%13];\n\t"
        "@%%q6 ld.shared.f32 %6, [%14];\n\t"
        "@%%q7 ld.shared.f32 %7, [%15];\n\t"
        "}\n"
        : "+f"(*value0), "+f"(*value1), "+f"(*value2), "+f"(*value3), "+f"(*value4), "+f"(*value5), "+f"(*value6), "+f"(*value7)
        : "r"(address0), "r"(address1), "r"(address2), "r"(address3), "r"(address4), "r"(address5), "r"(address6), "r"(address7),
          "r"(predicate0), "r"(predicate1), "r"(predicate2), "r"(predicate3), "r"(predicate4), "r"(predicate5), "r"(predicate6), "r"(predicate7)
        : "memory");
}

#define CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_INPUT(input_operand, scratch_operand) \
    "mov.b32 %%marker, %%marker_bits;\n\t" \
    "add.rn.ftz.f32 %%anchor, %" #input_operand ", %%marker;\n\t" \
    "mov.b32 %%anchor_bits, %%anchor;\n\t" \
    "st.volatile.shared.u32 [%" #scratch_operand "], %%anchor_bits;\n\t" \
    "brkpt;\n\t" \
    "add.u32 %%marker_bits, %%marker_bits, 1;\n\t"

#define CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(output_operand) \
    "mov.b32 %%marker, %%marker_bits;\n\t" \
    "add.rn.ftz.f32 %" #output_operand ", %" #output_operand ", %%marker;\n\t" \
    "brkpt;\n\t" \
    "add.u32 %%marker_bits, %%marker_bits, 1;\n\t"

template <uint32_t KernelIndex>
static __device__ __forceinline__ void
cusr_tile_static_eval_template_site(
    uint32_t scratch_address,
    float value0, float value1, float value2, float value3,
    float value4, float value5, float value6, float value7,
    CusrTileStaticEvalTemplateValues<8u>* values)
{
    asm volatile(
        "{\n\t"
        ".reg .f32 %%anchor, %%marker;\n\t"
        ".reg .u32 %%anchor_bits, %%marker_bits;\n\t"
        "brkpt;\n\t"
        "mov.u32 %%marker_bits, %17;\n\t"
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_INPUT(8, 16)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_INPUT(9, 16)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_INPUT(10, 16)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_INPUT(11, 16)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_INPUT(12, 16)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_INPUT(13, 16)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_INPUT(14, 16)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_INPUT(15, 16)
        /* Direct eval keeps the shared target role by aliasing input 0. */
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_INPUT(8, 16)
        "add.u32 %%marker_bits, %%marker_bits, 7;\n\t"
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(0)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(1)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(2)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(3)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(4)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(5)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(6)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(7)
        "}\n"
        : "+f"(values->value_000), "+f"(values->value_001), "+f"(values->value_002), "+f"(values->value_003),
          "+f"(values->value_004), "+f"(values->value_005), "+f"(values->value_006), "+f"(values->value_007)
        : "f"(value0), "f"(value1), "f"(value2), "f"(value3), "f"(value4), "f"(value5), "f"(value6), "f"(value7),
          "r"(scratch_address),
          "n"(CUSR_TILE_STATIC_EVAL_TEMPLATE_FIRST_MARKER_BITS + KernelIndex * CUSR_TILE_STATIC_EVAL_TEMPLATE_MARKER_STRIDE)
        : "memory");
    cusr_tile_static_eval_template_emit_brkpt_pad<8u * CUSR_TILE_STATIC_EVAL_TEMPLATE_PAD_INSTRUCTIONS_PER_AST>();
}

template <uint32_t KernelIndex>
static __device__ __forceinline__ void
cusr_tile_static_eval_template_site(
    uint32_t scratch_address,
    float value0, float value1, float value2, float value3,
    float value4, float value5, float value6, float value7,
    CusrTileStaticEvalTemplateValues<16u>* values)
{
    asm volatile(
        "{\n\t"
        ".reg .f32 %%anchor, %%marker;\n\t"
        ".reg .u32 %%anchor_bits, %%marker_bits;\n\t"
        "brkpt;\n\t"
        "mov.u32 %%marker_bits, %25;\n\t"
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_INPUT(16, 24)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_INPUT(17, 24)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_INPUT(18, 24)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_INPUT(19, 24)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_INPUT(20, 24)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_INPUT(21, 24)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_INPUT(22, 24)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_INPUT(23, 24)
        /* Direct eval keeps the shared target role by aliasing input 0. */
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_INPUT(16, 24)
        "add.u32 %%marker_bits, %%marker_bits, 7;\n\t"
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(0)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(1)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(2)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(3)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(4)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(5)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(6)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(7)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(8)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(9)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(10)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(11)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(12)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(13)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(14)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(15)
        "}\n"
        : "+f"(values->value_000), "+f"(values->value_001), "+f"(values->value_002), "+f"(values->value_003),
          "+f"(values->value_004), "+f"(values->value_005), "+f"(values->value_006), "+f"(values->value_007),
          "+f"(values->value_008), "+f"(values->value_009), "+f"(values->value_010), "+f"(values->value_011),
          "+f"(values->value_012), "+f"(values->value_013), "+f"(values->value_014), "+f"(values->value_015)
        : "f"(value0), "f"(value1), "f"(value2), "f"(value3), "f"(value4), "f"(value5), "f"(value6), "f"(value7),
          "r"(scratch_address),
          "n"(CUSR_TILE_STATIC_EVAL_TEMPLATE_FIRST_MARKER_BITS + KernelIndex * CUSR_TILE_STATIC_EVAL_TEMPLATE_MARKER_STRIDE)
        : "memory");
    cusr_tile_static_eval_template_emit_brkpt_pad<16u * CUSR_TILE_STATIC_EVAL_TEMPLATE_PAD_INSTRUCTIONS_PER_AST>();
}

template <uint32_t KernelIndex>
static __device__ __forceinline__ void
cusr_tile_static_eval_template_site(
    uint32_t scratch_address,
    float value0, float value1, float value2, float value3,
    float value4, float value5, float value6, float value7,
    CusrTileStaticEvalTemplateValues<32u>* values)
{
    asm volatile(
        "{\n\t"
        ".reg .f32 %%anchor, %%marker;\n\t"
        ".reg .u32 %%anchor_bits, %%marker_bits;\n\t"
        "brkpt;\n\t"
        "mov.u32 %%marker_bits, %41;\n\t"
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_INPUT(32, 40)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_INPUT(33, 40)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_INPUT(34, 40)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_INPUT(35, 40)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_INPUT(36, 40)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_INPUT(37, 40)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_INPUT(38, 40)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_INPUT(39, 40)
        /* Direct eval keeps the shared target role by aliasing input 0. */
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_INPUT(32, 40)
        "add.u32 %%marker_bits, %%marker_bits, 7;\n\t"
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(0)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(1)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(2)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(3)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(4)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(5)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(6)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(7)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(8)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(9)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(10)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(11)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(12)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(13)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(14)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(15)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(16)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(17)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(18)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(19)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(20)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(21)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(22)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(23)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(24)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(25)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(26)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(27)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(28)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(29)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(30)
        CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT(31)
        "}\n"
        : "+f"(values->value_000), "+f"(values->value_001), "+f"(values->value_002), "+f"(values->value_003),
          "+f"(values->value_004), "+f"(values->value_005), "+f"(values->value_006), "+f"(values->value_007),
          "+f"(values->value_008), "+f"(values->value_009), "+f"(values->value_010), "+f"(values->value_011),
          "+f"(values->value_012), "+f"(values->value_013), "+f"(values->value_014), "+f"(values->value_015),
          "+f"(values->value_016), "+f"(values->value_017), "+f"(values->value_018), "+f"(values->value_019),
          "+f"(values->value_020), "+f"(values->value_021), "+f"(values->value_022), "+f"(values->value_023),
          "+f"(values->value_024), "+f"(values->value_025), "+f"(values->value_026), "+f"(values->value_027),
          "+f"(values->value_028), "+f"(values->value_029), "+f"(values->value_030), "+f"(values->value_031)
        : "f"(value0), "f"(value1), "f"(value2), "f"(value3), "f"(value4), "f"(value5), "f"(value6), "f"(value7),
          "r"(scratch_address),
          "n"(CUSR_TILE_STATIC_EVAL_TEMPLATE_FIRST_MARKER_BITS + KernelIndex * CUSR_TILE_STATIC_EVAL_TEMPLATE_MARKER_STRIDE)
        : "memory");
    cusr_tile_static_eval_template_emit_brkpt_pad<32u * CUSR_TILE_STATIC_EVAL_TEMPLATE_PAD_INSTRUCTIONS_PER_AST>();
}

#undef CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_OUTPUT
#undef CUSR_TILE_STATIC_EVAL_TEMPLATE_ANCHOR_INPUT

static __device__ __forceinline__ void
cusr_tile_static_eval_template_store_values(
    const CusrTileStaticEvalTemplateValues<8u>* values,
    size_t row,
    size_t num_rows,
    uint32_t setting,
    uint32_t num_settings,
    uint32_t num_asts,
    float* output_values)
{
    if (num_asts > 0u) output_values[((size_t)0u * num_settings + setting) * num_rows + row] = values->value_000;
    if (num_asts > 1u) output_values[((size_t)1u * num_settings + setting) * num_rows + row] = values->value_001;
    if (num_asts > 2u) output_values[((size_t)2u * num_settings + setting) * num_rows + row] = values->value_002;
    if (num_asts > 3u) output_values[((size_t)3u * num_settings + setting) * num_rows + row] = values->value_003;
    if (num_asts > 4u) output_values[((size_t)4u * num_settings + setting) * num_rows + row] = values->value_004;
    if (num_asts > 5u) output_values[((size_t)5u * num_settings + setting) * num_rows + row] = values->value_005;
    if (num_asts > 6u) output_values[((size_t)6u * num_settings + setting) * num_rows + row] = values->value_006;
    if (num_asts > 7u) output_values[((size_t)7u * num_settings + setting) * num_rows + row] = values->value_007;
}

static __device__ __forceinline__ void
cusr_tile_static_eval_template_store_values(
    const CusrTileStaticEvalTemplateValues<16u>* values,
    size_t row,
    size_t num_rows,
    uint32_t setting,
    uint32_t num_settings,
    uint32_t num_asts,
    float* output_values)
{
    if (num_asts > 0u) output_values[((size_t)0u * num_settings + setting) * num_rows + row] = values->value_000;
    if (num_asts > 1u) output_values[((size_t)1u * num_settings + setting) * num_rows + row] = values->value_001;
    if (num_asts > 2u) output_values[((size_t)2u * num_settings + setting) * num_rows + row] = values->value_002;
    if (num_asts > 3u) output_values[((size_t)3u * num_settings + setting) * num_rows + row] = values->value_003;
    if (num_asts > 4u) output_values[((size_t)4u * num_settings + setting) * num_rows + row] = values->value_004;
    if (num_asts > 5u) output_values[((size_t)5u * num_settings + setting) * num_rows + row] = values->value_005;
    if (num_asts > 6u) output_values[((size_t)6u * num_settings + setting) * num_rows + row] = values->value_006;
    if (num_asts > 7u) output_values[((size_t)7u * num_settings + setting) * num_rows + row] = values->value_007;
    if (num_asts > 8u) output_values[((size_t)8u * num_settings + setting) * num_rows + row] = values->value_008;
    if (num_asts > 9u) output_values[((size_t)9u * num_settings + setting) * num_rows + row] = values->value_009;
    if (num_asts > 10u) output_values[((size_t)10u * num_settings + setting) * num_rows + row] = values->value_010;
    if (num_asts > 11u) output_values[((size_t)11u * num_settings + setting) * num_rows + row] = values->value_011;
    if (num_asts > 12u) output_values[((size_t)12u * num_settings + setting) * num_rows + row] = values->value_012;
    if (num_asts > 13u) output_values[((size_t)13u * num_settings + setting) * num_rows + row] = values->value_013;
    if (num_asts > 14u) output_values[((size_t)14u * num_settings + setting) * num_rows + row] = values->value_014;
    if (num_asts > 15u) output_values[((size_t)15u * num_settings + setting) * num_rows + row] = values->value_015;
}

static __device__ __forceinline__ void
cusr_tile_static_eval_template_store_values(
    const CusrTileStaticEvalTemplateValues<32u>* values,
    size_t row,
    size_t num_rows,
    uint32_t setting,
    uint32_t num_settings,
    uint32_t num_asts,
    float* output_values)
{
    if (num_asts > 0u) output_values[((size_t)0u * num_settings + setting) * num_rows + row] = values->value_000;
    if (num_asts > 1u) output_values[((size_t)1u * num_settings + setting) * num_rows + row] = values->value_001;
    if (num_asts > 2u) output_values[((size_t)2u * num_settings + setting) * num_rows + row] = values->value_002;
    if (num_asts > 3u) output_values[((size_t)3u * num_settings + setting) * num_rows + row] = values->value_003;
    if (num_asts > 4u) output_values[((size_t)4u * num_settings + setting) * num_rows + row] = values->value_004;
    if (num_asts > 5u) output_values[((size_t)5u * num_settings + setting) * num_rows + row] = values->value_005;
    if (num_asts > 6u) output_values[((size_t)6u * num_settings + setting) * num_rows + row] = values->value_006;
    if (num_asts > 7u) output_values[((size_t)7u * num_settings + setting) * num_rows + row] = values->value_007;
    if (num_asts > 8u) output_values[((size_t)8u * num_settings + setting) * num_rows + row] = values->value_008;
    if (num_asts > 9u) output_values[((size_t)9u * num_settings + setting) * num_rows + row] = values->value_009;
    if (num_asts > 10u) output_values[((size_t)10u * num_settings + setting) * num_rows + row] = values->value_010;
    if (num_asts > 11u) output_values[((size_t)11u * num_settings + setting) * num_rows + row] = values->value_011;
    if (num_asts > 12u) output_values[((size_t)12u * num_settings + setting) * num_rows + row] = values->value_012;
    if (num_asts > 13u) output_values[((size_t)13u * num_settings + setting) * num_rows + row] = values->value_013;
    if (num_asts > 14u) output_values[((size_t)14u * num_settings + setting) * num_rows + row] = values->value_014;
    if (num_asts > 15u) output_values[((size_t)15u * num_settings + setting) * num_rows + row] = values->value_015;
    if (num_asts > 16u) output_values[((size_t)16u * num_settings + setting) * num_rows + row] = values->value_016;
    if (num_asts > 17u) output_values[((size_t)17u * num_settings + setting) * num_rows + row] = values->value_017;
    if (num_asts > 18u) output_values[((size_t)18u * num_settings + setting) * num_rows + row] = values->value_018;
    if (num_asts > 19u) output_values[((size_t)19u * num_settings + setting) * num_rows + row] = values->value_019;
    if (num_asts > 20u) output_values[((size_t)20u * num_settings + setting) * num_rows + row] = values->value_020;
    if (num_asts > 21u) output_values[((size_t)21u * num_settings + setting) * num_rows + row] = values->value_021;
    if (num_asts > 22u) output_values[((size_t)22u * num_settings + setting) * num_rows + row] = values->value_022;
    if (num_asts > 23u) output_values[((size_t)23u * num_settings + setting) * num_rows + row] = values->value_023;
    if (num_asts > 24u) output_values[((size_t)24u * num_settings + setting) * num_rows + row] = values->value_024;
    if (num_asts > 25u) output_values[((size_t)25u * num_settings + setting) * num_rows + row] = values->value_025;
    if (num_asts > 26u) output_values[((size_t)26u * num_settings + setting) * num_rows + row] = values->value_026;
    if (num_asts > 27u) output_values[((size_t)27u * num_settings + setting) * num_rows + row] = values->value_027;
    if (num_asts > 28u) output_values[((size_t)28u * num_settings + setting) * num_rows + row] = values->value_028;
    if (num_asts > 29u) output_values[((size_t)29u * num_settings + setting) * num_rows + row] = values->value_029;
    if (num_asts > 30u) output_values[((size_t)30u * num_settings + setting) * num_rows + row] = values->value_030;
    if (num_asts > 31u) output_values[((size_t)31u * num_settings + setting) * num_rows + row] = values->value_031;
}

template <uint32_t KernelIndex, uint32_t AstCapacity, uint32_t TileRows, uint32_t ThreadsPerCta>
static __device__ __forceinline__ void
cusr_tile_static_eval_template_f32(
    const float* __restrict__ X,
    size_t num_rows,
    uint32_t num_columns,
    size_t leading_dim,
    const uint8_t* __restrict__ leaf_masks,
    const uint32_t* __restrict__ leaf_words,
    size_t leaf_words_stride,
    uint32_t num_settings,
    uint32_t num_asts,
    float* __restrict__ output_values,
    float* x_tile,
    uint32_t* site_scratch)
{
    static_assert(AstCapacity == 8u || AstCapacity == 16u || AstCapacity == 32u, "unsupported AST capacity");
    const uint32_t shared_stride = cusr_tile_static_eval_template_shared_stride(num_columns);
    const uint32_t scratch_address = (uint32_t)__cvta_generic_to_shared((void*)site_scratch);
    const uint32_t row_limit = cusr_tile_static_eval_template_load_tile_f32<TileRows, ThreadsPerCta>(X, num_rows, num_columns, leading_dim, x_tile, shared_stride);
    const size_t row_base = (size_t)blockIdx.x * TileRows;

    __syncthreads();

    for (uint32_t setting_base = 0u; setting_base < num_settings; setting_base += ThreadsPerCta) {
        const uint32_t setting = setting_base + threadIdx.x;

        if (setting >= num_settings) {
            continue;
        }

        const uint8_t mask = leaf_masks[setting];
        const uint32_t* const words = leaf_words + (size_t)setting * leaf_words_stride;
        const uint32_t word0 = words[0u];
        const uint32_t word1 = words[1u];
        const uint32_t word2 = words[2u];
        const uint32_t word3 = words[3u];
        const uint32_t word4 = words[4u];
        const uint32_t word5 = words[5u];
        const uint32_t word6 = words[6u];
        const uint32_t word7 = words[7u];
        const uint32_t is_column0 = cusr_tile_static_eval_template_leaf_is_column(mask, 0u);
        const uint32_t is_column1 = cusr_tile_static_eval_template_leaf_is_column(mask, 1u);
        const uint32_t is_column2 = cusr_tile_static_eval_template_leaf_is_column(mask, 2u);
        const uint32_t is_column3 = cusr_tile_static_eval_template_leaf_is_column(mask, 3u);
        const uint32_t is_column4 = cusr_tile_static_eval_template_leaf_is_column(mask, 4u);
        const uint32_t is_column5 = cusr_tile_static_eval_template_leaf_is_column(mask, 5u);
        const uint32_t is_column6 = cusr_tile_static_eval_template_leaf_is_column(mask, 6u);
        const uint32_t is_column7 = cusr_tile_static_eval_template_leaf_is_column(mask, 7u);
        const uint32_t x_tile_address = (uint32_t)__cvta_generic_to_shared((const void*)x_tile);
        const uint32_t shared_stride_bytes = shared_stride * sizeof(float);
        float value0 = __uint_as_float(word0);
        float value1 = __uint_as_float(word1);
        float value2 = __uint_as_float(word2);
        float value3 = __uint_as_float(word3);
        float value4 = __uint_as_float(word4);
        float value5 = __uint_as_float(word5);
        float value6 = __uint_as_float(word6);
        float value7 = __uint_as_float(word7);
        uint32_t address0 = x_tile_address + word0 * sizeof(float);
        uint32_t address1 = x_tile_address + word1 * sizeof(float);
        uint32_t address2 = x_tile_address + word2 * sizeof(float);
        uint32_t address3 = x_tile_address + word3 * sizeof(float);
        uint32_t address4 = x_tile_address + word4 * sizeof(float);
        uint32_t address5 = x_tile_address + word5 * sizeof(float);
        uint32_t address6 = x_tile_address + word6 * sizeof(float);
        uint32_t address7 = x_tile_address + word7 * sizeof(float);
        CusrTileStaticEvalTemplateValues<AstCapacity> values = {};

        #pragma unroll 1
        for (uint32_t eval_row = 0u; eval_row < row_limit; ++eval_row) {
            cusr_tile_static_eval_template_predicated_shared_load8_f32(
                &value0, &value1, &value2, &value3, &value4, &value5, &value6, &value7,
                address0, address1, address2, address3, address4, address5, address6, address7,
                is_column0, is_column1, is_column2, is_column3, is_column4, is_column5, is_column6, is_column7);

            cusr_tile_static_eval_template_site<KernelIndex>(scratch_address, value0, value1, value2, value3, value4, value5, value6, value7, &values);
            cusr_tile_static_eval_template_store_values(&values, row_base + eval_row, num_rows, setting, num_settings, num_asts, output_values);

            address0 += shared_stride_bytes;
            address1 += shared_stride_bytes;
            address2 += shared_stride_bytes;
            address3 += shared_stride_bytes;
            address4 += shared_stride_bytes;
            address5 += shared_stride_bytes;
            address6 += shared_stride_bytes;
            address7 += shared_stride_bytes;
        }
    }
}

template <uint32_t KernelIndex, uint32_t AstCapacity, uint32_t TileRows, uint32_t ThreadsPerCta>
__global__ void
cusr_tile_static_eval_template_kernel_f32(
    const float* __restrict__ X,
    size_t num_rows,
    uint32_t num_columns,
    size_t leading_dim,
    const uint8_t* __restrict__ leaf_masks,
    const uint32_t* __restrict__ leaf_words,
    size_t leaf_words_stride,
    uint32_t num_settings,
    uint32_t num_asts,
    float* __restrict__ output_values)
{
    extern __shared__ float x_tile[];
    __shared__ uint32_t cusr_sass_site_scratch[1];

    cusr_tile_static_eval_template_f32<KernelIndex, AstCapacity, TileRows, ThreadsPerCta>(
        X,
        num_rows,
        num_columns,
        leading_dim,
        leaf_masks,
        leaf_words,
        leaf_words_stride,
        num_settings,
        num_asts,
        output_values,
        x_tile,
        cusr_sass_site_scratch);
}

#define CUSR_TILE_STATIC_EVAL_TEMPLATE_INSTANTIATE(Name, KernelIndex, AstCapacity, TileRows, ThreadsPerCta) \
    extern "C" __global__ __launch_bounds__(ThreadsPerCta) void Name( \
        const float* __restrict__ X, \
        size_t num_rows, \
        uint32_t num_columns, \
        size_t leading_dim, \
        const uint8_t* __restrict__ leaf_masks, \
        const uint32_t* __restrict__ leaf_words, \
        size_t leaf_words_stride, \
        uint32_t num_settings, \
        uint32_t num_asts, \
        float* __restrict__ output_values) \
    { \
        extern __shared__ float x_tile[]; \
        __shared__ uint32_t cusr_sass_site_scratch[1]; \
        cusr_tile_static_eval_template_f32<KernelIndex, AstCapacity, TileRows, ThreadsPerCta>( \
            X, num_rows, num_columns, leading_dim, leaf_masks, leaf_words, leaf_words_stride, \
            num_settings, num_asts, output_values, x_tile, cusr_sass_site_scratch); \
    }

#endif /* CUSR_TILE_STATIC_EVAL_TEMPLATE_CUH_INCLUDED */
