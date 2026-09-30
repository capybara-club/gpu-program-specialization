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
#ifndef CUSR_TILE_STATIC_MSE_TEMPLATE_CUH_INCLUDED
#define CUSR_TILE_STATIC_MSE_TEMPLATE_CUH_INCLUDED

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

#define CUSR_TILE_STATIC_MSE_TEMPLATE_FIRST_MARKER_BITS 0x7fc0ffeeu
#define CUSR_TILE_STATIC_MSE_TEMPLATE_MARKER_STRIDE 128u
#define CUSR_TILE_STATIC_MSE_TEMPLATE_TARGET_MARKER_OFFSET 8u
#define CUSR_TILE_STATIC_MSE_TEMPLATE_SSE_MARKER_OFFSET 16u
#define CUSR_TILE_STATIC_MSE_TEMPLATE_PAD_INSTRUCTIONS_PER_AST 64u

template <uint32_t NumBrkpts>
static __device__ __forceinline__ void
cusr_tile_static_mse_template_emit_brkpt_pad(void)
{
    #pragma unroll
    for (uint32_t i = 0u; i < NumBrkpts; ++i) {
        asm volatile("brkpt;" ::: "memory");
    }
}

template <uint32_t AstCapacity>
struct CusrTileStaticMseTemplateSse;

/* Named fields keep every SSE accumulator in a scalar register. */
template <>
struct CusrTileStaticMseTemplateSse<8u> {
    float sse_000;
    float sse_001;
    float sse_002;
    float sse_003;
    float sse_004;
    float sse_005;
    float sse_006;
    float sse_007;
};

template <>
struct CusrTileStaticMseTemplateSse<16u> {
    float sse_000;
    float sse_001;
    float sse_002;
    float sse_003;
    float sse_004;
    float sse_005;
    float sse_006;
    float sse_007;
    float sse_008;
    float sse_009;
    float sse_010;
    float sse_011;
    float sse_012;
    float sse_013;
    float sse_014;
    float sse_015;
};

template <>
struct CusrTileStaticMseTemplateSse<32u> {
    float sse_000;
    float sse_001;
    float sse_002;
    float sse_003;
    float sse_004;
    float sse_005;
    float sse_006;
    float sse_007;
    float sse_008;
    float sse_009;
    float sse_010;
    float sse_011;
    float sse_012;
    float sse_013;
    float sse_014;
    float sse_015;
    float sse_016;
    float sse_017;
    float sse_018;
    float sse_019;
    float sse_020;
    float sse_021;
    float sse_022;
    float sse_023;
    float sse_024;
    float sse_025;
    float sse_026;
    float sse_027;
    float sse_028;
    float sse_029;
    float sse_030;
    float sse_031;
};

static __device__ __forceinline__ uint32_t
cusr_tile_static_mse_template_leaf_is_column(uint8_t mask, uint32_t leaf)
{
    return (((uint32_t)mask) >> leaf) & 1u;
}

static __device__ __forceinline__ uint32_t
cusr_tile_static_mse_template_shared_stride(uint32_t num_columns)
{
    return num_columns | 1u;
}

template <uint32_t TileRows, uint32_t ThreadsPerCta>
static __device__ __forceinline__ uint32_t
cusr_tile_static_mse_template_load_tile_f32(
    const float* __restrict__ X,
    const float* __restrict__ target,
    size_t num_rows,
    uint32_t num_columns,
    size_t leading_dim,
    float* x_tile,
    float* target_tile,
    uint32_t shared_stride)
{
    const size_t row_base = (size_t)blockIdx.x * TileRows;

    #pragma unroll
    for (uint32_t tile_row = threadIdx.x; tile_row < TileRows; tile_row += ThreadsPerCta) {
        const size_t row = row_base + tile_row;
        const bool row_valid = row < num_rows;
        target_tile[tile_row] = row_valid ? target[row] : 0.0f;

        #pragma unroll 1
        for (uint32_t column = 0u; column < num_columns; ++column) {
            x_tile[(size_t)tile_row * shared_stride + column] = row_valid ? X[(size_t)column * leading_dim + row] : 0.0f;
        }
    }

    return row_base + TileRows <= num_rows ? TileRows : (row_base < num_rows ? (uint32_t)(num_rows - row_base) : 0u);
}

static __device__ __forceinline__ void
cusr_tile_static_mse_template_predicated_shared_load8_f32(
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

/*
 * Keep every anchor in one asm island so all inputs and SSE accumulators remain
 * live together. ptxas folds marker_bits into immediate FADDs; inspection
 * rejects the cubin if those marker instructions do not have the expected form.
 */
#define CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_INPUT(input_operand, scratch_operand) \
    "mov.b32 %%marker, %%marker_bits;\n\t" \
    "add.rn.ftz.f32 %%anchor, %" #input_operand ", %%marker;\n\t" \
    "mov.b32 %%anchor_bits, %%anchor;\n\t" \
    "st.volatile.shared.u32 [%" #scratch_operand "], %%anchor_bits;\n\t" \
    "brkpt;\n\t" \
    "add.u32 %%marker_bits, %%marker_bits, 1;\n\t"

#define CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(output_operand) \
    "mov.b32 %%marker, %%marker_bits;\n\t" \
    "add.rn.ftz.f32 %" #output_operand ", %" #output_operand ", %%marker;\n\t" \
    "brkpt;\n\t" \
    "add.u32 %%marker_bits, %%marker_bits, 1;\n\t"

template <uint32_t KernelIndex>
static __device__ __forceinline__ void
cusr_tile_static_mse_template_site(
    uint32_t scratch_address,
    float value0, float value1, float value2, float value3,
    float value4, float value5, float value6, float value7,
    float target,
    CusrTileStaticMseTemplateSse<8u>* sse)
{
    asm volatile(
        "{\n\t"
        ".reg .f32 %%anchor, %%marker;\n\t"
        ".reg .u32 %%anchor_bits, %%marker_bits;\n\t"
        "brkpt;\n\t"
        "mov.u32 %%marker_bits, %18;\n\t"
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_INPUT(8, 17)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_INPUT(9, 17)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_INPUT(10, 17)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_INPUT(11, 17)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_INPUT(12, 17)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_INPUT(13, 17)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_INPUT(14, 17)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_INPUT(15, 17)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_INPUT(16, 17)
        "add.u32 %%marker_bits, %%marker_bits, 7;\n\t"
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(0)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(1)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(2)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(3)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(4)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(5)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(6)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(7)
        "}\n"
        : "+f"(sse->sse_000), "+f"(sse->sse_001), "+f"(sse->sse_002), "+f"(sse->sse_003),
          "+f"(sse->sse_004), "+f"(sse->sse_005), "+f"(sse->sse_006), "+f"(sse->sse_007)
        : "f"(value0), "f"(value1), "f"(value2), "f"(value3), "f"(value4), "f"(value5), "f"(value6), "f"(value7), "f"(target),
          "r"(scratch_address),
          "n"(CUSR_TILE_STATIC_MSE_TEMPLATE_FIRST_MARKER_BITS + KernelIndex * CUSR_TILE_STATIC_MSE_TEMPLATE_MARKER_STRIDE)
        : "memory");
    cusr_tile_static_mse_template_emit_brkpt_pad<8u * CUSR_TILE_STATIC_MSE_TEMPLATE_PAD_INSTRUCTIONS_PER_AST>();
}

template <uint32_t KernelIndex>
static __device__ __forceinline__ void
cusr_tile_static_mse_template_site(
    uint32_t scratch_address,
    float value0, float value1, float value2, float value3,
    float value4, float value5, float value6, float value7,
    float target,
    CusrTileStaticMseTemplateSse<16u>* sse)
{
    asm volatile(
        "{\n\t"
        ".reg .f32 %%anchor, %%marker;\n\t"
        ".reg .u32 %%anchor_bits, %%marker_bits;\n\t"
        "brkpt;\n\t"
        "mov.u32 %%marker_bits, %26;\n\t"
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_INPUT(16, 25)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_INPUT(17, 25)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_INPUT(18, 25)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_INPUT(19, 25)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_INPUT(20, 25)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_INPUT(21, 25)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_INPUT(22, 25)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_INPUT(23, 25)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_INPUT(24, 25)
        "add.u32 %%marker_bits, %%marker_bits, 7;\n\t"
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(0)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(1)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(2)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(3)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(4)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(5)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(6)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(7)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(8)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(9)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(10)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(11)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(12)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(13)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(14)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(15)
        "}\n"
        : "+f"(sse->sse_000), "+f"(sse->sse_001), "+f"(sse->sse_002), "+f"(sse->sse_003),
          "+f"(sse->sse_004), "+f"(sse->sse_005), "+f"(sse->sse_006), "+f"(sse->sse_007),
          "+f"(sse->sse_008), "+f"(sse->sse_009), "+f"(sse->sse_010), "+f"(sse->sse_011),
          "+f"(sse->sse_012), "+f"(sse->sse_013), "+f"(sse->sse_014), "+f"(sse->sse_015)
        : "f"(value0), "f"(value1), "f"(value2), "f"(value3), "f"(value4), "f"(value5), "f"(value6), "f"(value7), "f"(target),
          "r"(scratch_address),
          "n"(CUSR_TILE_STATIC_MSE_TEMPLATE_FIRST_MARKER_BITS + KernelIndex * CUSR_TILE_STATIC_MSE_TEMPLATE_MARKER_STRIDE)
        : "memory");
    cusr_tile_static_mse_template_emit_brkpt_pad<16u * CUSR_TILE_STATIC_MSE_TEMPLATE_PAD_INSTRUCTIONS_PER_AST>();
}

template <uint32_t KernelIndex>
static __device__ __forceinline__ void
cusr_tile_static_mse_template_site(
    uint32_t scratch_address,
    float value0, float value1, float value2, float value3,
    float value4, float value5, float value6, float value7,
    float target,
    CusrTileStaticMseTemplateSse<32u>* sse)
{
    asm volatile(
        "{\n\t"
        ".reg .f32 %%anchor, %%marker;\n\t"
        ".reg .u32 %%anchor_bits, %%marker_bits;\n\t"
        "brkpt;\n\t"
        "mov.u32 %%marker_bits, %42;\n\t"
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_INPUT(32, 41)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_INPUT(33, 41)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_INPUT(34, 41)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_INPUT(35, 41)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_INPUT(36, 41)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_INPUT(37, 41)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_INPUT(38, 41)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_INPUT(39, 41)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_INPUT(40, 41)
        "add.u32 %%marker_bits, %%marker_bits, 7;\n\t"
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(0)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(1)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(2)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(3)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(4)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(5)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(6)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(7)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(8)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(9)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(10)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(11)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(12)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(13)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(14)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(15)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(16)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(17)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(18)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(19)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(20)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(21)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(22)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(23)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(24)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(25)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(26)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(27)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(28)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(29)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(30)
        CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE(31)
        "}\n"
        : "+f"(sse->sse_000), "+f"(sse->sse_001), "+f"(sse->sse_002), "+f"(sse->sse_003),
          "+f"(sse->sse_004), "+f"(sse->sse_005), "+f"(sse->sse_006), "+f"(sse->sse_007),
          "+f"(sse->sse_008), "+f"(sse->sse_009), "+f"(sse->sse_010), "+f"(sse->sse_011),
          "+f"(sse->sse_012), "+f"(sse->sse_013), "+f"(sse->sse_014), "+f"(sse->sse_015),
          "+f"(sse->sse_016), "+f"(sse->sse_017), "+f"(sse->sse_018), "+f"(sse->sse_019),
          "+f"(sse->sse_020), "+f"(sse->sse_021), "+f"(sse->sse_022), "+f"(sse->sse_023),
          "+f"(sse->sse_024), "+f"(sse->sse_025), "+f"(sse->sse_026), "+f"(sse->sse_027),
          "+f"(sse->sse_028), "+f"(sse->sse_029), "+f"(sse->sse_030), "+f"(sse->sse_031)
        : "f"(value0), "f"(value1), "f"(value2), "f"(value3), "f"(value4), "f"(value5), "f"(value6), "f"(value7), "f"(target),
          "r"(scratch_address),
          "n"(CUSR_TILE_STATIC_MSE_TEMPLATE_FIRST_MARKER_BITS + KernelIndex * CUSR_TILE_STATIC_MSE_TEMPLATE_MARKER_STRIDE)
        : "memory");
    cusr_tile_static_mse_template_emit_brkpt_pad<32u * CUSR_TILE_STATIC_MSE_TEMPLATE_PAD_INSTRUCTIONS_PER_AST>();
}

#undef CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_SSE
#undef CUSR_TILE_STATIC_MSE_TEMPLATE_ANCHOR_INPUT

static __device__ __forceinline__ void
cusr_tile_static_mse_template_atomic_add(const CusrTileStaticMseTemplateSse<8u>* sse, uint32_t setting, uint32_t num_settings, uint32_t num_asts, float* output_sse)
{
    if (num_asts > 0u) atomicAdd(output_sse + (size_t)0u * num_settings + setting, sse->sse_000);
    if (num_asts > 1u) atomicAdd(output_sse + (size_t)1u * num_settings + setting, sse->sse_001);
    if (num_asts > 2u) atomicAdd(output_sse + (size_t)2u * num_settings + setting, sse->sse_002);
    if (num_asts > 3u) atomicAdd(output_sse + (size_t)3u * num_settings + setting, sse->sse_003);
    if (num_asts > 4u) atomicAdd(output_sse + (size_t)4u * num_settings + setting, sse->sse_004);
    if (num_asts > 5u) atomicAdd(output_sse + (size_t)5u * num_settings + setting, sse->sse_005);
    if (num_asts > 6u) atomicAdd(output_sse + (size_t)6u * num_settings + setting, sse->sse_006);
    if (num_asts > 7u) atomicAdd(output_sse + (size_t)7u * num_settings + setting, sse->sse_007);
}

static __device__ __forceinline__ void
cusr_tile_static_mse_template_atomic_add(const CusrTileStaticMseTemplateSse<16u>* sse, uint32_t setting, uint32_t num_settings, uint32_t num_asts, float* output_sse)
{
    if (num_asts > 0u) atomicAdd(output_sse + (size_t)0u * num_settings + setting, sse->sse_000);
    if (num_asts > 1u) atomicAdd(output_sse + (size_t)1u * num_settings + setting, sse->sse_001);
    if (num_asts > 2u) atomicAdd(output_sse + (size_t)2u * num_settings + setting, sse->sse_002);
    if (num_asts > 3u) atomicAdd(output_sse + (size_t)3u * num_settings + setting, sse->sse_003);
    if (num_asts > 4u) atomicAdd(output_sse + (size_t)4u * num_settings + setting, sse->sse_004);
    if (num_asts > 5u) atomicAdd(output_sse + (size_t)5u * num_settings + setting, sse->sse_005);
    if (num_asts > 6u) atomicAdd(output_sse + (size_t)6u * num_settings + setting, sse->sse_006);
    if (num_asts > 7u) atomicAdd(output_sse + (size_t)7u * num_settings + setting, sse->sse_007);
    if (num_asts > 8u) atomicAdd(output_sse + (size_t)8u * num_settings + setting, sse->sse_008);
    if (num_asts > 9u) atomicAdd(output_sse + (size_t)9u * num_settings + setting, sse->sse_009);
    if (num_asts > 10u) atomicAdd(output_sse + (size_t)10u * num_settings + setting, sse->sse_010);
    if (num_asts > 11u) atomicAdd(output_sse + (size_t)11u * num_settings + setting, sse->sse_011);
    if (num_asts > 12u) atomicAdd(output_sse + (size_t)12u * num_settings + setting, sse->sse_012);
    if (num_asts > 13u) atomicAdd(output_sse + (size_t)13u * num_settings + setting, sse->sse_013);
    if (num_asts > 14u) atomicAdd(output_sse + (size_t)14u * num_settings + setting, sse->sse_014);
    if (num_asts > 15u) atomicAdd(output_sse + (size_t)15u * num_settings + setting, sse->sse_015);
}

static __device__ __forceinline__ void
cusr_tile_static_mse_template_atomic_add(const CusrTileStaticMseTemplateSse<32u>* sse, uint32_t setting, uint32_t num_settings, uint32_t num_asts, float* output_sse)
{
    if (num_asts > 0u) atomicAdd(output_sse + (size_t)0u * num_settings + setting, sse->sse_000);
    if (num_asts > 1u) atomicAdd(output_sse + (size_t)1u * num_settings + setting, sse->sse_001);
    if (num_asts > 2u) atomicAdd(output_sse + (size_t)2u * num_settings + setting, sse->sse_002);
    if (num_asts > 3u) atomicAdd(output_sse + (size_t)3u * num_settings + setting, sse->sse_003);
    if (num_asts > 4u) atomicAdd(output_sse + (size_t)4u * num_settings + setting, sse->sse_004);
    if (num_asts > 5u) atomicAdd(output_sse + (size_t)5u * num_settings + setting, sse->sse_005);
    if (num_asts > 6u) atomicAdd(output_sse + (size_t)6u * num_settings + setting, sse->sse_006);
    if (num_asts > 7u) atomicAdd(output_sse + (size_t)7u * num_settings + setting, sse->sse_007);
    if (num_asts > 8u) atomicAdd(output_sse + (size_t)8u * num_settings + setting, sse->sse_008);
    if (num_asts > 9u) atomicAdd(output_sse + (size_t)9u * num_settings + setting, sse->sse_009);
    if (num_asts > 10u) atomicAdd(output_sse + (size_t)10u * num_settings + setting, sse->sse_010);
    if (num_asts > 11u) atomicAdd(output_sse + (size_t)11u * num_settings + setting, sse->sse_011);
    if (num_asts > 12u) atomicAdd(output_sse + (size_t)12u * num_settings + setting, sse->sse_012);
    if (num_asts > 13u) atomicAdd(output_sse + (size_t)13u * num_settings + setting, sse->sse_013);
    if (num_asts > 14u) atomicAdd(output_sse + (size_t)14u * num_settings + setting, sse->sse_014);
    if (num_asts > 15u) atomicAdd(output_sse + (size_t)15u * num_settings + setting, sse->sse_015);
    if (num_asts > 16u) atomicAdd(output_sse + (size_t)16u * num_settings + setting, sse->sse_016);
    if (num_asts > 17u) atomicAdd(output_sse + (size_t)17u * num_settings + setting, sse->sse_017);
    if (num_asts > 18u) atomicAdd(output_sse + (size_t)18u * num_settings + setting, sse->sse_018);
    if (num_asts > 19u) atomicAdd(output_sse + (size_t)19u * num_settings + setting, sse->sse_019);
    if (num_asts > 20u) atomicAdd(output_sse + (size_t)20u * num_settings + setting, sse->sse_020);
    if (num_asts > 21u) atomicAdd(output_sse + (size_t)21u * num_settings + setting, sse->sse_021);
    if (num_asts > 22u) atomicAdd(output_sse + (size_t)22u * num_settings + setting, sse->sse_022);
    if (num_asts > 23u) atomicAdd(output_sse + (size_t)23u * num_settings + setting, sse->sse_023);
    if (num_asts > 24u) atomicAdd(output_sse + (size_t)24u * num_settings + setting, sse->sse_024);
    if (num_asts > 25u) atomicAdd(output_sse + (size_t)25u * num_settings + setting, sse->sse_025);
    if (num_asts > 26u) atomicAdd(output_sse + (size_t)26u * num_settings + setting, sse->sse_026);
    if (num_asts > 27u) atomicAdd(output_sse + (size_t)27u * num_settings + setting, sse->sse_027);
    if (num_asts > 28u) atomicAdd(output_sse + (size_t)28u * num_settings + setting, sse->sse_028);
    if (num_asts > 29u) atomicAdd(output_sse + (size_t)29u * num_settings + setting, sse->sse_029);
    if (num_asts > 30u) atomicAdd(output_sse + (size_t)30u * num_settings + setting, sse->sse_030);
    if (num_asts > 31u) atomicAdd(output_sse + (size_t)31u * num_settings + setting, sse->sse_031);
}

template <uint32_t KernelIndex, uint32_t AstCapacity, uint32_t TileRows, uint32_t ThreadsPerCta>
static __device__ __forceinline__ void
cusr_tile_static_mse_template_f32(
    const float* __restrict__ X,
    const float* __restrict__ target,
    size_t num_rows,
    uint32_t num_columns,
    size_t leading_dim,
    const uint8_t* __restrict__ leaf_masks,
    const uint32_t* __restrict__ leaf_words,
    size_t leaf_words_stride,
    uint32_t num_settings,
    uint32_t num_asts,
    float* __restrict__ output_sse,
    float* shared,
    uint32_t* site_scratch)
{
    static_assert(AstCapacity == 8u || AstCapacity == 16u || AstCapacity == 32u, "unsupported AST capacity");
    const uint32_t shared_stride = cusr_tile_static_mse_template_shared_stride(num_columns);
    float* const x_tile = shared;
    float* const target_tile = shared + (size_t)TileRows * shared_stride;
    const uint32_t scratch_address = (uint32_t)__cvta_generic_to_shared((void*)site_scratch);
    const uint32_t row_limit = cusr_tile_static_mse_template_load_tile_f32<TileRows, ThreadsPerCta>(X, target, num_rows, num_columns, leading_dim, x_tile, target_tile, shared_stride);

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
        const uint32_t is_column0 = cusr_tile_static_mse_template_leaf_is_column(mask, 0u);
        const uint32_t is_column1 = cusr_tile_static_mse_template_leaf_is_column(mask, 1u);
        const uint32_t is_column2 = cusr_tile_static_mse_template_leaf_is_column(mask, 2u);
        const uint32_t is_column3 = cusr_tile_static_mse_template_leaf_is_column(mask, 3u);
        const uint32_t is_column4 = cusr_tile_static_mse_template_leaf_is_column(mask, 4u);
        const uint32_t is_column5 = cusr_tile_static_mse_template_leaf_is_column(mask, 5u);
        const uint32_t is_column6 = cusr_tile_static_mse_template_leaf_is_column(mask, 6u);
        const uint32_t is_column7 = cusr_tile_static_mse_template_leaf_is_column(mask, 7u);
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
        CusrTileStaticMseTemplateSse<AstCapacity> sse = {};

        #pragma unroll 1
        for (uint32_t eval_row = 0u; eval_row < row_limit; ++eval_row) {
            cusr_tile_static_mse_template_predicated_shared_load8_f32(
                &value0, &value1, &value2, &value3, &value4, &value5, &value6, &value7,
                address0, address1, address2, address3, address4, address5, address6, address7,
                is_column0, is_column1, is_column2, is_column3, is_column4, is_column5, is_column6, is_column7);

            cusr_tile_static_mse_template_site<KernelIndex>(scratch_address, value0, value1, value2, value3, value4, value5, value6, value7, target_tile[eval_row], &sse);

            address0 += shared_stride_bytes;
            address1 += shared_stride_bytes;
            address2 += shared_stride_bytes;
            address3 += shared_stride_bytes;
            address4 += shared_stride_bytes;
            address5 += shared_stride_bytes;
            address6 += shared_stride_bytes;
            address7 += shared_stride_bytes;
        }

        cusr_tile_static_mse_template_atomic_add(&sse, setting, num_settings, num_asts, output_sse);
    }
}

template <uint32_t KernelIndex, uint32_t AstCapacity, uint32_t TileRows, uint32_t ThreadsPerCta>
__global__ void
cusr_tile_static_mse_template_kernel_f32(
    const float* __restrict__ X,
    const float* __restrict__ target,
    size_t num_rows,
    uint32_t num_columns,
    size_t leading_dim,
    const uint8_t* __restrict__ leaf_masks,
    const uint32_t* __restrict__ leaf_words,
    size_t leaf_words_stride,
    uint32_t num_settings,
    uint32_t num_asts,
    float* __restrict__ output_sse)
{
    extern __shared__ float shared[];
    __shared__ uint32_t cusr_sass_site_scratch[1];

    cusr_tile_static_mse_template_f32<KernelIndex, AstCapacity, TileRows, ThreadsPerCta>(
        X,
        target,
        num_rows,
        num_columns,
        leading_dim,
        leaf_masks,
        leaf_words,
        leaf_words_stride,
        num_settings,
        num_asts,
        output_sse,
        shared,
        cusr_sass_site_scratch);
}

#define CUSR_TILE_STATIC_MSE_TEMPLATE_INSTANTIATE(Name, KernelIndex, AstCapacity, TileRows, ThreadsPerCta) \
    extern "C" __global__ __launch_bounds__(ThreadsPerCta) void Name( \
        const float* __restrict__ X, \
        const float* __restrict__ target, \
        size_t num_rows, \
        uint32_t num_columns, \
        size_t leading_dim, \
        const uint8_t* __restrict__ leaf_masks, \
        const uint32_t* __restrict__ leaf_words, \
        size_t leaf_words_stride, \
        uint32_t num_settings, \
        uint32_t num_asts, \
        float* __restrict__ output_sse) \
    { \
        extern __shared__ float shared[]; \
        __shared__ uint32_t cusr_sass_site_scratch[1]; \
        cusr_tile_static_mse_template_f32<KernelIndex, AstCapacity, TileRows, ThreadsPerCta>( \
            X, target, num_rows, num_columns, leading_dim, leaf_masks, leaf_words, leaf_words_stride, \
            num_settings, num_asts, output_sse, shared, cusr_sass_site_scratch); \
    }

#endif /* CUSR_TILE_STATIC_MSE_TEMPLATE_CUH_INCLUDED */
