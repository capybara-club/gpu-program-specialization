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
#ifndef CUSR_AST_SETTINGS_EXPERIMENT_KERNELS_CUH_INCLUDED
#define CUSR_AST_SETTINGS_EXPERIMENT_KERNELS_CUH_INCLUDED

#include <cuda_runtime.h>

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#define CUSR_AST_SETTINGS_TILE_ROWS 128u
#define CUSR_AST_SETTINGS_THREADS 128u

enum CusrAstSettingsExpression {
    CUSR_AST_SETTINGS_SIMPLE_ALU = 0,
    CUSR_AST_SETTINGS_COMPLEX_ALU = 1,
    CUSR_AST_SETTINGS_LIGHT_MUFU = 2,
    CUSR_AST_SETTINGS_HEAVY_MUFU = 3
};

struct CusrAstSettingsLeafValues {
    float value0;
    float value1;
    float value2;
    float value3;
    float value4;
    float value5;
    float value6;
    float value7;
};

struct CusrAstSettingsSettingState {
    uint8_t mask;
    uint32_t word0;
    uint32_t word1;
    uint32_t word2;
    uint32_t word3;
    uint32_t word4;
    uint32_t word5;
    uint32_t word6;
    uint32_t word7;
    CusrAstSettingsLeafValues constants;
};

template <int Leaf>
static __device__ __forceinline__ float cusr_ast_settings_leaf(const CusrAstSettingsLeafValues& values)
{
    if constexpr (Leaf == 0) return values.value0;
    if constexpr (Leaf == 1) return values.value1;
    if constexpr (Leaf == 2) return values.value2;
    if constexpr (Leaf == 3) return values.value3;
    if constexpr (Leaf == 4) return values.value4;
    if constexpr (Leaf == 5) return values.value5;
    if constexpr (Leaf == 6) return values.value6;
    return values.value7;
}

template <int Op>
static __device__ __forceinline__ float cusr_ast_settings_binary(float lhs, float rhs)
{
    if constexpr (Op == 0) return lhs + rhs;
    if constexpr (Op == 1) return lhs * rhs;
    if constexpr (Op == 2) return fminf(lhs, rhs);
    return fmaxf(lhs, rhs);
}

template <int Op>
static __device__ __forceinline__ float cusr_ast_settings_mufu(float value)
{
    if constexpr (Op == 0) return __sinf(value);
    if constexpr (Op == 1) return __cosf(value);
    if constexpr (Op == 2) return exp2f(value * 0.125f);
    return rsqrtf(fabsf(value) + 0.25f);
}

static __host__ __device__ constexpr uint32_t cusr_ast_settings_hash32(uint32_t value)
{
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value;
}

template <int Ast, int Node>
static __host__ __device__ constexpr int cusr_ast_settings_binary_op()
{
    return (int)(cusr_ast_settings_hash32((uint32_t)Ast * 0xc2b2ae35u + (uint32_t)Node * 0x27d4eb2fu) & 3u);
}

template <int Ast, int Term>
static __host__ __device__ constexpr int cusr_ast_settings_mufu_op()
{
    return (int)(cusr_ast_settings_hash32((uint32_t)Ast * 0x9e3779b9u + (uint32_t)Term * 0x85ebca6bu) & 3u);
}

template <int Ast, int Term>
static __host__ __device__ constexpr int cusr_ast_settings_leaf_index()
{
    return (Ast + Term * 3) & 7;
}

template <int Ast, bool LightMufu, bool HeavyMufu, int Term>
static __device__ __forceinline__ float cusr_ast_settings_complex_leaf(const CusrAstSettingsLeafValues& values)
{
    constexpr int leaf = cusr_ast_settings_leaf_index<Ast, Term>();
    const float value = cusr_ast_settings_leaf<leaf>(values);

    if constexpr (HeavyMufu) return cusr_ast_settings_mufu<cusr_ast_settings_mufu_op<Ast, Term>()>(value);
    if constexpr (LightMufu && Term == 0) return cusr_ast_settings_mufu<cusr_ast_settings_mufu_op<Ast, Term>()>(value);
    return value;
}

template <int Ast, bool LightMufu, bool HeavyMufu>
static __device__ __forceinline__ float cusr_ast_settings_complex_expression(const CusrAstSettingsLeafValues& values)
{
    const float leaf0 = cusr_ast_settings_complex_leaf<Ast, LightMufu, HeavyMufu, 0>(values);
    const float leaf1 = cusr_ast_settings_complex_leaf<Ast, LightMufu, HeavyMufu, 1>(values);
    const float leaf2 = cusr_ast_settings_complex_leaf<Ast, LightMufu, HeavyMufu, 2>(values);
    const float leaf3 = cusr_ast_settings_complex_leaf<Ast, LightMufu, HeavyMufu, 3>(values);
    const float leaf4 = cusr_ast_settings_complex_leaf<Ast, LightMufu, HeavyMufu, 4>(values);
    const float leaf5 = cusr_ast_settings_complex_leaf<Ast, LightMufu, HeavyMufu, 5>(values);
    const float leaf6 = cusr_ast_settings_complex_leaf<Ast, LightMufu, HeavyMufu, 6>(values);
    const float leaf7 = cusr_ast_settings_complex_leaf<Ast, LightMufu, HeavyMufu, 7>(values);
    const float node0 = cusr_ast_settings_binary<cusr_ast_settings_binary_op<Ast, 0>()>(leaf0, leaf1);
    const float node1 = cusr_ast_settings_binary<cusr_ast_settings_binary_op<Ast, 1>()>(leaf2, leaf3);
    const float node2 = cusr_ast_settings_binary<cusr_ast_settings_binary_op<Ast, 2>()>(node0, node1);
    const float node3 = cusr_ast_settings_binary<cusr_ast_settings_binary_op<Ast, 3>()>(leaf4, leaf5);
    const float node4 = cusr_ast_settings_binary<cusr_ast_settings_binary_op<Ast, 4>()>(leaf6, leaf7);
    const float node5 = cusr_ast_settings_binary<cusr_ast_settings_binary_op<Ast, 5>()>(node3, node4);
    return cusr_ast_settings_binary<cusr_ast_settings_binary_op<Ast, 6>()>(node2, node5);
}

template <int Expression, int Ast>
static __device__ __forceinline__ float cusr_ast_settings_expression(const CusrAstSettingsLeafValues& values)
{
    if constexpr (Expression == CUSR_AST_SETTINGS_SIMPLE_ALU) {
        constexpr int lhs = Ast & 7;
        constexpr int rhs = (Ast * 3 + 1) & 7;
        constexpr int tail = (Ast * 5 + 2) & 7;
        const float scale = 1.0009765625f + (float)Ast * 0.000244140625f;
        return (cusr_ast_settings_leaf<lhs>(values) + cusr_ast_settings_leaf<rhs>(values)) * scale + cusr_ast_settings_leaf<tail>(values);
    } else if constexpr (Expression == CUSR_AST_SETTINGS_COMPLEX_ALU) {
        return cusr_ast_settings_complex_expression<Ast, false, false>(values);
    } else if constexpr (Expression == CUSR_AST_SETTINGS_LIGHT_MUFU) {
        return cusr_ast_settings_complex_expression<Ast, true, false>(values);
    } else {
        return cusr_ast_settings_complex_expression<Ast, false, true>(values);
    }
}

template <int Expression, int Ast, int NumAsts>
static __device__ __forceinline__ void cusr_ast_settings_accumulate(float* sse, const CusrAstSettingsLeafValues& values, float target)
{
    const float prediction = cusr_ast_settings_expression<Expression, Ast>(values);
    const float error = prediction - target;
    sse[Ast] = error * error + sse[Ast];

    if constexpr (Ast + 1 < NumAsts) cusr_ast_settings_accumulate<Expression, Ast + 1, NumAsts>(sse, values, target);
}

static __device__ __forceinline__ uint32_t cusr_ast_settings_shared_stride(uint32_t num_columns)
{
    return num_columns | 1u;
}

static __device__ __forceinline__ CusrAstSettingsSettingState cusr_ast_settings_load_setting(
    const uint8_t* __restrict__ leaf_masks,
    const uint32_t* __restrict__ leaf_words,
    size_t leaf_words_stride,
    uint32_t setting)
{
    CusrAstSettingsSettingState state;
    const uint32_t* const words = leaf_words + (size_t)setting * leaf_words_stride;

    state.mask = leaf_masks[setting];
    state.word0 = words[0];
    state.word1 = words[1];
    state.word2 = words[2];
    state.word3 = words[3];
    state.word4 = words[4];
    state.word5 = words[5];
    state.word6 = words[6];
    state.word7 = words[7];
    state.constants.value0 = __uint_as_float(state.word0);
    state.constants.value1 = __uint_as_float(state.word1);
    state.constants.value2 = __uint_as_float(state.word2);
    state.constants.value3 = __uint_as_float(state.word3);
    state.constants.value4 = __uint_as_float(state.word4);
    state.constants.value5 = __uint_as_float(state.word5);
    state.constants.value6 = __uint_as_float(state.word6);
    state.constants.value7 = __uint_as_float(state.word7);
    return state;
}

static __device__ __forceinline__ CusrAstSettingsLeafValues cusr_ast_settings_load_row(
    const float* x_tile,
    uint32_t shared_stride,
    uint32_t row,
    const CusrAstSettingsSettingState& state)
{
    CusrAstSettingsLeafValues values = state.constants;
    const float* const x_row = x_tile + (size_t)row * shared_stride;

    if ((state.mask & 0x01u) != 0u) values.value0 = x_row[state.word0];
    if ((state.mask & 0x02u) != 0u) values.value1 = x_row[state.word1];
    if ((state.mask & 0x04u) != 0u) values.value2 = x_row[state.word2];
    if ((state.mask & 0x08u) != 0u) values.value3 = x_row[state.word3];
    if ((state.mask & 0x10u) != 0u) values.value4 = x_row[state.word4];
    if ((state.mask & 0x20u) != 0u) values.value5 = x_row[state.word5];
    if ((state.mask & 0x40u) != 0u) values.value6 = x_row[state.word6];
    if ((state.mask & 0x80u) != 0u) values.value7 = x_row[state.word7];
    return values;
}

static __device__ __forceinline__ uint32_t cusr_ast_settings_load_tile(
    const float* __restrict__ X,
    const float* __restrict__ target,
    size_t num_rows,
    uint32_t num_columns,
    size_t leading_dim,
    size_t tile_idx,
    float* x_tile,
    float* target_tile,
    uint32_t shared_stride)
{
    const uint32_t tile_row = threadIdx.x;
    const size_t row_base = tile_idx * CUSR_AST_SETTINGS_TILE_ROWS;
    const size_t row = row_base + tile_row;
    const bool row_valid = row < num_rows;

    target_tile[tile_row] = row_valid ? target[row] : 0.0f;

    #pragma unroll 1
    for (uint32_t column = 0u; column < num_columns; ++column) {
        x_tile[(size_t)tile_row * shared_stride + column] = row_valid ? X[(size_t)column * leading_dim + row] : 0.0f;
    }

    return row_base + CUSR_AST_SETTINGS_TILE_ROWS <= num_rows
        ? CUSR_AST_SETTINGS_TILE_ROWS
        : (row_base < num_rows ? (uint32_t)(num_rows - row_base) : 0u);
}

template <int Expression, int NumAsts>
static __global__ __launch_bounds__(CUSR_AST_SETTINGS_THREADS) void cusr_ast_settings_row_tile_kernel(
    const float* __restrict__ X,
    const float* __restrict__ target,
    size_t num_rows,
    uint32_t num_columns,
    size_t leading_dim,
    const uint8_t* __restrict__ leaf_masks,
    const uint32_t* __restrict__ leaf_words,
    size_t leaf_words_stride,
    uint32_t num_settings,
    float* __restrict__ output)
{
    extern __shared__ float shared[];
    const uint32_t shared_stride = cusr_ast_settings_shared_stride(num_columns);
    float* const x_tile = shared;
    float* const target_tile = x_tile + (size_t)CUSR_AST_SETTINGS_TILE_ROWS * shared_stride;
    const size_t tile_idx = blockIdx.x;
    const uint32_t row_limit = cusr_ast_settings_load_tile(X, target, num_rows, num_columns, leading_dim, tile_idx, x_tile, target_tile, shared_stride);

    __syncthreads();

    for (uint32_t setting_base = 0u; setting_base < num_settings; setting_base += CUSR_AST_SETTINGS_THREADS) {
        const uint32_t setting = setting_base + threadIdx.x;

        if (setting < num_settings) {
            const CusrAstSettingsSettingState state = cusr_ast_settings_load_setting(leaf_masks, leaf_words, leaf_words_stride, setting);
            float sse[NumAsts] = {};

            #pragma unroll 1
            for (uint32_t row = 0u; row < row_limit; ++row) {
                const CusrAstSettingsLeafValues values = cusr_ast_settings_load_row(x_tile, shared_stride, row, state);
                cusr_ast_settings_accumulate<Expression, 0, NumAsts>(sse, values, target_tile[row]);
            }

            #pragma unroll
            for (int ast = 0; ast < NumAsts; ++ast) {
                const size_t output_idx = (size_t)ast * num_settings + setting;
                atomicAdd(output + output_idx, sse[ast]);
            }
        }
    }
}

#endif /* CUSR_AST_SETTINGS_EXPERIMENT_KERNELS_CUH_INCLUDED */
