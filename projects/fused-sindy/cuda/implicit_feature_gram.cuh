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
#ifndef IMPLICIT_FEATURE_GRAM_CUH_INCLUDE
#define IMPLICIT_FEATURE_GRAM_CUH_INCLUDE

#ifndef IMPLICIT_SINDY_FEATURE_GRAM_EXTERNAL_EVAL
#include "implicit_sindy.h"
#endif
#include "implicit_sindy_internal_constants.h"

#ifdef IMPLICIT_SINDY_FEATURE_GRAM_EXTERNAL_EVAL
typedef int int32_t;
typedef unsigned int uint32_t;
typedef long long int64_t;
#else
#include <stdint.h>
#endif

namespace implicit_sindy_feature_gram_kernel {

constexpr int kFeatures = IMPLICIT_SINDY_INTERNAL_FEATURES;
constexpr int kLeaves = IMPLICIT_SINDY_INTERNAL_LEAVES;
constexpr int kPrimitiveMax = IMPLICIT_SINDY_INTERNAL_PRIMITIVE_FEATURES_MAX;
constexpr int kRowsPerTile = IMPLICIT_SINDY_INTERNAL_ROWS_PER_TILE;
constexpr int kTargetRhsMax = IMPLICIT_SINDY_INTERNAL_TARGET_RHS_MAX;
constexpr int kThreads = 128;
constexpr int kWarpSize = 32;
constexpr int kWarps = kThreads / kWarpSize;
constexpr int kGramElements = kFeatures * kFeatures;
constexpr int kGramValuesPerThread = (kGramElements + kThreads - 1) / kThreads;

struct SharedStorage {
    float implicit_panel[kFeatures * kRowsPerTile];
    float primitive[kPrimitiveMax * kRowsPerTile];
    float target[kRowsPerTile];
    float leaf_constants[kFeatures * kLeaves];
    const float* leaf_ptrs[kFeatures * kLeaves];
    int leaf_strides[kFeatures * kLeaves];
    float warp_x_sum[kWarps * kFeatures];
    float target_y_sum[kTargetRhsMax];
    float target_yy[kTargetRhsMax];
};

__device__ __forceinline__
float
u32_as_f32(
    uint32_t bits
) {
    union {
        uint32_t u;
        float f;
    } value;
    value.u = bits;
    return value.f;
}

__device__ __forceinline__
float
warp_reduce_sum(
    float value
) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__
float
subwarp4_reduce_sum_mask(
    uint32_t mask,
    float value
) {
    value += __shfl_down_sync(mask, value, 2, 4);
    value += __shfl_down_sync(mask, value, 1, 4);
    return value;
}

__device__ __forceinline__
float
subwarp4_reduce_sum(
    float value
) {
    value = subwarp4_reduce_sum_mask(
        0xffffffffu,
        value
    );
    return value;
}

#ifdef IMPLICIT_SINDY_FEATURE_GRAM_EXTERNAL_EVAL
template <int KernelIdx>
__device__ __forceinline__
void
implicit_feature_eval_selector(
    float* implicit_panel,
    const float* const* leaf_ptrs,
    const int* leaf_strides,
    int active_rows
);
#else
#define IMPLICIT_SINDY_FEATURE_INJECT_SITE(SITE_NAME, value, leaf) do { \
    const float leaf0 = (leaf)[0]; \
    const float leaf1 = (leaf)[1]; \
    const float leaf2 = (leaf)[2]; \
    const float leaf3 = (leaf)[3]; \
    const float leaf4 = (leaf)[4]; \
    const float leaf5 = (leaf)[5]; \
    const float leaf6 = (leaf)[6]; \
    const float leaf7 = (leaf)[7]; \
    asm volatile( \
        "{\n\t" \
        ".reg .f32 %%_x0;\n\t" \
        ".reg .f32 %%_x1;\n\t" \
        ".reg .f32 %%_x2;\n\t" \
        ".reg .f32 %%_x3;\n\t" \
        ".reg .f32 %%_x4;\n\t" \
        ".reg .f32 %%_x5;\n\t" \
        ".reg .f32 %%_x6;\n\t" \
        ".reg .f32 %%_x7;\n\t" \
        ".reg .f32 %%_x8;\n\t" \
        "mov.f32 %%_x0, %0;\n\t" \
        "mov.f32 %%_x1, %1;\n\t" \
        "mov.f32 %%_x2, %2;\n\t" \
        "mov.f32 %%_x3, %3;\n\t" \
        "mov.f32 %%_x4, %4;\n\t" \
        "mov.f32 %%_x5, %5;\n\t" \
        "mov.f32 %%_x6, %6;\n\t" \
        "mov.f32 %%_x7, %7;\n\t" \
        "mov.f32 %%_x8, %8;\n\t" \
        "// PTX_INJECT_START " SITE_NAME "\n\t" \
        "// _x0 m f32 F32 value\n\t" \
        "// _x1 i f32 F32 leaf0\n\t" \
        "// _x2 i f32 F32 leaf1\n\t" \
        "// _x3 i f32 F32 leaf2\n\t" \
        "// _x4 i f32 F32 leaf3\n\t" \
        "// _x5 i f32 F32 leaf4\n\t" \
        "// _x6 i f32 F32 leaf5\n\t" \
        "// _x7 i f32 F32 leaf6\n\t" \
        "// _x8 i f32 F32 leaf7\n\t" \
        "// PTX_INJECT_END\n\t" \
        "mov.f32 %0, %%_x0;\n\t" \
        "}" \
        : "+f"(value) \
        : "f"(leaf0), "f"(leaf1), "f"(leaf2), "f"(leaf3), \
          "f"(leaf4), "f"(leaf5), "f"(leaf6), "f"(leaf7)); \
} while (0)

#define IMPLICIT_SINDY_SITE_CASE(FEATURE_ID, SITE_ID) \
    case FEATURE_ID: { \
        float value = 0.0f; \
        IMPLICIT_SINDY_FEATURE_INJECT_SITE("implicit_feature_" #SITE_ID, value, leaf); \
        return value; \
    }

template <int KernelIdx>
__device__ __forceinline__
float
evaluate_feature_site(
    int feature_idx,
    const float leaf[kLeaves]
) {
    if constexpr (KernelIdx == 0) {
        switch (feature_idx) {
            IMPLICIT_SINDY_SITE_CASE(0, 0);
            IMPLICIT_SINDY_SITE_CASE(1, 1);
            IMPLICIT_SINDY_SITE_CASE(2, 2);
            IMPLICIT_SINDY_SITE_CASE(3, 3);
            IMPLICIT_SINDY_SITE_CASE(4, 4);
            IMPLICIT_SINDY_SITE_CASE(5, 5);
            IMPLICIT_SINDY_SITE_CASE(6, 6);
            IMPLICIT_SINDY_SITE_CASE(7, 7);
            IMPLICIT_SINDY_SITE_CASE(8, 8);
            IMPLICIT_SINDY_SITE_CASE(9, 9);
            IMPLICIT_SINDY_SITE_CASE(10, 10);
            IMPLICIT_SINDY_SITE_CASE(11, 11);
            IMPLICIT_SINDY_SITE_CASE(12, 12);
            IMPLICIT_SINDY_SITE_CASE(13, 13);
            IMPLICIT_SINDY_SITE_CASE(14, 14);
            IMPLICIT_SINDY_SITE_CASE(15, 15);
            IMPLICIT_SINDY_SITE_CASE(16, 16);
            IMPLICIT_SINDY_SITE_CASE(17, 17);
            IMPLICIT_SINDY_SITE_CASE(18, 18);
            IMPLICIT_SINDY_SITE_CASE(19, 19);
            IMPLICIT_SINDY_SITE_CASE(20, 20);
            IMPLICIT_SINDY_SITE_CASE(21, 21);
            IMPLICIT_SINDY_SITE_CASE(22, 22);
            IMPLICIT_SINDY_SITE_CASE(23, 23);
            IMPLICIT_SINDY_SITE_CASE(24, 24);
            IMPLICIT_SINDY_SITE_CASE(25, 25);
            IMPLICIT_SINDY_SITE_CASE(26, 26);
            IMPLICIT_SINDY_SITE_CASE(27, 27);
            IMPLICIT_SINDY_SITE_CASE(28, 28);
            IMPLICIT_SINDY_SITE_CASE(29, 29);
            IMPLICIT_SINDY_SITE_CASE(30, 30);
            IMPLICIT_SINDY_SITE_CASE(31, 31);
            default:
                return 0.0f;
        }
    } else if constexpr (KernelIdx == 1) {
        switch (feature_idx) {
            IMPLICIT_SINDY_SITE_CASE(0, 32);
            IMPLICIT_SINDY_SITE_CASE(1, 33);
            IMPLICIT_SINDY_SITE_CASE(2, 34);
            IMPLICIT_SINDY_SITE_CASE(3, 35);
            IMPLICIT_SINDY_SITE_CASE(4, 36);
            IMPLICIT_SINDY_SITE_CASE(5, 37);
            IMPLICIT_SINDY_SITE_CASE(6, 38);
            IMPLICIT_SINDY_SITE_CASE(7, 39);
            IMPLICIT_SINDY_SITE_CASE(8, 40);
            IMPLICIT_SINDY_SITE_CASE(9, 41);
            IMPLICIT_SINDY_SITE_CASE(10, 42);
            IMPLICIT_SINDY_SITE_CASE(11, 43);
            IMPLICIT_SINDY_SITE_CASE(12, 44);
            IMPLICIT_SINDY_SITE_CASE(13, 45);
            IMPLICIT_SINDY_SITE_CASE(14, 46);
            IMPLICIT_SINDY_SITE_CASE(15, 47);
            IMPLICIT_SINDY_SITE_CASE(16, 48);
            IMPLICIT_SINDY_SITE_CASE(17, 49);
            IMPLICIT_SINDY_SITE_CASE(18, 50);
            IMPLICIT_SINDY_SITE_CASE(19, 51);
            IMPLICIT_SINDY_SITE_CASE(20, 52);
            IMPLICIT_SINDY_SITE_CASE(21, 53);
            IMPLICIT_SINDY_SITE_CASE(22, 54);
            IMPLICIT_SINDY_SITE_CASE(23, 55);
            IMPLICIT_SINDY_SITE_CASE(24, 56);
            IMPLICIT_SINDY_SITE_CASE(25, 57);
            IMPLICIT_SINDY_SITE_CASE(26, 58);
            IMPLICIT_SINDY_SITE_CASE(27, 59);
            IMPLICIT_SINDY_SITE_CASE(28, 60);
            IMPLICIT_SINDY_SITE_CASE(29, 61);
            IMPLICIT_SINDY_SITE_CASE(30, 62);
            IMPLICIT_SINDY_SITE_CASE(31, 63);
            default:
                return 0.0f;
        }
    } else if constexpr (KernelIdx == 2) {
        switch (feature_idx) {
            IMPLICIT_SINDY_SITE_CASE(0, 64);
            IMPLICIT_SINDY_SITE_CASE(1, 65);
            IMPLICIT_SINDY_SITE_CASE(2, 66);
            IMPLICIT_SINDY_SITE_CASE(3, 67);
            IMPLICIT_SINDY_SITE_CASE(4, 68);
            IMPLICIT_SINDY_SITE_CASE(5, 69);
            IMPLICIT_SINDY_SITE_CASE(6, 70);
            IMPLICIT_SINDY_SITE_CASE(7, 71);
            IMPLICIT_SINDY_SITE_CASE(8, 72);
            IMPLICIT_SINDY_SITE_CASE(9, 73);
            IMPLICIT_SINDY_SITE_CASE(10, 74);
            IMPLICIT_SINDY_SITE_CASE(11, 75);
            IMPLICIT_SINDY_SITE_CASE(12, 76);
            IMPLICIT_SINDY_SITE_CASE(13, 77);
            IMPLICIT_SINDY_SITE_CASE(14, 78);
            IMPLICIT_SINDY_SITE_CASE(15, 79);
            IMPLICIT_SINDY_SITE_CASE(16, 80);
            IMPLICIT_SINDY_SITE_CASE(17, 81);
            IMPLICIT_SINDY_SITE_CASE(18, 82);
            IMPLICIT_SINDY_SITE_CASE(19, 83);
            IMPLICIT_SINDY_SITE_CASE(20, 84);
            IMPLICIT_SINDY_SITE_CASE(21, 85);
            IMPLICIT_SINDY_SITE_CASE(22, 86);
            IMPLICIT_SINDY_SITE_CASE(23, 87);
            IMPLICIT_SINDY_SITE_CASE(24, 88);
            IMPLICIT_SINDY_SITE_CASE(25, 89);
            IMPLICIT_SINDY_SITE_CASE(26, 90);
            IMPLICIT_SINDY_SITE_CASE(27, 91);
            IMPLICIT_SINDY_SITE_CASE(28, 92);
            IMPLICIT_SINDY_SITE_CASE(29, 93);
            IMPLICIT_SINDY_SITE_CASE(30, 94);
            IMPLICIT_SINDY_SITE_CASE(31, 95);
            default:
                return 0.0f;
        }
    } else if constexpr (KernelIdx == 3) {
        switch (feature_idx) {
            IMPLICIT_SINDY_SITE_CASE(0, 96);
            IMPLICIT_SINDY_SITE_CASE(1, 97);
            IMPLICIT_SINDY_SITE_CASE(2, 98);
            IMPLICIT_SINDY_SITE_CASE(3, 99);
            IMPLICIT_SINDY_SITE_CASE(4, 100);
            IMPLICIT_SINDY_SITE_CASE(5, 101);
            IMPLICIT_SINDY_SITE_CASE(6, 102);
            IMPLICIT_SINDY_SITE_CASE(7, 103);
            IMPLICIT_SINDY_SITE_CASE(8, 104);
            IMPLICIT_SINDY_SITE_CASE(9, 105);
            IMPLICIT_SINDY_SITE_CASE(10, 106);
            IMPLICIT_SINDY_SITE_CASE(11, 107);
            IMPLICIT_SINDY_SITE_CASE(12, 108);
            IMPLICIT_SINDY_SITE_CASE(13, 109);
            IMPLICIT_SINDY_SITE_CASE(14, 110);
            IMPLICIT_SINDY_SITE_CASE(15, 111);
            IMPLICIT_SINDY_SITE_CASE(16, 112);
            IMPLICIT_SINDY_SITE_CASE(17, 113);
            IMPLICIT_SINDY_SITE_CASE(18, 114);
            IMPLICIT_SINDY_SITE_CASE(19, 115);
            IMPLICIT_SINDY_SITE_CASE(20, 116);
            IMPLICIT_SINDY_SITE_CASE(21, 117);
            IMPLICIT_SINDY_SITE_CASE(22, 118);
            IMPLICIT_SINDY_SITE_CASE(23, 119);
            IMPLICIT_SINDY_SITE_CASE(24, 120);
            IMPLICIT_SINDY_SITE_CASE(25, 121);
            IMPLICIT_SINDY_SITE_CASE(26, 122);
            IMPLICIT_SINDY_SITE_CASE(27, 123);
            IMPLICIT_SINDY_SITE_CASE(28, 124);
            IMPLICIT_SINDY_SITE_CASE(29, 125);
            IMPLICIT_SINDY_SITE_CASE(30, 126);
            IMPLICIT_SINDY_SITE_CASE(31, 127);
            default:
                return 0.0f;
        }
    } else {
        return 0.0f;
    }
}

#undef IMPLICIT_SINDY_SITE_CASE
#undef IMPLICIT_SINDY_FEATURE_INJECT_SITE
#endif

__device__ __forceinline__
void
setup_leaf_refs(
    const int32_t* leaf_masks,
    const int32_t* leaf_words,
    int64_t leaf_words_feature_stride,
    int64_t num_primitive_features,
    SharedStorage& storage,
    int setting_idx
) {
    const int tid = static_cast<int>(threadIdx.x);

    for (int linear = tid; linear < kFeatures * kLeaves; linear += kThreads) {
        const int feature = linear / kLeaves;
        const int leaf = linear - feature * kLeaves;
        const int setting_feature = setting_idx * kFeatures + feature;
        const uint32_t mask = static_cast<uint32_t>(leaf_masks[setting_feature]);
        const uint32_t word = static_cast<uint32_t>(
            leaf_words[leaf + static_cast<int64_t>(setting_feature) * leaf_words_feature_stride]);

        if (((mask >> leaf) & 1u) != 0u) {
            const int col = static_cast<int>(word);
            if (col >= 0 && col < static_cast<int>(num_primitive_features)) {
                storage.leaf_ptrs[linear] = storage.primitive + col * kRowsPerTile;
                storage.leaf_strides[linear] = 1;
            } else {
                storage.leaf_constants[linear] = 0.0f;
                storage.leaf_ptrs[linear] = storage.leaf_constants + linear;
                storage.leaf_strides[linear] = 0;
            }
        } else {
            storage.leaf_constants[linear] = u32_as_f32(word);
            storage.leaf_ptrs[linear] = storage.leaf_constants + linear;
            storage.leaf_strides[linear] = 0;
        }
    }
}

__device__ __forceinline__
void
clear_warp_stats(
    SharedStorage& storage
) {
    const int tid = static_cast<int>(threadIdx.x);

    for (int linear = tid; linear < kWarps * kFeatures; linear += kThreads) {
        storage.warp_x_sum[linear] = 0.0f;
    }
    for (int rhs_idx = tid; rhs_idx < kTargetRhsMax; rhs_idx += kThreads) {
        storage.target_y_sum[rhs_idx] = 0.0f;
        storage.target_yy[rhs_idx] = 0.0f;
    }
}

__device__ __forceinline__
void
load_primitive_tile(
    const float* primitive_features,
    int64_t primitive_feature_stride,
    int64_t num_primitive_features,
    SharedStorage& storage,
    int row0,
    int row_end
) {
    const int tid = static_cast<int>(threadIdx.x);
    const int cols = static_cast<int>(num_primitive_features);
    const int total = kPrimitiveMax * kRowsPerTile;

    for (int linear = tid; linear < total; linear += kThreads) {
        const int col = linear / kRowsPerTile;
        const int row = linear - col * kRowsPerTile;
        const int global_row = row0 + row;
        float value = 0.0f;

        if (col < cols && global_row < row_end) {
            value = primitive_features[
                static_cast<int64_t>(global_row) +
                static_cast<int64_t>(col) * primitive_feature_stride
            ];
        }
        storage.primitive[linear] = value;
    }
}

__device__ __forceinline__
void
load_target_tile(
    const float* targets,
    int64_t target_rhs_stride,
    SharedStorage& storage,
    int rhs_idx,
    int row0,
    int row_end
) {
    const int tid = static_cast<int>(threadIdx.x);

    for (int row = tid; row < kRowsPerTile; row += kThreads) {
        const int global_row = row0 + row;
        storage.target[row] = global_row < row_end
            ? targets[static_cast<int64_t>(global_row) + static_cast<int64_t>(rhs_idx) * target_rhs_stride]
            : 0.0f;
    }
}

template <int KernelIdx>
__device__ __forceinline__
void
evaluate_implicit_tile(
    SharedStorage& storage,
    int row0,
    int row_end
) {
#ifdef IMPLICIT_SINDY_FEATURE_GRAM_EXTERNAL_EVAL
    const int active_rows = row0 + kRowsPerTile <= row_end
        ? kRowsPerTile
        : row_end - row0;
    implicit_feature_eval_selector<KernelIdx>(
        storage.implicit_panel,
        storage.leaf_ptrs,
        storage.leaf_strides,
        active_rows
    );
#else
    const int tid = static_cast<int>(threadIdx.x);
    const int total = kFeatures * kRowsPerTile;

    for (int linear = tid; linear < total; linear += kThreads) {
        const int feature = linear / kRowsPerTile;
        const int row = linear - feature * kRowsPerTile;
        const int global_row = row0 + row;
        float leaf[kLeaves];
        float value = 0.0f;

        #pragma unroll
        for (int leaf_idx = 0; leaf_idx < kLeaves; ++leaf_idx) {
            const int ref_idx = feature * kLeaves + leaf_idx;
            leaf[leaf_idx] = storage.leaf_ptrs[ref_idx][row * storage.leaf_strides[ref_idx]];
        }

        if (global_row < row_end) {
            value = evaluate_feature_site<KernelIdx>(
                feature,
                leaf
            );
        }
        storage.implicit_panel[linear] = value;
    }
#endif
}

__device__ __forceinline__
void
accumulate_implicit_x_sum_warp(
    SharedStorage& storage
) {
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & (kWarpSize - 1);
    const int warp = tid / kWarpSize;

    for (int feature = 0; feature < kFeatures; ++feature) {
        float local_x_sum = 0.0f;

        for (int row = tid; row < kRowsPerTile; row += kThreads) {
            const float x = storage.implicit_panel[feature * kRowsPerTile + row];
            local_x_sum += x;
        }

        local_x_sum = warp_reduce_sum(local_x_sum);
        if (lane == 0) {
            const int offset = warp * kFeatures + feature;
            storage.warp_x_sum[offset] += local_x_sum;
        }
    }
}

template <int RhsIdx>
__device__ __forceinline__
void
accumulate_implicit_xty_warp(
    SharedStorage& storage,
    float xty_accum[kTargetRhsMax]
) {
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 3;
    const int feature = tid >> 2;
    float local_xty = 0.0f;
    float local_y_sum = 0.0f;
    float local_yy = 0.0f;

    for (int row = lane; row < kRowsPerTile; row += 4) {
        const float x = storage.implicit_panel[feature * kRowsPerTile + row];
        const float y = storage.target[row];
        local_xty += x * y;
        if (feature == 0) {
            local_y_sum += y;
            local_yy += y * y;
        }
    }

    local_xty = subwarp4_reduce_sum(local_xty);
    if (feature == 0) {
        local_y_sum = subwarp4_reduce_sum_mask(
            0x0000000fu,
            local_y_sum
        );
        local_yy = subwarp4_reduce_sum_mask(
            0x0000000fu,
            local_yy
        );
    }

    if (lane == 0) {
        xty_accum[RhsIdx] += local_xty;
        if (feature == 0) {
            storage.target_y_sum[RhsIdx] += local_y_sum;
            storage.target_yy[RhsIdx] += local_yy;
        }
    }
}

template <int RhsIdx>
__device__ __forceinline__
void
process_target_rhs_tile(
    const float* targets,
    int64_t target_rhs_stride,
    SharedStorage& storage,
    int target_rhs_count,
    int row0,
    int row_end,
    float xty_accum[kTargetRhsMax]
) {
    if (RhsIdx < target_rhs_count) {
        load_target_tile(
            targets,
            target_rhs_stride,
            storage,
            RhsIdx,
            row0,
            row_end
        );
        __syncthreads();

        accumulate_implicit_xty_warp<RhsIdx>(
            storage,
            xty_accum
        );
        __syncthreads();
    }

    if constexpr (RhsIdx + 1 < kTargetRhsMax) {
        process_target_rhs_tile<RhsIdx + 1>(
            targets,
            target_rhs_stride,
            storage,
            target_rhs_count,
            row0,
            row_end,
            xty_accum
        );
    }
}

__device__ __forceinline__
void
accumulate_gram_tile(
    const SharedStorage& storage,
    float gram_accum[kGramValuesPerThread]
) {
    const int tid = static_cast<int>(threadIdx.x);

    #pragma unroll
    for (int slot = 0; slot < kGramValuesPerThread; ++slot) {
        const int linear = tid + slot * kThreads;
        if (linear < kGramElements) {
            const int row_feature = linear % kFeatures;
            const int col_feature = linear / kFeatures;
            float sum = 0.0f;

            #pragma unroll
            for (int row = 0; row < kRowsPerTile; ++row) {
                const float lhs = storage.implicit_panel[row_feature * kRowsPerTile + row];
                const float rhs = storage.implicit_panel[col_feature * kRowsPerTile + row];
                sum += lhs * rhs;
            }
            gram_accum[slot] += sum;
        }
    }
}

template <int RhsIdx>
__device__ __forceinline__
void
write_target_stats(
    float* xty_out,
    int64_t xty_rhs_stride,
    float* y_sum_out,
    float* yy_out,
    const SharedStorage& storage,
    const float xty_accum[kTargetRhsMax],
    int num_target_rhs,
    int output_idx
) {
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 3;
    const int feature = tid >> 2;

    if (RhsIdx < num_target_rhs) {
        if (lane == 0) {
            xty_out[
                (static_cast<int64_t>(output_idx) * num_target_rhs + RhsIdx) * xty_rhs_stride +
                feature
            ] = xty_accum[RhsIdx];
        }
        if (tid == 0) {
            y_sum_out[static_cast<int64_t>(output_idx) * num_target_rhs + RhsIdx] =
                storage.target_y_sum[RhsIdx];
            yy_out[static_cast<int64_t>(output_idx) * num_target_rhs + RhsIdx] =
                storage.target_yy[RhsIdx];
        }
    }

    if constexpr (RhsIdx + 1 < kTargetRhsMax) {
        write_target_stats<RhsIdx + 1>(
            xty_out,
            xty_rhs_stride,
            y_sum_out,
            yy_out,
            storage,
            xty_accum,
            num_target_rhs,
            output_idx
        );
    }
}

__device__ __forceinline__
void
write_final_stats(
    float* x_sum_out,
    const SharedStorage& storage,
    int output_idx
) {
    const int tid = static_cast<int>(threadIdx.x);

    if (tid < kFeatures) {
        float x_sum = 0.0f;
        for (int warp = 0; warp < kWarps; ++warp) {
            const int offset = warp * kFeatures + tid;
            x_sum += storage.warp_x_sum[offset];
        }
        x_sum_out[static_cast<int64_t>(output_idx) * kFeatures + tid] = x_sum;
    }
}

template <int KernelIdx>
__device__ __forceinline__
void
implicit_feature_gram_kernel_device(
    const float* primitive_features,
    int64_t row_count,
    int64_t primitive_feature_stride,
    int64_t num_primitive_features,
    const float* targets,
    int64_t target_rhs_stride,
    int64_t num_target_rhs,
    const int32_t* leaf_masks,
    const int32_t* leaf_words,
    int64_t leaf_words_feature_stride,
    float* gram,
    int64_t gram_col_stride,
    float* x_sum,
    float* xty,
    int64_t xty_rhs_stride,
    float* y_sum,
    float* yy
) {
    __shared__ __align__(16) SharedStorage storage;
    float gram_accum[kGramValuesPerThread];
    float xty_accum[kTargetRhsMax];
    const int setting_idx = static_cast<int>(blockIdx.x);
    const int output_idx = setting_idx;
    const int target_rhs_count = static_cast<int>(num_target_rhs);
    int row_begin = 0;
    int row_end = static_cast<int>(row_count);

    if (target_rhs_count <= 0 || target_rhs_count > kTargetRhsMax) {
        return;
    }

    #pragma unroll
    for (int slot = 0; slot < kGramValuesPerThread; ++slot) {
        gram_accum[slot] = 0.0f;
    }
    #pragma unroll
    for (int rhs_idx = 0; rhs_idx < kTargetRhsMax; ++rhs_idx) {
        xty_accum[rhs_idx] = 0.0f;
    }

    setup_leaf_refs(
        leaf_masks,
        leaf_words,
        leaf_words_feature_stride,
        num_primitive_features,
        storage,
        setting_idx
    );
    clear_warp_stats(storage);
    __syncthreads();

    for (int row0 = row_begin; row0 < row_end; row0 += kRowsPerTile) {
        load_primitive_tile(
            primitive_features,
            primitive_feature_stride,
            num_primitive_features,
            storage,
            row0,
            row_end
        );
        __syncthreads();

        evaluate_implicit_tile<KernelIdx>(
            storage,
            row0,
            row_end
        );
        __syncthreads();

        accumulate_implicit_x_sum_warp(storage);
        accumulate_gram_tile(storage, gram_accum);

        process_target_rhs_tile<0>(
            targets,
            target_rhs_stride,
            storage,
            target_rhs_count,
            row0,
            row_end,
            xty_accum
        );
    }

    #pragma unroll
    for (int slot = 0; slot < kGramValuesPerThread; ++slot) {
        const int linear = static_cast<int>(threadIdx.x) + slot * kThreads;
        if (linear < kGramElements) {
            const int row_feature = linear % kFeatures;
            const int col_feature = linear / kFeatures;
            gram[
                static_cast<int64_t>(output_idx) * kFeatures * gram_col_stride +
                static_cast<int64_t>(row_feature) +
                static_cast<int64_t>(col_feature) * gram_col_stride
            ] = gram_accum[slot];
        }
    }

    write_final_stats(
        x_sum,
        storage,
        output_idx
    );
    write_target_stats<0>(
        xty,
        xty_rhs_stride,
        y_sum,
        yy,
        storage,
        xty_accum,
        target_rhs_count,
        output_idx
    );
}

}  // namespace implicit_sindy_feature_gram_kernel

#endif /* IMPLICIT_FEATURE_GRAM_CUH_INCLUDE */
