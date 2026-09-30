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
typedef unsigned int u32;
typedef unsigned long long u64;

static __device__ __forceinline__
uint4
philox4x32_round(
    uint4 ctr,
    uint2 key
) {
    const u32 m0 = 0xD2511F53u;
    const u32 m1 = 0xCD9E8D57u;
    u32 hi0 = __umulhi(m0, ctr.x);
    u32 hi1 = __umulhi(m1, ctr.z);
    u32 lo0 = m0 * ctr.x;
    u32 lo1 = m1 * ctr.z;
    return make_uint4(hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0);
}

static __device__ __forceinline__
uint4
philox4x32_10(
    uint4 ctr,
    uint2 key
) {
    const u32 w0 = 0x9E3779B9u;
    const u32 w1 = 0xBB67AE85u;
    #pragma unroll
    for (int i = 0; i < 10; ++i) {
        ctr = philox4x32_round(ctr, key);
        key.x += w0;
        key.y += w1;
    }
    return ctr;
}

static __device__ __forceinline__
float
u32_uniform_01(
    u32 x
) {
    return (float)x * 2.3283064e-10f + 1.1641532e-10f;
}

static __device__ __forceinline__
int
constant_word(
    u32 x
) {
    switch (x % 7u) {
        case 0u: return __float_as_int(-2.0f);
        case 1u: return __float_as_int(-1.0f);
        case 2u: return __float_as_int(-0.5f);
        case 3u: return __float_as_int(0.0f);
        case 4u: return __float_as_int(0.5f);
        case 5u: return __float_as_int(1.0f);
        default: return __float_as_int(2.0f);
    }
}

extern "C" __global__
void
implicit_sindy_map_elites_generate_settings_philox(
    int* leaf_masks,
    int* leaf_words,
    const int* inherited_masks,
    const int* inherited_words,
    const int* inherited_terms,
    long long num_cohorts,
    long long num_settings,
    long long leaf_masks_cohort_stride,
    long long leaf_words_cohort_stride,
    long long leaf_words_feature_stride,
    long long inherited_masks_cohort_stride,
    long long inherited_words_cohort_stride,
    long long inherited_words_feature_stride,
    int primitive_cols,
    unsigned long long seed,
    unsigned long long generation,
    unsigned long long start_cohort_seed,
    float feature_leaf_probability
) {
    long long cohort_idx = (long long)blockIdx.z;
    long long setting_idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    int feature_idx = (int)blockIdx.y;
    if (cohort_idx >= num_cohorts || setting_idx >= num_settings || feature_idx >= 32) return;

    long long mask_idx = cohort_idx * leaf_masks_cohort_stride + setting_idx * 32 + feature_idx;
    long long word_base = cohort_idx * leaf_words_cohort_stride + (setting_idx * 32 + feature_idx) * leaf_words_feature_stride;
    int inherited_count = inherited_terms[cohort_idx];

    if (feature_idx < inherited_count) {
        long long inherited_mask_idx = cohort_idx * inherited_masks_cohort_stride + feature_idx;
        long long inherited_word_base = cohort_idx * inherited_words_cohort_stride + feature_idx * inherited_words_feature_stride;
        leaf_masks[mask_idx] = inherited_masks[inherited_mask_idx];
        #pragma unroll
        for (int leaf_idx = 0; leaf_idx < 8; ++leaf_idx) {
            leaf_words[word_base + leaf_idx] = inherited_words[inherited_word_base + leaf_idx];
        }
        return;
    }

    unsigned long long global_cohort = start_cohort_seed + (unsigned long long)cohort_idx;
    uint2 key = make_uint2((u32)seed, (u32)(seed >> 32));
    int mask = 0;

    #pragma unroll
    for (int leaf_idx = 0; leaf_idx < 8; ++leaf_idx) {
        u64 linear = (((((u64)generation * (u64)num_cohorts + (u64)cohort_idx) *
                        (u64)num_settings + (u64)setting_idx) *
                       32ull + (u64)feature_idx) *
                      8ull + (u64)leaf_idx);
        uint4 ctr = make_uint4(
            (u32)linear,
            (u32)(linear >> 32),
            (u32)global_cohort,
            (u32)((global_cohort >> 32) ^ generation)
        );
        uint4 rnd = philox4x32_10(ctr, key);
        int use_feature = u32_uniform_01(rnd.x) < feature_leaf_probability;
        int word;
        if (use_feature != 0 && primitive_cols > 0) {
            mask |= (1 << leaf_idx);
            word = (int)(rnd.y % (u32)primitive_cols);
        } else {
            word = constant_word(rnd.y);
        }
        leaf_words[word_base + leaf_idx] = word;
    }
    leaf_masks[mask_idx] = mask;
}
