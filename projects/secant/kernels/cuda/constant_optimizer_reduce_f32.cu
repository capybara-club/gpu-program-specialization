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
static __forceinline__ __device__ uint4 secant_optimizer_philox_round(uint4 c, uint2 k) {
    const unsigned int hi0 = __umulhi(0xd2511f53u, c.x);
    const unsigned int hi1 = __umulhi(0xcd9e8d57u, c.z);
    const unsigned int lo0 = 0xd2511f53u * c.x;
    const unsigned int lo1 = 0xcd9e8d57u * c.z;
    return make_uint4(hi1 ^ c.y ^ k.x, lo1, hi0 ^ c.w ^ k.y, lo0);
}

static __forceinline__ __device__ uint4 secant_optimizer_philox4x32_10(uint4 c, uint2 k) {
    #pragma unroll
    for (int round = 0; round < 10; ++round) {
        c = secant_optimizer_philox_round(c, k);
        k.x += 0x9e3779b9u;
        k.y += 0xbb67ae85u;
    }
    return c;
}

static __forceinline__ __device__ uint4 secant_optimizer_random4(
    unsigned long long setting,
    unsigned long long ast_index,
    unsigned long long seed,
    unsigned long long generation,
    unsigned long long iteration,
    unsigned int group
) {
    const uint4 counter = make_uint4(
        (unsigned int)setting,
        (unsigned int)(setting >> 32),
        (unsigned int)ast_index,
        (unsigned int)(ast_index >> 32));
    const uint2 key = make_uint2(
        (unsigned int)seed ^ (unsigned int)generation * 0x9e3779b9u ^
            (unsigned int)(iteration >> 32) * 0x85ebca6bu ^ group * 0x27d4eb2du,
        (unsigned int)(seed >> 32) ^ (unsigned int)(generation >> 32) * 0xbb67ae85u ^
            (unsigned int)iteration * 0xc2b2ae35u ^ group * 0x165667b1u);
    return secant_optimizer_philox4x32_10(counter, key);
}

static __forceinline__ __device__ float secant_optimizer_delta(unsigned int word) {
    return (float)(word >> 8) * 1.1920928955078125e-7f - 1.0f;
}

extern "C" __global__
void secant_cuda_constant_optimizer_reduce_f32(
    const float* __restrict__ sse,
    size_t sse_leading_dimension,
    size_t num_asts,
    size_t num_settings,
    float* __restrict__ current_constants,
    size_t constants_leading_dimension,
    size_t num_constants,
    unsigned long long ast_index_base,
    unsigned long long seed,
    unsigned long long generation,
    unsigned long long iteration,
    float perturbation_scale,
    float* __restrict__ best_sse,
    unsigned int* __restrict__ best_settings
) {
    __shared__ float partial_sse[256];
    __shared__ unsigned int partial_setting[256];
    const size_t ast = blockIdx.x;
    float local_sse = __int_as_float(0x7f800000);
    unsigned int local_setting = 0xffffffffu;

    if (ast >= num_asts) {
        return;
    }
    for (size_t setting = threadIdx.x; setting < num_settings; setting += blockDim.x) {
        float value = sse[ast * sse_leading_dimension + setting];

        if (!(value >= 0.0f)) {
            value = __int_as_float(0x7f800000);
        }
        if (value < local_sse || (value == local_sse && setting < local_setting)) {
            local_sse = value;
            local_setting = (unsigned int)setting;
        }
    }
    partial_sse[threadIdx.x] = local_sse;
    partial_setting[threadIdx.x] = local_setting;
    __syncthreads();

    for (unsigned int offset = blockDim.x / 2u; offset != 0u; offset >>= 1u) {
        if (threadIdx.x < offset) {
            const float other_sse = partial_sse[threadIdx.x + offset];
            const unsigned int other_setting = partial_setting[threadIdx.x + offset];

            if (other_sse < partial_sse[threadIdx.x] ||
                (other_sse == partial_sse[threadIdx.x] && other_setting < partial_setting[threadIdx.x])) {
                partial_sse[threadIdx.x] = other_sse;
                partial_setting[threadIdx.x] = other_setting;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0u) {
        const unsigned int setting = partial_setting[0];
        float* const constants = current_constants + ast * constants_leading_dimension;

        if (best_sse != nullptr) {
            best_sse[ast] = partial_sse[0];
        }
        if (best_settings != nullptr) {
            best_settings[ast] = setting;
        }
        if (setting != 0u) {
            for (size_t group = 0u; group < (num_constants + 3u) / 4u; ++group) {
                const uint4 random = secant_optimizer_random4(
                    setting, ast_index_base + ast, seed, generation, iteration, (unsigned int)group);
                const unsigned int words[4] = {random.x, random.y, random.z, random.w};

                #pragma unroll
                for (unsigned int lane = 0u; lane < 4u; ++lane) {
                    const size_t constant_idx = group * 4u + lane;

                    if (constant_idx < num_constants) {
                        constants[constant_idx] += perturbation_scale * secant_optimizer_delta(words[lane]);
                    }
                }
            }
        }
    }
}
