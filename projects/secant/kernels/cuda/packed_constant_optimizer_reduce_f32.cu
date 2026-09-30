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
static __forceinline__ __device__ uint4 secant_packed_optimizer_philox_round(uint4 c, uint2 k) {
    const unsigned int hi0 = __umulhi(0xd2511f53u, c.x);
    const unsigned int hi1 = __umulhi(0xcd9e8d57u, c.z);
    const unsigned int lo0 = 0xd2511f53u * c.x;
    const unsigned int lo1 = 0xcd9e8d57u * c.z;
    return make_uint4(hi1 ^ c.y ^ k.x, lo1, hi0 ^ c.w ^ k.y, lo0);
}

static __forceinline__ __device__ uint4 secant_packed_optimizer_philox4x32_10(uint4 c, uint2 k) {
    #pragma unroll
    for (int round = 0; round < 10; ++round) {
        c = secant_packed_optimizer_philox_round(c, k);
        k.x += 0x9e3779b9u;
        k.y += 0xbb67ae85u;
    }
    return c;
}

static __forceinline__ __device__ uint4 secant_packed_optimizer_random4(
    unsigned long long setting,
    unsigned long long seed,
    unsigned long long generation,
    unsigned long long iteration,
    unsigned int group
) {
    const uint4 counter = make_uint4((unsigned int)setting, (unsigned int)(setting >> 32), 0u, 0u);
    const uint2 key = make_uint2(
        (unsigned int)seed ^ (unsigned int)generation * 0x9e3779b9u ^
            (unsigned int)(iteration >> 32) * 0x85ebca6bu ^ group * 0x27d4eb2du,
        (unsigned int)(seed >> 32) ^ (unsigned int)(generation >> 32) * 0xbb67ae85u ^
            (unsigned int)iteration * 0xc2b2ae35u ^ group * 0x165667b1u);
    return secant_packed_optimizer_philox4x32_10(counter, key);
}

static __forceinline__ __device__ float secant_packed_optimizer_delta(unsigned int word) {
    return (float)(word >> 8) * 1.1920928955078125e-7f - 1.0f;
}

extern "C" __global__
void secant_cuda_packed_constant_optimizer_reduce_f32(
    const float* __restrict__ sse,
    size_t sse_leading_dimension,
    size_t num_asts,
    size_t num_settings,
    size_t num_constants,
    unsigned long long seed,
    unsigned long long generation,
    unsigned long long iteration,
    unsigned int update_mode,
    size_t num_elites,
    float momentum,
    float scale_learning_rate,
    float scale_failure_decay,
    float minimum_scale,
    float maximum_scale,
    float* __restrict__ best,
    size_t best_leading_dimension
) {
    __shared__ float partial_sse[256];
    __shared__ unsigned int partial_setting[256];
    __shared__ float elite_sse[16];
    __shared__ unsigned int elite_setting[16];
    const size_t ast = blockIdx.x;
    const size_t selected_count = update_mode == 1u ? 1u : num_elites;

    if (ast >= num_asts) {
        return;
    }
    for (size_t elite_idx = 0u; elite_idx < selected_count; ++elite_idx) {
        float local_sse = __int_as_float(0x7f800000);
        unsigned int local_setting = 0xffffffffu;

        for (size_t setting = threadIdx.x; setting < num_settings; setting += blockDim.x) {
            float value = sse[ast * sse_leading_dimension + setting];

            if (!(value >= 0.0f)) {
                value = __int_as_float(0x7f800000);
            }
            const bool after_previous = elite_idx == 0u || value > elite_sse[elite_idx - 1u] ||
                (value == elite_sse[elite_idx - 1u] && setting > elite_setting[elite_idx - 1u]);
            if (after_previous &&
                (value < local_sse || (value == local_sse && setting < local_setting))) {
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
            elite_sse[elite_idx] = partial_sse[0];
            elite_setting[elite_idx] = partial_setting[0];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0u) {
        const unsigned int setting = elite_setting[0];
        float* const result = best + ast * best_leading_dimension;
        float* const centers = result + 1u;
        float* const scales = centers + num_constants;
        float* const velocities = scales + num_constants;
        float* const incumbent_constants = velocities + num_constants;
        const float candidate_sse = elite_sse[0];

        if (update_mode == 1u) {
            if (candidate_sse < result[0]) {
                result[0] = candidate_sse;
                for (size_t group = 0u; group < (num_constants + 3u) / 4u; ++group) {
                    const uint4 random = setting == 0u
                        ? make_uint4(0u, 0u, 0u, 0u)
                        : secant_packed_optimizer_random4(
                            setting, seed, generation, iteration, (unsigned int)group);
                    const unsigned int words[4] = {random.x, random.y, random.z, random.w};

                    #pragma unroll
                    for (unsigned int lane = 0u; lane < 4u; ++lane) {
                        const size_t constant_idx = group * 4u + lane;

                        if (constant_idx < num_constants) {
                            const float old_center = centers[constant_idx];
                            const float old_scale = scales[constant_idx];
                            const float jitter = setting == 0u
                                ? 0.0f
                                : secant_packed_optimizer_delta(words[lane]);
                            const float random_step = old_scale * jitter;
                            const float center = old_center + momentum * velocities[constant_idx] + random_step;
                            float target_scale = 2.0f * fabsf(random_step);
                            float scale;

                            if (target_scale < minimum_scale) {
                                target_scale = minimum_scale;
                            }
                            scale = old_scale + scale_learning_rate * (target_scale - old_scale);
                            if (scale < minimum_scale) {
                                scale = minimum_scale;
                            } else if (scale > maximum_scale) {
                                scale = maximum_scale;
                            }
                            centers[constant_idx] = center;
                            scales[constant_idx] = scale;
                            velocities[constant_idx] = center - old_center;
                            incumbent_constants[constant_idx] = center;
                        }
                    }
                }
            } else {
                for (size_t constant_idx = 0u; constant_idx < num_constants; ++constant_idx) {
                    float scale = scales[constant_idx] * scale_failure_decay;

                    if (scale < minimum_scale) {
                        scale = minimum_scale;
                    } else if (scale > maximum_scale) {
                        scale = maximum_scale;
                    }
                    scales[constant_idx] = scale;
                    velocities[constant_idx] *= momentum;
                }
            }
        } else {
            const bool improved = candidate_sse < result[0];

            for (size_t group = 0u; group < (num_constants + 3u) / 4u; ++group) {
                float jitter_sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                float jitter_squared_sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                float winner_jitter[4] = {0.0f, 0.0f, 0.0f, 0.0f};

                for (size_t elite_idx = 0u; elite_idx < selected_count; ++elite_idx) {
                    const unsigned int elite = elite_setting[elite_idx];
                    const uint4 random = elite == 0u
                        ? make_uint4(0u, 0u, 0u, 0u)
                        : secant_packed_optimizer_random4(
                            elite, seed, generation, iteration, (unsigned int)group);
                    const unsigned int words[4] = {random.x, random.y, random.z, random.w};

                    #pragma unroll
                    for (unsigned int lane = 0u; lane < 4u; ++lane) {
                        const float jitter = elite == 0u ? 0.0f : secant_packed_optimizer_delta(words[lane]);

                        if (elite_idx == 0u) {
                            winner_jitter[lane] = jitter;
                        }
                        jitter_sum[lane] += jitter;
                        jitter_squared_sum[lane] += jitter * jitter;
                    }
                }
                #pragma unroll
                for (unsigned int lane = 0u; lane < 4u; ++lane) {
                    const size_t constant_idx = group * 4u + lane;

                    if (constant_idx < num_constants) {
                        const float old_center = centers[constant_idx];
                        const float old_scale = scales[constant_idx];
                        const float mean = jitter_sum[lane] / (float)selected_count;
                        float variance = jitter_squared_sum[lane] / (float)selected_count - mean * mean;
                        float target_scale;
                        float scale;

                        if (variance < 0.0f) {
                            variance = 0.0f;
                        }
                        centers[constant_idx] = old_center + old_scale * mean;
                        velocities[constant_idx] = 0.0f;
                        target_scale = old_scale * sqrtf(3.0f * variance);
                        if (target_scale < 2.0f * fabsf(old_scale * winner_jitter[lane])) {
                            target_scale = 2.0f * fabsf(old_scale * winner_jitter[lane]);
                        }
                        if (target_scale < minimum_scale) {
                            target_scale = minimum_scale;
                        }
                        scale = old_scale + scale_learning_rate * (target_scale - old_scale);
                        if (scale < minimum_scale) {
                            scale = minimum_scale;
                        } else if (scale > maximum_scale) {
                            scale = maximum_scale;
                        }
                        scales[constant_idx] = scale;
                        if (improved) {
                            incumbent_constants[constant_idx] = old_center + old_scale * winner_jitter[lane];
                        }
                    }
                }
            }
            if (improved) {
                result[0] = candidate_sse;
            }
        }
    }
}
