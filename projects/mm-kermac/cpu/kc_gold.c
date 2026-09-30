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
#include <kermac_cpu.h>
#include <k_internal.h>

#include <math.h>

typedef enum {
    KERMAC_GOLD_OP_GEMM,
    KERMAC_GOLD_OP_L1,
    KERMAC_GOLD_OP_L2
} KermacGoldOp;

static
inline
void
compute_gemm_gold_batch(
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t L,
    const float* a,
    int64_t lda,
    int64_t batch_stride_a,
    const float* b,
    int64_t ldb,
    int64_t batch_stride_b,
    float* c,
    int64_t ldc,
    int64_t batch_stride_c
) {
    for (int64_t l = 0; l < L; ++l) {
        const float* a_l = a + l * batch_stride_a;
        const float* b_l = b + l * batch_stride_b;
        float* c_l = c + l * batch_stride_c;

        for (int64_t m = 0; m < M; ++m) {
            for (int64_t n = 0; n < N; ++n) {
                float accum = 0.0f;
                for (int64_t k = 0; k < K; ++k) {
                    int64_t a_idx = k * lda + m;
                    int64_t b_idx = k * ldb + n;
                    accum += a_l[a_idx] * b_l[b_idx];
                }
                int64_t c_idx = n * ldc + m;
                c_l[c_idx] = accum;
            }
        }
    }
}

static
inline
void
compute_l1_gold_batch(
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t L,
    const float* a,
    int64_t lda,
    int64_t batch_stride_a,
    const float* b,
    int64_t ldb,
    int64_t batch_stride_b,
    float* c,
    int64_t ldc,
    int64_t batch_stride_c
) {
    for (int64_t l = 0; l < L; ++l) {
        const float* a_l = a + l * batch_stride_a;
        const float* b_l = b + l * batch_stride_b;
        float* c_l = c + l * batch_stride_c;

        for (int64_t m = 0; m < M; ++m) {
            for (int64_t n = 0; n < N; ++n) {
                float accum = 0.0f;
                for (int64_t k = 0; k < K; ++k) {
                    int64_t a_idx = k * lda + m;
                    int64_t b_idx = k * ldb + n;
                    float diff = a_l[a_idx] - b_l[b_idx];
                    accum += fabsf(diff);
                }
                int64_t c_idx = n * ldc + m;
                c_l[c_idx] = accum;
            }
        }
    }
}

static
inline
void
compute_l2_gold_batch(
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t L,
    const float* a,
    int64_t lda,
    int64_t batch_stride_a,
    const float* b,
    int64_t ldb,
    int64_t batch_stride_b,
    float* c,
    int64_t ldc,
    int64_t batch_stride_c
) {
    for (int64_t l = 0; l < L; ++l) {
        const float* a_l = a + l * batch_stride_a;
        const float* b_l = b + l * batch_stride_b;
        float* c_l = c + l * batch_stride_c;

        for (int64_t m = 0; m < M; ++m) {
            for (int64_t n = 0; n < N; ++n) {
                float accum = 0.0f;
                for (int64_t k = 0; k < K; ++k) {
                    int64_t a_idx = k * lda + m;
                    int64_t b_idx = k * ldb + n;
                    float diff = a_l[a_idx] - b_l[b_idx];
                    accum += diff * diff;
                }
                int64_t c_idx = n * ldc + m;
                c_l[c_idx] = accum;
            }
        }
    }
}

static
KermacResult
kermac_gold_cpu_impl(
    KermacStackAllocator* hsa,
    KermacTensor a,
    KermacTensor b,
    KermacTensor c,
    CUstream stream,
    KermacGoldOp op
) {
    KermacMemorySpace a_space = a.memory.stack_allocator->memory_space;
    KermacMemorySpace b_space = b.memory.stack_allocator->memory_space;
    KermacMemorySpace c_space = c.memory.stack_allocator->memory_space;

    if (a_space != KERMAC_MEMORY_SPACE_DEVICE && a_space != KERMAC_MEMORY_SPACE_HOST) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (b_space != KERMAC_MEMORY_SPACE_DEVICE && b_space != KERMAC_MEMORY_SPACE_HOST) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (c_space != KERMAC_MEMORY_SPACE_DEVICE && c_space != KERMAC_MEMORY_SPACE_HOST) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    if (a.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (b.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (c.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }

    if (a.num_modes != 2 && a.num_modes != 3) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (b.num_modes != 2 && b.num_modes != 3) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (c.num_modes != 2 && c.num_modes != 3) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    int64_t M = c.extent[0];
    int64_t N = c.extent[1];
    int64_t K = a.extent[1];
    int64_t L = c.num_modes == 2 ? 1 : c.extent[2];
    int64_t L_a = a.num_modes == 2 ? 1 : a.extent[2];
    int64_t L_b = b.num_modes == 2 ? 1 : b.extent[2];

    if (M != a.extent[0] || K != a.extent[1] || L != L_a) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (N != b.extent[0] || K != b.extent[1] || L != L_b) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    bool hsa_is_dry = hsa->is_dry;
    bool a_is_dry = a.memory.stack_allocator->is_dry;
    bool b_is_dry = b.memory.stack_allocator->is_dry;
    bool c_is_dry = c.memory.stack_allocator->is_dry;

    bool is_dry;
    if (hsa_is_dry && a_is_dry && b_is_dry && c_is_dry) {
        is_dry = true;
    } else if (!hsa_is_dry && !a_is_dry && !b_is_dry && !c_is_dry) {
        is_dry = false;
    } else if (!hsa_is_dry && a_is_dry && b_is_dry && c_is_dry) {
        is_dry = true;
    } else {
        _KERMAC_ERROR( KERMAC_ERROR_INCONSISTENT_ALLOCATION );
    }

    bool a_on_device = a_space == KERMAC_MEMORY_SPACE_DEVICE;
    bool b_on_device = b_space == KERMAC_MEMORY_SPACE_DEVICE;
    bool c_on_device = c_space == KERMAC_MEMORY_SPACE_DEVICE;

    KermacTensor h_a = {0};
    KermacTensor h_b = {0};
    KermacTensor h_c = {0};

    bool h_a_created = false;
    bool h_b_created = false;
    bool h_c_created = false;

    if (a_on_device) {
        _KERMAC_CHECK_RET( kermac_tensor_create(&h_a, KERMAC_DATA_TYPE_FLOAT, a.extent, hsa) );
        h_a_created = true;
        _KERMAC_CHECK_RET( kermac_tensor_copy(a, h_a, stream) );
    } else {
        h_a = a;
    }

    if (b_on_device) {
        _KERMAC_CHECK_RET( kermac_tensor_create(&h_b, KERMAC_DATA_TYPE_FLOAT, b.extent, hsa) );
        h_b_created = true;
        _KERMAC_CHECK_RET( kermac_tensor_copy(b, h_b, stream) );
    } else {
        h_b = b;
    }

    if (c_on_device) {
        _KERMAC_CHECK_RET( kermac_tensor_create(&h_c, KERMAC_DATA_TYPE_FLOAT, c.extent, hsa) );
        h_c_created = true;
        _KERMAC_CHECK_RET( kermac_tensor_copy(c, h_c, stream) );
    } else {
        h_c = c;
    }

    if (!is_dry) {
        float* h_a_ptr = NULL;
        float* h_b_ptr = NULL;
        float* h_c_ptr = NULL;

        _KERMAC_CHECK_RET( kermac_memory_pointer(h_a.memory, (void**)&h_a_ptr) );
        _KERMAC_CHECK_RET( kermac_memory_pointer(h_b.memory, (void**)&h_b_ptr) );
        _KERMAC_CHECK_RET( kermac_memory_pointer(h_c.memory, (void**)&h_c_ptr) );

        int64_t lda = h_a.stride[1];
        int64_t ldb = h_b.stride[1];
        int64_t ldc = h_c.stride[1];
        int64_t batch_stride_a = h_a.num_modes == 2 ? 0 : h_a.stride[2];
        int64_t batch_stride_b = h_b.num_modes == 2 ? 0 : h_b.stride[2];
        int64_t batch_stride_c = h_c.num_modes == 2 ? 0 : h_c.stride[2];

        switch (op) {
            case KERMAC_GOLD_OP_GEMM:
                compute_gemm_gold_batch(
                    M, N, K, L,
                    h_a_ptr, lda, batch_stride_a,
                    h_b_ptr, ldb, batch_stride_b,
                    h_c_ptr, ldc, batch_stride_c
                );
                break;
            case KERMAC_GOLD_OP_L1:
                compute_l1_gold_batch(
                    M, N, K, L,
                    h_a_ptr, lda, batch_stride_a,
                    h_b_ptr, ldb, batch_stride_b,
                    h_c_ptr, ldc, batch_stride_c
                );
                break;
            case KERMAC_GOLD_OP_L2:
                compute_l2_gold_batch(
                    M, N, K, L,
                    h_a_ptr, lda, batch_stride_a,
                    h_b_ptr, ldb, batch_stride_b,
                    h_c_ptr, ldc, batch_stride_c
                );
                break;
            default:
                _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
        }
    }

    if (a_on_device) {
        _KERMAC_CHECK_RET( kermac_tensor_copy(h_a, a, stream) );
    }
    if (b_on_device) {
        _KERMAC_CHECK_RET( kermac_tensor_copy(h_b, b, stream) );
    }
    if (c_on_device) {
        _KERMAC_CHECK_RET( kermac_tensor_copy(h_c, c, stream) );
    }

    if (h_c_created) {
        _KERMAC_CHECK_RET( kermac_tensor_destroy(h_c) );
    }
    if (h_b_created) {
        _KERMAC_CHECK_RET( kermac_tensor_destroy(h_b) );
    }
    if (h_a_created) {
        _KERMAC_CHECK_RET( kermac_tensor_destroy(h_a) );
    }

    return KERMAC_SUCCESS;
}

KERMAC_PUBLIC_DEF
KermacResult
kermac_gemm_gold_cpu(
    KermacStackAllocator* hsa,
    KermacTensor a,
    KermacTensor b,
    KermacTensor c,
    CUstream stream
) {
    return kermac_gold_cpu_impl(hsa, a, b, c, stream, KERMAC_GOLD_OP_GEMM);
}

KERMAC_PUBLIC_DEF
KermacResult
kermac_l1_gold_cpu(
    KermacStackAllocator* hsa,
    KermacTensor a,
    KermacTensor b,
    KermacTensor c,
    CUstream stream
) {
    return kermac_gold_cpu_impl(hsa, a, b, c, stream, KERMAC_GOLD_OP_L1);
}

KERMAC_PUBLIC_DEF
KermacResult
kermac_l2_gold_cpu(
    KermacStackAllocator* hsa,
    KermacTensor a,
    KermacTensor b,
    KermacTensor c,
    CUstream stream
) {
    return kermac_gold_cpu_impl(hsa, a, b, c, stream, KERMAC_GOLD_OP_L2);
}
