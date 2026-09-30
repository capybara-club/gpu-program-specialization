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
    KERMAC_P_NORM_L1,
    KERMAC_P_NORM_L2,
    KERMAC_P_NORM_P
} KermacPNormType;

static
inline
void
compute_p_norm_kernel_batch(
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
    int64_t batch_stride_c,
    KermacPNormType norm_type,
    float p_power
) {
    const float p_recip = norm_type == KERMAC_P_NORM_P ? (1.0f / p_power) : 0.0f;

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
                    float diff = b_l[b_idx] - a_l[a_idx];
                    if (norm_type == KERMAC_P_NORM_L1) {
                        diff = fabsf(diff);
                    } else if (norm_type == KERMAC_P_NORM_L2) {
                        diff = diff * diff;
                    } else {
                        diff = powf(fabsf(diff), p_power);
                    }
                    accum += diff;
                }
                if (norm_type == KERMAC_P_NORM_L2) {
                    accum = sqrtf(accum);
                } else if (norm_type == KERMAC_P_NORM_P) {
                    accum = powf(accum, p_recip);
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
compute_p_norm_kernel_gradient(
    int64_t M,
    int64_t N,
    int64_t D,
    int64_t C,
    const float* kernel_matrix,
    int64_t ld_kernel,
    const float* data_n,
    int64_t ld_data_n,
    const float* solution,
    int64_t ld_solution,
    const float* data_m,
    int64_t ld_data_m,
    float* gradient,
    int64_t ld_gradient_d,
    int64_t ld_gradient_c,
    KermacPNormType norm_type,
    float p_power
) {
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t d = 0; d < D; ++d) {
            for (int64_t c = 0; c < C; ++c) {
                float accum = 0.0f;
                for (int64_t n = 0; n < N; ++n) {
                    float mn = kernel_matrix[n * ld_kernel + m];
                    float nd = data_n[d * ld_data_n + n];
                    float md = data_m[d * ld_data_m + m];
                    float nc = solution[c * ld_solution + n];
                    float diff = nd - md;
                    float sign = copysignf(1.0f, diff);
                    if (norm_type == KERMAC_P_NORM_L1) {
                        diff = sign;
                    } else if (norm_type == KERMAC_P_NORM_L2) {
                        diff = diff;
                    } else {
                        diff = powf(fabsf(diff), p_power) * sign;
                    }
                    accum += diff * mn * nc;
                }
                gradient[c * ld_gradient_c + d * ld_gradient_d + m] = accum;
            }
        }
    }
}

static
KermacResult
kermac_p_norm_kernel_cpu_impl(
    KermacStackAllocator* hsa,
    float p_power,
    KermacTensor a,
    KermacTensor b,
    KermacTensor c,
    CUstream stream
) {
    if (a.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (b.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (c.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
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
    if (a.num_modes != 3 || b.num_modes != 3 || c.num_modes != 3) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    int64_t M = c.extent[0];
    int64_t N = c.extent[1];
    int64_t K = a.extent[1];
    int64_t L = c.extent[2];

    if (a.extent[0] != M || a.extent[1] != K || a.extent[2] != L) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (b.extent[0] != N || b.extent[1] != K || b.extent[2] != L) {
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

    KermacTensor h_a = {0};
    KermacTensor h_b = {0};
    KermacTensor h_c = {0};

    _KERMAC_CHECK_RET( kermac_tensor_create(&h_a, KERMAC_DATA_TYPE_FLOAT, a.extent, hsa) );
    _KERMAC_CHECK_RET( kermac_tensor_create(&h_b, KERMAC_DATA_TYPE_FLOAT, b.extent, hsa) );
    _KERMAC_CHECK_RET( kermac_tensor_create(&h_c, KERMAC_DATA_TYPE_FLOAT, c.extent, hsa) );

    _KERMAC_CHECK_RET( kermac_tensor_copy(a, h_a, stream) );
    _KERMAC_CHECK_RET( kermac_tensor_copy(b, h_b, stream) );

    if (!is_dry) {
        float* h_a_ptr;
        float* h_b_ptr;
        float* h_c_ptr;

        _KERMAC_CHECK_RET( kermac_memory_pointer(h_a.memory, (void**)&h_a_ptr) );
        _KERMAC_CHECK_RET( kermac_memory_pointer(h_b.memory, (void**)&h_b_ptr) );
        _KERMAC_CHECK_RET( kermac_memory_pointer(h_c.memory, (void**)&h_c_ptr) );

        KermacPNormType norm_type = KERMAC_P_NORM_P;
        if (p_power == 1.0f) {
            norm_type = KERMAC_P_NORM_L1;
        } else if (p_power == 2.0f) {
            norm_type = KERMAC_P_NORM_L2;
        }

        compute_p_norm_kernel_batch(
            M, N, K, L,
            h_a_ptr, h_a.stride[1], h_a.stride[2],
            h_b_ptr, h_b.stride[1], h_b.stride[2],
            h_c_ptr, h_c.stride[1], h_c.stride[2],
            norm_type,
            p_power
        );
    }

    _KERMAC_CHECK_RET( kermac_tensor_copy(h_c, c, stream) );

    _KERMAC_CHECK_RET( kermac_tensor_destroy(h_c) );
    _KERMAC_CHECK_RET( kermac_tensor_destroy(h_b) );
    _KERMAC_CHECK_RET( kermac_tensor_destroy(h_a) );

    return KERMAC_SUCCESS;
}

static
KermacResult
kermac_p_norm_kernel_gradient_cpu_impl(
    KermacStackAllocator* hsa,
    float p_power,
    KermacTensor kernel_matrix,
    KermacTensor data_n,
    KermacTensor solution,
    KermacTensor data_m,
    KermacTensor gradient,
    CUstream stream
) {
    if (kernel_matrix.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (data_n.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (solution.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (data_m.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (gradient.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }

    if (kernel_matrix.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (data_n.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (solution.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (data_m.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (gradient.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }

    if (kernel_matrix.num_modes != 2) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (data_n.num_modes != 2) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (solution.num_modes != 2) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (data_m.num_modes != 2) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (gradient.num_modes != 3) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    int64_t M = data_m.extent[0];
    int64_t N = data_n.extent[0];
    int64_t D = data_m.extent[1];
    int64_t C = solution.extent[1];

    if (kernel_matrix.extent[0] != M || kernel_matrix.extent[1] != N) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (data_n.extent[1] != D) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (solution.extent[0] != N) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (gradient.extent[0] != M || gradient.extent[1] != D || gradient.extent[2] != C) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    bool hsa_is_dry = hsa->is_dry;
    bool kernel_is_dry = kernel_matrix.memory.stack_allocator->is_dry;
    bool data_n_is_dry = data_n.memory.stack_allocator->is_dry;
    bool solution_is_dry = solution.memory.stack_allocator->is_dry;
    bool data_m_is_dry = data_m.memory.stack_allocator->is_dry;
    bool gradient_is_dry = gradient.memory.stack_allocator->is_dry;

    bool is_dry;
    if (hsa_is_dry && kernel_is_dry && data_n_is_dry && solution_is_dry && data_m_is_dry && gradient_is_dry) {
        is_dry = true;
    } else if (!hsa_is_dry && !kernel_is_dry && !data_n_is_dry && !solution_is_dry && !data_m_is_dry && !gradient_is_dry) {
        is_dry = false;
    } else if (!hsa_is_dry && kernel_is_dry && data_n_is_dry && solution_is_dry && data_m_is_dry && gradient_is_dry) {
        is_dry = true;
    } else {
        _KERMAC_ERROR( KERMAC_ERROR_INCONSISTENT_ALLOCATION );
    }

    KermacTensor h_kernel = {0};
    KermacTensor h_data_n = {0};
    KermacTensor h_solution = {0};
    KermacTensor h_data_m = {0};
    KermacTensor h_gradient = {0};

    _KERMAC_CHECK_RET( kermac_tensor_create(&h_kernel, KERMAC_DATA_TYPE_FLOAT, kernel_matrix.extent, hsa) );
    _KERMAC_CHECK_RET( kermac_tensor_create(&h_data_n, KERMAC_DATA_TYPE_FLOAT, data_n.extent, hsa) );
    _KERMAC_CHECK_RET( kermac_tensor_create(&h_solution, KERMAC_DATA_TYPE_FLOAT, solution.extent, hsa) );
    _KERMAC_CHECK_RET( kermac_tensor_create(&h_data_m, KERMAC_DATA_TYPE_FLOAT, data_m.extent, hsa) );
    _KERMAC_CHECK_RET( kermac_tensor_create(&h_gradient, KERMAC_DATA_TYPE_FLOAT, gradient.extent, hsa) );

    _KERMAC_CHECK_RET( kermac_tensor_copy(kernel_matrix, h_kernel, stream) );
    _KERMAC_CHECK_RET( kermac_tensor_copy(data_n, h_data_n, stream) );
    _KERMAC_CHECK_RET( kermac_tensor_copy(solution, h_solution, stream) );
    _KERMAC_CHECK_RET( kermac_tensor_copy(data_m, h_data_m, stream) );

    if (!is_dry) {
        float* h_kernel_ptr;
        float* h_data_n_ptr;
        float* h_solution_ptr;
        float* h_data_m_ptr;
        float* h_gradient_ptr;

        _KERMAC_CHECK_RET( kermac_memory_pointer(h_kernel.memory, (void**)&h_kernel_ptr) );
        _KERMAC_CHECK_RET( kermac_memory_pointer(h_data_n.memory, (void**)&h_data_n_ptr) );
        _KERMAC_CHECK_RET( kermac_memory_pointer(h_solution.memory, (void**)&h_solution_ptr) );
        _KERMAC_CHECK_RET( kermac_memory_pointer(h_data_m.memory, (void**)&h_data_m_ptr) );
        _KERMAC_CHECK_RET( kermac_memory_pointer(h_gradient.memory, (void**)&h_gradient_ptr) );

        KermacPNormType norm_type = KERMAC_P_NORM_P;
        float p_power_grad = p_power - 1.0f;
        if (p_power == 1.0f) {
            norm_type = KERMAC_P_NORM_L1;
            p_power_grad = 0.0f;
        } else if (p_power == 2.0f) {
            norm_type = KERMAC_P_NORM_L2;
            p_power_grad = 1.0f;
        }

        compute_p_norm_kernel_gradient(
            M, N, D, C,
            h_kernel_ptr, h_kernel.stride[1],
            h_data_n_ptr, h_data_n.stride[1],
            h_solution_ptr, h_solution.stride[1],
            h_data_m_ptr, h_data_m.stride[1],
            h_gradient_ptr, h_gradient.stride[1], h_gradient.stride[2],
            norm_type,
            p_power_grad
        );
    }

    _KERMAC_CHECK_RET( kermac_tensor_copy(h_gradient, gradient, stream) );

    _KERMAC_CHECK_RET( kermac_tensor_destroy(h_gradient) );
    _KERMAC_CHECK_RET( kermac_tensor_destroy(h_data_m) );
    _KERMAC_CHECK_RET( kermac_tensor_destroy(h_solution) );
    _KERMAC_CHECK_RET( kermac_tensor_destroy(h_data_n) );
    _KERMAC_CHECK_RET( kermac_tensor_destroy(h_kernel) );

    return KERMAC_SUCCESS;
}

KERMAC_PUBLIC_DEF
KermacResult
kermac_p_norm_kernel_cpu(
    KermacStackAllocator* hsa,
    float p_power,
    KermacTensor a,
    KermacTensor b,
    KermacTensor c,
    CUstream stream
) {
    return kermac_p_norm_kernel_cpu_impl(hsa, p_power, a, b, c, stream);
}

KERMAC_PUBLIC_DEF
KermacResult
kermac_p_norm_kernel_gradient_cpu(
    KermacStackAllocator* hsa,
    float p_power,
    KermacTensor kernel_matrix,
    KermacTensor data_n,
    KermacTensor solution,
    KermacTensor data_m,
    KermacTensor gradient,
    CUstream stream
) {
    return kermac_p_norm_kernel_gradient_cpu_impl(
        hsa,
        p_power,
        kernel_matrix,
        data_n,
        solution,
        data_m,
        gradient,
        stream
    );
}
