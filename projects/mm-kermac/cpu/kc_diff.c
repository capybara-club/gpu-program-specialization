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

KERMAC_PUBLIC_DEF
KermacResult
kermac_max_diff_cpu(
    KermacStackAllocator* hsa,
    KermacTensor a,
    KermacTensor b,
    float* max_diff_out,
    CUstream stream
) {
    if (a.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (b.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_DEVICE) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }
    if (a.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (b.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (a.num_modes != b.num_modes) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    for (int i = 0; i < a.num_modes; i++) {
        if (a.extent[i] != b.extent[i]) {
            _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
        }
    }
    bool hsa_is_dry = hsa->is_dry;
    bool a_is_dry = a.memory.stack_allocator->is_dry;
    bool b_is_dry = b.memory.stack_allocator->is_dry;

    bool is_dry;
    if (hsa_is_dry && a_is_dry && b_is_dry) {
        is_dry = true;
    } else if (!hsa_is_dry && !a_is_dry && !b_is_dry) {
        is_dry = false;
    } else if (!hsa_is_dry && a_is_dry && b_is_dry) {
        is_dry = true;
    } else {
        _KERMAC_ERROR( KERMAC_ERROR_INCONSISTENT_ALLOCATION );
    }

    KermacTensor h_a = {0};
    KermacTensor h_b = {0};

    _KERMAC_CHECK_RET( kermac_tensor_create(&h_a, KERMAC_DATA_TYPE_FLOAT, a.extent, hsa) );
    _KERMAC_CHECK_RET( kermac_tensor_create(&h_b, KERMAC_DATA_TYPE_FLOAT, b.extent, hsa) );

    _KERMAC_CHECK_RET( kermac_tensor_copy(a, h_a, stream) );
    _KERMAC_CHECK_RET( kermac_tensor_copy(b, h_b, stream) );

    if (!is_dry) {
        float *h_a_ptr;
        float *h_b_ptr;

        _KERMAC_CHECK_RET( kermac_memory_pointer(h_a.memory, (void**)&h_a_ptr) );
        _KERMAC_CHECK_RET( kermac_memory_pointer(h_b.memory, (void**)&h_b_ptr) );

        double max_abs = 0.0;
        for (int64_t e3 = 0; e3 < _KERMAC_MAX(1, h_a.extent[3]); e3++) {
            for (int64_t e2 = 0; e2 < _KERMAC_MAX(1, h_a.extent[2]); e2++) {
                for (int64_t e1 = 0; e1 < _KERMAC_MAX(1, h_a.extent[1]); e1++) {
                    for (int64_t e0 = 0; e0 < _KERMAC_MAX(1, h_a.extent[0]); e0++) {
                        int64_t h_a_idx = h_a.stride[0] * e0 + h_a.stride[1] * e1 + h_a.stride[2] * e2 + h_a.stride[3] * e3;
                        int64_t h_b_idx = h_b.stride[0] * e0 + h_b.stride[1] * e1 + h_b.stride[2] * e2 + h_b.stride[3] * e3;
                        double v_a = (double)h_a_ptr[h_a_idx];
                        double v_b = (double)h_b_ptr[h_b_idx];
                        double diff = v_a - v_b;
                        diff = fabs(diff);
                        if (diff > max_abs) {
                            max_abs = diff;
                        }
                    }
                }
            }
        }
        *max_diff_out = max_abs;
    }

    _KERMAC_CHECK_RET( kermac_tensor_destroy(h_b) );
    _KERMAC_CHECK_RET( kermac_tensor_destroy(h_a) );

    return KERMAC_SUCCESS;
}