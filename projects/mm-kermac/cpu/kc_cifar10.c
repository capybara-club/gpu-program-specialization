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
#include <cifar10.h>

KERMAC_PUBLIC_DEF
KermacResult
kermac_cifar10_dims(
    int64_t* num_train,
    int64_t* num_test,
    int64_t* num_dims,
    int64_t* num_labels
) {
    int64_t num_rows, num_cols;
    Cifar10CResult cifar_result;
    cifar_result =
        cifar10_c_dims(
            num_train,
            num_test,
            &num_rows,
            &num_cols,
            num_labels
        );
    if (cifar_result != CIFAR10_C_RESULT_SUCCESS) {
        _KERMAC_ERROR( KERMAC_ERROR_THIRDPARTY );
    }
    *num_dims = num_rows * num_cols * CIFAR10_C_NUM_CHANNELS;

    return KERMAC_SUCCESS;
}

KERMAC_PUBLIC_DEF
KermacResult
kermac_cifar10_load(
    KermacTensor x_train,
    KermacTensor y_train,
    KermacTensor x_test,
    KermacTensor y_test
) {
    float *x_train_ptr, *y_train_ptr, *x_test_ptr, *y_test_ptr;

    _KERMAC_CHECK_RET( kermac_memory_pointer(x_train.memory, (void**)&x_train_ptr) );
    _KERMAC_CHECK_RET( kermac_memory_pointer(y_train.memory, (void**)&y_train_ptr) );
    _KERMAC_CHECK_RET( kermac_memory_pointer(x_test.memory, (void**)&x_test_ptr) );
    _KERMAC_CHECK_RET( kermac_memory_pointer(y_test.memory, (void**)&y_test_ptr) );
    Cifar10CResult cifar_result;

    int64_t num_train, num_test, num_rows, num_cols, num_labels;
    cifar_result =
        cifar10_c_dims(
            &num_train,
            &num_test,
            &num_rows,
            &num_cols,
            &num_labels
        );
    if (cifar_result != CIFAR10_C_RESULT_SUCCESS) {
        _KERMAC_ERROR( KERMAC_ERROR_THIRDPARTY );
    }

    num_train = x_train.extent[0];
    num_test = x_test.extent[0];

    int64_t plane = num_rows * num_cols;
    int64_t stride_n = x_train.stride[0];
    int64_t stride_w = x_train.stride[1];
    int64_t stride_h = x_train.stride[1] * num_cols;
    int64_t stride_c = x_train.stride[1] * plane;

    cifar_result =
        cifar10_c_load_images_f32(
            true,
            num_train,
            x_train_ptr,
            stride_n,
            stride_c,
            stride_h,
            stride_w
        );
    if (cifar_result != CIFAR10_C_RESULT_SUCCESS) {
        _KERMAC_ERROR( KERMAC_ERROR_THIRDPARTY );
    }

    cifar_result =
        cifar10_c_load_labels_onehot_f32(
            true,
            num_train,
            y_train_ptr,
            y_train.stride[0],
            y_train.stride[1]
        );
    if (cifar_result != CIFAR10_C_RESULT_SUCCESS) {
        _KERMAC_ERROR( KERMAC_ERROR_THIRDPARTY );
    }

    stride_n = x_test.stride[0];
    stride_w = x_test.stride[1];
    stride_h = x_test.stride[1] * num_cols;
    stride_c = x_test.stride[1] * plane;

    cifar_result =
        cifar10_c_load_images_f32(
            false,
            num_test,
            x_test_ptr,
            stride_n,
            stride_c,
            stride_h,
            stride_w
        );
    if (cifar_result != CIFAR10_C_RESULT_SUCCESS) {
        _KERMAC_ERROR( KERMAC_ERROR_THIRDPARTY );
    }

    cifar_result =
        cifar10_c_load_labels_onehot_f32(
            false,
            num_test,
            y_test_ptr,
            y_test.stride[0],
            y_test.stride[1]
        );
    if (cifar_result != CIFAR10_C_RESULT_SUCCESS) {
        _KERMAC_ERROR( KERMAC_ERROR_THIRDPARTY );
    }

    return KERMAC_SUCCESS;
}

KERMAC_PUBLIC_DEF
KermacResult
kermac_cifar10_load_u32(
    KermacTensor x_train,
    KermacTensor y_train,
    KermacTensor x_test,
    KermacTensor y_test
) {
    uint32_t *x_train_ptr, *x_test_ptr;
    uint8_t *y_train_ptr, *y_test_ptr;

    _KERMAC_CHECK_RET( kermac_memory_pointer(x_train.memory, (void**)&x_train_ptr) );
    _KERMAC_CHECK_RET( kermac_memory_pointer(y_train.memory, (void**)&y_train_ptr) );
    _KERMAC_CHECK_RET( kermac_memory_pointer(x_test.memory, (void**)&x_test_ptr) );
    _KERMAC_CHECK_RET( kermac_memory_pointer(y_test.memory, (void**)&y_test_ptr) );
    Cifar10CResult cifar_result;

    int64_t num_train, num_test, num_rows, num_cols, num_labels;
    cifar_result =
        cifar10_c_dims(
            &num_train,
            &num_test,
            &num_rows,
            &num_cols,
            &num_labels
        );
    if (cifar_result != CIFAR10_C_RESULT_SUCCESS) {
        _KERMAC_ERROR( KERMAC_ERROR_THIRDPARTY );
    }

    num_train = x_train.extent[0];
    num_test = x_test.extent[0];

    cifar_result =
        cifar10_c_load_images_u32(
            true,
            num_train,
            x_train_ptr,
            x_train.stride[0],
            x_train.stride[1] * num_cols,
            x_train.stride[1]
        );
    if (cifar_result != CIFAR10_C_RESULT_SUCCESS) {
        _KERMAC_ERROR( KERMAC_ERROR_THIRDPARTY );
    }

    cifar_result =
        cifar10_c_load_labels_u8(
            true,
            num_train,
            y_train_ptr,
            y_train.stride[0]
        );
    if (cifar_result != CIFAR10_C_RESULT_SUCCESS) {
        _KERMAC_ERROR( KERMAC_ERROR_THIRDPARTY );
    }

    cifar_result =
        cifar10_c_load_images_u32(
            false,
            num_test,
            x_test_ptr,
            x_test.stride[0],
            x_test.stride[1] * num_cols,
            x_test.stride[1]
        );
    if (cifar_result != CIFAR10_C_RESULT_SUCCESS) {
        _KERMAC_ERROR( KERMAC_ERROR_THIRDPARTY );
    }

    cifar_result =
        cifar10_c_load_labels_u8(
            false,
            num_test,
            y_test_ptr,
            y_test.stride[0]
        );
    if (cifar_result != CIFAR10_C_RESULT_SUCCESS) {
        _KERMAC_ERROR( KERMAC_ERROR_THIRDPARTY );
    }

    return KERMAC_SUCCESS;
}
