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
#ifndef SECANT_HSACO_GENERATE_INTERNAL_H_INCLUDED
#define SECANT_HSACO_GENERATE_INTERNAL_H_INCLUDED

#include "secant_hsaco.h"

#include <stddef.h>

#define _SECANT_HSACO_MAX_WAVES 16u
#define _SECANT_HSACO_MAX_REGISTERS 256u
#define _SECANT_HSACO_FIRST_MARKER_BITS 0x7fc0ffeeu

#define _SECANT_HSACO_ERROR_RET(ans) do { SecantResult _secant_hsaco_result = (ans); return _secant_hsaco_result; } while (0)
#define _SECANT_HSACO_CHECK_RET(ans) do { SecantResult _secant_hsaco_check_result = (ans); if (_secant_hsaco_check_result != SECANT_SUCCESS) { _SECANT_HSACO_ERROR_RET(_secant_hsaco_check_result); } } while (0)

typedef enum _SecantHsacoShape {
    _SECANT_HSACO_SHAPE_MATERIALIZE = 1,
    _SECANT_HSACO_SHAPE_SSE = 2,
    _SECANT_HSACO_SHAPE_DYNAMIC_CONSTANT_SSE = 3
} _SecantHsacoShape;

int _secant_hsaco_checked_mul(
    size_t lhs,
    size_t rhs,
    size_t* result_ret
);

SecantResult _secant_hsaco_write(
    char* buffer,
    size_t buffer_size,
    size_t* offset,
    const char* format,
    ...
);

SecantResult _secant_hsaco_write_finish(
    char* buffer,
    size_t buffer_size,
    size_t offset,
    size_t* size_ret
);

SecantResult _secant_hsaco_emit_skeleton(
    _SecantHsacoShape shape,
    size_t kernel_idx,
    size_t asts_per_kernel,
    size_t num_inputs,
    size_t num_targets,
    size_t patch_capacity_instructions,
    char* buffer,
    size_t buffer_size,
    size_t* offset
);

SecantResult _secant_hsaco_validate_generate_args(
    _SecantHsacoShape shape,
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    size_t num_targets,
    size_t patch_capacity_instructions,
    size_t* marker_count_ret
);

#endif /* SECANT_HSACO_GENERATE_INTERNAL_H_INCLUDED */
