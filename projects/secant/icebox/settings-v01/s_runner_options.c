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
#include "s_runner_internal.h"

#include <stdlib.h>
#include <string.h>

SecantResult
_secant_runner_options_copy(
    const char* const* options,
    size_t num_options,
    _SecantRunnerOptions* copied_ret
) {
    size_t storage_size = 0u;
    size_t option_idx;
    size_t offset = 0u;

    if (copied_ret == NULL || (num_options != 0u && options == NULL)) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    memset(copied_ret, 0, sizeof(*copied_ret));
    for (option_idx = 0u; option_idx < num_options; ++option_idx) {
        size_t option_size;

        if (options[option_idx] == NULL) {
            return SECANT_ERROR_INVALID_VALUE;
        }
        option_size = strlen(options[option_idx]) + 1u;
        if (option_size > SIZE_MAX - storage_size) {
            return SECANT_ERROR_OVERFLOW;
        }
        storage_size += option_size;
    }
    if (num_options > SIZE_MAX / sizeof(*copied_ret->values)) {
        return SECANT_ERROR_OVERFLOW;
    }
    if (num_options != 0u) {
        copied_ret->values = (const char**)malloc(num_options * sizeof(*copied_ret->values));
        copied_ret->storage = (char*)malloc(storage_size);
        if (copied_ret->values == NULL || copied_ret->storage == NULL) {
            _secant_runner_options_destroy(copied_ret);
            return SECANT_ERROR_ALLOCATION_FAILED;
        }
    }
    for (option_idx = 0u; option_idx < num_options; ++option_idx) {
        const size_t option_size = strlen(options[option_idx]) + 1u;

        copied_ret->values[option_idx] = copied_ret->storage + offset;
        memcpy(copied_ret->storage + offset, options[option_idx], option_size);
        offset += option_size;
    }
    copied_ret->count = num_options;
    return SECANT_SUCCESS;
}

void
_secant_runner_options_destroy(
    _SecantRunnerOptions* options
) {
    if (options == NULL) {
        return;
    }
    free(options->storage);
    free(options->values);
    memset(options, 0, sizeof(*options));
}
