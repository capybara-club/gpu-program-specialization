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
#include "secant_sindy.h"

const char* secant_sindy_result_to_string(SecantSindyResult result) {
    static const char* const names[SECANT_SINDY_RESULT_COUNT] = {
        "SECANT_SINDY_SUCCESS",
        "SECANT_SINDY_ERROR_INVALID_VALUE",
        "SECANT_SINDY_ERROR_UNSUPPORTED_VERSION",
        "SECANT_SINDY_ERROR_UNSUPPORTED_SHAPE",
        "SECANT_SINDY_ERROR_OVERFLOW",
        "SECANT_SINDY_ERROR_INSUFFICIENT_BUFFER",
        "SECANT_SINDY_ERROR_FORMAT"
    };

    return result >= 0 && result < SECANT_SINDY_RESULT_COUNT ? names[result] : "SECANT_SINDY_ERROR_UNKNOWN";
}
