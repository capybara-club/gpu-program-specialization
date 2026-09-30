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
#include "secant.h"

const char*
secant_result_to_string(SecantResult result) {
    static const char* const names[SECANT_RESULT_COUNT] = {
        "SECANT_SUCCESS",
        "SECANT_ERROR_INVALID_VALUE",
        "SECANT_ERROR_OVERFLOW",
        "SECANT_ERROR_INSUFFICIENT_BUFFER",
        "SECANT_ERROR_FORMAT",
        "SECANT_ERROR_ALLOCATION_FAILED",
        "SECANT_ERROR_COMPILE_FAILED",
        "SECANT_ERROR_PARSE_FAILED",
        "SECANT_ERROR_SKELETON_NOT_FOUND",
        "SECANT_ERROR_BAD_PROGRAM",
        "SECANT_ERROR_STACK_OVERFLOW",
        "SECANT_ERROR_STACK_UNDERFLOW",
        "SECANT_ERROR_TOO_MANY_ARGS",
        "SECANT_ERROR_UNSUPPORTED_OP",
        "SECANT_ERROR_ROUTINE_IDX_OUT_OF_BOUNDS",
        "SECANT_ERROR_ROUTINE_ARG_IDX_OUT_OF_BOUNDS",
        "SECANT_ERROR_ROUTINE_DEPTH_EXCEEDED",
        "SECANT_ERROR_INSUFFICIENT_PATCH_SPACE",
        "SECANT_ERROR_REGISTER_OVERFLOW",
        "SECANT_ERROR_UNSUPPORTED_ARCHITECTURE",
        "SECANT_ERROR_INVALID_STATE",
        "SECANT_ERROR_UNSUPPORTED_SHAPE",
        "SECANT_ERROR_UNEXPECTED_INSTRUCTION",
        "SECANT_ERROR_BACKEND_UNAVAILABLE",
        "SECANT_ERROR_BAD_BINARY",
        "SECANT_ERROR_BAD_PATCH_SITE",
        "SECANT_ERROR_FUNCTION_NOT_FOUND",
        "SECANT_ERROR_PATCH_SITE_NOT_FOUND",
        "SECANT_ERROR_REGISTER_COUNT_NOT_FOUND",
        "SECANT_ERROR_THREAD_FAILED",
        "SECANT_ERROR_DRIVER_FAILED",
        "SECANT_ERROR_EAGER_LOADING_REQUIRED",
        "SECANT_ERROR_UNSUPPORTED_VERSION",
        "SECANT_ERROR_COMPLETION_UNKNOWN"
    };

    return result >= SECANT_SUCCESS && result < SECANT_RESULT_COUNT
        ? names[result]
        : "SECANT_ERROR_UNKNOWN";
}
