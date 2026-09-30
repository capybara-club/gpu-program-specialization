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
#include "internal.h"

const char*
secant_sr_result_to_string(SecantSRResult result) {
    static const char* const names[] = {
        "SECANT_SR_SUCCESS",
        "SECANT_SR_ERROR_INVALID_VALUE",
        "SECANT_SR_ERROR_OVERFLOW",
        "SECANT_SR_ERROR_INSUFFICIENT_BUFFER",
        "SECANT_SR_ERROR_BAD_PROGRAM",
        "SECANT_SR_ERROR_ARENA_EXHAUSTED",
        "SECANT_SR_ERROR_POPULATION_FULL",
        "SECANT_SR_ERROR_NOT_SCORED",
        "SECANT_SR_ERROR_SECANT",
        "SECANT_SR_ERROR_SEARCH_STALLED"
    };

    return result >= SECANT_SR_SUCCESS && result < SECANT_SR_RESULT_NUM_ENUMS
        ? names[result]
        : "SECANT_SR_RESULT_UNKNOWN";
}

int
secant_sr_checked_add(size_t left, size_t right, size_t* result_ret) {
    if (left > SIZE_MAX - right) {
        return 0;
    }
    *result_ret = left + right;
    return 1;
}

int
secant_sr_checked_mul(size_t left, size_t right, size_t* result_ret) {
    if (left != 0u && right > SIZE_MAX / left) {
        return 0;
    }
    *result_ret = left * right;
    return 1;
}

static int
secant_sr_align_up(size_t value, size_t alignment, size_t* result_ret) {
    const size_t remainder = value % alignment;

    return remainder == 0u
        ? (*result_ret = value, 1)
        : secant_sr_checked_add(value, alignment - remainder, result_ret);
}

int
secant_sr_layout_add(size_t count, size_t element_size, size_t* offset) {
    size_t aligned_offset;
    size_t bytes;

    if (!secant_sr_align_up(*offset, sizeof(SecantSRAlignment), &aligned_offset) ||
        !secant_sr_checked_mul(count, element_size, &bytes) ||
        !secant_sr_checked_add(aligned_offset, bytes, offset)) {
        return 0;
    }
    return 1;
}

void*
secant_sr_layout_take(void* storage, size_t count, size_t element_size, size_t* offset) {
    size_t aligned_offset;
    size_t bytes;
    unsigned char* result;

    if (!secant_sr_align_up(*offset, sizeof(SecantSRAlignment), &aligned_offset) ||
        !secant_sr_checked_mul(count, element_size, &bytes) ||
        !secant_sr_checked_add(aligned_offset, bytes, offset)) {
        return NULL;
    }
    result = (unsigned char*)storage + aligned_offset;
    memset(result, 0, bytes);
    return result;
}

uint64_t
secant_sr_rng_next(SecantSRSearch search) {
    uint64_t value = search->rng_state;

    value ^= value >> 12u;
    value ^= value << 25u;
    value ^= value >> 27u;
    search->rng_state = value;
    return value * UINT64_C(2685821657736338717);
}

size_t
secant_sr_rng_bounded(SecantSRSearch search, size_t upper_bound) {
    const uint64_t threshold = (uint64_t)(-upper_bound) % upper_bound;
    uint64_t value;

    do {
        value = secant_sr_rng_next(search);
    } while (value < threshold);
    return (size_t)(value % upper_bound);
}

double
secant_sr_rng_unit(SecantSRSearch search) {
    return (double)(secant_sr_rng_next(search) >> 11u) * (1.0 / 9007199254740992.0);
}
