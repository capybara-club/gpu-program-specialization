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
#include "odr_grammar_internal.h"

uint64_t odr_mix(uint64_t x) {
    x += UINT64_C(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

uint64_t odr_random_below(uint64_t seed, uint64_t *counter, uint64_t bound) {
    uint64_t x, threshold = (UINT64_C(0) - bound) % bound;
    do { x = odr_mix(seed + (*counter)++); } while (x < threshold);
    return x % bound;
}

static size_t cell(uint64_t *map, size_t capacity, uint64_t key) {
    size_t i = (size_t)(odr_mix(key) % capacity);
    while (map[2*i] != UINT64_MAX && map[2*i] != key) i = (i+1) % capacity;
    return i;
}

/* Partial Fisher-Yates; map uses 2*(2*count+1) uint64 words. Old keys need
 * not be deleted because later draws never revisit an earlier prefix index. */
void odr_sample_ranks(uint64_t total, size_t count, uint64_t seed,
                      uint64_t *ranks, uint64_t *map) {
    size_t i, capacity = 2*count+1;
    uint64_t counter = 0;
    for (i=0; i<capacity; ++i) map[2*i] = UINT64_MAX;
    for (i=0; i<count; ++i) {
        uint64_t j = (uint64_t)i + odr_random_below(seed, &counter, total-i);
        size_t cj = cell(map, capacity, j), ci = cell(map, capacity, i);
        uint64_t chosen = map[2*cj] == UINT64_MAX ? j : map[2*cj+1];
        uint64_t replacement = map[2*ci] == UINT64_MAX ? (uint64_t)i : map[2*ci+1];
        map[2*cj] = j;
        map[2*cj+1] = replacement;
        ranks[i] = chosen;
    }
}

const char *odr_grammar_sampling_profile(void) {
    return "odezza.c99-grammar.splitmix64-fisher-yates.v1";
}
