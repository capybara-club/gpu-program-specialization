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
#ifndef O_SHA256_H
#define O_SHA256_H

#include "odezza.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef struct OSha256 {
    uint32_t state[8];
    uint64_t total_bytes;
    unsigned char block[64];
    size_t block_bytes;
} OSha256;

static inline OdezzaResult o_sha256_transform(OSha256 *sha, const unsigned char block[64]) {
    static const uint32_t constants[64] = {0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
                                           0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
                                           0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
                                           0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
                                           0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
                                           0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
                                           0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
                                           0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
    uint32_t words[64];
    uint32_t a;
    uint32_t b;
    uint32_t c;
    uint32_t d;
    uint32_t e;
    uint32_t f;
    uint32_t g;
    uint32_t h;
    size_t index;

    if (sha == NULL || block == NULL) {
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    }
    for (index = 0u; index < 16u; ++index) {
        const unsigned char *word = block + index * 4u;
        words[index] = ((uint32_t)word[0] << 24u) | ((uint32_t)word[1] << 16u) | ((uint32_t)word[2] << 8u) | (uint32_t)word[3];
    }
    for (index = 16u; index < 64u; ++index) {
        uint32_t value0 = words[index - 15u];
        uint32_t value1 = words[index - 2u];
        uint32_t s0 = ((value0 >> 7u) | (value0 << 25u)) ^ ((value0 >> 18u) | (value0 << 14u)) ^ (value0 >> 3u);
        uint32_t s1 = ((value1 >> 17u) | (value1 << 15u)) ^ ((value1 >> 19u) | (value1 << 13u)) ^ (value1 >> 10u);
        words[index] = words[index - 16u] + s0 + words[index - 7u] + s1;
    }

    a = sha->state[0];
    b = sha->state[1];
    c = sha->state[2];
    d = sha->state[3];
    e = sha->state[4];
    f = sha->state[5];
    g = sha->state[6];
    h = sha->state[7];
    for (index = 0u; index < 64u; ++index) {
        uint32_t sum1 = ((e >> 6u) | (e << 26u)) ^ ((e >> 11u) | (e << 21u)) ^ ((e >> 25u) | (e << 7u));
        uint32_t choice = (e & f) ^ ((~e) & g);
        uint32_t temporary1 = h + sum1 + choice + constants[index] + words[index];
        uint32_t sum0 = ((a >> 2u) | (a << 30u)) ^ ((a >> 13u) | (a << 19u)) ^ ((a >> 22u) | (a << 10u));
        uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temporary2 = sum0 + majority;
        h = g;
        g = f;
        f = e;
        e = d + temporary1;
        d = c;
        c = b;
        b = a;
        a = temporary1 + temporary2;
    }
    sha->state[0] += a;
    sha->state[1] += b;
    sha->state[2] += c;
    sha->state[3] += d;
    sha->state[4] += e;
    sha->state[5] += f;
    sha->state[6] += g;
    sha->state[7] += h;
    return ODEZZA_SUCCESS;
}

static inline OdezzaResult o_sha256_init(OSha256 *sha) {
    static const uint32_t initial[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au, 0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    if (sha == NULL) {
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    }
    memcpy(sha->state, initial, sizeof(initial));
    sha->total_bytes = 0u;
    sha->block_bytes = 0u;
    return ODEZZA_SUCCESS;
}

static inline OdezzaResult o_sha256_update(OSha256 *sha, const void *data, size_t data_size) {
    const unsigned char *input = (const unsigned char *)data;
    size_t available;

    if (sha == NULL || (data == NULL && data_size != 0u)) {
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    }
    if (data_size > UINT64_MAX - sha->total_bytes) {
        return ODEZZA_ERROR_OVERFLOW;
    }
    sha->total_bytes += (uint64_t)data_size;
    while (data_size != 0u) {
        available = 64u - sha->block_bytes;
        if (available > data_size) {
            available = data_size;
        }
        memcpy(sha->block + sha->block_bytes, input, available);
        sha->block_bytes += available;
        input += available;
        data_size -= available;
        if (sha->block_bytes == 64u) {
            OdezzaResult result = o_sha256_transform(sha, sha->block);
            if (result != ODEZZA_SUCCESS) {
                return result;
            }
            sha->block_bytes = 0u;
        }
    }
    return ODEZZA_SUCCESS;
}

static inline OdezzaResult o_sha256_final(OSha256 *sha, unsigned char digest[32]) {
    uint64_t bit_count;
    size_t index;
    OdezzaResult result;

    if (sha == NULL || digest == NULL) {
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    }
    if (sha->total_bytes > UINT64_MAX / 8u) {
        return ODEZZA_ERROR_OVERFLOW;
    }
    bit_count = sha->total_bytes * 8u;
    sha->block[sha->block_bytes++] = 0x80u;
    if (sha->block_bytes > 56u) {
        memset(sha->block + sha->block_bytes, 0, 64u - sha->block_bytes);
        result = o_sha256_transform(sha, sha->block);
        if (result != ODEZZA_SUCCESS) {
            return result;
        }
        sha->block_bytes = 0u;
    }
    memset(sha->block + sha->block_bytes, 0, 56u - sha->block_bytes);
    for (index = 0u; index < 8u; ++index) {
        sha->block[63u - index] = (unsigned char)(bit_count >> (index * 8u));
    }
    result = o_sha256_transform(sha, sha->block);
    if (result != ODEZZA_SUCCESS) {
        return result;
    }
    for (index = 0u; index < 8u; ++index) {
        digest[index * 4u] = (unsigned char)(sha->state[index] >> 24u);
        digest[index * 4u + 1u] = (unsigned char)(sha->state[index] >> 16u);
        digest[index * 4u + 2u] = (unsigned char)(sha->state[index] >> 8u);
        digest[index * 4u + 3u] = (unsigned char)sha->state[index];
    }
    return ODEZZA_SUCCESS;
}

#endif
