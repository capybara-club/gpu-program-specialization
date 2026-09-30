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
#ifndef SECANT_SR_CUBIN_CACHE_H_INCLUDED
#define SECANT_SR_CUBIN_CACHE_H_INCLUDED

#include "secant.h"

#include <stddef.h>
#include <stdint.h>

typedef struct SecantSRCubinCache {
    void* database;
} SecantSRCubinCache;

int secant_sr_cubin_cache_open(const char* path, SecantSRCubinCache* cache);

int secant_sr_cubin_cache_lookup(
    SecantSRCubinCache* cache,
    const SecantCubinRecipeHeader* recipe,
    int compute_capability_major,
    int compute_capability_minor,
    int nvrtc_major,
    int nvrtc_minor,
    int ptxas_opt_level,
    int nvrtc_no_cache,
    unsigned char** cubin_ret,
    size_t* cubin_size_ret,
    double* compile_seconds_ret,
    int* hit_ret
);

int secant_sr_cubin_cache_store(
    SecantSRCubinCache* cache,
    const SecantCubinRecipeHeader* recipe,
    int compute_capability_major,
    int compute_capability_minor,
    int nvrtc_major,
    int nvrtc_minor,
    int ptxas_opt_level,
    int nvrtc_no_cache,
    const unsigned char* cubin,
    size_t cubin_size,
    double compile_seconds
);

int secant_sr_cubin_cache_artifact_lookup(
    SecantSRCubinCache* cache,
    const char* artifact_name,
    uint32_t artifact_version,
    int compute_capability_major,
    int compute_capability_minor,
    int nvrtc_major,
    int nvrtc_minor,
    int ptxas_opt_level,
    int nvrtc_no_cache,
    unsigned char** cubin_ret,
    size_t* cubin_size_ret,
    double* compile_seconds_ret,
    int* hit_ret
);

int secant_sr_cubin_cache_artifact_store(
    SecantSRCubinCache* cache,
    const char* artifact_name,
    uint32_t artifact_version,
    int compute_capability_major,
    int compute_capability_minor,
    int nvrtc_major,
    int nvrtc_minor,
    int ptxas_opt_level,
    int nvrtc_no_cache,
    const unsigned char* cubin,
    size_t cubin_size,
    double compile_seconds
);

void secant_sr_cubin_cache_close(SecantSRCubinCache* cache);

#endif /* SECANT_SR_CUBIN_CACHE_H_INCLUDED */
