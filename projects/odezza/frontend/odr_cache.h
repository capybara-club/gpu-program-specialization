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
#ifndef ODR_CACHE_H
#define ODR_CACHE_H
#include <stddef.h>
#include <stdint.h>
typedef struct OdrCache OdrCache;
/* Host-local, disposable compiler artifacts only. No CUDA or request policy.
 * One connection per caller; a held key lock must be released before close.
 * open honors ODEZZA_TEMPLATE_CACHE_BYTES (default 256 MiB). */
int odr_cache_open(const char *directory,OdrCache **out,char *error,size_t error_size);
int odr_cache_lock(OdrCache *cache,const char *key);
void odr_cache_unlock(OdrCache *cache);
/* get: 0=hit, 1=miss (including checksum invalidation), -1=error.
 * Returned bytes are owned by the caller and must be freed. */
int odr_cache_get(OdrCache *cache,const char *key,void **data,size_t *size);
int odr_cache_remove(OdrCache *cache,const char *key);
/* put: 0=stored, 1=larger than budget (not stored), -1=error. */
int odr_cache_put(OdrCache *cache,const char *key,const void *data,size_t size,double compile_seconds);
uint64_t odr_cache_evictions(const OdrCache *cache);
uint64_t odr_cache_invalidations(const OdrCache *cache);
const char *odr_cache_error(const OdrCache *cache);
void odr_cache_close(OdrCache *cache);
#endif
