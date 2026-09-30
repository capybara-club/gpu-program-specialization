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
/* Explicit benchmark-only opt-in for CUDA 13.1 / glibc 2.43 / GCC 15.
 * -U_GNU_SOURCE avoids conflicting new rsqrt declarations. libstdc++ still
 * expects these two existing libc entry points. Match installed pthread.h;
 * do not modify vendor headers or link this adapter into the Odezza service.
 * Use a compatible toolchain for a normal production nvcc build. */
#ifndef ODEZZA_TOPK_GLIBC_COMPAT_H
#define ODEZZA_TOPK_GLIBC_COMPAT_H
#include <pthread.h>
#if !defined(__GLIBC__) || !defined(__TIMESIZE) || __TIMESIZE != 64 || defined(__USE_GNU)
#error This benchmark compatibility opt-in requires 64-bit glibc with GNU feature declarations disabled
#endif
#ifdef __cplusplus
extern "C" {
#endif
extern int pthread_cond_clockwait(pthread_cond_t *,pthread_mutex_t *,clockid_t,const struct timespec *);
extern int pthread_mutex_clocklock(pthread_mutex_t *,clockid_t,const struct timespec *) __THROWNL;
#ifdef __cplusplus
}
#endif
#endif
