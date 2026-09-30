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
#ifndef SECANT_RUNNER_FAULT_HOOKS_H
#define SECANT_RUNNER_FAULT_HOOKS_H
#include <pthread.h>
#include <stdlib.h>
void *test_malloc(size_t);
void *test_calloc(size_t, size_t);
void test_free(void *);
int test_pthread_create(pthread_t *, const pthread_attr_t *, void *(*)(void *), void *);
int test_pthread_join(pthread_t, void **);
int test_pthread_mutex_init(pthread_mutex_t *, const pthread_mutexattr_t *);
int test_pthread_cond_init(pthread_cond_t *, const pthread_condattr_t *);
int test_pthread_mutex_destroy(pthread_mutex_t *);
int test_pthread_cond_destroy(pthread_cond_t *);
#define malloc test_malloc
#define calloc test_calloc
#define free test_free
#define pthread_create test_pthread_create
#define pthread_join test_pthread_join
#define pthread_mutex_init test_pthread_mutex_init
#define pthread_cond_init test_pthread_cond_init
#define pthread_mutex_destroy test_pthread_mutex_destroy
#define pthread_cond_destroy test_pthread_cond_destroy
#endif
