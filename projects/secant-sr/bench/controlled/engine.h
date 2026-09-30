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
#ifndef BENCH_ENGINE_H
#define BENCH_ENGINE_H
#include "secant_sr_cuda.h"
SecantResult bench_cuda_create(const SRConfig *,const float *,const float *,size_t,const float *,const SRCudaOptions *,SRCudaScorer *,SRCudaStats *);
SecantResult bench_cuda_score(SRCudaScorer,const SecantAstProgramSet *,SRScore *,size_t,SRCudaStats *);
SecantResult bench_cuda_score_banks(SRCudaScorer,const SecantAstProgramSet *,const float *,size_t,size_t,SRScore *,size_t,SRCudaStats *);
SecantResult bench_cuda_destroy(SRCudaScorer);
/* Diagnostic oracle only: resolved expressions, static materialization kernel,
 * no selectors, no coefficient tables, no GPU squared-error reduction. */
void bench_materialize(size_t,const float *,size_t,const uint8_t *const *,size_t,float *);
#endif
