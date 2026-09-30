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
#ifndef SECANT_SR_LM_INTERNAL_H
#define SECANT_SR_LM_INTERNAL_H
#include "internal.h"
#include "secant_sr_lm.h"
#include "lm_layout.h"
typedef struct SRLMBatch {
  SRConfig config;
  SRLMOptions options;
  uint8_t operators[32];
  SRCandidate *models;
  size_t *generations, *first, *length;
  SRLMProgram *programs;
  SRLMState *states;
  uint8_t *storage;
  const uint8_t **asts;
  float *jitter;
  size_t count, program_count, state_count;
  SRLMStats stats;
} SRLMBatch;
SecantResult sr_lm_batch_create(const SRConfig *, const SRLMOptions *, SRLMBatch *);
void sr_lm_batch_free(SRLMBatch *);
SecantResult sr_lm_batch_prepare(SRLMBatch *, const SRModel *, size_t,
    const float *, size_t, uint64_t);
SecantResult sr_lm_batch_finish(SRLMBatch *, SecantAstProgramSet *);
#endif
