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
#ifndef SECANT_SR_ARCHIVE_H
#define SECANT_SR_ARCHIVE_H
#include "secant_sr.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Passive, bounded archive of winning bindings for final coefficient fitting.
 * It never affects GP selection, mutation, or random state. Diversity means
 * distinct ordered operators, input columns, fixed literals and parameter
 * sharing; adjustable values and parameter names are excluded from the key.
 * This is not algebraic equivalence or behavioral diversity. */
typedef struct SRArchiveImpl *SRArchive;
typedef struct SRArchiveStats {
  uint64_t visited, eligible, retained, duplicates, score_pruned;
} SRArchiveStats;
SecantResult secant_sr_archive_create(const SRConfig *, size_t capacity, SRArchive *);
void secant_sr_archive_destroy(SRArchive);
/* Copies a finite model's winning binding. Models with no active adjustable
 * parameters are ignored. Same-key entries keep the lower training SSE; a full
 * archive evicts its highest SSE. Allocations occur only at create. */
SecantResult secant_sr_archive_offer(SRArchive, const SRModel *, const float *banks, size_t elements);
size_t secant_sr_archive_count(SRArchive);
/* Borrowed model with one choice per leaf and configuration zero. Coefficients
 * are stored fitted centers; their canonical IDs preserve parameter sharing.
 * Valid until the next offer/destroy. Order is ascending training SSE. */
SecantResult secant_sr_archive_model(SRArchive, size_t index, SRModel *);
SRArchiveStats secant_sr_archive_stats(SRArchive);
#ifdef __cplusplus
}
#endif
#endif
