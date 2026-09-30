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
#ifndef O_SPECIALIZE_SCORING_CUBIN_H
#define O_SPECIALIZE_SCORING_CUBIN_H

#include "o_odezza_internal.h"

/*
 * Trusted pipeline path. The caller must have copied the exact validated,
 * prespecialized template into cubin immediately before this call. Candidate
 * AST bytecode remains fully validated; redundant whole-CUBIN identity checks
 * and clearing of the already-clean patch arena are omitted.
 */
OdezzaResult o_specialize_scoring_cubin_systems_fresh(
    void *cubin,
    size_t cubin_size,
    const OdezzaScoringCubinInspection *inspection,
    const OdezzaScoringPrespecialization *prespecialization,
    uint32_t active_toggle_count,
    const OdezzaScoringSystem *systems,
    size_t system_count,
    void *workspace,
    size_t workspace_size,
    OdezzaScoringSpecializationReport *report_ret
);

#endif
