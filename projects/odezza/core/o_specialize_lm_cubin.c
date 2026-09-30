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
#include "o_odezza_internal.h"
#include "o_sha256.h"
#include <string.h>
#define TRY O_RETURN_IF_ERROR
static void put32(unsigned char *p, uint32_t v) {
    unsigned i;
    for (i = 0; i < 4; ++i)
        p[i] = (unsigned char)(v >> (8 * i));
}
static OdezzaResult instruction(unsigned char *data, size_t size, size_t at, OSassEncoding enc) {
    OSassInstruction words;
    if (enc.status)
        return enc.status;
    TRY(o_check_range(size, at, 16));
    words.word0 = enc.word0;
    words.word1 = enc.word1;
    memcpy(data + at, &words, 16);
    return ODEZZA_SUCCESS;
}
OdezzaResult o_lm_specialize(const void *original, size_t size, const OLmInspection *p,
                             const OdezzaAstProgram *rhs, uint32_t bits, OLmExpressions *expressions,
                             OSassInstruction *code, void *output, uint32_t *registers) {
    uint32_t k, high;
    size_t at, i;
    unsigned char hash[32];
    OSha256 sha;
    if (!p || !original || !output || !expressions || !code || !registers || original == output ||
        size != p->cubin_size)
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    TRY(o_sha256_init(&sha));
    TRY(o_sha256_update(&sha, original, size));
    TRY(o_sha256_final(&sha, hash));
    if (memcmp(hash, p->cubin_id, 32))
        return ODEZZA_ERROR_FORMAT;
    TRY(o_lm_expressions(rhs, &p->shape, bits, expressions));
    memcpy(output, original, size);
    high = p->registers;
    for (k = 0; k < p->site_count; ++k) {
        const OLmSite *s = &p->sites[k];
        OdezzaScoringCubinInspection shared;
        size_t used, prefix = s->preserves_predicate ? 1u : 0u;
        uint32_t need, roots[24];
        memset(&shared, 0, sizeof(shared));
        shared.architecture = p->architecture;
        /* Preserve the predicate in a private register: PTXAS may reuse its
         * original save register for a later output within the same site. */
        shared.register_count = p->registers + (uint32_t)prefix;
        shared.state_capacity = p->shape.state_count;
        shared.constant_capacity = p->shape.parameter_count;
        shared.input_count = s->input_count;
        shared.output_count = s->output_count;
        shared.input_registers = s->inputs;
        shared.output_registers = s->outputs;
        shared.final_output_registers = s->outputs;
        shared.available_registers = s->available;
        shared.available_register_count = s->available_count;
        shared.permutation_register = s->permutation;
        shared.predicate_register = s->predicate;
        shared.toggle_test_instruction = s->toggle;
        for (i = 0; i < s->output_count; ++i)
            roots[i] = expressions->roots[s->output_indices[i]];
        if (p->shape.site_patch_capacity <= 2u * prefix + 1u)
            return ODEZZA_ERROR_SPECIALIZATION_CAPACITY;
        TRY(o_sass_compile_group(&shared, expressions, roots, s->output_count, bits, code,
                                 p->shape.site_patch_capacity - 2u * prefix - 1u, &used, &need));
        if (need > high)
            high = need;
        if (high > 253u)
            return ODEZZA_ERROR_REGISTER_PRESSURE;
        for (at = s->scaffold; at < s->continuation; at += 16)
            TRY(instruction(output, size, at, o_sass_nop(0)));
        for (i = 0; i < s->input_count; ++i)
            TRY(instruction(output, size, s->input_offsets[i],
                            o_sass_mov(s->inputs[i], s->sources[i], s->waits[i])));
        TRY(instruction(output, size, s->entry, o_sass_branch((s->patch - s->entry) / 16, p->architecture)));
        if (prefix) {
            OSassInstruction save = s->predicate_save;
            save.word0 = (save.word0 & ~(UINT64_C(255) << 16)) | ((uint64_t)p->registers << 16);
            save.word1 = (save.word1 & ~(UINT64_C(15) << 40)) | (UINT64_C(12) << 40);
            memcpy((unsigned char *)output + s->patch, &save, 16);
        }
        for (i = 0; i < used; ++i) {
            TRY(o_check_range(size, s->patch + (prefix + i) * 16, 16));
            memcpy((unsigned char *)output + s->patch + (prefix + i) * 16, &code[i], 16);
        }
        used += prefix;
        if (prefix) {
            OSassInstruction restore = s->predicate_restore;
            restore.word0 = (restore.word0 & ~(UINT64_C(255) << 24)) | ((uint64_t)p->registers << 24);
            restore.word1 = (restore.word1 & ~(UINT64_C(15) << 40)) | (UINT64_C(12) << 40);
            memcpy((unsigned char *)output + s->patch + used * 16, &restore, 16);
            ++used;
        }
        TRY(instruction(output, size, s->patch + used * 16,
                        o_sass_branch((s->continuation - s->patch) / 16 - used, p->architecture)));
    }
    *registers = high > p->registers ? high + 2u : p->registers;
    for (i = 0; i < p->register_offset_count; ++i) {
        TRY(o_check_range(size, p->register_offsets[i], 4));
        put32((unsigned char *)output + p->register_offsets[i], *registers);
    }
    for (i = 0; i < p->header_offset_count; ++i) {
        TRY(o_check_range(size, p->header_offsets[i], 1));
        ((unsigned char *)output)[p->header_offsets[i]] = (unsigned char)*registers;
    }
    return ODEZZA_SUCCESS;
}
