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
#define BPT UINT64_C(0x000000040000795c)
static OdezzaResult words(const OElf *e, size_t at, uint64_t *a, uint64_t *b) {
    TRY(o_read_u64(e->data, e->size, at, a));
    return o_read_u64(e->data, e->size, at + 8, b);
}
static OdezzaResult inspect_site(const OElf *e, const OFunction *f, OLmSite *s, uint32_t k,
                                 uint32_t capacity) {
    size_t at, end = f->file_offset + f->size, first = end, last = 0, ofirst = end, olast = 0;
    uint32_t marker = 0x4fc40000u + 256u * k, i, j, salts = 0, toggles = 0, selects = 0, entries = 0;
    uint8_t seen_in[24] = {0}, seen_out[24] = {0}, salted = 255;
    uint64_t a, b;
    for (at = f->file_offset; at < end; at += 16) {
        uint32_t imm;
        TRY(words(e, at, &a, &b));
        if ((a & 65535u) != 0x7421u)
            continue;
        imm = (uint32_t)(a >> 32);
        if (imm >= marker && imm < marker + s->input_count) {
            i = imm - marker;
            if (seen_in[i]++)
                return ODEZZA_ERROR_FORMAT;
            s->input_offsets[i] = at;
            s->inputs[i] = (uint8_t)(a >> 16);
            s->sources[i] = (uint8_t)(a >> 24);
            s->waits[i] = (uint8_t)((b >> 52) & 63u);
            if (at < first)
                first = at;
            if (at > last)
                last = at;
        }
        if (imm >= marker + 128u && imm < marker + 128u + s->output_count) {
            i = imm - marker - 128u;
            if (seen_out[i]++)
                return ODEZZA_ERROR_FORMAT;
            s->outputs[i] = (uint8_t)(a >> 16);
            if (at < ofirst)
                ofirst = at;
            if (at > olast)
                olast = at;
        }
    }
    for (i = 0; i < s->input_count; ++i)
        if (!seen_in[i])
            return ODEZZA_ERROR_FORMAT;
    for (i = 0; i < s->output_count; ++i)
        if (!seen_out[i])
            return ODEZZA_ERROR_FORMAT;
    if (last >= ofirst)
        return ODEZZA_ERROR_FORMAT;
    for (at = olast + 16; at < end; at += 16) {
        TRY(words(e, at, &a, &b));
        if (a == BPT)
            break;
    }
    if (at >= end || (size_t)capacity > (end - at) / 16)
        return ODEZZA_ERROR_FORMAT;
    s->patch = at;
    s->continuation = at + (size_t)capacity * 16;
    for (; at < s->continuation; at += 16) {
        TRY(words(e, at, &a, &b));
        if (a != BPT)
            return ODEZZA_ERROR_FORMAT;
    }
    s->scaffold = SIZE_MAX;
    at = first - f->file_offset > (s->input_count + 4u) * 16u ? first - (s->input_count + 4u) * 16u
                                                              : f->file_offset;
    for (; at < first; at += 16) {
        TRY(words(e, at, &a, &b));
        if (a == BPT)
            s->scaffold = at;
    }
    if (s->scaffold == SIZE_MAX)
        return ODEZZA_ERROR_FORMAT;
    for (at = last + 16; at < ofirst; at += 16) {
        TRY(words(e, at, &a, &b));
        if (a == BPT) {
            s->entry = at;
            ++entries;
        }
    }
    if (entries != 1)
        return ODEZZA_ERROR_FORMAT;
    s->available_count = 0;
    /* A read/write operand's incoming register may also hold a CUDA value
     * that remains live after this site. Copying it into the destination does
     * not prove the source is dead. Only asm-local scratch can be recycled. */
    for (at = s->scaffold; at < s->patch; at += 16) {
        uint32_t op;
        TRY(words(e, at, &a, &b));
        op = (uint32_t)(a & 65535u);
        if ((op == 0x7810u || op == 0x7835u) && a >> 32 == k + 1u) {
            ++salts;
            salted = (uint8_t)(a >> 16);
            s->permutation = (uint8_t)(a >> 24);
        } else if (op == 0x7812u && (a & 0xffffffu) == 0xff7812u && a >> 32 == 1u) {
            ++toggles;
            s->predicate = (uint32_t)((b >> 17) & 7u);
            if (salted == 255 || (uint8_t)(a >> 24) != salted)
                return ODEZZA_ERROR_FORMAT;
            s->toggle.word0 = a;
            s->toggle.word1 = b;
        } else if (op == 0x7208u || ((op & 0xfffu) == 0x221u && op >> 12 < 7u)) {
            uint32_t pred = op == 0x7208u ? (uint32_t)((b >> 23) & 7u) : op >> 12;
            if (!toggles || pred != s->predicate)
                return ODEZZA_ERROR_FORMAT;
            ++selects;
            s->available[s->available_count++] = (uint8_t)(a >> 16);
        }
    }
    if (salts != 1 || toggles != 1 || selects != 1 || s->predicate >= 7u || s->permutation == 255)
        return ODEZZA_ERROR_FORMAT;
    /* PTXAS may spill a live CUDA predicate around the asm-local predicate.
     * Those P2R/ISETP instructions are inside the region we replace, even
     * though they were not in the inline PTX. Preserve the matched pair. */
    {
        uint32_t saves = 0, restores = 0;
        uint8_t saved = 255;
        for (at = s->entry + 16; at < s->patch; at += 16) {
            TRY(words(e, at, &a, &b));
            if ((a & 65535u) == 0x7803u) {
                if (saves++ || (uint8_t)(a >> 24) != 255 ||
                    (a >> 32) != (UINT64_C(1) << s->predicate) ||
                    (b & ((UINT64_C(1) << 40) - 1u)) != 0)
                    return ODEZZA_ERROR_FORMAT;
                saved = (uint8_t)(a >> 16);
                s->predicate_save.word0 = a; s->predicate_save.word1 = b;
            } else if ((a & 65535u) == 0x720cu) {
                uint64_t rest = b & ((UINT64_C(1) << 40) - 1u);
                if (!saves || restores++ || (uint8_t)(a >> 24) != saved ||
                    (a >> 32) != 255 || ((b >> 17) & 7u) != s->predicate ||
                    (rest & ~(UINT64_C(7) << 17)) != UINT64_C(0x3f05270))
                    return ODEZZA_ERROR_FORMAT;
                s->predicate_restore.word0 = a; s->predicate_restore.word1 = b;
            }
        }
        if (saves != restores) return ODEZZA_ERROR_FORMAT;
        s->preserves_predicate = saves;
        if (saves && saved == 255) return ODEZZA_ERROR_FORMAT;
    }
    {
        uint32_t moves = 0;
        uint8_t source = s->permutation;
        for (at = s->scaffold; at < s->patch; at += 16) {
            TRY(words(e, at, &a, &b));
            if ((a & 65535u) == 0x7202u && (uint8_t)(a >> 16) == s->permutation) {
                source = (uint8_t)(a >> 32);
                ++moves;
            }
        }
        if (moves > 1)
            return ODEZZA_ERROR_FORMAT;
        s->permutation = source;
    }
    for (i = 0; i < s->input_count; ++i) {
        if (s->inputs[i] == 255 || s->inputs[i] == s->permutation)
            return ODEZZA_ERROR_FORMAT;
        for (j = 0; j < i; ++j)
            if (s->inputs[i] == s->inputs[j])
                return ODEZZA_ERROR_FORMAT;
    }
    for (i = 0; i < s->output_count; ++i) {
        if (s->outputs[i] == 255)
            return ODEZZA_ERROR_FORMAT;
        for (j = 0; j < i; ++j)
            if (s->outputs[i] == s->outputs[j])
                return ODEZZA_ERROR_FORMAT;
        for (j = 0; j < s->input_count; ++j)
            if (s->outputs[i] == s->inputs[j])
                return ODEZZA_ERROR_FORMAT;
    }
    {
        size_t count = 0, raw = s->available_count;
        for (i = 0; i < raw; ++i) {
            uint8_t v = s->available[i];
            int reserved = v == 255 || v == s->permutation ||
                (s->preserves_predicate && v == (uint8_t)(s->predicate_save.word0 >> 16));
            for (j = 0; j < s->input_count; ++j)
                if (v == s->inputs[j])
                    reserved = 1;
            for (j = 0; j < s->output_count; ++j)
                if (v == s->outputs[j])
                    reserved = 1;
            for (j = 0; j < count; ++j)
                if (v == s->available[j])
                    reserved = 1;
            if (!reserved)
                s->available[count++] = v;
        }
        s->available_count = count;
    }
    return ODEZZA_SUCCESS;
}
OdezzaResult o_lm_inspect(const void *data, size_t size, const OdezzaLmShape *shape, OLmInspection *p) {
    OElf e;
    OFunction f;
    size_t i, j;
    OSha256 hash;
    if (!p)
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    memset(p, 0, sizeof(*p));
    TRY(o_lm_site_layout(shape, p));
    TRY(o_elf_init(data, size, &e));
    TRY(o_find_function(&e, "odezza_trajectory_lm", &f));
    p->cubin_size = size;
    p->architecture = e.flags & 255u;
    if (p->architecture < 30u)
        p->architecture = (e.flags >> 8) & 255u;
    if (!o_sass_architecture_supported(p->architecture))
        return ODEZZA_ERROR_UNSUPPORTED;
    if (f.size % 16u)
        return ODEZZA_ERROR_FORMAT;
    TRY(o_check_range(size, f.file_offset, f.size));
    for (i = 0; i < e.section_count; ++i) {
        OElfSection section;
        int match;
        TRY(o_elf_section(&e, i, &section));
        TRY(o_section_name_contains(&e, &section, ".nv.info", &match));
        if (match) {
            TRY(o_check_range(size, section.offset, section.size));
            for (j = 0; j + 12 <= section.size; j += 4) {
                size_t at = section.offset + j;
                uint16_t n;
                uint32_t sym, value;
                if (e.data[at] != 4 || e.data[at + 1] != 0x2f)
                    continue;
                TRY(o_read_u16(e.data, size, at + 2, &n));
                TRY(o_read_u32(e.data, size, at + 4, &sym));
                if (n != 8 || sym != f.symbol_index)
                    continue;
                TRY(o_read_u32(e.data, size, at + 8, &value));
                if (p->register_offset_count >= 32 || (p->registers && p->registers != value))
                    return ODEZZA_ERROR_FORMAT;
                p->registers = value;
                p->register_offsets[p->register_offset_count++] = at + 8;
            }
        }
        TRY(o_section_name_starts_with(&e, &section, ".text.", &match));
        if (match && (section.info & 0xffffffu) == f.symbol_index && section.info >> 24) {
            uint32_t value = section.info >> 24;
            if (p->header_offset_count >= 32 || (p->registers && p->registers != value))
                return ODEZZA_ERROR_FORMAT;
            p->registers = value;
            p->header_offsets[p->header_offset_count++] = e.section_table_offset + i * 64 + 47;
        }
    }
    if (!p->registers || p->registers > 255)
        return ODEZZA_ERROR_FORMAT;
    /* A valid template can exhaust the register file before specialization.
     * Keep rejecting it, but distinguish capacity from malformed ELF/SASS. */
    if (p->registers == 255)
        return ODEZZA_ERROR_REGISTER_PRESSURE;
    for (i = 0; i < p->site_count; ++i) {
        TRY(inspect_site(&e, &f, &p->sites[i], (uint32_t)i, shape->site_patch_capacity));
        for (j = 0; j < i; ++j)
            if (p->sites[i].scaffold < p->sites[j].continuation &&
                p->sites[j].scaffold < p->sites[i].continuation)
                return ODEZZA_ERROR_FORMAT;
    }
    TRY(o_sha256_init(&hash));
    TRY(o_sha256_update(&hash, data, size));
    return o_sha256_final(&hash, p->cubin_id);
}
