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
#include "o_sass.h"
#include <stdio.h>
#include <string.h>

#define CHECK(condition) do { if (!(condition)) { fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 0; } } while (0)

static int golden(OSassEncoding encoded, uint64_t word0, uint64_t word1, const char *name) {
    if (encoded.status != ODEZZA_SUCCESS || encoded.word0 != word0 || encoded.word1 != word1) {
        fprintf(stderr, "%s: got %016llx %016llx (result %d)\n", name,
                (unsigned long long)encoded.word0, (unsigned long long)encoded.word1, encoded.status);
        return 0;
    }
    return 1;
}

/* Frozen words from the pre-refactor encoder, independently reproduced by
 * python/odezza/sass.py. Include wait fields, all six MUFU barrier slots,
 * predicate endpoints, bit 31, branch packing boundaries, and RZ sources. */
static int test_golden(void) {
    const OSassInstruction prototype = {UINT64_C(0x0000000106ff7812), UINT64_C(0x000fc0000782c0ff)};
    CHECK(golden(o_sass_nop(37u), UINT64_C(0x0000000000007918), UINT64_C(0x025fc00000000000), "o_sass_nop(37u)"));
    CHECK(golden(o_sass_mov(17u, 23u, 37u), UINT64_C(0x0000001700117202), UINT64_C(0x025fcc0000000f00), "o_sass_mov(17u, 23u, 37u)"));
    CHECK(golden(o_sass_mov(254u, 255u, 63u), UINT64_C(0x000000ff00fe7202), UINT64_C(0x03ffcc0000000f00), "o_sass_mov(254u, 255u, 63u)"));
    CHECK(golden(o_sass_fadd_register(17u, 23u, 41u, 37u), UINT64_C(0x0000002917117221), UINT64_C(0x025fcc0000000000), "o_sass_fadd_register(17u, 23u, 41u, 37u)"));
    CHECK(golden(o_sass_fadd_immediate(17u, 23u, 0xbf812345u, 37u), UINT64_C(0xbf81234517117421), UINT64_C(0x025fcc0000010000), "o_sass_fadd_immediate(17u, 23u, 0xbf812345u, 37u)"));
    CHECK(golden(o_sass_fmul_register(17u, 23u, 41u, 37u), UINT64_C(0x0000002917117220), UINT64_C(0x025fcc0000410000), "o_sass_fmul_register(17u, 23u, 41u, 37u)"));
    CHECK(golden(o_sass_fmul_immediate(17u, 23u, 0x3e22f983u, 37u, O_SASS_MULTIPLY_NORMAL), UINT64_C(0x3e22f98317117820), UINT64_C(0x025fcc0000410000), "o_sass_fmul_immediate(17u, 23u, 0x3e22f983u, 37u, O_SASS_MULTIPLY_NORMAL)"));
    CHECK(golden(o_sass_fmul_immediate(17u, 23u, 0x3e22f983u, 37u, O_SASS_MULTIPLY_RZ), UINT64_C(0x3e22f98317117820), UINT64_C(0x025fcc000040c000), "o_sass_fmul_immediate(17u, 23u, 0x3e22f983u, 37u, O_SASS_MULTIPLY_RZ)"));
    CHECK(golden(o_sass_fsel(17u, 23u, 41u, 0u), UINT64_C(0x0000002917117208), UINT64_C(0x000fcc0000000000), "o_sass_fsel(17u, 23u, 41u, 0u)"));
    CHECK(golden(o_sass_fsel(17u, 23u, 41u, 6u), UINT64_C(0x0000002917117208), UINT64_C(0x000fcc0003000000), "o_sass_fsel(17u, 23u, 41u, 6u)"));
    CHECK(golden(o_sass_absolute(17u, 23u, 37u), UINT64_C(0x800000ff17117221), UINT64_C(0x025fcc0000010200), "o_sass_absolute(17u, 23u, 37u)"));
    CHECK(golden(o_sass_fmnmx(17u, 23u, 41u, 0, 37u), UINT64_C(0x0000002917117209), UINT64_C(0x025fcc0003810000), "o_sass_fmnmx(17u, 23u, 41u, 0, 37u)"));
    CHECK(golden(o_sass_fmnmx(17u, 23u, 41u, 1, 37u), UINT64_C(0x0000002917117209), UINT64_C(0x025fcc0007810000), "o_sass_fmnmx(17u, 23u, 41u, 1, 37u)"));
    CHECK(golden(o_sass_ffma(17u, 23u, 41u, 61u, 37u), UINT64_C(0x0000002917117223), UINT64_C(0x025fcc000001003d), "o_sass_ffma(17u, 23u, 41u, 61u, 37u)"));
    CHECK(golden(o_sass_mufu(17u, 23u, 37u, 0u, O_SASS_MUFU_COS), UINT64_C(0x0000001700117308), UINT64_C(0x025e260000000000), "o_sass_mufu(17u, 23u, 37u, 0u, O_SASS_MUFU_COS)"));
    CHECK(golden(o_sass_mufu(17u, 23u, 37u, 1u, O_SASS_MUFU_COS), UINT64_C(0x0000001700117308), UINT64_C(0x025e660000000000), "o_sass_mufu(17u, 23u, 37u, 1u, O_SASS_MUFU_COS)"));
    CHECK(golden(o_sass_mufu(17u, 23u, 37u, 2u, O_SASS_MUFU_COS), UINT64_C(0x0000001700117308), UINT64_C(0x025ea60000000000), "o_sass_mufu(17u, 23u, 37u, 2u, O_SASS_MUFU_COS)"));
    CHECK(golden(o_sass_mufu(17u, 23u, 37u, 3u, O_SASS_MUFU_COS), UINT64_C(0x0000001700117308), UINT64_C(0x025ee60000000000), "o_sass_mufu(17u, 23u, 37u, 3u, O_SASS_MUFU_COS)"));
    CHECK(golden(o_sass_mufu(17u, 23u, 37u, 4u, O_SASS_MUFU_COS), UINT64_C(0x0000001700117308), UINT64_C(0x025f260000000000), "o_sass_mufu(17u, 23u, 37u, 4u, O_SASS_MUFU_COS)"));
    CHECK(golden(o_sass_mufu(17u, 23u, 37u, 5u, O_SASS_MUFU_COS), UINT64_C(0x0000001700117308), UINT64_C(0x025f660000000000), "o_sass_mufu(17u, 23u, 37u, 5u, O_SASS_MUFU_COS)"));
    CHECK(golden(o_sass_mufu(17u, 23u, 37u, 0u, O_SASS_MUFU_SIN), UINT64_C(0x0000001700117308), UINT64_C(0x025e260000000400), "o_sass_mufu(17u, 23u, 37u, 0u, O_SASS_MUFU_SIN)"));
    CHECK(golden(o_sass_mufu(17u, 23u, 37u, 1u, O_SASS_MUFU_SIN), UINT64_C(0x0000001700117308), UINT64_C(0x025e660000000400), "o_sass_mufu(17u, 23u, 37u, 1u, O_SASS_MUFU_SIN)"));
    CHECK(golden(o_sass_mufu(17u, 23u, 37u, 2u, O_SASS_MUFU_SIN), UINT64_C(0x0000001700117308), UINT64_C(0x025ea60000000400), "o_sass_mufu(17u, 23u, 37u, 2u, O_SASS_MUFU_SIN)"));
    CHECK(golden(o_sass_mufu(17u, 23u, 37u, 3u, O_SASS_MUFU_SIN), UINT64_C(0x0000001700117308), UINT64_C(0x025ee60000000400), "o_sass_mufu(17u, 23u, 37u, 3u, O_SASS_MUFU_SIN)"));
    CHECK(golden(o_sass_mufu(17u, 23u, 37u, 4u, O_SASS_MUFU_SIN), UINT64_C(0x0000001700117308), UINT64_C(0x025f260000000400), "o_sass_mufu(17u, 23u, 37u, 4u, O_SASS_MUFU_SIN)"));
    CHECK(golden(o_sass_mufu(17u, 23u, 37u, 5u, O_SASS_MUFU_SIN), UINT64_C(0x0000001700117308), UINT64_C(0x025f660000000400), "o_sass_mufu(17u, 23u, 37u, 5u, O_SASS_MUFU_SIN)"));
    CHECK(golden(o_sass_mufu(17u, 23u, 37u, 0u, O_SASS_MUFU_EX2), UINT64_C(0x0000001700117308), UINT64_C(0x025e260000000800), "o_sass_mufu(17u, 23u, 37u, 0u, O_SASS_MUFU_EX2)"));
    CHECK(golden(o_sass_mufu(17u, 23u, 37u, 1u, O_SASS_MUFU_EX2), UINT64_C(0x0000001700117308), UINT64_C(0x025e660000000800), "o_sass_mufu(17u, 23u, 37u, 1u, O_SASS_MUFU_EX2)"));
    CHECK(golden(o_sass_mufu(17u, 23u, 37u, 2u, O_SASS_MUFU_EX2), UINT64_C(0x0000001700117308), UINT64_C(0x025ea60000000800), "o_sass_mufu(17u, 23u, 37u, 2u, O_SASS_MUFU_EX2)"));
    CHECK(golden(o_sass_mufu(17u, 23u, 37u, 3u, O_SASS_MUFU_EX2), UINT64_C(0x0000001700117308), UINT64_C(0x025ee60000000800), "o_sass_mufu(17u, 23u, 37u, 3u, O_SASS_MUFU_EX2)"));
    CHECK(golden(o_sass_mufu(17u, 23u, 37u, 4u, O_SASS_MUFU_EX2), UINT64_C(0x0000001700117308), UINT64_C(0x025f260000000800), "o_sass_mufu(17u, 23u, 37u, 4u, O_SASS_MUFU_EX2)"));
    CHECK(golden(o_sass_mufu(17u, 23u, 37u, 5u, O_SASS_MUFU_EX2), UINT64_C(0x0000001700117308), UINT64_C(0x025f660000000800), "o_sass_mufu(17u, 23u, 37u, 5u, O_SASS_MUFU_EX2)"));
    CHECK(golden(o_sass_mufu(17u, 23u, 37u, 0u, O_SASS_MUFU_LG2), UINT64_C(0x0000001700117308), UINT64_C(0x025e260000000c00), "o_sass_mufu(17u, 23u, 37u, 0u, O_SASS_MUFU_LG2)"));
    CHECK(golden(o_sass_mufu(17u, 23u, 37u, 1u, O_SASS_MUFU_LG2), UINT64_C(0x0000001700117308), UINT64_C(0x025e660000000c00), "o_sass_mufu(17u, 23u, 37u, 1u, O_SASS_MUFU_LG2)"));
    CHECK(golden(o_sass_mufu(17u, 23u, 37u, 2u, O_SASS_MUFU_LG2), UINT64_C(0x0000001700117308), UINT64_C(0x025ea60000000c00), "o_sass_mufu(17u, 23u, 37u, 2u, O_SASS_MUFU_LG2)"));
    CHECK(golden(o_sass_mufu(17u, 23u, 37u, 3u, O_SASS_MUFU_LG2), UINT64_C(0x0000001700117308), UINT64_C(0x025ee60000000c00), "o_sass_mufu(17u, 23u, 37u, 3u, O_SASS_MUFU_LG2)"));
    CHECK(golden(o_sass_mufu(17u, 23u, 37u, 4u, O_SASS_MUFU_LG2), UINT64_C(0x0000001700117308), UINT64_C(0x025f260000000c00), "o_sass_mufu(17u, 23u, 37u, 4u, O_SASS_MUFU_LG2)"));
    CHECK(golden(o_sass_mufu(17u, 23u, 37u, 5u, O_SASS_MUFU_LG2), UINT64_C(0x0000001700117308), UINT64_C(0x025f660000000c00), "o_sass_mufu(17u, 23u, 37u, 5u, O_SASS_MUFU_LG2)"));
    CHECK(golden(o_sass_mufu(17u, 23u, 37u, 0u, O_SASS_MUFU_RCP), UINT64_C(0x0000001700117308), UINT64_C(0x025e260000001000), "o_sass_mufu(17u, 23u, 37u, 0u, O_SASS_MUFU_RCP)"));
    CHECK(golden(o_sass_mufu(17u, 23u, 37u, 1u, O_SASS_MUFU_RCP), UINT64_C(0x0000001700117308), UINT64_C(0x025e660000001000), "o_sass_mufu(17u, 23u, 37u, 1u, O_SASS_MUFU_RCP)"));
    CHECK(golden(o_sass_mufu(17u, 23u, 37u, 2u, O_SASS_MUFU_RCP), UINT64_C(0x0000001700117308), UINT64_C(0x025ea60000001000), "o_sass_mufu(17u, 23u, 37u, 2u, O_SASS_MUFU_RCP)"));
    CHECK(golden(o_sass_mufu(17u, 23u, 37u, 3u, O_SASS_MUFU_RCP), UINT64_C(0x0000001700117308), UINT64_C(0x025ee60000001000), "o_sass_mufu(17u, 23u, 37u, 3u, O_SASS_MUFU_RCP)"));
    CHECK(golden(o_sass_mufu(17u, 23u, 37u, 4u, O_SASS_MUFU_RCP), UINT64_C(0x0000001700117308), UINT64_C(0x025f260000001000), "o_sass_mufu(17u, 23u, 37u, 4u, O_SASS_MUFU_RCP)"));
    CHECK(golden(o_sass_mufu(17u, 23u, 37u, 5u, O_SASS_MUFU_RCP), UINT64_C(0x0000001700117308), UINT64_C(0x025f660000001000), "o_sass_mufu(17u, 23u, 37u, 5u, O_SASS_MUFU_RCP)"));
    CHECK(golden(o_sass_mufu(17u, 23u, 37u, 0u, O_SASS_MUFU_RSQ), UINT64_C(0x0000001700117308), UINT64_C(0x025e260000001400), "o_sass_mufu(17u, 23u, 37u, 0u, O_SASS_MUFU_RSQ)"));
    CHECK(golden(o_sass_mufu(17u, 23u, 37u, 1u, O_SASS_MUFU_RSQ), UINT64_C(0x0000001700117308), UINT64_C(0x025e660000001400), "o_sass_mufu(17u, 23u, 37u, 1u, O_SASS_MUFU_RSQ)"));
    CHECK(golden(o_sass_mufu(17u, 23u, 37u, 2u, O_SASS_MUFU_RSQ), UINT64_C(0x0000001700117308), UINT64_C(0x025ea60000001400), "o_sass_mufu(17u, 23u, 37u, 2u, O_SASS_MUFU_RSQ)"));
    CHECK(golden(o_sass_mufu(17u, 23u, 37u, 3u, O_SASS_MUFU_RSQ), UINT64_C(0x0000001700117308), UINT64_C(0x025ee60000001400), "o_sass_mufu(17u, 23u, 37u, 3u, O_SASS_MUFU_RSQ)"));
    CHECK(golden(o_sass_mufu(17u, 23u, 37u, 4u, O_SASS_MUFU_RSQ), UINT64_C(0x0000001700117308), UINT64_C(0x025f260000001400), "o_sass_mufu(17u, 23u, 37u, 4u, O_SASS_MUFU_RSQ)"));
    CHECK(golden(o_sass_mufu(17u, 23u, 37u, 5u, O_SASS_MUFU_RSQ), UINT64_C(0x0000001700117308), UINT64_C(0x025f660000001400), "o_sass_mufu(17u, 23u, 37u, 5u, O_SASS_MUFU_RSQ)"));
    CHECK(golden(o_sass_mufu(17u, 23u, 37u, 0u, O_SASS_MUFU_SQRT), UINT64_C(0x0000001700117308), UINT64_C(0x025e260000002000), "o_sass_mufu(17u, 23u, 37u, 0u, O_SASS_MUFU_SQRT)"));
    CHECK(golden(o_sass_mufu(17u, 23u, 37u, 1u, O_SASS_MUFU_SQRT), UINT64_C(0x0000001700117308), UINT64_C(0x025e660000002000), "o_sass_mufu(17u, 23u, 37u, 1u, O_SASS_MUFU_SQRT)"));
    CHECK(golden(o_sass_mufu(17u, 23u, 37u, 2u, O_SASS_MUFU_SQRT), UINT64_C(0x0000001700117308), UINT64_C(0x025ea60000002000), "o_sass_mufu(17u, 23u, 37u, 2u, O_SASS_MUFU_SQRT)"));
    CHECK(golden(o_sass_mufu(17u, 23u, 37u, 3u, O_SASS_MUFU_SQRT), UINT64_C(0x0000001700117308), UINT64_C(0x025ee60000002000), "o_sass_mufu(17u, 23u, 37u, 3u, O_SASS_MUFU_SQRT)"));
    CHECK(golden(o_sass_mufu(17u, 23u, 37u, 4u, O_SASS_MUFU_SQRT), UINT64_C(0x0000001700117308), UINT64_C(0x025f260000002000), "o_sass_mufu(17u, 23u, 37u, 4u, O_SASS_MUFU_SQRT)"));
    CHECK(golden(o_sass_mufu(17u, 23u, 37u, 5u, O_SASS_MUFU_SQRT), UINT64_C(0x0000001700117308), UINT64_C(0x025f660000002000), "o_sass_mufu(17u, 23u, 37u, 5u, O_SASS_MUFU_SQRT)"));
    CHECK(golden(o_sass_mufu(17u, 23u, 37u, 0u, O_SASS_MUFU_TANH), UINT64_C(0x0000001700117308), UINT64_C(0x025e260000002400), "o_sass_mufu(17u, 23u, 37u, 0u, O_SASS_MUFU_TANH)"));
    CHECK(golden(o_sass_mufu(17u, 23u, 37u, 1u, O_SASS_MUFU_TANH), UINT64_C(0x0000001700117308), UINT64_C(0x025e660000002400), "o_sass_mufu(17u, 23u, 37u, 1u, O_SASS_MUFU_TANH)"));
    CHECK(golden(o_sass_mufu(17u, 23u, 37u, 2u, O_SASS_MUFU_TANH), UINT64_C(0x0000001700117308), UINT64_C(0x025ea60000002400), "o_sass_mufu(17u, 23u, 37u, 2u, O_SASS_MUFU_TANH)"));
    CHECK(golden(o_sass_mufu(17u, 23u, 37u, 3u, O_SASS_MUFU_TANH), UINT64_C(0x0000001700117308), UINT64_C(0x025ee60000002400), "o_sass_mufu(17u, 23u, 37u, 3u, O_SASS_MUFU_TANH)"));
    CHECK(golden(o_sass_mufu(17u, 23u, 37u, 4u, O_SASS_MUFU_TANH), UINT64_C(0x0000001700117308), UINT64_C(0x025f260000002400), "o_sass_mufu(17u, 23u, 37u, 4u, O_SASS_MUFU_TANH)"));
    CHECK(golden(o_sass_mufu(17u, 23u, 37u, 5u, O_SASS_MUFU_TANH), UINT64_C(0x0000001700117308), UINT64_C(0x025f660000002400), "o_sass_mufu(17u, 23u, 37u, 5u, O_SASS_MUFU_TANH)"));
    CHECK(golden(o_sass_branch(1u, 89u), UINT64_C(0x0000000000007947), UINT64_C(0x000fcc0003800000), "o_sass_branch(1u, 89u)"));
    CHECK(golden(o_sass_branch(2u, 89u), UINT64_C(0x0000001000007947), UINT64_C(0x000fcc0003800000), "o_sass_branch(2u, 89u)"));
    CHECK(golden(o_sass_branch(64u, 89u), UINT64_C(0x000003f000007947), UINT64_C(0x000fcc0003800000), "o_sass_branch(64u, 89u)"));
    CHECK(golden(o_sass_branch(65u, 89u), UINT64_C(0x0000040000007947), UINT64_C(0x000fcc0003800000), "o_sass_branch(65u, 89u)"));
    CHECK(golden(o_sass_branch(65536u, 89u), UINT64_C(0x000ffff000007947), UINT64_C(0x000fcc0003800000), "o_sass_branch(65536u, 89u)"));
    CHECK(golden(o_sass_branch(134217728u, 89u), UINT64_C(0x7ffffff000007947), UINT64_C(0x000fcc0003800000), "o_sass_branch(134217728u, 89u)"));
    CHECK(golden(o_sass_branch(1u, 90u), UINT64_C(0x0000000000007947), UINT64_C(0x000fcc0003800000), "o_sass_branch(1u, 90u)"));
    CHECK(golden(o_sass_branch(2u, 90u), UINT64_C(0x0000000000047947), UINT64_C(0x000fcc0003800000), "o_sass_branch(2u, 90u)"));
    CHECK(golden(o_sass_branch(64u, 90u), UINT64_C(0x0000000000fc7947), UINT64_C(0x000fcc0003800000), "o_sass_branch(64u, 90u)"));
    CHECK(golden(o_sass_branch(65u, 90u), UINT64_C(0x0000000400007947), UINT64_C(0x000fcc0003800000), "o_sass_branch(65u, 90u)"));
    CHECK(golden(o_sass_branch(65536u, 90u), UINT64_C(0x00000ffc00fc7947), UINT64_C(0x000fcc0003800000), "o_sass_branch(65536u, 90u)"));
    CHECK(golden(o_sass_branch(134217728u, 90u), UINT64_C(0x007ffffc00fc7947), UINT64_C(0x000fcc0003800000), "o_sass_branch(134217728u, 90u)"));
    CHECK(golden(o_sass_branch(1u, 120u), UINT64_C(0x0000000000007947), UINT64_C(0x000fcc0003800000), "o_sass_branch(1u, 120u)"));
    CHECK(golden(o_sass_branch(2u, 120u), UINT64_C(0x0000000000047947), UINT64_C(0x000fcc0003800000), "o_sass_branch(2u, 120u)"));
    CHECK(golden(o_sass_branch(64u, 120u), UINT64_C(0x0000000000fc7947), UINT64_C(0x000fcc0003800000), "o_sass_branch(64u, 120u)"));
    CHECK(golden(o_sass_branch(65u, 120u), UINT64_C(0x0000000400007947), UINT64_C(0x000fcc0003800000), "o_sass_branch(65u, 120u)"));
    CHECK(golden(o_sass_branch(65536u, 120u), UINT64_C(0x00000ffc00fc7947), UINT64_C(0x000fcc0003800000), "o_sass_branch(65536u, 120u)"));
    CHECK(golden(o_sass_branch(134217728u, 120u), UINT64_C(0x007ffffc00fc7947), UINT64_C(0x000fcc0003800000), "o_sass_branch(134217728u, 120u)"));
    CHECK(golden(o_sass_toggle_test(prototype, 23u, 1u, 0u), UINT64_C(0x0000000117ff7812), UINT64_C(0x000fcc000782c0ff), "o_sass_toggle_test(prototype, 23u, 1u, 0u)"));
    CHECK(golden(o_sass_toggle_test(prototype, 23u, 1u, 1u), UINT64_C(0x0000000217ff7812), UINT64_C(0x000fcc000782c0ff), "o_sass_toggle_test(prototype, 23u, 1u, 1u)"));
    CHECK(golden(o_sass_toggle_test(prototype, 23u, 1u, 31u), UINT64_C(0x8000000017ff7812), UINT64_C(0x000fcc000782c0ff), "o_sass_toggle_test(prototype, 23u, 1u, 31u)"));
    return 1;
}

static int test_invalid_operands(void) {
    OSassInstruction instruction = {1u, 2u};
    OSassInstruction prototype = {UINT64_C(0x0000000106ff7812), UINT64_C(0x000fc0000782c0ff)};
    OSassEncoding invalid[] = {
        o_sass_mov(256u, 0u, 0u), o_sass_mov(255u, 0u, 0u), o_sass_mov(0u, 256u, 0u), o_sass_nop(64u),
        o_sass_fadd_register(0u, 0u, 256u, 0u), o_sass_fadd_immediate(0u, 0u, 0u, 64u),
        o_sass_fmul_register(0u, 0u, 256u, 0u), o_sass_fmul_immediate(0u, 0u, 0u, 0u, (OSassMultiplyMode)2),
        o_sass_fsel(0u, 0u, 0u, 7u), o_sass_fsel(0u, 256u, 0u, 0u), o_sass_absolute(0u, 256u, 0u),
        o_sass_fmnmx(0u, 0u, 0u, 2, 0u), o_sass_ffma(0u, 0u, 0u, 256u, 0u),
        o_sass_mufu(0u, 0u, 0u, 6u, O_SASS_MUFU_RCP), o_sass_mufu(0u, 0u, 0u, 0u, (OSassMufu)-1),
        o_sass_toggle_test(prototype, 0u, 1u, 32u), o_sass_toggle_test(prototype, 255u, 1u, 0u),
        o_sass_branch(0u, 120u)
    };
    size_t i;
    for (i = 0u; i < sizeof(invalid) / sizeof(invalid[0]); ++i) {
        CHECK(o_sass_instruction(invalid[i], &instruction) == ODEZZA_ERROR_INVALID_ARGUMENT);
        CHECK(instruction.word0 == 1u && instruction.word1 == 2u);
    }
    CHECK(o_sass_toggle_test(prototype, 0u, 2u, 0u).status == ODEZZA_ERROR_FORMAT);
    prototype.word0 ^= 1u;
    CHECK(o_sass_toggle_test(prototype, 0u, 1u, 0u).status == ODEZZA_ERROR_FORMAT);
    CHECK(o_sass_branch(1u, 80u).status == ODEZZA_ERROR_UNSUPPORTED);
    CHECK(o_sass_branch(134217729u, 89u).status == ODEZZA_ERROR_OVERFLOW);
    CHECK(o_sass_branch(134217729u, 120u).status == ODEZZA_ERROR_OVERFLOW);
    CHECK(o_sass_branch(SIZE_MAX, 90u).status == ODEZZA_ERROR_OVERFLOW);
    CHECK(o_sass_instruction(o_sass_nop(0u), NULL) == ODEZZA_ERROR_INVALID_ARGUMENT);
    return 1;
}

static int test_writer(void) {
    OSassInstruction buffer[2] = {{1u, 2u}, {3u, 4u}};
    OSassInstruction snapshot[2];
    OSassWriter writer;
    memcpy(snapshot, buffer, sizeof(buffer));
    CHECK(o_sass_writer_init(&writer, buffer, 1u, 2u) == ODEZZA_SUCCESS);
    CHECK(o_sass_writer_emit(&writer, o_sass_mov(256u, 0u, 0u)) == ODEZZA_ERROR_INVALID_ARGUMENT);
    CHECK(writer.count == 0u && writer.pending_wait_mask == 2u && memcmp(buffer, snapshot, sizeof(buffer)) == 0);
    CHECK(o_sass_writer_emit(&writer, o_sass_mov(1u, 2u, 4u)) == ODEZZA_SUCCESS);
    CHECK(writer.count == 1u && writer.pending_wait_mask == 0u && (buffer[0].word1 >> 52u) == 6u);
    memcpy(snapshot, buffer, sizeof(buffer));
    writer.pending_wait_mask = 8u;
    CHECK(o_sass_writer_emit(&writer, o_sass_nop(0u)) == ODEZZA_ERROR_SPECIALIZATION_CAPACITY);
    CHECK(writer.count == 1u && writer.pending_wait_mask == 8u && memcmp(buffer, snapshot, sizeof(buffer)) == 0);
    writer.count = SIZE_MAX;
    CHECK(o_sass_writer_emit(&writer, o_sass_nop(0u)) == ODEZZA_ERROR_SPECIALIZATION_CAPACITY);
    CHECK(memcmp(buffer, snapshot, sizeof(buffer)) == 0);
    CHECK(o_sass_writer_init(&writer, NULL, 0u, 0u) == ODEZZA_SUCCESS);
    CHECK(o_sass_writer_emit(&writer, o_sass_nop(0u)) == ODEZZA_ERROR_SPECIALIZATION_CAPACITY);
    CHECK(o_sass_writer_init(&writer, NULL, 1u, 0u) == ODEZZA_ERROR_INVALID_ARGUMENT);
    CHECK(o_sass_writer_init(&writer, buffer, SIZE_MAX, 0u) == ODEZZA_ERROR_OVERFLOW);
    CHECK(o_sass_writer_init(&writer, buffer, 1u, 64u) == ODEZZA_ERROR_INVALID_ARGUMENT);
    return 1;
}

static int test_store(void) {
    unsigned char bytes[20];
    unsigned char snapshot[20];
    static const unsigned char expected[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    OSassInstruction instruction = {UINT64_C(0x0706050403020100), UINT64_C(0x0f0e0d0c0b0a0908)};
    memset(bytes, 0xa5, sizeof(bytes));
    memcpy(snapshot, bytes, sizeof(bytes));
    CHECK(o_sass_store(bytes, sizeof(bytes), SIZE_MAX, instruction) == ODEZZA_ERROR_INSUFFICIENT_BUFFER);
    CHECK(o_sass_store(bytes, sizeof(bytes), 5u, instruction) == ODEZZA_ERROR_INSUFFICIENT_BUFFER);
    CHECK(memcmp(bytes, snapshot, sizeof(bytes)) == 0);
    CHECK(o_sass_store(bytes, sizeof(bytes), 3u, instruction) == ODEZZA_SUCCESS);
    CHECK(memcmp(bytes + 3u, expected, sizeof(expected)) == 0);
    CHECK(bytes[0] == 0xa5 && bytes[2] == 0xa5 && bytes[19] == 0xa5);
    CHECK(o_sass_store(NULL, 16u, 0u, instruction) == ODEZZA_ERROR_INVALID_ARGUMENT);
    return 1;
}

int main(void) {
    if (!test_golden() || !test_invalid_operands() || !test_writer() || !test_store()) return 1;
    puts("SASS encodings, field rejection, emission, and byte bounds: verified");
    return 0;
}
