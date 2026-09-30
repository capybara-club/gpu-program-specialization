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
#include "o_scoring_fixture.h"
#include "o_sha256.h"
#include <stdio.h>

static uint32_t rng = 0x6d2171abu;
static uint32_t random_word(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }
static void expression(uint8_t *bytes, size_t *count, unsigned depth) {
    static const uint8_t unary[] = {ODEZZA_AST_NEG_F32, ODEZZA_AST_ABS_F32, ODEZZA_AST_SQRT_F32, ODEZZA_AST_RCP_F32,
        ODEZZA_AST_SIN_F32, ODEZZA_AST_COS_F32, ODEZZA_AST_EX2_F32, ODEZZA_AST_LG2_F32, ODEZZA_AST_RSQRT_F32,
        ODEZZA_AST_TANH_F32, ODEZZA_AST_EXP_F32, ODEZZA_AST_LOG_F32};
    static const uint8_t binary[] = {ODEZZA_AST_ADD_F32, ODEZZA_AST_SUB_F32, ODEZZA_AST_MUL_F32, ODEZZA_AST_DIV_F32,
        ODEZZA_AST_MIN_F32, ODEZZA_AST_MAX_F32};
    unsigned kind = random_word() % 8u;
    if (depth == 0u || kind == 0u) {
        bytes[(*count)++] = ODEZZA_AST_STATE_F32;
        bytes[(*count)++] = random_word() % 2u;
    } else if (kind == 1u) {
        bytes[(*count)++] = ODEZZA_AST_CONSTANT_F32;
        bytes[(*count)++] = random_word() % 4u;
    } else if (kind == 2u) {
        bytes[(*count)++] = ODEZZA_AST_LITERAL_F32;
        bytes[(*count)++] = 0u; bytes[(*count)++] = 0u; bytes[(*count)++] = 128u; bytes[(*count)++] = 63u;
    } else if (kind == 3u) {
        uint8_t bit = random_word() % 32u;
        bytes[(*count)++] = ODEZZA_AST_STATE_F32; bytes[(*count)++] = 0u;
        bytes[(*count)++] = ODEZZA_AST_CONSTANT_F32; bytes[(*count)++] = 0u;
        bytes[(*count)++] = ODEZZA_AST_STATE_F32; bytes[(*count)++] = 1u;
        bytes[(*count)++] = ODEZZA_AST_CONSTANT_F32; bytes[(*count)++] = 1u;
        bytes[(*count)++] = ODEZZA_AST_TOGGLE4_F32;
        bytes[(*count)++] = bit; bytes[(*count)++] = (bit + 1u) % 32u;
    } else if (kind == 4u) {
        expression(bytes, count, depth - 1u);
        bytes[(*count)++] = unary[random_word() % (sizeof(unary) / sizeof(unary[0]))];
    } else if (kind == 5u) {
        expression(bytes, count, depth - 1u); expression(bytes, count, depth - 1u); expression(bytes, count, depth - 1u);
        bytes[(*count)++] = ODEZZA_AST_FMA_F32;
    } else {
        expression(bytes, count, depth - 1u); expression(bytes, count, depth - 1u);
        bytes[(*count)++] = binary[random_word() % (sizeof(binary) / sizeof(binary[0]))];
    }
}

int main(int argc, char **argv) {
    static const uint32_t architectures[] = {89u, 90u, 120u};
    static const uint8_t fixed[] = {ODEZZA_AST_STATE_F32, 0u, ODEZZA_AST_CONSTANT_F32, 0u, ODEZZA_AST_MUL_F32, ODEZZA_AST_RETURN_F32};
    FILE *output = NULL;
    OSha256 sha;
    unsigned char digest[32];
    char hex[65];
    unsigned successes = 0u, rejected = 0u;
    size_t arch, sample;
    if (argc > 2 || (argc == 2 && (output = fopen(argv[1], "wb")) == NULL)) return 1;
    if (o_sha256_init(&sha) != ODEZZA_SUCCESS) return 1;
    for (arch = 0u; arch < 3u; ++arch) {
        for (sample = 0u; sample < 1001u; ++sample) {
            uint8_t program[4096]; size_t count = 0u, i;
            unsigned char cubin[CUBIN_BYTES], snapshot[CUBIN_BYTES];
            OdezzaScoringCubinInspection inspection;
            OdezzaScoringPrespecialization prespecialization;
            OdezzaScoringRhs fixed_rhs = {0u, {fixed, sizeof(fixed)}};
            OdezzaScoringRhs candidate;
            OdezzaScoringSystem system;
            OdezzaResult result;
            AlignedBytes workspace;
            initialize_inspection(cubin, &inspection);
            inspection.architecture = architectures[arch];
            inspection.system_patch_capacity = 128u;
            if (sample == 0u) {
                for (i = 0u; i < 7u; ++i) { program[count++] = ODEZZA_AST_STATE_F32; program[count++] = 0u; program[count++] = ODEZZA_AST_RCP_F32; }
                for (i = 0u; i < 6u; ++i) program[count++] = ODEZZA_AST_ADD_F32;
            } else expression(program, &count, 4u);
            program[count++] = ODEZZA_AST_RETURN_F32;
            candidate.state_index = 1u; candidate.program.bytes = program; candidate.program.byte_count = count;
            system.rhs = &candidate; system.rhs_count = 1u;
            if (odezza_prespecialize_scoring_cubin(cubin, sizeof(cubin), &inspection, 2u, 4u, &fixed_rhs, 1u,
                                                  workspace.bytes, sizeof(workspace.bytes), &prespecialization) != ODEZZA_SUCCESS) return 1;
            memcpy(snapshot, cubin, sizeof(cubin));
            result = odezza_specialize_scoring_cubin_systems(cubin, sizeof(cubin), &inspection, &prespecialization, 32u, &system, 1u,
                                                             workspace.bytes, sizeof(workspace.bytes), NULL);
            if (result == ODEZZA_SUCCESS) ++successes;
            else if (result == ODEZZA_ERROR_SPECIALIZATION_CAPACITY || result == ODEZZA_ERROR_REGISTER_PRESSURE) {
                ++rejected; if (memcmp(cubin, snapshot, sizeof(cubin)) != 0) return 1;
            } else { fprintf(stderr, "sample %zu sm%u failed %d\n", sample, architectures[arch], result); return 1; }
            {
                unsigned char status[4] = {(unsigned char)result, 0u, 0u, 0u};
                if (o_sha256_update(&sha, status, sizeof(status)) != ODEZZA_SUCCESS ||
                    o_sha256_update(&sha, cubin, sizeof(cubin)) != ODEZZA_SUCCESS) return 1;
                if (output && (fwrite(status, sizeof(status), 1u, output) != 1u || fwrite(cubin, sizeof(cubin), 1u, output) != 1u)) return 1;
            }
        }
    }
    if (output && fclose(output)) return 1;
    if (o_sha256_final(&sha, digest) != ODEZZA_SUCCESS) return 1;
    for (sample = 0u; sample < sizeof(digest); ++sample) snprintf(hex + sample * 2u, 3u, "%02x", digest[sample]);
    /* Frozen from both the original C99 specializer and the extracted writer. */
    if (strcmp(hex, "15b05e22d2a38453a253b3d806e3af98d88601bc3d76b8f24f0b7f5e31ccb5cd") != 0) {
        fprintf(stderr, "corpus instruction bytes changed: %s\n", hex);
        return 1;
    }
    printf("%u successful assemblies, %u capacity rejections; 3 architecture encodings\n", successes, rejected);
    return 0;
}
