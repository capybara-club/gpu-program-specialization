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
#include "secant_instructions.h"
#include "../s_ast_internal.h"

#include <string.h>

int
main(void) {
    static const SecantAstInstruction encoded[] = {
        secant_ast_encode_column_f32(127u),
        secant_ast_encode_constant_f32_bits(
            SECANT_F32_BITS_ONE_AND_HALF),
        secant_ast_encode_add_f32,
        secant_ast_encode_routine_f32(254u),
        secant_ast_encode_return_f32
    };
    SecantAstInstruction written[sizeof(encoded)];
    size_t written_size = 0u;

    if (secant_internal_ast_instruction_num_args(
            SECANT_AST_INSTRUCTION_TYPE_ROUTINE_F32) !=
            SECANT_INTERNAL_AST_DYNAMIC_ARITY ||
        secant_internal_ast_instruction_num_args(
            SECANT_AST_INSTRUCTION_TYPE_COLUMN_F32) != 0u ||
        secant_internal_ast_instruction_num_args(
            SECANT_AST_INSTRUCTION_TYPE_SQRT_F32) != 1u ||
        secant_internal_ast_instruction_num_args(
            SECANT_AST_INSTRUCTION_TYPE_ADD_S32) != 2u ||
        secant_internal_ast_instruction_num_args(
            SECANT_AST_INSTRUCTION_TYPE_FMA_U32) != 3u ||
        !secant_internal_ast_instruction_is_f32(SECANT_AST_INSTRUCTION_TYPE_ADD_F32) ||
        secant_internal_ast_instruction_is_f32(SECANT_AST_INSTRUCTION_TYPE_ADD_S32) ||
        secant_internal_ast_instruction_is_f32(SECANT_AST_INSTRUCTION_TYPE_ADD_U32) ||
        sizeof(SecantAstInstruction) != 1u ||
        sizeof(encoded) != 11u ||
        secant_ast_instruction_type_get(encoded) !=
            SECANT_AST_INSTRUCTION_TYPE_COLUMN_F32 ||
        secant_ast_index_get(encoded) != 127u ||
        secant_ast_instruction_size_get(encoded) != 2u ||
        secant_ast_instruction_size_get(encoded + 2u) != 5u ||
        secant_ast_constant_f32_bits_get(encoded + 2u) != 0x3fc00000u ||
        secant_ast_instruction_type_get(encoded + 8u) !=
            SECANT_AST_INSTRUCTION_TYPE_ROUTINE_F32 ||
        secant_ast_index_get(encoded + 8u) != 254u) {
        return 1;
    }
    if (secant_internal_ast_instruction_write(
            SECANT_AST_INSTRUCTION_TYPE_COLUMN_F32,
            127u,
            0u,
            written,
            sizeof(written),
            &written_size) != SECANT_SUCCESS ||
        secant_internal_ast_instruction_write(
            SECANT_AST_INSTRUCTION_TYPE_CONSTANT_BITS_F32,
            0u,
            0x3fc00000u,
            written,
            sizeof(written),
            &written_size) != SECANT_SUCCESS ||
        secant_internal_ast_instruction_write(
            SECANT_AST_INSTRUCTION_TYPE_ADD_F32,
            0u,
            0u,
            written,
            sizeof(written),
            &written_size) != SECANT_SUCCESS ||
        secant_internal_ast_instruction_write(
            SECANT_AST_INSTRUCTION_TYPE_ROUTINE_F32,
            254u,
            0u,
            written,
            sizeof(written),
            &written_size) != SECANT_SUCCESS ||
        secant_internal_ast_instruction_write(
            SECANT_AST_INSTRUCTION_TYPE_RETURN_F32,
            0u,
            0u,
            written,
            sizeof(written),
            &written_size) != SECANT_SUCCESS ||
        written_size != sizeof(encoded) ||
        memcmp(encoded, written, sizeof(encoded)) != 0) {
        return 1;
    }
    return 0;
}
