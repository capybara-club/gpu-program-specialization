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
#ifndef SECANT_INSTRUCTIONS_H_INCLUDED
#define SECANT_INSTRUCTIONS_H_INCLUDED

#include "secant.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/**
 * @file secant_instructions.h
 * @brief AST bytecode helpers and initializer encoders.
 */

#define SECANT_AST_MAX_PROGRAM_INSTRUCTIONS 1024u
#define SECANT_AST_MAX_INSTRUCTION_BYTES 10u
#define SECANT_AST_MAX_PROGRAM_BYTES (SECANT_AST_MAX_PROGRAM_INSTRUCTIONS * SECANT_AST_MAX_INSTRUCTION_BYTES)
#define SECANT_AST_MAX_STACK_DEPTH 128u
#define SECANT_AST_MAX_INSTRUCTION_ARGS 4u
#define SECANT_AST_MAX_INPUTS 128u
#define SECANT_AST_MAX_TOGGLE_BITS 32u
#define SECANT_AST_MAX_ROUTINES 255u

#define SECANT_F32_BITS_POSITIVE_ZERO 0x00000000u
#define SECANT_F32_BITS_NEGATIVE_ZERO 0x80000000u
#define SECANT_F32_BITS_ONE_MILLIONTH 0x358637bdu
#define SECANT_F32_BITS_ONE_HUNDREDTH 0x3c23d70au
#define SECANT_F32_BITS_QUARTER 0x3e800000u
#define SECANT_F32_BITS_HALF 0x3f000000u
#define SECANT_F32_BITS_ONE 0x3f800000u
#define SECANT_F32_BITS_ONE_AND_HALF 0x3fc00000u
#define SECANT_F32_BITS_TWO 0x40000000u
#define SECANT_F32_BITS_TWO_AND_HALF 0x40200000u
#define SECANT_F32_BITS_NEGATIVE_ONE 0xbf800000u
#define SECANT_F32_BITS_PI 0x40490fdbu
#define SECANT_F32_BITS_LN_TWO 0x3f317218u
#define SECANT_F32_BITS_LOG2_E 0x3fb8aa3bu

/**
 * Returns the decoded operation type at an instruction boundary.
 *
 * @param[in] instruction Pointer to the first byte of an encoded instruction;
 *     must not be NULL.
 * @return Decoded instruction type. The returned enum can be invalid when the
 *     first byte is not a defined opcode.
 */
static inline SecantAstInstructionType secant_ast_instruction_type_get(
    const SecantAstInstruction* instruction
) {
    return (SecantAstInstructionType)instruction[0];
}

/**
 * Returns the encoded byte width of one instruction.
 *
 * @param[in] instruction Pointer to the first byte of an encoded instruction;
 *     must not be NULL.
 * @return Instruction width in bytes, or zero for an invalid opcode.
 */
static inline size_t secant_ast_instruction_size_get(
    const SecantAstInstruction* instruction
) {
    const SecantAstInstructionType instruction_type = secant_ast_instruction_type_get(instruction);

    switch (instruction_type) {
        case SECANT_AST_INSTRUCTION_TYPE_AFFINE_BANK_F32:
            return 10u;
        case SECANT_AST_INSTRUCTION_TYPE_CONSTANT_BITS_F32:
        case SECANT_AST_INSTRUCTION_TYPE_CONSTANT_S32:
        case SECANT_AST_INSTRUCTION_TYPE_CONSTANT_U32:
            return 5u;
        case SECANT_AST_INSTRUCTION_TYPE_ROUTINE_F32:
        case SECANT_AST_INSTRUCTION_TYPE_ROUTINE_ARG_F32:
        case SECANT_AST_INSTRUCTION_TYPE_BANK_CONSTANT_F32:
        case SECANT_AST_INSTRUCTION_TYPE_TOGGLE2_F32:
        case SECANT_AST_INSTRUCTION_TYPE_COLUMN_F32:
        case SECANT_AST_INSTRUCTION_TYPE_INPUT_S32:
        case SECANT_AST_INSTRUCTION_TYPE_ROUTINE_S32:
        case SECANT_AST_INSTRUCTION_TYPE_ROUTINE_ARG_S32:
        case SECANT_AST_INSTRUCTION_TYPE_INPUT_U32:
        case SECANT_AST_INSTRUCTION_TYPE_ROUTINE_U32:
        case SECANT_AST_INSTRUCTION_TYPE_ROUTINE_ARG_U32:
            return 2u;
        case SECANT_AST_INSTRUCTION_TYPE_TOGGLE4_F32:
            return 3u;
        case 0xb7: case 0xb8: case 0xb9:
            return 0u;
        default:
            return instruction_type > SECANT_AST_INSTRUCTION_TYPE_NONE &&
                    instruction_type < SECANT_AST_INSTRUCTION_TYPE_ONE_PAST_LAST
                ? 1u
                : 0u;
    }
}

/**
 * Returns the index carried by an input, routine call, or routine argument.
 *
 * @param[in] instruction Pointer to a decoded input, routine, or routine-argument
 *     instruction with its complete payload available.
 * @return The encoded input, routine, or argument index.
 * @pre The caller has established that `instruction` has one of the supported
 *     indexed instruction types.
 */
static inline SecantAstIdx secant_ast_index_get(
    const SecantAstInstruction* instruction
) {
    return instruction[1];
}

/**
 * Returns the exact little-endian payload of a scalar constant.
 *
 * @param[in] instruction Pointer to a five-byte f32, s32, or u32 constant
 *     instruction.
 * @return The four-byte payload reassembled as uint32_t.
 * @pre The caller has validated the constant instruction and available bytes.
 */
static inline uint32_t secant_ast_constant_bits_get(
    const SecantAstInstruction* instruction
) {
    return (uint32_t)instruction[1] |
        ((uint32_t)instruction[2] << 8u) |
        ((uint32_t)instruction[3] << 16u) |
        ((uint32_t)instruction[4] << 24u);
}

/** Affine bank leaf payload: opcode, slot, scale bits, offset bits (LE).
 * Both transforms are finite. Multiplication and addition round separately;
 * this deliberately matches the native writer, without implicit FMA fusion. */
static inline uint32_t secant_ast_affine_bank_scale_bits_get(const uint8_t *p) {
    return secant_ast_constant_bits_get(p + 1);
}
static inline uint32_t secant_ast_affine_bank_offset_bits_get(const uint8_t *p) {
    return secant_ast_constant_bits_get(p + 5);
}
static inline float secant_ast_affine_bank_scale_get(const uint8_t *p) {
    uint32_t bits = secant_ast_affine_bank_scale_bits_get(p);
    float f; memcpy(&f, &bits, sizeof(f)); return f;
}
static inline float secant_ast_affine_bank_offset_get(const uint8_t *p) {
    uint32_t bits = secant_ast_affine_bank_offset_bits_get(p);
    float f; memcpy(&f, &bits, sizeof(f)); return f;
}
static inline void secant_ast_affine_bank_write(uint8_t *p, uint8_t slot,
                                               float scale, float offset) {
    uint32_t a, b; unsigned i;
    memcpy(&a, &scale, 4); memcpy(&b, &offset, 4);
    p[0] = SECANT_AST_INSTRUCTION_TYPE_AFFINE_BANK_F32; p[1] = slot;
    for (i = 0; i < 4; ++i) { p[2+i] = (uint8_t)(a >> (8*i)); p[6+i] = (uint8_t)(b >> (8*i)); }
}

/**
 * Returns the exact f32 object representation stored by a constant.
 *
 * @param[in] instruction Pointer to a five-byte f32 constant instruction.
 * @return The encoded IEEE-754 binary32 bits without conversion.
 */
static inline uint32_t secant_ast_constant_f32_bits_get(
    const SecantAstInstruction* instruction
) {
    return secant_ast_constant_bits_get(instruction);
}

/**
 * Returns the f32 value stored by a constant.
 *
 * @param[in] instruction Pointer to a five-byte f32 constant instruction.
 * @return The float with the exact encoded object representation.
 */
static inline float secant_ast_constant_f32_get(
    const SecantAstInstruction* instruction
) {
    const uint32_t bits = secant_ast_constant_bits_get(instruction);
    float value;

    memcpy(&value, &bits, sizeof(value));
    return value;
}

/**
 * Returns the s32 value stored by a constant.
 *
 * @param[in] instruction Pointer to a five-byte s32 constant instruction.
 * @return The encoded signed 32-bit value.
 */
static inline int32_t secant_ast_constant_s32_get(
    const SecantAstInstruction* instruction
) {
    return (int32_t)secant_ast_constant_bits_get(instruction);
}

/**
 * Returns the u32 value stored by a constant.
 *
 * @param[in] instruction Pointer to a five-byte u32 constant instruction.
 * @return The encoded unsigned 32-bit value.
 */
static inline uint32_t secant_ast_constant_u32_get(
    const SecantAstInstruction* instruction
) {
    return secant_ast_constant_bits_get(instruction);
}

/**
 * @name AST initializer-list encoders
 *
 * These macros encode complete instructions as comma-separated byte
 * initializers. Payload-bearing encoders expand to multiple comma-separated
 * bytes and are intended for uint8_t array initializers. Runtime builders
 * should write the same byte layout directly. Input, routine, and
 * routine-argument indices must fit in uint8_t; f32 input indices must also be
 * less than SECANT_AST_MAX_INPUTS. The f32 constant encoder takes
 * object-representation bits, such as SECANT_F32_BITS_ONE.
 *
 * Example:
 * @code{.c}
 * static const SecantAstInstruction add_one[] = {
 *     secant_ast_encode_column_f32(0),
 *     secant_ast_encode_constant_f32_bits(SECANT_F32_BITS_ONE),
 *     secant_ast_encode_add_f32,
 *     secant_ast_encode_return_f32
 * };
 * @endcode
 * @{
 */
#define SECANT_AST_ENCODE_U8(type_, value_) ((uint8_t)(type_)), ((uint8_t)(value_))
#define SECANT_AST_ENCODE_U32(type_, value_) \
    ((uint8_t)(type_)), ((uint8_t)((uint32_t)(value_) >> 0u)), ((uint8_t)((uint32_t)(value_) >> 8u)), \
    ((uint8_t)((uint32_t)(value_) >> 16u)), ((uint8_t)((uint32_t)(value_) >> 24u))
#define secant_ast_encode_column_f32(input_idx_) \
    SECANT_AST_ENCODE_U8(SECANT_AST_INSTRUCTION_TYPE_COLUMN_F32, input_idx_)
#define secant_ast_encode_input_s32(input_idx_) SECANT_AST_ENCODE_U8(SECANT_AST_INSTRUCTION_TYPE_INPUT_S32, input_idx_)
#define secant_ast_encode_input_u32(input_idx_) SECANT_AST_ENCODE_U8(SECANT_AST_INSTRUCTION_TYPE_INPUT_U32, input_idx_)
#define secant_ast_encode_constant_f32_bits(bits_) SECANT_AST_ENCODE_U32(SECANT_AST_INSTRUCTION_TYPE_CONSTANT_BITS_F32, bits_)
#define secant_ast_encode_constant_s32(value_) \
    SECANT_AST_ENCODE_U32(SECANT_AST_INSTRUCTION_TYPE_CONSTANT_S32, (uint32_t)(int32_t)(value_))
#define secant_ast_encode_constant_u32(value_) SECANT_AST_ENCODE_U32(SECANT_AST_INSTRUCTION_TYPE_CONSTANT_U32, value_)
#define secant_ast_encode_routine_f32(routine_idx_) SECANT_AST_ENCODE_U8(SECANT_AST_INSTRUCTION_TYPE_ROUTINE_F32, routine_idx_)
#define secant_ast_encode_routine_s32(routine_idx_) SECANT_AST_ENCODE_U8(SECANT_AST_INSTRUCTION_TYPE_ROUTINE_S32, routine_idx_)
#define secant_ast_encode_routine_u32(routine_idx_) SECANT_AST_ENCODE_U8(SECANT_AST_INSTRUCTION_TYPE_ROUTINE_U32, routine_idx_)
#define secant_ast_encode_routine_arg_f32(arg_idx_) SECANT_AST_ENCODE_U8(SECANT_AST_INSTRUCTION_TYPE_ROUTINE_ARG_F32, arg_idx_)
#define secant_ast_encode_routine_arg_s32(arg_idx_) SECANT_AST_ENCODE_U8(SECANT_AST_INSTRUCTION_TYPE_ROUTINE_ARG_S32, arg_idx_)
#define secant_ast_encode_routine_arg_u32(arg_idx_) SECANT_AST_ENCODE_U8(SECANT_AST_INSTRUCTION_TYPE_ROUTINE_ARG_U32, arg_idx_)
#define secant_ast_encode_add_f32 ((uint8_t)SECANT_AST_INSTRUCTION_TYPE_ADD_F32)
#define secant_ast_encode_sub_f32 ((uint8_t)SECANT_AST_INSTRUCTION_TYPE_SUB_F32)
#define secant_ast_encode_mul_f32 ((uint8_t)SECANT_AST_INSTRUCTION_TYPE_MUL_F32)
#define secant_ast_encode_div_f32 ((uint8_t)SECANT_AST_INSTRUCTION_TYPE_DIV_F32)
#define secant_ast_encode_neg_f32 ((uint8_t)SECANT_AST_INSTRUCTION_TYPE_NEG_F32)
#define secant_ast_encode_sqrt_f32 ((uint8_t)SECANT_AST_INSTRUCTION_TYPE_SQRT_F32)
#define secant_ast_encode_rcp_f32 ((uint8_t)SECANT_AST_INSTRUCTION_TYPE_RCP_F32)
#define secant_ast_encode_abs_f32 ((uint8_t)SECANT_AST_INSTRUCTION_TYPE_ABS_F32)
#define secant_ast_encode_min_f32 ((uint8_t)SECANT_AST_INSTRUCTION_TYPE_MIN_F32)
#define secant_ast_encode_max_f32 ((uint8_t)SECANT_AST_INSTRUCTION_TYPE_MAX_F32)
#define secant_ast_encode_fma_f32 ((uint8_t)SECANT_AST_INSTRUCTION_TYPE_FMA_F32)
#define secant_ast_encode_sin_f32 ((uint8_t)SECANT_AST_INSTRUCTION_TYPE_SIN_F32)
#define secant_ast_encode_cos_f32 ((uint8_t)SECANT_AST_INSTRUCTION_TYPE_COS_F32)
#define secant_ast_encode_ex2_f32 ((uint8_t)SECANT_AST_INSTRUCTION_TYPE_EX2_F32)
#define secant_ast_encode_lg2_f32 ((uint8_t)SECANT_AST_INSTRUCTION_TYPE_LG2_F32)
#define secant_ast_encode_rsqrt_f32 ((uint8_t)SECANT_AST_INSTRUCTION_TYPE_RSQRT_F32)
#define secant_ast_encode_tanh_f32 ((uint8_t)SECANT_AST_INSTRUCTION_TYPE_TANH_F32)
#define secant_ast_encode_exp_f32 ((uint8_t)SECANT_AST_INSTRUCTION_TYPE_EXP_F32)
#define secant_ast_encode_log_f32 ((uint8_t)SECANT_AST_INSTRUCTION_TYPE_LOG_F32)
#define secant_ast_encode_return_f32 ((uint8_t)SECANT_AST_INSTRUCTION_TYPE_RETURN_F32)
#define secant_ast_encode_add_s32 ((uint8_t)SECANT_AST_INSTRUCTION_TYPE_ADD_S32)
#define secant_ast_encode_sub_s32 ((uint8_t)SECANT_AST_INSTRUCTION_TYPE_SUB_S32)
#define secant_ast_encode_mul_s32 ((uint8_t)SECANT_AST_INSTRUCTION_TYPE_MUL_S32)
#define secant_ast_encode_div_s32 ((uint8_t)SECANT_AST_INSTRUCTION_TYPE_DIV_S32)
#define secant_ast_encode_neg_s32 ((uint8_t)SECANT_AST_INSTRUCTION_TYPE_NEG_S32)
#define secant_ast_encode_abs_s32 ((uint8_t)SECANT_AST_INSTRUCTION_TYPE_ABS_S32)
#define secant_ast_encode_min_s32 ((uint8_t)SECANT_AST_INSTRUCTION_TYPE_MIN_S32)
#define secant_ast_encode_max_s32 ((uint8_t)SECANT_AST_INSTRUCTION_TYPE_MAX_S32)
#define secant_ast_encode_fma_s32 ((uint8_t)SECANT_AST_INSTRUCTION_TYPE_FMA_S32)
#define secant_ast_encode_return_s32 ((uint8_t)SECANT_AST_INSTRUCTION_TYPE_RETURN_S32)
#define secant_ast_encode_add_u32 ((uint8_t)SECANT_AST_INSTRUCTION_TYPE_ADD_U32)
#define secant_ast_encode_sub_u32 ((uint8_t)SECANT_AST_INSTRUCTION_TYPE_SUB_U32)
#define secant_ast_encode_mul_u32 ((uint8_t)SECANT_AST_INSTRUCTION_TYPE_MUL_U32)
#define secant_ast_encode_div_u32 ((uint8_t)SECANT_AST_INSTRUCTION_TYPE_DIV_U32)
#define secant_ast_encode_min_u32 ((uint8_t)SECANT_AST_INSTRUCTION_TYPE_MIN_U32)
#define secant_ast_encode_max_u32 ((uint8_t)SECANT_AST_INSTRUCTION_TYPE_MAX_U32)
#define secant_ast_encode_fma_u32 ((uint8_t)SECANT_AST_INSTRUCTION_TYPE_FMA_U32)
#define secant_ast_encode_and_u32 ((uint8_t)SECANT_AST_INSTRUCTION_TYPE_AND_U32)
#define secant_ast_encode_or_u32 ((uint8_t)SECANT_AST_INSTRUCTION_TYPE_OR_U32)
#define secant_ast_encode_xor_u32 ((uint8_t)SECANT_AST_INSTRUCTION_TYPE_XOR_U32)
#define secant_ast_encode_not_u32 ((uint8_t)SECANT_AST_INSTRUCTION_TYPE_NOT_U32)
#define secant_ast_encode_shl_u32 ((uint8_t)SECANT_AST_INSTRUCTION_TYPE_SHL_U32)
#define secant_ast_encode_shr_u32 ((uint8_t)SECANT_AST_INSTRUCTION_TYPE_SHR_U32)
#define secant_ast_encode_return_u32 ((uint8_t)SECANT_AST_INSTRUCTION_TYPE_RETURN_U32)

/** @} */

/** Toggles consume two/four preceding direct leaves in listed order.
 * bit0 is the low selection bit. Reusing bits couples sites. */
#define secant_ast_encode_bank_constant_f32(idx) SECANT_AST_INSTRUCTION_TYPE_BANK_CONSTANT_F32, (uint8_t)(idx)
#define secant_ast_encode_toggle2_f32(bit) SECANT_AST_INSTRUCTION_TYPE_TOGGLE2_F32, (uint8_t)(bit)
#define secant_ast_encode_toggle4_f32(bit0, bit1) SECANT_AST_INSTRUCTION_TYPE_TOGGLE4_F32, (uint8_t)(bit0), (uint8_t)(bit1)

#endif /* SECANT_INSTRUCTIONS_H_INCLUDED */
