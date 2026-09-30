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
#ifndef CUSR_AST_H_INCLUDED
#define CUSR_AST_H_INCLUDED

#include <stdint.h>

#if defined(__cplusplus)
#if __cplusplus >= 202002L
#include <bit>
#else
#include <cstring>
#endif
#endif

#ifndef CUSR_AST_MAX_PROGRAM_INSTRUCTIONS
#define CUSR_AST_MAX_PROGRAM_INSTRUCTIONS 1024u
#endif

#ifndef CUSR_AST_MAX_INSTRUCTION_ARGS
#define CUSR_AST_MAX_INSTRUCTION_ARGS 4u
#endif

typedef uint32_t CusrAstIdx;

typedef enum CusrAstInstructionType {
    CUSR_AST_INSTRUCTION_TYPE_NONE = 0,
    CUSR_AST_INSTRUCTION_TYPE_INPUT = 1,
    CUSR_AST_INSTRUCTION_TYPE_CONSTANT_BITS = 2,
    CUSR_AST_INSTRUCTION_TYPE_OP = 3,
    CUSR_AST_INSTRUCTION_TYPE_ROUTINE = 4,
    CUSR_AST_INSTRUCTION_TYPE_RETURN = 5,
    CUSR_AST_INSTRUCTION_TYPE_ROUTINE_ARG = 6
} CusrAstInstructionType;

typedef enum CusrAstOp {
    CUSR_AST_OP_NONE = 0,
    CUSR_AST_OP_ADD = 1,
    CUSR_AST_OP_SUB = 2,
    CUSR_AST_OP_MUL = 3,
    CUSR_AST_OP_DIV = 4,
    CUSR_AST_OP_NEG = 5,
    CUSR_AST_OP_SQRT = 6,
    CUSR_AST_OP_RCP = 7,
    CUSR_AST_OP_ABS = 8,
    CUSR_AST_OP_MIN = 9,
    CUSR_AST_OP_MAX = 10,
    CUSR_AST_OP_FMA = 11,
    CUSR_AST_OP_SIN = 12,
    CUSR_AST_OP_COS = 13,
    CUSR_AST_OP_EX2 = 14,
    CUSR_AST_OP_LG2 = 15,
    CUSR_AST_OP_RSQRT = 16,
    CUSR_AST_OP_TANH = 17,
    CUSR_AST_OP_NUM_ENUMS = 18
} CusrAstOp;

typedef union CusrAstPayload {
    CusrAstIdx idx;
    uint32_t bits;
    uint32_t op;
    float f;
} CusrAstPayload;

typedef struct CusrAstInstruction {
    uint16_t instruction_type;
    uint16_t aux;
    CusrAstPayload payload;
} CusrAstInstruction;

#ifdef __cplusplus
#if __cplusplus >= 202002L
#define CUSR_AST_CPP_FUNC constexpr
#define CUSR_AST_CPP_CONST constexpr
#else
#define CUSR_AST_CPP_FUNC inline
#define CUSR_AST_CPP_CONST const
#endif

static CUSR_AST_CPP_FUNC CusrAstPayload cusr_ast_payload_idx(CusrAstIdx value) { CusrAstPayload payload = {}; payload.idx = value; return payload; }
static CUSR_AST_CPP_FUNC CusrAstPayload cusr_ast_payload_bits(uint32_t value) { CusrAstPayload payload = {}; payload.bits = value; return payload; }
static CUSR_AST_CPP_FUNC CusrAstPayload cusr_ast_payload_op(CusrAstOp value) { CusrAstPayload payload = {}; payload.op = (uint32_t)value; return payload; }
#if __cplusplus >= 202002L
static constexpr uint32_t cusr_ast_float_bits(float value) { return std::bit_cast<uint32_t>(value); }
#else
static inline uint32_t cusr_ast_float_bits(float value) { uint32_t bits; std::memcpy(&bits, &value, sizeof(bits)); return bits; }
#endif
static CUSR_AST_CPP_FUNC CusrAstInstruction cusr_ast_make_instruction(uint16_t instruction_type, uint16_t aux, CusrAstPayload payload) { return CusrAstInstruction{ instruction_type, aux, payload }; }
static CUSR_AST_CPP_FUNC CusrAstInstruction cusr_ast_encode_input(CusrAstIdx input_idx) { return cusr_ast_make_instruction(CUSR_AST_INSTRUCTION_TYPE_INPUT, 0u, cusr_ast_payload_idx(input_idx)); }
static CUSR_AST_CPP_FUNC CusrAstInstruction cusr_ast_encode_constant_bits(uint32_t bits) { return cusr_ast_make_instruction(CUSR_AST_INSTRUCTION_TYPE_CONSTANT_BITS, 0u, cusr_ast_payload_bits(bits)); }
static CUSR_AST_CPP_FUNC CusrAstInstruction cusr_ast_encode_constant(float value) { return cusr_ast_encode_constant_bits(cusr_ast_float_bits(value)); }
static CUSR_AST_CPP_FUNC CusrAstInstruction cusr_ast_encode_op(CusrAstOp op, uint16_t num_args) { return cusr_ast_make_instruction(CUSR_AST_INSTRUCTION_TYPE_OP, num_args, cusr_ast_payload_op(op)); }
static CUSR_AST_CPP_FUNC CusrAstInstruction cusr_ast_encode_routine(CusrAstIdx routine_idx, uint16_t num_args) { return cusr_ast_make_instruction(CUSR_AST_INSTRUCTION_TYPE_ROUTINE, num_args, cusr_ast_payload_idx(routine_idx)); }
static CUSR_AST_CPP_FUNC CusrAstInstruction cusr_ast_encode_routine_arg(CusrAstIdx arg_idx) { return cusr_ast_make_instruction(CUSR_AST_INSTRUCTION_TYPE_ROUTINE_ARG, 0u, cusr_ast_payload_idx(arg_idx)); }
static CUSR_AST_CPP_CONST CusrAstInstruction cusr_ast_encode_add = cusr_ast_encode_op(CUSR_AST_OP_ADD, 2u);
static CUSR_AST_CPP_CONST CusrAstInstruction cusr_ast_encode_sub = cusr_ast_encode_op(CUSR_AST_OP_SUB, 2u);
static CUSR_AST_CPP_CONST CusrAstInstruction cusr_ast_encode_mul = cusr_ast_encode_op(CUSR_AST_OP_MUL, 2u);
static CUSR_AST_CPP_CONST CusrAstInstruction cusr_ast_encode_div = cusr_ast_encode_op(CUSR_AST_OP_DIV, 2u);
static CUSR_AST_CPP_CONST CusrAstInstruction cusr_ast_encode_neg = cusr_ast_encode_op(CUSR_AST_OP_NEG, 1u);
static CUSR_AST_CPP_CONST CusrAstInstruction cusr_ast_encode_sqrt = cusr_ast_encode_op(CUSR_AST_OP_SQRT, 1u);
static CUSR_AST_CPP_CONST CusrAstInstruction cusr_ast_encode_rcp = cusr_ast_encode_op(CUSR_AST_OP_RCP, 1u);
static CUSR_AST_CPP_CONST CusrAstInstruction cusr_ast_encode_abs = cusr_ast_encode_op(CUSR_AST_OP_ABS, 1u);
static CUSR_AST_CPP_CONST CusrAstInstruction cusr_ast_encode_min = cusr_ast_encode_op(CUSR_AST_OP_MIN, 2u);
static CUSR_AST_CPP_CONST CusrAstInstruction cusr_ast_encode_max = cusr_ast_encode_op(CUSR_AST_OP_MAX, 2u);
static CUSR_AST_CPP_CONST CusrAstInstruction cusr_ast_encode_fma = cusr_ast_encode_op(CUSR_AST_OP_FMA, 3u);
static CUSR_AST_CPP_CONST CusrAstInstruction cusr_ast_encode_sin = cusr_ast_encode_op(CUSR_AST_OP_SIN, 1u);
static CUSR_AST_CPP_CONST CusrAstInstruction cusr_ast_encode_cos = cusr_ast_encode_op(CUSR_AST_OP_COS, 1u);
static CUSR_AST_CPP_CONST CusrAstInstruction cusr_ast_encode_ex2 = cusr_ast_encode_op(CUSR_AST_OP_EX2, 1u);
static CUSR_AST_CPP_CONST CusrAstInstruction cusr_ast_encode_lg2 = cusr_ast_encode_op(CUSR_AST_OP_LG2, 1u);
static CUSR_AST_CPP_CONST CusrAstInstruction cusr_ast_encode_rsqrt = cusr_ast_encode_op(CUSR_AST_OP_RSQRT, 1u);
static CUSR_AST_CPP_CONST CusrAstInstruction cusr_ast_encode_tanh = cusr_ast_encode_op(CUSR_AST_OP_TANH, 1u);
static CUSR_AST_CPP_CONST CusrAstInstruction cusr_ast_encode_return = cusr_ast_make_instruction(CUSR_AST_INSTRUCTION_TYPE_RETURN, 0u, cusr_ast_payload_bits(0u));

#undef CUSR_AST_CPP_CONST
#undef CUSR_AST_CPP_FUNC
#else
#define cusr_ast_encode_input(input_idx_) ((CusrAstInstruction){ CUSR_AST_INSTRUCTION_TYPE_INPUT, 0u, { .idx = (CusrAstIdx)(input_idx_) } })
#define cusr_ast_encode_constant_bits(bits_) ((CusrAstInstruction){ CUSR_AST_INSTRUCTION_TYPE_CONSTANT_BITS, 0u, { .bits = (uint32_t)(bits_) } })
#define cusr_ast_encode_constant(value) ((CusrAstInstruction){ CUSR_AST_INSTRUCTION_TYPE_CONSTANT_BITS, 0u, { .f = (float)(value) } })
#define cusr_ast_encode_op(op_, num_args_) ((CusrAstInstruction){ CUSR_AST_INSTRUCTION_TYPE_OP, (uint16_t)(num_args_), { .op = (uint32_t)(op_) } })
#define cusr_ast_encode_add cusr_ast_encode_op(CUSR_AST_OP_ADD, 2u)
#define cusr_ast_encode_sub cusr_ast_encode_op(CUSR_AST_OP_SUB, 2u)
#define cusr_ast_encode_mul cusr_ast_encode_op(CUSR_AST_OP_MUL, 2u)
#define cusr_ast_encode_div cusr_ast_encode_op(CUSR_AST_OP_DIV, 2u)
#define cusr_ast_encode_neg cusr_ast_encode_op(CUSR_AST_OP_NEG, 1u)
#define cusr_ast_encode_sqrt cusr_ast_encode_op(CUSR_AST_OP_SQRT, 1u)
#define cusr_ast_encode_rcp cusr_ast_encode_op(CUSR_AST_OP_RCP, 1u)
#define cusr_ast_encode_abs cusr_ast_encode_op(CUSR_AST_OP_ABS, 1u)
#define cusr_ast_encode_min cusr_ast_encode_op(CUSR_AST_OP_MIN, 2u)
#define cusr_ast_encode_max cusr_ast_encode_op(CUSR_AST_OP_MAX, 2u)
#define cusr_ast_encode_fma cusr_ast_encode_op(CUSR_AST_OP_FMA, 3u)
#define cusr_ast_encode_sin cusr_ast_encode_op(CUSR_AST_OP_SIN, 1u)
#define cusr_ast_encode_cos cusr_ast_encode_op(CUSR_AST_OP_COS, 1u)
#define cusr_ast_encode_ex2 cusr_ast_encode_op(CUSR_AST_OP_EX2, 1u)
#define cusr_ast_encode_lg2 cusr_ast_encode_op(CUSR_AST_OP_LG2, 1u)
#define cusr_ast_encode_rsqrt cusr_ast_encode_op(CUSR_AST_OP_RSQRT, 1u)
#define cusr_ast_encode_tanh cusr_ast_encode_op(CUSR_AST_OP_TANH, 1u)
#define cusr_ast_encode_routine(routine_idx_, num_args_) ((CusrAstInstruction){ CUSR_AST_INSTRUCTION_TYPE_ROUTINE, (uint16_t)(num_args_), { .idx = (CusrAstIdx)(routine_idx_) } })
#define cusr_ast_encode_routine_arg(arg_idx_) ((CusrAstInstruction){ CUSR_AST_INSTRUCTION_TYPE_ROUTINE_ARG, 0u, { .idx = (CusrAstIdx)(arg_idx_) } })
#define cusr_ast_encode_return ((CusrAstInstruction){ CUSR_AST_INSTRUCTION_TYPE_RETURN, 0u, { .bits = 0u } })
#endif

#endif /* CUSR_AST_H_INCLUDED */
