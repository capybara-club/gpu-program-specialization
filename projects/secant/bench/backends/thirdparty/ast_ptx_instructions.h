/*
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
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @file
 * @brief Concrete f32 instruction catalog and encoders for AST PTX.
 *
 * This header keeps AST PTX's f32 PTX instruction names, arities, and public
 * encode helpers separate from the generic compiler in ast_ptx.h.
 */

#ifndef AST_PTX_INSTRUCTIONS_H_INCLUDE
#define AST_PTX_INSTRUCTIONS_H_INCLUDE

#include "ast_ptx.h"

#ifdef __cplusplus

/**
 * \brief Encodes an input leaf by input-register index.
 */
constexpr AstPtxInstruction ast_ptx_encode_input(AstPtxIdx input_idx) {
    AstPtxInstruction instruction{};
    instruction.instruction_type = AST_PTX_INSTRUCTION_TYPE_INPUT;
    instruction.aux = 0u;
    instruction.payload.idx = input_idx;
    return instruction;
}

/**
 * \brief Encodes a routine-local argument leaf by argument index.
 */
constexpr AstPtxInstruction ast_ptx_encode_routine_arg(AstPtxIdx arg_idx) {
    AstPtxInstruction instruction{};
    instruction.instruction_type = AST_PTX_INSTRUCTION_TYPE_ROUTINE_ARG;
    instruction.aux = 0u;
    instruction.payload.idx = arg_idx;
    return instruction;
}

/**
 * \brief Encodes an f32 immediate constant leaf.
 */
constexpr AstPtxInstruction ast_ptx_encode_constant(float c) {
    AstPtxInstruction instruction{};
    instruction.instruction_type = AST_PTX_INSTRUCTION_TYPE_CONSTANT;
    instruction.aux = 0u;
    instruction.payload.f = c;
    return instruction;
}

/**
 * \brief Encodes a concrete PTX instruction and its f32 argument count.
 */
constexpr AstPtxInstruction ast_ptx_encode_ptx_instruction(
    AstPtxIdx instruction_idx,
    uint16_t num_args
) {
    AstPtxInstruction instruction{};
    instruction.instruction_type = AST_PTX_INSTRUCTION_TYPE_PTX_INSTRUCTION;
    instruction.aux = num_args;
    instruction.payload.idx = instruction_idx;
    return instruction;
}

/**
 * \brief Encodes a routine call and the number of f32 values it consumes.
 */
constexpr AstPtxInstruction ast_ptx_encode_routine(
    AstPtxIdx routine_idx,
    uint16_t num_args
) {
    AstPtxInstruction instruction{};
    instruction.instruction_type = AST_PTX_INSTRUCTION_TYPE_ROUTINE;
    instruction.aux = num_args;
    instruction.payload.idx = routine_idx;
    return instruction;
}

/**
 * \brief Encodes an internal generated-register value.
 */
constexpr AstPtxInstruction ast_ptx_encode_register(AstPtxIdx register_idx) {
    AstPtxInstruction instruction{};
    instruction.instruction_type = AST_PTX_INSTRUCTION_TYPE_REGISTER;
    instruction.aux = 0u;
    instruction.payload.idx = register_idx;
    return instruction;
}

/**
 * \brief Encodes the return sentinel for a program or routine.
 */
constexpr AstPtxInstruction _ast_ptx_encode_return() {
    AstPtxInstruction instruction{};
    instruction.instruction_type = AST_PTX_INSTRUCTION_TYPE_RETURN;
    instruction.aux = 0u;
    instruction.payload.idx = 0u;
    return instruction;
}

/**
 * \brief Return sentinel instruction for C++ users.
 */
static const AstPtxInstruction ast_ptx_encode_return = _ast_ptx_encode_return();

#else

/**
 * \brief Encodes an input leaf by input-register index.
 */
#define ast_ptx_encode_input(input_idx) ((AstPtxInstruction){ .instruction_type = AST_PTX_INSTRUCTION_TYPE_INPUT, .aux = 0u, .payload = { .idx = (AstPtxIdx)(input_idx) } })
/**
 * \brief Encodes a routine-local argument leaf by argument index.
 */
#define ast_ptx_encode_routine_arg(arg_idx) ((AstPtxInstruction){ .instruction_type = AST_PTX_INSTRUCTION_TYPE_ROUTINE_ARG, .aux = 0u, .payload = { .idx = (AstPtxIdx)(arg_idx) } })
/**
 * \brief Encodes an f32 immediate constant leaf.
 */
#define ast_ptx_encode_constant(c) ((AstPtxInstruction){ .instruction_type = AST_PTX_INSTRUCTION_TYPE_CONSTANT, .aux = 0u, .payload = { .f = (float)(c) } })
/**
 * \brief Encodes a concrete PTX instruction and its f32 argument count.
 */
#define ast_ptx_encode_ptx_instruction(instruction_idx, num_args) ((AstPtxInstruction){ .instruction_type = AST_PTX_INSTRUCTION_TYPE_PTX_INSTRUCTION, .aux = (uint16_t)(num_args), .payload = { .idx = (AstPtxIdx)(instruction_idx) } })
/**
 * \brief Encodes a routine call and the number of f32 values it consumes.
 */
#define ast_ptx_encode_routine(routine_idx, num_args) ((AstPtxInstruction){ .instruction_type = AST_PTX_INSTRUCTION_TYPE_ROUTINE, .aux = (uint16_t)(num_args), .payload = { .idx = (AstPtxIdx)(routine_idx) } })
/**
 * \brief Encodes an internal generated-register value.
 */
#define ast_ptx_encode_register(register_idx) ((AstPtxInstruction){ .instruction_type = AST_PTX_INSTRUCTION_TYPE_REGISTER, .aux = 0u, .payload = { .idx = (AstPtxIdx)(register_idx) } })
/**
 * \brief Return sentinel instruction for C users.
 */
#define ast_ptx_encode_return ((AstPtxInstruction){ .instruction_type = AST_PTX_INSTRUCTION_TYPE_RETURN, .aux = 0u, .payload = { .idx = 0u } })

#endif

/**
 * \brief Built-in f32 PTX instructions supported by ast_ptx_interpreter.h.
 *
 * \details ast_ptx.h does not depend on this enum directly; callers pass the
 * corresponding PTX name table and instruction count into ast_ptx_compile.
 */
typedef enum {
    AST_PTX_PTX_INSTRUCTION_COPYSIGN_F32 = 0,
    AST_PTX_PTX_INSTRUCTION_ADD_FTZ_F32 = 1,
    AST_PTX_PTX_INSTRUCTION_SUB_FTZ_F32 = 2,
    AST_PTX_PTX_INSTRUCTION_MUL_FTZ_F32 = 3,
    AST_PTX_PTX_INSTRUCTION_FMA_RN_FTZ_F32 = 4,
    AST_PTX_PTX_INSTRUCTION_DIV_APPROX_FTZ_F32 = 5,
    AST_PTX_PTX_INSTRUCTION_ABS_FTZ_F32 = 6,
    AST_PTX_PTX_INSTRUCTION_NEG_FTZ_F32 = 7,
    AST_PTX_PTX_INSTRUCTION_MIN_FTZ_F32 = 8,
    AST_PTX_PTX_INSTRUCTION_MAX_FTZ_F32 = 9,
    AST_PTX_PTX_INSTRUCTION_RCP_APPROX_FTZ_F32 = 10,
    AST_PTX_PTX_INSTRUCTION_SQRT_APPROX_FTZ_F32 = 11,
    AST_PTX_PTX_INSTRUCTION_RSQRT_APPROX_FTZ_F32 = 12,
    AST_PTX_PTX_INSTRUCTION_SIN_APPROX_FTZ_F32 = 13,
    AST_PTX_PTX_INSTRUCTION_COS_APPROX_FTZ_F32 = 14,
    AST_PTX_PTX_INSTRUCTION_LG2_APPROX_FTZ_F32 = 15,
    AST_PTX_PTX_INSTRUCTION_EX2_APPROX_FTZ_F32 = 16,
    AST_PTX_PTX_INSTRUCTION_TANH_APPROX_F32 = 17,
    AST_PTX_PTX_INSTRUCTION_NUM_ENUMS = 18
} AstPtxPtxInstruction;

/**
 * \brief PTX instruction strings indexed by AstPtxPtxInstruction.
 */
#ifdef __cplusplus
static constexpr const char* const ast_ptx_ptx_instruction_names[AST_PTX_PTX_INSTRUCTION_NUM_ENUMS] = {
#else
static const char* const ast_ptx_ptx_instruction_names[AST_PTX_PTX_INSTRUCTION_NUM_ENUMS] = {
#endif
    "copysign.f32",        /* AST_PTX_PTX_INSTRUCTION_COPYSIGN_F32 */
    "add.ftz.f32",         /* AST_PTX_PTX_INSTRUCTION_ADD_FTZ_F32 */
    "sub.ftz.f32",         /* AST_PTX_PTX_INSTRUCTION_SUB_FTZ_F32 */
    "mul.ftz.f32",         /* AST_PTX_PTX_INSTRUCTION_MUL_FTZ_F32 */
    "fma.rn.ftz.f32",      /* AST_PTX_PTX_INSTRUCTION_FMA_RN_FTZ_F32 */
    "div.approx.ftz.f32",  /* AST_PTX_PTX_INSTRUCTION_DIV_APPROX_FTZ_F32 */
    "abs.ftz.f32",         /* AST_PTX_PTX_INSTRUCTION_ABS_FTZ_F32 */
    "neg.ftz.f32",         /* AST_PTX_PTX_INSTRUCTION_NEG_FTZ_F32 */
    "min.ftz.f32",         /* AST_PTX_PTX_INSTRUCTION_MIN_FTZ_F32 */
    "max.ftz.f32",         /* AST_PTX_PTX_INSTRUCTION_MAX_FTZ_F32 */
    "rcp.approx.ftz.f32",  /* AST_PTX_PTX_INSTRUCTION_RCP_APPROX_FTZ_F32 */
    "sqrt.approx.ftz.f32", /* AST_PTX_PTX_INSTRUCTION_SQRT_APPROX_FTZ_F32 */
    "rsqrt.approx.ftz.f32",/* AST_PTX_PTX_INSTRUCTION_RSQRT_APPROX_FTZ_F32 */
    "sin.approx.ftz.f32",  /* AST_PTX_PTX_INSTRUCTION_SIN_APPROX_FTZ_F32 */
    "cos.approx.ftz.f32",  /* AST_PTX_PTX_INSTRUCTION_COS_APPROX_FTZ_F32 */
    "lg2.approx.ftz.f32",  /* AST_PTX_PTX_INSTRUCTION_LG2_APPROX_FTZ_F32 */
    "ex2.approx.ftz.f32",  /* AST_PTX_PTX_INSTRUCTION_EX2_APPROX_FTZ_F32 */
    "tanh.approx.f32"      /* AST_PTX_PTX_INSTRUCTION_TANH_APPROX_F32 */
};

/**
 * \brief Number of f32 operands consumed by each AstPtxPtxInstruction.
 */
#ifdef __cplusplus
static constexpr uint16_t ast_ptx_ptx_instruction_num_args[AST_PTX_PTX_INSTRUCTION_NUM_ENUMS] = {
#else
static const uint16_t ast_ptx_ptx_instruction_num_args[AST_PTX_PTX_INSTRUCTION_NUM_ENUMS] = {
#endif
    2u, /* AST_PTX_PTX_INSTRUCTION_COPYSIGN_F32 */
    2u, /* AST_PTX_PTX_INSTRUCTION_ADD_FTZ_F32 */
    2u, /* AST_PTX_PTX_INSTRUCTION_SUB_FTZ_F32 */
    2u, /* AST_PTX_PTX_INSTRUCTION_MUL_FTZ_F32 */
    3u, /* AST_PTX_PTX_INSTRUCTION_FMA_RN_FTZ_F32 */
    2u, /* AST_PTX_PTX_INSTRUCTION_DIV_APPROX_FTZ_F32 */
    1u, /* AST_PTX_PTX_INSTRUCTION_ABS_FTZ_F32 */
    1u, /* AST_PTX_PTX_INSTRUCTION_NEG_FTZ_F32 */
    2u, /* AST_PTX_PTX_INSTRUCTION_MIN_FTZ_F32 */
    2u, /* AST_PTX_PTX_INSTRUCTION_MAX_FTZ_F32 */
    1u, /* AST_PTX_PTX_INSTRUCTION_RCP_APPROX_FTZ_F32 */
    1u, /* AST_PTX_PTX_INSTRUCTION_SQRT_APPROX_FTZ_F32 */
    1u, /* AST_PTX_PTX_INSTRUCTION_RSQRT_APPROX_FTZ_F32 */
    1u, /* AST_PTX_PTX_INSTRUCTION_SIN_APPROX_FTZ_F32 */
    1u, /* AST_PTX_PTX_INSTRUCTION_COS_APPROX_FTZ_F32 */
    1u, /* AST_PTX_PTX_INSTRUCTION_LG2_APPROX_FTZ_F32 */
    1u, /* AST_PTX_PTX_INSTRUCTION_EX2_APPROX_FTZ_F32 */
    1u  /* AST_PTX_PTX_INSTRUCTION_TANH_APPROX_F32 */
};

#ifdef __cplusplus

/**
 * \brief Convenience encodes for built-in f32 PTX instructions.
 */
static const AstPtxInstruction ast_ptx_encode_ptx_instruction_copysign_f32 = ast_ptx_encode_ptx_instruction(AST_PTX_PTX_INSTRUCTION_COPYSIGN_F32, ast_ptx_ptx_instruction_num_args[AST_PTX_PTX_INSTRUCTION_COPYSIGN_F32]);
static const AstPtxInstruction ast_ptx_encode_ptx_instruction_add_ftz_f32 = ast_ptx_encode_ptx_instruction(AST_PTX_PTX_INSTRUCTION_ADD_FTZ_F32, ast_ptx_ptx_instruction_num_args[AST_PTX_PTX_INSTRUCTION_ADD_FTZ_F32]);
static const AstPtxInstruction ast_ptx_encode_ptx_instruction_sub_ftz_f32 = ast_ptx_encode_ptx_instruction(AST_PTX_PTX_INSTRUCTION_SUB_FTZ_F32, ast_ptx_ptx_instruction_num_args[AST_PTX_PTX_INSTRUCTION_SUB_FTZ_F32]);
static const AstPtxInstruction ast_ptx_encode_ptx_instruction_mul_ftz_f32 = ast_ptx_encode_ptx_instruction(AST_PTX_PTX_INSTRUCTION_MUL_FTZ_F32, ast_ptx_ptx_instruction_num_args[AST_PTX_PTX_INSTRUCTION_MUL_FTZ_F32]);
static const AstPtxInstruction ast_ptx_encode_ptx_instruction_fma_rn_ftz_f32 = ast_ptx_encode_ptx_instruction(AST_PTX_PTX_INSTRUCTION_FMA_RN_FTZ_F32, ast_ptx_ptx_instruction_num_args[AST_PTX_PTX_INSTRUCTION_FMA_RN_FTZ_F32]);
static const AstPtxInstruction ast_ptx_encode_ptx_instruction_div_approx_ftz_f32 = ast_ptx_encode_ptx_instruction(AST_PTX_PTX_INSTRUCTION_DIV_APPROX_FTZ_F32, ast_ptx_ptx_instruction_num_args[AST_PTX_PTX_INSTRUCTION_DIV_APPROX_FTZ_F32]);
static const AstPtxInstruction ast_ptx_encode_ptx_instruction_abs_ftz_f32 = ast_ptx_encode_ptx_instruction(AST_PTX_PTX_INSTRUCTION_ABS_FTZ_F32, ast_ptx_ptx_instruction_num_args[AST_PTX_PTX_INSTRUCTION_ABS_FTZ_F32]);
static const AstPtxInstruction ast_ptx_encode_ptx_instruction_neg_ftz_f32 = ast_ptx_encode_ptx_instruction(AST_PTX_PTX_INSTRUCTION_NEG_FTZ_F32, ast_ptx_ptx_instruction_num_args[AST_PTX_PTX_INSTRUCTION_NEG_FTZ_F32]);
static const AstPtxInstruction ast_ptx_encode_ptx_instruction_min_ftz_f32 = ast_ptx_encode_ptx_instruction(AST_PTX_PTX_INSTRUCTION_MIN_FTZ_F32, ast_ptx_ptx_instruction_num_args[AST_PTX_PTX_INSTRUCTION_MIN_FTZ_F32]);
static const AstPtxInstruction ast_ptx_encode_ptx_instruction_max_ftz_f32 = ast_ptx_encode_ptx_instruction(AST_PTX_PTX_INSTRUCTION_MAX_FTZ_F32, ast_ptx_ptx_instruction_num_args[AST_PTX_PTX_INSTRUCTION_MAX_FTZ_F32]);
static const AstPtxInstruction ast_ptx_encode_ptx_instruction_rcp_approx_ftz_f32 = ast_ptx_encode_ptx_instruction(AST_PTX_PTX_INSTRUCTION_RCP_APPROX_FTZ_F32, ast_ptx_ptx_instruction_num_args[AST_PTX_PTX_INSTRUCTION_RCP_APPROX_FTZ_F32]);
static const AstPtxInstruction ast_ptx_encode_ptx_instruction_sqrt_approx_ftz_f32 = ast_ptx_encode_ptx_instruction(AST_PTX_PTX_INSTRUCTION_SQRT_APPROX_FTZ_F32, ast_ptx_ptx_instruction_num_args[AST_PTX_PTX_INSTRUCTION_SQRT_APPROX_FTZ_F32]);
static const AstPtxInstruction ast_ptx_encode_ptx_instruction_rsqrt_approx_ftz_f32 = ast_ptx_encode_ptx_instruction(AST_PTX_PTX_INSTRUCTION_RSQRT_APPROX_FTZ_F32, ast_ptx_ptx_instruction_num_args[AST_PTX_PTX_INSTRUCTION_RSQRT_APPROX_FTZ_F32]);
static const AstPtxInstruction ast_ptx_encode_ptx_instruction_sin_approx_ftz_f32 = ast_ptx_encode_ptx_instruction(AST_PTX_PTX_INSTRUCTION_SIN_APPROX_FTZ_F32, ast_ptx_ptx_instruction_num_args[AST_PTX_PTX_INSTRUCTION_SIN_APPROX_FTZ_F32]);
static const AstPtxInstruction ast_ptx_encode_ptx_instruction_cos_approx_ftz_f32 = ast_ptx_encode_ptx_instruction(AST_PTX_PTX_INSTRUCTION_COS_APPROX_FTZ_F32, ast_ptx_ptx_instruction_num_args[AST_PTX_PTX_INSTRUCTION_COS_APPROX_FTZ_F32]);
static const AstPtxInstruction ast_ptx_encode_ptx_instruction_lg2_approx_ftz_f32 = ast_ptx_encode_ptx_instruction(AST_PTX_PTX_INSTRUCTION_LG2_APPROX_FTZ_F32, ast_ptx_ptx_instruction_num_args[AST_PTX_PTX_INSTRUCTION_LG2_APPROX_FTZ_F32]);
static const AstPtxInstruction ast_ptx_encode_ptx_instruction_ex2_approx_ftz_f32 = ast_ptx_encode_ptx_instruction(AST_PTX_PTX_INSTRUCTION_EX2_APPROX_FTZ_F32, ast_ptx_ptx_instruction_num_args[AST_PTX_PTX_INSTRUCTION_EX2_APPROX_FTZ_F32]);
static const AstPtxInstruction ast_ptx_encode_ptx_instruction_tanh_approx_f32 = ast_ptx_encode_ptx_instruction(AST_PTX_PTX_INSTRUCTION_TANH_APPROX_F32, ast_ptx_ptx_instruction_num_args[AST_PTX_PTX_INSTRUCTION_TANH_APPROX_F32]);

#else

/**
 * \brief Convenience encodes for built-in f32 PTX instructions.
 */
#define ast_ptx_encode_ptx_instruction_copysign_f32 ast_ptx_encode_ptx_instruction(AST_PTX_PTX_INSTRUCTION_COPYSIGN_F32, ast_ptx_ptx_instruction_num_args[AST_PTX_PTX_INSTRUCTION_COPYSIGN_F32])
#define ast_ptx_encode_ptx_instruction_add_ftz_f32 ast_ptx_encode_ptx_instruction(AST_PTX_PTX_INSTRUCTION_ADD_FTZ_F32, ast_ptx_ptx_instruction_num_args[AST_PTX_PTX_INSTRUCTION_ADD_FTZ_F32])
#define ast_ptx_encode_ptx_instruction_sub_ftz_f32 ast_ptx_encode_ptx_instruction(AST_PTX_PTX_INSTRUCTION_SUB_FTZ_F32, ast_ptx_ptx_instruction_num_args[AST_PTX_PTX_INSTRUCTION_SUB_FTZ_F32])
#define ast_ptx_encode_ptx_instruction_mul_ftz_f32 ast_ptx_encode_ptx_instruction(AST_PTX_PTX_INSTRUCTION_MUL_FTZ_F32, ast_ptx_ptx_instruction_num_args[AST_PTX_PTX_INSTRUCTION_MUL_FTZ_F32])
#define ast_ptx_encode_ptx_instruction_fma_rn_ftz_f32 ast_ptx_encode_ptx_instruction(AST_PTX_PTX_INSTRUCTION_FMA_RN_FTZ_F32, ast_ptx_ptx_instruction_num_args[AST_PTX_PTX_INSTRUCTION_FMA_RN_FTZ_F32])
#define ast_ptx_encode_ptx_instruction_div_approx_ftz_f32 ast_ptx_encode_ptx_instruction(AST_PTX_PTX_INSTRUCTION_DIV_APPROX_FTZ_F32, ast_ptx_ptx_instruction_num_args[AST_PTX_PTX_INSTRUCTION_DIV_APPROX_FTZ_F32])
#define ast_ptx_encode_ptx_instruction_abs_ftz_f32 ast_ptx_encode_ptx_instruction(AST_PTX_PTX_INSTRUCTION_ABS_FTZ_F32, ast_ptx_ptx_instruction_num_args[AST_PTX_PTX_INSTRUCTION_ABS_FTZ_F32])
#define ast_ptx_encode_ptx_instruction_neg_ftz_f32 ast_ptx_encode_ptx_instruction(AST_PTX_PTX_INSTRUCTION_NEG_FTZ_F32, ast_ptx_ptx_instruction_num_args[AST_PTX_PTX_INSTRUCTION_NEG_FTZ_F32])
#define ast_ptx_encode_ptx_instruction_min_ftz_f32 ast_ptx_encode_ptx_instruction(AST_PTX_PTX_INSTRUCTION_MIN_FTZ_F32, ast_ptx_ptx_instruction_num_args[AST_PTX_PTX_INSTRUCTION_MIN_FTZ_F32])
#define ast_ptx_encode_ptx_instruction_max_ftz_f32 ast_ptx_encode_ptx_instruction(AST_PTX_PTX_INSTRUCTION_MAX_FTZ_F32, ast_ptx_ptx_instruction_num_args[AST_PTX_PTX_INSTRUCTION_MAX_FTZ_F32])
#define ast_ptx_encode_ptx_instruction_rcp_approx_ftz_f32 ast_ptx_encode_ptx_instruction(AST_PTX_PTX_INSTRUCTION_RCP_APPROX_FTZ_F32, ast_ptx_ptx_instruction_num_args[AST_PTX_PTX_INSTRUCTION_RCP_APPROX_FTZ_F32])
#define ast_ptx_encode_ptx_instruction_sqrt_approx_ftz_f32 ast_ptx_encode_ptx_instruction(AST_PTX_PTX_INSTRUCTION_SQRT_APPROX_FTZ_F32, ast_ptx_ptx_instruction_num_args[AST_PTX_PTX_INSTRUCTION_SQRT_APPROX_FTZ_F32])
#define ast_ptx_encode_ptx_instruction_rsqrt_approx_ftz_f32 ast_ptx_encode_ptx_instruction(AST_PTX_PTX_INSTRUCTION_RSQRT_APPROX_FTZ_F32, ast_ptx_ptx_instruction_num_args[AST_PTX_PTX_INSTRUCTION_RSQRT_APPROX_FTZ_F32])
#define ast_ptx_encode_ptx_instruction_sin_approx_ftz_f32 ast_ptx_encode_ptx_instruction(AST_PTX_PTX_INSTRUCTION_SIN_APPROX_FTZ_F32, ast_ptx_ptx_instruction_num_args[AST_PTX_PTX_INSTRUCTION_SIN_APPROX_FTZ_F32])
#define ast_ptx_encode_ptx_instruction_cos_approx_ftz_f32 ast_ptx_encode_ptx_instruction(AST_PTX_PTX_INSTRUCTION_COS_APPROX_FTZ_F32, ast_ptx_ptx_instruction_num_args[AST_PTX_PTX_INSTRUCTION_COS_APPROX_FTZ_F32])
#define ast_ptx_encode_ptx_instruction_lg2_approx_ftz_f32 ast_ptx_encode_ptx_instruction(AST_PTX_PTX_INSTRUCTION_LG2_APPROX_FTZ_F32, ast_ptx_ptx_instruction_num_args[AST_PTX_PTX_INSTRUCTION_LG2_APPROX_FTZ_F32])
#define ast_ptx_encode_ptx_instruction_ex2_approx_ftz_f32 ast_ptx_encode_ptx_instruction(AST_PTX_PTX_INSTRUCTION_EX2_APPROX_FTZ_F32, ast_ptx_ptx_instruction_num_args[AST_PTX_PTX_INSTRUCTION_EX2_APPROX_FTZ_F32])
#define ast_ptx_encode_ptx_instruction_tanh_approx_f32 ast_ptx_encode_ptx_instruction(AST_PTX_PTX_INSTRUCTION_TANH_APPROX_F32, ast_ptx_ptx_instruction_num_args[AST_PTX_PTX_INSTRUCTION_TANH_APPROX_F32])

#endif

#endif /* AST_PTX_INSTRUCTIONS_H_INCLUDE */
