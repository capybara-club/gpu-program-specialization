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
 * @brief Small f32 CPU interpreter for AST PTX instruction arrays.
 *
 * The interpreter is intended for checking AST behavior on the CPU before or
 * alongside PTX generation. It follows the concrete f32 instruction catalog in
 * ast_ptx_instructions.h.
 */

#ifndef AST_PTX_INTERPRETER_H_INCLUDE
#define AST_PTX_INTERPRETER_H_INCLUDE

#include "ast_ptx_instructions.h"

#include <math.h>

/**
 * \brief Maximum number of f32 values held by the interpreter stack.
 */
#ifndef AST_PTX_STACK_DEPTH
#define AST_PTX_STACK_DEPTH 256u
#endif

/**
 * \brief Maximum nested routine-call depth accepted by the interpreter.
 */
#ifndef AST_PTX_ROUTINE_DEPTH
#define AST_PTX_ROUTINE_DEPTH 8u
#endif

/**
 * \brief Maximum number of instructions scanned before requiring a return.
 */
#ifndef AST_PTX_MAX_INSTRUCTIONS
#define AST_PTX_MAX_INSTRUCTIONS 65536u
#endif

/**
 * \brief Maximum number of f32 operands accepted by one PTX instruction or routine call.
 */
#ifndef AST_PTX_MAX_INSTRUCTION_ARGS
#define AST_PTX_MAX_INSTRUCTION_ARGS 4u
#endif

#define AST_PTX_INTERPRETER_ERROR(ans) do { AstPtxResult ast_ptx_result = (ans); return ast_ptx_result; } while (0)
#define AST_PTX_INTERPRETER_CHECK_RET(call) do { AstPtxResult ast_ptx_check_ret = (call); if (ast_ptx_check_ret != AST_PTX_SUCCESS) { AST_PTX_INTERPRETER_ERROR(ast_ptx_check_ret); } } while (0)

/**
 * \brief Pushes one f32 value onto the interpreter stack.
 */
static AstPtxResult
ast_ptx_interpret_push(
    float* stack,
    size_t* stack_size,
    float value
) {
    if (*stack_size >= AST_PTX_STACK_DEPTH) {
        AST_PTX_INTERPRETER_ERROR(AST_PTX_ERROR_STACK_OVERFLOW);
    }

    stack[*stack_size] = value;
    *stack_size += 1u;
    return AST_PTX_SUCCESS;
}

/**
 * \brief Evaluates one built-in f32 PTX instruction on CPU values.
 *
 * \param[in] ptx_instruction_idx Index from AstPtxPtxInstruction.
 * \param[in] args Contiguous f32 instruction arguments in source order.
 * \param[in] num_args Number of entries in args.
 * \param[out] value_ret Result value.
 */
static AstPtxResult
ast_ptx_interpret_ptx_instruction(
    AstPtxIdx ptx_instruction_idx,
    const float* args,
    size_t num_args,
    float* value_ret
) {
#define AST_PTX_INTERPRET_CHECK_NUM_ARGS(n) \
    do { \
        if (num_args != (n)) { \
            AST_PTX_INTERPRETER_ERROR(AST_PTX_ERROR_TOO_MANY_ARGS); \
        } \
    } while (0)

    switch ((AstPtxPtxInstruction)ptx_instruction_idx) {
        case AST_PTX_PTX_INSTRUCTION_COPYSIGN_F32:
            AST_PTX_INTERPRET_CHECK_NUM_ARGS(2u);
            *value_ret = copysignf(args[0], args[1]);
            break;
        case AST_PTX_PTX_INSTRUCTION_ADD_FTZ_F32:
            AST_PTX_INTERPRET_CHECK_NUM_ARGS(2u);
            *value_ret = args[0] + args[1];
            break;
        case AST_PTX_PTX_INSTRUCTION_SUB_FTZ_F32:
            AST_PTX_INTERPRET_CHECK_NUM_ARGS(2u);
            *value_ret = args[0] - args[1];
            break;
        case AST_PTX_PTX_INSTRUCTION_MUL_FTZ_F32:
            AST_PTX_INTERPRET_CHECK_NUM_ARGS(2u);
            *value_ret = args[0] * args[1];
            break;
        case AST_PTX_PTX_INSTRUCTION_FMA_RN_FTZ_F32:
            AST_PTX_INTERPRET_CHECK_NUM_ARGS(3u);
            *value_ret = fmaf(args[0], args[1], args[2]);
            break;
        case AST_PTX_PTX_INSTRUCTION_DIV_APPROX_FTZ_F32:
            AST_PTX_INTERPRET_CHECK_NUM_ARGS(2u);
            *value_ret = args[0] / args[1];
            break;
        case AST_PTX_PTX_INSTRUCTION_ABS_FTZ_F32:
            AST_PTX_INTERPRET_CHECK_NUM_ARGS(1u);
            *value_ret = fabsf(args[0]);
            break;
        case AST_PTX_PTX_INSTRUCTION_NEG_FTZ_F32:
            AST_PTX_INTERPRET_CHECK_NUM_ARGS(1u);
            *value_ret = -args[0];
            break;
        case AST_PTX_PTX_INSTRUCTION_MIN_FTZ_F32:
            AST_PTX_INTERPRET_CHECK_NUM_ARGS(2u);
            *value_ret = fminf(args[0], args[1]);
            break;
        case AST_PTX_PTX_INSTRUCTION_MAX_FTZ_F32:
            AST_PTX_INTERPRET_CHECK_NUM_ARGS(2u);
            *value_ret = fmaxf(args[0], args[1]);
            break;
        case AST_PTX_PTX_INSTRUCTION_RCP_APPROX_FTZ_F32:
            AST_PTX_INTERPRET_CHECK_NUM_ARGS(1u);
            *value_ret = 1.0f / args[0];
            break;
        case AST_PTX_PTX_INSTRUCTION_SQRT_APPROX_FTZ_F32:
            AST_PTX_INTERPRET_CHECK_NUM_ARGS(1u);
            *value_ret = sqrtf(args[0]);
            break;
        case AST_PTX_PTX_INSTRUCTION_RSQRT_APPROX_FTZ_F32:
            AST_PTX_INTERPRET_CHECK_NUM_ARGS(1u);
            *value_ret = 1.0f / sqrtf(args[0]);
            break;
        case AST_PTX_PTX_INSTRUCTION_SIN_APPROX_FTZ_F32:
            AST_PTX_INTERPRET_CHECK_NUM_ARGS(1u);
            *value_ret = sinf(args[0]);
            break;
        case AST_PTX_PTX_INSTRUCTION_COS_APPROX_FTZ_F32:
            AST_PTX_INTERPRET_CHECK_NUM_ARGS(1u);
            *value_ret = cosf(args[0]);
            break;
        case AST_PTX_PTX_INSTRUCTION_LG2_APPROX_FTZ_F32:
            AST_PTX_INTERPRET_CHECK_NUM_ARGS(1u);
            *value_ret = log2f(args[0]);
            break;
        case AST_PTX_PTX_INSTRUCTION_EX2_APPROX_FTZ_F32:
            AST_PTX_INTERPRET_CHECK_NUM_ARGS(1u);
            *value_ret = exp2f(args[0]);
            break;
        case AST_PTX_PTX_INSTRUCTION_TANH_APPROX_F32:
            AST_PTX_INTERPRET_CHECK_NUM_ARGS(1u);
            *value_ret = tanhf(args[0]);
            break;
        case AST_PTX_PTX_INSTRUCTION_NUM_ENUMS:
        default:
            AST_PTX_INTERPRETER_ERROR(AST_PTX_ERROR_UNSUPPORTED_INSTRUCTION);
    }

#undef AST_PTX_INTERPRET_CHECK_NUM_ARGS

    return AST_PTX_SUCCESS;
}

/**
 * \brief Interprets one program or routine frame.
 *
 * \details The frame must leave exactly one value above its entry stack depth
 * when it reaches ast_ptx_encode_return.
 */
static AstPtxResult
ast_ptx_interpret_frame(
    const AstPtxInstruction* instructions,
    const AstPtxInstruction* const* routines,
    size_t num_routines,
    const float* inputs,
    size_t num_inputs,
    const float* routine_args,
    size_t num_routine_args,
    size_t frame_depth,
    float* stack,
    size_t* stack_size
) {
    const size_t base_stack_size = *stack_size;

    for (size_t instruction_idx = 0u;
         instruction_idx < AST_PTX_MAX_INSTRUCTIONS;
         ++instruction_idx) {
        const AstPtxInstruction instruction = instructions[instruction_idx];

        switch ((AstPtxInstructionType)instruction.instruction_type) {
            case AST_PTX_INSTRUCTION_TYPE_INPUT: {
                if (instruction.payload.idx >= num_inputs) {
                    AST_PTX_INTERPRETER_ERROR(AST_PTX_ERROR_INPUT_IDX_OUT_OF_BOUNDS);
                }

                AST_PTX_INTERPRETER_CHECK_RET(
                    ast_ptx_interpret_push(
                        stack,
                        stack_size,
                        inputs[instruction.payload.idx]
                    )
                );
            } break;

            case AST_PTX_INSTRUCTION_TYPE_CONSTANT: {
                AST_PTX_INTERPRETER_CHECK_RET(
                    ast_ptx_interpret_push(stack, stack_size, instruction.payload.f)
                );
            } break;

            case AST_PTX_INSTRUCTION_TYPE_ROUTINE_ARG: {
                if (routine_args == NULL ||
                    instruction.payload.idx >= num_routine_args) {
                    AST_PTX_INTERPRETER_ERROR(AST_PTX_ERROR_ROUTINE_ARG_IDX_OUT_OF_BOUNDS);
                }

                AST_PTX_INTERPRETER_CHECK_RET(
                    ast_ptx_interpret_push(
                        stack,
                        stack_size,
                        routine_args[instruction.payload.idx]
                    )
                );
            } break;

            case AST_PTX_INSTRUCTION_TYPE_PTX_INSTRUCTION: {
                const size_t num_args = instruction.aux;
                if (num_args > AST_PTX_MAX_INSTRUCTION_ARGS) {
                    AST_PTX_INTERPRETER_ERROR(AST_PTX_ERROR_TOO_MANY_ARGS);
                }

                if (*stack_size < num_args) {
                    AST_PTX_INTERPRETER_ERROR(AST_PTX_ERROR_STACK_UNDERFLOW);
                }

                const size_t arg_start_idx = *stack_size - num_args;
                float value = 0.0f;

                AST_PTX_INTERPRETER_CHECK_RET(
                    ast_ptx_interpret_ptx_instruction(
                        instruction.payload.idx,
                        &stack[arg_start_idx],
                        num_args,
                        &value
                    )
                );

                *stack_size = arg_start_idx;

                AST_PTX_INTERPRETER_CHECK_RET(
                    ast_ptx_interpret_push(stack, stack_size, value)
                );
            } break;

            case AST_PTX_INSTRUCTION_TYPE_ROUTINE: {
                const AstPtxIdx routine_idx = instruction.payload.idx;
                if (instruction.aux > AST_PTX_MAX_INSTRUCTION_ARGS) {
                    AST_PTX_INTERPRETER_ERROR(AST_PTX_ERROR_TOO_MANY_ARGS);
                }

                if (routines == NULL ||
                    routine_idx >= num_routines ||
                    routines[routine_idx] == NULL) {
                    AST_PTX_INTERPRETER_ERROR(AST_PTX_ERROR_ROUTINE_IDX_OUT_OF_BOUNDS);
                }

                const size_t next_frame_depth = frame_depth + 1u;
                if (next_frame_depth > AST_PTX_ROUTINE_DEPTH) {
                    AST_PTX_INTERPRETER_ERROR(AST_PTX_ERROR_ROUTINE_DEPTH_EXCEEDED);
                }

                const size_t num_args = instruction.aux;
                if (*stack_size < num_args) {
                    AST_PTX_INTERPRETER_ERROR(AST_PTX_ERROR_STACK_UNDERFLOW);
                }

                const size_t arg_start_idx = *stack_size - num_args;
                float next_routine_args[AST_PTX_MAX_INSTRUCTION_ARGS];

                for (size_t arg_idx = 0u; arg_idx < num_args; ++arg_idx) {
                    next_routine_args[arg_idx] = stack[arg_start_idx + arg_idx];
                }

                *stack_size = arg_start_idx;

                AST_PTX_INTERPRETER_CHECK_RET(
                    ast_ptx_interpret_frame(
                        routines[routine_idx],
                        routines,
                        num_routines,
                        inputs,
                        num_inputs,
                        next_routine_args,
                        num_args,
                        next_frame_depth,
                        stack,
                        stack_size
                    )
                );
            } break;

            case AST_PTX_INSTRUCTION_TYPE_RETURN: {
                if (*stack_size != base_stack_size + 1u) {
                    AST_PTX_INTERPRETER_ERROR(AST_PTX_ERROR_INVALID_VALUE);
                }

                return AST_PTX_SUCCESS;
            } break;

            case AST_PTX_INSTRUCTION_TYPE_REGISTER:
            case AST_PTX_INSTRUCTION_TYPE_NONE:
            case AST_PTX_INSTRUCTION_TYPE_NUM_ENUMS:
            default:
                AST_PTX_INTERPRETER_ERROR(AST_PTX_ERROR_BAD_INSTRUCTION_MAYBE_FORGOT_RETURN);
        }
    }

    AST_PTX_INTERPRETER_ERROR(AST_PTX_ERROR_BAD_INSTRUCTION_MAYBE_FORGOT_RETURN);
}

/**
 * \brief Interprets a top-level AST PTX program and returns its single f32 value.
 *
 * \param[in] instructions Top-level postorder instruction array, terminated by return.
 * \param[in] routines Routine instruction arrays indexed by ROUTINE instructions.
 * \param[in] num_routines Number of entries in routines.
 * \param[in] inputs f32 input values indexed by INPUT instructions.
 * \param[in] num_inputs Number of entries in inputs.
 * \param[out] output_ret Interpreted top-level result.
 *
 * \return AST_PTX_SUCCESS on success, otherwise an AstPtxResult error.
 */
static AstPtxResult
ast_ptx_interpret(
    const AstPtxInstruction* instructions,
    const AstPtxInstruction* const* routines,
    size_t num_routines,
    const float* inputs,
    size_t num_inputs,
    float* output_ret
) {
    if (instructions == NULL ||
        output_ret == NULL ||
        (inputs == NULL && num_inputs != 0u)) {
        AST_PTX_INTERPRETER_ERROR(AST_PTX_ERROR_INVALID_VALUE);
    }

    float stack[AST_PTX_STACK_DEPTH];
    size_t stack_size = 0u;

    AST_PTX_INTERPRETER_CHECK_RET(
        ast_ptx_interpret_frame(
            instructions,
            routines,
            num_routines,
            inputs,
            num_inputs,
            NULL,
            0u,
            0u,
            stack,
            &stack_size
        )
    );

    if (stack_size != 1u) {
        AST_PTX_INTERPRETER_ERROR(AST_PTX_ERROR_INVALID_VALUE);
    }

    *output_ret = stack[0];
    return AST_PTX_SUCCESS;
}

#undef AST_PTX_INTERPRETER_CHECK_RET
#undef AST_PTX_INTERPRETER_ERROR

#endif /* AST_PTX_INTERPRETER_H_INCLUDE */
