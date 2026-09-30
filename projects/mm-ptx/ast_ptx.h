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
 * @brief Single-header compiler for postorder f32 AST instruction arrays.
 *
 * AST PTX compiles a postorder AST into a PTX stub that yields one f32 value.
 * The concrete f32 PTX instruction catalog is intentionally kept outside this
 * core header.
 */

#ifndef AST_PTX_H_INCLUDE
#define AST_PTX_H_INCLUDE

#define AST_PTX_VERSION_MAJOR 0 //!< AST PTX major version.
#define AST_PTX_VERSION_MINOR 1 //!< AST PTX minor version.
#define AST_PTX_VERSION_PATCH 0 //!< AST PTX patch version.

/**
 * \brief String representation of the AST PTX library version.
 */
#define AST_PTX_VERSION_STRING "0.1.0"

/**
 * \brief Integer representation of the AST PTX library version.
 */
#define AST_PTX_VERSION \
    (AST_PTX_VERSION_MAJOR * 10000 + AST_PTX_VERSION_MINOR * 100 + AST_PTX_VERSION_PATCH)

#ifdef __cplusplus
#define AST_PTX_PUBLIC_DEC extern "C"
#define AST_PTX_PUBLIC_DEF extern "C"
#else
#define AST_PTX_PUBLIC_DEC extern
#define AST_PTX_PUBLIC_DEF
#endif

#include <stddef.h>
#include <stdint.h>

/**
 * \brief Packs AstPtxInstruction so the encoded instruction remains 8 bytes.
 */
#ifndef AST_PTX_PACKED
#if defined(_MSC_VER)
#define AST_PTX_PACKED
#else
#define AST_PTX_PACKED __attribute__((packed))
#endif
#endif

/**
 * \brief Index type used for inputs, routines, PTX instructions, and registers.
 */
typedef uint32_t AstPtxIdx;

/**
 * \brief AST PTX result codes.
 */
typedef enum {
    /** Operation completed successfully. */
    AST_PTX_SUCCESS = 0,
    /** An internal error occurred. */
    AST_PTX_ERROR_INTERNAL = 1,
    /** The supplied output buffer was too small. */
    AST_PTX_ERROR_INSUFFICIENT_BUFFER = 2,
    /** A required pointer or value was invalid. */
    AST_PTX_ERROR_INVALID_VALUE = 3,
    /** Instruction traversal reached an invalid instruction, commonly from a missing return. */
    AST_PTX_ERROR_BAD_INSTRUCTION_MAYBE_FORGOT_RETURN = 4,
    /** An input index was outside the provided input-register or input-value array. */
    AST_PTX_ERROR_INPUT_IDX_OUT_OF_BOUNDS = 5,
    /** A PTX instruction index was outside the provided PTX instruction-name array. */
    AST_PTX_ERROR_PTX_INSTRUCTION_IDX_OUT_OF_BOUNDS = 6,
    /** The compile or interpreter value stack overflowed. */
    AST_PTX_ERROR_STACK_OVERFLOW = 7,
    /** An instruction required more stack values than were available. */
    AST_PTX_ERROR_STACK_UNDERFLOW = 8,
    /** A PTX instruction or routine call requested more arguments than supported. */
    AST_PTX_ERROR_TOO_MANY_ARGS = 9,
    /** The instruction is not supported by the active implementation. */
    AST_PTX_ERROR_UNSUPPORTED_INSTRUCTION = 10,
    /** A routine index was outside the provided routine array. */
    AST_PTX_ERROR_ROUTINE_IDX_OUT_OF_BOUNDS = 11,
    /** A routine argument index was outside the active routine call's argument list. */
    AST_PTX_ERROR_ROUTINE_ARG_IDX_OUT_OF_BOUNDS = 12,
    /** Routine call depth exceeded AST_PTX_ROUTINE_DEPTH. */
    AST_PTX_ERROR_ROUTINE_DEPTH_EXCEEDED = 13,
    /** The number of result enums. */
    AST_PTX_RESULT_NUM_ENUMS = 14
} AstPtxResult;

/**
 * \brief AST PTX instruction types.
 *
 * \details Public programs are postorder arrays terminated by
 * AST_PTX_INSTRUCTION_TYPE_RETURN. REGISTER is used internally by the compiler
 * to represent generated temporary registers.
 */
typedef enum {
    /** Invalid or empty instruction. */
    AST_PTX_INSTRUCTION_TYPE_NONE = 0,
    /** Pushes an input register or interpreter input value by index. */
    AST_PTX_INSTRUCTION_TYPE_INPUT = 1,
    /** Pushes a fixed f32 immediate value. */
    AST_PTX_INSTRUCTION_TYPE_CONSTANT = 2,
    /** Consumes aux stack values and emits or interprets a PTX instruction. */
    AST_PTX_INSTRUCTION_TYPE_PTX_INSTRUCTION = 3,
    /** Calls a routine by index, consuming aux arguments. */
    AST_PTX_INSTRUCTION_TYPE_ROUTINE = 4,
    /** Terminates a program or routine and returns the one stack value. */
    AST_PTX_INSTRUCTION_TYPE_RETURN = 5,
    /** Pushes one argument from the active routine frame. */
    AST_PTX_INSTRUCTION_TYPE_ROUTINE_ARG = 6,
    /** Internal generated-register value. */
    AST_PTX_INSTRUCTION_TYPE_REGISTER = 7,
    /** The number of instruction type enums. */
    AST_PTX_INSTRUCTION_TYPE_NUM_ENUMS = 8
} AstPtxInstructionType;

/**
 * \brief Payload for AstPtxInstruction.
 *
 * \details Constants use f. Indexed instructions use idx.
 */
typedef union AST_PTX_PACKED {
    float f;
    AstPtxIdx idx;
} AstPtxPayload;

/**
 * \brief Fixed-width AST PTX instruction.
 *
 * \details aux is the number of f32 arguments for PTX instructions and routine
 * calls. For other public instructions it is currently zero.
 */
typedef struct AST_PTX_PACKED {
    uint16_t instruction_type;
    uint16_t aux;
    AstPtxPayload payload;
} AstPtxInstruction;

/**
 * \brief Compiles a postorder AST instruction array into PTX.
 *
 * \param[in] output_register_name Bare PTX register name receiving the top-level result, without a leading percent sign.
 * \param[in] input_register_names Bare PTX register names indexed by INPUT instructions, without leading percent signs.
 * \param[in] num_input_registers Number of entries in input_register_names.
 * \param[in] ptx_instruction_names PTX instruction strings indexed by PTX instruction encodes.
 * \param[in] ptx_instruction_num_args Expected f32 argument counts indexed by PTX instruction encodes.
 * \param[in] num_ptx_instructions Number of entries in ptx_instruction_names.
 * \param[in] routines Routine instruction arrays indexed by ROUTINE instructions.
 * \param[in] num_routines Number of entries in routines.
 * \param[in] instructions Top-level postorder instruction array, terminated by return.
 * \param[out] buffer Output PTX buffer. Pass NULL to measure.
 * \param[in] buffer_size Size of buffer in bytes.
 * \param[out] buffer_bytes_written_ret Number of bytes required or written, excluding the null terminator.
 *
 * \return AST_PTX_SUCCESS on success, otherwise an AstPtxResult error.
 */
AST_PTX_PUBLIC_DEC AstPtxResult
ast_ptx_compile(
    const char* output_register_name,
    const char* const* input_register_names,  size_t num_input_registers,
    const char* const* ptx_instruction_names,
    const uint16_t* ptx_instruction_num_args, size_t num_ptx_instructions,
    const AstPtxInstruction* const* routines, size_t num_routines,
    const AstPtxInstruction* instructions,
    char* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_ret
);

#endif /* AST_PTX_H_INCLUDE */

#ifdef AST_PTX_IMPLEMENTATION
#ifndef AST_PTX_IMPLEMENTATION_ONCE
#define AST_PTX_IMPLEMENTATION_ONCE

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#define AST_PTX_STATIC_ASSERT(cond, msg) typedef char static_assertion_##msg[(cond) ? 1 : -1]
#define _AST_PTX_ERROR(ans) do { AstPtxResult ast_ptx_result = (ans); return ast_ptx_result; } while (0)
#define _AST_PTX_CHECK_RET(call) do { AstPtxResult ast_ptx_check_ret = (call); if (ast_ptx_check_ret != AST_PTX_SUCCESS) { _AST_PTX_ERROR(ast_ptx_check_ret); } } while (0)

AST_PTX_STATIC_ASSERT(sizeof(AstPtxInstruction) == 8, ast_ptx_instruction_must_be_eight_bytes);
AST_PTX_STATIC_ASSERT(AST_PTX_INSTRUCTION_TYPE_NUM_ENUMS <= UINT16_MAX, ast_ptx_instruction_type_must_fit_in_uint16);

#ifndef AST_PTX_STACK_DEPTH
#define AST_PTX_STACK_DEPTH 256u
#endif

#ifndef AST_PTX_ROUTINE_DEPTH
#define AST_PTX_ROUTINE_DEPTH 8u
#endif

#ifndef AST_PTX_MAX_INSTRUCTIONS
#define AST_PTX_MAX_INSTRUCTIONS 65536u
#endif

#ifndef AST_PTX_MAX_INSTRUCTION_ARGS
#define AST_PTX_MAX_INSTRUCTION_ARGS 4u
#endif

#ifndef AST_PTX_REGISTER_NAME
#define AST_PTX_REGISTER_NAME "ast"
#endif

#ifndef AST_PTX_TABBING
#define AST_PTX_TABBING "\t"
#endif

static AstPtxResult
_ast_ptx_snprintf_append(
    char* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_ret,
    const char* format,
    ...
) {
    va_list args;
    va_start(args, format);

    const size_t written_before = *buffer_bytes_written_ret;
    int bytes_written = 0;

    if (buffer == NULL || buffer_size == 0u || written_before >= buffer_size) {
        bytes_written = vsnprintf(NULL, 0u, format, args);
    } else {
        bytes_written = vsnprintf(
            buffer + written_before,
            buffer_size - written_before,
            format,
            args
        );
    }

    va_end(args);

    if (bytes_written < 0) {
        _AST_PTX_ERROR(AST_PTX_ERROR_INTERNAL);
    }

    const size_t num_bytes_written = (size_t)bytes_written;
    *buffer_bytes_written_ret = written_before + num_bytes_written;

    if (buffer != NULL) {
        if (buffer_size == 0u ||
            written_before >= buffer_size ||
            num_bytes_written >= buffer_size - written_before) {
            _AST_PTX_ERROR(AST_PTX_ERROR_INSUFFICIENT_BUFFER);
        }
    }

    return AST_PTX_SUCCESS;
}

static AstPtxResult
_ast_ptx_push_value(
    AstPtxInstruction* stack,
    size_t* stack_size,
    AstPtxInstruction value
) {
    if (*stack_size >= AST_PTX_STACK_DEPTH) {
        _AST_PTX_ERROR(AST_PTX_ERROR_STACK_OVERFLOW);
    }

    stack[*stack_size] = value;
    *stack_size += 1u;
    return AST_PTX_SUCCESS;
}

static AstPtxResult
_ast_ptx_pop_value(
    AstPtxInstruction* stack,
    size_t* stack_size,
    AstPtxInstruction* value_ret
) {
    if (*stack_size == 0u) {
        _AST_PTX_ERROR(AST_PTX_ERROR_STACK_UNDERFLOW);
    }

    *stack_size -= 1u;
    *value_ret = stack[*stack_size];
    return AST_PTX_SUCCESS;
}

static uint32_t
_ast_ptx_f32_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static AstPtxResult
_ast_ptx_write_value(
    const AstPtxInstruction* value,
    const char* const* input_register_names,
    size_t num_input_registers,
    char* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_ret
) {
    switch ((AstPtxInstructionType)value->instruction_type) {
        case AST_PTX_INSTRUCTION_TYPE_INPUT: {
            if (input_register_names != NULL &&
                value->payload.idx < num_input_registers &&
                input_register_names[value->payload.idx] != NULL) {
                _AST_PTX_CHECK_RET(
                    _ast_ptx_snprintf_append(
                        buffer,
                        buffer_size,
                        buffer_bytes_written_ret,
                        "%%%s",
                        input_register_names[value->payload.idx]
                    )
                );
            } else {
                _AST_PTX_ERROR(AST_PTX_ERROR_INPUT_IDX_OUT_OF_BOUNDS);
            }
        } break;

        case AST_PTX_INSTRUCTION_TYPE_CONSTANT: {
            _AST_PTX_CHECK_RET(
                _ast_ptx_snprintf_append(
                    buffer,
                    buffer_size,
                    buffer_bytes_written_ret,
                    "0f%08x",
                    (unsigned)_ast_ptx_f32_bits(value->payload.f)
                )
            );
        } break;

        case AST_PTX_INSTRUCTION_TYPE_REGISTER: {
            _AST_PTX_CHECK_RET(
                _ast_ptx_snprintf_append(
                    buffer,
                    buffer_size,
                    buffer_bytes_written_ret,
                    "%%" AST_PTX_REGISTER_NAME "%u",
                    (unsigned)value->payload.idx
                )
            );
        } break;

        case AST_PTX_INSTRUCTION_TYPE_ROUTINE_ARG:
            _AST_PTX_ERROR(AST_PTX_ERROR_UNSUPPORTED_INSTRUCTION);

        case AST_PTX_INSTRUCTION_TYPE_NONE:
        case AST_PTX_INSTRUCTION_TYPE_PTX_INSTRUCTION:
        case AST_PTX_INSTRUCTION_TYPE_ROUTINE:
        case AST_PTX_INSTRUCTION_TYPE_RETURN:
        case AST_PTX_INSTRUCTION_TYPE_NUM_ENUMS:
        default: {
            _AST_PTX_ERROR(AST_PTX_ERROR_BAD_INSTRUCTION_MAYBE_FORGOT_RETURN);
        } break;
    }

    return AST_PTX_SUCCESS;
}

static AstPtxResult
_ast_ptx_write_register_declarations(
    AstPtxIdx num_registers,
    char* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_ret
) {
    _AST_PTX_CHECK_RET(
        _ast_ptx_snprintf_append(
            buffer,
            buffer_size,
            buffer_bytes_written_ret,
            AST_PTX_TABBING "{\n"
        )
    );

    if (num_registers != 0u) {
        _AST_PTX_CHECK_RET(
            _ast_ptx_snprintf_append(
                buffer,
                buffer_size,
                buffer_bytes_written_ret,
                AST_PTX_TABBING ".reg .f32 %%" AST_PTX_REGISTER_NAME "<%u>;\n",
                (unsigned)num_registers
            )
        );
    }

    return AST_PTX_SUCCESS;
}

static AstPtxResult
_ast_ptx_prefix_register_declarations(
    AstPtxIdx num_registers,
    char* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_ret
) {
    size_t register_declaration_bytes_written = 0u;

    _AST_PTX_CHECK_RET(
        _ast_ptx_write_register_declarations(
            num_registers,
            NULL,
            0u,
            &register_declaration_bytes_written
        )
    );

    if (buffer == NULL) {
        *buffer_bytes_written_ret += register_declaration_bytes_written;
        return AST_PTX_SUCCESS;
    }

    const size_t body_bytes_written = *buffer_bytes_written_ret;

    if (body_bytes_written >= buffer_size) {
        _AST_PTX_ERROR(AST_PTX_ERROR_INSUFFICIENT_BUFFER);
    }

    if (register_declaration_bytes_written >= buffer_size ||
        body_bytes_written >= buffer_size - register_declaration_bytes_written) {
        _AST_PTX_ERROR(AST_PTX_ERROR_INSUFFICIENT_BUFFER);
    }

    memmove(
        buffer + register_declaration_bytes_written,
        buffer,
        body_bytes_written
    );

    char first_body_char = '\0';
    if (body_bytes_written != 0u) {
        first_body_char = buffer[register_declaration_bytes_written];
    }

    size_t register_declaration_bytes_written_temp = 0u;

    _AST_PTX_CHECK_RET(
        _ast_ptx_write_register_declarations(
            num_registers,
            buffer,
            register_declaration_bytes_written + 1u,
            &register_declaration_bytes_written_temp
        )
    );

    if (register_declaration_bytes_written_temp !=
        register_declaration_bytes_written) {
        _AST_PTX_ERROR(AST_PTX_ERROR_INTERNAL);
    }

    if (body_bytes_written != 0u) {
        buffer[register_declaration_bytes_written] = first_body_char;
    }

    *buffer_bytes_written_ret =
        body_bytes_written + register_declaration_bytes_written;
    buffer[*buffer_bytes_written_ret] = '\0';

    return AST_PTX_SUCCESS;
}

static AstPtxResult
_ast_ptx_compile_frame(
    const char* output_register_name,
    const char* const* input_register_names,
    size_t num_input_registers,
    const char* const* ptx_instruction_names,
    const uint16_t* ptx_instruction_num_args,
    size_t num_ptx_instructions,
    const AstPtxInstruction* const* routines,
    size_t num_routines,
    const AstPtxInstruction* instructions,
    const AstPtxInstruction* routine_args,
    size_t num_routine_args,
    size_t frame_depth,
    AstPtxInstruction* stack,
    size_t* stack_size,
    AstPtxIdx* next_register,
    char* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_ret
) {
    const size_t base_stack_size = *stack_size;

    for (size_t instruction_idx = 0u;
         instruction_idx < AST_PTX_MAX_INSTRUCTIONS;
         ++instruction_idx) {
        const AstPtxInstruction instruction = instructions[instruction_idx];

        switch ((AstPtxInstructionType)instruction.instruction_type) {
            case AST_PTX_INSTRUCTION_TYPE_INPUT: {
                _AST_PTX_CHECK_RET(
                    _ast_ptx_push_value(stack, stack_size, instruction)
                );
            } break;

            case AST_PTX_INSTRUCTION_TYPE_CONSTANT: {
                _AST_PTX_CHECK_RET(
                    _ast_ptx_push_value(stack, stack_size, instruction)
                );
            } break;

            case AST_PTX_INSTRUCTION_TYPE_ROUTINE_ARG: {
                const AstPtxIdx routine_arg_idx = instruction.payload.idx;
                if (routine_args == NULL ||
                    routine_arg_idx >= num_routine_args) {
                    _AST_PTX_ERROR(AST_PTX_ERROR_ROUTINE_ARG_IDX_OUT_OF_BOUNDS);
                }

                _AST_PTX_CHECK_RET(
                    _ast_ptx_push_value(
                        stack,
                        stack_size,
                        routine_args[routine_arg_idx]
                    )
                );
            } break;

            case AST_PTX_INSTRUCTION_TYPE_PTX_INSTRUCTION: {
                const AstPtxIdx ptx_instruction_idx = instruction.payload.idx;
                if (instruction.aux > AST_PTX_MAX_INSTRUCTION_ARGS) {
                    _AST_PTX_ERROR(AST_PTX_ERROR_TOO_MANY_ARGS);
                }

                if (ptx_instruction_names == NULL ||
                    ptx_instruction_idx >= num_ptx_instructions ||
                    ptx_instruction_names[ptx_instruction_idx] == NULL) {
                    _AST_PTX_ERROR(AST_PTX_ERROR_PTX_INSTRUCTION_IDX_OUT_OF_BOUNDS);
                }

                if (ptx_instruction_num_args == NULL) {
                    _AST_PTX_ERROR(AST_PTX_ERROR_INVALID_VALUE);
                }

                const size_t num_args = instruction.aux;
                if (num_args != ptx_instruction_num_args[ptx_instruction_idx]) {
                    _AST_PTX_ERROR(AST_PTX_ERROR_INVALID_VALUE);
                }

                if (*stack_size < num_args) {
                    _AST_PTX_ERROR(AST_PTX_ERROR_STACK_UNDERFLOW);
                }

                const size_t arg_start_idx = *stack_size - num_args;

                const char* ptx_instruction_name = ptx_instruction_names[ptx_instruction_idx];

                const AstPtxIdx output_register_idx = *next_register;
                *next_register += 1u;

                _AST_PTX_CHECK_RET(
                    _ast_ptx_snprintf_append(
                        buffer,
                        buffer_size,
                        buffer_bytes_written_ret,
                        AST_PTX_TABBING "%s %%" AST_PTX_REGISTER_NAME "%u",
                        ptx_instruction_name,
                        (unsigned)output_register_idx
                    )
                );

                for (size_t arg_idx = 0u; arg_idx < num_args; ++arg_idx) {
                    _AST_PTX_CHECK_RET(
                        _ast_ptx_snprintf_append(
                            buffer,
                            buffer_size,
                            buffer_bytes_written_ret,
                            ", "
                        )
                    );

                    _AST_PTX_CHECK_RET(
                        _ast_ptx_write_value(
                            &stack[arg_start_idx + arg_idx],
                            input_register_names,
                            num_input_registers,
                            buffer,
                            buffer_size,
                            buffer_bytes_written_ret
                        )
                    );
                }

                _AST_PTX_CHECK_RET(
                    _ast_ptx_snprintf_append(
                        buffer,
                        buffer_size,
                        buffer_bytes_written_ret,
                        ";\n"
                    )
                );

                *stack_size -= num_args;

                AstPtxInstruction value;
                value.instruction_type = AST_PTX_INSTRUCTION_TYPE_REGISTER;
                value.aux = 0u;
                value.payload.idx = output_register_idx;
                _AST_PTX_CHECK_RET(
                    _ast_ptx_push_value(stack, stack_size, value)
                );
            } break;

            case AST_PTX_INSTRUCTION_TYPE_ROUTINE: {
                const AstPtxIdx routine_idx = instruction.payload.idx;
                if (instruction.aux > AST_PTX_MAX_INSTRUCTION_ARGS) {
                    _AST_PTX_ERROR(AST_PTX_ERROR_TOO_MANY_ARGS);
                }

                if (routines == NULL ||
                    routine_idx >= num_routines ||
                    routines[routine_idx] == NULL) {
                    _AST_PTX_ERROR(AST_PTX_ERROR_ROUTINE_IDX_OUT_OF_BOUNDS);
                }

                const size_t next_frame_depth = frame_depth + 1u;
                if (next_frame_depth > AST_PTX_ROUTINE_DEPTH) {
                    _AST_PTX_ERROR(AST_PTX_ERROR_ROUTINE_DEPTH_EXCEEDED);
                }

                const size_t num_args = instruction.aux;
                if (*stack_size < num_args) {
                    _AST_PTX_ERROR(AST_PTX_ERROR_STACK_UNDERFLOW);
                }

                const size_t arg_start_idx = *stack_size - num_args;
                AstPtxInstruction next_routine_args[AST_PTX_MAX_INSTRUCTION_ARGS];

                for (size_t arg_idx = 0u; arg_idx < num_args; ++arg_idx) {
                    next_routine_args[arg_idx] = stack[arg_start_idx + arg_idx];
                }

                *stack_size = arg_start_idx;

                _AST_PTX_CHECK_RET(
                    _ast_ptx_compile_frame(
                        output_register_name,
                        input_register_names,
                        num_input_registers,
                        ptx_instruction_names,
                        ptx_instruction_num_args,
                        num_ptx_instructions,
                        routines,
                        num_routines,
                        routines[routine_idx],
                        next_routine_args,
                        num_args,
                        next_frame_depth,
                        stack,
                        stack_size,
                        next_register,
                        buffer,
                        buffer_size,
                        buffer_bytes_written_ret
                    )
                );
            } break;

            case AST_PTX_INSTRUCTION_TYPE_RETURN: {
                AstPtxInstruction value;

                if (*stack_size != base_stack_size + 1u) {
                    _AST_PTX_ERROR(AST_PTX_ERROR_INVALID_VALUE);
                }

                if (frame_depth != 0u) {
                    return AST_PTX_SUCCESS;
                }

                _AST_PTX_CHECK_RET(
                    _ast_ptx_pop_value(stack, stack_size, &value)
                );

                _AST_PTX_CHECK_RET(
                    _ast_ptx_snprintf_append(
                        buffer,
                        buffer_size,
                        buffer_bytes_written_ret,
                        AST_PTX_TABBING "mov.f32 %%%s, ",
                        output_register_name
                    )
                );

                _AST_PTX_CHECK_RET(
                    _ast_ptx_write_value(
                        &value,
                        input_register_names,
                        num_input_registers,
                        buffer,
                        buffer_size,
                        buffer_bytes_written_ret
                    )
                );

                _AST_PTX_CHECK_RET(
                    _ast_ptx_snprintf_append(
                        buffer,
                        buffer_size,
                        buffer_bytes_written_ret,
                        ";\n"
                    )
                );

                return AST_PTX_SUCCESS;
            } break;

            case AST_PTX_INSTRUCTION_TYPE_REGISTER:
            case AST_PTX_INSTRUCTION_TYPE_NONE:
            case AST_PTX_INSTRUCTION_TYPE_NUM_ENUMS:
            default: {
                _AST_PTX_ERROR(AST_PTX_ERROR_BAD_INSTRUCTION_MAYBE_FORGOT_RETURN);
            } break;
        }
    }

    _AST_PTX_ERROR(AST_PTX_ERROR_BAD_INSTRUCTION_MAYBE_FORGOT_RETURN);
}

AST_PTX_PUBLIC_DEF AstPtxResult
ast_ptx_compile(
    const char* output_register_name,
    const char* const* input_register_names,
    size_t num_input_registers,
    const char* const* ptx_instruction_names,
    const uint16_t* ptx_instruction_num_args,
    size_t num_ptx_instructions,
    const AstPtxInstruction* const* routines,
    size_t num_routines,
    const AstPtxInstruction* instructions,
    char* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_ret
) {
    size_t bytes_written = 0u;
    if (buffer_bytes_written_ret == NULL) {
        buffer_bytes_written_ret = &bytes_written;
    }

    *buffer_bytes_written_ret = 0u;

    if (buffer != NULL && buffer_size != 0u) {
        buffer[0] = '\0';
    }

    if (output_register_name == NULL) {
        _AST_PTX_ERROR(AST_PTX_ERROR_INVALID_VALUE);
    }

    if (instructions == NULL) {
        _AST_PTX_ERROR(AST_PTX_ERROR_INVALID_VALUE);
    }

    AstPtxInstruction stack[AST_PTX_STACK_DEPTH];
    size_t stack_size = 0u;
    AstPtxIdx next_register = 0u;

    _AST_PTX_CHECK_RET(
        _ast_ptx_compile_frame(
            output_register_name,
            input_register_names,
            num_input_registers,
            ptx_instruction_names,
            ptx_instruction_num_args,
            num_ptx_instructions,
            routines,
            num_routines,
            instructions,
            NULL,
            0u,
            0u,
            stack,
            &stack_size,
            &next_register,
            buffer,
            buffer_size,
            buffer_bytes_written_ret
        )
    );

    _AST_PTX_CHECK_RET(
        _ast_ptx_snprintf_append(
            buffer,
            buffer_size,
            buffer_bytes_written_ret,
            AST_PTX_TABBING "}"
        )
    );

    _AST_PTX_CHECK_RET(
        _ast_ptx_prefix_register_declarations(
            next_register,
            buffer,
            buffer_size,
            buffer_bytes_written_ret
        )
    );

    return AST_PTX_SUCCESS;
}

#endif /* AST_PTX_IMPLEMENTATION_ONCE */
#endif /* AST_PTX_IMPLEMENTATION */
