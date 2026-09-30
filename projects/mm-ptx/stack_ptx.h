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
 * @brief This file contains all headers and source for Stack PTX.
 */

#ifndef STACK_PTX_H_INCLUDE
#define STACK_PTX_H_INCLUDE

#define STACK_PTX_VERSION_MAJOR 1 //!< Stack PTX major version.
#define STACK_PTX_VERSION_MINOR 1 //!< Stack PTX minor version.
#define STACK_PTX_VERSION_PATCH 1 //!< Stack PTX patch version.

/**
 * \brief String representation of the Stack PTX library version (e.g., "1.1.1").
 */
#define STACK_PTX_VERSION_STRING "1.1.1"

#define STACK_PTX_VERSION (STACK_PTX_VERSION_MAJOR * 10000 + STACK_PTX_VERSION_MINOR * 100 + STACK_PTX_VERSION_PATCH)

#ifdef __cplusplus
#define STACK_PTX_PUBLIC_DEC extern "C"
#define STACK_PTX_PUBLIC_DEF extern "C"
#else
#define STACK_PTX_PUBLIC_DEC extern
#define STACK_PTX_PUBLIC_DEF
#endif

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
/**
 * \mainpage Stack PTX: A library for compiling valid PTX from this stack instruction language.
 *
 * \section usage Usage
 *
 * Fill this out.
 */

/**
 * \brief Need to pack structs to fit StackPtxInstruction inside 8 bytes.
 */
#define STACK_PTX_PACKED __attribute__((packed))

typedef enum {
    STACK_PTX_SUCCESS                                   = 0,
	/** Stack PTX internal error occured.*/
	STACK_PTX_ERROR_INTERNAL                            = 1,
	/** An unexpected instruction type was processed internally. */
	STACK_PTX_ERROR_WRONG_TYPE_DISPATCH                 = 2,
	/** The buffer passed in is not large enough.*/
	STACK_PTX_ERROR_INSUFFICIENT_BUFFER                 = 3,
	/** An value passed to the function is wrong.*/
	STACK_PTX_ERROR_INVALID_VALUE                       = 4,
	/** AST is not large enough to contain program.*/
	STACK_PTX_ERROR_INSUFFICIENT_AST_SIZE               = 5,
	/** AST visit size is not large enough to traverse AST.*/
	STACK_PTX_ERROR_INSUFFICIENT_AST_VISIT_SIZE         = 6,
	/** A StackPtxInstruction has an invaid instruction type, did you forget to terminate the 
	 * instruction array with a return? */
	STACK_PTX_ERROR_BAD_INSTRUCTION_MAYBE_FORGOT_RETURN = 7,
	/** The buffer passed in is not large enough.*/
	STACK_PTX_ERROR_INSUFFICIENT_WORKSPACE              = 8,
	/** The PTX Instruction index is out of bounds */
	STACK_PTX_ERROR_PTX_INSTRUCTION_IDX_OUT_OF_BOUNDS  	= 9,
	/** The meta instruction index is out of bounds */
	STACK_PTX_ERROR_META_INSTRUCTION_IDX_OUT_OF_BOUNDS  = 10,
	/** The input index is out of bounds */
	STACK_PTX_ERROR_INPUT_IDX_OUT_OF_BOUNDS             = 11,
	/** The register index is out of bounds */
	STACK_PTX_ERROR_REGISTER_IDX_OUT_OF_BOUNDS          = 12,
	/** The special register index is out of bounds */
	STACK_PTX_ERROR_SPECIAL_REGISTER_IDX_OUT_OF_BOUNDS	= 13,
	/** The stack index is out of bounds */
	STACK_PTX_ERROR_STACK_IDX_OUT_OF_BOUNDS             = 14,
	/** The argument index is out of bounds */
	STACK_PTX_ERROR_ARG_IDX_OUT_OF_BOUNDS              	= 15,
	/** The routines index is out of bounds */
	STACK_PTX_ERROR_ROUTINES_IDX_OUT_OF_BOUNDS          = 16,
	/** The store index is out of bounds */
	STACK_PTX_ERROR_STORE_IDX_OUT_OF_BOUNDS             = 17,
	/** The load index is out of bounds */
	STACK_PTX_ERROR_LOAD_IDX_OUT_OF_BOUNDS             	= 18,
	/** The number of result enums.*/
	STACK_PTX_RESULT_NUM_ELEMS                         	= 19
} StackPtxResult;

/**
 * \brief Stack PTX instruction types.
 */
typedef enum {
	STACK_PTX_INSTRUCTION_TYPE_NONE,
	STACK_PTX_INSTRUCTION_TYPE_PTX,
	STACK_PTX_INSTRUCTION_TYPE_CONSTANT,
	STACK_PTX_INSTRUCTION_TYPE_INPUT,
	STACK_PTX_INSTRUCTION_TYPE_SPECIAL,
	STACK_PTX_INSTRUCTION_TYPE_META,
	STACK_PTX_INSTRUCTION_TYPE_META_CONSTANT,
	STACK_PTX_INSTRUCTION_TYPE_ROUTINE,
	STACK_PTX_INSTRUCTION_TYPE_REGISTER,
	STACK_PTX_INSTRUCTION_TYPE_AST_IDX,
	STACK_PTX_INSTRUCTION_TYPE_RETURN,
	STACK_PTX_INSTRUCTION_TYPE_STORE,
	STACK_PTX_INSTRUCTION_TYPE_LOAD,
	STACK_PTX_INSTRUCTION_TYPE_NUM_ENUMS
} StackPtxInstructionType;

/**
 * \brief Stack PTX Meta instruction types.
 *
 * \details Meta instructions manipulate the stack. For example the
 * STACK_PTX_META_INSTRUCTION_DUP instruction operating on the
 * STACK_PTX_STACK_TYPE_F32 stack takes the top value of the STACK_PTX_STACK_TYPE_F32
 * stack and pushes a duplicate on the stack. As with PTX Instructions in Stack PTX, if
 * the required operands are not found within the respective stacks the instruction becomes
 * a no-op.
 */
typedef enum {
	/**
	 * \brief Used to duplicate a value at the top of the respective stack. Should be encoded 
	 * with the function or macro `stack_ptx_encode_meta`.
	 */
	STACK_PTX_META_INSTRUCTION_DUP,
	/**
	 * \brief Used to duplicate a value at a depth indicated by the value popped from the top of
	 * the meta_stack at for the indicated stack. If no value is found in the meta_stack, this is a no-op. Should be encoded 
	 * with the function or macro `stack_ptx_encode_meta`.
	 */
	STACK_PTX_META_INSTRUCTION_YANK_DUP,
	/**
	 * \brief Used to swap the two values at the top of the indicated stack. Should be encoded 
	 * with the function or macro `stack_ptx_encode_meta`.
	 */
	STACK_PTX_META_INSTRUCTION_SWAP,
	/**
	 * \brief Used to swap the value at the top of the indicated stack with the value at a 
	 * depth obtained from the meta_stack. 1,2,3 becomes 2,1,3. 
	 * Should be encoded with the function or macro `stack_ptx_encode_meta`.
	 */
	STACK_PTX_META_INSTRUCTION_SWAP_WITH,
	/**
	 * \brief Replaces the value at the top of the indicated stack with the value at a 
	 * depth obtained from the meta_stack. Should be encoded with the function or macro `stack_ptx_encode_meta`.
	 */
	STACK_PTX_META_INSTRUCTION_REPLACE,
	/**
	 * \brief Drops the top x values of the indicated stack where x is obtained from the meta_stack. 
	 * Should be encoded with the function or macro `stack_ptx_encode_meta`.
	 */
	STACK_PTX_META_INSTRUCTION_DROP,
	/**
	 * \brief Clears all values from the indicated stack.
	 * Should be encoded with the function or macro `stack_ptx_encode_meta`.
	 */
	STACK_PTX_META_INSTRUCTION_CLEAR,
	/**
	 * \brief Takes the value at the top of the indicated stack and pushes it 2 deep.
	 * 1,2,3 becomes 2,3,1.
	 * Should be encoded with the function or macro `stack_ptx_encode_meta`.
	 */
	STACK_PTX_META_INSTRUCTION_ROTATE,
	/**
	 * \brief Reverses the indicated stack. 1,2,3,4 becomes 4,3,2,1.
	 * Should be encoded with the function or macro `stack_ptx_encode_meta`.
	 */
	STACK_PTX_META_INSTRUCTION_REVERSE,
	/**
	 * \brief The number of enums.
	 */
	STACK_PTX_META_INSTRUCTION_NUM_ENUMS
} StackPtxMetaInstruction;

typedef uint32_t 	StackPtxRegisterCounter;
typedef uint32_t 	StackPtxMetaConstant;
typedef uint32_t 	StackPtxAstIdx;

typedef uint16_t 	StackPtxStackIdx;
typedef uint16_t 	StackPtxArgIdx;

typedef uint32_t 	StackPtxIdx;

/**
 * \brief A tagged union for StackPtxInstruction that is packed in 4 bytes.
 */
typedef union STACK_PTX_PACKED {
	uint32_t u;
	int32_t s;
	float f;
	StackPtxMetaConstant meta_constant;
	StackPtxRegisterCounter reg;
	StackPtxAstIdx ast_idx;
} StackPtxPayload;

/**
 * \brief Represents a Stack PTX instruction.
 */
typedef struct STACK_PTX_PACKED {
	uint16_t instruction_type;
	uint16_t aux; // stack_idx for public stack-typed ops, ret_idx for internal return placeholders
	StackPtxPayload payload;
} StackPtxInstruction;

/**
 * \brief Represents a Stack type and how many vector elements.
 * A non-zero num_vec_elems means PTX output will include braces {}.
 */
typedef struct STACK_PTX_PACKED {
	StackPtxStackIdx stack_idx;
	uint16_t num_vec_elems;
} StackPtxArgTypeInfo;

typedef struct STACK_PTX_PACKED {
	StackPtxStackIdx stack_idx;
	uint16_t num_values;
} StackPtxPtxInstructionStackRequirement;

typedef struct STACK_PTX_PACKED {
	StackPtxArgIdx arg_types[4];
	StackPtxArgIdx ret_types[2];
	uint16_t num_unique_stacks;
	bool is_aligned;
	StackPtxPtxInstructionStackRequirement unique_stacks[4];
} StackPtxPtxInstructionDescriptor;

typedef struct STACK_PTX_PACKED {
	StackPtxArgIdx arg_type_idx;
} StackPtxSpecialRegisterDescriptor;

/**
 * \brief Settings for the Stack PTX compiler.
 */
typedef struct {
	size_t max_ast_size;
	size_t max_ast_to_visit_stack_depth;
	size_t stack_size;
	size_t max_frame_depth;
	size_t store_size;
} StackPtxCompilerInfo;

/**
 * \brief Information about how Stack operations are assigned and executed.
 */
typedef struct {
	const char* const* 				ptx_instruction_strings;
	const StackPtxPtxInstructionDescriptor* ptx_instruction_descriptors;
	size_t 							num_ptx_instructions;
	const char* const* 				special_register_strings;
	const StackPtxSpecialRegisterDescriptor* special_register_descriptors;
	size_t 							num_special_registers;
	const char* const*				stack_literal_prefixes;
	size_t 							num_stacks;
	const StackPtxArgTypeInfo* 		arg_type_info;
	size_t 							num_arg_types;
} StackPtxStackInfo;

/**
 * \brief A representation of a register, both PTX name and Stack type.
 */
typedef struct {
	const char* name;
	StackPtxStackIdx stack_idx;
} StackPtxRegister;

/**
 * \brief Converts a StackPtxResult enum value to a human-readable string.
 *
 * \param[in] result The StackPtxResult enum value to convert.
 * \return A null-terminated string describing the result. The string "STACK_PTX_ERROR_INVALID_RESULT_ENUM"
 * is returned if `result` is out of bounds.
 * \remarks Thread-safe.
 */
STACK_PTX_PUBLIC_DEC const char* stack_ptx_result_to_string(StackPtxResult result);

/**
 * \brief Compiles StackPtx instructions into a buffer.
 * 
 * \param[in] compiler_info_ref A pointer to a StackPtxCompilerInfo.
 * \param[in] stack_info_ref A pointer to a StackPtxStackInfo.
 * \param[out] workspace_in_bytes_out The number of bytes to be supplied in the workspace memory.
 *
 */
STACK_PTX_PUBLIC_DEC StackPtxResult stack_ptx_compile_workspace_size(
	const StackPtxCompilerInfo* compiler_info_ref,
	const StackPtxStackInfo* stack_info_ref,
	size_t* workspace_in_bytes_out
);

/**
 * \brief Compiles StackPtx instructions into a buffer.
 * 
 * \details The fields ordered by index are to be referenced by their corresponding StackPtxInstruction 
 * and the index specified by that instruction.
 *
 * \param[in] compiler The StackPtxCompiler to use for compilation.
 * \param[in] compiler_info_ref A pointer to a StackPtxCompilerInfo.
 * \param[in] stack_info_ref A pointer to a StackPtxStackInfo.
 * \param[in] instructions Array of StackPtxInstruction to compile. Terminated with Return instruction.
 * \param[in] registers A pointer to a StackPtxRegister array.
 * \param[in] num_registers The number of elements in the regsiter array.
 * \param[in] routines The routines array ordered by index. Can be NULL if none are used.
 * \param[in] num_routines The number of elements in the routines array.
 * \param[in] requests Array of indices for which register to be assigned as a result.
 * \param[in] num_requests Number of requests in the array.
 * \param[in] execution_limit Limit on execution during compilation.
 * \param[in] workspace The workspace allocated by the user that was measured by `stack_ptx_compile_workspace_size`
 * \param[in] workspace_in_bytes The number of bytes supplied in the workspace memory.
 * \param[out] buffer Buffer to write the compiled output to.
 * \param[in] buffer_size Size of the output buffer.
 * \param[out] buffer_bytes_written_ret Pointer to return the number of bytes written to the buffer.
 * \return StackPtxResult indicating success or failure.
 * \remarks Thread-safe.
 */
STACK_PTX_PUBLIC_DEC StackPtxResult stack_ptx_compile(
	const StackPtxCompilerInfo* compiler_info_ref,
	const StackPtxStackInfo* 	stack_info_ref,
    const StackPtxInstruction* 	instructions,
	const StackPtxRegister* 	registers,
	size_t 						num_registers,
	const StackPtxInstruction** routines,
	size_t 						num_routines,
	const size_t* 				requests,
    size_t 						num_requests,
	size_t 						execution_limit,
	void* 						workspace,
	size_t 						workspace_in_bytes,
	char* 						buffer,
	size_t 						buffer_size,
	size_t* 					buffer_bytes_written_ret
);

#endif // STACK_PTX_H_INCLUDE

#ifdef STACK_PTX_IMPLEMENTATION
#ifndef STACK_PTX_IMPLEMENTATION_ONCE
#define STACK_PTX_IMPLEMENTATION_ONCE

#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define STACK_PTX_STATIC_ASSERT(cond, msg) typedef char static_assertion_##msg[(cond) ? 1 : -1]
#define STACK_PTX_MAX(a, b) ((a) > (b) ? (a) : (b))

#define _STACK_PTX_ALIGNMENT 16 // Standard malloc alignment
#define _STACK_PTX_ALIGNMENT_UP(size, align) (((size) + (align) - 1) & ~((align) - 1))

// StackPtxInstruction should all fit inside 8 bytes
STACK_PTX_STATIC_ASSERT(sizeof(StackPtxInstruction) == 8, stack_ptx_instruction_is_not_eight_bytes);
STACK_PTX_STATIC_ASSERT(STACK_PTX_INSTRUCTION_TYPE_NUM_ENUMS <= UINT16_MAX, stack_ptx_instruction_type_num_enums_must_fit_in_uint16);
STACK_PTX_STATIC_ASSERT(STACK_PTX_META_INSTRUCTION_NUM_ENUMS <= UINT16_MAX, stack_ptx_meta_instruction_num_enums_must_fit_in_uint16);

#define STACK_PTX_MAX_NUM_STACKS UINT16_MAX

static
inline
StackPtxStackIdx
_stack_ptx_instruction_stack_idx(
	StackPtxInstruction instruction
) {
	return (StackPtxStackIdx)instruction.aux;
}

static
inline
size_t
_stack_ptx_instruction_ret_idx(
	StackPtxInstruction instruction
) {
	return (size_t)instruction.aux;
}

static
inline
void
_stack_ptx_instruction_set_stack_idx(
	StackPtxInstruction* instruction,
	StackPtxStackIdx stack_idx
) {
	instruction->aux = stack_idx;
}

static
inline
void
_stack_ptx_instruction_set_ret_idx(
	StackPtxInstruction* instruction,
	size_t ret_idx
) {
	instruction->aux = (uint16_t)ret_idx;
}

static
inline
StackPtxResult
_stack_ptx_check_ret_idx_range(
	size_t ret_idx
) {
	if (ret_idx > UINT16_MAX) {
		return STACK_PTX_ERROR_INVALID_VALUE;
	}
	return STACK_PTX_SUCCESS;
}

/**
 * \brief Can modify to change default Stack PTX tabbing for output PTX.
 */
#ifndef STACK_PTX_TABBING
#define STACK_PTX_TABBING "\t"
#endif

/**
 * \brief The max number of arguments in PTX instructions.
 */
#define STACK_PTX_MAX_NUM_PTX_ARGS 	4
/**
 * \brief The max number of returns in PTX instructions. Uses %p | %q syntax.
 */
#define STACK_PTX_MAX_NUM_PTX_RETS 	2

#ifdef STACK_PTX_DEBUG
#include <assert.h>
#define _STACK_PTX_ERROR(ans)                                                            		\
	do {                                                                                     	\
		StackPtxResult _result = (ans);                                                       	\
		const char* error_name = stack_ptx_result_to_string(_result);                         	\
		fprintf(stderr, "STACK_PTX_ERROR: %s \n  %s %d\n", error_name, __FILE__, __LINE__);  	\
		assert(0);                                                                           	\
		exit(1);                                                                             	\
	} while(0);

#define _STACK_PTX_CHECK_RET(ans)                                                            	 \
	do {                                                                                         \
		StackPtxResult _result = (ans);                                                          \
		if (_result != STACK_PTX_SUCCESS) {                                                      \
			const char* error_name = stack_ptx_result_to_string(_result);                        \
			fprintf(stderr, "STACK_PTX_CHECK: %s \n  %s %d\n", error_name, __FILE__, __LINE__);  \
			assert(0);                                                                           \
			exit(1);                                                                             \
			return _result;                                                                      \
		}                                                                                        \
	} while(0);
#else
#define _STACK_PTX_ERROR(ans)                        		\
	do {                                                 	\
		StackPtxResult _result = (ans);                   	\
		return _result;                                   	\
	} while(0);

#define _STACK_PTX_CHECK_RET(ans)                    		\
	do {                                                 	\
		StackPtxResult _result = (ans);                   	\
		if (_result != STACK_PTX_SUCCESS) return _result;	\
	} while(0);
#endif // STACK_PTX_DEBUG

typedef size_t StackPtxStackPtr;
typedef size_t StackPtxFramePtr;

typedef struct {
	const StackPtxInstruction* instructions;
	StackPtxStackPtr sp;
} StackPtxStackFrame;

typedef struct {
	StackPtxCompilerInfo compiler_info;
	StackPtxStackInfo stack_info;

	const StackPtxRegister* 	registers;
	size_t 						num_registers;
	const StackPtxInstruction** routines;
	size_t 						num_routines;

	StackPtxInstruction* ast;
	StackPtxAstIdx ast_size;
	
	StackPtxAstIdx* ast_to_visit_stack;
	StackPtxAstIdx ast_to_visit_stack_max_depth_usage;
	StackPtxAstIdx ast_to_visit_stack_ptr;

	StackPtxAstIdx* stacks;
	StackPtxStackPtr* stack_ptrs;
	StackPtxStackPtr* request_stack_ptrs;

	// Counters for registers
	StackPtxRegisterCounter* register_counters;

	// For meta instructions
	StackPtxMetaConstant* meta_stack;
	StackPtxStackPtr meta_stack_ptr;

	// For routines
	StackPtxStackFrame* stack_frames;
	StackPtxFramePtr frame_ptr;

	// Store/Load
	StackPtxInstruction* store;
} StackPtxCompiler;

STACK_PTX_PUBLIC_DEF
const char*
stack_ptx_result_to_string(
	StackPtxResult result
) {
	switch(result) {
		case STACK_PTX_SUCCESS:										return "STACK_PTX_SUCCESS";
		case STACK_PTX_ERROR_INTERNAL:								return "STACK_PTX_ERROR_INTERNAL";
		case STACK_PTX_ERROR_WRONG_TYPE_DISPATCH:					return "STACK_PTX_ERROR_WRONG_TYPE_DISPATCH";
		case STACK_PTX_ERROR_INSUFFICIENT_BUFFER:					return "STACK_PTX_ERROR_INSUFFICIENT_BUFFER";
		case STACK_PTX_ERROR_INVALID_VALUE:							return "STACK_PTX_ERROR_INVALID_VALUE";
		case STACK_PTX_ERROR_INSUFFICIENT_AST_SIZE:					return "STACK_PTX_ERROR_INSUFFICIENT_AST_SIZE";
		case STACK_PTX_ERROR_INSUFFICIENT_AST_VISIT_SIZE:			return "STACK_PTX_ERROR_INSUFFICIENT_AST_VISIT_SIZE";
		case STACK_PTX_ERROR_BAD_INSTRUCTION_MAYBE_FORGOT_RETURN: 	return "STACK_PTX_ERROR_BAD_INSTRUCTION_MAYBE_FORGOT_RETURN";
		case STACK_PTX_ERROR_INSUFFICIENT_WORKSPACE:				return "STACK_PTX_ERROR_INSUFFICIENT_WORKSPACE";
		case STACK_PTX_ERROR_PTX_INSTRUCTION_IDX_OUT_OF_BOUNDS:		return "STACK_PTX_ERROR_PTX_INSTRUCTION_IDX_OUT_OF_BOUNDS";
		case STACK_PTX_ERROR_META_INSTRUCTION_IDX_OUT_OF_BOUNDS:	return "STACK_PTX_ERROR_META_INSTRUCTION_IDX_OUT_OF_BOUNDS";
		case STACK_PTX_ERROR_INPUT_IDX_OUT_OF_BOUNDS:				return "STACK_PTX_ERROR_INPUT_IDX_OUT_OF_BOUNDS";
		case STACK_PTX_ERROR_REGISTER_IDX_OUT_OF_BOUNDS:			return "STACK_PTX_ERROR_REGISTER_IDX_OUT_OF_BOUNDS";
		case STACK_PTX_ERROR_SPECIAL_REGISTER_IDX_OUT_OF_BOUNDS:	return "STACK_PTX_ERROR_SPECIAL_REGISTER_IDX_OUT_OF_BOUNDS";
		case STACK_PTX_ERROR_STACK_IDX_OUT_OF_BOUNDS:				return "STACK_PTX_ERROR_STACK_IDX_OUT_OF_BOUNDS";
		case STACK_PTX_ERROR_ARG_IDX_OUT_OF_BOUNDS:					return "STACK_PTX_ERROR_ARG_IDX_OUT_OF_BOUNDS";
		case STACK_PTX_ERROR_ROUTINES_IDX_OUT_OF_BOUNDS:			return "STACK_PTX_ERROR_ROUTINES_IDX_OUT_OF_BOUNDS";
		case STACK_PTX_ERROR_STORE_IDX_OUT_OF_BOUNDS:				return "STACK_PTX_ERROR_STORE_IDX_OUT_OF_BOUNDS";
		case STACK_PTX_ERROR_LOAD_IDX_OUT_OF_BOUNDS:				return "STACK_PTX_ERROR_LOAD_IDX_OUT_OF_BOUNDS";
		case STACK_PTX_RESULT_NUM_ELEMS: break;
	}
	return "STACK_PTX_ERROR_INVALID_RESULT_ENUM";
}

static
inline
StackPtxResult
_stack_ptx_check_stack_type_range(
	const StackPtxCompiler* compiler,
	StackPtxStackIdx stack_idx
) {
	if (stack_idx >= compiler->stack_info.num_stacks) {
		_STACK_PTX_ERROR( STACK_PTX_ERROR_STACK_IDX_OUT_OF_BOUNDS );
	}
	return STACK_PTX_SUCCESS;
}

static
inline
StackPtxResult
_stack_ptx_check_arg_type_range(
	const StackPtxCompiler* compiler,
	StackPtxArgIdx arg_idx
) {
	if (arg_idx >= compiler->stack_info.num_arg_types) {
		_STACK_PTX_ERROR( STACK_PTX_ERROR_ARG_IDX_OUT_OF_BOUNDS );
	}
	return STACK_PTX_SUCCESS;
}

static
inline
size_t
_stack_ptx_arg_type_num_stack_elems(
	const StackPtxCompiler* compiler,
	StackPtxArgIdx arg_idx
) {
	return STACK_PTX_MAX(compiler->stack_info.arg_type_info[arg_idx].num_vec_elems, 1);
}

static
inline
StackPtxResult
_stack_ptx_get_ptx_instruction_descriptor(
	const StackPtxCompiler* compiler,
	StackPtxInstruction instruction,
	StackPtxIdx* instruction_idx_out,
	const StackPtxPtxInstructionDescriptor** descriptor_out
) {
	if (instruction.instruction_type != STACK_PTX_INSTRUCTION_TYPE_PTX) {
		_STACK_PTX_ERROR( STACK_PTX_ERROR_WRONG_TYPE_DISPATCH );
	}
	if (compiler->stack_info.ptx_instruction_descriptors == NULL) {
		_STACK_PTX_ERROR( STACK_PTX_ERROR_PTX_INSTRUCTION_IDX_OUT_OF_BOUNDS );
	}

	StackPtxIdx instruction_idx = instruction.payload.u;
	if (instruction_idx >= compiler->stack_info.num_ptx_instructions) {
		_STACK_PTX_ERROR( STACK_PTX_ERROR_PTX_INSTRUCTION_IDX_OUT_OF_BOUNDS );
	}

	if (instruction_idx_out != NULL) {
		*instruction_idx_out = instruction_idx;
	}
	if (descriptor_out != NULL) {
		*descriptor_out = &compiler->stack_info.ptx_instruction_descriptors[instruction_idx];
	}

	return STACK_PTX_SUCCESS;
}

static
inline
StackPtxResult
_stack_ptx_get_special_register_descriptor(
	const StackPtxCompiler* compiler,
	StackPtxInstruction instruction,
	StackPtxIdx* special_register_idx_out,
	const StackPtxSpecialRegisterDescriptor** descriptor_out
) {
	if (instruction.instruction_type != STACK_PTX_INSTRUCTION_TYPE_SPECIAL) {
		_STACK_PTX_ERROR( STACK_PTX_ERROR_WRONG_TYPE_DISPATCH );
	}
	if (compiler->stack_info.special_register_descriptors == NULL) {
		_STACK_PTX_ERROR( STACK_PTX_ERROR_SPECIAL_REGISTER_IDX_OUT_OF_BOUNDS );
	}

	StackPtxIdx special_register_idx = instruction.payload.u;
	if (special_register_idx >= compiler->stack_info.num_special_registers) {
		_STACK_PTX_ERROR( STACK_PTX_ERROR_SPECIAL_REGISTER_IDX_OUT_OF_BOUNDS );
	}

	if (special_register_idx_out != NULL) {
		*special_register_idx_out = special_register_idx;
	}
	if (descriptor_out != NULL) {
		*descriptor_out = &compiler->stack_info.special_register_descriptors[special_register_idx];
	}

	return STACK_PTX_SUCCESS;
}

static
inline
StackPtxResult
_stack_ptx_snprintf_append(
    char* buffer,
    size_t buffer_size,
    size_t* total_bytes_ref,
    const char* fmt,
    ...
) {
    va_list args;
    va_start(args, fmt);
    int bytes =
		vsnprintf(
			buffer ? buffer + *total_bytes_ref : NULL,
			buffer ? buffer_size - *total_bytes_ref : 0,
			fmt,
			args
		);
    va_end(args);
    if (bytes < 0) {
		_STACK_PTX_ERROR( STACK_PTX_ERROR_INVALID_VALUE );
    }
    *total_bytes_ref += (size_t)bytes;
	if (buffer != NULL && *total_bytes_ref > buffer_size) {
        _STACK_PTX_ERROR( STACK_PTX_ERROR_INSUFFICIENT_BUFFER );
	}
    return STACK_PTX_SUCCESS;
}

__attribute__((warn_unused_result))
static
inline
StackPtxResult
_stack_ptx_print_constant(
	const StackPtxCompiler* compiler,
	StackPtxInstruction instruction,
	char* buffer,
	size_t buffer_size,
	size_t* buffer_bytes_written_ret
) {
	StackPtxStackIdx stack_idx = _stack_ptx_instruction_stack_idx(instruction);
	_STACK_PTX_CHECK_RET( _stack_ptx_check_stack_type_range(compiler, stack_idx) );
	const char* literal_prefix = compiler->stack_info.stack_literal_prefixes[stack_idx];

	const char hex_prefix = strcmp("f32", literal_prefix) == 0 ? 'f' : 'x';
	const char*  const unsigned_hex_postfix = strcmp("u32", literal_prefix) == 0 ? "U" : "";
	uint32_t constant = instruction.payload.u;
	uint32_t* hex_ptr = &constant;

	return _stack_ptx_snprintf_append(
		buffer,
		buffer_size,
		buffer_bytes_written_ret,
		"0%c%08X%s",
		hex_prefix,
		*hex_ptr,
		unsigned_hex_postfix
	);
}

__attribute__((warn_unused_result))
static
inline
StackPtxResult
_stack_ptx_print_register(
	const StackPtxCompiler* compiler,
	StackPtxInstruction instruction,
	char* buffer,
	size_t buffer_size,
	size_t* buffer_bytes_written_ret
) {
	StackPtxStackIdx stack_idx = _stack_ptx_instruction_stack_idx(instruction);
	_STACK_PTX_CHECK_RET( _stack_ptx_check_stack_type_range(compiler, stack_idx) );
	return _stack_ptx_snprintf_append(
		buffer,
		buffer_size,
		buffer_bytes_written_ret,
		"%%_s%u_%u",
		(unsigned)stack_idx,
		(unsigned)instruction.payload.reg
	);
}

static
inline
StackPtxResult
_stack_ptx_print_input(
	const StackPtxCompiler* compiler,
	StackPtxInstruction instruction,
	char* buffer,
	size_t buffer_size,
	size_t* buffer_bytes_written_ret
) {
	StackPtxIdx registers_idx = instruction.payload.u;
	if (registers_idx >= compiler->num_registers) {
		_STACK_PTX_ERROR( STACK_PTX_ERROR_INPUT_IDX_OUT_OF_BOUNDS );
	}

	return _stack_ptx_snprintf_append(
		buffer,
		buffer_size,
		buffer_bytes_written_ret,
		"%%%s",
		compiler->registers[registers_idx].name
	);
}

static
inline
StackPtxResult
_stack_ptx_print_special(
	const StackPtxCompiler* compiler,
	StackPtxInstruction instruction,
	char* buffer,
	size_t buffer_size,
	size_t* buffer_bytes_written_ret
) {
	if (compiler->stack_info.special_register_strings == NULL) {
		_STACK_PTX_ERROR( STACK_PTX_ERROR_SPECIAL_REGISTER_IDX_OUT_OF_BOUNDS );
	}
	StackPtxIdx special_register_idx;
	const StackPtxSpecialRegisterDescriptor* descriptor;
	_STACK_PTX_CHECK_RET(
		_stack_ptx_get_special_register_descriptor(
			compiler,
			instruction,
			&special_register_idx,
			&descriptor
		)
	);

	StackPtxArgIdx arg_idx = descriptor->arg_type_idx;
	_STACK_PTX_CHECK_RET( _stack_ptx_check_arg_type_range(compiler, arg_idx) );
	
	size_t num_vec_elems = compiler->stack_info.arg_type_info[arg_idx].num_vec_elems;

	if (num_vec_elems == 0) {
		_STACK_PTX_CHECK_RET(
			_stack_ptx_snprintf_append(
				buffer,
				buffer_size,
				buffer_bytes_written_ret,
				"%%%s",
				compiler->stack_info.special_register_strings[special_register_idx]
			)
		);
	} else {
		static const char vector_accessor[] = {
			'x','y','z','w'
		};
		_STACK_PTX_CHECK_RET(
			_stack_ptx_snprintf_append(
				buffer,
				buffer_size,
				buffer_bytes_written_ret,
				"%%%s.%c",
				compiler->stack_info.special_register_strings[special_register_idx],
					vector_accessor[_stack_ptx_instruction_ret_idx(instruction)]
			)
		);
	}

	return STACK_PTX_SUCCESS;
}

static
inline
StackPtxResult
_stack_ptx_ptx_instruction_num_args(
	const StackPtxCompiler* compiler,
	StackPtxInstruction instruction,
	size_t* num_args_flat_out,
	size_t* num_args_out
) {
	*num_args_flat_out = 0;
	*num_args_out = 0;

	const StackPtxPtxInstructionDescriptor* descriptor;
	_STACK_PTX_CHECK_RET( _stack_ptx_get_ptx_instruction_descriptor(compiler, instruction, NULL, &descriptor) );

	for (size_t arg_num = 0; arg_num < STACK_PTX_MAX_NUM_PTX_ARGS; arg_num++) {
		StackPtxArgIdx arg_idx = descriptor->arg_types[arg_num];
		if (arg_idx == compiler->stack_info.num_arg_types) {
			break;
		}
		_STACK_PTX_CHECK_RET( _stack_ptx_check_arg_type_range(compiler, arg_idx) );

		size_t num_stack_elems = _stack_ptx_arg_type_num_stack_elems(compiler, arg_idx);
		*num_args_flat_out += num_stack_elems;
		*num_args_out += 1;
	}

	return STACK_PTX_SUCCESS;
}

static
inline
StackPtxResult
_stack_ptx_ptx_instruction_num_rets(
	const StackPtxCompiler* compiler,
	StackPtxInstruction instruction,
	size_t* num_rets_flat_out,
	size_t* num_rets_out
) {
	*num_rets_flat_out = 0;
	*num_rets_out = 0;

	const StackPtxPtxInstructionDescriptor* descriptor;
	_STACK_PTX_CHECK_RET( _stack_ptx_get_ptx_instruction_descriptor(compiler, instruction, NULL, &descriptor) );

	for (size_t arg_num = 0; arg_num < STACK_PTX_MAX_NUM_PTX_RETS; arg_num++) {
		StackPtxArgIdx arg_idx = descriptor->ret_types[arg_num];
		if (arg_idx == compiler->stack_info.num_arg_types) {
			break;
		}

		_STACK_PTX_CHECK_RET( _stack_ptx_check_arg_type_range(compiler, arg_idx) );

		size_t num_stack_elems = _stack_ptx_arg_type_num_stack_elems(compiler, arg_idx);
		*num_rets_flat_out += num_stack_elems;
		*num_rets_out += 1;
	}

	return STACK_PTX_SUCCESS;
}

static
inline
StackPtxResult
_stack_ptx_ast_run_ptx(
	StackPtxCompiler* compiler,
	StackPtxInstruction instruction
) {
	if (instruction.instruction_type != STACK_PTX_INSTRUCTION_TYPE_PTX) {
		_STACK_PTX_ERROR( STACK_PTX_ERROR_WRONG_TYPE_DISPATCH );
	}

	const StackPtxPtxInstructionDescriptor* descriptor;
	_STACK_PTX_CHECK_RET( _stack_ptx_get_ptx_instruction_descriptor(compiler, instruction, NULL, &descriptor) );

	size_t num_args_flat, num_args_unused;
	size_t num_rets_flat, num_rets;
	_STACK_PTX_CHECK_RET( _stack_ptx_ptx_instruction_num_args(compiler, instruction, &num_args_flat, &num_args_unused) );
	_STACK_PTX_CHECK_RET( _stack_ptx_ptx_instruction_num_rets(compiler, instruction, &num_rets_flat, &num_rets) );
	(void)num_args_unused;
	_STACK_PTX_CHECK_RET( _stack_ptx_check_ret_idx_range(num_rets_flat) );

	for (size_t i = 0; i < descriptor->num_unique_stacks; i++) {
		StackPtxStackIdx stack_idx = descriptor->unique_stacks[i].stack_idx;
		_STACK_PTX_CHECK_RET( _stack_ptx_check_stack_type_range(compiler, stack_idx) );
		if (compiler->stack_ptrs[stack_idx] < descriptor->unique_stacks[i].num_values) {
			return STACK_PTX_SUCCESS;
		}
	}

	if (compiler->ast_size + num_args_flat + num_rets_flat > compiler->compiler_info.max_ast_size) {
		_STACK_PTX_ERROR( STACK_PTX_ERROR_INSUFFICIENT_AST_SIZE );
	}

	for (size_t arg_num = 0; arg_num < STACK_PTX_MAX_NUM_PTX_ARGS; arg_num++) {
		StackPtxArgIdx arg_idx = descriptor->arg_types[arg_num];
		if (arg_idx == compiler->stack_info.num_arg_types) {
			break;
		}

		_STACK_PTX_CHECK_RET( _stack_ptx_check_arg_type_range(compiler, arg_idx) );

		StackPtxStackIdx stack_idx = compiler->stack_info.arg_type_info[arg_idx].stack_idx;
		_STACK_PTX_CHECK_RET( _stack_ptx_check_stack_type_range(compiler, stack_idx) );
		size_t num_stack_elems = _stack_ptx_arg_type_num_stack_elems(compiler, arg_idx);
		for (size_t i = 0; i < num_stack_elems; i++) {
			StackPtxStackPtr stack_ptr = --compiler->stack_ptrs[stack_idx];
			StackPtxAstIdx ast_idx = compiler->stacks[stack_idx * compiler->compiler_info.stack_size + stack_ptr];
			StackPtxInstruction ast_instruction = {};

			ast_instruction.instruction_type = STACK_PTX_INSTRUCTION_TYPE_AST_IDX;
			ast_instruction.payload.ast_idx = ast_idx;

			compiler->ast[compiler->ast_size++] = ast_instruction;
		}
	}

	// Now that we have the number of total return arguments
	// We need to write a copy of the instruction for each of the ret args.
	// This gives a target in the stack values for which return value they belong to
	// We can then rewrite these instructions in place as registers if they get their assignments
	size_t ret_idx = 0;
	// Return placeholders are rewritten later from the last return AST entry backward.
	// Emit stack placeholders in reverse return order so each typed stack points at
	// the matching rewritten return register.
	for (size_t ret_num_rev = num_rets; ret_num_rev > 0; ret_num_rev--) {
		size_t ret_num = ret_num_rev - 1;
		StackPtxArgIdx arg_idx = descriptor->ret_types[ret_num];

		_STACK_PTX_CHECK_RET( _stack_ptx_check_arg_type_range(compiler, arg_idx) );
		
		StackPtxStackIdx stack_idx = compiler->stack_info.arg_type_info[arg_idx].stack_idx;
		_STACK_PTX_CHECK_RET( _stack_ptx_check_stack_type_range(compiler, stack_idx) );
		size_t num_stack_elems = _stack_ptx_arg_type_num_stack_elems(compiler, arg_idx);
		for (size_t i = 0; i < num_stack_elems; i++) {
			_stack_ptx_instruction_set_ret_idx(&instruction, num_rets_flat - ret_idx - 1);
			ret_idx++;
			StackPtxAstIdx ast_idx = compiler->ast_size++;
			compiler->ast[ast_idx] = instruction;
			if (compiler->stack_ptrs[stack_idx] < compiler->compiler_info.stack_size) {
				StackPtxStackPtr stack_ptr = compiler->stack_ptrs[stack_idx]++;
				compiler->stacks[stack_idx * compiler->compiler_info.stack_size + stack_ptr] = ast_idx;
			}
		}
	}

	return STACK_PTX_SUCCESS;
}

static
inline
StackPtxResult
_stack_ptx_ast_run_meta_constant(
	StackPtxCompiler* compiler,
	StackPtxInstruction instruction
) {
	if (instruction.instruction_type != STACK_PTX_INSTRUCTION_TYPE_META_CONSTANT) {
		_STACK_PTX_ERROR( STACK_PTX_ERROR_WRONG_TYPE_DISPATCH );
	}

	if (compiler->meta_stack_ptr < compiler->compiler_info.stack_size) {
		compiler->meta_stack[compiler->meta_stack_ptr++] = instruction.payload.meta_constant;
	}

	return STACK_PTX_SUCCESS;
}

static
inline
StackPtxResult
_stack_ptx_ast_run_meta(
	StackPtxCompiler* compiler,
	StackPtxInstruction instruction
) {
	if (instruction.instruction_type != STACK_PTX_INSTRUCTION_TYPE_META) {
		_STACK_PTX_ERROR( STACK_PTX_ERROR_WRONG_TYPE_DISPATCH );
	}

	StackPtxStackIdx stack_idx = _stack_ptx_instruction_stack_idx(instruction);
	_STACK_PTX_CHECK_RET( _stack_ptx_check_stack_type_range(compiler, stack_idx) );
	StackPtxMetaInstruction meta_idx = (StackPtxMetaInstruction)instruction.payload.u;
	StackPtxStackPtr*  const meta_stack_ptr = &compiler->meta_stack_ptr;
	StackPtxMetaConstant*  const meta_stack = compiler->meta_stack;

	if (stack_idx >= compiler->stack_info.num_stacks) {
		_STACK_PTX_ERROR( STACK_PTX_ERROR_INVALID_VALUE );
	}

	StackPtxStackPtr* const stack_ptr = &compiler->stack_ptrs[stack_idx];
	StackPtxAstIdx* stack = &compiler->stacks[stack_idx * compiler->compiler_info.stack_size];

		switch(meta_idx) {
			case STACK_PTX_META_INSTRUCTION_DUP: {
				// Duplicate the register at the top of the respective stack
				if ((*stack_ptr) > 0 && (*stack_ptr) < compiler->compiler_info.stack_size) {
					StackPtxAstIdx ast_idx = stack[(*stack_ptr)-1];
					stack[(*stack_ptr)++] = ast_idx;
				}
				return STACK_PTX_SUCCESS;
			};
		case STACK_PTX_META_INSTRUCTION_YANK_DUP: {
			if ((*meta_stack_ptr) > 0) {
				StackPtxMetaConstant depth = meta_stack[--(*meta_stack_ptr)];
				if ((*stack_ptr) > depth && (*stack_ptr) < compiler->compiler_info.stack_size) {
					StackPtxStackPtr stack_idx = (*stack_ptr) - depth - 1;
					StackPtxAstIdx ast_idx = stack[stack_idx];
					stack[(*stack_ptr)++] = ast_idx;
				}
			}
			return STACK_PTX_SUCCESS;
		};
		case STACK_PTX_META_INSTRUCTION_SWAP: {
			if ((*stack_ptr) > 1) {
				StackPtxAstIdx ast_idx_top = stack[--(*stack_ptr)];
				StackPtxAstIdx ast_idx_bottom = stack[--(*stack_ptr)];
				stack[(*stack_ptr)++] = ast_idx_top;
				stack[(*stack_ptr)++] = ast_idx_bottom;
			}
			return STACK_PTX_SUCCESS;
		};
		case STACK_PTX_META_INSTRUCTION_SWAP_WITH: {
			if ((*meta_stack_ptr) > 0) {
				StackPtxMetaConstant depth = meta_stack[--(*meta_stack_ptr)];
				if ((*stack_ptr) > depth && (*stack_ptr) < compiler->compiler_info.stack_size) {
					StackPtxStackPtr stack_idx = (*stack_ptr) - depth - 1;
					StackPtxAstIdx current_top = stack[(*stack_ptr)-1];
					StackPtxAstIdx at_depth = stack[stack_idx];
					stack[(*stack_ptr)-1] = at_depth;
					stack[stack_idx] = current_top;
				}
			}
			return STACK_PTX_SUCCESS;
		};
		case STACK_PTX_META_INSTRUCTION_REPLACE: {
			if ((*meta_stack_ptr) > 0) {
				StackPtxMetaConstant depth = meta_stack[--(*meta_stack_ptr)];
				if ((*stack_ptr) > depth && (*stack_ptr) < compiler->compiler_info.stack_size) {
					StackPtxStackPtr stack_idx = (*stack_ptr) - depth - 1;
					StackPtxAstIdx current_top = stack[(*stack_ptr)-1];
					stack[stack_idx] = current_top;
					stack[--(*stack_ptr)] = 0;
				}
			}
			return STACK_PTX_SUCCESS;
		};
		case STACK_PTX_META_INSTRUCTION_DROP: {
			if ((*meta_stack_ptr) > 0) {
				StackPtxMetaConstant amount = meta_stack[--(*meta_stack_ptr)];
				if ((*stack_ptr) > amount) {
					for (size_t d = 0; d < amount; d++) {
						stack[--(*stack_ptr)] = 0;
					}
				}
			}
			return STACK_PTX_SUCCESS;
		};
		case STACK_PTX_META_INSTRUCTION_CLEAR: {
			for (size_t i = 0; i < (*stack_ptr); i++) {
				stack[i] = 0;
			}
			(*stack_ptr) = 0;
			return STACK_PTX_SUCCESS;
		};
		case STACK_PTX_META_INSTRUCTION_ROTATE: {
			if ((*stack_ptr) > 2) {
				StackPtxAstIdx ast_idx_top = stack[--(*stack_ptr)];
				StackPtxAstIdx ast_idx_middle = stack[--(*stack_ptr)];
				StackPtxAstIdx ast_idx_bottom = stack[--(*stack_ptr)];
				stack[(*stack_ptr)++] = ast_idx_top;
				stack[(*stack_ptr)++] = ast_idx_bottom;
				stack[(*stack_ptr)++] = ast_idx_middle;
			}
			return STACK_PTX_SUCCESS;
		};
		case STACK_PTX_META_INSTRUCTION_REVERSE: {
			if ((*stack_ptr) == 0) {
				return STACK_PTX_SUCCESS;
			}
			size_t start = 0;
			size_t end = (*stack_ptr) - 1;
			StackPtxAstIdx temp;

			while (start < end) {
				temp = stack[start];
				stack[start] = stack[end];
				stack[end] = temp;

				start++;
				end--;
				}
				return STACK_PTX_SUCCESS;
			};
			case STACK_PTX_META_INSTRUCTION_NUM_ENUMS: break;
		}
		_STACK_PTX_ERROR( STACK_PTX_ERROR_META_INSTRUCTION_IDX_OUT_OF_BOUNDS );
}

static
inline
StackPtxResult
_stack_ptx_ast_run_store(
	StackPtxCompiler* compiler,
	StackPtxInstruction instruction
) {
	if (instruction.instruction_type != STACK_PTX_INSTRUCTION_TYPE_STORE) {
		_STACK_PTX_ERROR( STACK_PTX_ERROR_WRONG_TYPE_DISPATCH );
	}

	StackPtxStackIdx stack_idx = _stack_ptx_instruction_stack_idx(instruction);
	_STACK_PTX_CHECK_RET( _stack_ptx_check_stack_type_range(compiler, stack_idx) );
	StackPtxIdx store_idx = instruction.payload.u;

	if (store_idx >= compiler->compiler_info.store_size) {
		_STACK_PTX_ERROR( STACK_PTX_ERROR_STORE_IDX_OUT_OF_BOUNDS );
	}

	if (compiler->stack_ptrs[stack_idx] > 0) {
		StackPtxStackPtr stack_ptr = --compiler->stack_ptrs[stack_idx];
		StackPtxAstIdx ast_idx = compiler->stacks[stack_idx * compiler->compiler_info.stack_size + stack_ptr];
		StackPtxInstruction instruction = {};
		instruction.instruction_type = STACK_PTX_INSTRUCTION_TYPE_STORE;
			_stack_ptx_instruction_set_stack_idx(&instruction, stack_idx);
		instruction.payload.ast_idx = ast_idx;

		compiler->store[store_idx] = instruction;
	}

	return STACK_PTX_SUCCESS;
}

static
inline
StackPtxResult
_stack_ptx_ast_run_load(
	StackPtxCompiler* compiler,
	StackPtxInstruction instruction
) {
	if (instruction.instruction_type != STACK_PTX_INSTRUCTION_TYPE_LOAD) {
		_STACK_PTX_ERROR( STACK_PTX_ERROR_WRONG_TYPE_DISPATCH );
	}

	StackPtxIdx load_idx = instruction.payload.u;

	if (load_idx >= compiler->compiler_info.store_size) {
		_STACK_PTX_ERROR( STACK_PTX_ERROR_LOAD_IDX_OUT_OF_BOUNDS );
	}

	instruction = compiler->store[load_idx];
	StackPtxStackIdx stack_idx = _stack_ptx_instruction_stack_idx(instruction);
	_STACK_PTX_CHECK_RET( _stack_ptx_check_stack_type_range(compiler, stack_idx) );
	if (instruction.instruction_type != STACK_PTX_INSTRUCTION_TYPE_NONE) {
		StackPtxStackPtr* const stack_ptr = &compiler->stack_ptrs[stack_idx];
		StackPtxAstIdx* stack = &compiler->stacks[stack_idx * compiler->compiler_info.stack_size];
	
		if ((*stack_ptr) < compiler->compiler_info.stack_size) {
			stack[(*stack_ptr)++] = instruction.payload.ast_idx;
		}

	}

	return STACK_PTX_SUCCESS;
}

static
inline
StackPtxResult
_stack_ptx_ast_run(
	StackPtxCompiler* compiler,
	size_t execution_limit
) {
	for (size_t step = 0; step < execution_limit; step++) {
		StackPtxStackFrame* stack_frame = &compiler->stack_frames[compiler->frame_ptr];

		StackPtxInstruction instruction = stack_frame->instructions[stack_frame->sp++];
		StackPtxInstructionType instruction_type = (StackPtxInstructionType)instruction.instruction_type;

		switch(instruction_type) {
			case STACK_PTX_INSTRUCTION_TYPE_RETURN: {
				if (compiler->frame_ptr-- <= 0) {
					return STACK_PTX_SUCCESS;
				}
			} break;
			case STACK_PTX_INSTRUCTION_TYPE_INPUT: {
				StackPtxIdx register_idx = instruction.payload.u;
				if (compiler->registers == NULL || register_idx >= compiler->num_registers) {
					_STACK_PTX_ERROR( STACK_PTX_ERROR_REGISTER_IDX_OUT_OF_BOUNDS );
				}
				// If it's an input then we grab the stack type from the declared registers.
				StackPtxStackIdx stack_idx = compiler->registers[register_idx].stack_idx;
				_STACK_PTX_CHECK_RET( _stack_ptx_check_stack_type_range(compiler, stack_idx) );
				if (compiler->ast_size >= compiler->compiler_info.max_ast_size) {
					_STACK_PTX_ERROR( STACK_PTX_ERROR_INSUFFICIENT_AST_SIZE );
				}
				StackPtxAstIdx idx = compiler->ast_size++;
				compiler->ast[idx] = instruction;
				if (compiler->stack_ptrs[stack_idx] < compiler->compiler_info.stack_size) {
					StackPtxStackPtr stack_ptr = compiler->stack_ptrs[stack_idx]++;
					compiler->stacks[stack_idx * compiler->compiler_info.stack_size + stack_ptr] = idx;
				}
			} break;
			case STACK_PTX_INSTRUCTION_TYPE_CONSTANT: {
					StackPtxStackIdx stack_idx = _stack_ptx_instruction_stack_idx(instruction);
				_STACK_PTX_CHECK_RET( _stack_ptx_check_stack_type_range(compiler, stack_idx) );
				if (compiler->ast_size >= compiler->compiler_info.max_ast_size) {
					_STACK_PTX_ERROR( STACK_PTX_ERROR_INSUFFICIENT_AST_SIZE );
				}
				StackPtxAstIdx idx = compiler->ast_size++;
				compiler->ast[idx] = instruction;
				if (compiler->stack_ptrs[stack_idx] < compiler->compiler_info.stack_size) {
					StackPtxStackPtr stack_ptr = compiler->stack_ptrs[stack_idx]++;
					compiler->stacks[stack_idx * compiler->compiler_info.stack_size + stack_ptr] = idx;
				}
			} break;
			case STACK_PTX_INSTRUCTION_TYPE_SPECIAL: {
				const StackPtxSpecialRegisterDescriptor* descriptor;
				_STACK_PTX_CHECK_RET(
					_stack_ptx_get_special_register_descriptor(compiler, instruction, NULL, &descriptor)
				);
				StackPtxArgIdx arg_idx = descriptor->arg_type_idx;
				_STACK_PTX_CHECK_RET( _stack_ptx_check_arg_type_range(compiler, arg_idx) );

				StackPtxStackIdx stack_idx = compiler->stack_info.arg_type_info[arg_idx].stack_idx;
				_STACK_PTX_CHECK_RET( _stack_ptx_check_stack_type_range(compiler, stack_idx) );
				size_t num_stack_elems = _stack_ptx_arg_type_num_stack_elems(compiler, arg_idx);
				if (compiler->ast_size + num_stack_elems > compiler->compiler_info.max_ast_size) {
					_STACK_PTX_ERROR( STACK_PTX_ERROR_INSUFFICIENT_AST_SIZE );
				}
				for (size_t i = 0; i < num_stack_elems; i++) {
					StackPtxAstIdx idx = compiler->ast_size++;
						_stack_ptx_instruction_set_ret_idx(&instruction, i);
					compiler->ast[idx] = instruction;
					if (compiler->stack_ptrs[stack_idx] < compiler->compiler_info.stack_size) {
						StackPtxStackPtr stack_ptr = compiler->stack_ptrs[stack_idx]++;
						compiler->stacks[stack_idx * compiler->compiler_info.stack_size + stack_ptr] = idx;
					}
				}
			} break;
			case STACK_PTX_INSTRUCTION_TYPE_PTX: {
				_STACK_PTX_CHECK_RET( _stack_ptx_ast_run_ptx(compiler, instruction) );
			} break;
			case STACK_PTX_INSTRUCTION_TYPE_META: {
				_STACK_PTX_CHECK_RET( _stack_ptx_ast_run_meta(compiler, instruction) );
			} break;
			case STACK_PTX_INSTRUCTION_TYPE_META_CONSTANT: {
				_STACK_PTX_CHECK_RET( _stack_ptx_ast_run_meta_constant(compiler, instruction) );
			} break;
			case STACK_PTX_INSTRUCTION_TYPE_STORE: {
				_STACK_PTX_CHECK_RET( _stack_ptx_ast_run_store(compiler, instruction) );
			} break;
			case STACK_PTX_INSTRUCTION_TYPE_LOAD: {
				_STACK_PTX_CHECK_RET( _stack_ptx_ast_run_load(compiler, instruction) );
			} break;
			case STACK_PTX_INSTRUCTION_TYPE_ROUTINE: {
				if (compiler->routines == NULL) {
					_STACK_PTX_ERROR( STACK_PTX_ERROR_ROUTINES_IDX_OUT_OF_BOUNDS );
				}

				StackPtxIdx routine_idx = instruction.payload.u;
				if (routine_idx >= compiler->num_routines) {
					_STACK_PTX_ERROR( STACK_PTX_ERROR_ROUTINES_IDX_OUT_OF_BOUNDS );
				}
				const StackPtxInstruction* routine = compiler->routines[routine_idx];

				if (routine == NULL) {
					_STACK_PTX_ERROR( STACK_PTX_ERROR_ROUTINES_IDX_OUT_OF_BOUNDS );
				}

				if (compiler->frame_ptr >= compiler->compiler_info.max_frame_depth - 1) {
					break;
				}

				StackPtxStackFrame* stack_frame = &compiler->stack_frames[++compiler->frame_ptr];
				stack_frame->instructions = routine;
				stack_frame->sp = 0;
			} break;
			default:
				_STACK_PTX_ERROR( STACK_PTX_ERROR_BAD_INSTRUCTION_MAYBE_FORGOT_RETURN );
		}
	}

	return STACK_PTX_SUCCESS;
}

static
inline
StackPtxResult
_stack_ptx_compile_ptx(
	StackPtxCompiler* compiler,
	StackPtxInstruction instruction,
	StackPtxAstIdx ast_idx,
	char* buffer,
	size_t buffer_size,
	size_t* buffer_bytes_written_ret
) {
	if (instruction.instruction_type != STACK_PTX_INSTRUCTION_TYPE_PTX) {
		_STACK_PTX_ERROR( STACK_PTX_ERROR_WRONG_TYPE_DISPATCH );
	}

	if (compiler->stack_info.ptx_instruction_strings == NULL) {
		_STACK_PTX_ERROR( STACK_PTX_ERROR_PTX_INSTRUCTION_IDX_OUT_OF_BOUNDS );
	}

	const size_t ret_start_idx = ast_idx + _stack_ptx_instruction_ret_idx(instruction);
	instruction = compiler->ast[ret_start_idx];
	StackPtxIdx instruction_idx;
	const StackPtxPtxInstructionDescriptor* descriptor;
	_STACK_PTX_CHECK_RET(
		_stack_ptx_get_ptx_instruction_descriptor(compiler, instruction, &instruction_idx, &descriptor)
	);

	size_t num_args_flat, num_args, num_rets_flat, num_rets;
	_STACK_PTX_CHECK_RET( _stack_ptx_ptx_instruction_num_args(compiler, instruction, &num_args_flat, &num_args) );
	_STACK_PTX_CHECK_RET( _stack_ptx_ptx_instruction_num_rets(compiler, instruction, &num_rets_flat, &num_rets) );
	const size_t args_start_idx = ret_start_idx - num_rets_flat;

	bool all_evalutated = true;
	for (size_t i = 0; i < num_args_flat; i++) {
		size_t args_idx = args_start_idx - i;
		StackPtxInstruction arg_instruction = compiler->ast[args_idx];
		if (arg_instruction.instruction_type != STACK_PTX_INSTRUCTION_TYPE_AST_IDX) {
			_STACK_PTX_ERROR( STACK_PTX_ERROR_WRONG_TYPE_DISPATCH );
		}
		StackPtxAstIdx arg_ast_idx = arg_instruction.payload.ast_idx;
		arg_instruction = compiler->ast[arg_ast_idx];
		switch(arg_instruction.instruction_type) {
			case STACK_PTX_INSTRUCTION_TYPE_CONSTANT:
			case STACK_PTX_INSTRUCTION_TYPE_REGISTER:
			case STACK_PTX_INSTRUCTION_TYPE_INPUT:
				break;
			case STACK_PTX_INSTRUCTION_TYPE_SPECIAL:
			case STACK_PTX_INSTRUCTION_TYPE_PTX: {
				if (all_evalutated) {
					all_evalutated = false;
					// This is the first argument we've encountered that needs to be evaluated
					// Make sure this node gets evaluated again after the arguments are evaluated
					if (compiler->ast_to_visit_stack_ptr >= compiler->compiler_info.max_ast_to_visit_stack_depth) {
						_STACK_PTX_ERROR( STACK_PTX_ERROR_INSUFFICIENT_AST_VISIT_SIZE );
					}
					compiler->ast_to_visit_stack[compiler->ast_to_visit_stack_ptr++] = ast_idx;
					compiler->ast_to_visit_stack_max_depth_usage =
						STACK_PTX_MAX(
							compiler->ast_to_visit_stack_max_depth_usage,
							compiler->ast_to_visit_stack_ptr
						);
				}
				if (compiler->ast_to_visit_stack_ptr >= compiler->compiler_info.max_ast_to_visit_stack_depth) {
					_STACK_PTX_ERROR( STACK_PTX_ERROR_INSUFFICIENT_AST_VISIT_SIZE );
				}
				compiler->ast_to_visit_stack[compiler->ast_to_visit_stack_ptr++] = arg_ast_idx;
				compiler->ast_to_visit_stack_max_depth_usage =
					STACK_PTX_MAX(
						compiler->ast_to_visit_stack_max_depth_usage,
						compiler->ast_to_visit_stack_ptr
					);
			} break;
			default:
				_STACK_PTX_ERROR( STACK_PTX_ERROR_INVALID_VALUE );
		}
	}

	if (!all_evalutated) {
		return STACK_PTX_SUCCESS;
	}

	_STACK_PTX_CHECK_RET(
		_stack_ptx_snprintf_append(
			buffer,
			buffer_size,
			buffer_bytes_written_ret,
			STACK_PTX_TABBING "%s ",
			compiler->stack_info.ptx_instruction_strings[instruction_idx]
		)
	);
	// Now go through all rets and rewrite them to register assignments
	// Start printing out ptx assembly string
	size_t ret_ast_idx = 0;
	for (size_t arg_num = 0; arg_num < num_rets; arg_num++) {
		StackPtxArgIdx arg_idx = descriptor->ret_types[arg_num];
		_STACK_PTX_CHECK_RET( _stack_ptx_check_arg_type_range(compiler, arg_idx) );
		
		StackPtxStackIdx stack_idx = compiler->stack_info.arg_type_info[arg_idx].stack_idx;
		_STACK_PTX_CHECK_RET( _stack_ptx_check_stack_type_range(compiler, stack_idx) );
		size_t num_vec_elems = compiler->stack_info.arg_type_info[arg_idx].num_vec_elems;
		size_t num_stack_elems = _stack_ptx_arg_type_num_stack_elems(compiler, arg_idx);

		if (arg_num > 0) {
			_STACK_PTX_CHECK_RET(
				_stack_ptx_snprintf_append(
					buffer,
					buffer_size,
					buffer_bytes_written_ret,
					"|"
				)
			);
		}
		if (num_vec_elems > 0) {
			_STACK_PTX_CHECK_RET(
				_stack_ptx_snprintf_append(
					buffer,
					buffer_size,
					buffer_bytes_written_ret,
					"\n\t\t{"
				)
			);
		}
		for (size_t i = 0; i < num_stack_elems; i++) {
			StackPtxInstruction this_ret_instruction = {};

				this_ret_instruction.instruction_type = STACK_PTX_INSTRUCTION_TYPE_REGISTER;
				_stack_ptx_instruction_set_stack_idx(&this_ret_instruction, stack_idx);
				this_ret_instruction.payload.reg = compiler->register_counters[stack_idx]++;

			if (i != 0) {
				_STACK_PTX_CHECK_RET(
					_stack_ptx_snprintf_append(
						buffer,
						buffer_size,
						buffer_bytes_written_ret,
						", "
					)
				);
			}

			_STACK_PTX_CHECK_RET(
				_stack_ptx_print_register(
					compiler,
					this_ret_instruction,
					buffer,
					buffer_size,
					buffer_bytes_written_ret
				)
			);

			compiler->ast[ret_start_idx - ret_ast_idx] = this_ret_instruction;
			ret_ast_idx++;
		}
		if (num_vec_elems > 0) {
			_STACK_PTX_CHECK_RET(
				_stack_ptx_snprintf_append(
					buffer,
					buffer_size,
					buffer_bytes_written_ret,
					"}"
				)
			);
		}
	}

	size_t arg_ast_idx = 0;
	for (size_t i = 0; i < num_args; i++) {
		StackPtxArgIdx arg_idx = descriptor->arg_types[i];

		_STACK_PTX_CHECK_RET( _stack_ptx_check_arg_type_range(compiler, arg_idx) );
		size_t num_vec_elems = compiler->stack_info.arg_type_info[arg_idx].num_vec_elems;
		size_t num_stack_elems = _stack_ptx_arg_type_num_stack_elems(compiler, arg_idx);
		const char* maybe_vector_start_bracket = num_vec_elems > 0 ? "\n\t\t{" : "";
		const char* maybe_vector_end_bracket = num_vec_elems > 0 ? "}" : "";

		_STACK_PTX_CHECK_RET(
			_stack_ptx_snprintf_append(
				buffer,
				buffer_size,
				buffer_bytes_written_ret,
				", %s",
				maybe_vector_start_bracket
			)
		);

		for (size_t n = 0; n < num_stack_elems; n++) {
			const char* maybe_comma_string = n == 0 ? "" : ", ";
			_STACK_PTX_CHECK_RET(
				_stack_ptx_snprintf_append(
					buffer,
					buffer_size,
					buffer_bytes_written_ret,
					"%s",
					maybe_comma_string
				)
			);
			StackPtxInstruction arg_instruction = compiler->ast[args_start_idx - num_args_flat + 1 + arg_ast_idx];
			if (arg_instruction.instruction_type != STACK_PTX_INSTRUCTION_TYPE_AST_IDX) {
				_STACK_PTX_ERROR( STACK_PTX_ERROR_WRONG_TYPE_DISPATCH );
			}
			arg_instruction = compiler->ast[arg_instruction.payload.ast_idx];
			switch(arg_instruction.instruction_type) {
				case STACK_PTX_INSTRUCTION_TYPE_INPUT: {
					_STACK_PTX_CHECK_RET(
						_stack_ptx_print_input(compiler, arg_instruction, buffer, buffer_size, buffer_bytes_written_ret)
					);
				} break;
				case STACK_PTX_INSTRUCTION_TYPE_REGISTER: {
					_STACK_PTX_CHECK_RET(
						_stack_ptx_print_register(compiler, arg_instruction, buffer, buffer_size, buffer_bytes_written_ret)
					);
				} break;
				case STACK_PTX_INSTRUCTION_TYPE_CONSTANT: {
					_STACK_PTX_CHECK_RET(
						_stack_ptx_print_constant(compiler, arg_instruction, buffer, buffer_size, buffer_bytes_written_ret)
					);
				} break;
				default:
					_STACK_PTX_ERROR( STACK_PTX_ERROR_INVALID_VALUE );
			}
			arg_ast_idx++;

		}
		_STACK_PTX_CHECK_RET(
			_stack_ptx_snprintf_append(
				buffer,
				buffer_size,
				buffer_bytes_written_ret,
				"%s",
				maybe_vector_end_bracket
			)
		);
	}
	_STACK_PTX_CHECK_RET(
		_stack_ptx_snprintf_append(
			buffer,
			buffer_size,
			buffer_bytes_written_ret,
			";\n"
		)
	);

	return STACK_PTX_SUCCESS;
}

static
inline
StackPtxResult
_stack_ptx_compile_special(
	StackPtxCompiler* compiler,
	StackPtxInstruction instruction,
	StackPtxAstIdx ast_idx,
	char* buffer,
	size_t buffer_size,
	size_t* buffer_bytes_written_ret
) {
	const StackPtxSpecialRegisterDescriptor* descriptor;
	_STACK_PTX_CHECK_RET(
		_stack_ptx_get_special_register_descriptor(compiler, instruction, NULL, &descriptor)
	);

	StackPtxArgIdx arg_idx = descriptor->arg_type_idx;
	_STACK_PTX_CHECK_RET( _stack_ptx_check_arg_type_range(compiler, arg_idx) );

	StackPtxStackIdx stack_idx = compiler->stack_info.arg_type_info[arg_idx].stack_idx;
	_STACK_PTX_CHECK_RET( _stack_ptx_check_stack_type_range(compiler, stack_idx) );

	StackPtxInstruction this_register_instruction = {};
	this_register_instruction.instruction_type = STACK_PTX_INSTRUCTION_TYPE_REGISTER;
	_stack_ptx_instruction_set_stack_idx(&this_register_instruction, stack_idx);
	this_register_instruction.payload.reg = compiler->register_counters[stack_idx]++;

	const char* literal_prefix = compiler->stack_info.stack_literal_prefixes[stack_idx];
	_STACK_PTX_CHECK_RET(
		_stack_ptx_snprintf_append(
			buffer,
			buffer_size,
			buffer_bytes_written_ret,
			STACK_PTX_TABBING "mov.%s ",
			literal_prefix
		)
	);

	_STACK_PTX_CHECK_RET(
		_stack_ptx_print_register(
			compiler,
			this_register_instruction, 
			buffer, 
			buffer_size, 
			buffer_bytes_written_ret
		)
	);

	_STACK_PTX_CHECK_RET(
		_stack_ptx_snprintf_append(
			buffer,
			buffer_size,
			buffer_bytes_written_ret,
			", "
		)
	);

	_STACK_PTX_CHECK_RET(
		_stack_ptx_print_special(
			compiler,
			instruction, 
			buffer, 
			buffer_size, 
			buffer_bytes_written_ret)
	);

	_STACK_PTX_CHECK_RET(
		_stack_ptx_snprintf_append(
			buffer,
			buffer_size,
			buffer_bytes_written_ret,
			";\n"
		)
	);

	compiler->ast[ast_idx] = this_register_instruction;

	return STACK_PTX_SUCCESS;
}


static
inline
StackPtxResult
_stack_ptx_compile(
	StackPtxCompiler* compiler,
    const size_t* requests,
    size_t num_requests,
	char* buffer,
	size_t buffer_size,
	size_t* buffer_bytes_written_ret
) {
	memcpy(
		compiler->request_stack_ptrs,
		compiler->stack_ptrs,
		compiler->stack_info.num_stacks * sizeof(StackPtxStackPtr)
	);

	for (size_t i = 0; i < num_requests; i++) {
		StackPtxStackIdx stack_idx = compiler->registers[requests[i]].stack_idx;
		_STACK_PTX_CHECK_RET( _stack_ptx_check_stack_type_range(compiler, stack_idx) );

		if (compiler->request_stack_ptrs[stack_idx] > 0) {
			StackPtxStackPtr stack_ptr = --compiler->request_stack_ptrs[stack_idx];
			StackPtxAstIdx ast_idx = compiler->stacks[stack_idx * compiler->compiler_info.stack_size + stack_ptr];
			if (compiler->ast_to_visit_stack_ptr >= compiler->compiler_info.max_ast_to_visit_stack_depth) {
				_STACK_PTX_ERROR( STACK_PTX_ERROR_INSUFFICIENT_AST_VISIT_SIZE );
			}
			compiler->ast_to_visit_stack[compiler->ast_to_visit_stack_ptr++] = ast_idx;
			compiler->ast_to_visit_stack_max_depth_usage = 
				STACK_PTX_MAX(
					compiler->ast_to_visit_stack_max_depth_usage, 
					compiler->ast_to_visit_stack_ptr
				);
		}
	}

	while (compiler->ast_to_visit_stack_ptr != 0) {
		StackPtxAstIdx ast_idx = compiler->ast_to_visit_stack[--compiler->ast_to_visit_stack_ptr];
		StackPtxInstruction instruction = compiler->ast[ast_idx];
		switch(instruction.instruction_type) {
			case STACK_PTX_INSTRUCTION_TYPE_CONSTANT:
			case STACK_PTX_INSTRUCTION_TYPE_REGISTER:
			case STACK_PTX_INSTRUCTION_TYPE_INPUT:
				continue;
			case STACK_PTX_INSTRUCTION_TYPE_SPECIAL: {
				_STACK_PTX_CHECK_RET(
					_stack_ptx_compile_special(
						compiler,
						instruction,
						ast_idx,
						buffer,
						buffer_size,
						buffer_bytes_written_ret
					)
				);
			} break;
			case STACK_PTX_INSTRUCTION_TYPE_PTX: {
				_STACK_PTX_CHECK_RET(
					_stack_ptx_compile_ptx(
						compiler,
						instruction,
						ast_idx,
						buffer,
						buffer_size,
						buffer_bytes_written_ret
					)
				);
			} break;
			default:
				_STACK_PTX_ERROR( STACK_PTX_ERROR_INVALID_VALUE );
		}
	}

	for (size_t i = 0; i < num_requests; i++) {
		const char* request_name = compiler->registers[requests[i]].name;
		StackPtxStackIdx stack_idx = compiler->registers[requests[i]].stack_idx;
		_STACK_PTX_CHECK_RET( _stack_ptx_check_stack_type_range(compiler, stack_idx) );

		if (compiler->stack_ptrs[stack_idx] > 0) {
			StackPtxStackPtr stack_ptr = --compiler->stack_ptrs[stack_idx];
			StackPtxAstIdx ast_idx = compiler->stacks[stack_idx * compiler->compiler_info.stack_size + stack_ptr];
			StackPtxInstruction instruction = compiler->ast[ast_idx];
			_STACK_PTX_CHECK_RET( _stack_ptx_check_stack_type_range(compiler, stack_idx) );
			
			const char* literal_prefix = compiler->stack_info.stack_literal_prefixes[stack_idx];
			_STACK_PTX_CHECK_RET(
				_stack_ptx_snprintf_append(
					buffer,
					buffer_size,
					buffer_bytes_written_ret,
					STACK_PTX_TABBING "mov.%s %%%s, ",
					literal_prefix,
					request_name
				)
			);
			switch(instruction.instruction_type) {
				case STACK_PTX_INSTRUCTION_TYPE_INPUT: {
					_STACK_PTX_CHECK_RET(
						_stack_ptx_print_input(compiler, instruction, buffer, buffer_size, buffer_bytes_written_ret)
					);
				} break;
				case STACK_PTX_INSTRUCTION_TYPE_REGISTER: {
					_STACK_PTX_CHECK_RET(
						_stack_ptx_print_register(compiler, instruction, buffer, buffer_size, buffer_bytes_written_ret)
					);
				} break;
				case STACK_PTX_INSTRUCTION_TYPE_CONSTANT: {
					_STACK_PTX_CHECK_RET(
						_stack_ptx_print_constant(compiler, instruction, buffer, buffer_size, buffer_bytes_written_ret)
					);
				} break;
				default:
					_STACK_PTX_ERROR( STACK_PTX_ERROR_INVALID_VALUE );
			}
			_STACK_PTX_CHECK_RET(
				_stack_ptx_snprintf_append(
					buffer,
					buffer_size,
					buffer_bytes_written_ret,
					";\n"
				)
			);
		}
	}
	return STACK_PTX_SUCCESS;
}

static
inline
StackPtxResult
_stack_ptx_write_register_declaration(
	const StackPtxCompiler* compiler,
	char* buffer,
	size_t buffer_size,
	size_t* buffer_bytes_written_ret
) {

	_STACK_PTX_CHECK_RET(
		_stack_ptx_snprintf_append(buffer, buffer_size, buffer_bytes_written_ret,
			STACK_PTX_TABBING "{\n"
		)
	);

	for (StackPtxStackIdx stack_idx = 0; stack_idx < compiler->stack_info.num_stacks; stack_idx++) {
		StackPtxRegisterCounter reg = compiler->register_counters[stack_idx];
		if (reg == 0) continue;
		const char* literal_prefix = compiler->stack_info.stack_literal_prefixes[stack_idx];
		_STACK_PTX_CHECK_RET(
			_stack_ptx_snprintf_append(buffer, buffer_size, buffer_bytes_written_ret,
				STACK_PTX_TABBING ".reg .%s %%_s%u_<%u>;\n",
				literal_prefix,
				(unsigned)stack_idx,
				(unsigned)reg
			)
		);
	}

	return STACK_PTX_SUCCESS;
}

static
inline
StackPtxResult
_stack_ptx_setup_compiler_workspace(
	const StackPtxCompilerInfo* compiler_info_ref,
	const StackPtxStackInfo* stack_info_ref,
	void* workspace,
	size_t workspace_in_bytes,
	size_t* workspace_in_bytes_out,
	StackPtxCompiler** compiler_out
) {
	const size_t max_ast_size = compiler_info_ref->max_ast_size;
	const size_t max_ast_to_visit_stack_depth = compiler_info_ref->max_ast_to_visit_stack_depth;
	const size_t stack_size = compiler_info_ref->stack_size;
	const size_t max_frame_depth = compiler_info_ref->max_frame_depth;
	const size_t num_stacks = stack_info_ref->num_stacks;
	const size_t store_size = compiler_info_ref->store_size;

	const size_t compiler_num_bytes = 			sizeof(StackPtxCompiler);
	const size_t ast_num_bytes = 				max_ast_size * sizeof(StackPtxInstruction);
	const size_t ast_to_visit_num_bytes = 		max_ast_to_visit_stack_depth * sizeof(StackPtxAstIdx);
	const size_t stacks_num_bytes = 			num_stacks * stack_size * sizeof(StackPtxAstIdx);
	const size_t stack_ptrs_num_bytes = 		num_stacks * sizeof(StackPtxStackPtr);
	const size_t request_stack_ptrs_num_bytes = num_stacks * sizeof(StackPtxStackPtr);
	const size_t register_counters_num_bytes = 	num_stacks * sizeof(StackPtxRegisterCounter);
	const size_t meta_stack_num_bytes = 		stack_size * sizeof(StackPtxMetaConstant);
	const size_t stack_frames_num_bytes = 		max_frame_depth * sizeof(StackPtxStackFrame);
	const size_t store_num_bytes = 				store_size * sizeof(StackPtxInstruction);

	const size_t compiler_offset = 				0;
	const size_t ast_offset =					compiler_offset + 					_STACK_PTX_ALIGNMENT_UP(compiler_num_bytes, _STACK_PTX_ALIGNMENT);
	const size_t ast_to_visit_offset =			ast_offset + 						_STACK_PTX_ALIGNMENT_UP(ast_num_bytes, _STACK_PTX_ALIGNMENT);
	const size_t stacks_offset =				ast_to_visit_offset + 				_STACK_PTX_ALIGNMENT_UP(ast_to_visit_num_bytes, _STACK_PTX_ALIGNMENT);
	const size_t stack_ptrs_offset =			stacks_offset + 					_STACK_PTX_ALIGNMENT_UP(stacks_num_bytes, _STACK_PTX_ALIGNMENT);
	const size_t request_stack_ptrs_offset =    stack_ptrs_offset + 				_STACK_PTX_ALIGNMENT_UP(stack_ptrs_num_bytes, _STACK_PTX_ALIGNMENT);
	const size_t register_counters_offset = 	request_stack_ptrs_offset + 		_STACK_PTX_ALIGNMENT_UP(request_stack_ptrs_num_bytes, _STACK_PTX_ALIGNMENT);
	const size_t meta_stack_offset =			register_counters_offset + 			_STACK_PTX_ALIGNMENT_UP(register_counters_num_bytes, _STACK_PTX_ALIGNMENT);
	const size_t stack_frames_offset =			meta_stack_offset + 				_STACK_PTX_ALIGNMENT_UP(meta_stack_num_bytes, _STACK_PTX_ALIGNMENT);
	const size_t store_offset =					stack_frames_offset +				_STACK_PTX_ALIGNMENT_UP(stack_frames_num_bytes, _STACK_PTX_ALIGNMENT);
	const size_t total_size = 					store_offset + store_num_bytes + 	_STACK_PTX_ALIGNMENT;

	if (workspace_in_bytes_out != NULL) {
		*workspace_in_bytes_out = total_size;
	}

	if (compiler_out == NULL) {
		return STACK_PTX_SUCCESS;
	}

	*compiler_out = NULL;

	if (workspace == NULL) {
		_STACK_PTX_ERROR( STACK_PTX_ERROR_INVALID_VALUE );
	}

	if (total_size > workspace_in_bytes) {
		_STACK_PTX_ERROR( STACK_PTX_ERROR_INSUFFICIENT_WORKSPACE );
	}

	const uintptr_t address = (uintptr_t)workspace;
	const uintptr_t aligned_address = _STACK_PTX_ALIGNMENT_UP(address, _STACK_PTX_ALIGNMENT);
	void* const aligned_workspace = (void*)aligned_address;

	StackPtxCompiler* const compiler = (StackPtxCompiler*)((char*)aligned_workspace + compiler_offset);

	compiler->compiler_info = 						*compiler_info_ref;
	compiler->stack_info = 							*stack_info_ref;
	compiler->ast = 								(StackPtxInstruction*)((char*)aligned_workspace + ast_offset);
	compiler->ast_size = 							0;
	compiler->ast_to_visit_stack = 					(StackPtxAstIdx*)((char*)aligned_workspace + ast_to_visit_offset);
	compiler->ast_to_visit_stack_max_depth_usage = 	0;
	compiler->ast_to_visit_stack_ptr = 				0;
	compiler->stacks = 								(StackPtxAstIdx*)((char*)aligned_workspace + stacks_offset);
	compiler->stack_ptrs = 							(StackPtxStackPtr*)((char*)aligned_workspace + stack_ptrs_offset);
	compiler->request_stack_ptrs = 					(StackPtxStackPtr*)((char*)aligned_workspace + request_stack_ptrs_offset);
	compiler->register_counters = 					(StackPtxRegisterCounter*)((char*)aligned_workspace + register_counters_offset);
	compiler->meta_stack = 							(StackPtxMetaConstant*)((char*)aligned_workspace + meta_stack_offset);
	compiler->meta_stack_ptr = 						0;
	compiler->stack_frames = 						(StackPtxStackFrame*)((char*)aligned_workspace + stack_frames_offset);
	compiler->frame_ptr = 							0;
	compiler->store = 								(StackPtxInstruction*)((char*)aligned_workspace + store_offset);

	memset(compiler->stacks, 0, stacks_num_bytes);
	memset(compiler->stack_ptrs, 0, stack_ptrs_num_bytes);
	memset(compiler->request_stack_ptrs, 0, request_stack_ptrs_num_bytes);
	memset(compiler->register_counters, 0, register_counters_num_bytes);
	memset(compiler->stack_frames, 0, stack_frames_num_bytes);
	memset(compiler->store, 0, store_num_bytes);

	*compiler_out = compiler;

	return STACK_PTX_SUCCESS;
}

STACK_PTX_PUBLIC_DEF
StackPtxResult
stack_ptx_compile_workspace_size(
	const StackPtxCompilerInfo* compiler_info_ref,
	const StackPtxStackInfo* stack_info_ref,
	size_t* workspace_in_bytes_out
) {
	_STACK_PTX_CHECK_RET( _stack_ptx_setup_compiler_workspace(
		compiler_info_ref,
		stack_info_ref,
		NULL,
		0,
		workspace_in_bytes_out,
		NULL
	) );

	return STACK_PTX_SUCCESS;
}

STACK_PTX_PUBLIC_DEF
StackPtxResult
stack_ptx_compile(
	const StackPtxCompilerInfo* compiler_info_ref,
	const StackPtxStackInfo* stack_info_ref,
	const StackPtxInstruction* instructions,
	const StackPtxRegister* registers,
	size_t num_registers,
	const StackPtxInstruction** routines,
	size_t num_routines,
	const size_t* requests,
	size_t num_requests,
	size_t execution_limit,
	void* workspace,
	size_t workspace_in_bytes,
	char* buffer,
	size_t buffer_size,
	size_t* buffer_bytes_written_ret
) {
	if (stack_info_ref->num_stacks > STACK_PTX_MAX_NUM_STACKS) {
		_STACK_PTX_ERROR( STACK_PTX_ERROR_INVALID_VALUE );
	}

	StackPtxCompiler* compiler = NULL;
	_STACK_PTX_CHECK_RET( _stack_ptx_setup_compiler_workspace(
		compiler_info_ref,
		stack_info_ref,
		workspace,
		workspace_in_bytes,
		NULL,
		&compiler
	) );

	compiler->registers = registers;
	compiler->num_registers = num_registers;
	compiler->routines = routines;
	compiler->num_routines = num_routines;

	*buffer_bytes_written_ret = 0;

	if (compiler->compiler_info.max_frame_depth == 0) {
		if (buffer != NULL) {
			if (buffer_size == 0) {
				_STACK_PTX_ERROR( STACK_PTX_ERROR_INSUFFICIENT_BUFFER );
			}
			buffer[0] = '\0';
		}
		return STACK_PTX_SUCCESS;
	}

	// Set the stack frame to the immediate instructions passed in.
	compiler->stack_frames[compiler->frame_ptr].instructions = instructions;

	_STACK_PTX_CHECK_RET( _stack_ptx_ast_run(compiler, execution_limit) );

	_STACK_PTX_CHECK_RET(
		_stack_ptx_compile(
			compiler,
			requests,
			num_requests,
			buffer,
			buffer_size,
			buffer_bytes_written_ret
		)
	);

	_STACK_PTX_CHECK_RET(
		_stack_ptx_snprintf_append(
			buffer,
			buffer_size,
			buffer_bytes_written_ret,
			STACK_PTX_TABBING "}"
		)
	);

	size_t register_declaration_bytes_written = 0;

	_STACK_PTX_CHECK_RET(
		_stack_ptx_write_register_declaration(
			compiler,
			NULL,
			0ull,
			&register_declaration_bytes_written
		)
	);

	if (buffer == NULL) {
		*buffer_bytes_written_ret += register_declaration_bytes_written;
		return STACK_PTX_SUCCESS;
	}

	// Guard the exact-fit cases that would otherwise leave no room to prefix the
	// register declarations or to write the final string terminator after memmove.
	if (*buffer_bytes_written_ret >= buffer_size) {
		_STACK_PTX_ERROR( STACK_PTX_ERROR_INSUFFICIENT_BUFFER );
	}

	if (register_declaration_bytes_written >= buffer_size ||
		*buffer_bytes_written_ret >= buffer_size - register_declaration_bytes_written) {
		_STACK_PTX_ERROR( STACK_PTX_ERROR_INSUFFICIENT_BUFFER );
	}

	memmove(buffer + register_declaration_bytes_written, buffer, *buffer_bytes_written_ret);

	size_t register_declaration_bytes_written_temp = 0;

	_STACK_PTX_CHECK_RET(
		_stack_ptx_write_register_declaration(
			compiler,
			buffer,
			register_declaration_bytes_written,
			&register_declaration_bytes_written_temp
		)
	);

	// replace snprintf null termination with new line for registers
	buffer[register_declaration_bytes_written_temp-1] = '\n';
	*buffer_bytes_written_ret += register_declaration_bytes_written;
	// add in null termination after memmove
	buffer[*buffer_bytes_written_ret] = '\0';

	return STACK_PTX_SUCCESS;
}

#endif // STACK_PTX_IMPLEMENTATION_ONCE
#endif // STACK_PTX_IMPLEMENTATION
