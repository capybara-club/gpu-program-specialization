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
/*
 * SPDX-FileCopyrightText: 2026 Charles Durham
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef STACK_PTX_EMIT_BASE_H_INCLUDE
#define STACK_PTX_EMIT_BASE_H_INCLUDE

#include <stack_ptx.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef STACK_PTX_EMIT_PUBLIC_DEC
#define STACK_PTX_EMIT_PUBLIC_DEC extern
#endif

#ifndef STACK_PTX_EMIT_PUBLIC_DEF
#define STACK_PTX_EMIT_PUBLIC_DEF
#endif

typedef enum {
	STACK_PTX_EMIT_SUCCESS = 0,
	STACK_PTX_EMIT_ERROR_STACK_PTX = 1,
	STACK_PTX_EMIT_ERROR_INSUFFICIENT_BUFFER = 2,
	STACK_PTX_EMIT_ERROR_INVALID_VALUE = 3,
	STACK_PTX_EMIT_ERROR_UNSUPPORTED_VECTOR = 4,
	STACK_PTX_EMIT_ERROR_UNSUPPORTED_MULTI_RETURN = 5,
	STACK_PTX_EMIT_ERROR_MISSING_STACK_TYPE = 6,
	STACK_PTX_EMIT_ERROR_MISSING_INSTRUCTION_MAPPING = 7,
	STACK_PTX_EMIT_ERROR_MISSING_SPECIAL_REGISTER_MAPPING = 8,
	STACK_PTX_EMIT_ERROR_INVALID_IDENTIFIER = 9,
	STACK_PTX_EMIT_ERROR_ALLOCATION_FAILED = 10,
	STACK_PTX_EMIT_ERROR_BACKEND_NOT_CONFIGURED = 11,
	STACK_PTX_EMIT_ERROR_UNSUPPORTED_BACKEND = 12,
	STACK_PTX_EMIT_RESULT_NUM_ELEMS = 13
} StackPtxEmitResult;

typedef enum {
	STACK_PTX_EMIT_OP_KIND_INVALID = 0,
	STACK_PTX_EMIT_OP_KIND_PASSTHROUGH,
	STACK_PTX_EMIT_OP_KIND_INFIX,
	STACK_PTX_EMIT_OP_KIND_PREFIX,
	STACK_PTX_EMIT_OP_KIND_FUNC,
	STACK_PTX_EMIT_OP_KIND_CAST,
	STACK_PTX_EMIT_OP_KIND_SELECT,
	STACK_PTX_EMIT_OP_KIND_MULADD,
	STACK_PTX_EMIT_OP_KIND_RECIPROCAL,
	STACK_PTX_EMIT_OP_KIND_RSQRT,
	STACK_PTX_EMIT_OP_KIND_NUM_ENUMS
} StackPtxEmitOpKind;

typedef struct {
	StackPtxEmitOpKind kind;
	const char* text;
} StackPtxEmitInstructionInfo;

typedef enum {
	STACK_PTX_EMIT_LANGUAGE_PTX = 0,
	STACK_PTX_EMIT_LANGUAGE_C = 1,
	STACK_PTX_EMIT_LANGUAGE_CPP = 2,
	STACK_PTX_EMIT_LANGUAGE_CUDA = 3,
	STACK_PTX_EMIT_LANGUAGE_RUST = 4,
	STACK_PTX_EMIT_LANGUAGE_NUMPY = 5,
	STACK_PTX_EMIT_LANGUAGE_PYTORCH = 6,
	STACK_PTX_EMIT_LANGUAGE_LATEX = 7,
	STACK_PTX_EMIT_LANGUAGE_MARKDOWN = 8,
	STACK_PTX_EMIT_LANGUAGE_NUM_ENUMS = 9
} StackPtxEmitLanguage;

typedef enum {
	STACK_PTX_EMIT_VECTOR_KIND_NONE = 0,
	STACK_PTX_EMIT_VECTOR_KIND_AVX = 1,
	STACK_PTX_EMIT_VECTOR_KIND_NEON = 2,
	STACK_PTX_EMIT_VECTOR_KIND_NUM_ENUMS = 3
} StackPtxEmitVectorKind;

typedef struct {
	StackPtxEmitVectorKind kind;
	uint16_t bits;
} StackPtxEmitVectorInfo;

#define STACK_PTX_EMIT_OP_INVALID        { STACK_PTX_EMIT_OP_KIND_INVALID, NULL }
#define STACK_PTX_EMIT_OP_PASSTHROUGH    { STACK_PTX_EMIT_OP_KIND_PASSTHROUGH, NULL }
#define STACK_PTX_EMIT_OP_INFIX(TEXT)    { STACK_PTX_EMIT_OP_KIND_INFIX, (TEXT) }
#define STACK_PTX_EMIT_OP_PREFIX(TEXT)   { STACK_PTX_EMIT_OP_KIND_PREFIX, (TEXT) }
#define STACK_PTX_EMIT_OP_FUNC(TEXT)     { STACK_PTX_EMIT_OP_KIND_FUNC, (TEXT) }
#define STACK_PTX_EMIT_OP_CAST(TEXT)     { STACK_PTX_EMIT_OP_KIND_CAST, (TEXT) }
#define STACK_PTX_EMIT_OP_SELECT         { STACK_PTX_EMIT_OP_KIND_SELECT, NULL }
#define STACK_PTX_EMIT_OP_MULADD         { STACK_PTX_EMIT_OP_KIND_MULADD, NULL }
#define STACK_PTX_EMIT_OP_RECIPROCAL     { STACK_PTX_EMIT_OP_KIND_RECIPROCAL, NULL }
#define STACK_PTX_EMIT_OP_RSQRT          { STACK_PTX_EMIT_OP_KIND_RSQRT, NULL }

STACK_PTX_EMIT_PUBLIC_DEC
const char*
stack_ptx_emit_result_to_string(
	StackPtxEmitResult result
);

STACK_PTX_EMIT_PUBLIC_DEC
StackPtxEmitResult
stack_ptx_emit_workspace_size(
	const StackPtxCompilerInfo* compiler_info_ref,
	const StackPtxStackInfo* stack_info_ref,
	size_t* workspace_in_bytes_out,
	StackPtxResult* stack_ptx_result_out
);

#ifdef __cplusplus
}
#endif

#endif // STACK_PTX_EMIT_BASE_H_INCLUDE

#ifdef STACK_PTX_EMIT_BASE_IMPLEMENTATION
#ifndef STACK_PTX_IMPLEMENTATION_ONCE
#error "stack_ptx_emit_base.h implementation requires stack_ptx.h implementation in the same translation unit before this include"
#endif
#ifndef STACK_PTX_EMIT_BASE_IMPLEMENTATION_ONCE
#define STACK_PTX_EMIT_BASE_IMPLEMENTATION_ONCE

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define STACK_PTX_EMIT_MAX_EXPR_BYTES 8192

typedef enum {
	STACK_PTX_EMIT_OUTPUT_STYLE_POINTER = 0,
	STACK_PTX_EMIT_OUTPUT_STYLE_RUST_MUT = 1,
	STACK_PTX_EMIT_OUTPUT_STYLE_ARRAY = 2,
	STACK_PTX_EMIT_OUTPUT_STYLE_PLAIN = 3
} StackPtxEmitOutputStyle;

typedef struct {
	char* buffer;
	size_t buffer_size;
	size_t bytes_written;
} StackPtxEmitWriter;

typedef struct {
	StackPtxCompiler* compiler;
	const size_t* requests;
	size_t num_requests;
} StackPtxEmitAstView;

typedef struct {
	char* current;
	size_t remaining;
} StackPtxEmitBumpAllocator;

static
inline
size_t
_stack_ptx_emit_align_up(
	size_t value,
	size_t alignment
) {
	if (alignment == 0) {
		return value;
	}
	size_t remainder = value % alignment;
	return remainder == 0 ? value : (value + alignment - remainder);
}

static
inline
StackPtxEmitResult
_stack_ptx_emit_bump_init(
	void* workspace,
	size_t workspace_in_bytes,
	size_t prefix_in_bytes,
	StackPtxEmitBumpAllocator* allocator_out
) {
	if (workspace == NULL || allocator_out == NULL || prefix_in_bytes > workspace_in_bytes) {
		return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
	}

	size_t aligned_prefix = _stack_ptx_emit_align_up(prefix_in_bytes, sizeof(void*));
	if (aligned_prefix > workspace_in_bytes) {
		return STACK_PTX_EMIT_ERROR_ALLOCATION_FAILED;
	}

	allocator_out->current = (char*)workspace + aligned_prefix;
	allocator_out->remaining = workspace_in_bytes - aligned_prefix;
	return STACK_PTX_EMIT_SUCCESS;
}

static
inline
StackPtxEmitResult
_stack_ptx_emit_bump_alloc(
	StackPtxEmitBumpAllocator* allocator,
	size_t num_bytes,
	size_t alignment,
	void** allocation_out
) {
	if (allocator == NULL || allocation_out == NULL || alignment == 0) {
		return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
	}

	uintptr_t current_addr = (uintptr_t)allocator->current;
	uintptr_t aligned_addr = (uintptr_t)_stack_ptx_emit_align_up((size_t)current_addr, alignment);
	size_t padding = (size_t)(aligned_addr - current_addr);
	if (padding > allocator->remaining) {
		return STACK_PTX_EMIT_ERROR_ALLOCATION_FAILED;
	}
	if (num_bytes > allocator->remaining - padding) {
		return STACK_PTX_EMIT_ERROR_ALLOCATION_FAILED;
	}

	allocator->current = (char*)aligned_addr + num_bytes;
	allocator->remaining -= padding + num_bytes;
	*allocation_out = (void*)aligned_addr;
	return STACK_PTX_EMIT_SUCCESS;
}

static
inline
StackPtxEmitResult
_stack_ptx_emit_writer_append(
	StackPtxEmitWriter* writer,
	const char* fmt,
	...
) {
	if (writer == NULL || fmt == NULL) {
		return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
	}

	va_list args;
	va_start(args, fmt);
	int bytes = vsnprintf(
		writer->buffer != NULL ? writer->buffer + writer->bytes_written : NULL,
		writer->buffer != NULL && writer->bytes_written < writer->buffer_size
			? writer->buffer_size - writer->bytes_written
			: 0,
		fmt,
		args
	);
	va_end(args);
	if (bytes < 0) {
		return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
	}

	writer->bytes_written += (size_t)bytes;
	if (writer->buffer != NULL && writer->bytes_written >= writer->buffer_size) {
		return STACK_PTX_EMIT_ERROR_INSUFFICIENT_BUFFER;
	}
	return STACK_PTX_EMIT_SUCCESS;
}

static
inline
bool
_stack_ptx_emit_is_identifier(
	const char* name
) {
	if (name == NULL || name[0] == '\0') {
		return false;
	}
	unsigned char c0 = (unsigned char)name[0];
	if (!(c0 == '_' || (c0 >= 'A' && c0 <= 'Z') || (c0 >= 'a' && c0 <= 'z'))) {
		return false;
	}
	for (size_t i = 1; name[i] != '\0'; i++) {
		unsigned char c = (unsigned char)name[i];
		if (!(c == '_' || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9'))) {
			return false;
		}
	}
	return true;
}

static
inline
bool
_stack_ptx_emit_request_contains(
	const size_t* requests,
	size_t num_requests,
	size_t register_idx
) {
	for (size_t i = 0; i < num_requests; i++) {
		if (requests[i] == register_idx) {
			return true;
		}
	}
	return false;
}

static
inline
StackPtxEmitResult
_stack_ptx_emit_prepare_ast(
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
	StackPtxEmitAstView* ast_view_out,
	StackPtxResult* stack_ptx_result_out
) {
	if (compiler_info_ref == NULL ||
		stack_info_ref == NULL ||
		instructions == NULL ||
		registers == NULL ||
		ast_view_out == NULL) {
		if (stack_ptx_result_out != NULL) {
			*stack_ptx_result_out = STACK_PTX_SUCCESS;
		}
		return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
	}

	StackPtxCompiler* compiler = NULL;
	StackPtxResult stack_result = _stack_ptx_setup_compiler_workspace(
		compiler_info_ref,
		stack_info_ref,
		workspace,
		workspace_in_bytes,
		NULL,
		&compiler
	);
	if (stack_ptx_result_out != NULL) {
		*stack_ptx_result_out = stack_result;
	}
	if (stack_result != STACK_PTX_SUCCESS) {
		return STACK_PTX_EMIT_ERROR_STACK_PTX;
	}

	compiler->registers = registers;
	compiler->num_registers = num_registers;
	compiler->routines = routines;
	compiler->num_routines = num_routines;
	compiler->stack_frames[compiler->frame_ptr].instructions = instructions;

	stack_result = _stack_ptx_ast_run(compiler, execution_limit);
	if (stack_ptx_result_out != NULL) {
		*stack_ptx_result_out = stack_result;
	}
	if (stack_result != STACK_PTX_SUCCESS) {
		return STACK_PTX_EMIT_ERROR_STACK_PTX;
	}

	ast_view_out->compiler = compiler;
	ast_view_out->requests = requests;
	ast_view_out->num_requests = num_requests;
	return STACK_PTX_EMIT_SUCCESS;
}

STACK_PTX_EMIT_PUBLIC_DEF
const char*
stack_ptx_emit_result_to_string(
	StackPtxEmitResult result
) {
	switch (result) {
		case STACK_PTX_EMIT_SUCCESS: return "STACK_PTX_EMIT_SUCCESS";
		case STACK_PTX_EMIT_ERROR_STACK_PTX: return "STACK_PTX_EMIT_ERROR_STACK_PTX";
		case STACK_PTX_EMIT_ERROR_INSUFFICIENT_BUFFER: return "STACK_PTX_EMIT_ERROR_INSUFFICIENT_BUFFER";
		case STACK_PTX_EMIT_ERROR_INVALID_VALUE: return "STACK_PTX_EMIT_ERROR_INVALID_VALUE";
		case STACK_PTX_EMIT_ERROR_UNSUPPORTED_VECTOR: return "STACK_PTX_EMIT_ERROR_UNSUPPORTED_VECTOR";
		case STACK_PTX_EMIT_ERROR_UNSUPPORTED_MULTI_RETURN: return "STACK_PTX_EMIT_ERROR_UNSUPPORTED_MULTI_RETURN";
		case STACK_PTX_EMIT_ERROR_MISSING_STACK_TYPE: return "STACK_PTX_EMIT_ERROR_MISSING_STACK_TYPE";
		case STACK_PTX_EMIT_ERROR_MISSING_INSTRUCTION_MAPPING: return "STACK_PTX_EMIT_ERROR_MISSING_INSTRUCTION_MAPPING";
		case STACK_PTX_EMIT_ERROR_MISSING_SPECIAL_REGISTER_MAPPING: return "STACK_PTX_EMIT_ERROR_MISSING_SPECIAL_REGISTER_MAPPING";
		case STACK_PTX_EMIT_ERROR_INVALID_IDENTIFIER: return "STACK_PTX_EMIT_ERROR_INVALID_IDENTIFIER";
		case STACK_PTX_EMIT_ERROR_ALLOCATION_FAILED: return "STACK_PTX_EMIT_ERROR_ALLOCATION_FAILED";
		case STACK_PTX_EMIT_ERROR_BACKEND_NOT_CONFIGURED: return "STACK_PTX_EMIT_ERROR_BACKEND_NOT_CONFIGURED";
		case STACK_PTX_EMIT_ERROR_UNSUPPORTED_BACKEND: return "STACK_PTX_EMIT_ERROR_UNSUPPORTED_BACKEND";
		case STACK_PTX_EMIT_RESULT_NUM_ELEMS: break;
	}
	return "STACK_PTX_EMIT_ERROR_INVALID_RESULT_ENUM";
}

STACK_PTX_EMIT_PUBLIC_DEF
StackPtxEmitResult
stack_ptx_emit_workspace_size(
	const StackPtxCompilerInfo* compiler_info_ref,
	const StackPtxStackInfo* stack_info_ref,
	size_t* workspace_in_bytes_out,
	StackPtxResult* stack_ptx_result_out
) {
	StackPtxResult stack_result = stack_ptx_compile_workspace_size(
		compiler_info_ref,
		stack_info_ref,
		workspace_in_bytes_out
	);
	if (stack_ptx_result_out != NULL) {
		*stack_ptx_result_out = stack_result;
	}
	return stack_result == STACK_PTX_SUCCESS ? STACK_PTX_EMIT_SUCCESS : STACK_PTX_EMIT_ERROR_STACK_PTX;
}

#endif
#endif
