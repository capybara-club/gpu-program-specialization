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

#ifndef STACK_PTX_EMIT_H_INCLUDE
#define STACK_PTX_EMIT_H_INCLUDE

#ifdef STACK_PTX_EMIT_IMPLEMENTATION
#define STACK_PTX_EMIT_BASE_IMPLEMENTATION
#define STACK_PTX_EMIT_PTX_IMPLEMENTATION
#define STACK_PTX_EMIT_C_IMPLEMENTATION
#define STACK_PTX_EMIT_CUDA_IMPLEMENTATION
#define STACK_PTX_EMIT_RUST_IMPLEMENTATION
#define STACK_PTX_EMIT_NUMPY_IMPLEMENTATION
#define STACK_PTX_EMIT_LATEX_IMPLEMENTATION
#define STACK_PTX_EMIT_MARKDOWN_IMPLEMENTATION
#endif

#include <stack_ptx_emit_base.h>
#include <stack_ptx_emit_ptx.h>
#include <stack_ptx_emit_c.h>
#include <stack_ptx_emit_cuda.h>
#include <stack_ptx_emit_rust.h>
#include <stack_ptx_emit_numpy.h>
#include <stack_ptx_emit_latex.h>
#include <stack_ptx_emit_markdown.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	StackPtxEmitLanguage language;
	StackPtxEmitVectorInfo vector;
	union {
		StackPtxEmitPtxSettings ptx;
		StackPtxEmitCSettings c;
		StackPtxEmitCudaSettings cuda;
		StackPtxEmitRustSettings rust;
		StackPtxEmitNumpySettings numpy;
		StackPtxEmitLatexSettings latex;
		StackPtxEmitMarkdownSettings markdown;
	} as;
} StackPtxEmitSettings;

STACK_PTX_EMIT_PUBLIC_DEC
StackPtxEmitResult
stack_ptx_emit_compile(
	const StackPtxCompilerInfo* compiler_info_ref,
	const StackPtxStackInfo* stack_info_ref,
	const StackPtxEmitSettings* settings_ref,
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
	size_t* buffer_bytes_written_ret,
	StackPtxResult* stack_ptx_result_out
);

#ifdef __cplusplus
}
#endif

#endif // STACK_PTX_EMIT_H_INCLUDE

#ifdef STACK_PTX_EMIT_IMPLEMENTATION
#ifndef STACK_PTX_EMIT_IMPLEMENTATION_ONCE
#define STACK_PTX_EMIT_IMPLEMENTATION_ONCE

STACK_PTX_EMIT_PUBLIC_DEF
StackPtxEmitResult
stack_ptx_emit_compile(
	const StackPtxCompilerInfo* compiler_info_ref,
	const StackPtxStackInfo* stack_info_ref,
	const StackPtxEmitSettings* settings_ref,
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
	size_t* buffer_bytes_written_ret,
	StackPtxResult* stack_ptx_result_out
) {
	if (settings_ref == NULL) {
		if (stack_ptx_result_out != NULL) {
			*stack_ptx_result_out = STACK_PTX_SUCCESS;
		}
		return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
	}

	if (settings_ref->vector.kind != STACK_PTX_EMIT_VECTOR_KIND_NONE) {
		if (stack_ptx_result_out != NULL) {
			*stack_ptx_result_out = STACK_PTX_SUCCESS;
		}
		return STACK_PTX_EMIT_ERROR_UNSUPPORTED_BACKEND;
	}

	switch (settings_ref->language) {
		case STACK_PTX_EMIT_LANGUAGE_PTX:
			return stack_ptx_emit_ptx_compile(
				compiler_info_ref,
				stack_info_ref,
				&settings_ref->as.ptx,
				instructions,
				registers,
				num_registers,
				routines,
				num_routines,
				requests,
				num_requests,
				execution_limit,
				workspace,
				workspace_in_bytes,
				buffer,
				buffer_size,
				buffer_bytes_written_ret,
				stack_ptx_result_out
			);
		case STACK_PTX_EMIT_LANGUAGE_C:
			return stack_ptx_emit_c_compile(
				compiler_info_ref,
				stack_info_ref,
				&settings_ref->as.c,
				instructions,
				registers,
				num_registers,
				routines,
				num_routines,
				requests,
				num_requests,
				execution_limit,
				workspace,
				workspace_in_bytes,
				buffer,
				buffer_size,
				buffer_bytes_written_ret,
				stack_ptx_result_out
			);
		case STACK_PTX_EMIT_LANGUAGE_CUDA:
			return stack_ptx_emit_cuda_compile(
				compiler_info_ref,
				stack_info_ref,
				&settings_ref->as.cuda,
				instructions,
				registers,
				num_registers,
				routines,
				num_routines,
				requests,
				num_requests,
				execution_limit,
				workspace,
				workspace_in_bytes,
				buffer,
				buffer_size,
				buffer_bytes_written_ret,
				stack_ptx_result_out
			);
		case STACK_PTX_EMIT_LANGUAGE_RUST:
			return stack_ptx_emit_rust_compile(
				compiler_info_ref,
				stack_info_ref,
				&settings_ref->as.rust,
				instructions,
				registers,
				num_registers,
				routines,
				num_routines,
				requests,
				num_requests,
				execution_limit,
				workspace,
				workspace_in_bytes,
				buffer,
				buffer_size,
				buffer_bytes_written_ret,
				stack_ptx_result_out
			);
		case STACK_PTX_EMIT_LANGUAGE_NUMPY:
			return stack_ptx_emit_numpy_compile(
				compiler_info_ref,
				stack_info_ref,
				&settings_ref->as.numpy,
				instructions,
				registers,
				num_registers,
				routines,
				num_routines,
				requests,
				num_requests,
				execution_limit,
				workspace,
				workspace_in_bytes,
				buffer,
				buffer_size,
				buffer_bytes_written_ret,
				stack_ptx_result_out
			);
		case STACK_PTX_EMIT_LANGUAGE_LATEX:
			return stack_ptx_emit_latex_compile(
				compiler_info_ref,
				stack_info_ref,
				&settings_ref->as.latex,
				instructions,
				registers,
				num_registers,
				routines,
				num_routines,
				requests,
				num_requests,
				execution_limit,
				workspace,
				workspace_in_bytes,
				buffer,
				buffer_size,
				buffer_bytes_written_ret,
				stack_ptx_result_out
			);
		case STACK_PTX_EMIT_LANGUAGE_MARKDOWN:
			return stack_ptx_emit_markdown_compile(
				compiler_info_ref,
				stack_info_ref,
				&settings_ref->as.markdown,
				instructions,
				registers,
				num_registers,
				routines,
				num_routines,
				requests,
				num_requests,
				execution_limit,
				workspace,
				workspace_in_bytes,
				buffer,
				buffer_size,
				buffer_bytes_written_ret,
				stack_ptx_result_out
			);
		case STACK_PTX_EMIT_LANGUAGE_CPP:
		case STACK_PTX_EMIT_LANGUAGE_PYTORCH:
		case STACK_PTX_EMIT_LANGUAGE_NUM_ENUMS:
			break;
	}

	if (stack_ptx_result_out != NULL) {
		*stack_ptx_result_out = STACK_PTX_SUCCESS;
	}
	return STACK_PTX_EMIT_ERROR_UNSUPPORTED_BACKEND;
}

#endif
#endif
