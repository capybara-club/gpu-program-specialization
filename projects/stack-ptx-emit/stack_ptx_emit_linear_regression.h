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

#ifndef STACK_PTX_EMIT_LINEAR_REGRESSION_H_INCLUDE
#define STACK_PTX_EMIT_LINEAR_REGRESSION_H_INCLUDE

#ifdef STACK_PTX_EMIT_LINEAR_REGRESSION_IMPLEMENTATION
#define STACK_PTX_EMIT_LINEAR_REGRESSION_BASE_IMPLEMENTATION
#define STACK_PTX_EMIT_LINEAR_REGRESSION_C_IMPLEMENTATION
#define STACK_PTX_EMIT_LINEAR_REGRESSION_CUDA_IMPLEMENTATION
#define STACK_PTX_EMIT_LINEAR_REGRESSION_RUST_IMPLEMENTATION
#define STACK_PTX_EMIT_LINEAR_REGRESSION_NUMPY_IMPLEMENTATION
#define STACK_PTX_EMIT_LINEAR_REGRESSION_LATEX_IMPLEMENTATION
#define STACK_PTX_EMIT_LINEAR_REGRESSION_MARKDOWN_IMPLEMENTATION
#endif

#include <stack_ptx_emit_linear_regression_base.h>
#include <stack_ptx_emit_linear_regression_c.h>
#include <stack_ptx_emit_linear_regression_cuda.h>
#include <stack_ptx_emit_linear_regression_rust.h>
#include <stack_ptx_emit_linear_regression_numpy.h>
#include <stack_ptx_emit_linear_regression_latex.h>
#include <stack_ptx_emit_linear_regression_markdown.h>

#ifdef __cplusplus
extern "C" {
#endif

STACK_PTX_EMIT_PUBLIC_DEC
StackPtxEmitLinearRegressionResult
stack_ptx_emit_linear_regression_compile(
	const StackPtxCompilerInfo* compiler_info_ref,
	const StackPtxStackInfo* stack_info_ref,
	const StackPtxEmitSettings* settings_ref,
	const StackPtxEmitLinearRegressionInfo* regression_info_ref,
	void* workspace,
	size_t workspace_in_bytes,
	char* buffer,
	size_t buffer_size,
	size_t* buffer_bytes_written_ret,
	StackPtxEmitResult* emit_result_out,
	StackPtxResult* stack_ptx_result_out
);

#ifdef __cplusplus
}
#endif

#endif // STACK_PTX_EMIT_LINEAR_REGRESSION_H_INCLUDE

#ifdef STACK_PTX_EMIT_LINEAR_REGRESSION_IMPLEMENTATION
#ifndef STACK_PTX_EMIT_LINEAR_REGRESSION_IMPLEMENTATION_ONCE
#define STACK_PTX_EMIT_LINEAR_REGRESSION_IMPLEMENTATION_ONCE

STACK_PTX_EMIT_PUBLIC_DEF
StackPtxEmitLinearRegressionResult
stack_ptx_emit_linear_regression_compile(
	const StackPtxCompilerInfo* compiler_info_ref,
	const StackPtxStackInfo* stack_info_ref,
	const StackPtxEmitSettings* settings_ref,
	const StackPtxEmitLinearRegressionInfo* regression_info_ref,
	void* workspace,
	size_t workspace_in_bytes,
	char* buffer,
	size_t buffer_size,
	size_t* buffer_bytes_written_ret,
	StackPtxEmitResult* emit_result_out,
	StackPtxResult* stack_ptx_result_out
) {
	if (settings_ref == NULL) {
		return STACK_PTX_EMIT_LINEAR_REGRESSION_ERROR_INVALID_VALUE;
	}
	if (settings_ref->vector.kind != STACK_PTX_EMIT_VECTOR_KIND_NONE) {
		return STACK_PTX_EMIT_LINEAR_REGRESSION_ERROR_UNSUPPORTED_BACKEND;
	}
	switch (settings_ref->language) {
		case STACK_PTX_EMIT_LANGUAGE_C:
			return stack_ptx_emit_linear_regression_c_compile(
				compiler_info_ref,
				stack_info_ref,
				&settings_ref->as.c,
				regression_info_ref,
				workspace,
				workspace_in_bytes,
				buffer,
				buffer_size,
				buffer_bytes_written_ret,
				emit_result_out,
				stack_ptx_result_out
			);
		case STACK_PTX_EMIT_LANGUAGE_CUDA:
			return stack_ptx_emit_linear_regression_cuda_compile(
				compiler_info_ref,
				stack_info_ref,
				&settings_ref->as.cuda,
				regression_info_ref,
				workspace,
				workspace_in_bytes,
				buffer,
				buffer_size,
				buffer_bytes_written_ret,
				emit_result_out,
				stack_ptx_result_out
			);
		case STACK_PTX_EMIT_LANGUAGE_RUST:
			return stack_ptx_emit_linear_regression_rust_compile(
				compiler_info_ref,
				stack_info_ref,
				&settings_ref->as.rust,
				regression_info_ref,
				workspace,
				workspace_in_bytes,
				buffer,
				buffer_size,
				buffer_bytes_written_ret,
				emit_result_out,
				stack_ptx_result_out
			);
		case STACK_PTX_EMIT_LANGUAGE_NUMPY:
			return stack_ptx_emit_linear_regression_numpy_compile(
				compiler_info_ref,
				stack_info_ref,
				&settings_ref->as.numpy,
				regression_info_ref,
				workspace,
				workspace_in_bytes,
				buffer,
				buffer_size,
				buffer_bytes_written_ret,
				emit_result_out,
				stack_ptx_result_out
			);
		case STACK_PTX_EMIT_LANGUAGE_LATEX:
			return stack_ptx_emit_linear_regression_latex_compile(
				compiler_info_ref,
				stack_info_ref,
				&settings_ref->as.latex,
				regression_info_ref,
				workspace,
				workspace_in_bytes,
				buffer,
				buffer_size,
				buffer_bytes_written_ret,
				emit_result_out,
				stack_ptx_result_out
			);
		case STACK_PTX_EMIT_LANGUAGE_MARKDOWN:
			return stack_ptx_emit_linear_regression_markdown_compile(
				compiler_info_ref,
				stack_info_ref,
				&settings_ref->as.markdown,
				regression_info_ref,
				workspace,
				workspace_in_bytes,
				buffer,
				buffer_size,
				buffer_bytes_written_ret,
				emit_result_out,
				stack_ptx_result_out
			);
		case STACK_PTX_EMIT_LANGUAGE_PTX:
		case STACK_PTX_EMIT_LANGUAGE_CPP:
		case STACK_PTX_EMIT_LANGUAGE_PYTORCH:
		case STACK_PTX_EMIT_LANGUAGE_NUM_ENUMS:
			break;
	}
	return STACK_PTX_EMIT_LINEAR_REGRESSION_ERROR_UNSUPPORTED_BACKEND;
}

#endif
#endif
