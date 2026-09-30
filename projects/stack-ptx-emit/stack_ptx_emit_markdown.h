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

#ifndef STACK_PTX_EMIT_MARKDOWN_H_INCLUDE
#define STACK_PTX_EMIT_MARKDOWN_H_INCLUDE

#ifdef STACK_PTX_EMIT_MARKDOWN_IMPLEMENTATION
#define STACK_PTX_EMIT_BASE_IMPLEMENTATION
#define STACK_PTX_EMIT_LATEX_IMPLEMENTATION
#endif

#include <stack_ptx_emit_latex.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	const char* document_title;
	const char* output_symbol;
	const char* const* special_register_exprs;
	size_t num_special_register_exprs;
	bool use_input_indices;
	size_t factor_expression_min_bytes;
} StackPtxEmitMarkdownSettings;

STACK_PTX_EMIT_PUBLIC_DEC
StackPtxEmitResult
stack_ptx_emit_markdown_compile(
	const StackPtxCompilerInfo* compiler_info_ref,
	const StackPtxStackInfo* stack_info_ref,
	const StackPtxEmitMarkdownSettings* settings_ref,
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

#endif // STACK_PTX_EMIT_MARKDOWN_H_INCLUDE

#ifdef STACK_PTX_EMIT_MARKDOWN_IMPLEMENTATION
#ifndef STACK_PTX_EMIT_MARKDOWN_IMPLEMENTATION_ONCE
#define STACK_PTX_EMIT_MARKDOWN_IMPLEMENTATION_ONCE

STACK_PTX_EMIT_PUBLIC_DEF
StackPtxEmitResult
stack_ptx_emit_markdown_compile(
	const StackPtxCompilerInfo* compiler_info_ref,
	const StackPtxStackInfo* stack_info_ref,
	const StackPtxEmitMarkdownSettings* settings_ref,
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
	if (buffer_bytes_written_ret == NULL ||
		compiler_info_ref == NULL ||
		stack_info_ref == NULL ||
		instructions == NULL ||
		registers == NULL ||
		requests == NULL ||
		num_requests == 0) {
		if (stack_ptx_result_out != NULL) {
			*stack_ptx_result_out = STACK_PTX_SUCCESS;
		}
		return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
	}

	StackPtxEmitMarkdownSettings default_settings = {
		.document_title = NULL,
		.output_symbol = "y",
		.special_register_exprs = NULL,
		.num_special_register_exprs = 0,
		.use_input_indices = true,
		.factor_expression_min_bytes = 0
	};
	const StackPtxEmitMarkdownSettings* actual_settings =
		settings_ref != NULL ? settings_ref : &default_settings;

	StackPtxEmitLatexSettings latex_settings = {
		.document_title = NULL,
		.output_symbol = actual_settings->output_symbol,
		.special_register_exprs = actual_settings->special_register_exprs,
		.num_special_register_exprs = actual_settings->num_special_register_exprs,
		.emit_document = false,
		.use_input_indices = actual_settings->use_input_indices
	};

	StackPtxEmitResult result = _stack_ptx_emit_latex_validate(
		&latex_settings,
		registers,
		num_registers,
		requests,
		num_requests,
		stack_ptx_result_out
	);
	if (result != STACK_PTX_EMIT_SUCCESS) {
		*buffer_bytes_written_ret = 0;
		return result;
	}

	StackPtxEmitAstView ast_view = {};
	result = _stack_ptx_emit_prepare_ast(
		compiler_info_ref,
		stack_info_ref,
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
		&ast_view,
		stack_ptx_result_out
	);
	if (result != STACK_PTX_EMIT_SUCCESS) {
		*buffer_bytes_written_ret = 0;
		return result;
	}

	StackPtxEmitWriter writer = {
		.buffer = buffer,
		.buffer_size = buffer_size,
		.bytes_written = 0
	};
	StackPtxEmitLatexCompiler latex_compiler = {
		.ast_view = ast_view,
		.settings = &latex_settings
	};

	if (actual_settings->document_title != NULL && actual_settings->document_title[0] != '\0') {
		result = _stack_ptx_emit_writer_append(&writer, "# ");
		if (result == STACK_PTX_EMIT_SUCCESS) {
			result = _stack_ptx_emit_latex_append_escaped_text(&writer, actual_settings->document_title);
		}
		if (result == STACK_PTX_EMIT_SUCCESS) {
			result = _stack_ptx_emit_writer_append(&writer, "\n\n");
		}
		if (result != STACK_PTX_EMIT_SUCCESS) {
			*buffer_bytes_written_ret = writer.bytes_written;
			return result;
		}
	}

	memcpy(
		latex_compiler.ast_view.compiler->request_stack_ptrs,
		latex_compiler.ast_view.compiler->stack_ptrs,
		latex_compiler.ast_view.compiler->stack_info.num_stacks * sizeof(StackPtxStackPtr)
	);

	const char* output_symbol =
		(actual_settings->output_symbol != NULL && actual_settings->output_symbol[0] != '\0')
			? actual_settings->output_symbol
			: "y";

	for (size_t i = 0; i < num_requests; i++) {
		if (i != 0) {
			result = _stack_ptx_emit_writer_append(&writer, "\n");
			if (result != STACK_PTX_EMIT_SUCCESS) {
				*buffer_bytes_written_ret = writer.bytes_written;
				return result;
			}
		}

		result = _stack_ptx_emit_writer_append(&writer, "$$\n");
		if (result != STACK_PTX_EMIT_SUCCESS) {
			*buffer_bytes_written_ret = writer.bytes_written;
			return result;
		}

		if (num_requests == 1) {
			result = _stack_ptx_emit_latex_append_name(&writer, output_symbol);
		} else {
			result = _stack_ptx_emit_writer_append(&writer, "\\mathrm{");
			if (result == STACK_PTX_EMIT_SUCCESS) {
				result = _stack_ptx_emit_latex_append_escaped_text(&writer, output_symbol);
			}
			if (result == STACK_PTX_EMIT_SUCCESS) {
				result = _stack_ptx_emit_writer_append(&writer, "}_{%zu}", i + 1);
			}
		}
		if (result == STACK_PTX_EMIT_SUCCESS) {
			result = _stack_ptx_emit_writer_append(&writer, " = ");
		}
		if (result != STACK_PTX_EMIT_SUCCESS) {
			*buffer_bytes_written_ret = writer.bytes_written;
			return result;
		}

		size_t request_idx = requests[i];
		StackPtxStackIdx stack_idx = latex_compiler.ast_view.compiler->registers[request_idx].stack_idx;
		StackPtxResult stack_result = _stack_ptx_check_stack_type_range(
			latex_compiler.ast_view.compiler,
			stack_idx
		);
		if (stack_ptx_result_out != NULL) {
			*stack_ptx_result_out = stack_result;
		}
		if (stack_result != STACK_PTX_SUCCESS) {
			*buffer_bytes_written_ret = writer.bytes_written;
			return STACK_PTX_EMIT_ERROR_STACK_PTX;
		}
		if (latex_compiler.ast_view.compiler->request_stack_ptrs[stack_idx] == 0) {
			*buffer_bytes_written_ret = writer.bytes_written;
			return STACK_PTX_EMIT_ERROR_STACK_PTX;
		}
		StackPtxStackPtr stack_ptr = --latex_compiler.ast_view.compiler->request_stack_ptrs[stack_idx];
		StackPtxAstIdx ast_idx = latex_compiler.ast_view.compiler->stacks[
			stack_idx * latex_compiler.ast_view.compiler->compiler_info.stack_size + stack_ptr
		];

		result = _stack_ptx_emit_latex_render_ast(
			&latex_compiler,
			ast_idx,
			STACK_PTX_EMIT_LATEX_PRECEDENCE_LOWEST,
			false,
			&writer,
			stack_ptx_result_out
		);
		if (result == STACK_PTX_EMIT_SUCCESS) {
			result = _stack_ptx_emit_writer_append(&writer, "\n$$\n");
		}
		if (result != STACK_PTX_EMIT_SUCCESS) {
			*buffer_bytes_written_ret = writer.bytes_written;
			return result;
		}
	}

	*buffer_bytes_written_ret = writer.bytes_written;
	if (buffer != NULL) {
		buffer[writer.bytes_written] = '\0';
	}
	if (stack_ptx_result_out != NULL) {
		*stack_ptx_result_out = STACK_PTX_SUCCESS;
	}
	return STACK_PTX_EMIT_SUCCESS;
}

#endif
#endif
