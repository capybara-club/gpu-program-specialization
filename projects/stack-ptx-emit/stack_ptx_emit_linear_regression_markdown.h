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

#ifndef STACK_PTX_EMIT_LINEAR_REGRESSION_MARKDOWN_H_INCLUDE
#define STACK_PTX_EMIT_LINEAR_REGRESSION_MARKDOWN_H_INCLUDE

#ifdef STACK_PTX_EMIT_LINEAR_REGRESSION_MARKDOWN_IMPLEMENTATION
#define STACK_PTX_EMIT_LINEAR_REGRESSION_BASE_IMPLEMENTATION
#define STACK_PTX_EMIT_LINEAR_REGRESSION_LATEX_IMPLEMENTATION
#endif

#include <stack_ptx_emit_linear_regression_latex.h>

#ifdef __cplusplus
extern "C" {
#endif

STACK_PTX_EMIT_PUBLIC_DEC
StackPtxEmitLinearRegressionResult
stack_ptx_emit_linear_regression_markdown_compile(
	const StackPtxCompilerInfo* compiler_info_ref,
	const StackPtxStackInfo* stack_info_ref,
	const StackPtxEmitMarkdownSettings* settings_ref,
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

#endif // STACK_PTX_EMIT_LINEAR_REGRESSION_MARKDOWN_H_INCLUDE

#ifdef STACK_PTX_EMIT_LINEAR_REGRESSION_MARKDOWN_IMPLEMENTATION
#ifndef STACK_PTX_EMIT_LINEAR_REGRESSION_MARKDOWN_IMPLEMENTATION_ONCE
#define STACK_PTX_EMIT_LINEAR_REGRESSION_MARKDOWN_IMPLEMENTATION_ONCE

static
inline
StackPtxEmitResult
_stack_ptx_emit_linear_regression_markdown_feature_ast(
	const StackPtxCompilerInfo* compiler_info_ref,
	const StackPtxStackInfo* stack_info_ref,
	const StackPtxEmitLinearRegressionInfo* regression_info_ref,
	size_t feature_idx,
	StackPtxEmitBumpAllocator* allocator,
	void* workspace,
	size_t workspace_in_bytes,
	StackPtxEmitAstView* ast_view_out,
	StackPtxAstIdx* root_ast_idx_out,
	StackPtxResult* stack_ptx_result_out
) {
	if (compiler_info_ref == NULL ||
		stack_info_ref == NULL ||
		regression_info_ref == NULL ||
		allocator == NULL ||
		ast_view_out == NULL ||
		root_ast_idx_out == NULL) {
		return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
	}

	StackPtxRegister* registers = NULL;
	size_t num_registers = 0;
	StackPtxEmitResult emit_result = _stack_ptx_emit_linear_regression_alloc_feature_registers(
		regression_info_ref,
		NULL,
		allocator,
		&registers,
		&num_registers
	);
	if (emit_result != STACK_PTX_EMIT_SUCCESS) {
		return emit_result;
	}

	size_t requests[1] = { regression_info_ref->input_dim };
	emit_result = _stack_ptx_emit_prepare_ast(
		compiler_info_ref,
		stack_info_ref,
		regression_info_ref->feature_programs[feature_idx].instructions,
		registers,
		num_registers,
		regression_info_ref->routines,
		regression_info_ref->num_routines,
		requests,
		1,
		regression_info_ref->execution_limit,
		workspace,
		workspace_in_bytes,
		ast_view_out,
		stack_ptx_result_out
	);
	if (emit_result != STACK_PTX_EMIT_SUCCESS) {
		return emit_result;
	}

	memcpy(
		ast_view_out->compiler->request_stack_ptrs,
		ast_view_out->compiler->stack_ptrs,
		ast_view_out->compiler->stack_info.num_stacks * sizeof(StackPtxStackPtr)
	);

	StackPtxStackIdx stack_idx = ast_view_out->compiler->registers[regression_info_ref->input_dim].stack_idx;
	StackPtxResult stack_result = _stack_ptx_check_stack_type_range(ast_view_out->compiler, stack_idx);
	if (stack_ptx_result_out != NULL) {
		*stack_ptx_result_out = stack_result;
	}
	if (stack_result != STACK_PTX_SUCCESS) {
		return STACK_PTX_EMIT_ERROR_STACK_PTX;
	}
	if (ast_view_out->compiler->request_stack_ptrs[stack_idx] == 0) {
		return STACK_PTX_EMIT_ERROR_STACK_PTX;
	}

	StackPtxStackPtr stack_ptr = --ast_view_out->compiler->request_stack_ptrs[stack_idx];
	*root_ast_idx_out = ast_view_out->compiler->stacks[
		stack_idx * ast_view_out->compiler->compiler_info.stack_size + stack_ptr
	];
	return STACK_PTX_EMIT_SUCCESS;
}

static
inline
StackPtxEmitResult
_stack_ptx_emit_linear_regression_markdown_measure_ast(
	const StackPtxEmitLatexCompiler* latex_compiler,
	StackPtxAstIdx ast_idx,
	size_t* bytes_out,
	StackPtxResult* stack_ptx_result_out
) {
	if (latex_compiler == NULL || bytes_out == NULL) {
		return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
	}

	StackPtxEmitWriter writer = {
		.buffer = NULL,
		.buffer_size = 0,
		.bytes_written = 0
	};
	StackPtxEmitResult emit_result = _stack_ptx_emit_latex_render_ast(
		latex_compiler,
		ast_idx,
		STACK_PTX_EMIT_LATEX_PRECEDENCE_LOWEST,
		false,
		&writer,
		stack_ptx_result_out
	);
	if (emit_result != STACK_PTX_EMIT_SUCCESS) {
		return emit_result;
	}
	*bytes_out = writer.bytes_written;
	return STACK_PTX_EMIT_SUCCESS;
}

static
inline
StackPtxEmitResult
_stack_ptx_emit_linear_regression_markdown_ptx_children(
	const StackPtxEmitLatexCompiler* latex_compiler,
	StackPtxAstIdx ast_idx,
	StackPtxAstIdx* child_ast_indices,
	size_t* num_children_out,
	StackPtxResult* stack_ptx_result_out
) {
	if (latex_compiler == NULL ||
		child_ast_indices == NULL ||
		num_children_out == NULL) {
		return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
	}
	if (ast_idx >= latex_compiler->ast_view.compiler->ast_size) {
		if (stack_ptx_result_out != NULL) {
			*stack_ptx_result_out = STACK_PTX_ERROR_INVALID_VALUE;
		}
		return STACK_PTX_EMIT_ERROR_STACK_PTX;
	}

	StackPtxInstruction instruction = latex_compiler->ast_view.compiler->ast[ast_idx];
	if (instruction.instruction_type != STACK_PTX_INSTRUCTION_TYPE_PTX) {
		*num_children_out = 0;
		return STACK_PTX_EMIT_SUCCESS;
	}

	StackPtxIdx instruction_idx = 0;
	const StackPtxPtxInstructionDescriptor* descriptor = NULL;
	StackPtxResult stack_result = _stack_ptx_get_ptx_instruction_descriptor(
		latex_compiler->ast_view.compiler,
		instruction,
		&instruction_idx,
		&descriptor
	);
	if (stack_ptx_result_out != NULL) {
		*stack_ptx_result_out = stack_result;
	}
	if (stack_result != STACK_PTX_SUCCESS) {
		return STACK_PTX_EMIT_ERROR_STACK_PTX;
	}
	(void)instruction_idx;

	size_t num_args_flat = 0;
	size_t num_args = 0;
	size_t num_rets_flat = 0;
	size_t num_rets = 0;
	stack_result = _stack_ptx_ptx_instruction_num_args(
		latex_compiler->ast_view.compiler,
		instruction,
		&num_args_flat,
		&num_args
	);
	if (stack_ptx_result_out != NULL) {
		*stack_ptx_result_out = stack_result;
	}
	if (stack_result != STACK_PTX_SUCCESS) {
		return STACK_PTX_EMIT_ERROR_STACK_PTX;
	}

	stack_result = _stack_ptx_ptx_instruction_num_rets(
		latex_compiler->ast_view.compiler,
		instruction,
		&num_rets_flat,
		&num_rets
	);
	if (stack_ptx_result_out != NULL) {
		*stack_ptx_result_out = stack_result;
	}
	if (stack_result != STACK_PTX_SUCCESS) {
		return STACK_PTX_EMIT_ERROR_STACK_PTX;
	}
	if (num_rets != 1 || num_rets_flat != 1) {
		return STACK_PTX_EMIT_ERROR_UNSUPPORTED_MULTI_RETURN;
	}

	size_t args_start_idx = ast_idx + _stack_ptx_instruction_ret_idx(instruction) - num_rets_flat;
	size_t arg_ast_idx = 0;
	for (size_t i = 0; i < num_args; i++) {
		StackPtxArgIdx arg_type_idx = descriptor->arg_types[i];
		if (arg_type_idx >= latex_compiler->ast_view.compiler->stack_info.num_arg_types) {
			return STACK_PTX_EMIT_ERROR_STACK_PTX;
		}
		if (latex_compiler->ast_view.compiler->stack_info.arg_type_info[arg_type_idx].num_vec_elems != 0 ||
			_stack_ptx_arg_type_num_stack_elems(latex_compiler->ast_view.compiler, arg_type_idx) != 1) {
			return STACK_PTX_EMIT_ERROR_UNSUPPORTED_VECTOR;
		}
		StackPtxInstruction arg_instruction =
			latex_compiler->ast_view.compiler->ast[args_start_idx - num_args_flat + 1 + arg_ast_idx];
		if (arg_instruction.instruction_type != STACK_PTX_INSTRUCTION_TYPE_AST_IDX) {
			return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
		}
		child_ast_indices[i] = arg_instruction.payload.ast_idx;
		arg_ast_idx++;
	}

	*num_children_out = num_args;
	return STACK_PTX_EMIT_SUCCESS;
}

static
inline
StackPtxEmitResult
_stack_ptx_emit_linear_regression_markdown_collect_factors(
	const StackPtxEmitLatexCompiler* latex_compiler,
	StackPtxAstIdx ast_idx,
	bool is_root,
	size_t factor_expression_min_bytes,
	uint8_t* visit_state,
	size_t* factor_number_by_ast,
	StackPtxAstIdx* factor_order,
	size_t* factor_count,
	StackPtxResult* stack_ptx_result_out
) {
	if (latex_compiler == NULL ||
		visit_state == NULL ||
		factor_number_by_ast == NULL ||
		factor_order == NULL ||
		factor_count == NULL) {
		return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
	}

	if (visit_state[ast_idx] == 2) {
		return STACK_PTX_EMIT_SUCCESS;
	}
	if (visit_state[ast_idx] == 1) {
		return STACK_PTX_EMIT_ERROR_STACK_PTX;
	}
	visit_state[ast_idx] = 1;

	StackPtxInstruction instruction = latex_compiler->ast_view.compiler->ast[ast_idx];
	if (instruction.instruction_type == STACK_PTX_INSTRUCTION_TYPE_PTX) {
		StackPtxAstIdx child_ast_indices[STACK_PTX_MAX_NUM_PTX_ARGS] = {};
		size_t num_children = 0;
		StackPtxEmitResult emit_result =
			_stack_ptx_emit_linear_regression_markdown_ptx_children(
				latex_compiler,
				ast_idx,
				child_ast_indices,
				&num_children,
				stack_ptx_result_out
			);
		if (emit_result != STACK_PTX_EMIT_SUCCESS) {
			return emit_result;
		}

		for (size_t i = 0; i < num_children; i++) {
			emit_result = _stack_ptx_emit_linear_regression_markdown_collect_factors(
				latex_compiler,
				child_ast_indices[i],
				false,
				factor_expression_min_bytes,
				visit_state,
				factor_number_by_ast,
				factor_order,
				factor_count,
				stack_ptx_result_out
			);
			if (emit_result != STACK_PTX_EMIT_SUCCESS) {
				return emit_result;
			}
		}

		if (!is_root && factor_expression_min_bytes != 0) {
			size_t expr_bytes = 0;
			emit_result = _stack_ptx_emit_linear_regression_markdown_measure_ast(
				latex_compiler,
				ast_idx,
				&expr_bytes,
				stack_ptx_result_out
			);
			if (emit_result != STACK_PTX_EMIT_SUCCESS) {
				return emit_result;
			}
			if (expr_bytes >= factor_expression_min_bytes) {
				*factor_count += 1;
				factor_number_by_ast[ast_idx] = *factor_count;
				factor_order[*factor_count - 1] = ast_idx;
			}
		}
	}

	visit_state[ast_idx] = 2;
	return STACK_PTX_EMIT_SUCCESS;
}

static
inline
StackPtxEmitResult
_stack_ptx_emit_linear_regression_markdown_feature_cell(
	const StackPtxCompilerInfo* compiler_info_ref,
	const StackPtxStackInfo* stack_info_ref,
	const StackPtxEmitLatexSettings* latex_settings_ref,
	size_t factor_expression_min_bytes,
	const StackPtxEmitLinearRegressionInfo* regression_info_ref,
	size_t feature_idx,
	void* workspace,
	size_t workspace_in_bytes,
	StackPtxEmitWriter* writer,
	StackPtxResult* stack_ptx_result_out
) {
	StackPtxEmitBumpAllocator allocator = {};
	StackPtxEmitResult emit_result = _stack_ptx_emit_linear_regression_init_allocator(
		compiler_info_ref,
		stack_info_ref,
		regression_info_ref,
		workspace,
		workspace_in_bytes,
		&allocator,
		stack_ptx_result_out
	);
	if (emit_result != STACK_PTX_EMIT_SUCCESS) {
		return emit_result;
	}

	StackPtxEmitAstView ast_view = {};
	StackPtxAstIdx root_ast_idx = 0;
	emit_result = _stack_ptx_emit_linear_regression_markdown_feature_ast(
		compiler_info_ref,
		stack_info_ref,
		regression_info_ref,
		feature_idx,
		&allocator,
		workspace,
		workspace_in_bytes,
		&ast_view,
		&root_ast_idx,
		stack_ptx_result_out
	);
	if (emit_result != STACK_PTX_EMIT_SUCCESS) {
		return emit_result;
	}

	StackPtxEmitLatexCompiler latex_compiler = {
		.ast_view = ast_view,
		.settings = latex_settings_ref
	};

	size_t ast_size = latex_compiler.ast_view.compiler->ast_size;
	uint8_t* visit_state = NULL;
	size_t* factor_number_by_ast = NULL;
	StackPtxAstIdx* factor_order = NULL;
	emit_result = _stack_ptx_emit_bump_alloc(
		&allocator,
		ast_size * sizeof(uint8_t),
		_Alignof(uint8_t),
		(void**)&visit_state
	);
	if (emit_result != STACK_PTX_EMIT_SUCCESS) {
		return emit_result;
	}
	emit_result = _stack_ptx_emit_bump_alloc(
		&allocator,
		ast_size * sizeof(size_t),
		_Alignof(size_t),
		(void**)&factor_number_by_ast
	);
	if (emit_result != STACK_PTX_EMIT_SUCCESS) {
		return emit_result;
	}
	emit_result = _stack_ptx_emit_bump_alloc(
		&allocator,
		ast_size * sizeof(StackPtxAstIdx),
		_Alignof(StackPtxAstIdx),
		(void**)&factor_order
	);
	if (emit_result != STACK_PTX_EMIT_SUCCESS) {
		return emit_result;
	}
	for (size_t i = 0; i < ast_size; i++) {
		visit_state[i] = 0;
		factor_number_by_ast[i] = 0;
		factor_order[i] = 0;
	}

	size_t factor_count = 0;
	emit_result = _stack_ptx_emit_linear_regression_markdown_collect_factors(
		&latex_compiler,
		root_ast_idx,
		true,
		factor_expression_min_bytes,
		visit_state,
		factor_number_by_ast,
		factor_order,
		&factor_count,
		stack_ptx_result_out
	);
	if (emit_result != STACK_PTX_EMIT_SUCCESS) {
		return emit_result;
	}

	if (factor_count == 0) {
		emit_result = _stack_ptx_emit_writer_append(writer, "$");
		if (emit_result == STACK_PTX_EMIT_SUCCESS) {
			emit_result = _stack_ptx_emit_latex_render_ast(
				&latex_compiler,
				root_ast_idx,
				STACK_PTX_EMIT_LATEX_PRECEDENCE_LOWEST,
				false,
				writer,
				stack_ptx_result_out
			);
		}
		if (emit_result == STACK_PTX_EMIT_SUCCESS) {
			emit_result = _stack_ptx_emit_writer_append(writer, "$");
		}
		return emit_result;
	}

	size_t num_special_register_exprs =
		latex_settings_ref != NULL ? latex_settings_ref->num_special_register_exprs : 0;
	const char** factor_exprs = NULL;
	emit_result = _stack_ptx_emit_bump_alloc(
		&allocator,
		(num_special_register_exprs + ast_size) * sizeof(const char*),
		_Alignof(const char*),
		(void**)&factor_exprs
	);
	if (emit_result != STACK_PTX_EMIT_SUCCESS) {
		return emit_result;
	}
	for (size_t i = 0; i < num_special_register_exprs; i++) {
		factor_exprs[i] = latex_settings_ref->special_register_exprs[i];
	}

	char* factor_name_storage = NULL;
	StackPtxInstruction* original_factor_instructions = NULL;
	StackPtxInstruction* placeholder_factor_instructions = NULL;
	emit_result = _stack_ptx_emit_bump_alloc(
		&allocator,
		ast_size * 32,
		_Alignof(char),
		(void**)&factor_name_storage
	);
	if (emit_result != STACK_PTX_EMIT_SUCCESS) {
		return emit_result;
	}
	emit_result = _stack_ptx_emit_bump_alloc(
		&allocator,
		ast_size * sizeof(StackPtxInstruction),
		_Alignof(StackPtxInstruction),
		(void**)&original_factor_instructions
	);
	if (emit_result != STACK_PTX_EMIT_SUCCESS) {
		return emit_result;
	}
	emit_result = _stack_ptx_emit_bump_alloc(
		&allocator,
		ast_size * sizeof(StackPtxInstruction),
		_Alignof(StackPtxInstruction),
		(void**)&placeholder_factor_instructions
	);
	if (emit_result != STACK_PTX_EMIT_SUCCESS) {
		return emit_result;
	}
	for (size_t i = 0; i < factor_count; i++) {
		char* factor_name = factor_name_storage + (i * 32);
		int bytes = snprintf(factor_name, 32, "u_%zu", i + 1);
		if (bytes < 0 || bytes >= 32) {
			return STACK_PTX_EMIT_ERROR_INSUFFICIENT_BUFFER;
		}
		factor_exprs[num_special_register_exprs + i] = factor_name;
		original_factor_instructions[i] = latex_compiler.ast_view.compiler->ast[factor_order[i]];
		placeholder_factor_instructions[i] = (StackPtxInstruction){
			.instruction_type = STACK_PTX_INSTRUCTION_TYPE_SPECIAL,
			.aux = 0,
			.payload = { .u = (uint32_t)(num_special_register_exprs + i) }
		};
		latex_compiler.ast_view.compiler->ast[factor_order[i]] = placeholder_factor_instructions[i];
	}

	StackPtxEmitLatexSettings factor_settings = *latex_settings_ref;
	factor_settings.special_register_exprs = factor_exprs;
	factor_settings.num_special_register_exprs = num_special_register_exprs + factor_count;

	StackPtxEmitLatexCompiler factor_compiler = {
		.ast_view = latex_compiler.ast_view,
		.settings = &factor_settings
	};

	emit_result = _stack_ptx_emit_writer_append(writer, "$\\begin{aligned} &");
	if (emit_result == STACK_PTX_EMIT_SUCCESS) {
		emit_result = _stack_ptx_emit_latex_render_ast(
			&factor_compiler,
			root_ast_idx,
			STACK_PTX_EMIT_LATEX_PRECEDENCE_LOWEST,
			false,
			writer,
			stack_ptx_result_out
		);
	}
	if (emit_result == STACK_PTX_EMIT_SUCCESS) {
		emit_result = _stack_ptx_emit_writer_append(writer, " \\\\ &\\qquad \\text{where}");
	}
	if (emit_result != STACK_PTX_EMIT_SUCCESS) {
		return emit_result;
	}

	for (size_t i = 0; i < factor_count; i++) {
		StackPtxAstIdx factor_ast_idx = factor_order[i];
		latex_compiler.ast_view.compiler->ast[factor_ast_idx] = original_factor_instructions[i];

		emit_result = _stack_ptx_emit_writer_append(writer, " \\\\ &\\qquad\\qquad u_%zu = ", i + 1);
		if (emit_result == STACK_PTX_EMIT_SUCCESS) {
			emit_result = _stack_ptx_emit_latex_render_ast(
				&factor_compiler,
				factor_ast_idx,
				STACK_PTX_EMIT_LATEX_PRECEDENCE_LOWEST,
				false,
				writer,
				stack_ptx_result_out
			);
		}

		latex_compiler.ast_view.compiler->ast[factor_ast_idx] = placeholder_factor_instructions[i];
		if (emit_result != STACK_PTX_EMIT_SUCCESS) {
			return emit_result;
		}
	}

	return _stack_ptx_emit_writer_append(writer, " \\end{aligned}$");
}

STACK_PTX_EMIT_PUBLIC_DEF
StackPtxEmitLinearRegressionResult
stack_ptx_emit_linear_regression_markdown_compile(
	const StackPtxCompilerInfo* compiler_info_ref,
	const StackPtxStackInfo* stack_info_ref,
	const StackPtxEmitMarkdownSettings* settings_ref,
	const StackPtxEmitLinearRegressionInfo* regression_info_ref,
	void* workspace,
	size_t workspace_in_bytes,
	char* buffer,
	size_t buffer_size,
	size_t* buffer_bytes_written_ret,
	StackPtxEmitResult* emit_result_out,
	StackPtxResult* stack_ptx_result_out
) {
	if (emit_result_out != NULL) {
		*emit_result_out = STACK_PTX_EMIT_SUCCESS;
	}
	if (stack_ptx_result_out != NULL) {
		*stack_ptx_result_out = STACK_PTX_SUCCESS;
	}
	if (buffer_bytes_written_ret == NULL ||
		compiler_info_ref == NULL ||
		stack_info_ref == NULL ||
		regression_info_ref == NULL ||
		regression_info_ref->feature_programs == NULL ||
		regression_info_ref->means == NULL ||
		regression_info_ref->stddevs == NULL ||
		regression_info_ref->beta_standardized == NULL ||
		regression_info_ref->input_dim == 0 ||
		regression_info_ref->num_feature_programs == 0 ||
		workspace == NULL ||
		workspace_in_bytes == 0) {
		return STACK_PTX_EMIT_LINEAR_REGRESSION_ERROR_INVALID_VALUE;
	}

	for (size_t i = 0; i < regression_info_ref->num_feature_programs; i++) {
		if (regression_info_ref->feature_programs[i].instructions == NULL ||
			regression_info_ref->stddevs[i] == 0.0f) {
			return STACK_PTX_EMIT_LINEAR_REGRESSION_ERROR_INVALID_VALUE;
		}
	}

	StackPtxEmitMarkdownSettings default_settings = {
		.document_title = "Linear Regression Model Specification",
		.output_symbol = "y",
		.special_register_exprs = NULL,
		.num_special_register_exprs = 0,
		.use_input_indices = true,
		.factor_expression_min_bytes = 96
	};
	const StackPtxEmitMarkdownSettings* actual_settings =
		settings_ref != NULL ? settings_ref : &default_settings;

	StackPtxEmitLatexSettings latex_settings = {
		.document_title = NULL,
		.output_symbol = actual_settings->output_symbol,
		.special_register_exprs = actual_settings->special_register_exprs,
		.num_special_register_exprs = actual_settings->num_special_register_exprs,
		.emit_document = false,
		.use_input_indices = true
	};
	size_t factor_expression_min_bytes =
		actual_settings->factor_expression_min_bytes == 0
			? default_settings.factor_expression_min_bytes
			: actual_settings->factor_expression_min_bytes;

	StackPtxEmitWriter writer = {
		.buffer = buffer,
		.buffer_size = buffer_size,
		.bytes_written = 0
	};

	const char* title =
		(actual_settings->document_title != NULL && actual_settings->document_title[0] != '\0')
			? actual_settings->document_title
			: "Linear Regression Model Specification";
	StackPtxEmitResult emit_result = _stack_ptx_emit_writer_append(&writer, "# ");
	if (emit_result == STACK_PTX_EMIT_SUCCESS) {
		emit_result = _stack_ptx_emit_latex_append_escaped_text(&writer, title);
	}
	if (emit_result == STACK_PTX_EMIT_SUCCESS) {
		emit_result = _stack_ptx_emit_writer_append(&writer, "\n\n");
	}
	if (emit_result != STACK_PTX_EMIT_SUCCESS) {
		if (emit_result_out != NULL) {
			*emit_result_out = emit_result;
		}
		*buffer_bytes_written_ret = writer.bytes_written;
		return _stack_ptx_emit_linear_regression_map_emit_result(emit_result);
	}

	emit_result = _stack_ptx_emit_writer_append(
		&writer,
		"## Inputs\n\n"
		"For one observation, let\n\n"
		"$$\n"
		"x = (x_1, \\ldots, x_{%zu}) \\in \\mathbb{R}^{%zu}.\n"
		"$$\n",
		regression_info_ref->input_dim,
		regression_info_ref->input_dim
	);
	if (emit_result != STACK_PTX_EMIT_SUCCESS) {
		if (emit_result_out != NULL) {
			*emit_result_out = emit_result;
		}
		*buffer_bytes_written_ret = writer.bytes_written;
		return _stack_ptx_emit_linear_regression_map_emit_result(emit_result);
	}

	float intercept = _stack_ptx_emit_linear_regression_compute_intercept(regression_info_ref);
	emit_result = _stack_ptx_emit_writer_append(
		&writer,
		"\n## Model Form\n\n"
		"$$\n"
		"\\hat{y}(x) = \\alpha + \\sum_{j=1}^{%zu} B_j \\phi_j(x),\n"
		"\\qquad\n"
		"\\alpha = ",
		regression_info_ref->num_feature_programs
	);
	if (emit_result == STACK_PTX_EMIT_SUCCESS) {
		emit_result = _stack_ptx_emit_latex_append_float_literal(&writer, intercept);
	}
	if (emit_result == STACK_PTX_EMIT_SUCCESS) {
		emit_result = _stack_ptx_emit_writer_append(&writer, ".\n$$\n");
	}
	if (emit_result != STACK_PTX_EMIT_SUCCESS) {
		if (emit_result_out != NULL) {
			*emit_result_out = emit_result;
		}
		*buffer_bytes_written_ret = writer.bytes_written;
		return _stack_ptx_emit_linear_regression_map_emit_result(emit_result);
	}

	emit_result = _stack_ptx_emit_writer_append(
		&writer,
		"\n## Feature Table\n\n"
		"<div align=\"center\">\n\n"
		"<table>\n"
		"<thead>\n"
		"<tr><th>$j$</th><th>$\\beta_j$</th><th>$B_j$</th><th>$\\phi_j(x)$</th></tr>\n"
		"</thead>\n"
		"<tbody>\n"
	);
	if (emit_result != STACK_PTX_EMIT_SUCCESS) {
		if (emit_result_out != NULL) {
			*emit_result_out = emit_result;
		}
		*buffer_bytes_written_ret = writer.bytes_written;
		return _stack_ptx_emit_linear_regression_map_emit_result(emit_result);
	}

	for (size_t i = 0; i < regression_info_ref->num_feature_programs; i++) {
		float beta_unstandardized =
			regression_info_ref->beta_standardized[i] / regression_info_ref->stddevs[i];

		emit_result = _stack_ptx_emit_writer_append(
			&writer,
			"<tr><td>%zu</td><td>",
			i + 1
		);
		if (emit_result == STACK_PTX_EMIT_SUCCESS) {
			emit_result = _stack_ptx_emit_latex_append_float_literal(
				&writer,
				regression_info_ref->beta_standardized[i]
			);
		}
		if (emit_result == STACK_PTX_EMIT_SUCCESS) {
			emit_result = _stack_ptx_emit_writer_append(&writer, "</td><td>");
		}
		if (emit_result == STACK_PTX_EMIT_SUCCESS) {
			emit_result = _stack_ptx_emit_latex_append_float_literal(
				&writer,
				beta_unstandardized
			);
		}
		if (emit_result == STACK_PTX_EMIT_SUCCESS) {
			emit_result = _stack_ptx_emit_writer_append(&writer, "</td><td>");
		}
		if (emit_result == STACK_PTX_EMIT_SUCCESS) {
			emit_result = _stack_ptx_emit_linear_regression_markdown_feature_cell(
				compiler_info_ref,
				stack_info_ref,
				&latex_settings,
				factor_expression_min_bytes,
				regression_info_ref,
				i,
				workspace,
				workspace_in_bytes,
				&writer,
				stack_ptx_result_out
			);
		}
		if (emit_result == STACK_PTX_EMIT_SUCCESS) {
			emit_result = _stack_ptx_emit_writer_append(&writer, "</td></tr>\n");
		}
		if (emit_result != STACK_PTX_EMIT_SUCCESS) {
			if (emit_result_out != NULL) {
				*emit_result_out = emit_result;
			}
			*buffer_bytes_written_ret = writer.bytes_written;
			return _stack_ptx_emit_linear_regression_map_emit_result(emit_result);
		}
	}

	emit_result = _stack_ptx_emit_writer_append(&writer, "</tbody>\n</table>\n\n</div>\n");
	if (emit_result != STACK_PTX_EMIT_SUCCESS) {
		if (emit_result_out != NULL) {
			*emit_result_out = emit_result;
		}
		*buffer_bytes_written_ret = writer.bytes_written;
		return _stack_ptx_emit_linear_regression_map_emit_result(emit_result);
	}

	*buffer_bytes_written_ret = writer.bytes_written;
	if (buffer != NULL) {
		buffer[writer.bytes_written] = '\0';
	}
	if (emit_result_out != NULL) {
		*emit_result_out = STACK_PTX_EMIT_SUCCESS;
	}
	if (stack_ptx_result_out != NULL) {
		*stack_ptx_result_out = STACK_PTX_SUCCESS;
	}
	return STACK_PTX_EMIT_LINEAR_REGRESSION_SUCCESS;
}

#endif
#endif
