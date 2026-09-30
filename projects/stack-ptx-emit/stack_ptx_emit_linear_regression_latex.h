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

#ifndef STACK_PTX_EMIT_LINEAR_REGRESSION_LATEX_H_INCLUDE
#define STACK_PTX_EMIT_LINEAR_REGRESSION_LATEX_H_INCLUDE

#ifdef STACK_PTX_EMIT_LINEAR_REGRESSION_LATEX_IMPLEMENTATION
#define STACK_PTX_EMIT_LINEAR_REGRESSION_BASE_IMPLEMENTATION
#endif

#include <stack_ptx_emit_linear_regression_base.h>

#ifdef __cplusplus
extern "C" {
#endif

STACK_PTX_EMIT_PUBLIC_DEC
StackPtxEmitLinearRegressionResult
stack_ptx_emit_linear_regression_latex_compile(
	const StackPtxCompilerInfo* compiler_info_ref,
	const StackPtxStackInfo* stack_info_ref,
	const StackPtxEmitLatexSettings* settings_ref,
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

#endif // STACK_PTX_EMIT_LINEAR_REGRESSION_LATEX_H_INCLUDE

#ifdef STACK_PTX_EMIT_LINEAR_REGRESSION_LATEX_IMPLEMENTATION
#ifndef STACK_PTX_EMIT_LINEAR_REGRESSION_LATEX_IMPLEMENTATION_ONCE
#define STACK_PTX_EMIT_LINEAR_REGRESSION_LATEX_IMPLEMENTATION_ONCE

#include <math.h>
#include <stdint.h>
#include <stdio.h>

static
inline
StackPtxEmitResult
_stack_ptx_emit_linear_regression_latex_identifier(
	const StackPtxEmitLinearRegressionInfo* regression_info_ref,
	size_t input_idx,
	char* fallback_buffer,
	size_t fallback_buffer_size,
	const char** identifier_out
) {
	if (identifier_out == NULL) {
		return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
	}

	if (regression_info_ref->input_names != NULL &&
		input_idx < regression_info_ref->num_input_names &&
		regression_info_ref->input_names[input_idx] != NULL &&
		regression_info_ref->input_names[input_idx][0] != '\0') {
		*identifier_out = regression_info_ref->input_names[input_idx];
		return STACK_PTX_EMIT_SUCCESS;
	}

	int bytes = snprintf(fallback_buffer, fallback_buffer_size, "x_%zu", input_idx);
	if (bytes < 0 || (size_t)bytes >= fallback_buffer_size) {
		return STACK_PTX_EMIT_ERROR_INSUFFICIENT_BUFFER;
	}
	*identifier_out = fallback_buffer;
	return STACK_PTX_EMIT_SUCCESS;
}

static
inline
StackPtxEmitResult
_stack_ptx_emit_linear_regression_latex_feature_expr(
	const StackPtxCompilerInfo* compiler_info_ref,
	const StackPtxStackInfo* stack_info_ref,
	const StackPtxEmitLatexSettings* settings_ref,
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

	StackPtxRegister* registers = NULL;
	size_t num_registers = 0;
	emit_result = _stack_ptx_emit_linear_regression_alloc_feature_registers(
		regression_info_ref,
		NULL,
		&allocator,
		&registers,
		&num_registers
	);
	if (emit_result != STACK_PTX_EMIT_SUCCESS) {
		return emit_result;
	}

	size_t requests[1] = { regression_info_ref->input_dim };

	StackPtxEmitLatexSettings local_settings = *settings_ref;
	local_settings.emit_document = false;
	local_settings.use_input_indices = true;

	StackPtxEmitAstView ast_view = {};
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
		&ast_view,
		stack_ptx_result_out
	);
	if (emit_result != STACK_PTX_EMIT_SUCCESS) {
		return emit_result;
	}

	StackPtxEmitLatexCompiler latex_compiler = {
		.ast_view = ast_view,
		.settings = &local_settings
	};
	return _stack_ptx_emit_latex_render_request_expr(
		&latex_compiler,
		0,
		writer,
		stack_ptx_result_out
	);
}

static
inline
StackPtxEmitResult
_stack_ptx_emit_linear_regression_latex_document_prefix(
	const StackPtxEmitLatexSettings* settings_ref,
	StackPtxEmitWriter* writer
) {
	StackPtxEmitResult result = _stack_ptx_emit_writer_append(
		writer,
		"\\documentclass[11pt,landscape]{article}\n"
		"\\usepackage[margin=0.8in]{geometry}\n"
		"\\usepackage{amsmath}\n"
		"\\usepackage{amssymb}\n"
		"\\usepackage{array}\n"
		"\\usepackage{longtable}\n"
		"\\usepackage[T1]{fontenc}\n"
		"\\usepackage{lmodern}\n"
		"\\pagestyle{plain}\n"
	);
	if (result != STACK_PTX_EMIT_SUCCESS) {
		return result;
	}

	const char* title =
		(settings_ref != NULL &&
		 settings_ref->document_title != NULL &&
		 settings_ref->document_title[0] != '\0')
			? settings_ref->document_title
			: "Linear Regression Model Specification";

	result = _stack_ptx_emit_writer_append(writer, "\\title{");
	if (result != STACK_PTX_EMIT_SUCCESS) {
		return result;
	}
	result = _stack_ptx_emit_latex_append_escaped_text(writer, title);
	if (result != STACK_PTX_EMIT_SUCCESS) {
		return result;
	}
	return _stack_ptx_emit_writer_append(writer, "}\n\\date{}\n\\begin{document}\n\\maketitle\n\n");
}

STACK_PTX_EMIT_PUBLIC_DEF
StackPtxEmitLinearRegressionResult
stack_ptx_emit_linear_regression_latex_compile(
	const StackPtxCompilerInfo* compiler_info_ref,
	const StackPtxStackInfo* stack_info_ref,
	const StackPtxEmitLatexSettings* settings_ref,
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
		settings_ref == NULL ||
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

	StackPtxEmitWriter writer = {
		.buffer = buffer,
		.buffer_size = buffer_size,
		.bytes_written = 0
	};
	StackPtxEmitResult emit_result =
		_stack_ptx_emit_linear_regression_latex_document_prefix(settings_ref, &writer);
	if (emit_result != STACK_PTX_EMIT_SUCCESS) {
		if (emit_result_out != NULL) {
			*emit_result_out = emit_result;
		}
		*buffer_bytes_written_ret = writer.bytes_written;
		return _stack_ptx_emit_linear_regression_map_emit_result(emit_result);
	}

	emit_result = _stack_ptx_emit_writer_append(
		&writer,
		"\\section*{Inputs}\n\n"
		"For one observation, let\n"
		"\\[\n"
		"x = (x_1, \\ldots, x_{%zu}) \\in \\mathbb{R}^{%zu}.\n"
		"\\]\n",
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

	if (regression_info_ref->input_names != NULL && regression_info_ref->num_input_names != 0) {
		emit_result = _stack_ptx_emit_writer_append(
			&writer,
			"\n"
			"The coordinates are:\n"
			"\\[\n"
			"\\begin{aligned}\n"
		);
		if (emit_result != STACK_PTX_EMIT_SUCCESS) {
			if (emit_result_out != NULL) {
				*emit_result_out = emit_result;
			}
			*buffer_bytes_written_ret = writer.bytes_written;
			return _stack_ptx_emit_linear_regression_map_emit_result(emit_result);
		}

		for (size_t i = 0; i < regression_info_ref->input_dim; i++) {
			const char* identifier = NULL;
			char fallback_buffer[32];
			emit_result = _stack_ptx_emit_linear_regression_latex_identifier(
				regression_info_ref,
				i,
				fallback_buffer,
				sizeof(fallback_buffer),
				&identifier
			);
			if (emit_result != STACK_PTX_EMIT_SUCCESS) {
				if (emit_result_out != NULL) {
					*emit_result_out = emit_result;
				}
				*buffer_bytes_written_ret = writer.bytes_written;
				return _stack_ptx_emit_linear_regression_map_emit_result(emit_result);
			}

			emit_result = _stack_ptx_emit_writer_append(&writer, "x_{%zu} &= ", i + 1);
			if (emit_result == STACK_PTX_EMIT_SUCCESS) {
				emit_result = _stack_ptx_emit_latex_append_name(&writer, identifier);
			}
			if (emit_result == STACK_PTX_EMIT_SUCCESS) {
				emit_result = _stack_ptx_emit_writer_append(
					&writer,
					i + 1 == regression_info_ref->input_dim ? "\n" : ", \\\\\n"
				);
			}
			if (emit_result != STACK_PTX_EMIT_SUCCESS) {
				if (emit_result_out != NULL) {
					*emit_result_out = emit_result;
				}
				*buffer_bytes_written_ret = writer.bytes_written;
				return _stack_ptx_emit_linear_regression_map_emit_result(emit_result);
			}
		}

		emit_result = _stack_ptx_emit_writer_append(&writer, "\\end{aligned}\n\\]\n");
		if (emit_result != STACK_PTX_EMIT_SUCCESS) {
			if (emit_result_out != NULL) {
				*emit_result_out = emit_result;
			}
			*buffer_bytes_written_ret = writer.bytes_written;
			return _stack_ptx_emit_linear_regression_map_emit_result(emit_result);
		}
	}

	float intercept = _stack_ptx_emit_linear_regression_compute_intercept(regression_info_ref);
	emit_result = _stack_ptx_emit_writer_append(
		&writer,
		"\n"
		"\\section*{Model Form}\n\n"
		"\\[\n"
		"\\hat{y}(x) = \\alpha + \\sum_{j=1}^{%zu} B_j \\phi_j(x),\n"
		"\\qquad\n"
		"\\alpha = ",
		regression_info_ref->num_feature_programs
	);
	if (emit_result == STACK_PTX_EMIT_SUCCESS) {
		emit_result = _stack_ptx_emit_latex_append_float_literal(&writer, intercept);
	}
	if (emit_result == STACK_PTX_EMIT_SUCCESS) {
		emit_result = _stack_ptx_emit_writer_append(&writer, ".\n\\]\n\n");
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
		"\\setlength{\\LTleft}{0pt}\n"
		"\\setlength{\\LTright}{0pt}\n"
		"\\small\n"
		"\\renewcommand{\\arraystretch}{1.25}\n\n"
		"\\begin{longtable}{|\n"
		">{\\centering\\arraybackslash}p{0.035\\linewidth}|\n"
		">{\\centering\\arraybackslash}p{0.055\\linewidth}|\n"
		">{\\raggedright\\arraybackslash}p{0.830\\linewidth}|}\n"
		"\\hline\n"
		"$j$ & $B_j$ & $\\phi_j(x)$ \\\\\n"
		"\\hline\n"
		"\\endfirsthead\n"
		"\\hline\n"
		"$j$ & $B_j$ & $\\phi_j(x)$ \\\\\n"
		"\\hline\n"
		"\\endhead\n"
	);
	if (emit_result != STACK_PTX_EMIT_SUCCESS) {
		if (emit_result_out != NULL) {
			*emit_result_out = emit_result;
		}
		*buffer_bytes_written_ret = writer.bytes_written;
		return _stack_ptx_emit_linear_regression_map_emit_result(emit_result);
	}

	for (size_t i = 0; i < regression_info_ref->num_feature_programs; i++) {
		float beta_value = regression_info_ref->beta_standardized[i] / regression_info_ref->stddevs[i];
		emit_result = _stack_ptx_emit_writer_append(&writer, "%zu & $", i + 1);
		if (emit_result == STACK_PTX_EMIT_SUCCESS) {
			emit_result = _stack_ptx_emit_latex_append_float_literal(&writer, beta_value);
		}
		if (emit_result == STACK_PTX_EMIT_SUCCESS) {
			emit_result = _stack_ptx_emit_writer_append(&writer, "$ & $\\displaystyle ");
		}
		if (emit_result == STACK_PTX_EMIT_SUCCESS) {
			emit_result = _stack_ptx_emit_linear_regression_latex_feature_expr(
				compiler_info_ref,
				stack_info_ref,
				settings_ref,
				regression_info_ref,
				i,
				workspace,
				workspace_in_bytes,
				&writer,
				stack_ptx_result_out
			);
		}
		if (emit_result == STACK_PTX_EMIT_SUCCESS) {
			emit_result = _stack_ptx_emit_writer_append(&writer, "$ \\\\\n\\hline\n");
		}
		if (emit_result != STACK_PTX_EMIT_SUCCESS) {
			if (emit_result_out != NULL) {
				*emit_result_out = emit_result;
			}
			*buffer_bytes_written_ret = writer.bytes_written;
			return _stack_ptx_emit_linear_regression_map_emit_result(emit_result);
		}
	}

	emit_result = _stack_ptx_emit_writer_append(&writer, "\\end{longtable}\n\n\\end{document}\n");
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
