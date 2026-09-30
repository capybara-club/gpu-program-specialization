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

#ifndef STACK_PTX_EMIT_LINEAR_REGRESSION_C_H_INCLUDE
#define STACK_PTX_EMIT_LINEAR_REGRESSION_C_H_INCLUDE

#ifdef STACK_PTX_EMIT_LINEAR_REGRESSION_C_IMPLEMENTATION
#define STACK_PTX_EMIT_LINEAR_REGRESSION_BASE_IMPLEMENTATION
#endif

#include <stack_ptx_emit_linear_regression_base.h>

#ifdef __cplusplus
extern "C" {
#endif

STACK_PTX_EMIT_PUBLIC_DEC
StackPtxEmitLinearRegressionResult
stack_ptx_emit_linear_regression_c_compile(
	const StackPtxCompilerInfo* compiler_info_ref,
	const StackPtxStackInfo* stack_info_ref,
	const StackPtxEmitCSettings* settings_ref,
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

#endif // STACK_PTX_EMIT_LINEAR_REGRESSION_C_H_INCLUDE

#ifdef STACK_PTX_EMIT_LINEAR_REGRESSION_C_IMPLEMENTATION
#ifndef STACK_PTX_EMIT_LINEAR_REGRESSION_C_IMPLEMENTATION_ONCE
#define STACK_PTX_EMIT_LINEAR_REGRESSION_C_IMPLEMENTATION_ONCE

static
inline
StackPtxEmitResult
_stack_ptx_emit_linear_regression_c_float_literal(
	float value,
	char* buffer,
	size_t buffer_size
) {
	char number[64];
	int number_bytes = snprintf(number, sizeof(number), "%.9g", (double)value);
	if (number_bytes < 0 || (size_t)number_bytes >= sizeof(number)) {
		return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
	}
	bool has_decimal_or_exponent = false;
	for (size_t i = 0; number[i] != '\0'; i++) {
		if (number[i] == '.' || number[i] == 'e' || number[i] == 'E') {
			has_decimal_or_exponent = true;
			break;
		}
	}
	int bytes = snprintf(buffer, buffer_size, has_decimal_or_exponent ? "%sf" : "%s.0f", number);
	if (bytes < 0 || (size_t)bytes >= buffer_size) {
		return STACK_PTX_EMIT_ERROR_INSUFFICIENT_BUFFER;
	}
	return STACK_PTX_EMIT_SUCCESS;
}

STACK_PTX_EMIT_PUBLIC_DEF
StackPtxEmitLinearRegressionResult
stack_ptx_emit_linear_regression_c_compile(
	const StackPtxCompilerInfo* compiler_info_ref,
	const StackPtxStackInfo* stack_info_ref,
	const StackPtxEmitCSettings* settings_ref,
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
	if (settings_ref == NULL ||
		buffer_bytes_written_ret == NULL ||
		regression_info_ref == NULL ||
		regression_info_ref->function_name == NULL ||
		regression_info_ref->function_name[0] == '\0' ||
		workspace == NULL ||
		workspace_in_bytes == 0) {
		return STACK_PTX_EMIT_LINEAR_REGRESSION_ERROR_INVALID_VALUE;
	}

	StackPtxEmitLinearRegressionResult regression_result =
		_stack_ptx_emit_linear_regression_validate_info(regression_info_ref);
	if (regression_result != STACK_PTX_EMIT_LINEAR_REGRESSION_SUCCESS) {
		return regression_result;
	}

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
		if (emit_result_out != NULL) {
			*emit_result_out = emit_result;
		}
		return _stack_ptx_emit_linear_regression_map_emit_result(emit_result);
	}

	const char** input_identifiers = NULL;
	emit_result = _stack_ptx_emit_linear_regression_alloc_input_identifiers(
		regression_info_ref,
		&allocator,
		&input_identifiers
	);
	if (emit_result != STACK_PTX_EMIT_SUCCESS) {
		if (emit_result_out != NULL) {
			*emit_result_out = emit_result;
		}
		return _stack_ptx_emit_linear_regression_map_emit_result(emit_result);
	}

	StackPtxRegister* feature_registers = NULL;
	size_t num_feature_registers = 0;
	emit_result = _stack_ptx_emit_linear_regression_alloc_feature_registers(
		regression_info_ref,
		input_identifiers,
		&allocator,
		&feature_registers,
		&num_feature_registers
	);
	if (emit_result != STACK_PTX_EMIT_SUCCESS) {
		if (emit_result_out != NULL) {
			*emit_result_out = emit_result;
		}
		return _stack_ptx_emit_linear_regression_map_emit_result(emit_result);
	}

	float intercept = _stack_ptx_emit_linear_regression_compute_intercept(regression_info_ref);
	char intercept_literal[64];
	emit_result = _stack_ptx_emit_linear_regression_c_float_literal(
		intercept,
		intercept_literal,
		sizeof(intercept_literal)
	);
	if (emit_result != STACK_PTX_EMIT_SUCCESS) {
		if (emit_result_out != NULL) {
			*emit_result_out = emit_result;
		}
		return _stack_ptx_emit_linear_regression_map_emit_result(emit_result);
	}

	StackPtxEmitWriter writer = {
		.buffer = buffer,
		.buffer_size = buffer_size,
		.bytes_written = 0
	};
	if (regression_info_ref->emit_spec_comment) {
		emit_result = _stack_ptx_emit_writer_append(
			&writer,
			"/**\n"
			" * Evaluate the hard-coded linear regression model for one observation.\n"
			" *\n"
			" * The generated function applies %zu derived feature programs to an input\n"
			" * vector of length %zu and returns the resulting scalar prediction.\n"
			" *\n"
			" * @param x Input vector with layout:\n",
			regression_info_ref->num_feature_programs,
			regression_info_ref->input_dim
		);
		for (size_t i = 0; emit_result == STACK_PTX_EMIT_SUCCESS &&
				i < regression_info_ref->input_dim; i++) {
			emit_result = _stack_ptx_emit_writer_append(
				&writer,
				" *   x[%zu] = %s\n",
				i,
				input_identifiers[i]
			);
		}
		if (emit_result == STACK_PTX_EMIT_SUCCESS) {
			emit_result = _stack_ptx_emit_writer_append(
				&writer,
				" * @return Regression prediction for the provided observation.\n"
				" *\n"
				" * Learned parameters:\n"
				" *\n"
			);
		}
		if (emit_result == STACK_PTX_EMIT_SUCCESS) {
			size_t table_bytes = 0;
			emit_result = _stack_ptx_emit_linear_regression_beta_table(
				regression_info_ref,
				" *   ",
				writer.buffer != NULL ? writer.buffer + writer.bytes_written : NULL,
				writer.buffer != NULL && writer.bytes_written < writer.buffer_size
					? writer.buffer_size - writer.bytes_written
					: 0,
				&table_bytes
			);
			writer.bytes_written += table_bytes;
		}
		if (emit_result == STACK_PTX_EMIT_SUCCESS) {
			emit_result = _stack_ptx_emit_writer_append(&writer, " */\n");
		}
		if (emit_result != STACK_PTX_EMIT_SUCCESS) {
			if (emit_result_out != NULL) {
				*emit_result_out = emit_result;
			}
			*buffer_bytes_written_ret = writer.bytes_written;
			return _stack_ptx_emit_linear_regression_map_emit_result(emit_result);
		}
	}

	const char* qualifiers =
		(settings_ref->function_qualifiers != NULL && settings_ref->function_qualifiers[0] != '\0')
			? settings_ref->function_qualifiers
			: "static inline";
	const char* f32_type = settings_ref->stack_type_names[STACK_PTX_STACK_TYPE_F32];
	emit_result = _stack_ptx_emit_writer_append(
		&writer,
		"%s %s %s(const %s *x) {\n",
		qualifiers,
		f32_type,
		regression_info_ref->function_name,
		f32_type
	);
	if (emit_result != STACK_PTX_EMIT_SUCCESS) {
		if (emit_result_out != NULL) {
			*emit_result_out = emit_result;
		}
		*buffer_bytes_written_ret = writer.bytes_written;
		return _stack_ptx_emit_linear_regression_map_emit_result(emit_result);
	}

	for (size_t i = 0; i < regression_info_ref->input_dim; i++) {
		emit_result = _stack_ptx_emit_writer_append(
			&writer,
			"\tconst %s %s = x[%zu];\n",
			f32_type,
			input_identifiers[i],
			i
		);
		if (emit_result != STACK_PTX_EMIT_SUCCESS) {
			if (emit_result_out != NULL) {
				*emit_result_out = emit_result;
			}
			*buffer_bytes_written_ret = writer.bytes_written;
			return _stack_ptx_emit_linear_regression_map_emit_result(emit_result);
		}
	}

	emit_result = _stack_ptx_emit_writer_append(&writer, "\n");
	if (emit_result != STACK_PTX_EMIT_SUCCESS) {
		if (emit_result_out != NULL) {
			*emit_result_out = emit_result;
		}
		*buffer_bytes_written_ret = writer.bytes_written;
		return _stack_ptx_emit_linear_regression_map_emit_result(emit_result);
	}

	emit_result = _stack_ptx_emit_writer_append(
		&writer,
		"\tconst %s intercept = %s;\n",
		f32_type,
		intercept_literal
	);
	if (emit_result != STACK_PTX_EMIT_SUCCESS) {
		if (emit_result_out != NULL) {
			*emit_result_out = emit_result;
		}
		*buffer_bytes_written_ret = writer.bytes_written;
		return _stack_ptx_emit_linear_regression_map_emit_result(emit_result);
	}

	for (size_t i = 0; i < regression_info_ref->num_feature_programs; i++) {
		float beta_value = _stack_ptx_emit_linear_regression_compute_beta_unstandardized(
			regression_info_ref,
			i
		);
		char beta_literal[64];
		emit_result = _stack_ptx_emit_linear_regression_c_float_literal(
			beta_value,
			beta_literal,
			sizeof(beta_literal)
		);
		if (emit_result == STACK_PTX_EMIT_SUCCESS) {
			emit_result = _stack_ptx_emit_writer_append(
				&writer,
				"\tconst %s b%zu = %s;\n",
				f32_type,
				i,
				beta_literal
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

	emit_result = _stack_ptx_emit_writer_append(
		&writer,
		"\n\t%s accum = intercept;\n",
		f32_type
	);
	if (emit_result != STACK_PTX_EMIT_SUCCESS) {
		if (emit_result_out != NULL) {
			*emit_result_out = emit_result;
		}
		*buffer_bytes_written_ret = writer.bytes_written;
		return _stack_ptx_emit_linear_regression_map_emit_result(emit_result);
	}

	size_t feature_requests[1] = { regression_info_ref->input_dim };
	for (size_t i = 0; i < regression_info_ref->num_feature_programs; i++) {
		emit_result = _stack_ptx_emit_writer_append(
			&writer,
			"\n\t{\n\t\t%s feature_out = 0.0f;\n",
			f32_type
		);
		if (emit_result != STACK_PTX_EMIT_SUCCESS) {
			if (emit_result_out != NULL) {
				*emit_result_out = emit_result;
			}
			*buffer_bytes_written_ret = writer.bytes_written;
			return _stack_ptx_emit_linear_regression_map_emit_result(emit_result);
		}

		size_t feature_bytes = 0;
		emit_result = _stack_ptx_emit_c_compile_mode(
			compiler_info_ref,
			stack_info_ref,
			settings_ref,
			regression_info_ref->feature_programs[i].instructions,
			feature_registers,
			num_feature_registers,
			regression_info_ref->routines,
			regression_info_ref->num_routines,
			feature_requests,
			1,
			regression_info_ref->execution_limit,
			workspace,
			workspace_in_bytes,
			false,
			STACK_PTX_EMIT_OUTPUT_STYLE_PLAIN,
			"\t\t",
			writer.buffer != NULL ? writer.buffer + writer.bytes_written : NULL,
			writer.buffer != NULL && writer.bytes_written < writer.buffer_size
				? writer.buffer_size - writer.bytes_written
				: 0,
			&feature_bytes,
			stack_ptx_result_out
		);
		writer.bytes_written += feature_bytes;
		if (emit_result != STACK_PTX_EMIT_SUCCESS) {
			if (emit_result_out != NULL) {
				*emit_result_out = emit_result;
			}
			*buffer_bytes_written_ret = writer.bytes_written;
			return _stack_ptx_emit_linear_regression_map_emit_result(emit_result);
		}

		emit_result = _stack_ptx_emit_writer_append(
			&writer,
			"\t\taccum += b%zu * feature_out;\n\t}\n",
			i
		);
		if (emit_result != STACK_PTX_EMIT_SUCCESS) {
			if (emit_result_out != NULL) {
				*emit_result_out = emit_result;
			}
			*buffer_bytes_written_ret = writer.bytes_written;
			return _stack_ptx_emit_linear_regression_map_emit_result(emit_result);
		}
	}

	emit_result = _stack_ptx_emit_writer_append(&writer, "\n\treturn accum;\n}\n");
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
	return STACK_PTX_EMIT_LINEAR_REGRESSION_SUCCESS;
}

#endif
#endif
