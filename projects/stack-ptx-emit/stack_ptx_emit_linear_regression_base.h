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

#ifndef STACK_PTX_EMIT_LINEAR_REGRESSION_BASE_H_INCLUDE
#define STACK_PTX_EMIT_LINEAR_REGRESSION_BASE_H_INCLUDE

#ifdef STACK_PTX_EMIT_LINEAR_REGRESSION_BASE_IMPLEMENTATION
#define STACK_PTX_EMIT_IMPLEMENTATION
#endif

#include <stack_ptx_emit.h>
#include <stack_ptx_tools_descriptions.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
	STACK_PTX_EMIT_LINEAR_REGRESSION_SUCCESS = 0,
	STACK_PTX_EMIT_LINEAR_REGRESSION_ERROR_INVALID_VALUE = 1,
	STACK_PTX_EMIT_LINEAR_REGRESSION_ERROR_UNSUPPORTED_BACKEND = 2,
	STACK_PTX_EMIT_LINEAR_REGRESSION_ERROR_INSUFFICIENT_BUFFER = 3,
	STACK_PTX_EMIT_LINEAR_REGRESSION_ERROR_EMIT = 4,
	STACK_PTX_EMIT_LINEAR_REGRESSION_ERROR_STACK_PTX = 5,
	STACK_PTX_EMIT_LINEAR_REGRESSION_RESULT_NUM_ELEMS = 6
} StackPtxEmitLinearRegressionResult;

typedef struct {
	const char* name;
	const StackPtxInstruction* instructions;
} StackPtxEmitLinearRegressionFeatureProgram;

typedef struct {
	const char* function_name;

	size_t input_dim;
	const char* const* input_names;
	size_t num_input_names;

	const StackPtxEmitLinearRegressionFeatureProgram* feature_programs;
	size_t num_feature_programs;

	const float* means;
	const float* stddevs;
	const float* beta_standardized;

	const StackPtxInstruction** routines;
	size_t num_routines;
	size_t execution_limit;

	bool emit_spec_comment;
} StackPtxEmitLinearRegressionInfo;

STACK_PTX_EMIT_PUBLIC_DEC
const char*
stack_ptx_emit_linear_regression_result_to_string(
	StackPtxEmitLinearRegressionResult result
);

STACK_PTX_EMIT_PUBLIC_DEC
StackPtxEmitLinearRegressionResult
stack_ptx_emit_linear_regression_workspace_size(
	const StackPtxCompilerInfo* compiler_info_ref,
	const StackPtxStackInfo* stack_info_ref,
	const StackPtxEmitLinearRegressionInfo* regression_info_ref,
	size_t* workspace_in_bytes_out,
	StackPtxEmitResult* emit_result_out,
	StackPtxResult* stack_ptx_result_out
);

#ifdef __cplusplus
}
#endif

#endif // STACK_PTX_EMIT_LINEAR_REGRESSION_BASE_H_INCLUDE

#ifdef STACK_PTX_EMIT_LINEAR_REGRESSION_BASE_IMPLEMENTATION
#ifndef STACK_PTX_EMIT_LINEAR_REGRESSION_BASE_IMPLEMENTATION_ONCE
#define STACK_PTX_EMIT_LINEAR_REGRESSION_BASE_IMPLEMENTATION_ONCE

static
inline
StackPtxEmitLinearRegressionResult
_stack_ptx_emit_linear_regression_map_emit_result(
	StackPtxEmitResult emit_result
) {
	switch (emit_result) {
		case STACK_PTX_EMIT_SUCCESS:
			return STACK_PTX_EMIT_LINEAR_REGRESSION_SUCCESS;
		case STACK_PTX_EMIT_ERROR_INSUFFICIENT_BUFFER:
			return STACK_PTX_EMIT_LINEAR_REGRESSION_ERROR_INSUFFICIENT_BUFFER;
		case STACK_PTX_EMIT_ERROR_STACK_PTX:
			return STACK_PTX_EMIT_LINEAR_REGRESSION_ERROR_STACK_PTX;
		default:
			return STACK_PTX_EMIT_LINEAR_REGRESSION_ERROR_EMIT;
	}
}

static
inline
StackPtxEmitLinearRegressionResult
_stack_ptx_emit_linear_regression_validate_info(
	const StackPtxEmitLinearRegressionInfo* regression_info_ref
) {
	if (regression_info_ref == NULL ||
		regression_info_ref->feature_programs == NULL ||
		regression_info_ref->means == NULL ||
		regression_info_ref->stddevs == NULL ||
		regression_info_ref->beta_standardized == NULL ||
		regression_info_ref->num_feature_programs == 0 ||
		regression_info_ref->input_dim == 0) {
		return STACK_PTX_EMIT_LINEAR_REGRESSION_ERROR_INVALID_VALUE;
	}

	for (size_t i = 0; i < regression_info_ref->num_feature_programs; i++) {
		if (regression_info_ref->feature_programs[i].instructions == NULL ||
			regression_info_ref->stddevs[i] == 0.0f) {
			return STACK_PTX_EMIT_LINEAR_REGRESSION_ERROR_INVALID_VALUE;
		}
	}

	return STACK_PTX_EMIT_LINEAR_REGRESSION_SUCCESS;
}

static
inline
const char*
_stack_ptx_emit_linear_regression_identifier(
	const StackPtxEmitLinearRegressionInfo* regression_info_ref,
	size_t input_idx,
	char* fallback_buffer,
	size_t fallback_buffer_size
) {
	if (regression_info_ref->input_names != NULL &&
		input_idx < regression_info_ref->num_input_names &&
		regression_info_ref->input_names[input_idx] != NULL &&
		_stack_ptx_emit_is_identifier(regression_info_ref->input_names[input_idx])) {
		return regression_info_ref->input_names[input_idx];
	}
	snprintf(fallback_buffer, fallback_buffer_size, "x%zu", input_idx);
	return fallback_buffer;
}

static
inline
float
_stack_ptx_emit_linear_regression_compute_intercept(
	const StackPtxEmitLinearRegressionInfo* regression_info_ref
) {
	float intercept = 0.0f;
	for (size_t i = 0; i < regression_info_ref->num_feature_programs; i++) {
		intercept -= regression_info_ref->beta_standardized[i] *
			regression_info_ref->means[i] /
			regression_info_ref->stddevs[i];
	}
	return intercept;
}

static
inline
float
_stack_ptx_emit_linear_regression_compute_beta_unstandardized(
	const StackPtxEmitLinearRegressionInfo* regression_info_ref,
	size_t feature_idx
) {
	return regression_info_ref->beta_standardized[feature_idx] /
		regression_info_ref->stddevs[feature_idx];
}

static
inline
size_t
_stack_ptx_emit_linear_regression_max_size(
	size_t a,
	size_t b
) {
	return a > b ? a : b;
}

static
inline
StackPtxEmitResult
_stack_ptx_emit_linear_regression_format_float_text(
	float value,
	char* buffer,
	size_t buffer_size
) {
	if (buffer == NULL || buffer_size == 0) {
		return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
	}

	int bytes = snprintf(buffer, buffer_size, "%.9g", (double)value);
	if (bytes < 0 || (size_t)bytes >= buffer_size) {
		return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
	}
	return STACK_PTX_EMIT_SUCCESS;
}

static
inline
StackPtxEmitResult
_stack_ptx_emit_linear_regression_ascii_rule_row(
	StackPtxEmitWriter* writer,
	const char* row_prefix,
	size_t term_width,
	size_t standardized_width,
	size_t unstandardized_width
) {
	StackPtxEmitResult emit_result = _stack_ptx_emit_writer_append(writer, "%s+", row_prefix);
	if (emit_result != STACK_PTX_EMIT_SUCCESS) {
		return emit_result;
	}
	for (size_t i = 0; i < term_width + 2; i++) {
		emit_result = _stack_ptx_emit_writer_append(writer, "-");
		if (emit_result != STACK_PTX_EMIT_SUCCESS) {
			return emit_result;
		}
	}
	emit_result = _stack_ptx_emit_writer_append(writer, "+");
	if (emit_result != STACK_PTX_EMIT_SUCCESS) {
		return emit_result;
	}
	for (size_t i = 0; i < standardized_width + 2; i++) {
		emit_result = _stack_ptx_emit_writer_append(writer, "-");
		if (emit_result != STACK_PTX_EMIT_SUCCESS) {
			return emit_result;
		}
	}
	emit_result = _stack_ptx_emit_writer_append(writer, "+");
	if (emit_result != STACK_PTX_EMIT_SUCCESS) {
		return emit_result;
	}
	for (size_t i = 0; i < unstandardized_width + 2; i++) {
		emit_result = _stack_ptx_emit_writer_append(writer, "-");
		if (emit_result != STACK_PTX_EMIT_SUCCESS) {
			return emit_result;
		}
	}
	return _stack_ptx_emit_writer_append(writer, "+\n");
}

static
inline
StackPtxEmitResult
_stack_ptx_emit_linear_regression_ascii_value_row(
	StackPtxEmitWriter* writer,
	const char* row_prefix,
	size_t term_width,
	size_t standardized_width,
	size_t unstandardized_width,
	const char* term_text,
	const char* standardized_text,
	const char* unstandardized_text
) {
	return _stack_ptx_emit_writer_append(
		writer,
		"%s| %-*s | %-*s | %-*s |\n",
		row_prefix,
		(int)term_width,
		term_text,
		(int)standardized_width,
		standardized_text,
		(int)unstandardized_width,
		unstandardized_text
	);
}

static
inline
StackPtxEmitResult
_stack_ptx_emit_linear_regression_beta_table(
	const StackPtxEmitLinearRegressionInfo* regression_info_ref,
	const char* row_prefix,
	char* buffer,
	size_t buffer_size,
	size_t* buffer_bytes_written_ret
) {
	if (regression_info_ref == NULL ||
		row_prefix == NULL ||
		buffer_bytes_written_ret == NULL) {
		return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
	}

	const char* intercept_label = "intercept";
	const char* standardized_header = "beta_standardized";
	const char* unstandardized_header = "beta_unstandardized";
	size_t term_width = strlen("term");
	size_t standardized_width = strlen(standardized_header);
	size_t unstandardized_width = strlen(unstandardized_header);

	term_width = _stack_ptx_emit_linear_regression_max_size(term_width, strlen(intercept_label));
	standardized_width = _stack_ptx_emit_linear_regression_max_size(standardized_width, strlen("0.0"));

	char literal_buffer[64];
	StackPtxEmitResult emit_result = _stack_ptx_emit_linear_regression_format_float_text(
		_stack_ptx_emit_linear_regression_compute_intercept(regression_info_ref),
		literal_buffer,
		sizeof(literal_buffer)
	);
	if (emit_result != STACK_PTX_EMIT_SUCCESS) {
		return emit_result;
	}
	unstandardized_width = _stack_ptx_emit_linear_regression_max_size(
		unstandardized_width,
		strlen(literal_buffer)
	);

	for (size_t i = 0; i < regression_info_ref->num_feature_programs; i++) {
		char feature_name[64];
		int feature_name_bytes = snprintf(feature_name, sizeof(feature_name), "feature_%zu", i);
		if (feature_name_bytes < 0 || (size_t)feature_name_bytes >= sizeof(feature_name)) {
			return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
		}
		term_width = _stack_ptx_emit_linear_regression_max_size(term_width, strlen(feature_name));

		emit_result = _stack_ptx_emit_linear_regression_format_float_text(
			regression_info_ref->beta_standardized[i],
			literal_buffer,
			sizeof(literal_buffer)
		);
		if (emit_result != STACK_PTX_EMIT_SUCCESS) {
			return emit_result;
		}
		standardized_width = _stack_ptx_emit_linear_regression_max_size(
			standardized_width,
			strlen(literal_buffer)
		);

		emit_result = _stack_ptx_emit_linear_regression_format_float_text(
			_stack_ptx_emit_linear_regression_compute_beta_unstandardized(
				regression_info_ref,
				i
			),
			literal_buffer,
			sizeof(literal_buffer)
		);
		if (emit_result != STACK_PTX_EMIT_SUCCESS) {
			return emit_result;
		}
		unstandardized_width = _stack_ptx_emit_linear_regression_max_size(
			unstandardized_width,
			strlen(literal_buffer)
		);
	}

	StackPtxEmitWriter writer = {
		.buffer = buffer,
		.buffer_size = buffer_size,
		.bytes_written = 0
	};

	emit_result = _stack_ptx_emit_linear_regression_ascii_rule_row(
		&writer,
		row_prefix,
		term_width,
		standardized_width,
		unstandardized_width
	);
	if (emit_result != STACK_PTX_EMIT_SUCCESS) {
		*buffer_bytes_written_ret = writer.bytes_written;
		return emit_result;
	}

	emit_result = _stack_ptx_emit_linear_regression_ascii_value_row(
		&writer,
		row_prefix,
		term_width,
		standardized_width,
		unstandardized_width,
		"term",
		standardized_header,
		unstandardized_header
	);
	if (emit_result != STACK_PTX_EMIT_SUCCESS) {
		*buffer_bytes_written_ret = writer.bytes_written;
		return emit_result;
	}

	emit_result = _stack_ptx_emit_linear_regression_ascii_rule_row(
		&writer,
		row_prefix,
		term_width,
		standardized_width,
		unstandardized_width
	);
	if (emit_result != STACK_PTX_EMIT_SUCCESS) {
		*buffer_bytes_written_ret = writer.bytes_written;
		return emit_result;
	}

	emit_result = _stack_ptx_emit_linear_regression_format_float_text(
		_stack_ptx_emit_linear_regression_compute_intercept(regression_info_ref),
		literal_buffer,
		sizeof(literal_buffer)
	);
	if (emit_result != STACK_PTX_EMIT_SUCCESS) {
		return emit_result;
	}

	emit_result = _stack_ptx_emit_linear_regression_ascii_value_row(
		&writer,
		row_prefix,
		term_width,
		standardized_width,
		unstandardized_width,
		intercept_label,
		"0.0",
		literal_buffer
	);
	if (emit_result != STACK_PTX_EMIT_SUCCESS) {
		*buffer_bytes_written_ret = writer.bytes_written;
		return emit_result;
	}

	for (size_t i = 0; i < regression_info_ref->num_feature_programs; i++) {
		char feature_name[64];
		char standardized_literal[64];
		char unstandardized_literal[64];
		int feature_name_bytes = snprintf(feature_name, sizeof(feature_name), "feature_%zu", i);
		if (feature_name_bytes < 0 || (size_t)feature_name_bytes >= sizeof(feature_name)) {
			return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
		}

		emit_result = _stack_ptx_emit_linear_regression_format_float_text(
			regression_info_ref->beta_standardized[i],
			standardized_literal,
			sizeof(standardized_literal)
		);
		if (emit_result != STACK_PTX_EMIT_SUCCESS) {
			return emit_result;
		}
		emit_result = _stack_ptx_emit_linear_regression_format_float_text(
			_stack_ptx_emit_linear_regression_compute_beta_unstandardized(
				regression_info_ref,
				i
			),
			unstandardized_literal,
			sizeof(unstandardized_literal)
		);
		if (emit_result != STACK_PTX_EMIT_SUCCESS) {
			return emit_result;
		}

		emit_result = _stack_ptx_emit_linear_regression_ascii_value_row(
			&writer,
			row_prefix,
			term_width,
			standardized_width,
			unstandardized_width,
			feature_name,
			standardized_literal,
			unstandardized_literal
		);
		if (emit_result != STACK_PTX_EMIT_SUCCESS) {
			*buffer_bytes_written_ret = writer.bytes_written;
			return emit_result;
		}
	}

	emit_result = _stack_ptx_emit_linear_regression_ascii_rule_row(
		&writer,
		row_prefix,
		term_width,
		standardized_width,
		unstandardized_width
	);
	*buffer_bytes_written_ret = writer.bytes_written;
	return emit_result;
}

static
inline
size_t
_stack_ptx_emit_linear_regression_scratch_size(
	const StackPtxCompilerInfo* compiler_info_ref,
	const StackPtxStackInfo* stack_info_ref,
	const StackPtxEmitLinearRegressionInfo* regression_info_ref
) {
	size_t input_dim = regression_info_ref->input_dim;
	size_t max_ast_size = compiler_info_ref->max_ast_size;
	size_t num_special_registers = stack_info_ref->num_special_registers;

	size_t total = 0;
	total = _stack_ptx_emit_align_up(total, _Alignof(const char*));
	total += input_dim * sizeof(const char*);
	total = _stack_ptx_emit_align_up(total, _Alignof(char));
	total += input_dim * 64;

	total = _stack_ptx_emit_align_up(total, _Alignof(StackPtxRegister));
	total += (input_dim + 1) * sizeof(StackPtxRegister);
	total = _stack_ptx_emit_align_up(total, _Alignof(char));
	total += (input_dim + 1) * 32;

	total = _stack_ptx_emit_align_up(total, _Alignof(uint8_t));
	total += max_ast_size * sizeof(uint8_t);
	total = _stack_ptx_emit_align_up(total, _Alignof(size_t));
	total += max_ast_size * sizeof(size_t);
	total = _stack_ptx_emit_align_up(total, _Alignof(StackPtxAstIdx));
	total += max_ast_size * sizeof(StackPtxAstIdx);
	total = _stack_ptx_emit_align_up(total, _Alignof(const char*));
	total += (num_special_registers + max_ast_size) * sizeof(const char*);
	total = _stack_ptx_emit_align_up(total, _Alignof(char));
	total += max_ast_size * 32;
	total = _stack_ptx_emit_align_up(total, _Alignof(StackPtxInstruction));
	total += max_ast_size * sizeof(StackPtxInstruction);
	total = _stack_ptx_emit_align_up(total, _Alignof(StackPtxInstruction));
	total += max_ast_size * sizeof(StackPtxInstruction);

	return total;
}

static
inline
StackPtxEmitResult
_stack_ptx_emit_linear_regression_init_allocator(
	const StackPtxCompilerInfo* compiler_info_ref,
	const StackPtxStackInfo* stack_info_ref,
	const StackPtxEmitLinearRegressionInfo* regression_info_ref,
	void* workspace,
	size_t workspace_in_bytes,
	StackPtxEmitBumpAllocator* allocator_out,
	StackPtxResult* stack_ptx_result_out
) {
	if (allocator_out == NULL) {
		return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
	}

	size_t core_workspace_size = 0;
	StackPtxEmitResult emit_result = stack_ptx_emit_workspace_size(
		compiler_info_ref,
		stack_info_ref,
		&core_workspace_size,
		stack_ptx_result_out
	);
	if (emit_result != STACK_PTX_EMIT_SUCCESS) {
		return emit_result;
	}

	size_t required_workspace_size = core_workspace_size +
		_stack_ptx_emit_linear_regression_scratch_size(
			compiler_info_ref,
			stack_info_ref,
			regression_info_ref
		);
	if (workspace_in_bytes < required_workspace_size) {
		return STACK_PTX_EMIT_ERROR_ALLOCATION_FAILED;
	}

	return _stack_ptx_emit_bump_init(
		workspace,
		workspace_in_bytes,
		core_workspace_size,
		allocator_out
	);
}

static
inline
StackPtxEmitResult
_stack_ptx_emit_linear_regression_alloc_input_identifiers(
	const StackPtxEmitLinearRegressionInfo* regression_info_ref,
	StackPtxEmitBumpAllocator* allocator,
	const char*** input_identifiers_out
) {
	if (regression_info_ref == NULL || allocator == NULL || input_identifiers_out == NULL) {
		return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
	}

	const char** input_identifiers = NULL;
	StackPtxEmitResult emit_result = _stack_ptx_emit_bump_alloc(
		allocator,
		regression_info_ref->input_dim * sizeof(const char*),
		_Alignof(const char*),
		(void**)&input_identifiers
	);
	if (emit_result != STACK_PTX_EMIT_SUCCESS) {
		return emit_result;
	}

	char* fallback_storage = NULL;
	emit_result = _stack_ptx_emit_bump_alloc(
		allocator,
		regression_info_ref->input_dim * 64,
		_Alignof(char),
		(void**)&fallback_storage
	);
	if (emit_result != STACK_PTX_EMIT_SUCCESS) {
		return emit_result;
	}

	for (size_t i = 0; i < regression_info_ref->input_dim; i++) {
		input_identifiers[i] = _stack_ptx_emit_linear_regression_identifier(
			regression_info_ref,
			i,
			fallback_storage + (i * 64),
			64
		);
	}

	*input_identifiers_out = input_identifiers;
	return STACK_PTX_EMIT_SUCCESS;
}

static
inline
StackPtxEmitResult
_stack_ptx_emit_linear_regression_alloc_feature_registers(
	const StackPtxEmitLinearRegressionInfo* regression_info_ref,
	const char* const* input_identifiers,
	StackPtxEmitBumpAllocator* allocator,
	StackPtxRegister** registers_out,
	size_t* num_registers_out
) {
	if (regression_info_ref == NULL ||
		allocator == NULL ||
		registers_out == NULL ||
		num_registers_out == NULL) {
		return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
	}

	size_t num_registers = regression_info_ref->input_dim + 1;
	StackPtxRegister* registers = NULL;
	StackPtxEmitResult emit_result = _stack_ptx_emit_bump_alloc(
		allocator,
		num_registers * sizeof(StackPtxRegister),
		_Alignof(StackPtxRegister),
		(void**)&registers
	);
	if (emit_result != STACK_PTX_EMIT_SUCCESS) {
		return emit_result;
	}

	char* register_name_storage = NULL;
	emit_result = _stack_ptx_emit_bump_alloc(
		allocator,
		num_registers * 32,
		_Alignof(char),
		(void**)&register_name_storage
	);
	if (emit_result != STACK_PTX_EMIT_SUCCESS) {
		return emit_result;
	}

	for (size_t i = 0; i < regression_info_ref->input_dim; i++) {
		if (input_identifiers != NULL) {
			registers[i].name = input_identifiers[i];
		} else {
			char* register_name = register_name_storage + (i * 32);
			int bytes = snprintf(register_name, 32, "x%zu", i);
			if (bytes < 0 || bytes >= 32) {
				return STACK_PTX_EMIT_ERROR_INSUFFICIENT_BUFFER;
			}
			registers[i].name = register_name;
		}
		registers[i].stack_idx = STACK_PTX_STACK_TYPE_F32;
	}
	registers[regression_info_ref->input_dim].name = "feature_out";
	registers[regression_info_ref->input_dim].stack_idx = STACK_PTX_STACK_TYPE_F32;

	*registers_out = registers;
	*num_registers_out = num_registers;
	return STACK_PTX_EMIT_SUCCESS;
}

STACK_PTX_EMIT_PUBLIC_DEF
const char*
stack_ptx_emit_linear_regression_result_to_string(
	StackPtxEmitLinearRegressionResult result
) {
	switch (result) {
		case STACK_PTX_EMIT_LINEAR_REGRESSION_SUCCESS: return "STACK_PTX_EMIT_LINEAR_REGRESSION_SUCCESS";
		case STACK_PTX_EMIT_LINEAR_REGRESSION_ERROR_INVALID_VALUE: return "STACK_PTX_EMIT_LINEAR_REGRESSION_ERROR_INVALID_VALUE";
		case STACK_PTX_EMIT_LINEAR_REGRESSION_ERROR_UNSUPPORTED_BACKEND: return "STACK_PTX_EMIT_LINEAR_REGRESSION_ERROR_UNSUPPORTED_BACKEND";
		case STACK_PTX_EMIT_LINEAR_REGRESSION_ERROR_INSUFFICIENT_BUFFER: return "STACK_PTX_EMIT_LINEAR_REGRESSION_ERROR_INSUFFICIENT_BUFFER";
		case STACK_PTX_EMIT_LINEAR_REGRESSION_ERROR_EMIT: return "STACK_PTX_EMIT_LINEAR_REGRESSION_ERROR_EMIT";
		case STACK_PTX_EMIT_LINEAR_REGRESSION_ERROR_STACK_PTX: return "STACK_PTX_EMIT_LINEAR_REGRESSION_ERROR_STACK_PTX";
		case STACK_PTX_EMIT_LINEAR_REGRESSION_RESULT_NUM_ELEMS: break;
	}
	return "STACK_PTX_EMIT_LINEAR_REGRESSION_ERROR_INVALID_RESULT_ENUM";
}

STACK_PTX_EMIT_PUBLIC_DEF
StackPtxEmitLinearRegressionResult
stack_ptx_emit_linear_regression_workspace_size(
	const StackPtxCompilerInfo* compiler_info_ref,
	const StackPtxStackInfo* stack_info_ref,
	const StackPtxEmitLinearRegressionInfo* regression_info_ref,
	size_t* workspace_in_bytes_out,
	StackPtxEmitResult* emit_result_out,
	StackPtxResult* stack_ptx_result_out
) {
	if (workspace_in_bytes_out == NULL) {
		return STACK_PTX_EMIT_LINEAR_REGRESSION_ERROR_INVALID_VALUE;
	}
	StackPtxEmitLinearRegressionResult regression_validation =
		_stack_ptx_emit_linear_regression_validate_info(regression_info_ref);
	if (regression_validation != STACK_PTX_EMIT_LINEAR_REGRESSION_SUCCESS) {
		if (emit_result_out != NULL) {
			*emit_result_out = STACK_PTX_EMIT_SUCCESS;
		}
		if (stack_ptx_result_out != NULL) {
			*stack_ptx_result_out = STACK_PTX_SUCCESS;
		}
		return regression_validation;
	}

	size_t core_workspace_size = 0;
	StackPtxEmitResult emit_result = stack_ptx_emit_workspace_size(
		compiler_info_ref,
		stack_info_ref,
		&core_workspace_size,
		stack_ptx_result_out
	);
	if (emit_result_out != NULL) {
		*emit_result_out = emit_result;
	}
	if (emit_result != STACK_PTX_EMIT_SUCCESS) {
		return _stack_ptx_emit_linear_regression_map_emit_result(emit_result);
	}

	*workspace_in_bytes_out = core_workspace_size +
		_stack_ptx_emit_linear_regression_scratch_size(
			compiler_info_ref,
			stack_info_ref,
			regression_info_ref
		);
	return STACK_PTX_EMIT_LINEAR_REGRESSION_SUCCESS;
}

#endif
#endif
