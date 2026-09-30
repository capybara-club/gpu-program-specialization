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

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STACK_PTX_IMPLEMENTATION
#include <stack_ptx.h>

#define STACK_PTX_EMIT_IMPLEMENTATION
#include <stack_ptx_emit.h>

#define STACK_PTX_EMIT_LINEAR_REGRESSION_IMPLEMENTATION
#include <stack_ptx_emit_linear_regression.h>

#include <stack_ptx_tools_descriptions.h>
#include <stack_ptx_default_info.h>

#define STACK_PTX_LITERAL(x) ((StackPtxInstruction)x)

typedef struct {
	StackPtxEmitLanguage backend;
	bool emit_all;
	bool show_spec;
	const char* input_names_csv;
} ExampleCliOptions;

static const StackPtxEmitInstructionInfo c_instruction_info[STACK_PTX_PTX_INSTRUCTION_NUM_ENUMS] = {
	[STACK_PTX_PTX_INSTRUCTION_ABS_FTZ_F32] = STACK_PTX_EMIT_OP_FUNC("fabsf"),
	[STACK_PTX_PTX_INSTRUCTION_NEG_FTZ_F32] = STACK_PTX_EMIT_OP_PREFIX("-"),
	[STACK_PTX_PTX_INSTRUCTION_ADD_FTZ_F32] = STACK_PTX_EMIT_OP_INFIX("+"),
	[STACK_PTX_PTX_INSTRUCTION_SUB_FTZ_F32] = STACK_PTX_EMIT_OP_INFIX("-"),
	[STACK_PTX_PTX_INSTRUCTION_MUL_FTZ_F32] = STACK_PTX_EMIT_OP_INFIX("*"),
	[STACK_PTX_PTX_INSTRUCTION_MIN_FTZ_F32] = STACK_PTX_EMIT_OP_FUNC("fminf"),
	[STACK_PTX_PTX_INSTRUCTION_MAX_FTZ_F32] = STACK_PTX_EMIT_OP_FUNC("fmaxf"),
	[STACK_PTX_PTX_INSTRUCTION_COPYSIGN_F32] = STACK_PTX_EMIT_OP_FUNC("copysignf"),
	[STACK_PTX_PTX_INSTRUCTION_FMA_RN_FTZ_F32] = STACK_PTX_EMIT_OP_FUNC("fmaf"),
};

static const StackPtxEmitInstructionInfo rust_instruction_info[STACK_PTX_PTX_INSTRUCTION_NUM_ENUMS] = {
	[STACK_PTX_PTX_INSTRUCTION_ABS_FTZ_F32] = STACK_PTX_EMIT_OP_FUNC("f32::abs"),
	[STACK_PTX_PTX_INSTRUCTION_NEG_FTZ_F32] = STACK_PTX_EMIT_OP_PREFIX("-"),
	[STACK_PTX_PTX_INSTRUCTION_ADD_FTZ_F32] = STACK_PTX_EMIT_OP_INFIX("+"),
	[STACK_PTX_PTX_INSTRUCTION_SUB_FTZ_F32] = STACK_PTX_EMIT_OP_INFIX("-"),
	[STACK_PTX_PTX_INSTRUCTION_MUL_FTZ_F32] = STACK_PTX_EMIT_OP_INFIX("*"),
	[STACK_PTX_PTX_INSTRUCTION_MIN_FTZ_F32] = STACK_PTX_EMIT_OP_FUNC("f32::min"),
	[STACK_PTX_PTX_INSTRUCTION_MAX_FTZ_F32] = STACK_PTX_EMIT_OP_FUNC("f32::max"),
	[STACK_PTX_PTX_INSTRUCTION_COPYSIGN_F32] = STACK_PTX_EMIT_OP_FUNC("f32::copysign"),
	[STACK_PTX_PTX_INSTRUCTION_FMA_RN_FTZ_F32] = STACK_PTX_EMIT_OP_FUNC("f32::mul_add"),
};

static const StackPtxEmitInstructionInfo numpy_instruction_info[STACK_PTX_PTX_INSTRUCTION_NUM_ENUMS] = {
	[STACK_PTX_PTX_INSTRUCTION_ABS_FTZ_F32] = STACK_PTX_EMIT_OP_FUNC("np.abs"),
	[STACK_PTX_PTX_INSTRUCTION_NEG_FTZ_F32] = STACK_PTX_EMIT_OP_PREFIX("-"),
	[STACK_PTX_PTX_INSTRUCTION_ADD_FTZ_F32] = STACK_PTX_EMIT_OP_INFIX("+"),
	[STACK_PTX_PTX_INSTRUCTION_SUB_FTZ_F32] = STACK_PTX_EMIT_OP_INFIX("-"),
	[STACK_PTX_PTX_INSTRUCTION_MUL_FTZ_F32] = STACK_PTX_EMIT_OP_INFIX("*"),
	[STACK_PTX_PTX_INSTRUCTION_MIN_FTZ_F32] = STACK_PTX_EMIT_OP_FUNC("np.minimum"),
	[STACK_PTX_PTX_INSTRUCTION_MAX_FTZ_F32] = STACK_PTX_EMIT_OP_FUNC("np.maximum"),
	[STACK_PTX_PTX_INSTRUCTION_COPYSIGN_F32] = STACK_PTX_EMIT_OP_FUNC("np.copysign"),
	[STACK_PTX_PTX_INSTRUCTION_FMA_RN_FTZ_F32] = STACK_PTX_EMIT_OP_MULADD,
};

static const char* c_stack_type_names[STACK_PTX_STACK_TYPE_NUM_ENUMS] = {
	[STACK_PTX_STACK_TYPE_F32] = "float"
};

static const char* rust_stack_type_names[STACK_PTX_STACK_TYPE_NUM_ENUMS] = {
	[STACK_PTX_STACK_TYPE_F32] = "f32"
};

static const char* numpy_stack_type_names[STACK_PTX_STACK_TYPE_NUM_ENUMS] = {
	[STACK_PTX_STACK_TYPE_F32] = "np.float32"
};

static const StackPtxEmitCSettings base_c_settings = {
	.function_name = NULL,
	.function_qualifiers = "static inline",
	.extra_params = NULL,
	.stack_type_names = c_stack_type_names,
	.num_stack_type_names = STACK_PTX_ARRAY_NUM_ELEMS(c_stack_type_names),
	.instruction_info = c_instruction_info,
	.num_instruction_info = STACK_PTX_ARRAY_NUM_ELEMS(c_instruction_info),
	.special_register_exprs = NULL,
	.num_special_register_exprs = 0
};

static const StackPtxEmitCudaSettings base_cuda_settings = {
	.function_name = NULL,
	.function_qualifiers = "static __device__ __forceinline__",
	.extra_params = NULL,
	.stack_type_names = c_stack_type_names,
	.num_stack_type_names = STACK_PTX_ARRAY_NUM_ELEMS(c_stack_type_names),
	.instruction_info = c_instruction_info,
	.num_instruction_info = STACK_PTX_ARRAY_NUM_ELEMS(c_instruction_info),
	.special_register_exprs = NULL,
	.num_special_register_exprs = 0
};

static const StackPtxEmitRustSettings base_rust_settings = {
	.function_name = NULL,
	.function_qualifiers = "pub",
	.extra_params = NULL,
	.stack_type_names = rust_stack_type_names,
	.num_stack_type_names = STACK_PTX_ARRAY_NUM_ELEMS(rust_stack_type_names),
	.instruction_info = rust_instruction_info,
	.num_instruction_info = STACK_PTX_ARRAY_NUM_ELEMS(rust_instruction_info),
	.special_register_exprs = NULL,
	.num_special_register_exprs = 0
};

static const StackPtxEmitNumpySettings base_numpy_settings = {
	.function_name = NULL,
	.function_qualifiers = NULL,
	.extra_params = NULL,
	.instruction_info = numpy_instruction_info,
	.num_instruction_info = STACK_PTX_ARRAY_NUM_ELEMS(numpy_instruction_info),
	.special_register_exprs = NULL,
	.num_special_register_exprs = 0
};

static const StackPtxEmitLatexSettings base_latex_settings = {
	.document_title = "Linear Regression Model Specification",
	.output_symbol = "y",
	.special_register_exprs = NULL,
	.num_special_register_exprs = 0,
	.emit_document = true,
	.use_input_indices = true
};

static const StackPtxEmitMarkdownSettings base_markdown_settings = {
	.document_title = "Linear Regression Model Specification",
	.output_symbol = "y",
	.special_register_exprs = NULL,
	.num_special_register_exprs = 0,
	.use_input_indices = true,
	.factor_expression_min_bytes = 96
};

static
bool
parse_backend(
	const char* text,
	StackPtxEmitLanguage* backend_out
) {
	if (strcmp(text, "c") == 0) {
		*backend_out = STACK_PTX_EMIT_LANGUAGE_C;
		return true;
	}
	if (strcmp(text, "cuda") == 0) {
		*backend_out = STACK_PTX_EMIT_LANGUAGE_CUDA;
		return true;
	}
	if (strcmp(text, "rust") == 0) {
		*backend_out = STACK_PTX_EMIT_LANGUAGE_RUST;
		return true;
	}
	if (strcmp(text, "numpy") == 0) {
		*backend_out = STACK_PTX_EMIT_LANGUAGE_NUMPY;
		return true;
	}
	if (strcmp(text, "latex") == 0) {
		*backend_out = STACK_PTX_EMIT_LANGUAGE_LATEX;
		return true;
	}
	if (strcmp(text, "markdown") == 0) {
		*backend_out = STACK_PTX_EMIT_LANGUAGE_MARKDOWN;
		return true;
	}
	return false;
}

static
bool
parse_cli_options(
	int argc,
	char** argv,
	ExampleCliOptions* options_out
) {
	ExampleCliOptions options = {
		.backend = STACK_PTX_EMIT_LANGUAGE_C,
		.emit_all = true,
		.show_spec = false,
		.input_names_csv = NULL
	};

	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "--backend") == 0) {
			if (i + 1 >= argc) {
				return false;
			}
			const char* value = argv[++i];
			if (strcmp(value, "all") == 0) {
				options.emit_all = true;
			} else if (parse_backend(value, &options.backend)) {
				options.emit_all = false;
			} else {
				return false;
			}
		} else if (strcmp(argv[i], "--show-spec") == 0) {
			options.show_spec = true;
		} else if (strcmp(argv[i], "--input-names") == 0) {
			if (i + 1 >= argc) {
				return false;
			}
			options.input_names_csv = argv[++i];
		} else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
			return false;
		} else {
			return false;
		}
	}

	*options_out = options;
	return true;
}

static
void
print_usage(
	const char* argv0
) {
	printf("Usage: %s [--backend c|cuda|rust|numpy|latex|markdown|all] [--show-spec] [--input-names name0,name1,...]\n", argv0);
	printf("Print a hard-coded regression inference function built from Stack PTX feature programs.\n");
}

static
void
free_string_array(
	char** values,
	size_t count
) {
	if (values == NULL) {
		return;
	}
	for (size_t i = 0; i < count; i++) {
		free(values[i]);
	}
	free(values);
}

static
bool
parse_input_names_csv(
	const char* csv,
	size_t expected_count,
	char*** names_out,
	size_t* num_names_out
) {
	*names_out = NULL;
	*num_names_out = 0;

	if (csv == NULL) {
		return true;
	}

	char* csv_copy = strdup(csv);
	if (csv_copy == NULL) {
		return false;
	}

	char** names = (char**)calloc(expected_count, sizeof(char*));
	if (names == NULL) {
		free(csv_copy);
		return false;
	}

	size_t count = 0;
	char* save_ptr = NULL;
	for (char* token = strtok_r(csv_copy, ",", &save_ptr);
		token != NULL;
		token = strtok_r(NULL, ",", &save_ptr)) {
		while (*token == ' ' || *token == '\t') {
			token++;
		}
		size_t token_size = strlen(token);
		while (token_size > 0 &&
			(token[token_size - 1] == ' ' || token[token_size - 1] == '\t')) {
			token[--token_size] = '\0';
		}

		if (count >= expected_count || token[0] == '\0') {
			free_string_array(names, expected_count);
			free(csv_copy);
			return false;
		}

		names[count] = strdup(token);
		if (names[count] == NULL) {
			free_string_array(names, expected_count);
			free(csv_copy);
			return false;
		}
		count++;
	}

	free(csv_copy);
	if (count != expected_count) {
		free_string_array(names, expected_count);
		return false;
	}

	*names_out = names;
	*num_names_out = count;
	return true;
}

static
bool
make_emit_settings(
	StackPtxEmitLanguage backend,
	StackPtxEmitSettings* settings_out
) {
	if (settings_out == NULL) {
		return false;
	}
	StackPtxEmitSettings settings = {
		.language = backend,
		.vector = { .kind = STACK_PTX_EMIT_VECTOR_KIND_NONE, .bits = 0 }
	};
	switch (backend) {
		case STACK_PTX_EMIT_LANGUAGE_C:
			settings.as.c = base_c_settings;
			break;
		case STACK_PTX_EMIT_LANGUAGE_CUDA:
			settings.as.cuda = base_cuda_settings;
			break;
		case STACK_PTX_EMIT_LANGUAGE_RUST:
			settings.as.rust = base_rust_settings;
			break;
		case STACK_PTX_EMIT_LANGUAGE_NUMPY:
			settings.as.numpy = base_numpy_settings;
			break;
		case STACK_PTX_EMIT_LANGUAGE_LATEX:
			settings.as.latex = base_latex_settings;
			break;
		case STACK_PTX_EMIT_LANGUAGE_MARKDOWN:
			settings.as.markdown = base_markdown_settings;
			break;
		case STACK_PTX_EMIT_LANGUAGE_PTX:
		case STACK_PTX_EMIT_LANGUAGE_CPP:
		case STACK_PTX_EMIT_LANGUAGE_PYTORCH:
		case STACK_PTX_EMIT_LANGUAGE_NUM_ENUMS:
			return false;
	}
	*settings_out = settings;
	return true;
}

static
bool
generate_linear_regression_text(
	StackPtxEmitLanguage backend,
	bool emit_spec_comment,
	const char* const* input_names,
	size_t num_input_names,
	char** generated_text_out
) {
	static const StackPtxInstruction feature_0[] = {
		/* x_0 * x_1 + x_2 */
		STACK_PTX_LITERAL(stack_ptx_encode_input(2)),
		STACK_PTX_LITERAL(stack_ptx_encode_input(1)),
		STACK_PTX_LITERAL(stack_ptx_encode_input(0)),
		stack_ptx_encode_ptx_instruction_fma_rn_ftz_f32,
		STACK_PTX_LITERAL(stack_ptx_encode_return)
	};

	static const StackPtxInstruction feature_1[] = {
		/* abs(x_3 - x_1) */
		STACK_PTX_LITERAL(stack_ptx_encode_input(1)),
		STACK_PTX_LITERAL(stack_ptx_encode_input(3)),
		stack_ptx_encode_ptx_instruction_sub_ftz_f32,
		stack_ptx_encode_ptx_instruction_abs_ftz_f32,
		STACK_PTX_LITERAL(stack_ptx_encode_return)
	};

	static const StackPtxInstruction feature_2[] = {
		/* max(x_0, -x_2) */
		STACK_PTX_LITERAL(stack_ptx_encode_input(2)),
		stack_ptx_encode_ptx_instruction_neg_ftz_f32,
		STACK_PTX_LITERAL(stack_ptx_encode_input(0)),
		stack_ptx_encode_ptx_instruction_max_ftz_f32,
		STACK_PTX_LITERAL(stack_ptx_encode_return)
	};

	static const StackPtxEmitLinearRegressionFeatureProgram feature_programs[] = {
		{ .name = "interaction term", .instructions = feature_0 },
		{ .name = "abs gap", .instructions = feature_1 },
		{ .name = "score-cap", .instructions = feature_2 },
	};

	static const float means[] = { 2.5f, 1.0f, -0.5f };
	static const float stddevs[] = { 0.5f, 2.0f, 4.0f };
	static const float beta_standardized[] = { 1.25f, -0.75f, 0.5f };

	StackPtxEmitLinearRegressionInfo regression_info = {
		.function_name = "mock_linear_regression_infer",
		.input_dim = 4,
		.input_names = input_names,
		.num_input_names = num_input_names,
		.feature_programs = feature_programs,
		.num_feature_programs = STACK_PTX_ARRAY_NUM_ELEMS(feature_programs),
		.means = means,
		.stddevs = stddevs,
		.beta_standardized = beta_standardized,
		.routines = NULL,
		.num_routines = 0,
		.execution_limit = 1024,
		.emit_spec_comment = emit_spec_comment
	};

	StackPtxEmitSettings settings;
	if (!make_emit_settings(backend, &settings)) {
		fprintf(stderr, "unsupported backend\n");
		return false;
	}

	size_t workspace_size = 0;
	StackPtxEmitResult emit_result = STACK_PTX_EMIT_SUCCESS;
	StackPtxResult stack_result = STACK_PTX_SUCCESS;
	StackPtxEmitLinearRegressionResult regression_result =
		stack_ptx_emit_linear_regression_workspace_size(
			&compiler_info,
			&stack_ptx_stack_info,
			&regression_info,
			&workspace_size,
			&emit_result,
			&stack_result
		);
	if (regression_result != STACK_PTX_EMIT_LINEAR_REGRESSION_SUCCESS) {
		fprintf(
			stderr,
			"workspace_size failed: %s (emit=%s, stack_ptx=%s)\n",
			stack_ptx_emit_linear_regression_result_to_string(regression_result),
			stack_ptx_emit_result_to_string(emit_result),
			stack_ptx_result_to_string(stack_result)
		);
		return false;
	}

	void* workspace = malloc(workspace_size);
	if (workspace == NULL) {
		fprintf(stderr, "workspace allocation failed for %zu bytes\n", workspace_size);
		return false;
	}

	size_t required_bytes = 0;
	regression_result = stack_ptx_emit_linear_regression_compile(
		&compiler_info,
		&stack_ptx_stack_info,
		&settings,
		&regression_info,
		workspace,
		workspace_size,
		NULL,
		0,
		&required_bytes,
		&emit_result,
		&stack_result
	);
	if (regression_result != STACK_PTX_EMIT_LINEAR_REGRESSION_SUCCESS) {
		fprintf(
			stderr,
			"measure compile failed: %s (emit=%s, stack_ptx=%s)\n",
			stack_ptx_emit_linear_regression_result_to_string(regression_result),
			stack_ptx_emit_result_to_string(emit_result),
			stack_ptx_result_to_string(stack_result)
		);
		free(workspace);
		return false;
	}

	char* generated_text = (char*)malloc(required_bytes + 1);
	if (generated_text == NULL) {
		fprintf(stderr, "output allocation failed for %zu bytes\n", required_bytes + 1);
		free(workspace);
		return false;
	}

	regression_result = stack_ptx_emit_linear_regression_compile(
		&compiler_info,
		&stack_ptx_stack_info,
		&settings,
		&regression_info,
		workspace,
		workspace_size,
		generated_text,
		required_bytes + 1,
		&required_bytes,
		&emit_result,
		&stack_result
	);
	free(workspace);
	if (regression_result != STACK_PTX_EMIT_LINEAR_REGRESSION_SUCCESS) {
		fprintf(
			stderr,
			"emit compile failed: %s (emit=%s, stack_ptx=%s)\n",
			stack_ptx_emit_linear_regression_result_to_string(regression_result),
			stack_ptx_emit_result_to_string(emit_result),
			stack_ptx_result_to_string(stack_result)
		);
		free(generated_text);
		return false;
	}

	*generated_text_out = generated_text;
	return true;
}

static
bool
emit_backend_output(
	StackPtxEmitLanguage backend,
	bool emit_spec_comment,
	const char* header,
	const char* const* input_names,
	size_t num_input_names
) {
	char* generated_text = NULL;
	if (!generate_linear_regression_text(
			backend,
			emit_spec_comment,
			input_names,
			num_input_names,
			&generated_text)) {
		return false;
	}

	if (header != NULL) {
		printf("%s\n", header);
	}
	fputs(generated_text, stdout);
	free(generated_text);
	return true;
}

int
main(
	int argc,
	char** argv
) {
	ExampleCliOptions options;
	if (!parse_cli_options(argc, argv, &options)) {
		print_usage(argv[0]);
		return 0;
	}

	char** input_names = NULL;
	size_t num_input_names = 0;
	if (!parse_input_names_csv(options.input_names_csv, 4, &input_names, &num_input_names)) {
		fprintf(stderr, "failed to parse --input-names; expected exactly 4 comma-separated names\n");
		return 1;
	}

	bool ok = true;
	if (options.emit_all) {
		ok = emit_backend_output(
				STACK_PTX_EMIT_LANGUAGE_C,
				options.show_spec,
				"/* ==== C ==== */",
				(const char* const*)input_names,
				num_input_names
			) &&
			emit_backend_output(
				STACK_PTX_EMIT_LANGUAGE_CUDA,
				false,
				"\n/* ==== CUDA ==== */",
				(const char* const*)input_names,
				num_input_names
			) &&
			emit_backend_output(
				STACK_PTX_EMIT_LANGUAGE_RUST,
				false,
				"\n// ==== Rust ====",
				(const char* const*)input_names,
				num_input_names
			) &&
			emit_backend_output(
				STACK_PTX_EMIT_LANGUAGE_NUMPY,
				false,
				"\n# ==== NumPy ====",
				(const char* const*)input_names,
				num_input_names
			) &&
			emit_backend_output(
				STACK_PTX_EMIT_LANGUAGE_LATEX,
				false,
				"\n% ==== LaTeX ====",
				(const char* const*)input_names,
				num_input_names
			) &&
			emit_backend_output(
				STACK_PTX_EMIT_LANGUAGE_MARKDOWN,
				false,
				"\n# ==== Markdown ====",
				(const char* const*)input_names,
				num_input_names
			);
	} else {
		ok = emit_backend_output(
			options.backend,
			options.show_spec,
			NULL,
			(const char* const*)input_names,
			num_input_names
		);
	}

	free_string_array(input_names, num_input_names);
	return ok ? 0 : 1;
}
