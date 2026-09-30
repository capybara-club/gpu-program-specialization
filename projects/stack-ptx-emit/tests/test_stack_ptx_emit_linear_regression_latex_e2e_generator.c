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

enum {
	LATEX_INPUT_DIM = 5,
	LATEX_NUM_FEATURES = 4
};

static const StackPtxInstruction feature_0[] = {
	/* (x_0 * x_1) + x_2 */
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

static const StackPtxInstruction feature_3[] = {
	/* min(x_1 + x_4, x_0 * x_0) */
	STACK_PTX_LITERAL(stack_ptx_encode_input(4)),
	STACK_PTX_LITERAL(stack_ptx_encode_input(1)),
	stack_ptx_encode_ptx_instruction_add_ftz_f32,
	STACK_PTX_LITERAL(stack_ptx_encode_input(0)),
	STACK_PTX_LITERAL(stack_ptx_encode_input(0)),
	stack_ptx_encode_ptx_instruction_mul_ftz_f32,
	stack_ptx_encode_ptx_instruction_min_ftz_f32,
	STACK_PTX_LITERAL(stack_ptx_encode_return)
};

static const StackPtxEmitLinearRegressionFeatureProgram feature_programs[LATEX_NUM_FEATURES] = {
	{ .name = NULL, .instructions = feature_0 },
	{ .name = NULL, .instructions = feature_1 },
	{ .name = NULL, .instructions = feature_2 },
	{ .name = NULL, .instructions = feature_3 },
};

static const float means[LATEX_NUM_FEATURES] = { 2.0f, 1.0f, -0.5f, 1.5f };
static const float stddevs[LATEX_NUM_FEATURES] = { 2.0f, 4.0f, 0.5f, 3.0f };
static const float beta_standardized[LATEX_NUM_FEATURES] = { 4.0f, -8.0f, 1.0f, 6.0f };

static const char* input_names[LATEX_INPUT_DIM] = {
	"age",
	"income",
	"score",
	"balance",
	"bonus"
};

static const StackPtxEmitLatexSettings base_latex_settings = {
	.document_title = "Linear Regression Model Specification",
	.output_symbol = "y",
	.special_register_exprs = NULL,
	.num_special_register_exprs = 0,
	.emit_document = true,
	.use_input_indices = true
};

static
bool
write_text_file(
	const char* path,
	const char* text
) {
	FILE* file = fopen(path, "w");
	if (file == NULL) {
		return false;
	}
	size_t text_len = strlen(text);
	bool ok = fwrite(text, 1, text_len, file) == text_len;
	ok = ok && fclose(file) == 0;
	return ok;
}

static
bool
generate_latex_model(
	char** generated_text_out
) {
	StackPtxEmitLinearRegressionInfo regression_info = {
		.function_name = "linear_regression_infer",
		.input_dim = LATEX_INPUT_DIM,
		.input_names = input_names,
		.num_input_names = LATEX_INPUT_DIM,
		.feature_programs = feature_programs,
		.num_feature_programs = LATEX_NUM_FEATURES,
		.means = means,
		.stddevs = stddevs,
		.beta_standardized = beta_standardized,
		.routines = NULL,
		.num_routines = 0,
		.execution_limit = 1024,
		.emit_spec_comment = false
	};

	StackPtxEmitSettings settings = {
		.language = STACK_PTX_EMIT_LANGUAGE_LATEX,
		.vector = { .kind = STACK_PTX_EMIT_VECTOR_KIND_NONE, .bits = 0 },
		.as.latex = base_latex_settings
	};

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
		fprintf(stderr, "workspace_size failed\n");
		return false;
	}

	void* workspace = malloc(workspace_size);
	if (workspace == NULL) {
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
		free(workspace);
		fprintf(stderr, "measure emit failed\n");
		return false;
	}

	char* generated_text = (char*)malloc(required_bytes + 1);
	if (generated_text == NULL) {
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
		free(generated_text);
		fprintf(stderr, "emit failed\n");
		return false;
	}

	*generated_text_out = generated_text;
	return true;
}

int
main(
	int argc,
	char** argv
) {
	if (argc != 2) {
		fprintf(stderr, "Usage: %s OUTPUT_DIR\n", argv[0]);
		return 1;
	}

	char path[4096];
	int path_bytes = snprintf(path, sizeof(path), "%s/linear_regression_model.tex", argv[1]);
	if (path_bytes < 0 || (size_t)path_bytes >= sizeof(path)) {
		return 1;
	}

	char* generated_text = NULL;
	if (!generate_latex_model(&generated_text)) {
		return 1;
	}

	bool ok = write_text_file(path, generated_text);
	free(generated_text);
	return ok ? 0 : 1;
}
