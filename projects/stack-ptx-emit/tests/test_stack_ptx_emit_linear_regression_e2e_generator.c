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

#include <math.h>
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
	E2E_INPUT_DIM = 5,
	E2E_NUM_FEATURES = 4,
	E2E_NUM_ROWS = 5
};

static const StackPtxEmitInstructionInfo c_instruction_info[STACK_PTX_PTX_INSTRUCTION_NUM_ENUMS] = {
	[STACK_PTX_PTX_INSTRUCTION_ABS_FTZ_F32] = STACK_PTX_EMIT_OP_FUNC("fabsf"),
	[STACK_PTX_PTX_INSTRUCTION_NEG_FTZ_F32] = STACK_PTX_EMIT_OP_PREFIX("-"),
	[STACK_PTX_PTX_INSTRUCTION_ADD_FTZ_F32] = STACK_PTX_EMIT_OP_INFIX("+"),
	[STACK_PTX_PTX_INSTRUCTION_SUB_FTZ_F32] = STACK_PTX_EMIT_OP_INFIX("-"),
	[STACK_PTX_PTX_INSTRUCTION_MUL_FTZ_F32] = STACK_PTX_EMIT_OP_INFIX("*"),
	[STACK_PTX_PTX_INSTRUCTION_MIN_FTZ_F32] = STACK_PTX_EMIT_OP_FUNC("fminf"),
	[STACK_PTX_PTX_INSTRUCTION_MAX_FTZ_F32] = STACK_PTX_EMIT_OP_FUNC("fmaxf"),
};

static const StackPtxEmitInstructionInfo rust_instruction_info[STACK_PTX_PTX_INSTRUCTION_NUM_ENUMS] = {
	[STACK_PTX_PTX_INSTRUCTION_ABS_FTZ_F32] = STACK_PTX_EMIT_OP_FUNC("f32::abs"),
	[STACK_PTX_PTX_INSTRUCTION_NEG_FTZ_F32] = STACK_PTX_EMIT_OP_PREFIX("-"),
	[STACK_PTX_PTX_INSTRUCTION_ADD_FTZ_F32] = STACK_PTX_EMIT_OP_INFIX("+"),
	[STACK_PTX_PTX_INSTRUCTION_SUB_FTZ_F32] = STACK_PTX_EMIT_OP_INFIX("-"),
	[STACK_PTX_PTX_INSTRUCTION_MUL_FTZ_F32] = STACK_PTX_EMIT_OP_INFIX("*"),
	[STACK_PTX_PTX_INSTRUCTION_MIN_FTZ_F32] = STACK_PTX_EMIT_OP_FUNC("f32::min"),
	[STACK_PTX_PTX_INSTRUCTION_MAX_FTZ_F32] = STACK_PTX_EMIT_OP_FUNC("f32::max"),
};

static const StackPtxEmitInstructionInfo numpy_instruction_info[STACK_PTX_PTX_INSTRUCTION_NUM_ENUMS] = {
	[STACK_PTX_PTX_INSTRUCTION_ABS_FTZ_F32] = STACK_PTX_EMIT_OP_FUNC("np.abs"),
	[STACK_PTX_PTX_INSTRUCTION_NEG_FTZ_F32] = STACK_PTX_EMIT_OP_PREFIX("-"),
	[STACK_PTX_PTX_INSTRUCTION_ADD_FTZ_F32] = STACK_PTX_EMIT_OP_INFIX("+"),
	[STACK_PTX_PTX_INSTRUCTION_SUB_FTZ_F32] = STACK_PTX_EMIT_OP_INFIX("-"),
	[STACK_PTX_PTX_INSTRUCTION_MUL_FTZ_F32] = STACK_PTX_EMIT_OP_INFIX("*"),
	[STACK_PTX_PTX_INSTRUCTION_MIN_FTZ_F32] = STACK_PTX_EMIT_OP_FUNC("np.minimum"),
	[STACK_PTX_PTX_INSTRUCTION_MAX_FTZ_F32] = STACK_PTX_EMIT_OP_FUNC("np.maximum"),
};

static const char* c_stack_type_names[STACK_PTX_STACK_TYPE_NUM_ENUMS] = {
	[STACK_PTX_STACK_TYPE_F32] = "float"
};

static const char* rust_stack_type_names[STACK_PTX_STACK_TYPE_NUM_ENUMS] = {
	[STACK_PTX_STACK_TYPE_F32] = "f32"
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

static const StackPtxInstruction feature_0[] = {
	/* (x_0 * x_1) + x_2 */
	STACK_PTX_LITERAL(stack_ptx_encode_input(1)),
	STACK_PTX_LITERAL(stack_ptx_encode_input(0)),
	stack_ptx_encode_ptx_instruction_mul_ftz_f32,
	STACK_PTX_LITERAL(stack_ptx_encode_input(2)),
	stack_ptx_encode_ptx_instruction_add_ftz_f32,
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

static const StackPtxEmitLinearRegressionFeatureProgram feature_programs[E2E_NUM_FEATURES] = {
	{ .name = NULL, .instructions = feature_0 },
	{ .name = NULL, .instructions = feature_1 },
	{ .name = NULL, .instructions = feature_2 },
	{ .name = NULL, .instructions = feature_3 },
};

static const float means[E2E_NUM_FEATURES] = { 2.0f, 1.0f, -0.5f, 1.5f };
static const float stddevs[E2E_NUM_FEATURES] = { 2.0f, 4.0f, 0.5f, 3.0f };
static const float beta_standardized[E2E_NUM_FEATURES] = { 4.0f, -8.0f, 1.0f, 6.0f };

static const float dataset[E2E_NUM_ROWS][E2E_INPUT_DIM] = {
	{ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f },
	{ 2.0f, -1.0f, 4.0f, 0.0f, 3.0f },
	{ -3.0f, 2.0f, -1.0f, 5.0f, -2.0f },
	{ 0.5f, 1.5f, -2.0f, 1.0f, -1.0f },
	{ -1.25f, -0.5f, 2.0f, -3.0f, 4.5f },
};

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
open_output_file(
	const char* output_dir,
	const char* file_name,
	char* path_buffer,
	size_t path_buffer_size,
	FILE** file_out
) {
	int bytes = snprintf(path_buffer, path_buffer_size, "%s/%s", output_dir, file_name);
	if (bytes < 0 || (size_t)bytes >= path_buffer_size) {
		return false;
	}
	*file_out = fopen(path_buffer, "w");
	return *file_out != NULL;
}

static
void
format_float_literal(
	float value,
	const char* suffix,
	char* buffer,
	size_t buffer_size
) {
	char number[64];
	int number_bytes = snprintf(number, sizeof(number), "%.9g", value);
	if (number_bytes < 0 || (size_t)number_bytes >= sizeof(number)) {
		number[0] = '\0';
		return;
	}

	bool has_decimal_or_exponent = false;
	for (size_t i = 0; number[i] != '\0'; i++) {
		if (number[i] == '.' || number[i] == 'e' || number[i] == 'E') {
			has_decimal_or_exponent = true;
			break;
		}
	}

	snprintf(
		buffer,
		buffer_size,
		has_decimal_or_exponent ? "%s%s" : "%s.0%s",
		number,
		suffix
	);
}

static
float
reference_infer(
	const float* x
) {
	float betas[E2E_NUM_FEATURES];
	float intercept = 0.0f;
	for (size_t i = 0; i < E2E_NUM_FEATURES; i++) {
		betas[i] = beta_standardized[i] / stddevs[i];
		intercept -= beta_standardized[i] * means[i] / stddevs[i];
	}

	float phi_0 = (x[0] * x[1]) + x[2];
	float phi_1 = fabsf(x[3] - x[1]);
	float phi_2 = fmaxf(x[0], -x[2]);
	float phi_3 = fminf(x[1] + x[4], x[0] * x[0]);

	float y = intercept;
	y += betas[0] * phi_0;
	y += betas[1] * phi_1;
	y += betas[2] * phi_2;
	y += betas[3] * phi_3;
	return y;
}

static
bool
write_expected_file(
	const char* output_dir,
	uint32_t* expected_bits_out
) {
	char path[4096];
	FILE* file = NULL;
	if (!open_output_file(output_dir, "expected.txt", path, sizeof(path), &file)) {
		return false;
	}

	for (size_t row = 0; row < E2E_NUM_ROWS; row++) {
		union {
			float f;
			uint32_t u;
		} bits = { .f = reference_infer(dataset[row]) };
		expected_bits_out[row] = bits.u;
		if (fprintf(file, "0x%08X\n", (unsigned)bits.u) <= 0) {
			fclose(file);
			return false;
		}
	}

	return fclose(file) == 0;
}

static
bool
generate_linear_regression_text(
	StackPtxEmitLanguage backend,
	char** generated_text_out
) {
	StackPtxEmitLinearRegressionInfo regression_info = {
		.function_name = "linear_regression_infer",
		.input_dim = E2E_INPUT_DIM,
		.input_names = NULL,
		.num_input_names = 0,
		.feature_programs = feature_programs,
		.num_feature_programs = E2E_NUM_FEATURES,
		.means = means,
		.stddevs = stddevs,
		.beta_standardized = beta_standardized,
		.routines = NULL,
		.num_routines = 0,
		.execution_limit = 1024,
		.emit_spec_comment = false
	};

	StackPtxEmitSettings settings;
	if (!make_emit_settings(backend, &settings)) {
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
		return false;
	}

	*generated_text_out = generated_text;
	return true;
}

static
bool
write_model_file(
	const char* output_dir,
	StackPtxEmitLanguage backend,
	const char* file_name
) {
	char* generated_text = NULL;
	if (!generate_linear_regression_text(backend, &generated_text)) {
		return false;
	}

	char path[4096];
	bool ok = false;
	if (snprintf(path, sizeof(path), "%s/%s", output_dir, file_name) > 0) {
		ok = write_text_file(path, generated_text);
	}
	free(generated_text);
	return ok;
}

static
bool
write_c_runner(
	const char* output_dir,
	const uint32_t* expected_bits
) {
	char path[4096];
	FILE* file = NULL;
	if (!open_output_file(output_dir, "run_c.c", path, sizeof(path), &file)) {
		return false;
	}

	fprintf(file, "#include <assert.h>\n");
	fprintf(file, "#include <math.h>\n");
	fprintf(file, "#include <stdint.h>\n");
	fprintf(file, "#include <stdio.h>\n");
	fprintf(file, "#include \"linear_regression_model.c\"\n\n");
	fprintf(file, "static const float kInputs[%d][%d] = {\n", E2E_NUM_ROWS, E2E_INPUT_DIM);
	for (size_t row = 0; row < E2E_NUM_ROWS; row++) {
		fprintf(file, "\t{ ");
		for (size_t col = 0; col < E2E_INPUT_DIM; col++) {
			char literal[64];
			format_float_literal(dataset[row][col], "f", literal, sizeof(literal));
			fprintf(file, "%s%s", col == 0 ? "" : ", ", literal);
		}
		fprintf(file, " }%s\n", row + 1 == E2E_NUM_ROWS ? "" : ",");
	}
	fprintf(file, "};\n\n");
	fprintf(file, "static const uint32_t kExpected[%d] = {\n", E2E_NUM_ROWS);
	for (size_t row = 0; row < E2E_NUM_ROWS; row++) {
		fprintf(file, "\tUINT32_C(0x%08X)%s\n", (unsigned)expected_bits[row], row + 1 == E2E_NUM_ROWS ? "" : ",");
	}
	fprintf(file, "};\n\n");
	fprintf(file, "int main(void) {\n");
	fprintf(file, "\tfor (size_t row = 0; row < %d; row++) {\n", E2E_NUM_ROWS);
	fprintf(file, "\t\tunion { float f; uint32_t u; } bits;\n");
	fprintf(file, "\t\tbits.f = linear_regression_infer(kInputs[row]);\n");
	fprintf(file, "\t\tprintf(\"0x%%08X\\n\", (unsigned)bits.u);\n");
	fprintf(file, "\t\tassert(bits.u == kExpected[row]);\n");
	fprintf(file, "\t}\n");
	fprintf(file, "\treturn 0;\n");
	fprintf(file, "}\n");

	return fclose(file) == 0;
}

static
bool
write_rust_runner(
	const char* output_dir,
	const uint32_t* expected_bits
) {
	char path[4096];
	FILE* file = NULL;
	if (!open_output_file(output_dir, "run_rust.rs", path, sizeof(path), &file)) {
		return false;
	}

	fprintf(file, "include!(r#\"linear_regression_model.rs\"#);\n\n");
	fprintf(file, "const INPUTS: [[f32; %d]; %d] = [\n", E2E_INPUT_DIM, E2E_NUM_ROWS);
	for (size_t row = 0; row < E2E_NUM_ROWS; row++) {
		fprintf(file, "    [");
		for (size_t col = 0; col < E2E_INPUT_DIM; col++) {
			char literal[64];
			format_float_literal(dataset[row][col], "f32", literal, sizeof(literal));
			fprintf(file, "%s%s", col == 0 ? "" : ", ", literal);
		}
		fprintf(file, "]%s\n", row + 1 == E2E_NUM_ROWS ? "" : ",");
	}
	fprintf(file, "];\n\n");
	fprintf(file, "const EXPECTED: [u32; %d] = [\n", E2E_NUM_ROWS);
	for (size_t row = 0; row < E2E_NUM_ROWS; row++) {
		fprintf(file, "    0x%08Xu32%s\n", (unsigned)expected_bits[row], row + 1 == E2E_NUM_ROWS ? "" : ",");
	}
	fprintf(file, "];\n\n");
	fprintf(file, "fn main() {\n");
	fprintf(file, "    for row in 0..INPUTS.len() {\n");
	fprintf(file, "        let bits = linear_regression_infer(&INPUTS[row]).to_bits();\n");
	fprintf(file, "        println!(\"0x{:08X}\", bits);\n");
	fprintf(file, "        assert_eq!(bits, EXPECTED[row]);\n");
	fprintf(file, "    }\n");
	fprintf(file, "}\n");

	return fclose(file) == 0;
}

static
bool
write_cuda_runner(
	const char* output_dir,
	const uint32_t* expected_bits
) {
	char path[4096];
	FILE* file = NULL;
	if (!open_output_file(output_dir, "run_cuda.cu", path, sizeof(path), &file)) {
		return false;
	}

	fprintf(file, "#include <assert.h>\n");
	fprintf(file, "#include <cuda_runtime.h>\n");
	fprintf(file, "#include <stdint.h>\n");
	fprintf(file, "#include <stdio.h>\n");
	fprintf(file, "#include <stdlib.h>\n");
	fprintf(file, "#include \"linear_regression_model.cu\"\n\n");
	fprintf(file, "enum { ROWS = %d, COLS = %d };\n\n", E2E_NUM_ROWS, E2E_INPUT_DIM);
	fprintf(file, "static const float kInputs[ROWS][COLS] = {\n");
	for (size_t row = 0; row < E2E_NUM_ROWS; row++) {
		fprintf(file, "\t{ ");
		for (size_t col = 0; col < E2E_INPUT_DIM; col++) {
			char literal[64];
			format_float_literal(dataset[row][col], "f", literal, sizeof(literal));
			fprintf(file, "%s%s", col == 0 ? "" : ", ", literal);
		}
		fprintf(file, " }%s\n", row + 1 == E2E_NUM_ROWS ? "" : ",");
	}
	fprintf(file, "};\n\n");
	fprintf(file, "static const uint32_t kExpected[ROWS] = {\n");
	for (size_t row = 0; row < E2E_NUM_ROWS; row++) {
		fprintf(file, "\t0x%08Xu%s\n", (unsigned)expected_bits[row], row + 1 == E2E_NUM_ROWS ? "" : ",");
	}
	fprintf(file, "};\n\n");
	fprintf(file, "static void cuda_check(cudaError_t error, const char* message) {\n");
	fprintf(file, "\tif (error != cudaSuccess) {\n");
	fprintf(file, "\t\tfprintf(stderr, \"%%s: %%s\\n\", message, cudaGetErrorString(error));\n");
	fprintf(file, "\t\texit(1);\n");
	fprintf(file, "\t}\n");
	fprintf(file, "}\n\n");
	fprintf(file, "__global__ void run_kernel(const float* inputs, float* outputs) {\n");
	fprintf(file, "\tfor (int row = 0; row < ROWS; row++) {\n");
	fprintf(file, "\t\toutputs[row] = linear_regression_infer(inputs + row * COLS);\n");
	fprintf(file, "\t}\n");
	fprintf(file, "}\n\n");
	fprintf(file, "int main(void) {\n");
	fprintf(file, "\tfloat* d_inputs = NULL;\n");
	fprintf(file, "\tfloat* d_outputs = NULL;\n");
	fprintf(file, "\tfloat h_outputs[ROWS];\n");
	fprintf(file, "\tcuda_check(cudaMalloc((void**)&d_inputs, sizeof(kInputs)), \"cudaMalloc inputs failed\");\n");
	fprintf(file, "\tcuda_check(cudaMalloc((void**)&d_outputs, sizeof(h_outputs)), \"cudaMalloc outputs failed\");\n");
	fprintf(file, "\tcuda_check(cudaMemcpy(d_inputs, kInputs, sizeof(kInputs), cudaMemcpyHostToDevice), \"cudaMemcpy H2D failed\");\n");
	fprintf(file, "\trun_kernel<<<1, 1>>>(d_inputs, d_outputs);\n");
	fprintf(file, "\tcuda_check(cudaGetLastError(), \"kernel launch failed\");\n");
	fprintf(file, "\tcuda_check(cudaDeviceSynchronize(), \"cudaDeviceSynchronize failed\");\n");
	fprintf(file, "\tcuda_check(cudaMemcpy(h_outputs, d_outputs, sizeof(h_outputs), cudaMemcpyDeviceToHost), \"cudaMemcpy D2H failed\");\n");
	fprintf(file, "\tcuda_check(cudaFree(d_inputs), \"cudaFree inputs failed\");\n");
	fprintf(file, "\tcuda_check(cudaFree(d_outputs), \"cudaFree outputs failed\");\n");
	fprintf(file, "\tfor (int row = 0; row < ROWS; row++) {\n");
	fprintf(file, "\t\tunion { float f; uint32_t u; } bits;\n");
	fprintf(file, "\t\tbits.f = h_outputs[row];\n");
	fprintf(file, "\t\tprintf(\"0x%%08X\\n\", (unsigned)bits.u);\n");
	fprintf(file, "\t\tassert(bits.u == kExpected[row]);\n");
	fprintf(file, "\t}\n");
	fprintf(file, "\treturn 0;\n");
	fprintf(file, "}\n");

	return fclose(file) == 0;
}

static
bool
write_numpy_runner(
	const char* output_dir,
	const uint32_t* expected_bits
) {
	char path[4096];
	FILE* file = NULL;
	if (!open_output_file(output_dir, "run_numpy.py", path, sizeof(path), &file)) {
		return false;
	}

	fprintf(file, "import numpy as np\n");
	fprintf(file, "from linear_regression_model import linear_regression_infer\n\n");
	fprintf(file, "DATASET = np.array([\n");
	for (size_t row = 0; row < E2E_NUM_ROWS; row++) {
		fprintf(file, "    [");
		for (size_t col = 0; col < E2E_INPUT_DIM; col++) {
			char literal[64];
			format_float_literal(dataset[row][col], "", literal, sizeof(literal));
			fprintf(file, "%s%s", col == 0 ? "" : ", ", literal);
		}
		fprintf(file, "]%s\n", row + 1 == E2E_NUM_ROWS ? "" : ",");
	}
	fprintf(file, "], dtype=np.float32)\n\n");
	fprintf(file, "EXPECTED = np.array([\n");
	for (size_t row = 0; row < E2E_NUM_ROWS; row++) {
		fprintf(file, "    0x%08X%s\n", (unsigned)expected_bits[row], row + 1 == E2E_NUM_ROWS ? "" : ",");
	}
	fprintf(file, "], dtype=np.uint32)\n\n");
	fprintf(file, "outputs = np.asarray(linear_regression_infer(DATASET), dtype=np.float32)\n");
	fprintf(file, "bits = outputs.view(np.uint32)\n");
	fprintf(file, "for value in bits:\n");
	fprintf(file, "    print(f\"0x{int(value):08X}\")\n");
	fprintf(file, "assert bits.shape == EXPECTED.shape\n");
	fprintf(file, "assert np.array_equal(bits, EXPECTED)\n");

	return fclose(file) == 0;
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

	const char* output_dir = argv[1];
	uint32_t expected_bits[E2E_NUM_ROWS];

	if (!write_expected_file(output_dir, expected_bits)) {
		fprintf(stderr, "failed to write expected outputs\n");
		return 1;
	}

	if (!write_model_file(output_dir, STACK_PTX_EMIT_LANGUAGE_C, "linear_regression_model.c") ||
		!write_model_file(output_dir, STACK_PTX_EMIT_LANGUAGE_CUDA, "linear_regression_model.cu") ||
		!write_model_file(output_dir, STACK_PTX_EMIT_LANGUAGE_RUST, "linear_regression_model.rs") ||
		!write_model_file(output_dir, STACK_PTX_EMIT_LANGUAGE_NUMPY, "linear_regression_model.py")) {
		fprintf(stderr, "failed to write emitted model code\n");
		return 1;
	}

	if (!write_c_runner(output_dir, expected_bits) ||
		!write_rust_runner(output_dir, expected_bits) ||
		!write_cuda_runner(output_dir, expected_bits) ||
		!write_numpy_runner(output_dir, expected_bits)) {
		fprintf(stderr, "failed to write backend runners\n");
		return 1;
	}

	return 0;
}
