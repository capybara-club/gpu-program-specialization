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

#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#define PTX_INJECT_IMPLEMENTATION
#include <ptx_inject.h>

#define STACK_PTX_DEBUG
#define STACK_PTX_IMPLEMENTATION
#include <stack_ptx.h>

#define STACK_PTX_EMIT_IMPLEMENTATION
#include <stack_ptx_emit.h>

#include <stack_ptx_tools_descriptions.h>
#include <stack_ptx_default_info.h>

#include <check_result_helper.h>

#define STACK_PTX_LITERAL(x) ((StackPtxInstruction)x)

typedef enum {
	REGISTER_V_X,
	REGISTER_V_Y,
	REGISTER_V_Z,
	REGISTER_NUM_ENUMS
} Register;

typedef struct {
	uint32_t state;
} Rng;

typedef struct {
	StackPtxIdx instruction_idx;
	uint8_t arity;
} RandomOp;

typedef enum {
	EMIT_FLAG_PTX = 1u << 0,
	EMIT_FLAG_C = 1u << 1,
	EMIT_FLAG_RUST = 1u << 2,
	EMIT_FLAG_CUDA = 1u << 3,
	EMIT_FLAG_NUMPY = 1u << 4,
	EMIT_FLAG_LATEX = 1u << 5,
	EMIT_FLAG_MARKDOWN = 1u << 6
} EmitFlag;

typedef struct {
	uint32_t seed;
	size_t count;
	size_t steps;
	uint32_t emit_flags;
	const char* output_prefix;
	const char* ptx_kernel_path;
	const char* ptx_inject_name;
} CliOptions;

static const StackPtxRegister registers[] = {
	[REGISTER_V_X] = { .name = "v_x", .stack_idx = STACK_PTX_STACK_TYPE_F32 },
	[REGISTER_V_Y] = { .name = "v_y", .stack_idx = STACK_PTX_STACK_TYPE_F32 },
	[REGISTER_V_Z] = { .name = "v_z", .stack_idx = STACK_PTX_STACK_TYPE_F32 },
};

static const size_t requests[] = {
	REGISTER_V_Z
};

static const RandomOp unary_ops[] = {
	{ STACK_PTX_PTX_INSTRUCTION_ABS_FTZ_F32, 1 },
	{ STACK_PTX_PTX_INSTRUCTION_NEG_FTZ_F32, 1 },
};

static const RandomOp binary_ops[] = {
	{ STACK_PTX_PTX_INSTRUCTION_ADD_FTZ_F32, 2 },
	{ STACK_PTX_PTX_INSTRUCTION_SUB_FTZ_F32, 2 },
	{ STACK_PTX_PTX_INSTRUCTION_MUL_FTZ_F32, 2 },
};

static const RandomOp ternary_ops[] = {
	{ STACK_PTX_PTX_INSTRUCTION_FMA_RN_FTZ_F32, 3 },
};

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
	.document_title = NULL,
	.output_symbol = "y",
	.special_register_exprs = NULL,
	.num_special_register_exprs = 0,
	.emit_document = true,
	.use_input_indices = false
};

static const StackPtxEmitMarkdownSettings base_markdown_settings = {
	.document_title = NULL,
	.output_symbol = "y",
	.special_register_exprs = NULL,
	.num_special_register_exprs = 0,
	.use_input_indices = false,
	.factor_expression_min_bytes = 96
};

static
uint32_t
rng_next(
	Rng* rng
) {
	uint32_t x = rng->state;
	if (x == 0) {
		x = UINT32_C(0x6D2B79F5);
	}
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	rng->state = x;
	return x;
}

static
size_t
rng_bounded(
	Rng* rng,
	size_t bound
) {
	return bound == 0 ? 0 : (size_t)(rng_next(rng) % (uint32_t)bound);
}

static
float
rng_f32(
	Rng* rng
) {
	int32_t raw = (int32_t)(rng_next(rng) % 2001u) - 1000;
	return (float)raw / 100.0f;
}

static
bool
has_live_slot(
	const bool* live_slots,
	size_t num_slots
) {
	for (size_t i = 0; i < num_slots; i++) {
		if (live_slots[i]) {
			return true;
		}
	}
	return false;
}

static
size_t
random_live_slot(
	Rng* rng,
	const bool* live_slots,
	size_t num_slots
) {
	size_t live_indices[16];
	size_t live_count = 0;
	for (size_t i = 0; i < num_slots; i++) {
		if (live_slots[i]) {
			live_indices[live_count++] = i;
		}
	}
	ASSERT(live_count != 0);
	return live_indices[rng_bounded(rng, live_count)];
}

static
void
append_instruction(
	StackPtxInstruction* instructions,
	size_t max_instructions,
	size_t* num_instructions,
	StackPtxInstruction instruction
) {
	ASSERT(*num_instructions < max_instructions);
	instructions[(*num_instructions)++] = instruction;
}

static
void
generate_program(
	StackPtxInstruction* instructions,
	size_t max_instructions,
	size_t* num_instructions_out,
	uint32_t seed,
	size_t num_steps
) {
	enum {
		STORE_SLOT_COUNT = 4
	};

	enum Action {
		ACTION_PUSH_INPUT,
		ACTION_PUSH_CONSTANT,
		ACTION_UNARY,
		ACTION_BINARY,
		ACTION_TERNARY,
		ACTION_STORE,
		ACTION_LOAD,
		ACTION_DUP,
		ACTION_SWAP
	};

	Rng rng = {
		.state = seed
	};
	bool live_slots[STORE_SLOT_COUNT] = { false, false, false, false };
	size_t depth = 0;
	size_t num_instructions = 0;

	append_instruction(
		instructions,
		max_instructions,
		&num_instructions,
		STACK_PTX_LITERAL(stack_ptx_encode_input(REGISTER_V_X))
	);
	append_instruction(
		instructions,
		max_instructions,
		&num_instructions,
		STACK_PTX_LITERAL(stack_ptx_encode_input(REGISTER_V_Y))
	);
	depth = 2;

	for (size_t step = 0; step < num_steps; step++) {
		enum Action choices[16];
		size_t num_choices = 0;

		if (depth < 6) {
			choices[num_choices++] = ACTION_PUSH_INPUT;
			choices[num_choices++] = ACTION_PUSH_CONSTANT;
		}
		if (depth >= 1) {
			choices[num_choices++] = ACTION_UNARY;
			choices[num_choices++] = ACTION_DUP;
			choices[num_choices++] = ACTION_STORE;
		}
		if (depth >= 2) {
			choices[num_choices++] = ACTION_BINARY;
			choices[num_choices++] = ACTION_BINARY;
			choices[num_choices++] = ACTION_SWAP;
		}
		if (depth >= 3) {
			choices[num_choices++] = ACTION_TERNARY;
		}
		if (has_live_slot(live_slots, STORE_SLOT_COUNT)) {
			choices[num_choices++] = ACTION_LOAD;
		}

		enum Action action = choices[rng_bounded(&rng, num_choices)];

		switch (action) {
			case ACTION_PUSH_INPUT: {
				size_t register_idx = rng_bounded(&rng, 2) == 0 ? REGISTER_V_X : REGISTER_V_Y;
				append_instruction(
					instructions,
					max_instructions,
					&num_instructions,
					STACK_PTX_LITERAL(stack_ptx_encode_input(register_idx))
				);
				depth++;
			} break;
			case ACTION_PUSH_CONSTANT: {
				append_instruction(
					instructions,
					max_instructions,
					&num_instructions,
					STACK_PTX_LITERAL(stack_ptx_encode_constant_f32(rng_f32(&rng)))
				);
				depth++;
			} break;
			case ACTION_UNARY: {
				RandomOp op = unary_ops[rng_bounded(&rng, STACK_PTX_ARRAY_NUM_ELEMS(unary_ops))];
				append_instruction(
					instructions,
					max_instructions,
					&num_instructions,
					stack_ptx_ptx_instructions[op.instruction_idx]
				);
			} break;
			case ACTION_BINARY: {
				RandomOp op = binary_ops[rng_bounded(&rng, STACK_PTX_ARRAY_NUM_ELEMS(binary_ops))];
				append_instruction(
					instructions,
					max_instructions,
					&num_instructions,
					stack_ptx_ptx_instructions[op.instruction_idx]
				);
				depth--;
			} break;
			case ACTION_TERNARY: {
				RandomOp op = ternary_ops[rng_bounded(&rng, STACK_PTX_ARRAY_NUM_ELEMS(ternary_ops))];
				append_instruction(
					instructions,
					max_instructions,
					&num_instructions,
					stack_ptx_ptx_instructions[op.instruction_idx]
				);
				depth -= 2;
			} break;
			case ACTION_STORE: {
				size_t slot = rng_bounded(&rng, STORE_SLOT_COUNT);
				append_instruction(
					instructions,
					max_instructions,
					&num_instructions,
					STACK_PTX_LITERAL(stack_ptx_encode_store(STACK_PTX_STACK_TYPE_F32, slot))
				);
				live_slots[slot] = true;
				depth--;
			} break;
			case ACTION_LOAD: {
				size_t slot = random_live_slot(&rng, live_slots, STORE_SLOT_COUNT);
				append_instruction(
					instructions,
					max_instructions,
					&num_instructions,
					STACK_PTX_LITERAL(stack_ptx_encode_load(slot))
				);
				depth++;
			} break;
			case ACTION_DUP: {
				append_instruction(
					instructions,
					max_instructions,
					&num_instructions,
					STACK_PTX_LITERAL(stack_ptx_encode_meta_dup(STACK_PTX_STACK_TYPE_F32))
				);
				depth++;
			} break;
			case ACTION_SWAP: {
				append_instruction(
					instructions,
					max_instructions,
					&num_instructions,
					STACK_PTX_LITERAL(stack_ptx_encode_meta_swap(STACK_PTX_STACK_TYPE_F32))
				);
			} break;
		}
	}

	while (depth < 3) {
		append_instruction(
			instructions,
			max_instructions,
			&num_instructions,
			STACK_PTX_LITERAL(stack_ptx_encode_input(depth == 0 ? REGISTER_V_X : REGISTER_V_Y))
		);
		depth++;
	}

	{
		RandomOp final_op = ternary_ops[rng_bounded(&rng, STACK_PTX_ARRAY_NUM_ELEMS(ternary_ops))];
		append_instruction(
			instructions,
			max_instructions,
			&num_instructions,
			stack_ptx_ptx_instructions[final_op.instruction_idx]
		);
		depth -= 2;
	}

	append_instruction(
		instructions,
		max_instructions,
		&num_instructions,
		STACK_PTX_LITERAL(stack_ptx_encode_return)
	);

	*num_instructions_out = num_instructions;
}

static
uint32_t
program_seed_for_index(
	uint32_t seed,
	size_t index
) {
	return seed + (uint32_t)(UINT32_C(0x9E3779B9) * (uint32_t)index);
}

static
char*
compile_backend_stub(
	const StackPtxEmitSettings* settings_ref,
	const StackPtxInstruction* instructions,
	void* workspace,
	size_t workspace_size
) {
	size_t required = 0;
	StackPtxResult stack_result = STACK_PTX_SUCCESS;
	StackPtxEmitResult emit_result = stack_ptx_emit_compile(
		&compiler_info,
		&stack_ptx_stack_info,
		settings_ref,
		instructions,
		registers,
		STACK_PTX_ARRAY_NUM_ELEMS(registers),
		NULL,
		0,
		requests,
		STACK_PTX_ARRAY_NUM_ELEMS(requests),
		1024,
		workspace,
		workspace_size,
		NULL,
		0,
		&required,
		&stack_result
	);
	ASSERT(emit_result == STACK_PTX_EMIT_SUCCESS);
	ASSERT(stack_result == STACK_PTX_SUCCESS);

	char* buffer = (char*)malloc(required + 1);
	ASSERT(buffer != NULL);

	emit_result = stack_ptx_emit_compile(
		&compiler_info,
		&stack_ptx_stack_info,
		settings_ref,
		instructions,
		registers,
		STACK_PTX_ARRAY_NUM_ELEMS(registers),
		NULL,
		0,
		requests,
		STACK_PTX_ARRAY_NUM_ELEMS(requests),
		1024,
		workspace,
		workspace_size,
		buffer,
		required + 1,
		&required,
		&stack_result
	);
	ASSERT(emit_result == STACK_PTX_EMIT_SUCCESS);
	ASSERT(stack_result == STACK_PTX_SUCCESS);

	return buffer;
}

static
char*
compile_ptx_stub(
	const StackPtxInstruction* instructions,
	const StackPtxRegister* ptx_registers,
	size_t num_ptx_registers,
	void* workspace,
	size_t workspace_size
) {
	StackPtxEmitSettings settings = {
		.language = STACK_PTX_EMIT_LANGUAGE_PTX,
		.vector = { .kind = STACK_PTX_EMIT_VECTOR_KIND_NONE, .bits = 0 },
		.as.ptx = {
			.registers = ptx_registers,
			.num_registers = num_ptx_registers
		}
	};
	return compile_backend_stub(
		&settings,
		instructions,
		workspace,
		workspace_size
	);
}

static
char*
compile_c_like_stub(
	const StackPtxEmitLanguage backend,
	const StackPtxInstruction* instructions,
	void* workspace,
	size_t workspace_size,
	const char* function_name
) {
	StackPtxEmitSettings settings = {
		.language = backend,
		.vector = { .kind = STACK_PTX_EMIT_VECTOR_KIND_NONE, .bits = 0 }
	};
	switch (backend) {
		case STACK_PTX_EMIT_LANGUAGE_C:
			settings.as.c = base_c_settings;
			settings.as.c.function_name = function_name;
			break;
		case STACK_PTX_EMIT_LANGUAGE_CUDA:
			settings.as.cuda = base_cuda_settings;
			settings.as.cuda.function_name = function_name;
			break;
		case STACK_PTX_EMIT_LANGUAGE_RUST:
			settings.as.rust = base_rust_settings;
			settings.as.rust.function_name = function_name;
			break;
		case STACK_PTX_EMIT_LANGUAGE_NUMPY:
			settings.as.numpy = base_numpy_settings;
			settings.as.numpy.function_name = function_name;
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
			ASSERT(false);
			break;
	}
	return compile_backend_stub(
		&settings,
		instructions,
		workspace,
		workspace_size
	);
}

static
void
print_usage(
	const char* argv0
) {
	printf("Usage: %s [-o PREFIX] [-seed N] [-N COUNT] [--steps N] [-ptx] [-c] [-rust] [-cuda] [-numpy] [-latex] [-markdown]\n", argv0);
	printf("  -o, --output-prefix PATH  Output file prefix. If omitted, write emitted stubs to stdout.\n");
	printf("  -seed, --seed N           Base seed. Default: 1.\n");
	printf("  -N, --count N             Number of stubs to emit. Default: 1.\n");
	printf("  --steps N                 Random instruction steps before final return. Default: 12.\n");
	printf("  --ptx-kernel PATH         Optional PTX template used to resolve inject register names.\n");
	printf("  --ptx-inject-name NAME    PTX inject site name. Default: func.\n");
	printf("  -ptx                      Emit raw Stack PTX stub text. Default if -o is set and no language flags are set.\n");
	printf("  -c                        Emit scalar C function stubs.\n");
	printf("  -rust                     Emit Rust function stubs.\n");
	printf("  -cuda                     Emit native CUDA device function stubs.\n");
	printf("  -numpy                    Emit NumPy/Python function stubs.\n");
	printf("  -latex                    Emit LaTeX math stubs.\n");
	printf("  -markdown                 Emit Markdown math stubs.\n");
	printf("                            If -o is omitted and no language flags are set, all current backends are printed.\n");
}

static
char*
read_text_file(
	const char* path
) {
	FILE* file = fopen(path, "rb");
	if (file == NULL) {
		return NULL;
	}

	ASSERT(fseek(file, 0, SEEK_END) == 0);
	long file_size = ftell(file);
	ASSERT(file_size >= 0);
	ASSERT(fseek(file, 0, SEEK_SET) == 0);

	char* buffer = (char*)malloc((size_t)file_size + 1);
	ASSERT(buffer != NULL);
	size_t bytes_read = fread(buffer, 1, (size_t)file_size, file);
	ASSERT(bytes_read == (size_t)file_size);
	ASSERT(fclose(file) == 0);
	buffer[bytes_read] = '\0';
	return buffer;
}

static
bool
ensure_parent_directory(
	const char* path
) {
	char buffer[4096];
	size_t path_len = strlen(path);
	if (path_len >= sizeof(buffer)) {
		return false;
	}
	memcpy(buffer, path, path_len + 1);

	char* slash = strrchr(buffer, '/');
	if (slash == NULL) {
		return true;
	}
	*slash = '\0';
	if (buffer[0] == '\0') {
		return true;
	}

	for (char* p = buffer + 1; *p != '\0'; p++) {
		if (*p == '/') {
			*p = '\0';
			if (mkdir(buffer, 0777) != 0 && errno != EEXIST) {
				return false;
			}
			*p = '/';
		}
	}

	return mkdir(buffer, 0777) == 0 || errno == EEXIST;
}

static
bool
write_text_file(
	const char* path,
	const char* text
) {
	if (!ensure_parent_directory(path)) {
		return false;
	}
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
append_manifest_line(
	FILE* manifest,
	size_t index,
	uint32_t seed,
	const char* suffix,
	const char* path
) {
	return fprintf(
		manifest,
		"%03zu seed=%u kind=%s path=%s\n",
		index,
		seed,
		suffix,
		path
	) > 0;
}

static
void
print_terminal_stub(
	size_t index,
	uint32_t seed,
	const char* kind,
	const char* text
) {
	printf("=== stub_%03zu %s seed=%u ===\n", index, kind, seed);
	printf("%s", text);
	if (text[0] != '\0' && text[strlen(text) - 1] != '\n') {
		printf("\n");
	}
	printf("\n");
}

static
bool
parse_cli_options(
	int argc,
	char** argv,
	CliOptions* options_out
) {
	CliOptions options = {
		.seed = 1u,
		.count = 1,
		.steps = 12,
		.emit_flags = 0,
		.output_prefix = NULL,
		.ptx_kernel_path = NULL,
		.ptx_inject_name = "func"
	};

	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-o") == 0 ||
			strcmp(argv[i], "--output-prefix") == 0 ||
			strcmp(argv[i], "--prefix") == 0 ||
			strcmp(argv[i], "-path") == 0) {
			ASSERT(i + 1 < argc);
			options.output_prefix = argv[++i];
		} else if (strcmp(argv[i], "-seed") == 0 || strcmp(argv[i], "--seed") == 0) {
			ASSERT(i + 1 < argc);
			options.seed = (uint32_t)strtoul(argv[++i], NULL, 10);
		} else if (strcmp(argv[i], "-N") == 0 || strcmp(argv[i], "--count") == 0) {
			ASSERT(i + 1 < argc);
			options.count = (size_t)strtoull(argv[++i], NULL, 10);
		} else if (strcmp(argv[i], "--steps") == 0) {
			ASSERT(i + 1 < argc);
			options.steps = (size_t)strtoull(argv[++i], NULL, 10);
		} else if (strcmp(argv[i], "--ptx-kernel") == 0) {
			ASSERT(i + 1 < argc);
			options.ptx_kernel_path = argv[++i];
		} else if (strcmp(argv[i], "--ptx-inject-name") == 0) {
			ASSERT(i + 1 < argc);
			options.ptx_inject_name = argv[++i];
		} else if (strcmp(argv[i], "-ptx") == 0) {
			options.emit_flags |= EMIT_FLAG_PTX;
		} else if (strcmp(argv[i], "-c") == 0) {
			options.emit_flags |= EMIT_FLAG_C;
		} else if (strcmp(argv[i], "-rust") == 0) {
			options.emit_flags |= EMIT_FLAG_RUST;
		} else if (strcmp(argv[i], "-cuda") == 0) {
			options.emit_flags |= EMIT_FLAG_CUDA;
		} else if (strcmp(argv[i], "-numpy") == 0) {
			options.emit_flags |= EMIT_FLAG_NUMPY;
		} else if (strcmp(argv[i], "-latex") == 0) {
			options.emit_flags |= EMIT_FLAG_LATEX;
		} else if (strcmp(argv[i], "-markdown") == 0) {
			options.emit_flags |= EMIT_FLAG_MARKDOWN;
		} else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
			print_usage(argv[0]);
			exit(0);
		} else {
			fprintf(stderr, "Unknown option: %s\n", argv[i]);
			return false;
		}
	}

	if (options.count == 0) {
		return false;
	}
	if (options.emit_flags == 0) {
		if (options.output_prefix != NULL && options.output_prefix[0] != '\0') {
			options.emit_flags = EMIT_FLAG_PTX;
		} else {
			options.emit_flags =
				EMIT_FLAG_PTX |
				EMIT_FLAG_C |
				EMIT_FLAG_RUST |
				EMIT_FLAG_CUDA |
				EMIT_FLAG_NUMPY |
				EMIT_FLAG_LATEX |
				EMIT_FLAG_MARKDOWN;
		}
	}

	*options_out = options;
	return true;
}

int
main(
	int argc,
	char** argv
) {
	CliOptions options;
	if (!parse_cli_options(argc, argv, &options)) {
		print_usage(argv[0]);
		return 1;
	}

	size_t workspace_size = 0;
	stackPtxCheck(
		stack_ptx_compile_workspace_size(
			&compiler_info,
			&stack_ptx_stack_info,
			&workspace_size
		)
	);

	void* workspace = malloc(workspace_size);
	ASSERT(workspace != NULL);

	size_t max_instructions = options.steps + 8;
	StackPtxInstruction* instructions =
		(StackPtxInstruction*)malloc(max_instructions * sizeof(StackPtxInstruction));
	ASSERT(instructions != NULL);

	FILE* manifest = NULL;
	if (options.output_prefix != NULL && options.output_prefix[0] != '\0') {
		char manifest_path[4096];
		int manifest_bytes = snprintf(
			manifest_path,
			sizeof(manifest_path),
			"%s_manifest.txt",
			options.output_prefix
		);
		ASSERT(manifest_bytes > 0 && (size_t)manifest_bytes < sizeof(manifest_path));
		ASSERT(ensure_parent_directory(manifest_path));
		manifest = fopen(manifest_path, "w");
		ASSERT(manifest != NULL);
	}

	StackPtxRegister ptx_registers[STACK_PTX_ARRAY_NUM_ELEMS(registers)];
	memcpy(ptx_registers, registers, sizeof(ptx_registers));

	PtxInjectHandle ptx_inject = NULL;
	char* ptx_kernel_text = NULL;
	if (options.ptx_kernel_path != NULL) {
		ptx_kernel_text = read_text_file(options.ptx_kernel_path);
		ASSERT(ptx_kernel_text != NULL);

		ptxInjectCheck( ptx_inject_create(&ptx_inject, ptx_kernel_text) );

		size_t inject_idx = 0;
		ptxInjectCheck(
			ptx_inject_inject_info_by_name(
				ptx_inject,
				options.ptx_inject_name,
				&inject_idx,
				NULL,
				NULL
			)
		);

		ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, inject_idx, "v_x", NULL, &ptx_registers[REGISTER_V_X].name, NULL, NULL, NULL) );
		ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, inject_idx, "v_y", NULL, &ptx_registers[REGISTER_V_Y].name, NULL, NULL, NULL) );
		ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, inject_idx, "v_z", NULL, &ptx_registers[REGISTER_V_Z].name, NULL, NULL, NULL) );
	}

	for (size_t index = 0; index < options.count; index++) {
		uint32_t program_seed = program_seed_for_index(options.seed, index);
		size_t num_instructions = 0;
		generate_program(
			instructions,
			max_instructions,
			&num_instructions,
			program_seed,
			options.steps
		);
		(void)num_instructions;

		char function_name[128];
		int function_name_bytes = snprintf(
			function_name,
			sizeof(function_name),
			"stack_ptx_stub_%03zu",
			index
		);
		ASSERT(function_name_bytes > 0 && (size_t)function_name_bytes < sizeof(function_name));

		if ((options.emit_flags & EMIT_FLAG_PTX) != 0) {
			char* ptx_stub = compile_ptx_stub(
				instructions,
				ptx_registers,
				STACK_PTX_ARRAY_NUM_ELEMS(ptx_registers),
				workspace,
				workspace_size
			);
			if (manifest != NULL) {
				char path[4096];
				ASSERT(snprintf(path, sizeof(path), "%s_%03zu.ptx", options.output_prefix, index) > 0);
				ASSERT(write_text_file(path, ptx_stub));
				ASSERT(append_manifest_line(manifest, index, program_seed, "ptx", path));
				printf("%s\n", path);
			} else {
				print_terminal_stub(index, program_seed, "ptx", ptx_stub);
			}
			free(ptx_stub);
		}

		char* c_stub = NULL;
		if ((options.emit_flags & EMIT_FLAG_C) != 0) {
			c_stub = compile_c_like_stub(
				STACK_PTX_EMIT_LANGUAGE_C,
				instructions,
				workspace,
				workspace_size,
				function_name
			);
		}

		if ((options.emit_flags & EMIT_FLAG_C) != 0) {
			if (manifest != NULL) {
				char path[4096];
				ASSERT(snprintf(path, sizeof(path), "%s_%03zu.c", options.output_prefix, index) > 0);
				ASSERT(write_text_file(path, c_stub));
				ASSERT(append_manifest_line(manifest, index, program_seed, "c", path));
				printf("%s\n", path);
			} else {
				print_terminal_stub(index, program_seed, "c", c_stub);
			}
		}

		if ((options.emit_flags & EMIT_FLAG_RUST) != 0) {
			char* rust_stub = compile_c_like_stub(
				STACK_PTX_EMIT_LANGUAGE_RUST,
				instructions,
				workspace,
				workspace_size,
				function_name
			);
			if (manifest != NULL) {
				char path[4096];
				ASSERT(snprintf(path, sizeof(path), "%s_%03zu.rs", options.output_prefix, index) > 0);
				ASSERT(write_text_file(path, rust_stub));
				ASSERT(append_manifest_line(manifest, index, program_seed, "rust", path));
				printf("%s\n", path);
			} else {
				print_terminal_stub(index, program_seed, "rust", rust_stub);
			}
			free(rust_stub);
		}

		if ((options.emit_flags & EMIT_FLAG_NUMPY) != 0) {
			char* numpy_stub = compile_c_like_stub(
				STACK_PTX_EMIT_LANGUAGE_NUMPY,
				instructions,
				workspace,
				workspace_size,
				function_name
			);
			if (manifest != NULL) {
				char path[4096];
				ASSERT(snprintf(path, sizeof(path), "%s_%03zu.py", options.output_prefix, index) > 0);
				ASSERT(write_text_file(path, numpy_stub));
				ASSERT(append_manifest_line(manifest, index, program_seed, "numpy", path));
				printf("%s\n", path);
			} else {
				print_terminal_stub(index, program_seed, "numpy", numpy_stub);
			}
			free(numpy_stub);
		}

		if ((options.emit_flags & EMIT_FLAG_LATEX) != 0) {
			char* latex_stub = compile_c_like_stub(
				STACK_PTX_EMIT_LANGUAGE_LATEX,
				instructions,
				workspace,
				workspace_size,
				function_name
			);
			if (manifest != NULL) {
				char path[4096];
				ASSERT(snprintf(path, sizeof(path), "%s_%03zu.tex", options.output_prefix, index) > 0);
				ASSERT(write_text_file(path, latex_stub));
				ASSERT(append_manifest_line(manifest, index, program_seed, "latex", path));
				printf("%s\n", path);
			} else {
				print_terminal_stub(index, program_seed, "latex", latex_stub);
			}
			free(latex_stub);
		}

		if ((options.emit_flags & EMIT_FLAG_MARKDOWN) != 0) {
			char* markdown_stub = compile_c_like_stub(
				STACK_PTX_EMIT_LANGUAGE_MARKDOWN,
				instructions,
				workspace,
				workspace_size,
				function_name
			);
			if (manifest != NULL) {
				char path[4096];
				ASSERT(snprintf(path, sizeof(path), "%s_%03zu.md", options.output_prefix, index) > 0);
				ASSERT(write_text_file(path, markdown_stub));
				ASSERT(append_manifest_line(manifest, index, program_seed, "markdown", path));
				printf("%s\n", path);
			} else {
				print_terminal_stub(index, program_seed, "markdown", markdown_stub);
			}
			free(markdown_stub);
		}

		free(c_stub);

		if ((options.emit_flags & EMIT_FLAG_CUDA) != 0) {
			char* cuda_stub = compile_c_like_stub(
				STACK_PTX_EMIT_LANGUAGE_CUDA,
				instructions,
				workspace,
				workspace_size,
				function_name
			);
			if (manifest != NULL) {
				char path[4096];
				ASSERT(snprintf(path, sizeof(path), "%s_%03zu.cu", options.output_prefix, index) > 0);
				ASSERT(write_text_file(path, cuda_stub));
				ASSERT(append_manifest_line(manifest, index, program_seed, "cuda", path));
				printf("%s\n", path);
			} else {
				print_terminal_stub(index, program_seed, "cuda", cuda_stub);
			}
			free(cuda_stub);
		}
	}

	if (manifest != NULL) {
		ASSERT(fclose(manifest) == 0);
	}
	if (ptx_inject != NULL) {
		ptxInjectCheck( ptx_inject_destroy(ptx_inject) );
	}
	free(ptx_kernel_text);
	free(instructions);
	free(workspace);

	return 0;
}
