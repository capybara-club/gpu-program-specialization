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

typedef enum {
	BACKEND_FLAG_C = 1u << 0,
	BACKEND_FLAG_CUDA = 1u << 1,
	BACKEND_FLAG_RUST = 1u << 2,
	BACKEND_FLAG_NUMPY = 1u << 3,
	BACKEND_FLAG_LATEX = 1u << 4,
	BACKEND_FLAG_MARKDOWN = 1u << 5
} BackendFlag;

typedef struct {
	uint32_t state;
} Rng;

typedef struct {
	uint32_t operand;
	uint8_t arity;
	uint8_t kind;
} RandomOp;

typedef struct {
	uint32_t seed;
	size_t input_dim;
	size_t num_features;
	size_t steps;
	float eps;
	uint64_t allowed_function_mask;
	uint32_t backend_flags;
	bool emit_spec_comment;
	const char* function_name;
	const char* input_names_csv;
	size_t markdown_factor_expression_min_bytes;
} CliOptions;

typedef struct {
	char* name;
	StackPtxInstruction* instructions;
	size_t num_instructions;
} GeneratedFeature;

typedef struct {
	const char* name;
	RandomOp op;
} RandomFunctionSpec;

typedef enum {
	RANDOM_OP_KIND_PTX = 0,
	RANDOM_OP_KIND_ROUTINE
} RandomOpKind;

typedef enum {
	PROTECTED_ROUTINE_DIV = 0,
	PROTECTED_ROUTINE_RCP,
	PROTECTED_ROUTINE_SQRT,
	PROTECTED_ROUTINE_RSQRT,
	PROTECTED_ROUTINE_LG2,
	PROTECTED_ROUTINE_EX2,
	PROTECTED_ROUTINE_NUM_ENUMS
} ProtectedRoutineIdx;

enum {
	PROTECTED_SLOT_INPUT = 0,
	PROTECTED_SLOT_SECOND = 1
};

static const char* protected_routine_names[PROTECTED_ROUTINE_NUM_ENUMS] = {
	[PROTECTED_ROUTINE_DIV] = "protected_div",
	[PROTECTED_ROUTINE_RCP] = "protected_rcp",
	[PROTECTED_ROUTINE_SQRT] = "protected_sqrt",
	[PROTECTED_ROUTINE_RSQRT] = "protected_rsqrt",
	[PROTECTED_ROUTINE_LG2] = "protected_log2",
	[PROTECTED_ROUTINE_EX2] = "protected_exp2",
};

typedef enum {
	RANDOM_FUNCTION_ABS = 0,
	RANDOM_FUNCTION_NEG,
	RANDOM_FUNCTION_RCP,
	RANDOM_FUNCTION_SQRT,
	RANDOM_FUNCTION_RSQRT,
	RANDOM_FUNCTION_SIN,
	RANDOM_FUNCTION_COS,
	RANDOM_FUNCTION_LOG2,
	RANDOM_FUNCTION_EXP2,
	RANDOM_FUNCTION_TANH,
	RANDOM_FUNCTION_ADD,
	RANDOM_FUNCTION_SUB,
	RANDOM_FUNCTION_MUL,
	RANDOM_FUNCTION_DIV,
	RANDOM_FUNCTION_MIN,
	RANDOM_FUNCTION_MAX,
	RANDOM_FUNCTION_COPYSIGN,
	RANDOM_FUNCTION_FMA,
	RANDOM_FUNCTION_NUM_ENUMS
} RandomFunctionId;

static const RandomFunctionSpec random_function_specs[RANDOM_FUNCTION_NUM_ENUMS] = {
	[RANDOM_FUNCTION_ABS] = {
		.name = "abs",
		.op = { STACK_PTX_PTX_INSTRUCTION_ABS_FTZ_F32, 1, RANDOM_OP_KIND_PTX }
	},
	[RANDOM_FUNCTION_NEG] = {
		.name = "neg",
		.op = { STACK_PTX_PTX_INSTRUCTION_NEG_FTZ_F32, 1, RANDOM_OP_KIND_PTX }
	},
	[RANDOM_FUNCTION_RCP] = {
		.name = "rcp",
		.op = { PROTECTED_ROUTINE_RCP, 1, RANDOM_OP_KIND_ROUTINE }
	},
	[RANDOM_FUNCTION_SQRT] = {
		.name = "sqrt",
		.op = { PROTECTED_ROUTINE_SQRT, 1, RANDOM_OP_KIND_ROUTINE }
	},
	[RANDOM_FUNCTION_RSQRT] = {
		.name = "rsqrt",
		.op = { PROTECTED_ROUTINE_RSQRT, 1, RANDOM_OP_KIND_ROUTINE }
	},
	[RANDOM_FUNCTION_SIN] = {
		.name = "sin",
		.op = { STACK_PTX_PTX_INSTRUCTION_SIN_APPROX_FTZ_F32, 1, RANDOM_OP_KIND_PTX }
	},
	[RANDOM_FUNCTION_COS] = {
		.name = "cos",
		.op = { STACK_PTX_PTX_INSTRUCTION_COS_APPROX_FTZ_F32, 1, RANDOM_OP_KIND_PTX }
	},
	[RANDOM_FUNCTION_LOG2] = {
		.name = "log2",
		.op = { PROTECTED_ROUTINE_LG2, 1, RANDOM_OP_KIND_ROUTINE }
	},
	[RANDOM_FUNCTION_EXP2] = {
		.name = "exp2",
		.op = { PROTECTED_ROUTINE_EX2, 1, RANDOM_OP_KIND_ROUTINE }
	},
	[RANDOM_FUNCTION_TANH] = {
		.name = "tanh",
		.op = { STACK_PTX_PTX_INSTRUCTION_TANH_APPROX_F32, 1, RANDOM_OP_KIND_PTX }
	},
	[RANDOM_FUNCTION_ADD] = {
		.name = "add",
		.op = { STACK_PTX_PTX_INSTRUCTION_ADD_FTZ_F32, 2, RANDOM_OP_KIND_PTX }
	},
	[RANDOM_FUNCTION_SUB] = {
		.name = "sub",
		.op = { STACK_PTX_PTX_INSTRUCTION_SUB_FTZ_F32, 2, RANDOM_OP_KIND_PTX }
	},
	[RANDOM_FUNCTION_MUL] = {
		.name = "mul",
		.op = { STACK_PTX_PTX_INSTRUCTION_MUL_FTZ_F32, 2, RANDOM_OP_KIND_PTX }
	},
	[RANDOM_FUNCTION_DIV] = {
		.name = "div",
		.op = { PROTECTED_ROUTINE_DIV, 2, RANDOM_OP_KIND_ROUTINE }
	},
	[RANDOM_FUNCTION_MIN] = {
		.name = "min",
		.op = { STACK_PTX_PTX_INSTRUCTION_MIN_FTZ_F32, 2, RANDOM_OP_KIND_PTX }
	},
	[RANDOM_FUNCTION_MAX] = {
		.name = "max",
		.op = { STACK_PTX_PTX_INSTRUCTION_MAX_FTZ_F32, 2, RANDOM_OP_KIND_PTX }
	},
	[RANDOM_FUNCTION_COPYSIGN] = {
		.name = "copysign",
		.op = { STACK_PTX_PTX_INSTRUCTION_COPYSIGN_F32, 2, RANDOM_OP_KIND_PTX }
	},
	[RANDOM_FUNCTION_FMA] = {
		.name = "fma",
		.op = { STACK_PTX_PTX_INSTRUCTION_FMA_RN_FTZ_F32, 3, RANDOM_OP_KIND_PTX }
	},
};

static const StackPtxEmitInstructionInfo c_instruction_info[STACK_PTX_PTX_INSTRUCTION_NUM_ENUMS] = {
	[STACK_PTX_PTX_INSTRUCTION_ABS_FTZ_F32] = STACK_PTX_EMIT_OP_FUNC("fabsf"),
	[STACK_PTX_PTX_INSTRUCTION_NEG_FTZ_F32] = STACK_PTX_EMIT_OP_PREFIX("-"),
	[STACK_PTX_PTX_INSTRUCTION_ADD_FTZ_F32] = STACK_PTX_EMIT_OP_INFIX("+"),
	[STACK_PTX_PTX_INSTRUCTION_SUB_FTZ_F32] = STACK_PTX_EMIT_OP_INFIX("-"),
	[STACK_PTX_PTX_INSTRUCTION_MUL_FTZ_F32] = STACK_PTX_EMIT_OP_INFIX("*"),
	[STACK_PTX_PTX_INSTRUCTION_DIV_APPROX_FTZ_F32] = STACK_PTX_EMIT_OP_INFIX("/"),
	[STACK_PTX_PTX_INSTRUCTION_MIN_FTZ_F32] = STACK_PTX_EMIT_OP_FUNC("fminf"),
	[STACK_PTX_PTX_INSTRUCTION_MAX_FTZ_F32] = STACK_PTX_EMIT_OP_FUNC("fmaxf"),
	[STACK_PTX_PTX_INSTRUCTION_COPYSIGN_F32] = STACK_PTX_EMIT_OP_FUNC("copysignf"),
	[STACK_PTX_PTX_INSTRUCTION_FMA_RN_FTZ_F32] = STACK_PTX_EMIT_OP_FUNC("fmaf"),
	[STACK_PTX_PTX_INSTRUCTION_RCP_APPROX_FTZ_F32] = STACK_PTX_EMIT_OP_RECIPROCAL,
	[STACK_PTX_PTX_INSTRUCTION_SQRT_APPROX_FTZ_F32] = STACK_PTX_EMIT_OP_FUNC("sqrtf"),
	[STACK_PTX_PTX_INSTRUCTION_RSQRT_APPROX_FTZ_F32] = STACK_PTX_EMIT_OP_RSQRT,
	[STACK_PTX_PTX_INSTRUCTION_SIN_APPROX_FTZ_F32] = STACK_PTX_EMIT_OP_FUNC("sinf"),
	[STACK_PTX_PTX_INSTRUCTION_COS_APPROX_FTZ_F32] = STACK_PTX_EMIT_OP_FUNC("cosf"),
	[STACK_PTX_PTX_INSTRUCTION_LG2_APPROX_FTZ_F32] = STACK_PTX_EMIT_OP_FUNC("log2f"),
	[STACK_PTX_PTX_INSTRUCTION_EX2_APPROX_FTZ_F32] = STACK_PTX_EMIT_OP_FUNC("exp2f"),
	[STACK_PTX_PTX_INSTRUCTION_TANH_APPROX_F32] = STACK_PTX_EMIT_OP_FUNC("tanhf"),
};

static const StackPtxEmitInstructionInfo rust_instruction_info[STACK_PTX_PTX_INSTRUCTION_NUM_ENUMS] = {
	[STACK_PTX_PTX_INSTRUCTION_ABS_FTZ_F32] = STACK_PTX_EMIT_OP_FUNC("f32::abs"),
	[STACK_PTX_PTX_INSTRUCTION_NEG_FTZ_F32] = STACK_PTX_EMIT_OP_PREFIX("-"),
	[STACK_PTX_PTX_INSTRUCTION_ADD_FTZ_F32] = STACK_PTX_EMIT_OP_INFIX("+"),
	[STACK_PTX_PTX_INSTRUCTION_SUB_FTZ_F32] = STACK_PTX_EMIT_OP_INFIX("-"),
	[STACK_PTX_PTX_INSTRUCTION_MUL_FTZ_F32] = STACK_PTX_EMIT_OP_INFIX("*"),
	[STACK_PTX_PTX_INSTRUCTION_DIV_APPROX_FTZ_F32] = STACK_PTX_EMIT_OP_INFIX("/"),
	[STACK_PTX_PTX_INSTRUCTION_MIN_FTZ_F32] = STACK_PTX_EMIT_OP_FUNC("f32::min"),
	[STACK_PTX_PTX_INSTRUCTION_MAX_FTZ_F32] = STACK_PTX_EMIT_OP_FUNC("f32::max"),
	[STACK_PTX_PTX_INSTRUCTION_COPYSIGN_F32] = STACK_PTX_EMIT_OP_FUNC("f32::copysign"),
	[STACK_PTX_PTX_INSTRUCTION_FMA_RN_FTZ_F32] = STACK_PTX_EMIT_OP_FUNC("f32::mul_add"),
	[STACK_PTX_PTX_INSTRUCTION_RCP_APPROX_FTZ_F32] = STACK_PTX_EMIT_OP_RECIPROCAL,
	[STACK_PTX_PTX_INSTRUCTION_SQRT_APPROX_FTZ_F32] = STACK_PTX_EMIT_OP_FUNC("f32::sqrt"),
	[STACK_PTX_PTX_INSTRUCTION_RSQRT_APPROX_FTZ_F32] = STACK_PTX_EMIT_OP_RSQRT,
	[STACK_PTX_PTX_INSTRUCTION_SIN_APPROX_FTZ_F32] = STACK_PTX_EMIT_OP_FUNC("f32::sin"),
	[STACK_PTX_PTX_INSTRUCTION_COS_APPROX_FTZ_F32] = STACK_PTX_EMIT_OP_FUNC("f32::cos"),
	[STACK_PTX_PTX_INSTRUCTION_LG2_APPROX_FTZ_F32] = STACK_PTX_EMIT_OP_FUNC("f32::log2"),
	[STACK_PTX_PTX_INSTRUCTION_EX2_APPROX_FTZ_F32] = STACK_PTX_EMIT_OP_FUNC("f32::exp2"),
	[STACK_PTX_PTX_INSTRUCTION_TANH_APPROX_F32] = STACK_PTX_EMIT_OP_FUNC("f32::tanh"),
};

static const StackPtxEmitInstructionInfo numpy_instruction_info[STACK_PTX_PTX_INSTRUCTION_NUM_ENUMS] = {
	[STACK_PTX_PTX_INSTRUCTION_ABS_FTZ_F32] = STACK_PTX_EMIT_OP_FUNC("np.abs"),
	[STACK_PTX_PTX_INSTRUCTION_NEG_FTZ_F32] = STACK_PTX_EMIT_OP_PREFIX("-"),
	[STACK_PTX_PTX_INSTRUCTION_ADD_FTZ_F32] = STACK_PTX_EMIT_OP_INFIX("+"),
	[STACK_PTX_PTX_INSTRUCTION_SUB_FTZ_F32] = STACK_PTX_EMIT_OP_INFIX("-"),
	[STACK_PTX_PTX_INSTRUCTION_MUL_FTZ_F32] = STACK_PTX_EMIT_OP_INFIX("*"),
	[STACK_PTX_PTX_INSTRUCTION_DIV_APPROX_FTZ_F32] = STACK_PTX_EMIT_OP_INFIX("/"),
	[STACK_PTX_PTX_INSTRUCTION_MIN_FTZ_F32] = STACK_PTX_EMIT_OP_FUNC("np.minimum"),
	[STACK_PTX_PTX_INSTRUCTION_MAX_FTZ_F32] = STACK_PTX_EMIT_OP_FUNC("np.maximum"),
	[STACK_PTX_PTX_INSTRUCTION_COPYSIGN_F32] = STACK_PTX_EMIT_OP_FUNC("np.copysign"),
	[STACK_PTX_PTX_INSTRUCTION_FMA_RN_FTZ_F32] = STACK_PTX_EMIT_OP_MULADD,
	[STACK_PTX_PTX_INSTRUCTION_RCP_APPROX_FTZ_F32] = STACK_PTX_EMIT_OP_RECIPROCAL,
	[STACK_PTX_PTX_INSTRUCTION_SQRT_APPROX_FTZ_F32] = STACK_PTX_EMIT_OP_FUNC("np.sqrt"),
	[STACK_PTX_PTX_INSTRUCTION_RSQRT_APPROX_FTZ_F32] = STACK_PTX_EMIT_OP_RSQRT,
	[STACK_PTX_PTX_INSTRUCTION_SIN_APPROX_FTZ_F32] = STACK_PTX_EMIT_OP_FUNC("np.sin"),
	[STACK_PTX_PTX_INSTRUCTION_COS_APPROX_FTZ_F32] = STACK_PTX_EMIT_OP_FUNC("np.cos"),
	[STACK_PTX_PTX_INSTRUCTION_LG2_APPROX_FTZ_F32] = STACK_PTX_EMIT_OP_FUNC("np.log2"),
	[STACK_PTX_PTX_INSTRUCTION_EX2_APPROX_FTZ_F32] = STACK_PTX_EMIT_OP_FUNC("np.exp2"),
	[STACK_PTX_PTX_INSTRUCTION_TANH_APPROX_F32] = STACK_PTX_EMIT_OP_FUNC("np.tanh"),
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
rng_unit_f32(
	Rng* rng
) {
	return (float)(rng_next(rng) & UINT32_C(0x00FFFFFF)) / (float)UINT32_C(0x01000000);
}

static
float
rng_range_f32(
	Rng* rng,
	float min_value,
	float max_value
) {
	return min_value + (max_value - min_value) * rng_unit_f32(rng);
}

static
StackPtxInstruction
random_op_instruction(
	RandomOp op
) {
	if (op.kind == RANDOM_OP_KIND_ROUTINE) {
		return STACK_PTX_LITERAL(stack_ptx_encode_routine(op.operand));
	}
	return stack_ptx_ptx_instructions[op.operand];
}

static
const char*
routine_name_for_index(
	uint32_t routine_idx
) {
	if (routine_idx < STACK_PTX_ARRAY_NUM_ELEMS(protected_routine_names)) {
		return protected_routine_names[routine_idx];
	}
	return "routine";
}

static
uint64_t
random_function_flag(
	size_t function_idx
) {
	return UINT64_C(1) << function_idx;
}

static
uint64_t
random_function_default_mask(void) {
	return (random_function_flag(RANDOM_FUNCTION_NUM_ENUMS) - 1u);
}

static
const RandomFunctionSpec*
find_random_function_spec(
	const char* name,
	size_t* index_out
) {
	for (size_t i = 0; i < RANDOM_FUNCTION_NUM_ENUMS; i++) {
		if (strcmp(random_function_specs[i].name, name) == 0) {
			if (index_out != NULL) {
				*index_out = i;
			}
			return &random_function_specs[i];
		}
	}
	return NULL;
}

static
bool
build_allowed_ops(
	uint64_t mask,
	RandomOp* unary_ops_out,
	size_t* num_unary_ops_out,
	RandomOp* binary_ops_out,
	size_t* num_binary_ops_out,
	RandomOp* ternary_ops_out,
	size_t* num_ternary_ops_out
) {
	size_t num_unary_ops = 0;
	size_t num_binary_ops = 0;
	size_t num_ternary_ops = 0;
	for (size_t i = 0; i < RANDOM_FUNCTION_NUM_ENUMS; i++) {
		if ((mask & random_function_flag(i)) == 0) {
			continue;
		}
		RandomOp op = random_function_specs[i].op;
		if (op.arity == 1) {
			unary_ops_out[num_unary_ops++] = op;
		} else if (op.arity == 2) {
			binary_ops_out[num_binary_ops++] = op;
		} else if (op.arity == 3) {
			ternary_ops_out[num_ternary_ops++] = op;
		} else {
			return false;
		}
	}

	if (num_binary_ops == 0 && num_ternary_ops == 0) {
		return false;
	}

	*num_unary_ops_out = num_unary_ops;
	*num_binary_ops_out = num_binary_ops;
	*num_ternary_ops_out = num_ternary_ops;
	return true;
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
	return live_count == 0 ? 0 : live_indices[rng_bounded(rng, live_count)];
}

static
void
append_instruction(
	StackPtxInstruction* instructions,
	size_t max_instructions,
	size_t* num_instructions,
	StackPtxInstruction instruction
) {
	if (*num_instructions < max_instructions) {
		instructions[(*num_instructions)++] = instruction;
	}
}

static
void
generate_program(
	StackPtxInstruction* instructions,
	size_t max_instructions,
	size_t* num_instructions_out,
	uint32_t seed,
	size_t input_dim,
	size_t num_steps,
	const RandomOp* unary_ops,
	size_t num_unary_ops,
	const RandomOp* binary_ops,
	size_t num_binary_ops,
	const RandomOp* ternary_ops,
	size_t num_ternary_ops
) {
	enum {
		STORE_SLOT_BASE = 12,
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

	Rng rng = { .state = seed };
	bool live_slots[STORE_SLOT_COUNT] = { false, false, false, false };
	size_t depth = 0;
	size_t num_instructions = 0;

	append_instruction(
		instructions,
		max_instructions,
		&num_instructions,
		STACK_PTX_LITERAL(stack_ptx_encode_input(rng_bounded(&rng, input_dim)))
	);
	append_instruction(
		instructions,
		max_instructions,
		&num_instructions,
		STACK_PTX_LITERAL(stack_ptx_encode_input(rng_bounded(&rng, input_dim)))
	);
	depth = 2;

	for (size_t step = 0; step < num_steps; step++) {
		enum Action choices[24];
		size_t num_choices = 0;

		if (depth < 8) {
			choices[num_choices++] = ACTION_PUSH_INPUT;
			choices[num_choices++] = ACTION_PUSH_CONSTANT;
		}
		if (depth >= 1 && num_unary_ops > 0) {
			choices[num_choices++] = ACTION_UNARY;
			choices[num_choices++] = ACTION_UNARY;
		}
		if (depth >= 1) {
			choices[num_choices++] = ACTION_DUP;
			choices[num_choices++] = ACTION_STORE;
		}
		if (depth >= 2 && num_binary_ops > 0) {
			choices[num_choices++] = ACTION_BINARY;
			choices[num_choices++] = ACTION_BINARY;
		}
		if (depth >= 2) {
			choices[num_choices++] = ACTION_SWAP;
		}
		if (depth >= 3 && num_ternary_ops > 0) {
			choices[num_choices++] = ACTION_TERNARY;
		}
		if (has_live_slot(live_slots, STORE_SLOT_COUNT)) {
			choices[num_choices++] = ACTION_LOAD;
		}

		enum Action action = choices[rng_bounded(&rng, num_choices)];
		switch (action) {
			case ACTION_PUSH_INPUT: {
				append_instruction(
					instructions,
					max_instructions,
					&num_instructions,
					STACK_PTX_LITERAL(stack_ptx_encode_input(rng_bounded(&rng, input_dim)))
				);
				depth++;
			} break;
			case ACTION_PUSH_CONSTANT: {
				append_instruction(
					instructions,
					max_instructions,
					&num_instructions,
					STACK_PTX_LITERAL(stack_ptx_encode_constant_f32(rng_range_f32(&rng, -4.0f, 4.0f)))
				);
				depth++;
			} break;
			case ACTION_UNARY: {
				RandomOp op = unary_ops[rng_bounded(&rng, num_unary_ops)];
				append_instruction(
					instructions,
					max_instructions,
					&num_instructions,
					random_op_instruction(op)
				);
			} break;
			case ACTION_BINARY: {
				RandomOp op = binary_ops[rng_bounded(&rng, num_binary_ops)];
				append_instruction(
					instructions,
					max_instructions,
					&num_instructions,
					random_op_instruction(op)
				);
				depth--;
			} break;
			case ACTION_TERNARY: {
				RandomOp op = ternary_ops[rng_bounded(&rng, num_ternary_ops)];
				append_instruction(
					instructions,
					max_instructions,
					&num_instructions,
					random_op_instruction(op)
				);
				depth -= 2;
			} break;
			case ACTION_STORE: {
				size_t slot = rng_bounded(&rng, STORE_SLOT_COUNT);
				append_instruction(
					instructions,
					max_instructions,
					&num_instructions,
					STACK_PTX_LITERAL(stack_ptx_encode_store(
						STACK_PTX_STACK_TYPE_F32,
						STORE_SLOT_BASE + slot
					))
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
					STACK_PTX_LITERAL(stack_ptx_encode_load(STORE_SLOT_BASE + slot))
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

	while (depth > 1) {
		if (depth == 2 && num_binary_ops == 0) {
			append_instruction(
				instructions,
				max_instructions,
				&num_instructions,
				STACK_PTX_LITERAL(stack_ptx_encode_input(rng_bounded(&rng, input_dim)))
			);
			depth++;
			continue;
		}

		if (depth >= 3 && num_ternary_ops > 0 &&
			(num_binary_ops == 0 || rng_bounded(&rng, 2) == 0)) {
			RandomOp op = ternary_ops[rng_bounded(&rng, num_ternary_ops)];
			append_instruction(
				instructions,
				max_instructions,
				&num_instructions,
				random_op_instruction(op)
			);
			depth -= 2;
			continue;
		}

		if (depth >= 2 && num_binary_ops > 0) {
			RandomOp op = binary_ops[rng_bounded(&rng, num_binary_ops)];
			append_instruction(
				instructions,
				max_instructions,
				&num_instructions,
				random_op_instruction(op)
			);
			depth--;
			continue;
		}
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
bool
parse_size_arg(
	const char* text,
	size_t* value_out
) {
	char* end = NULL;
	unsigned long long value = strtoull(text, &end, 10);
	if (text[0] == '\0' || end == NULL || *end != '\0') {
		return false;
	}
	*value_out = (size_t)value;
	return true;
}

static
bool
parse_u32_arg(
	const char* text,
	uint32_t* value_out
) {
	char* end = NULL;
	unsigned long value = strtoul(text, &end, 10);
	if (text[0] == '\0' || end == NULL || *end != '\0') {
		return false;
	}
	*value_out = (uint32_t)value;
	return true;
}

static
bool
parse_f32_arg(
	const char* text,
	float* value_out
) {
	char* end = NULL;
	float value = strtof(text, &end);
	if (text[0] == '\0' || end == NULL || *end != '\0' || !(value > 0.0f) || !isfinite(value)) {
		return false;
	}
	*value_out = value;
	return true;
}

static
bool
parse_backend_name(
	const char* text,
	uint32_t* flag_out
) {
	if (strcmp(text, "c") == 0) {
		*flag_out = BACKEND_FLAG_C;
		return true;
	}
	if (strcmp(text, "cuda") == 0) {
		*flag_out = BACKEND_FLAG_CUDA;
		return true;
	}
	if (strcmp(text, "rust") == 0) {
		*flag_out = BACKEND_FLAG_RUST;
		return true;
	}
	if (strcmp(text, "numpy") == 0) {
		*flag_out = BACKEND_FLAG_NUMPY;
		return true;
	}
	if (strcmp(text, "latex") == 0) {
		*flag_out = BACKEND_FLAG_LATEX;
		return true;
	}
	if (strcmp(text, "markdown") == 0) {
		*flag_out = BACKEND_FLAG_MARKDOWN;
		return true;
	}
	return false;
}

static
bool
parse_backends_csv(
	const char* csv,
	uint32_t* backend_flags_out
) {
	uint32_t backend_flags = 0;
	char* copy = strdup(csv);
	if (copy == NULL) {
		return false;
	}

	char* save_ptr = NULL;
	for (char* token = strtok_r(copy, ",", &save_ptr);
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

		uint32_t flag = 0;
		if (token[0] == '\0' || !parse_backend_name(token, &flag)) {
			free(copy);
			return false;
		}
		backend_flags |= flag;
	}

	free(copy);
	if (backend_flags == 0) {
		return false;
	}
	*backend_flags_out = backend_flags;
	return true;
}

static
bool
parse_allowed_functions_csv(
	const char* csv,
	uint64_t* allowed_function_mask_out
) {
	uint64_t allowed_function_mask = 0;
	char* copy = strdup(csv);
	if (copy == NULL) {
		return false;
	}

	char* save_ptr = NULL;
	for (char* token = strtok_r(copy, ",", &save_ptr);
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

		if (strcmp(token, "all") == 0) {
			allowed_function_mask = random_function_default_mask();
			continue;
		}

		size_t function_idx = 0;
		if (token[0] == '\0' || find_random_function_spec(token, &function_idx) == NULL) {
			free(copy);
			return false;
		}
		allowed_function_mask |= random_function_flag(function_idx);
	}

	free(copy);
	if (allowed_function_mask == 0) {
		return false;
	}
	*allowed_function_mask_out = allowed_function_mask;
	return true;
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
	size_t* count_out
) {
	*names_out = NULL;
	*count_out = 0;
	if (csv == NULL) {
		return true;
	}

	char* copy = strdup(csv);
	if (copy == NULL) {
		return false;
	}

	char** names = (char**)calloc(expected_count, sizeof(char*));
	if (names == NULL) {
		free(copy);
		return false;
	}

	size_t count = 0;
	char* save_ptr = NULL;
	for (char* token = strtok_r(copy, ",", &save_ptr);
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
		if (token[0] == '\0' || count >= expected_count) {
			free_string_array(names, expected_count);
			free(copy);
			return false;
		}
		names[count] = strdup(token);
		if (names[count] == NULL) {
			free_string_array(names, expected_count);
			free(copy);
			return false;
		}
		count++;
	}

	free(copy);
	if (count != expected_count) {
		free_string_array(names, expected_count);
		return false;
	}

	*names_out = names;
	*count_out = count;
	return true;
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
		.input_dim = 8,
		.num_features = 6,
		.steps = 12,
		.eps = 1.0e-6f,
		.allowed_function_mask = random_function_default_mask(),
		.backend_flags = BACKEND_FLAG_C |
			BACKEND_FLAG_CUDA |
			BACKEND_FLAG_RUST |
			BACKEND_FLAG_NUMPY |
			BACKEND_FLAG_LATEX |
			BACKEND_FLAG_MARKDOWN,
		.emit_spec_comment = false,
		.function_name = "linear_regression_infer",
		.input_names_csv = NULL,
		.markdown_factor_expression_min_bytes = 96
	};

	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "--seed") == 0) {
			if (i + 1 >= argc || !parse_u32_arg(argv[++i], &options.seed)) {
				return false;
			}
		} else if (strcmp(argv[i], "--input-dim") == 0) {
			if (i + 1 >= argc || !parse_size_arg(argv[++i], &options.input_dim)) {
				return false;
			}
		} else if (strcmp(argv[i], "--num-features") == 0) {
			if (i + 1 >= argc || !parse_size_arg(argv[++i], &options.num_features)) {
				return false;
			}
		} else if (strcmp(argv[i], "--steps") == 0) {
			if (i + 1 >= argc || !parse_size_arg(argv[++i], &options.steps)) {
				return false;
			}
		} else if (strcmp(argv[i], "--eps") == 0) {
			if (i + 1 >= argc || !parse_f32_arg(argv[++i], &options.eps)) {
				return false;
			}
		} else if (strcmp(argv[i], "--backends") == 0) {
			if (i + 1 >= argc || !parse_backends_csv(argv[++i], &options.backend_flags)) {
				return false;
			}
		} else if (strcmp(argv[i], "--allowed-functions") == 0) {
			if (i + 1 >= argc ||
				!parse_allowed_functions_csv(argv[++i], &options.allowed_function_mask)) {
				return false;
			}
		} else if (strcmp(argv[i], "--function-name") == 0) {
			if (i + 1 >= argc) {
				return false;
			}
			options.function_name = argv[++i];
		} else if (strcmp(argv[i], "--input-names") == 0) {
			if (i + 1 >= argc) {
				return false;
			}
			options.input_names_csv = argv[++i];
		} else if (strcmp(argv[i], "--show-spec") == 0) {
			options.emit_spec_comment = true;
		} else if (strcmp(argv[i], "--markdown-factor-bytes") == 0) {
			if (i + 1 >= argc ||
				!parse_size_arg(argv[++i], &options.markdown_factor_expression_min_bytes)) {
				return false;
			}
		} else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
			return false;
		} else {
			return false;
		}
	}

	if (options.input_dim == 0 || options.num_features == 0) {
		return false;
	}
	*options_out = options;
	return true;
}

static
void
print_usage(
	const char* argv0
) {
	fprintf(stderr,
		"Usage: %s [--seed N] [--input-dim N] [--num-features N] [--steps N] [--eps X] "
		"[--backends c,cuda,rust,numpy,latex,markdown] "
		"[--allowed-functions all,abs,neg,rcp,sqrt,rsqrt,sin,cos,log2,exp2,tanh,add,sub,mul,div,min,max,copysign,fma] "
		"[--function-name NAME] [--input-names name0,name1,...] "
		"[--show-spec] [--markdown-factor-bytes N]\n",
		argv0
	);
}

static
bool
make_emit_settings(
	StackPtxEmitLanguage backend,
	const CliOptions* options,
	StackPtxEmitSettings* settings_out
) {
	if (settings_out == NULL || options == NULL) {
		return false;
	}

	StackPtxEmitSettings settings = {
		.language = backend,
		.vector = { .kind = STACK_PTX_EMIT_VECTOR_KIND_NONE, .bits = 0 }
	};
	switch (backend) {
		case STACK_PTX_EMIT_LANGUAGE_C:
			settings.as.c = base_c_settings;
			settings.as.c.function_name = options->function_name;
			break;
		case STACK_PTX_EMIT_LANGUAGE_CUDA:
			settings.as.cuda = base_cuda_settings;
			settings.as.cuda.function_name = options->function_name;
			break;
		case STACK_PTX_EMIT_LANGUAGE_RUST:
			settings.as.rust = base_rust_settings;
			settings.as.rust.function_name = options->function_name;
			break;
		case STACK_PTX_EMIT_LANGUAGE_NUMPY:
			settings.as.numpy = base_numpy_settings;
			settings.as.numpy.function_name = options->function_name;
			break;
		case STACK_PTX_EMIT_LANGUAGE_LATEX:
			settings.as.latex = base_latex_settings;
			break;
		case STACK_PTX_EMIT_LANGUAGE_MARKDOWN:
			settings.as.markdown = base_markdown_settings;
			settings.as.markdown.factor_expression_min_bytes =
				options->markdown_factor_expression_min_bytes;
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
compile_backend_output(
	StackPtxEmitLanguage backend,
	const CliOptions* options,
	const StackPtxEmitLinearRegressionInfo* regression_info,
	size_t workspace_size,
	char** output_text_out
) {
	StackPtxEmitSettings settings;
	if (!make_emit_settings(backend, options, &settings)) {
		return false;
	}

	void* workspace = malloc(workspace_size);
	if (workspace == NULL) {
		return false;
	}

	size_t required_bytes = 0;
	StackPtxEmitResult emit_result = STACK_PTX_EMIT_SUCCESS;
	StackPtxResult stack_result = STACK_PTX_SUCCESS;
	StackPtxEmitLinearRegressionResult regression_result =
		stack_ptx_emit_linear_regression_compile(
			&compiler_info,
			&stack_ptx_stack_info,
			&settings,
			regression_info,
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
			"compile measure failed for backend %d: %s (emit=%s, stack_ptx=%s)\n",
			(int)backend,
			stack_ptx_emit_linear_regression_result_to_string(regression_result),
			stack_ptx_emit_result_to_string(emit_result),
			stack_ptx_result_to_string(stack_result)
		);
		free(workspace);
		return false;
	}

	char* buffer = (char*)malloc(required_bytes + 1);
	if (buffer == NULL) {
		free(workspace);
		return false;
	}

	regression_result = stack_ptx_emit_linear_regression_compile(
		&compiler_info,
		&stack_ptx_stack_info,
		&settings,
		regression_info,
		workspace,
		workspace_size,
		buffer,
		required_bytes + 1,
		&required_bytes,
		&emit_result,
		&stack_result
	);
	free(workspace);
	if (regression_result != STACK_PTX_EMIT_LINEAR_REGRESSION_SUCCESS) {
		fprintf(
			stderr,
			"compile emit failed for backend %d: %s (emit=%s, stack_ptx=%s)\n",
			(int)backend,
			stack_ptx_emit_linear_regression_result_to_string(regression_result),
			stack_ptx_emit_result_to_string(emit_result),
			stack_ptx_result_to_string(stack_result)
		);
		free(buffer);
		return false;
	}

	*output_text_out = buffer;
	return true;
}

static
const char*
backend_name_for_flag(
	uint32_t flag
) {
	switch (flag) {
		case BACKEND_FLAG_C: return "c";
		case BACKEND_FLAG_CUDA: return "cuda";
		case BACKEND_FLAG_RUST: return "rust";
		case BACKEND_FLAG_NUMPY: return "numpy";
		case BACKEND_FLAG_LATEX: return "latex";
		case BACKEND_FLAG_MARKDOWN: return "markdown";
	}
	return "unknown";
}

static
StackPtxEmitLanguage
backend_language_for_flag(
	uint32_t flag
) {
	switch (flag) {
		case BACKEND_FLAG_C: return STACK_PTX_EMIT_LANGUAGE_C;
		case BACKEND_FLAG_CUDA: return STACK_PTX_EMIT_LANGUAGE_CUDA;
		case BACKEND_FLAG_RUST: return STACK_PTX_EMIT_LANGUAGE_RUST;
		case BACKEND_FLAG_NUMPY: return STACK_PTX_EMIT_LANGUAGE_NUMPY;
		case BACKEND_FLAG_LATEX: return STACK_PTX_EMIT_LANGUAGE_LATEX;
		case BACKEND_FLAG_MARKDOWN: return STACK_PTX_EMIT_LANGUAGE_MARKDOWN;
	}
	return STACK_PTX_EMIT_LANGUAGE_NUM_ENUMS;
}

static
void
json_write_escaped(
	FILE* file,
	const char* text
) {
	for (size_t i = 0; text[i] != '\0'; i++) {
		unsigned char c = (unsigned char)text[i];
		switch (c) {
			case '\\': fputs("\\\\", file); break;
			case '"': fputs("\\\"", file); break;
			case '\b': fputs("\\b", file); break;
			case '\f': fputs("\\f", file); break;
			case '\n': fputs("\\n", file); break;
			case '\r': fputs("\\r", file); break;
			case '\t': fputs("\\t", file); break;
			default:
				if (c < 0x20) {
					fprintf(file, "\\u%04X", (unsigned)c);
				} else {
					fputc((int)c, file);
				}
				break;
		}
	}
}

static
void
json_write_string(
	FILE* file,
	const char* text
) {
	fputc('"', file);
	json_write_escaped(file, text);
	fputc('"', file);
}

static
void
json_write_float(
	FILE* file,
	float value
) {
	char buffer[64];
	snprintf(buffer, sizeof(buffer), "%.9g", (double)value);
	fputs(buffer, file);
}

static
void
format_instruction_text(
	StackPtxInstruction instruction,
	char* buffer,
	size_t buffer_size
) {
	switch (instruction.instruction_type) {
		case STACK_PTX_INSTRUCTION_TYPE_INPUT:
			snprintf(buffer, buffer_size, "input(x_%u)", (unsigned)instruction.payload.u);
			return;
		case STACK_PTX_INSTRUCTION_TYPE_CONSTANT: {
			if (_stack_ptx_instruction_stack_idx(instruction) == STACK_PTX_STACK_TYPE_F32) {
				union {
					uint32_t u;
					float f;
				} constant = { .u = instruction.payload.u };
				snprintf(buffer, buffer_size, "const(%.9g)", (double)constant.f);
				return;
			}
			snprintf(buffer, buffer_size, "const");
			return;
		}
		case STACK_PTX_INSTRUCTION_TYPE_META:
			switch (instruction.payload.u) {
				case STACK_PTX_META_INSTRUCTION_DUP:
					snprintf(buffer, buffer_size, "dup");
					return;
				case STACK_PTX_META_INSTRUCTION_SWAP:
					snprintf(buffer, buffer_size, "swap");
					return;
				default:
					snprintf(buffer, buffer_size, "meta(%u)", (unsigned)instruction.payload.u);
					return;
			}
		case STACK_PTX_INSTRUCTION_TYPE_STORE:
			snprintf(buffer, buffer_size, "store(%u)", (unsigned)instruction.payload.u);
			return;
		case STACK_PTX_INSTRUCTION_TYPE_LOAD:
			snprintf(buffer, buffer_size, "load(%u)", (unsigned)instruction.payload.u);
			return;
		case STACK_PTX_INSTRUCTION_TYPE_PTX:
			if (instruction.payload.u < stack_ptx_stack_info.num_ptx_instructions) {
				snprintf(
					buffer,
					buffer_size,
					"%s",
					stack_ptx_stack_info.ptx_instruction_strings[instruction.payload.u]
				);
				return;
			}
			snprintf(buffer, buffer_size, "ptx(%u)", (unsigned)instruction.payload.u);
			return;
		case STACK_PTX_INSTRUCTION_TYPE_ROUTINE:
			snprintf(
				buffer,
				buffer_size,
				"%s",
				routine_name_for_index(instruction.payload.u)
			);
			return;
		case STACK_PTX_INSTRUCTION_TYPE_RETURN:
			snprintf(buffer, buffer_size, "return");
			return;
		default:
			snprintf(buffer, buffer_size, "instruction_type(%u)", (unsigned)instruction.instruction_type);
			return;
	}
}

static
bool
build_effective_input_names(
	size_t input_dim,
	char** parsed_input_names,
	size_t num_parsed_input_names,
	const char*** effective_names_out,
	char*** fallback_storage_out
) {
	if (parsed_input_names != NULL) {
		*effective_names_out = (const char**)parsed_input_names;
		*fallback_storage_out = NULL;
		return num_parsed_input_names == input_dim;
	}

	char** fallback_names = (char**)calloc(input_dim, sizeof(char*));
	if (fallback_names == NULL) {
		return false;
	}

	for (size_t i = 0; i < input_dim; i++) {
		fallback_names[i] = (char*)malloc(32);
		if (fallback_names[i] == NULL) {
			free_string_array(fallback_names, input_dim);
			return false;
		}
		snprintf(fallback_names[i], 32, "x%zu", i);
	}

	*effective_names_out = (const char**)fallback_names;
	*fallback_storage_out = fallback_names;
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

	char** parsed_input_names = NULL;
	size_t num_parsed_input_names = 0;
	if (!parse_input_names_csv(
			options.input_names_csv,
			options.input_dim,
			&parsed_input_names,
			&num_parsed_input_names)) {
		fprintf(stderr, "failed to parse --input-names\n");
		return 1;
	}

	const char** effective_input_names = NULL;
	char** fallback_input_names = NULL;
	if (!build_effective_input_names(
			options.input_dim,
			parsed_input_names,
			num_parsed_input_names,
			&effective_input_names,
			&fallback_input_names)) {
		fprintf(stderr, "failed to build input names\n");
		free_string_array(parsed_input_names, num_parsed_input_names);
		return 1;
	}

	StackPtxInstruction protected_div_routine[] = {
		STACK_PTX_LITERAL(stack_ptx_encode_store(STACK_PTX_STACK_TYPE_F32, PROTECTED_SLOT_INPUT)),
		STACK_PTX_LITERAL(stack_ptx_encode_store(STACK_PTX_STACK_TYPE_F32, PROTECTED_SLOT_SECOND)),
		STACK_PTX_LITERAL(stack_ptx_encode_load(PROTECTED_SLOT_SECOND)),
		STACK_PTX_LITERAL(stack_ptx_encode_ptx_instruction_abs_ftz_f32),
		STACK_PTX_LITERAL(stack_ptx_encode_constant_f32(options.eps)),
		STACK_PTX_LITERAL(stack_ptx_encode_ptx_instruction_max_ftz_f32),
		STACK_PTX_LITERAL(stack_ptx_encode_load(PROTECTED_SLOT_SECOND)),
		STACK_PTX_LITERAL(stack_ptx_encode_meta_swap(STACK_PTX_STACK_TYPE_F32)),
		STACK_PTX_LITERAL(stack_ptx_encode_ptx_instruction_copysign_f32),
		STACK_PTX_LITERAL(stack_ptx_encode_load(PROTECTED_SLOT_INPUT)),
		STACK_PTX_LITERAL(stack_ptx_encode_ptx_instruction_div_approx_ftz_f32),
		STACK_PTX_LITERAL(stack_ptx_encode_return),
	};
	StackPtxInstruction protected_rcp_routine[] = {
		STACK_PTX_LITERAL(stack_ptx_encode_store(STACK_PTX_STACK_TYPE_F32, PROTECTED_SLOT_INPUT)),
		STACK_PTX_LITERAL(stack_ptx_encode_load(PROTECTED_SLOT_INPUT)),
		STACK_PTX_LITERAL(stack_ptx_encode_ptx_instruction_abs_ftz_f32),
		STACK_PTX_LITERAL(stack_ptx_encode_constant_f32(options.eps)),
		STACK_PTX_LITERAL(stack_ptx_encode_ptx_instruction_max_ftz_f32),
		STACK_PTX_LITERAL(stack_ptx_encode_load(PROTECTED_SLOT_INPUT)),
		STACK_PTX_LITERAL(stack_ptx_encode_meta_swap(STACK_PTX_STACK_TYPE_F32)),
		STACK_PTX_LITERAL(stack_ptx_encode_ptx_instruction_copysign_f32),
		STACK_PTX_LITERAL(stack_ptx_encode_ptx_instruction_rcp_approx_ftz_f32),
		STACK_PTX_LITERAL(stack_ptx_encode_return),
	};
	StackPtxInstruction protected_sqrt_routine[] = {
		STACK_PTX_LITERAL(stack_ptx_encode_constant_f32(0.0f)),
		STACK_PTX_LITERAL(stack_ptx_encode_ptx_instruction_max_ftz_f32),
		STACK_PTX_LITERAL(stack_ptx_encode_ptx_instruction_sqrt_approx_ftz_f32),
		STACK_PTX_LITERAL(stack_ptx_encode_return),
	};
	StackPtxInstruction protected_rsqrt_routine[] = {
		STACK_PTX_LITERAL(stack_ptx_encode_ptx_instruction_abs_ftz_f32),
		STACK_PTX_LITERAL(stack_ptx_encode_constant_f32(options.eps)),
		STACK_PTX_LITERAL(stack_ptx_encode_ptx_instruction_max_ftz_f32),
		STACK_PTX_LITERAL(stack_ptx_encode_ptx_instruction_rsqrt_approx_ftz_f32),
		STACK_PTX_LITERAL(stack_ptx_encode_return),
	};
	StackPtxInstruction protected_lg2_routine[] = {
		STACK_PTX_LITERAL(stack_ptx_encode_ptx_instruction_abs_ftz_f32),
		STACK_PTX_LITERAL(stack_ptx_encode_constant_f32(options.eps)),
		STACK_PTX_LITERAL(stack_ptx_encode_ptx_instruction_max_ftz_f32),
		STACK_PTX_LITERAL(stack_ptx_encode_ptx_instruction_lg2_approx_ftz_f32),
		STACK_PTX_LITERAL(stack_ptx_encode_return),
	};
	StackPtxInstruction protected_ex2_routine[] = {
		STACK_PTX_LITERAL(stack_ptx_encode_constant_f32(-80.0f)),
		STACK_PTX_LITERAL(stack_ptx_encode_ptx_instruction_max_ftz_f32),
		STACK_PTX_LITERAL(stack_ptx_encode_constant_f32(80.0f)),
		STACK_PTX_LITERAL(stack_ptx_encode_ptx_instruction_min_ftz_f32),
		STACK_PTX_LITERAL(stack_ptx_encode_ptx_instruction_ex2_approx_ftz_f32),
		STACK_PTX_LITERAL(stack_ptx_encode_return),
	};
	const StackPtxInstruction* protected_routines[PROTECTED_ROUTINE_NUM_ENUMS] = {
		[PROTECTED_ROUTINE_DIV] = protected_div_routine,
		[PROTECTED_ROUTINE_RCP] = protected_rcp_routine,
		[PROTECTED_ROUTINE_SQRT] = protected_sqrt_routine,
		[PROTECTED_ROUTINE_RSQRT] = protected_rsqrt_routine,
		[PROTECTED_ROUTINE_LG2] = protected_lg2_routine,
		[PROTECTED_ROUTINE_EX2] = protected_ex2_routine,
	};

	RandomOp allowed_unary_ops[RANDOM_FUNCTION_NUM_ENUMS];
	RandomOp allowed_binary_ops[RANDOM_FUNCTION_NUM_ENUMS];
	RandomOp allowed_ternary_ops[RANDOM_FUNCTION_NUM_ENUMS];
	size_t num_allowed_unary_ops = 0;
	size_t num_allowed_binary_ops = 0;
	size_t num_allowed_ternary_ops = 0;
	if (!build_allowed_ops(
			options.allowed_function_mask,
			allowed_unary_ops,
			&num_allowed_unary_ops,
			allowed_binary_ops,
			&num_allowed_binary_ops,
			allowed_ternary_ops,
			&num_allowed_ternary_ops)) {
		fprintf(stderr, "allowed functions must include at least one binary or ternary operator\n");
		free_string_array(fallback_input_names, options.input_dim);
		free_string_array(parsed_input_names, num_parsed_input_names);
		return 1;
	}

	size_t max_instructions_per_feature = options.steps + 16;
	GeneratedFeature* features = (GeneratedFeature*)calloc(options.num_features, sizeof(GeneratedFeature));
	StackPtxInstruction* instruction_storage =
		(StackPtxInstruction*)calloc(options.num_features * max_instructions_per_feature, sizeof(StackPtxInstruction));
	StackPtxEmitLinearRegressionFeatureProgram* feature_programs =
		(StackPtxEmitLinearRegressionFeatureProgram*)calloc(options.num_features, sizeof(StackPtxEmitLinearRegressionFeatureProgram));
	float* means = (float*)calloc(options.num_features, sizeof(float));
	float* stddevs = (float*)calloc(options.num_features, sizeof(float));
	float* beta_standardized = (float*)calloc(options.num_features, sizeof(float));
	if (features == NULL ||
		instruction_storage == NULL ||
		feature_programs == NULL ||
		means == NULL ||
		stddevs == NULL ||
		beta_standardized == NULL) {
		fprintf(stderr, "allocation failed\n");
		free_string_array(fallback_input_names, options.input_dim);
		free_string_array(parsed_input_names, num_parsed_input_names);
		free(features);
		free(instruction_storage);
		free(feature_programs);
		free(means);
		free(stddevs);
		free(beta_standardized);
		return 1;
	}

	Rng rng = { .state = options.seed ^ UINT32_C(0xA5A5A5A5) };
	for (size_t i = 0; i < options.num_features; i++) {
		char* feature_name = (char*)malloc(48);
		if (feature_name == NULL) {
			fprintf(stderr, "feature name allocation failed\n");
			free_string_array(fallback_input_names, options.input_dim);
			free_string_array(parsed_input_names, num_parsed_input_names);
			free(features);
			free(instruction_storage);
			free(feature_programs);
			free(means);
			free(stddevs);
			free(beta_standardized);
			return 1;
		}
		snprintf(feature_name, 48, "feature_%zu", i + 1);
		features[i].name = feature_name;
		features[i].instructions = instruction_storage + (i * max_instructions_per_feature);
		generate_program(
			features[i].instructions,
			max_instructions_per_feature,
			&features[i].num_instructions,
			program_seed_for_index(options.seed, i + 1),
			options.input_dim,
			options.steps,
			allowed_unary_ops,
			num_allowed_unary_ops,
			allowed_binary_ops,
			num_allowed_binary_ops,
			allowed_ternary_ops,
			num_allowed_ternary_ops
		);
		feature_programs[i].name = NULL;
		feature_programs[i].instructions = features[i].instructions;
		means[i] = rng_range_f32(&rng, -3.0f, 3.0f);
		stddevs[i] = rng_range_f32(&rng, 0.5f, 4.5f);
		beta_standardized[i] = rng_range_f32(&rng, -4.0f, 4.0f);
	}

	StackPtxEmitLinearRegressionInfo regression_info = {
		.function_name = options.function_name,
		.input_dim = options.input_dim,
		.input_names = effective_input_names,
		.num_input_names = options.input_dim,
		.feature_programs = feature_programs,
		.num_feature_programs = options.num_features,
		.means = means,
		.stddevs = stddevs,
		.beta_standardized = beta_standardized,
		.routines = protected_routines,
		.num_routines = PROTECTED_ROUTINE_NUM_ENUMS,
		.execution_limit = 1024,
		.emit_spec_comment = options.emit_spec_comment
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
		fprintf(
			stderr,
			"workspace_size failed: %s (emit=%s, stack_ptx=%s)\n",
			stack_ptx_emit_linear_regression_result_to_string(regression_result),
			stack_ptx_emit_result_to_string(emit_result),
			stack_ptx_result_to_string(stack_result)
		);
		free_string_array(fallback_input_names, options.input_dim);
		free_string_array(parsed_input_names, num_parsed_input_names);
		for (size_t i = 0; i < options.num_features; i++) {
			free(features[i].name);
		}
		free(features);
		free(instruction_storage);
		free(feature_programs);
		free(means);
		free(stddevs);
		free(beta_standardized);
		return 1;
	}

	const uint32_t backend_order[] = {
		BACKEND_FLAG_C,
		BACKEND_FLAG_CUDA,
		BACKEND_FLAG_RUST,
		BACKEND_FLAG_NUMPY,
		BACKEND_FLAG_LATEX,
		BACKEND_FLAG_MARKDOWN
	};
	char* backend_outputs[STACK_PTX_ARRAY_NUM_ELEMS(backend_order)] = { NULL };
	for (size_t i = 0; i < STACK_PTX_ARRAY_NUM_ELEMS(backend_order); i++) {
		if ((options.backend_flags & backend_order[i]) == 0) {
			continue;
		}
		if (!compile_backend_output(
				backend_language_for_flag(backend_order[i]),
				&options,
				&regression_info,
				workspace_size,
				&backend_outputs[i])) {
			for (size_t j = 0; j < STACK_PTX_ARRAY_NUM_ELEMS(backend_order); j++) {
				free(backend_outputs[j]);
			}
			free_string_array(fallback_input_names, options.input_dim);
			free_string_array(parsed_input_names, num_parsed_input_names);
			for (size_t j = 0; j < options.num_features; j++) {
				free(features[j].name);
			}
			free(features);
			free(instruction_storage);
			free(feature_programs);
			free(means);
			free(stddevs);
			free(beta_standardized);
			return 1;
		}
	}

	FILE* out = stdout;
	fputs("{\n", out);
	fputs("  \"request\": {\n", out);
	fprintf(out, "    \"seed\": %u,\n", options.seed);
	fprintf(out, "    \"input_dim\": %zu,\n", options.input_dim);
	fprintf(out, "    \"num_features\": %zu,\n", options.num_features);
	fprintf(out, "    \"steps\": %zu,\n", options.steps);
	fprintf(out, "    \"eps\": %.9g,\n", (double)options.eps);
	fputs("    \"function_name\": ", out);
	json_write_string(out, options.function_name);
	fputs(",\n    \"show_spec\": ", out);
	fputs(options.emit_spec_comment ? "true" : "false", out);
	fprintf(out, ",\n    \"markdown_factor_expression_min_bytes\": %zu,\n", options.markdown_factor_expression_min_bytes);
	fputs("    \"allowed_functions\": [", out);
	bool first_function = true;
	for (size_t i = 0; i < RANDOM_FUNCTION_NUM_ENUMS; i++) {
		if ((options.allowed_function_mask & random_function_flag(i)) == 0) {
			continue;
		}
		if (!first_function) {
			fputs(", ", out);
		}
		json_write_string(out, random_function_specs[i].name);
		first_function = false;
	}
	fputs("],\n", out);
	fputs("    \"backends\": [", out);
	bool first_backend = true;
	for (size_t i = 0; i < STACK_PTX_ARRAY_NUM_ELEMS(backend_order); i++) {
		if ((options.backend_flags & backend_order[i]) == 0) {
			continue;
		}
		if (!first_backend) {
			fputs(", ", out);
		}
		json_write_string(out, backend_name_for_flag(backend_order[i]));
		first_backend = false;
	}
	fputs("]\n", out);
	fputs("  },\n", out);

	fputs("  \"regression\": {\n", out);
	fputs("    \"input_names\": [", out);
	for (size_t i = 0; i < options.input_dim; i++) {
		if (i != 0) {
			fputs(", ", out);
		}
		json_write_string(out, effective_input_names[i]);
	}
	fputs("],\n", out);
	fputs("    \"means\": [", out);
	for (size_t i = 0; i < options.num_features; i++) {
		if (i != 0) {
			fputs(", ", out);
		}
		json_write_float(out, means[i]);
	}
	fputs("],\n", out);
	fputs("    \"stddevs\": [", out);
	for (size_t i = 0; i < options.num_features; i++) {
		if (i != 0) {
			fputs(", ", out);
		}
		json_write_float(out, stddevs[i]);
	}
	fputs("],\n", out);
	fputs("    \"beta_standardized\": [", out);
	for (size_t i = 0; i < options.num_features; i++) {
		if (i != 0) {
			fputs(", ", out);
		}
		json_write_float(out, beta_standardized[i]);
	}
	fputs("],\n", out);
	fputs("    \"beta_unstandardized\": [", out);
	for (size_t i = 0; i < options.num_features; i++) {
		if (i != 0) {
			fputs(", ", out);
		}
		json_write_float(out, beta_standardized[i] / stddevs[i]);
	}
	fputs("]\n", out);
	fputs("  },\n", out);

	fputs("  \"features\": [\n", out);
	for (size_t i = 0; i < options.num_features; i++) {
		fprintf(out, "    {\n      \"index\": %zu,\n      \"name\": ", i + 1);
		json_write_string(out, features[i].name);
		fprintf(out, ",\n      \"instruction_count\": %zu,\n      \"program\": [", features[i].num_instructions);
		for (size_t j = 0; j < features[i].num_instructions; j++) {
			char instruction_text[128];
			format_instruction_text(features[i].instructions[j], instruction_text, sizeof(instruction_text));
			if (j != 0) {
				fputs(", ", out);
			}
			json_write_string(out, instruction_text);
		}
		fputs("]\n    }", out);
		if (i + 1 != options.num_features) {
			fputs(",", out);
		}
		fputs("\n", out);
	}
	fputs("  ],\n", out);

	fputs("  \"outputs\": {\n", out);
	bool first_output = true;
	for (size_t i = 0; i < STACK_PTX_ARRAY_NUM_ELEMS(backend_order); i++) {
		if (backend_outputs[i] == NULL) {
			continue;
		}
		if (!first_output) {
			fputs(",\n", out);
		}
		fputs("    ", out);
		json_write_string(out, backend_name_for_flag(backend_order[i]));
		fputs(": ", out);
		json_write_string(out, backend_outputs[i]);
		first_output = false;
	}
	fputs("\n  }\n", out);
	fputs("}\n", out);

	for (size_t i = 0; i < STACK_PTX_ARRAY_NUM_ELEMS(backend_order); i++) {
		free(backend_outputs[i]);
	}
	free_string_array(fallback_input_names, options.input_dim);
	free_string_array(parsed_input_names, num_parsed_input_names);
	for (size_t i = 0; i < options.num_features; i++) {
		free(features[i].name);
	}
	free(features);
	free(instruction_storage);
	free(feature_programs);
	free(means);
	free(stddevs);
	free(beta_standardized);
	return 0;
}
