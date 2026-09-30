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

#ifndef STACK_PTX_EMIT_LATEX_H_INCLUDE
#define STACK_PTX_EMIT_LATEX_H_INCLUDE

#ifdef STACK_PTX_EMIT_LATEX_IMPLEMENTATION
#define STACK_PTX_EMIT_BASE_IMPLEMENTATION
#endif

#include <stack_ptx_emit_base.h>
#include <stack_ptx_tools_descriptions.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	const char* document_title;
	const char* output_symbol;
	const char* const* special_register_exprs;
	size_t num_special_register_exprs;
	bool emit_document;
	bool use_input_indices;
} StackPtxEmitLatexSettings;

STACK_PTX_EMIT_PUBLIC_DEC
StackPtxEmitResult
stack_ptx_emit_latex_compile(
	const StackPtxCompilerInfo* compiler_info_ref,
	const StackPtxStackInfo* stack_info_ref,
	const StackPtxEmitLatexSettings* settings_ref,
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

#endif // STACK_PTX_EMIT_LATEX_H_INCLUDE

#ifdef STACK_PTX_EMIT_LATEX_IMPLEMENTATION
#ifndef STACK_PTX_EMIT_LATEX_IMPLEMENTATION_ONCE
#define STACK_PTX_EMIT_LATEX_IMPLEMENTATION_ONCE

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef struct {
	StackPtxEmitAstView ast_view;
	const StackPtxEmitLatexSettings* settings;
} StackPtxEmitLatexCompiler;

typedef enum {
	STACK_PTX_EMIT_LATEX_PRECEDENCE_LOWEST = 0,
	STACK_PTX_EMIT_LATEX_PRECEDENCE_ADD = 10,
	STACK_PTX_EMIT_LATEX_PRECEDENCE_MUL = 20,
	STACK_PTX_EMIT_LATEX_PRECEDENCE_PREFIX = 30,
	STACK_PTX_EMIT_LATEX_PRECEDENCE_FUNC = 40,
	STACK_PTX_EMIT_LATEX_PRECEDENCE_ATOM = 50
} StackPtxEmitLatexPrecedence;

static
inline
StackPtxEmitResult
_stack_ptx_emit_latex_append_escaped_text(
	StackPtxEmitWriter* writer,
	const char* text
) {
	if (writer == NULL || text == NULL) {
		return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
	}

	for (size_t i = 0; text[i] != '\0'; i++) {
		const char* replacement = NULL;
		switch (text[i]) {
			case '\\': replacement = "\\textbackslash{}"; break;
			case '{': replacement = "\\{"; break;
			case '}': replacement = "\\}"; break;
			case '_': replacement = "\\_"; break;
			case '%': replacement = "\\%"; break;
			case '&': replacement = "\\&"; break;
			case '#': replacement = "\\#"; break;
			case '$': replacement = "\\$"; break;
			default: break;
		}
		StackPtxEmitResult result = replacement == NULL
			? _stack_ptx_emit_writer_append(writer, "%c", text[i])
			: _stack_ptx_emit_writer_append(writer, "%s", replacement);
		if (result != STACK_PTX_EMIT_SUCCESS) {
			return result;
		}
	}
	return STACK_PTX_EMIT_SUCCESS;
}

static
inline
StackPtxEmitResult
_stack_ptx_emit_latex_append_name(
	StackPtxEmitWriter* writer,
	const char* name
) {
	StackPtxEmitResult result = _stack_ptx_emit_writer_append(writer, "\\mathrm{");
	if (result != STACK_PTX_EMIT_SUCCESS) {
		return result;
	}
	result = _stack_ptx_emit_latex_append_escaped_text(writer, name);
	if (result != STACK_PTX_EMIT_SUCCESS) {
		return result;
	}
	return _stack_ptx_emit_writer_append(writer, "}");
}

static
inline
StackPtxEmitResult
_stack_ptx_emit_latex_append_float_literal(
	StackPtxEmitWriter* writer,
	float value
) {
	if (writer == NULL) {
		return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
	}

	if (isnan(value)) {
		return _stack_ptx_emit_writer_append(writer, "\\mathrm{NaN}");
	}
	if (isinf(value)) {
		return _stack_ptx_emit_writer_append(
			writer,
			signbit(value) ? "-\\infty" : "\\infty"
		);
	}
	if (value == 0.0f && signbit(value)) {
		return _stack_ptx_emit_writer_append(writer, "-0");
	}

	char literal[64];
	int bytes = snprintf(literal, sizeof(literal), "%.9g", (double)value);
	if (bytes < 0 || (size_t)bytes >= sizeof(literal)) {
		return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
	}
	return _stack_ptx_emit_writer_append(writer, "%s", literal);
}

static
inline
StackPtxEmitResult
_stack_ptx_emit_latex_constant(
	const StackPtxEmitLatexCompiler* latex_compiler,
	StackPtxInstruction instruction,
	StackPtxEmitWriter* writer
) {
	if (latex_compiler == NULL || writer == NULL) {
		return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
	}

	StackPtxStackIdx stack_idx = _stack_ptx_instruction_stack_idx(instruction);
	if (stack_idx >= latex_compiler->ast_view.compiler->stack_info.num_stacks) {
		return STACK_PTX_EMIT_ERROR_STACK_PTX;
	}

	const char* literal_prefix =
		latex_compiler->ast_view.compiler->stack_info.stack_literal_prefixes[stack_idx];
	if (literal_prefix == NULL) {
		return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
	}

	if (strcmp(literal_prefix, "f32") == 0) {
		union {
			uint32_t u;
			float f;
		} constant = { .u = instruction.payload.u };
		return _stack_ptx_emit_latex_append_float_literal(writer, constant.f);
	}

	if (strcmp(literal_prefix, "u32") == 0) {
		return _stack_ptx_emit_writer_append(writer, "%u", (unsigned)instruction.payload.u);
	}

	if (strcmp(literal_prefix, "s32") == 0) {
		union {
			uint32_t u;
			int32_t s;
		} constant = { .u = instruction.payload.u };
		return _stack_ptx_emit_writer_append(writer, "%d", (int)constant.s);
	}

	if (strcmp(literal_prefix, "pred") == 0) {
		return _stack_ptx_emit_writer_append(
			writer,
			"%s",
			instruction.payload.u == 0 ? "\\mathrm{false}" : "\\mathrm{true}"
		);
	}

	return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
}

static
inline
StackPtxEmitResult
_stack_ptx_emit_latex_special(
	const StackPtxEmitLatexCompiler* latex_compiler,
	StackPtxInstruction instruction,
	StackPtxEmitWriter* writer
) {
	if (latex_compiler == NULL || writer == NULL) {
		return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
	}

	StackPtxIdx special_register_idx = instruction.payload.u;
	if (special_register_idx >= latex_compiler->settings->num_special_register_exprs) {
		return STACK_PTX_EMIT_ERROR_MISSING_SPECIAL_REGISTER_MAPPING;
	}
	const char* expr = latex_compiler->settings->special_register_exprs[special_register_idx];
	if (expr == NULL || expr[0] == '\0') {
		return STACK_PTX_EMIT_ERROR_MISSING_SPECIAL_REGISTER_MAPPING;
	}
	return _stack_ptx_emit_writer_append(writer, "%s", expr);
}

static
inline
StackPtxEmitLatexPrecedence
_stack_ptx_emit_latex_instruction_precedence(
	StackPtxIdx instruction_idx
) {
	switch (instruction_idx) {
		case STACK_PTX_PTX_INSTRUCTION_ADD_FTZ_F32:
		case STACK_PTX_PTX_INSTRUCTION_SUB_FTZ_F32:
		case STACK_PTX_PTX_INSTRUCTION_FMA_RN_FTZ_F32:
			return STACK_PTX_EMIT_LATEX_PRECEDENCE_ADD;
		case STACK_PTX_PTX_INSTRUCTION_MUL_FTZ_F32:
		case STACK_PTX_PTX_INSTRUCTION_DIV_APPROX_FTZ_F32:
			return STACK_PTX_EMIT_LATEX_PRECEDENCE_MUL;
		case STACK_PTX_PTX_INSTRUCTION_NEG_FTZ_F32:
			return STACK_PTX_EMIT_LATEX_PRECEDENCE_PREFIX;
		default:
			return STACK_PTX_EMIT_LATEX_PRECEDENCE_FUNC;
	}
}

static
inline
StackPtxEmitResult
_stack_ptx_emit_latex_ast_precedence(
	const StackPtxEmitLatexCompiler* latex_compiler,
	StackPtxAstIdx ast_idx,
	StackPtxEmitLatexPrecedence* precedence_out,
	StackPtxResult* stack_ptx_result_out
) {
	if (latex_compiler == NULL || precedence_out == NULL) {
		return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
	}
	if (ast_idx >= latex_compiler->ast_view.compiler->ast_size) {
		if (stack_ptx_result_out != NULL) {
			*stack_ptx_result_out = STACK_PTX_ERROR_INVALID_VALUE;
		}
		return STACK_PTX_EMIT_ERROR_STACK_PTX;
	}

	StackPtxInstruction instruction = latex_compiler->ast_view.compiler->ast[ast_idx];
	switch ((StackPtxInstructionType)instruction.instruction_type) {
		case STACK_PTX_INSTRUCTION_TYPE_INPUT:
		case STACK_PTX_INSTRUCTION_TYPE_REGISTER:
		case STACK_PTX_INSTRUCTION_TYPE_CONSTANT:
		case STACK_PTX_INSTRUCTION_TYPE_SPECIAL:
			*precedence_out = STACK_PTX_EMIT_LATEX_PRECEDENCE_ATOM;
			return STACK_PTX_EMIT_SUCCESS;
		case STACK_PTX_INSTRUCTION_TYPE_PTX: {
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
			(void)descriptor;
			*precedence_out = _stack_ptx_emit_latex_instruction_precedence(instruction_idx);
			return STACK_PTX_EMIT_SUCCESS;
		}
		default:
			return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
	}
}

static
inline
StackPtxEmitResult
_stack_ptx_emit_latex_render_ast(
	const StackPtxEmitLatexCompiler* latex_compiler,
	StackPtxAstIdx ast_idx,
	StackPtxEmitLatexPrecedence min_precedence,
	bool wrap_on_equal,
	StackPtxEmitWriter* writer,
	StackPtxResult* stack_ptx_result_out
);

static
inline
StackPtxEmitResult
_stack_ptx_emit_latex_render_children(
	const StackPtxEmitLatexCompiler* latex_compiler,
	StackPtxAstIdx child_ast_idx,
	StackPtxEmitLatexPrecedence min_precedence,
	bool wrap_on_equal,
	StackPtxEmitWriter* writer,
	StackPtxResult* stack_ptx_result_out
) {
	return _stack_ptx_emit_latex_render_ast(
		latex_compiler,
		child_ast_idx,
		min_precedence,
		wrap_on_equal,
		writer,
		stack_ptx_result_out
	);
}

static
inline
StackPtxEmitResult
_stack_ptx_emit_latex_render_fallback_function(
	const StackPtxEmitLatexCompiler* latex_compiler,
	StackPtxIdx instruction_idx,
	const StackPtxAstIdx* arg_ast_indices,
	size_t num_args,
	StackPtxEmitWriter* writer,
	StackPtxResult* stack_ptx_result_out
) {
	const char* ptx_name = NULL;
	if (instruction_idx < latex_compiler->ast_view.compiler->stack_info.num_ptx_instructions) {
		ptx_name = latex_compiler->ast_view.compiler->stack_info.ptx_instruction_strings[instruction_idx];
	}
	if (ptx_name == NULL) {
		ptx_name = "ptx";
	}

	StackPtxEmitResult result = _stack_ptx_emit_writer_append(writer, "\\operatorname{");
	if (result != STACK_PTX_EMIT_SUCCESS) {
		return result;
	}
	result = _stack_ptx_emit_latex_append_escaped_text(writer, ptx_name);
	if (result != STACK_PTX_EMIT_SUCCESS) {
		return result;
	}
	result = _stack_ptx_emit_writer_append(writer, "}\\left(");
	if (result != STACK_PTX_EMIT_SUCCESS) {
		return result;
	}

	for (size_t i = 0; i < num_args; i++) {
		result = _stack_ptx_emit_latex_render_children(
			latex_compiler,
			arg_ast_indices[i],
			STACK_PTX_EMIT_LATEX_PRECEDENCE_LOWEST,
			false,
			writer,
			stack_ptx_result_out
		);
		if (result != STACK_PTX_EMIT_SUCCESS) {
			return result;
		}
		if (i + 1 < num_args) {
			result = _stack_ptx_emit_writer_append(writer, ", ");
			if (result != STACK_PTX_EMIT_SUCCESS) {
				return result;
			}
		}
	}

	return _stack_ptx_emit_writer_append(writer, "\\right)");
}

static
inline
StackPtxEmitResult
_stack_ptx_emit_latex_render_ptx(
	const StackPtxEmitLatexCompiler* latex_compiler,
	StackPtxAstIdx ast_idx,
	StackPtxEmitWriter* writer,
	StackPtxResult* stack_ptx_result_out
) {
	StackPtxInstruction instruction = latex_compiler->ast_view.compiler->ast[ast_idx];

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

	StackPtxAstIdx arg_ast_indices[STACK_PTX_MAX_NUM_PTX_ARGS] = {};
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
		arg_ast_indices[i] = arg_instruction.payload.ast_idx;
		arg_ast_idx++;
	}

	StackPtxArgIdx ret_type_idx = descriptor->ret_types[0];
	if (ret_type_idx >= latex_compiler->ast_view.compiler->stack_info.num_arg_types) {
		return STACK_PTX_EMIT_ERROR_STACK_PTX;
	}
	if (latex_compiler->ast_view.compiler->stack_info.arg_type_info[ret_type_idx].num_vec_elems != 0 ||
		_stack_ptx_arg_type_num_stack_elems(latex_compiler->ast_view.compiler, ret_type_idx) != 1) {
		return STACK_PTX_EMIT_ERROR_UNSUPPORTED_VECTOR;
	}

	switch (instruction_idx) {
		case STACK_PTX_PTX_INSTRUCTION_ABS_FTZ_F32: {
			StackPtxEmitResult result = _stack_ptx_emit_writer_append(writer, "\\left\\lvert ");
			if (result != STACK_PTX_EMIT_SUCCESS) {
				return result;
			}
			result = _stack_ptx_emit_latex_render_children(
				latex_compiler,
				arg_ast_indices[0],
				STACK_PTX_EMIT_LATEX_PRECEDENCE_LOWEST,
				false,
				writer,
				stack_ptx_result_out
			);
			if (result != STACK_PTX_EMIT_SUCCESS) {
				return result;
			}
			return _stack_ptx_emit_writer_append(writer, "\\right\\rvert");
		}
		case STACK_PTX_PTX_INSTRUCTION_NEG_FTZ_F32: {
			StackPtxEmitResult result = _stack_ptx_emit_writer_append(writer, "-");
			if (result != STACK_PTX_EMIT_SUCCESS) {
				return result;
			}
			return _stack_ptx_emit_latex_render_children(
				latex_compiler,
				arg_ast_indices[0],
				STACK_PTX_EMIT_LATEX_PRECEDENCE_PREFIX,
				false,
				writer,
				stack_ptx_result_out
			);
		}
		case STACK_PTX_PTX_INSTRUCTION_ADD_FTZ_F32: {
			StackPtxEmitResult result = _stack_ptx_emit_latex_render_children(
				latex_compiler,
				arg_ast_indices[0],
				STACK_PTX_EMIT_LATEX_PRECEDENCE_ADD,
				false,
				writer,
				stack_ptx_result_out
			);
			if (result != STACK_PTX_EMIT_SUCCESS) {
				return result;
			}
			result = _stack_ptx_emit_writer_append(writer, " + ");
			if (result != STACK_PTX_EMIT_SUCCESS) {
				return result;
			}
			return _stack_ptx_emit_latex_render_children(
				latex_compiler,
				arg_ast_indices[1],
				STACK_PTX_EMIT_LATEX_PRECEDENCE_ADD,
				false,
				writer,
				stack_ptx_result_out
			);
		}
		case STACK_PTX_PTX_INSTRUCTION_SUB_FTZ_F32: {
			StackPtxEmitResult result = _stack_ptx_emit_latex_render_children(
				latex_compiler,
				arg_ast_indices[0],
				STACK_PTX_EMIT_LATEX_PRECEDENCE_ADD,
				false,
				writer,
				stack_ptx_result_out
			);
			if (result != STACK_PTX_EMIT_SUCCESS) {
				return result;
			}
			result = _stack_ptx_emit_writer_append(writer, " - ");
			if (result != STACK_PTX_EMIT_SUCCESS) {
				return result;
			}
			return _stack_ptx_emit_latex_render_children(
				latex_compiler,
				arg_ast_indices[1],
				STACK_PTX_EMIT_LATEX_PRECEDENCE_ADD,
				true,
				writer,
				stack_ptx_result_out
			);
		}
		case STACK_PTX_PTX_INSTRUCTION_MUL_FTZ_F32: {
			StackPtxEmitResult result = _stack_ptx_emit_latex_render_children(
				latex_compiler,
				arg_ast_indices[0],
				STACK_PTX_EMIT_LATEX_PRECEDENCE_MUL,
				false,
				writer,
				stack_ptx_result_out
			);
			if (result != STACK_PTX_EMIT_SUCCESS) {
				return result;
			}
			result = _stack_ptx_emit_writer_append(writer, " ");
			if (result != STACK_PTX_EMIT_SUCCESS) {
				return result;
			}
			return _stack_ptx_emit_latex_render_children(
				latex_compiler,
				arg_ast_indices[1],
				STACK_PTX_EMIT_LATEX_PRECEDENCE_MUL,
				false,
				writer,
				stack_ptx_result_out
			);
		}
		case STACK_PTX_PTX_INSTRUCTION_DIV_APPROX_FTZ_F32: {
			StackPtxEmitResult result = _stack_ptx_emit_writer_append(writer, "\\frac{");
			if (result != STACK_PTX_EMIT_SUCCESS) {
				return result;
			}
			result = _stack_ptx_emit_latex_render_children(
				latex_compiler,
				arg_ast_indices[0],
				STACK_PTX_EMIT_LATEX_PRECEDENCE_LOWEST,
				false,
				writer,
				stack_ptx_result_out
			);
			if (result != STACK_PTX_EMIT_SUCCESS) {
				return result;
			}
			result = _stack_ptx_emit_writer_append(writer, "}{");
			if (result != STACK_PTX_EMIT_SUCCESS) {
				return result;
			}
			result = _stack_ptx_emit_latex_render_children(
				latex_compiler,
				arg_ast_indices[1],
				STACK_PTX_EMIT_LATEX_PRECEDENCE_LOWEST,
				false,
				writer,
				stack_ptx_result_out
			);
			if (result != STACK_PTX_EMIT_SUCCESS) {
				return result;
			}
			return _stack_ptx_emit_writer_append(writer, "}");
		}
		case STACK_PTX_PTX_INSTRUCTION_MIN_FTZ_F32:
		case STACK_PTX_PTX_INSTRUCTION_MAX_FTZ_F32: {
			const char* fn_name =
				instruction_idx == STACK_PTX_PTX_INSTRUCTION_MIN_FTZ_F32 ? "\\min" : "\\max";
			StackPtxEmitResult result = _stack_ptx_emit_writer_append(writer, "%s\\!\\left(", fn_name);
			if (result != STACK_PTX_EMIT_SUCCESS) {
				return result;
			}
			result = _stack_ptx_emit_latex_render_children(
				latex_compiler,
				arg_ast_indices[0],
				STACK_PTX_EMIT_LATEX_PRECEDENCE_LOWEST,
				false,
				writer,
				stack_ptx_result_out
			);
			if (result != STACK_PTX_EMIT_SUCCESS) {
				return result;
			}
			result = _stack_ptx_emit_writer_append(writer, ", ");
			if (result != STACK_PTX_EMIT_SUCCESS) {
				return result;
			}
			result = _stack_ptx_emit_latex_render_children(
				latex_compiler,
				arg_ast_indices[1],
				STACK_PTX_EMIT_LATEX_PRECEDENCE_LOWEST,
				false,
				writer,
				stack_ptx_result_out
			);
			if (result != STACK_PTX_EMIT_SUCCESS) {
				return result;
			}
			return _stack_ptx_emit_writer_append(writer, "\\right)");
		}
		case STACK_PTX_PTX_INSTRUCTION_FMA_RN_FTZ_F32: {
			StackPtxEmitResult result = _stack_ptx_emit_latex_render_children(
				latex_compiler,
				arg_ast_indices[0],
				STACK_PTX_EMIT_LATEX_PRECEDENCE_MUL,
				false,
				writer,
				stack_ptx_result_out
			);
			if (result != STACK_PTX_EMIT_SUCCESS) {
				return result;
			}
			result = _stack_ptx_emit_writer_append(writer, " ");
			if (result != STACK_PTX_EMIT_SUCCESS) {
				return result;
			}
			result = _stack_ptx_emit_latex_render_children(
				latex_compiler,
				arg_ast_indices[1],
				STACK_PTX_EMIT_LATEX_PRECEDENCE_MUL,
				false,
				writer,
				stack_ptx_result_out
			);
			if (result != STACK_PTX_EMIT_SUCCESS) {
				return result;
			}
			result = _stack_ptx_emit_writer_append(writer, " + ");
			if (result != STACK_PTX_EMIT_SUCCESS) {
				return result;
			}
			return _stack_ptx_emit_latex_render_children(
				latex_compiler,
				arg_ast_indices[2],
				STACK_PTX_EMIT_LATEX_PRECEDENCE_ADD,
				false,
				writer,
				stack_ptx_result_out
			);
		}
		case STACK_PTX_PTX_INSTRUCTION_RCP_APPROX_FTZ_F32:
		{
			StackPtxEmitResult result = _stack_ptx_emit_writer_append(writer, "\\frac{1}{");
			if (result != STACK_PTX_EMIT_SUCCESS) {
				return result;
			}
			result = _stack_ptx_emit_latex_render_children(
				latex_compiler,
				arg_ast_indices[0],
				STACK_PTX_EMIT_LATEX_PRECEDENCE_LOWEST,
				false,
				writer,
				stack_ptx_result_out
			);
			if (result != STACK_PTX_EMIT_SUCCESS) {
				return result;
			}
			return _stack_ptx_emit_writer_append(writer, "}");
		}
		case STACK_PTX_PTX_INSTRUCTION_SQRT_APPROX_FTZ_F32:
		{
			StackPtxEmitResult result = _stack_ptx_emit_writer_append(writer, "\\sqrt{");
			if (result != STACK_PTX_EMIT_SUCCESS) {
				return result;
			}
			result = _stack_ptx_emit_latex_render_children(
				latex_compiler,
				arg_ast_indices[0],
				STACK_PTX_EMIT_LATEX_PRECEDENCE_LOWEST,
				false,
				writer,
				stack_ptx_result_out
			);
			if (result != STACK_PTX_EMIT_SUCCESS) {
				return result;
			}
			return _stack_ptx_emit_writer_append(writer, "}");
		}
		case STACK_PTX_PTX_INSTRUCTION_RSQRT_APPROX_FTZ_F32:
		{
			StackPtxEmitResult result = _stack_ptx_emit_writer_append(writer, "\\frac{1}{\\sqrt{");
			if (result != STACK_PTX_EMIT_SUCCESS) {
				return result;
			}
			result = _stack_ptx_emit_latex_render_children(
				latex_compiler,
				arg_ast_indices[0],
				STACK_PTX_EMIT_LATEX_PRECEDENCE_LOWEST,
				false,
				writer,
				stack_ptx_result_out
			);
			if (result != STACK_PTX_EMIT_SUCCESS) {
				return result;
			}
			return _stack_ptx_emit_writer_append(writer, "}}");
		}
		case STACK_PTX_PTX_INSTRUCTION_SIN_APPROX_FTZ_F32:
		{
			StackPtxEmitResult result = _stack_ptx_emit_writer_append(writer, "\\sin\\!\\left(");
			if (result != STACK_PTX_EMIT_SUCCESS) {
				return result;
			}
			result = _stack_ptx_emit_latex_render_children(
				latex_compiler,
				arg_ast_indices[0],
				STACK_PTX_EMIT_LATEX_PRECEDENCE_LOWEST,
				false,
				writer,
				stack_ptx_result_out
			);
			if (result != STACK_PTX_EMIT_SUCCESS) {
				return result;
			}
			return _stack_ptx_emit_writer_append(writer, "\\right)");
		}
		case STACK_PTX_PTX_INSTRUCTION_COS_APPROX_FTZ_F32:
		{
			StackPtxEmitResult result = _stack_ptx_emit_writer_append(writer, "\\cos\\!\\left(");
			if (result != STACK_PTX_EMIT_SUCCESS) {
				return result;
			}
			result = _stack_ptx_emit_latex_render_children(
				latex_compiler,
				arg_ast_indices[0],
				STACK_PTX_EMIT_LATEX_PRECEDENCE_LOWEST,
				false,
				writer,
				stack_ptx_result_out
			);
			if (result != STACK_PTX_EMIT_SUCCESS) {
				return result;
			}
			return _stack_ptx_emit_writer_append(writer, "\\right)");
		}
		case STACK_PTX_PTX_INSTRUCTION_TANH_APPROX_F32:
		{
			StackPtxEmitResult result = _stack_ptx_emit_writer_append(writer, "\\tanh\\!\\left(");
			if (result != STACK_PTX_EMIT_SUCCESS) {
				return result;
			}
			result = _stack_ptx_emit_latex_render_children(
				latex_compiler,
				arg_ast_indices[0],
				STACK_PTX_EMIT_LATEX_PRECEDENCE_LOWEST,
				false,
				writer,
				stack_ptx_result_out
			);
			if (result != STACK_PTX_EMIT_SUCCESS) {
				return result;
			}
			return _stack_ptx_emit_writer_append(writer, "\\right)");
		}
		case STACK_PTX_PTX_INSTRUCTION_LG2_APPROX_FTZ_F32:
		{
			StackPtxEmitResult result = _stack_ptx_emit_writer_append(writer, "\\log_2\\!\\left(");
			if (result != STACK_PTX_EMIT_SUCCESS) {
				return result;
			}
			result = _stack_ptx_emit_latex_render_children(
				latex_compiler,
				arg_ast_indices[0],
				STACK_PTX_EMIT_LATEX_PRECEDENCE_LOWEST,
				false,
				writer,
				stack_ptx_result_out
			);
			if (result != STACK_PTX_EMIT_SUCCESS) {
				return result;
			}
			return _stack_ptx_emit_writer_append(writer, "\\right)");
		}
		case STACK_PTX_PTX_INSTRUCTION_EX2_APPROX_FTZ_F32:
		{
			StackPtxEmitResult result = _stack_ptx_emit_writer_append(writer, "2^{");
			if (result != STACK_PTX_EMIT_SUCCESS) {
				return result;
			}
			result = _stack_ptx_emit_latex_render_children(
				latex_compiler,
				arg_ast_indices[0],
				STACK_PTX_EMIT_LATEX_PRECEDENCE_LOWEST,
				false,
				writer,
				stack_ptx_result_out
			);
			if (result != STACK_PTX_EMIT_SUCCESS) {
				return result;
			}
			return _stack_ptx_emit_writer_append(writer, "}");
		}
		case STACK_PTX_PTX_INSTRUCTION_COPYSIGN_F32: {
			StackPtxEmitResult result = _stack_ptx_emit_writer_append(writer, "\\operatorname{copysign}\\!\\left(");
			if (result != STACK_PTX_EMIT_SUCCESS) {
				return result;
			}
			result = _stack_ptx_emit_latex_render_children(
				latex_compiler,
				arg_ast_indices[0],
				STACK_PTX_EMIT_LATEX_PRECEDENCE_LOWEST,
				false,
				writer,
				stack_ptx_result_out
			);
			if (result != STACK_PTX_EMIT_SUCCESS) {
				return result;
			}
			result = _stack_ptx_emit_writer_append(writer, ", ");
			if (result != STACK_PTX_EMIT_SUCCESS) {
				return result;
			}
			result = _stack_ptx_emit_latex_render_children(
				latex_compiler,
				arg_ast_indices[1],
				STACK_PTX_EMIT_LATEX_PRECEDENCE_LOWEST,
				false,
				writer,
				stack_ptx_result_out
			);
			if (result != STACK_PTX_EMIT_SUCCESS) {
				return result;
			}
			return _stack_ptx_emit_writer_append(writer, "\\right)");
		}
		default:
			return _stack_ptx_emit_latex_render_fallback_function(
				latex_compiler,
				instruction_idx,
				arg_ast_indices,
				num_args,
				writer,
				stack_ptx_result_out
			);
	}
}

static
inline
StackPtxEmitResult
_stack_ptx_emit_latex_render_ast(
	const StackPtxEmitLatexCompiler* latex_compiler,
	StackPtxAstIdx ast_idx,
	StackPtxEmitLatexPrecedence min_precedence,
	bool wrap_on_equal,
	StackPtxEmitWriter* writer,
	StackPtxResult* stack_ptx_result_out
) {
	if (latex_compiler == NULL || writer == NULL) {
		return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
	}
	if (ast_idx >= latex_compiler->ast_view.compiler->ast_size) {
		if (stack_ptx_result_out != NULL) {
			*stack_ptx_result_out = STACK_PTX_ERROR_INVALID_VALUE;
		}
		return STACK_PTX_EMIT_ERROR_STACK_PTX;
	}

	StackPtxInstruction instruction = latex_compiler->ast_view.compiler->ast[ast_idx];
	StackPtxEmitLatexPrecedence precedence = STACK_PTX_EMIT_LATEX_PRECEDENCE_ATOM;
	StackPtxEmitResult result = _stack_ptx_emit_latex_ast_precedence(
		latex_compiler,
		ast_idx,
		&precedence,
		stack_ptx_result_out
	);
	if (result != STACK_PTX_EMIT_SUCCESS) {
		return result;
	}

	bool wrap = precedence < min_precedence || (wrap_on_equal && precedence == min_precedence);
	if (wrap) {
		result = _stack_ptx_emit_writer_append(writer, "\\left(");
		if (result != STACK_PTX_EMIT_SUCCESS) {
			return result;
		}
	}

	switch ((StackPtxInstructionType)instruction.instruction_type) {
		case STACK_PTX_INSTRUCTION_TYPE_INPUT: {
			if (latex_compiler->settings->use_input_indices) {
				result = _stack_ptx_emit_writer_append(
					writer,
					"x_{%u}",
					(unsigned)(instruction.payload.u + 1)
				);
			} else {
				StackPtxIdx register_idx = instruction.payload.u;
				if (register_idx >= latex_compiler->ast_view.compiler->num_registers) {
					return STACK_PTX_EMIT_ERROR_STACK_PTX;
				}
				result = _stack_ptx_emit_latex_append_name(
					writer,
					latex_compiler->ast_view.compiler->registers[register_idx].name
				);
			}
		} break;
		case STACK_PTX_INSTRUCTION_TYPE_REGISTER: {
			char register_name[128];
			int bytes = snprintf(
				register_name,
				sizeof(register_name),
				"r_{%u,%u}",
				(unsigned)_stack_ptx_instruction_stack_idx(instruction),
				(unsigned)instruction.payload.reg
			);
			if (bytes < 0 || (size_t)bytes >= sizeof(register_name)) {
				return STACK_PTX_EMIT_ERROR_INSUFFICIENT_BUFFER;
			}
			result = _stack_ptx_emit_writer_append(writer, "%s", register_name);
		} break;
		case STACK_PTX_INSTRUCTION_TYPE_CONSTANT:
			result = _stack_ptx_emit_latex_constant(latex_compiler, instruction, writer);
			break;
		case STACK_PTX_INSTRUCTION_TYPE_SPECIAL:
			result = _stack_ptx_emit_latex_special(latex_compiler, instruction, writer);
			break;
		case STACK_PTX_INSTRUCTION_TYPE_PTX:
			result = _stack_ptx_emit_latex_render_ptx(
				latex_compiler,
				ast_idx,
				writer,
				stack_ptx_result_out
			);
			break;
		default:
			return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
	}
	if (result != STACK_PTX_EMIT_SUCCESS) {
		return result;
	}

	if (wrap) {
		result = _stack_ptx_emit_writer_append(writer, "\\right)");
		if (result != STACK_PTX_EMIT_SUCCESS) {
			return result;
		}
	}

	return STACK_PTX_EMIT_SUCCESS;
}

static
inline
StackPtxEmitResult
_stack_ptx_emit_latex_render_request_expr(
	const StackPtxEmitLatexCompiler* latex_compiler,
	size_t request_number,
	StackPtxEmitWriter* writer,
	StackPtxResult* stack_ptx_result_out
) {
	if (latex_compiler == NULL || writer == NULL) {
		return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
	}
	if (request_number >= latex_compiler->ast_view.num_requests) {
		return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
	}

	memcpy(
		latex_compiler->ast_view.compiler->request_stack_ptrs,
		latex_compiler->ast_view.compiler->stack_ptrs,
		latex_compiler->ast_view.compiler->stack_info.num_stacks * sizeof(StackPtxStackPtr)
	);

	size_t request_idx = latex_compiler->ast_view.requests[request_number];
	if (request_idx >= latex_compiler->ast_view.compiler->num_registers) {
		if (stack_ptx_result_out != NULL) {
			*stack_ptx_result_out = STACK_PTX_ERROR_REGISTER_IDX_OUT_OF_BOUNDS;
		}
		return STACK_PTX_EMIT_ERROR_STACK_PTX;
	}

	StackPtxStackIdx stack_idx = latex_compiler->ast_view.compiler->registers[request_idx].stack_idx;
	StackPtxResult stack_result = _stack_ptx_check_stack_type_range(
		latex_compiler->ast_view.compiler,
		stack_idx
	);
	if (stack_ptx_result_out != NULL) {
		*stack_ptx_result_out = stack_result;
	}
	if (stack_result != STACK_PTX_SUCCESS) {
		return STACK_PTX_EMIT_ERROR_STACK_PTX;
	}
	if (latex_compiler->ast_view.compiler->request_stack_ptrs[stack_idx] == 0) {
		return STACK_PTX_EMIT_ERROR_STACK_PTX;
	}

	StackPtxStackPtr stack_ptr = --latex_compiler->ast_view.compiler->request_stack_ptrs[stack_idx];
	StackPtxAstIdx ast_idx = latex_compiler->ast_view.compiler->stacks[
		stack_idx * latex_compiler->ast_view.compiler->compiler_info.stack_size + stack_ptr
	];
	return _stack_ptx_emit_latex_render_ast(
		latex_compiler,
		ast_idx,
		STACK_PTX_EMIT_LATEX_PRECEDENCE_LOWEST,
		false,
		writer,
		stack_ptx_result_out
	);
}

static
inline
StackPtxEmitResult
_stack_ptx_emit_latex_validate(
	const StackPtxEmitLatexSettings* settings_ref,
	const StackPtxRegister* registers,
	size_t num_registers,
	const size_t* requests,
	size_t num_requests,
	StackPtxResult* stack_ptx_result_out
) {
	if (settings_ref == NULL || registers == NULL || requests == NULL || num_requests == 0) {
		if (stack_ptx_result_out != NULL) {
			*stack_ptx_result_out = STACK_PTX_SUCCESS;
		}
		return STACK_PTX_EMIT_ERROR_BACKEND_NOT_CONFIGURED;
	}

	for (size_t i = 0; i < num_requests; i++) {
		if (requests[i] >= num_registers) {
			if (stack_ptx_result_out != NULL) {
				*stack_ptx_result_out = STACK_PTX_ERROR_REGISTER_IDX_OUT_OF_BOUNDS;
			}
			return STACK_PTX_EMIT_ERROR_STACK_PTX;
		}
	}

	if (stack_ptx_result_out != NULL) {
		*stack_ptx_result_out = STACK_PTX_SUCCESS;
	}
	return STACK_PTX_EMIT_SUCCESS;
}

static
inline
StackPtxEmitResult
_stack_ptx_emit_latex_document_prefix(
	const StackPtxEmitLatexSettings* settings_ref,
	StackPtxEmitWriter* writer
) {
	StackPtxEmitResult result = _stack_ptx_emit_writer_append(
		writer,
		"\\documentclass[11pt]{article}\n"
		"\\usepackage[margin=1in]{geometry}\n"
		"\\usepackage{amsmath}\n"
		"\\usepackage{amssymb}\n"
		"\\usepackage[T1]{fontenc}\n"
		"\\usepackage{lmodern}\n"
		"\\pagestyle{plain}\n"
	);
	if (result != STACK_PTX_EMIT_SUCCESS) {
		return result;
	}
	if (settings_ref->document_title != NULL && settings_ref->document_title[0] != '\0') {
		result = _stack_ptx_emit_writer_append(writer, "\\title{");
		if (result != STACK_PTX_EMIT_SUCCESS) {
			return result;
		}
		result = _stack_ptx_emit_latex_append_escaped_text(writer, settings_ref->document_title);
		if (result != STACK_PTX_EMIT_SUCCESS) {
			return result;
		}
		result = _stack_ptx_emit_writer_append(writer, "}\n\\date{}\n\\begin{document}\n\\maketitle\n");
		if (result != STACK_PTX_EMIT_SUCCESS) {
			return result;
		}
		return STACK_PTX_EMIT_SUCCESS;
	}
	return _stack_ptx_emit_writer_append(writer, "\\begin{document}\n");
}

STACK_PTX_EMIT_PUBLIC_DEF
StackPtxEmitResult
stack_ptx_emit_latex_compile(
	const StackPtxCompilerInfo* compiler_info_ref,
	const StackPtxStackInfo* stack_info_ref,
	const StackPtxEmitLatexSettings* settings_ref,
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

	StackPtxEmitLatexSettings default_settings = {
		.document_title = NULL,
		.output_symbol = "y",
		.special_register_exprs = NULL,
		.num_special_register_exprs = 0,
		.emit_document = false,
		.use_input_indices = true
	};
	const StackPtxEmitLatexSettings* actual_settings =
		settings_ref != NULL ? settings_ref : &default_settings;

	StackPtxEmitResult result = _stack_ptx_emit_latex_validate(
		actual_settings,
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
		.settings = actual_settings
	};

	if (actual_settings->emit_document) {
		result = _stack_ptx_emit_latex_document_prefix(actual_settings, &writer);
		if (result != STACK_PTX_EMIT_SUCCESS) {
			*buffer_bytes_written_ret = writer.bytes_written;
			return result;
		}
	}

	const char* output_symbol =
		(actual_settings->output_symbol != NULL && actual_settings->output_symbol[0] != '\0')
			? actual_settings->output_symbol
			: "y";

	memcpy(
		latex_compiler.ast_view.compiler->request_stack_ptrs,
		latex_compiler.ast_view.compiler->stack_ptrs,
		latex_compiler.ast_view.compiler->stack_info.num_stacks * sizeof(StackPtxStackPtr)
	);

	for (size_t i = 0; i < num_requests; i++) {
		result = _stack_ptx_emit_writer_append(&writer, "\\[\n");
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
		if (result != STACK_PTX_EMIT_SUCCESS) {
			*buffer_bytes_written_ret = writer.bytes_written;
			return result;
		}

		result = _stack_ptx_emit_writer_append(&writer, " = ");
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
		if (result != STACK_PTX_EMIT_SUCCESS) {
			*buffer_bytes_written_ret = writer.bytes_written;
			return result;
		}

		result = _stack_ptx_emit_writer_append(&writer, "\n\\]\n");
		if (result != STACK_PTX_EMIT_SUCCESS) {
			*buffer_bytes_written_ret = writer.bytes_written;
			return result;
		}
	}

	if (actual_settings->emit_document) {
		result = _stack_ptx_emit_writer_append(&writer, "\\end{document}\n");
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
