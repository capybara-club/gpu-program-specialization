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

#ifndef STACK_PTX_EMIT_RUST_H_INCLUDE
#define STACK_PTX_EMIT_RUST_H_INCLUDE

#ifdef STACK_PTX_EMIT_RUST_IMPLEMENTATION
#define STACK_PTX_EMIT_BASE_IMPLEMENTATION
#endif

#include <stack_ptx_emit_base.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	const char* function_name;
	const char* function_qualifiers;
	const char* extra_params;
	const char* const* stack_type_names;
	size_t num_stack_type_names;
	const StackPtxEmitInstructionInfo* instruction_info;
	size_t num_instruction_info;
	const char* const* special_register_exprs;
	size_t num_special_register_exprs;
} StackPtxEmitRustSettings;

STACK_PTX_EMIT_PUBLIC_DEC
StackPtxEmitResult
stack_ptx_emit_rust_compile(
	const StackPtxCompilerInfo* compiler_info_ref,
	const StackPtxStackInfo* stack_info_ref,
	const StackPtxEmitRustSettings* settings_ref,
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

#endif // STACK_PTX_EMIT_RUST_H_INCLUDE

#ifdef STACK_PTX_EMIT_RUST_IMPLEMENTATION
#ifndef STACK_PTX_EMIT_RUST_IMPLEMENTATION_ONCE
#define STACK_PTX_EMIT_RUST_IMPLEMENTATION_ONCE

#include <math.h>
#include <stdint.h>
#include <stdio.h>

typedef struct {
	StackPtxEmitAstView ast_view;
	const StackPtxEmitRustSettings* settings;
	StackPtxEmitOutputStyle output_style;
} StackPtxEmitRustCompiler;

static
inline
StackPtxEmitResult
_stack_ptx_emit_rust_stack_type_name(
	const StackPtxEmitRustCompiler* rust_compiler,
	StackPtxStackIdx stack_idx,
	const char** type_name_out
) {
	if (type_name_out == NULL) {
		return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
	}
	*type_name_out = NULL;
	if (stack_idx >= rust_compiler->settings->num_stack_type_names) {
		return STACK_PTX_EMIT_ERROR_MISSING_STACK_TYPE;
	}
	const char* type_name = rust_compiler->settings->stack_type_names[stack_idx];
	if (type_name == NULL || type_name[0] == '\0') {
		return STACK_PTX_EMIT_ERROR_MISSING_STACK_TYPE;
	}
	*type_name_out = type_name;
	return STACK_PTX_EMIT_SUCCESS;
}

static
inline
StackPtxEmitResult
_stack_ptx_emit_rust_check_scalar_arg(
	const StackPtxCompiler* compiler,
	StackPtxArgIdx arg_idx
) {
	if (compiler->stack_info.arg_type_info[arg_idx].num_vec_elems != 0) {
		return STACK_PTX_EMIT_ERROR_UNSUPPORTED_VECTOR;
	}
	return STACK_PTX_EMIT_SUCCESS;
}

static
inline
StackPtxEmitResult
_stack_ptx_emit_rust_temp_name(
	StackPtxInstruction instruction,
	char* buffer,
	size_t buffer_size
) {
	if (buffer == NULL || buffer_size == 0) {
		return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
	}
	int bytes = snprintf(
		buffer,
		buffer_size,
		"f%u",
		(unsigned)instruction.payload.reg
	);
	if (bytes < 0 || (size_t)bytes >= buffer_size) {
		return STACK_PTX_EMIT_ERROR_INSUFFICIENT_BUFFER;
	}
	return STACK_PTX_EMIT_SUCCESS;
}

static
inline
StackPtxEmitResult
_stack_ptx_emit_rust_constant_expr(
	const StackPtxEmitRustCompiler* rust_compiler,
	StackPtxInstruction instruction,
	char* buffer,
	size_t buffer_size
) {
	if (rust_compiler == NULL || buffer == NULL || buffer_size == 0) {
		return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
	}

	StackPtxStackIdx stack_idx = _stack_ptx_instruction_stack_idx(instruction);
	if (stack_idx >= rust_compiler->ast_view.compiler->stack_info.num_stacks) {
		return STACK_PTX_EMIT_ERROR_STACK_PTX;
	}

	const char* literal_prefix = rust_compiler->ast_view.compiler->stack_info.stack_literal_prefixes[stack_idx];
	if (literal_prefix == NULL) {
		return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
	}

	if (strcmp(literal_prefix, "f32") == 0) {
		union {
			uint32_t u;
			float f;
		} constant = {
			.u = instruction.payload.u
		};
		if (isnan(constant.f)) {
			int bytes = snprintf(buffer, buffer_size, "f32::from_bits(0x%08Xu32)", (unsigned)instruction.payload.u);
			if (bytes < 0 || (size_t)bytes >= buffer_size) {
				return STACK_PTX_EMIT_ERROR_INSUFFICIENT_BUFFER;
			}
			return STACK_PTX_EMIT_SUCCESS;
		}
		if (isinf(constant.f)) {
			const char* inf_text = signbit(constant.f) ? "f32::NEG_INFINITY" : "f32::INFINITY";
			int bytes = snprintf(buffer, buffer_size, "%s", inf_text);
			if (bytes < 0 || (size_t)bytes >= buffer_size) {
				return STACK_PTX_EMIT_ERROR_INSUFFICIENT_BUFFER;
			}
			return STACK_PTX_EMIT_SUCCESS;
		}
		if (constant.f == 0.0f && signbit(constant.f)) {
			int bytes = snprintf(buffer, buffer_size, "(-0.0f32)");
			if (bytes < 0 || (size_t)bytes >= buffer_size) {
				return STACK_PTX_EMIT_ERROR_INSUFFICIENT_BUFFER;
			}
			return STACK_PTX_EMIT_SUCCESS;
		}
		char literal[64];
		int literal_bytes = snprintf(literal, sizeof(literal), "%.9g", (double)constant.f);
		if (literal_bytes < 0 || (size_t)literal_bytes >= sizeof(literal)) {
			return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
		}
		bool has_decimal_or_exponent = false;
		for (size_t i = 0; literal[i] != '\0'; i++) {
			if (literal[i] == '.' || literal[i] == 'e' || literal[i] == 'E') {
				has_decimal_or_exponent = true;
				break;
			}
		}
		int bytes = snprintf(buffer, buffer_size, has_decimal_or_exponent ? "%sf32" : "%s.0f32", literal);
		if (bytes < 0 || (size_t)bytes >= buffer_size) {
			return STACK_PTX_EMIT_ERROR_INSUFFICIENT_BUFFER;
		}
		return STACK_PTX_EMIT_SUCCESS;
	}

	if (strcmp(literal_prefix, "u32") == 0) {
		int bytes = snprintf(buffer, buffer_size, "0x%08Xu32", (unsigned)instruction.payload.u);
		if (bytes < 0 || (size_t)bytes >= buffer_size) {
			return STACK_PTX_EMIT_ERROR_INSUFFICIENT_BUFFER;
		}
		return STACK_PTX_EMIT_SUCCESS;
	}

	if (strcmp(literal_prefix, "s32") == 0) {
		int bytes = snprintf(
			buffer,
			buffer_size,
			"i32::from_ne_bytes(0x%08Xu32.to_ne_bytes())",
			(unsigned)instruction.payload.u
		);
		if (bytes < 0 || (size_t)bytes >= buffer_size) {
			return STACK_PTX_EMIT_ERROR_INSUFFICIENT_BUFFER;
		}
		return STACK_PTX_EMIT_SUCCESS;
	}

	if (strcmp(literal_prefix, "pred") == 0) {
		int bytes = snprintf(buffer, buffer_size, "%s", instruction.payload.u == 0 ? "false" : "true");
		if (bytes < 0 || (size_t)bytes >= buffer_size) {
			return STACK_PTX_EMIT_ERROR_INSUFFICIENT_BUFFER;
		}
		return STACK_PTX_EMIT_SUCCESS;
	}

	return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
}

static
inline
StackPtxEmitResult
_stack_ptx_emit_rust_special_expr(
	const StackPtxEmitRustCompiler* rust_compiler,
	StackPtxInstruction instruction,
	char* buffer,
	size_t buffer_size
) {
	if (rust_compiler == NULL || buffer == NULL || buffer_size == 0) {
		return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
	}
	StackPtxIdx special_register_idx = instruction.payload.u;
	if (special_register_idx >= rust_compiler->settings->num_special_register_exprs) {
		return STACK_PTX_EMIT_ERROR_MISSING_SPECIAL_REGISTER_MAPPING;
	}
	const char* expr = rust_compiler->settings->special_register_exprs[special_register_idx];
	if (expr == NULL || expr[0] == '\0') {
		return STACK_PTX_EMIT_ERROR_MISSING_SPECIAL_REGISTER_MAPPING;
	}
	int bytes = snprintf(buffer, buffer_size, "%s", expr);
	if (bytes < 0 || (size_t)bytes >= buffer_size) {
		return STACK_PTX_EMIT_ERROR_INSUFFICIENT_BUFFER;
	}
	return STACK_PTX_EMIT_SUCCESS;
}

static
inline
StackPtxEmitResult
_stack_ptx_emit_rust_value_expr(
	const StackPtxEmitRustCompiler* rust_compiler,
	StackPtxInstruction instruction,
	char* buffer,
	size_t buffer_size
) {
	if (rust_compiler == NULL || buffer == NULL || buffer_size == 0) {
		return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
	}

	switch ((StackPtxInstructionType)instruction.instruction_type) {
		case STACK_PTX_INSTRUCTION_TYPE_INPUT: {
			StackPtxIdx register_idx = instruction.payload.u;
			if (register_idx >= rust_compiler->ast_view.compiler->num_registers) {
				return STACK_PTX_EMIT_ERROR_STACK_PTX;
			}
			const char* name = rust_compiler->ast_view.compiler->registers[register_idx].name;
			if (!_stack_ptx_emit_is_identifier(name)) {
				return STACK_PTX_EMIT_ERROR_INVALID_IDENTIFIER;
			}
			bool is_output = _stack_ptx_emit_request_contains(
				rust_compiler->ast_view.requests,
				rust_compiler->ast_view.num_requests,
				register_idx
			);
			if (is_output && rust_compiler->output_style == STACK_PTX_EMIT_OUTPUT_STYLE_RUST_MUT) {
				int bytes = snprintf(buffer, buffer_size, "(*%s)", name);
				if (bytes < 0 || (size_t)bytes >= buffer_size) {
					return STACK_PTX_EMIT_ERROR_INSUFFICIENT_BUFFER;
				}
				return STACK_PTX_EMIT_SUCCESS;
			}
			int bytes = snprintf(buffer, buffer_size, "%s", name);
			if (bytes < 0 || (size_t)bytes >= buffer_size) {
				return STACK_PTX_EMIT_ERROR_INSUFFICIENT_BUFFER;
			}
			return STACK_PTX_EMIT_SUCCESS;
		}
		case STACK_PTX_INSTRUCTION_TYPE_REGISTER:
			return _stack_ptx_emit_rust_temp_name(instruction, buffer, buffer_size);
		case STACK_PTX_INSTRUCTION_TYPE_CONSTANT:
			return _stack_ptx_emit_rust_constant_expr(rust_compiler, instruction, buffer, buffer_size);
		case STACK_PTX_INSTRUCTION_TYPE_SPECIAL:
			return _stack_ptx_emit_rust_special_expr(rust_compiler, instruction, buffer, buffer_size);
		default:
			return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
	}
}

static
inline
StackPtxEmitResult
_stack_ptx_emit_rust_rhs(
	const StackPtxEmitInstructionInfo* instruction_info,
	const char* const* arg_exprs,
	size_t num_args,
	char* buffer,
	size_t buffer_size
) {
	if (instruction_info == NULL || buffer == NULL || buffer_size == 0) {
		return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
	}

	switch (instruction_info->kind) {
		case STACK_PTX_EMIT_OP_KIND_INVALID:
			return STACK_PTX_EMIT_ERROR_MISSING_INSTRUCTION_MAPPING;
		case STACK_PTX_EMIT_OP_KIND_PASSTHROUGH: {
			if (num_args != 1) {
				return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
			}
			int bytes = snprintf(buffer, buffer_size, "%s", arg_exprs[0]);
			if (bytes < 0 || (size_t)bytes >= buffer_size) {
				return STACK_PTX_EMIT_ERROR_INSUFFICIENT_BUFFER;
			}
		} break;
		case STACK_PTX_EMIT_OP_KIND_INFIX: {
			if (instruction_info->text == NULL || num_args != 2) {
				return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
			}
			int bytes = snprintf(buffer, buffer_size, "(%s %s %s)", arg_exprs[0], instruction_info->text, arg_exprs[1]);
			if (bytes < 0 || (size_t)bytes >= buffer_size) {
				return STACK_PTX_EMIT_ERROR_INSUFFICIENT_BUFFER;
			}
		} break;
		case STACK_PTX_EMIT_OP_KIND_PREFIX: {
			if (instruction_info->text == NULL || num_args != 1) {
				return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
			}
			int bytes = snprintf(buffer, buffer_size, "(%s%s)", instruction_info->text, arg_exprs[0]);
			if (bytes < 0 || (size_t)bytes >= buffer_size) {
				return STACK_PTX_EMIT_ERROR_INSUFFICIENT_BUFFER;
			}
		} break;
		case STACK_PTX_EMIT_OP_KIND_FUNC: {
			if (instruction_info->text == NULL || num_args == 0) {
				return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
			}
			StackPtxEmitWriter writer = {
				.buffer = buffer,
				.buffer_size = buffer_size,
				.bytes_written = 0
			};
			StackPtxEmitResult result = _stack_ptx_emit_writer_append(&writer, "%s(", instruction_info->text);
			if (result != STACK_PTX_EMIT_SUCCESS) {
				return result;
			}
			for (size_t i = 0; i < num_args; i++) {
				result = _stack_ptx_emit_writer_append(&writer, i == 0 ? "%s" : ", %s", arg_exprs[i]);
				if (result != STACK_PTX_EMIT_SUCCESS) {
					return result;
				}
			}
			result = _stack_ptx_emit_writer_append(&writer, ")");
			if (result != STACK_PTX_EMIT_SUCCESS) {
				return result;
			}
		} break;
		case STACK_PTX_EMIT_OP_KIND_CAST: {
			if (instruction_info->text == NULL || num_args != 1) {
				return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
			}
			int bytes = snprintf(buffer, buffer_size, "((%s) as %s)", arg_exprs[0], instruction_info->text);
			if (bytes < 0 || (size_t)bytes >= buffer_size) {
				return STACK_PTX_EMIT_ERROR_INSUFFICIENT_BUFFER;
			}
		} break;
		case STACK_PTX_EMIT_OP_KIND_SELECT: {
			if (num_args != 3) {
				return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
			}
			int bytes = snprintf(buffer, buffer_size, "(if %s { %s } else { %s })", arg_exprs[2], arg_exprs[0], arg_exprs[1]);
			if (bytes < 0 || (size_t)bytes >= buffer_size) {
				return STACK_PTX_EMIT_ERROR_INSUFFICIENT_BUFFER;
			}
		} break;
		case STACK_PTX_EMIT_OP_KIND_MULADD: {
			if (num_args != 3) {
				return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
			}
			int bytes = snprintf(buffer, buffer_size, "((%s * %s) + %s)", arg_exprs[0], arg_exprs[1], arg_exprs[2]);
			if (bytes < 0 || (size_t)bytes >= buffer_size) {
				return STACK_PTX_EMIT_ERROR_INSUFFICIENT_BUFFER;
			}
		} break;
		case STACK_PTX_EMIT_OP_KIND_RECIPROCAL: {
			if (num_args != 1) {
				return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
			}
			int bytes = snprintf(buffer, buffer_size, "(1.0f32 / (%s))", arg_exprs[0]);
			if (bytes < 0 || (size_t)bytes >= buffer_size) {
				return STACK_PTX_EMIT_ERROR_INSUFFICIENT_BUFFER;
			}
		} break;
		case STACK_PTX_EMIT_OP_KIND_RSQRT: {
			if (num_args != 1) {
				return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
			}
			int bytes = snprintf(buffer, buffer_size, "(1.0f32 / f32::sqrt(%s))", arg_exprs[0]);
			if (bytes < 0 || (size_t)bytes >= buffer_size) {
				return STACK_PTX_EMIT_ERROR_INSUFFICIENT_BUFFER;
			}
		} break;
		case STACK_PTX_EMIT_OP_KIND_NUM_ENUMS:
			return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
	}
	return STACK_PTX_EMIT_SUCCESS;
}

static
inline
StackPtxEmitResult
_stack_ptx_emit_rust_local_assignment(
	StackPtxEmitWriter* writer,
	const char* indent,
	const char* type_name,
	const char* local_name,
	const char* rhs
) {
	if (writer == NULL || indent == NULL || type_name == NULL || local_name == NULL || rhs == NULL) {
		return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
	}
	return _stack_ptx_emit_writer_append(writer, "%slet %s: %s = %s;\n", indent, local_name, type_name, rhs);
}

static
inline
StackPtxEmitResult
_stack_ptx_emit_rust_output_assignment(
	StackPtxEmitWriter* writer,
	StackPtxEmitOutputStyle output_style,
	const char* indent,
	const char* output_name,
	const char* rhs
) {
	if (writer == NULL || indent == NULL || output_name == NULL || rhs == NULL) {
		return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
	}
	if (output_style == STACK_PTX_EMIT_OUTPUT_STYLE_PLAIN) {
		return _stack_ptx_emit_writer_append(writer, "%s%s = %s;\n", indent, output_name, rhs);
	}
	return _stack_ptx_emit_writer_append(writer, "%s*%s = %s;\n", indent, output_name, rhs);
}

static
inline
StackPtxEmitResult
_stack_ptx_emit_rust_special(
	StackPtxEmitRustCompiler* rust_compiler,
	StackPtxInstruction instruction,
	StackPtxAstIdx ast_idx,
	const char* indent,
	StackPtxEmitWriter* writer,
	StackPtxResult* stack_ptx_result_out
) {
	const StackPtxSpecialRegisterDescriptor* descriptor = NULL;
	StackPtxResult stack_result = _stack_ptx_get_special_register_descriptor(
		rust_compiler->ast_view.compiler,
		instruction,
		NULL,
		&descriptor
	);
	if (stack_ptx_result_out != NULL) {
		*stack_ptx_result_out = stack_result;
	}
	if (stack_result != STACK_PTX_SUCCESS) {
		return STACK_PTX_EMIT_ERROR_STACK_PTX;
	}

	StackPtxArgIdx arg_idx = descriptor->arg_type_idx;
	StackPtxEmitResult result = _stack_ptx_emit_rust_check_scalar_arg(rust_compiler->ast_view.compiler, arg_idx);
	if (result != STACK_PTX_EMIT_SUCCESS) {
		return result;
	}

	StackPtxStackIdx stack_idx = rust_compiler->ast_view.compiler->stack_info.arg_type_info[arg_idx].stack_idx;
	const char* type_name = NULL;
	result = _stack_ptx_emit_rust_stack_type_name(rust_compiler, stack_idx, &type_name);
	if (result != STACK_PTX_EMIT_SUCCESS) {
		return result;
	}

	StackPtxInstruction register_instruction = {};
	register_instruction.instruction_type = STACK_PTX_INSTRUCTION_TYPE_REGISTER;
	_stack_ptx_instruction_set_stack_idx(&register_instruction, stack_idx);
	register_instruction.payload.reg = rust_compiler->ast_view.compiler->register_counters[0]++;

	char register_name[128];
	result = _stack_ptx_emit_rust_temp_name(register_instruction, register_name, sizeof(register_name));
	if (result != STACK_PTX_EMIT_SUCCESS) {
		return result;
	}

	char expr_buffer[STACK_PTX_EMIT_MAX_EXPR_BYTES];
	result = _stack_ptx_emit_rust_special_expr(rust_compiler, instruction, expr_buffer, sizeof(expr_buffer));
	if (result != STACK_PTX_EMIT_SUCCESS) {
		return result;
	}

	result = _stack_ptx_emit_rust_local_assignment(writer, indent, type_name, register_name, expr_buffer);
	if (result != STACK_PTX_EMIT_SUCCESS) {
		return result;
	}

	rust_compiler->ast_view.compiler->ast[ast_idx] = register_instruction;
	return STACK_PTX_EMIT_SUCCESS;
}

static
inline
StackPtxEmitResult
_stack_ptx_emit_rust_ptx(
	StackPtxEmitRustCompiler* rust_compiler,
	StackPtxInstruction instruction,
	StackPtxAstIdx ast_idx,
	const char* indent,
	StackPtxEmitWriter* writer,
	StackPtxResult* stack_ptx_result_out
) {
	const size_t ret_start_idx = ast_idx + _stack_ptx_instruction_ret_idx(instruction);
	instruction = rust_compiler->ast_view.compiler->ast[ret_start_idx];

	StackPtxIdx instruction_idx = 0;
	const StackPtxPtxInstructionDescriptor* descriptor = NULL;
	StackPtxResult stack_result = _stack_ptx_get_ptx_instruction_descriptor(
		rust_compiler->ast_view.compiler,
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
		rust_compiler->ast_view.compiler,
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
		rust_compiler->ast_view.compiler,
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

	for (size_t i = 0; i < num_args; i++) {
		StackPtxEmitResult result = _stack_ptx_emit_rust_check_scalar_arg(
			rust_compiler->ast_view.compiler,
			descriptor->arg_types[i]
		);
		if (result != STACK_PTX_EMIT_SUCCESS) {
			return result;
		}
	}
	for (size_t i = 0; i < num_rets; i++) {
		StackPtxEmitResult result = _stack_ptx_emit_rust_check_scalar_arg(
			rust_compiler->ast_view.compiler,
			descriptor->ret_types[i]
		);
		if (result != STACK_PTX_EMIT_SUCCESS) {
			return result;
		}
	}

	const size_t args_start_idx = ret_start_idx - num_rets_flat;
	bool all_evaluated = true;
	for (size_t i = 0; i < num_args_flat; i++) {
		size_t args_idx = args_start_idx - i;
		StackPtxInstruction arg_instruction = rust_compiler->ast_view.compiler->ast[args_idx];
		if (arg_instruction.instruction_type != STACK_PTX_INSTRUCTION_TYPE_AST_IDX) {
			return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
		}
		StackPtxAstIdx arg_ast_idx = arg_instruction.payload.ast_idx;
		arg_instruction = rust_compiler->ast_view.compiler->ast[arg_ast_idx];
		switch ((StackPtxInstructionType)arg_instruction.instruction_type) {
			case STACK_PTX_INSTRUCTION_TYPE_CONSTANT:
			case STACK_PTX_INSTRUCTION_TYPE_REGISTER:
			case STACK_PTX_INSTRUCTION_TYPE_INPUT:
				break;
			case STACK_PTX_INSTRUCTION_TYPE_SPECIAL:
			case STACK_PTX_INSTRUCTION_TYPE_PTX: {
				if (all_evaluated) {
					all_evaluated = false;
					if (rust_compiler->ast_view.compiler->ast_to_visit_stack_ptr >=
						rust_compiler->ast_view.compiler->compiler_info.max_ast_to_visit_stack_depth) {
						if (stack_ptx_result_out != NULL) {
							*stack_ptx_result_out = STACK_PTX_ERROR_INSUFFICIENT_AST_VISIT_SIZE;
						}
						return STACK_PTX_EMIT_ERROR_STACK_PTX;
					}
					rust_compiler->ast_view.compiler->ast_to_visit_stack[
						rust_compiler->ast_view.compiler->ast_to_visit_stack_ptr++
					] = ast_idx;
				}
				if (rust_compiler->ast_view.compiler->ast_to_visit_stack_ptr >=
					rust_compiler->ast_view.compiler->compiler_info.max_ast_to_visit_stack_depth) {
					if (stack_ptx_result_out != NULL) {
						*stack_ptx_result_out = STACK_PTX_ERROR_INSUFFICIENT_AST_VISIT_SIZE;
					}
					return STACK_PTX_EMIT_ERROR_STACK_PTX;
				}
				rust_compiler->ast_view.compiler->ast_to_visit_stack[
					rust_compiler->ast_view.compiler->ast_to_visit_stack_ptr++
				] = arg_ast_idx;
			} break;
			default:
				return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
		}
	}

	if (!all_evaluated) {
		return STACK_PTX_EMIT_SUCCESS;
	}

	if (instruction_idx >= rust_compiler->settings->num_instruction_info) {
		return STACK_PTX_EMIT_ERROR_MISSING_INSTRUCTION_MAPPING;
	}
	const StackPtxEmitInstructionInfo* instruction_info =
		&rust_compiler->settings->instruction_info[instruction_idx];
	if (instruction_info->kind == STACK_PTX_EMIT_OP_KIND_INVALID) {
		return STACK_PTX_EMIT_ERROR_MISSING_INSTRUCTION_MAPPING;
	}

	StackPtxArgIdx ret_arg_idx = descriptor->ret_types[0];
	StackPtxStackIdx ret_stack_idx = rust_compiler->ast_view.compiler->stack_info.arg_type_info[ret_arg_idx].stack_idx;
	const char* ret_type_name = NULL;
	StackPtxEmitResult result = _stack_ptx_emit_rust_stack_type_name(rust_compiler, ret_stack_idx, &ret_type_name);
	if (result != STACK_PTX_EMIT_SUCCESS) {
		return result;
	}

	const char* arg_exprs[STACK_PTX_MAX_NUM_PTX_ARGS];
	char arg_buffers[STACK_PTX_MAX_NUM_PTX_ARGS][STACK_PTX_EMIT_MAX_EXPR_BYTES];
	size_t arg_ast_idx = 0;
	for (size_t i = 0; i < num_args; i++) {
		StackPtxInstruction arg_instruction =
			rust_compiler->ast_view.compiler->ast[args_start_idx - num_args_flat + 1 + arg_ast_idx];
		if (arg_instruction.instruction_type != STACK_PTX_INSTRUCTION_TYPE_AST_IDX) {
			return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
		}
		arg_instruction = rust_compiler->ast_view.compiler->ast[arg_instruction.payload.ast_idx];
		result = _stack_ptx_emit_rust_value_expr(rust_compiler, arg_instruction, arg_buffers[i], sizeof(arg_buffers[i]));
		if (result != STACK_PTX_EMIT_SUCCESS) {
			return result;
		}
		arg_exprs[i] = arg_buffers[i];
		arg_ast_idx++;
	}

	char rhs_buffer[STACK_PTX_EMIT_MAX_EXPR_BYTES];
	result = _stack_ptx_emit_rust_rhs(instruction_info, arg_exprs, num_args, rhs_buffer, sizeof(rhs_buffer));
	if (result != STACK_PTX_EMIT_SUCCESS) {
		return result;
	}

	StackPtxInstruction register_instruction = {};
	register_instruction.instruction_type = STACK_PTX_INSTRUCTION_TYPE_REGISTER;
	_stack_ptx_instruction_set_stack_idx(&register_instruction, ret_stack_idx);
	register_instruction.payload.reg = rust_compiler->ast_view.compiler->register_counters[0]++;

	char register_name[128];
	result = _stack_ptx_emit_rust_temp_name(register_instruction, register_name, sizeof(register_name));
	if (result != STACK_PTX_EMIT_SUCCESS) {
		return result;
	}

	result = _stack_ptx_emit_rust_local_assignment(writer, indent, ret_type_name, register_name, rhs_buffer);
	if (result != STACK_PTX_EMIT_SUCCESS) {
		return result;
	}

	rust_compiler->ast_view.compiler->ast[ret_start_idx] = register_instruction;
	return STACK_PTX_EMIT_SUCCESS;
}

static
inline
StackPtxEmitResult
_stack_ptx_emit_rust_outputs(
	StackPtxEmitRustCompiler* rust_compiler,
	const char* indent,
	StackPtxEmitWriter* writer,
	StackPtxResult* stack_ptx_result_out
) {
	for (size_t i = 0; i < rust_compiler->ast_view.num_requests; i++) {
		size_t request_idx = rust_compiler->ast_view.requests[i];
		if (request_idx >= rust_compiler->ast_view.compiler->num_registers) {
			if (stack_ptx_result_out != NULL) {
				*stack_ptx_result_out = STACK_PTX_ERROR_REGISTER_IDX_OUT_OF_BOUNDS;
			}
			return STACK_PTX_EMIT_ERROR_STACK_PTX;
		}

		const StackPtxRegister* request_register = &rust_compiler->ast_view.compiler->registers[request_idx];
		const char* request_name = request_register->name;
		if (!_stack_ptx_emit_is_identifier(request_name)) {
			return STACK_PTX_EMIT_ERROR_INVALID_IDENTIFIER;
		}

		StackPtxStackIdx stack_idx = request_register->stack_idx;
		StackPtxResult stack_result = _stack_ptx_check_stack_type_range(rust_compiler->ast_view.compiler, stack_idx);
		if (stack_ptx_result_out != NULL) {
			*stack_ptx_result_out = stack_result;
		}
		if (stack_result != STACK_PTX_SUCCESS) {
			return STACK_PTX_EMIT_ERROR_STACK_PTX;
		}
		if (rust_compiler->ast_view.compiler->stack_ptrs[stack_idx] == 0) {
			continue;
		}

		StackPtxStackPtr stack_ptr = --rust_compiler->ast_view.compiler->stack_ptrs[stack_idx];
		StackPtxAstIdx ast_idx = rust_compiler->ast_view.compiler->stacks[
			stack_idx * rust_compiler->ast_view.compiler->compiler_info.stack_size + stack_ptr
		];
		StackPtxInstruction value_instruction = rust_compiler->ast_view.compiler->ast[ast_idx];

		char expr_buffer[STACK_PTX_EMIT_MAX_EXPR_BYTES];
		StackPtxEmitResult result = _stack_ptx_emit_rust_value_expr(rust_compiler, value_instruction, expr_buffer, sizeof(expr_buffer));
		if (result != STACK_PTX_EMIT_SUCCESS) {
			return result;
		}

		result = _stack_ptx_emit_rust_output_assignment(
			writer,
			rust_compiler->output_style,
			indent,
			request_name,
			expr_buffer
		);
		if (result != STACK_PTX_EMIT_SUCCESS) {
			return result;
		}
	}
	return STACK_PTX_EMIT_SUCCESS;
}

static
inline
StackPtxEmitResult
_stack_ptx_emit_rust_body(
	StackPtxEmitRustCompiler* rust_compiler,
	const char* indent,
	StackPtxEmitWriter* writer,
	StackPtxResult* stack_ptx_result_out
) {
	memcpy(
		rust_compiler->ast_view.compiler->request_stack_ptrs,
		rust_compiler->ast_view.compiler->stack_ptrs,
		rust_compiler->ast_view.compiler->stack_info.num_stacks * sizeof(StackPtxStackPtr)
	);

	for (size_t i = 0; i < rust_compiler->ast_view.num_requests; i++) {
		if (rust_compiler->ast_view.requests[i] >= rust_compiler->ast_view.compiler->num_registers) {
			if (stack_ptx_result_out != NULL) {
				*stack_ptx_result_out = STACK_PTX_ERROR_REGISTER_IDX_OUT_OF_BOUNDS;
			}
			return STACK_PTX_EMIT_ERROR_STACK_PTX;
		}

		StackPtxStackIdx stack_idx =
			rust_compiler->ast_view.compiler->registers[rust_compiler->ast_view.requests[i]].stack_idx;
		StackPtxResult stack_result = _stack_ptx_check_stack_type_range(rust_compiler->ast_view.compiler, stack_idx);
		if (stack_ptx_result_out != NULL) {
			*stack_ptx_result_out = stack_result;
		}
		if (stack_result != STACK_PTX_SUCCESS) {
			return STACK_PTX_EMIT_ERROR_STACK_PTX;
		}
		if (rust_compiler->ast_view.compiler->request_stack_ptrs[stack_idx] > 0) {
			StackPtxStackPtr stack_ptr = --rust_compiler->ast_view.compiler->request_stack_ptrs[stack_idx];
			StackPtxAstIdx ast_idx = rust_compiler->ast_view.compiler->stacks[
				stack_idx * rust_compiler->ast_view.compiler->compiler_info.stack_size + stack_ptr
			];
			if (rust_compiler->ast_view.compiler->ast_to_visit_stack_ptr >=
				rust_compiler->ast_view.compiler->compiler_info.max_ast_to_visit_stack_depth) {
				if (stack_ptx_result_out != NULL) {
					*stack_ptx_result_out = STACK_PTX_ERROR_INSUFFICIENT_AST_VISIT_SIZE;
				}
				return STACK_PTX_EMIT_ERROR_STACK_PTX;
			}
			rust_compiler->ast_view.compiler->ast_to_visit_stack[
				rust_compiler->ast_view.compiler->ast_to_visit_stack_ptr++
			] = ast_idx;
		}
	}

	while (rust_compiler->ast_view.compiler->ast_to_visit_stack_ptr != 0) {
		StackPtxAstIdx ast_idx =
			rust_compiler->ast_view.compiler->ast_to_visit_stack[--rust_compiler->ast_view.compiler->ast_to_visit_stack_ptr];
		StackPtxInstruction instruction = rust_compiler->ast_view.compiler->ast[ast_idx];
		switch ((StackPtxInstructionType)instruction.instruction_type) {
			case STACK_PTX_INSTRUCTION_TYPE_CONSTANT:
			case STACK_PTX_INSTRUCTION_TYPE_REGISTER:
			case STACK_PTX_INSTRUCTION_TYPE_INPUT:
				continue;
			case STACK_PTX_INSTRUCTION_TYPE_SPECIAL: {
				StackPtxEmitResult result = _stack_ptx_emit_rust_special(
					rust_compiler,
					instruction,
					ast_idx,
					indent,
					writer,
					stack_ptx_result_out
				);
				if (result != STACK_PTX_EMIT_SUCCESS) {
					return result;
				}
			} break;
			case STACK_PTX_INSTRUCTION_TYPE_PTX: {
				StackPtxEmitResult result = _stack_ptx_emit_rust_ptx(
					rust_compiler,
					instruction,
					ast_idx,
					indent,
					writer,
					stack_ptx_result_out
				);
				if (result != STACK_PTX_EMIT_SUCCESS) {
					return result;
				}
			} break;
			default:
				return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
		}
	}

	return _stack_ptx_emit_rust_outputs(rust_compiler, indent, writer, stack_ptx_result_out);
}

static
inline
StackPtxEmitResult
_stack_ptx_emit_rust_validate(
	const StackPtxEmitRustSettings* settings_ref,
	const StackPtxRegister* registers,
	size_t num_registers,
	const size_t* requests,
	size_t num_requests,
	bool emit_wrapper,
	StackPtxResult* stack_ptx_result_out
) {
	if (settings_ref == NULL ||
		settings_ref->instruction_info == NULL ||
		settings_ref->stack_type_names == NULL ||
		registers == NULL ||
		(emit_wrapper && settings_ref->function_name == NULL)) {
		if (stack_ptx_result_out != NULL) {
			*stack_ptx_result_out = STACK_PTX_SUCCESS;
		}
		return STACK_PTX_EMIT_ERROR_BACKEND_NOT_CONFIGURED;
	}

	if (emit_wrapper && !_stack_ptx_emit_is_identifier(settings_ref->function_name)) {
		if (stack_ptx_result_out != NULL) {
			*stack_ptx_result_out = STACK_PTX_SUCCESS;
		}
		return STACK_PTX_EMIT_ERROR_INVALID_IDENTIFIER;
	}

	for (size_t i = 0; i < num_requests; i++) {
		if (requests[i] >= num_registers) {
			if (stack_ptx_result_out != NULL) {
				*stack_ptx_result_out = STACK_PTX_ERROR_REGISTER_IDX_OUT_OF_BOUNDS;
			}
			return STACK_PTX_EMIT_ERROR_STACK_PTX;
		}
	}

	for (size_t i = 0; i < num_registers; i++) {
		if (!_stack_ptx_emit_is_identifier(registers[i].name)) {
			if (stack_ptx_result_out != NULL) {
				*stack_ptx_result_out = STACK_PTX_SUCCESS;
			}
			return STACK_PTX_EMIT_ERROR_INVALID_IDENTIFIER;
		}
		if (registers[i].stack_idx >= settings_ref->num_stack_type_names ||
			settings_ref->stack_type_names[registers[i].stack_idx] == NULL) {
			if (stack_ptx_result_out != NULL) {
				*stack_ptx_result_out = STACK_PTX_SUCCESS;
			}
			return STACK_PTX_EMIT_ERROR_MISSING_STACK_TYPE;
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
_stack_ptx_emit_rust_signature(
	const StackPtxEmitRustSettings* settings_ref,
	const StackPtxRegister* registers,
	size_t num_registers,
	const size_t* requests,
	size_t num_requests,
	StackPtxEmitWriter* writer
) {
	const char* qualifiers =
		(settings_ref->function_qualifiers != NULL && settings_ref->function_qualifiers[0] != '\0')
			? settings_ref->function_qualifiers
			: "pub";
	StackPtxEmitResult result = _stack_ptx_emit_writer_append(writer, "%s fn %s(", qualifiers, settings_ref->function_name);
	if (result != STACK_PTX_EMIT_SUCCESS) {
		return result;
	}

	bool emitted_param = false;
	if (settings_ref->extra_params != NULL && settings_ref->extra_params[0] != '\0') {
		result = _stack_ptx_emit_writer_append(writer, "%s", settings_ref->extra_params);
		if (result != STACK_PTX_EMIT_SUCCESS) {
			return result;
		}
		emitted_param = true;
	}

	for (size_t i = 0; i < num_registers; i++) {
		const char* type_name = settings_ref->stack_type_names[registers[i].stack_idx];
		bool is_output = _stack_ptx_emit_request_contains(requests, num_requests, i);
		result = _stack_ptx_emit_writer_append(
			writer,
			emitted_param ? ", %s: %s%s" : "%s: %s%s",
			registers[i].name,
			is_output ? "&mut " : "",
			type_name
		);
		if (result != STACK_PTX_EMIT_SUCCESS) {
			return result;
		}
		emitted_param = true;
	}

	return _stack_ptx_emit_writer_append(writer, ") {\n");
}

static
inline
StackPtxEmitResult
_stack_ptx_emit_rust_compile_mode(
	const StackPtxCompilerInfo* compiler_info_ref,
	const StackPtxStackInfo* stack_info_ref,
	const StackPtxEmitRustSettings* settings_ref,
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
	bool emit_wrapper,
	StackPtxEmitOutputStyle output_style,
	const char* indent,
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
		indent == NULL) {
		if (stack_ptx_result_out != NULL) {
			*stack_ptx_result_out = STACK_PTX_SUCCESS;
		}
		return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
	}

	StackPtxEmitResult result = _stack_ptx_emit_rust_validate(
		settings_ref,
		registers,
		num_registers,
		requests,
		num_requests,
		emit_wrapper,
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
	StackPtxEmitRustCompiler rust_compiler = {
		.ast_view = ast_view,
		.settings = settings_ref,
		.output_style = output_style
	};

	if (emit_wrapper) {
		result = _stack_ptx_emit_rust_signature(
			settings_ref,
			registers,
			num_registers,
			requests,
			num_requests,
			&writer
		);
		if (result != STACK_PTX_EMIT_SUCCESS) {
			*buffer_bytes_written_ret = writer.bytes_written;
			return result;
		}
	}

	result = _stack_ptx_emit_rust_body(&rust_compiler, indent, &writer, stack_ptx_result_out);
	if (result != STACK_PTX_EMIT_SUCCESS) {
		*buffer_bytes_written_ret = writer.bytes_written;
		return result;
	}

	if (emit_wrapper) {
		result = _stack_ptx_emit_writer_append(&writer, "}\n");
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

STACK_PTX_EMIT_PUBLIC_DEF
StackPtxEmitResult
stack_ptx_emit_rust_compile(
	const StackPtxCompilerInfo* compiler_info_ref,
	const StackPtxStackInfo* stack_info_ref,
	const StackPtxEmitRustSettings* settings_ref,
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
	return _stack_ptx_emit_rust_compile_mode(
		compiler_info_ref,
		stack_info_ref,
		settings_ref,
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
		true,
		STACK_PTX_EMIT_OUTPUT_STYLE_RUST_MUT,
		"\t",
		buffer,
		buffer_size,
		buffer_bytes_written_ret,
		stack_ptx_result_out
	);
}

#endif
#endif
