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

#ifndef STACK_PTX_EMIT_PTX_H_INCLUDE
#define STACK_PTX_EMIT_PTX_H_INCLUDE

#ifdef STACK_PTX_EMIT_PTX_IMPLEMENTATION
#define STACK_PTX_EMIT_BASE_IMPLEMENTATION
#endif

#include <stack_ptx_emit_base.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	const StackPtxRegister* registers;
	size_t num_registers;
} StackPtxEmitPtxSettings;

STACK_PTX_EMIT_PUBLIC_DEC
StackPtxEmitResult
stack_ptx_emit_ptx_compile(
	const StackPtxCompilerInfo* compiler_info_ref,
	const StackPtxStackInfo* stack_info_ref,
	const StackPtxEmitPtxSettings* settings_ref,
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

#endif // STACK_PTX_EMIT_PTX_H_INCLUDE

#ifdef STACK_PTX_EMIT_PTX_IMPLEMENTATION
#ifndef STACK_PTX_EMIT_PTX_IMPLEMENTATION_ONCE
#define STACK_PTX_EMIT_PTX_IMPLEMENTATION_ONCE

STACK_PTX_EMIT_PUBLIC_DEF
StackPtxEmitResult
stack_ptx_emit_ptx_compile(
	const StackPtxCompilerInfo* compiler_info_ref,
	const StackPtxStackInfo* stack_info_ref,
	const StackPtxEmitPtxSettings* settings_ref,
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
	if (compiler_info_ref == NULL ||
		stack_info_ref == NULL ||
		instructions == NULL ||
		registers == NULL ||
		buffer_bytes_written_ret == NULL) {
		if (stack_ptx_result_out != NULL) {
			*stack_ptx_result_out = STACK_PTX_SUCCESS;
		}
		return STACK_PTX_EMIT_ERROR_INVALID_VALUE;
	}

	const StackPtxRegister* ptx_registers = registers;
	size_t ptx_num_registers = num_registers;
	if (settings_ref != NULL && settings_ref->registers != NULL) {
		ptx_registers = settings_ref->registers;
		ptx_num_registers = settings_ref->num_registers;
	}

	StackPtxResult stack_result = stack_ptx_compile(
		compiler_info_ref,
		stack_info_ref,
		instructions,
		ptx_registers,
		ptx_num_registers,
		routines,
		num_routines,
		requests,
		num_requests,
		execution_limit,
		workspace,
		workspace_in_bytes,
		buffer,
		buffer_size,
		buffer_bytes_written_ret
	);
	if (stack_ptx_result_out != NULL) {
		*stack_ptx_result_out = stack_result;
	}
	return stack_result == STACK_PTX_SUCCESS ? STACK_PTX_EMIT_SUCCESS : STACK_PTX_EMIT_ERROR_STACK_PTX;
}

#endif
#endif
