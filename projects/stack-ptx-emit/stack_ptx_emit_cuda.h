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

#ifndef STACK_PTX_EMIT_CUDA_H_INCLUDE
#define STACK_PTX_EMIT_CUDA_H_INCLUDE

#ifdef STACK_PTX_EMIT_CUDA_IMPLEMENTATION
#define STACK_PTX_EMIT_BASE_IMPLEMENTATION
#define STACK_PTX_EMIT_C_IMPLEMENTATION
#endif

#include <stack_ptx_emit_c.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef StackPtxEmitCSettings StackPtxEmitCudaSettings;

STACK_PTX_EMIT_PUBLIC_DEC
StackPtxEmitResult
stack_ptx_emit_cuda_compile(
	const StackPtxCompilerInfo* compiler_info_ref,
	const StackPtxStackInfo* stack_info_ref,
	const StackPtxEmitCudaSettings* settings_ref,
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

#endif // STACK_PTX_EMIT_CUDA_H_INCLUDE

#ifdef STACK_PTX_EMIT_CUDA_IMPLEMENTATION
#ifndef STACK_PTX_EMIT_CUDA_IMPLEMENTATION_ONCE
#define STACK_PTX_EMIT_CUDA_IMPLEMENTATION_ONCE

static
inline
StackPtxEmitResult
_stack_ptx_emit_cuda_compile_mode(
	const StackPtxCompilerInfo* compiler_info_ref,
	const StackPtxStackInfo* stack_info_ref,
	const StackPtxEmitCudaSettings* settings_ref,
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
	return _stack_ptx_emit_c_compile_mode(
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
		emit_wrapper,
		output_style,
		indent,
		buffer,
		buffer_size,
		buffer_bytes_written_ret,
		stack_ptx_result_out
	);
}

STACK_PTX_EMIT_PUBLIC_DEF
StackPtxEmitResult
stack_ptx_emit_cuda_compile(
	const StackPtxCompilerInfo* compiler_info_ref,
	const StackPtxStackInfo* stack_info_ref,
	const StackPtxEmitCudaSettings* settings_ref,
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
	return _stack_ptx_emit_cuda_compile_mode(
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
		STACK_PTX_EMIT_OUTPUT_STYLE_POINTER,
		"\t",
		buffer,
		buffer_size,
		buffer_bytes_written_ret,
		stack_ptx_result_out
	);
}

#endif
#endif
