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

#pragma once

// Requires stack_ptx_example_descriptions.hpp and RoutineIdx enum entries.

static const StackPtxInstruction routine_library_cpp_f32_func_0[] = {
    stack_ptx::encode_constant_f32(1.0f),
    stack_ptx::encode_routine(ROUTINE_LIBRARY_CPP_F32_FUNC_1),
    stack_ptx::encode_return
};

static const StackPtxInstruction routine_library_cpp_f32_func_1[] = {
    stack_ptx::encode_constant_f32(3.0f),
    stack_ptx::encode_constant_f32(7.0f),
    stack_ptx::encode_ptx_instruction_mul_ftz_f32,
    stack_ptx::encode_return
};

static const StackPtxInstruction routine_library_cpp_f32_func_2[] = {
    stack_ptx::encode_constant_f32(30.0f),
    stack_ptx::encode_constant_f32(40.0f),
    stack_ptx::encode_ptx_instruction_add_ftz_f32,
    stack_ptx::encode_return
};
