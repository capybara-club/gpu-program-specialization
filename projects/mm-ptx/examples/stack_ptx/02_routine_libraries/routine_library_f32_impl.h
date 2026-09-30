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

// This is one of the routines we can now include as a library.
// "ROUTINE_LIBRARY_F32_1" will have a name because the user will
// add "ROUTINE_LIBRARY_F32" to a "RoutineIdx" enum.
static const StackPtxInstruction routine_library_f32_func_0[] = {
    stack_ptx_encode_constant_f32(1.0f),
    stack_ptx_encode_routine(ROUTINE_LIBRARY_F32_FUNC_1),
    stack_ptx_encode_return
};

static const StackPtxInstruction routine_library_f32_func_1[] = {
    stack_ptx_encode_constant_f32(3),
    stack_ptx_encode_constant_f32(7),
    stack_ptx_encode_ptx_instruction_mul_ftz_f32,
    stack_ptx_encode_return
};

static const StackPtxInstruction routine_library_f32_func_2[] = {
    stack_ptx_encode_constant_f32(30),
    stack_ptx_encode_constant_f32(40),
    stack_ptx_encode_ptx_instruction_add_ftz_f32,
    stack_ptx_encode_return
};
