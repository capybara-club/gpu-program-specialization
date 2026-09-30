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

// This is meant to be added to a "RoutineIdx" enum.
// It will declare the names so that the main program
// can refer to these routines as well as other routines.
#define ROUTINE_LIBRARY_F32     \
    ROUTINE_LIBRARY_F32_FUNC_0, \
    ROUTINE_LIBRARY_F32_FUNC_1, \
    ROUTINE_LIBRARY_F32_FUNC_2 

// This is meant to be added to an initializer array list of routines.
// "routine_library_f32_0" is going to be an array of StackPtxInstruction 
// declared in the impl.h header file.
#define ROUTINE_LIBRARY_F32_INITIALIZERS                        \
    [ROUTINE_LIBRARY_F32_FUNC_0] = routine_library_f32_func_0,  \
    [ROUTINE_LIBRARY_F32_FUNC_1] = routine_library_f32_func_1,  \
    [ROUTINE_LIBRARY_F32_FUNC_2] = routine_library_f32_func_2
