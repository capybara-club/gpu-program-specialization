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

// Requires STACK_PTX_STACK_TYPE_PHILOX from stack_ptx_example_descriptions.h.
#define ROUTINE_LIBRARY_PHILOX_REGISTER_DECL \
    REGISTER_PHILOX_KEY_X,                   \
    REGISTER_PHILOX_KEY_Y,                   \
    REGISTER_PHILOX_CTR_X,                   \
    REGISTER_PHILOX_CTR_Y,                   \
    REGISTER_PHILOX_CTR_Z,                   \
    REGISTER_PHILOX_CTR_W

#define ROUTINE_LIBRARY_PHILOX_REGISTER_IMPL                                     \
    [REGISTER_PHILOX_KEY_X] = {.name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_PHILOX}, \
    [REGISTER_PHILOX_KEY_Y] = {.name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_PHILOX}, \
    [REGISTER_PHILOX_CTR_X] = {.name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_PHILOX}, \
    [REGISTER_PHILOX_CTR_Y] = {.name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_PHILOX}, \
    [REGISTER_PHILOX_CTR_Z] = {.name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_PHILOX}, \
    [REGISTER_PHILOX_CTR_W] = {.name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_PHILOX}

// Request at least one philox register to keep the stack alive.
#define ROUTINE_LIBRARY_PHILOX_REQUEST REGISTER_PHILOX_CTR_X
#define PHILOX_REQUEST_IMPL ROUTINE_LIBRARY_PHILOX_REQUEST

// This is meant to be added to a "RoutineIdx" enum.
#define ROUTINE_LIBRARY_PHILOX             \
    ROUTINE_LIBRARY_PHILOX_PUSH_STATE,     \
    ROUTINE_LIBRARY_PHILOX_CURAND,         \
    ROUTINE_LIBRARY_PHILOX_CURAND_UNIFORM, \
    ROUTINE_LIBRARY_PHILOX_CURAND_NORMAL,  \
    ROUTINE_LIBRARY_PHILOX_UNIFORM_SCALE,  \
    ROUTINE_LIBRARY_PHILOX_BOX_MULLER,     \
    ROUTINE_LIBRARY_PHILOX_ROUND

// This is meant to be added to a routines initializer list.
#define ROUTINE_LIBRARY_PHILOX_INITIALIZERS                                       \
    [ROUTINE_LIBRARY_PHILOX_PUSH_STATE] = routine_library_philox_push_state,      \
    [ROUTINE_LIBRARY_PHILOX_CURAND] = routine_library_philox_curand,              \
    [ROUTINE_LIBRARY_PHILOX_CURAND_UNIFORM] = routine_library_philox_curand_uniform, \
    [ROUTINE_LIBRARY_PHILOX_CURAND_NORMAL] = routine_library_philox_curand_normal, \
    [ROUTINE_LIBRARY_PHILOX_UNIFORM_SCALE] = routine_library_philox_uniform_scale, \
    [ROUTINE_LIBRARY_PHILOX_BOX_MULLER] = routine_library_philox_box_muller,      \
    [ROUTINE_LIBRARY_PHILOX_ROUND] = routine_library_philox_round
