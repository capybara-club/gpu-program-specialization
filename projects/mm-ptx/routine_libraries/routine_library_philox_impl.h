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

#define PHILOX_W32_0                (0x9E3779B9)
#define PHILOX_W32_1                (0xBB67AE85)
#define PHILOX_M4x32_0              (0xD2511F53)
#define PHILOX_M4x32_1              (0xCD9E8D57)

#define CURAND_2POW32_INV           (2.3283064e-10f)
#define CURAND_2POW32_INV_HALF      (2.3283064e-10f/2.0f)
#define CURAND_2POW32_INV_2PI       (2.3283064e-10f * 6.2831855f)
#define CURAND_2POW32_INV_2PI_HALF  ((2.3283064e-10f * 6.2831855f)/2.0f)

#define PTX_LN2                     (0x1.62e43p-1f)

typedef enum {
    PHILOX_KEY_X_IDX = 10,
    PHILOX_KEY_Y_IDX = 11,
    PHILOX_CTR_X_IDX = 12,
    PHILOX_CTR_Y_IDX = 13,
    PHILOX_CTR_Z_IDX = 14,
    PHILOX_CTR_W_IDX = 15,
} PhiloxStoreIdx;

static const StackPtxInstruction routine_library_philox_push_state[] = {
    stack_ptx_encode_input(REGISTER_PHILOX_KEY_Y),
    stack_ptx_encode_input(REGISTER_PHILOX_KEY_X),
    stack_ptx_encode_input(REGISTER_PHILOX_CTR_W),
    stack_ptx_encode_input(REGISTER_PHILOX_CTR_Z),
    stack_ptx_encode_input(REGISTER_PHILOX_CTR_Y),
    stack_ptx_encode_input(REGISTER_PHILOX_CTR_X),
    stack_ptx_encode_return
};

static const StackPtxInstruction routine_library_philox_round[] = {
    stack_ptx_encode_store(STACK_PTX_STACK_TYPE_PHILOX, PHILOX_CTR_X_IDX),
    stack_ptx_encode_store(STACK_PTX_STACK_TYPE_PHILOX, PHILOX_CTR_Y_IDX),
    stack_ptx_encode_store(STACK_PTX_STACK_TYPE_PHILOX, PHILOX_CTR_Z_IDX),
    stack_ptx_encode_store(STACK_PTX_STACK_TYPE_PHILOX, PHILOX_CTR_W_IDX),
    stack_ptx_encode_store(STACK_PTX_STACK_TYPE_PHILOX, PHILOX_KEY_X_IDX),
    stack_ptx_encode_store(STACK_PTX_STACK_TYPE_PHILOX, PHILOX_KEY_Y_IDX),

    // key_y
    stack_ptx_encode_load(PHILOX_KEY_Y_IDX),
    stack_ptx_encode_constant_philox(PHILOX_W32_1),
    stack_ptx_encode_ptx_instruction_add_philox,

    // key_x
    stack_ptx_encode_load(PHILOX_KEY_X_IDX),
    stack_ptx_encode_constant_philox(PHILOX_W32_0),
    stack_ptx_encode_ptx_instruction_add_philox,

    // ctr_w
    stack_ptx_encode_load(PHILOX_CTR_X_IDX),
    stack_ptx_encode_constant_philox(PHILOX_M4x32_0),
    stack_ptx_encode_ptx_instruction_mul_lo_philox,

    // ctr_z
    stack_ptx_encode_load(PHILOX_CTR_X_IDX),
    stack_ptx_encode_constant_philox(PHILOX_M4x32_0),
    stack_ptx_encode_ptx_instruction_mul_hi_philox,
    stack_ptx_encode_load(PHILOX_CTR_W_IDX),
    stack_ptx_encode_ptx_instruction_xor_philox,
    stack_ptx_encode_load(PHILOX_KEY_Y_IDX),
    stack_ptx_encode_ptx_instruction_xor_philox,

    // ctr_y
    stack_ptx_encode_load(PHILOX_CTR_Z_IDX),
    stack_ptx_encode_constant_philox(PHILOX_M4x32_1),
    stack_ptx_encode_ptx_instruction_mul_lo_philox,

    // ctr_x
    stack_ptx_encode_load(PHILOX_CTR_Z_IDX),
    stack_ptx_encode_constant_philox(PHILOX_M4x32_1),
    stack_ptx_encode_ptx_instruction_mul_hi_philox,
    stack_ptx_encode_load(PHILOX_CTR_Y_IDX),
    stack_ptx_encode_ptx_instruction_xor_philox,
    stack_ptx_encode_load(PHILOX_KEY_X_IDX),
    stack_ptx_encode_ptx_instruction_xor_philox,
    stack_ptx_encode_return
};

static const StackPtxInstruction routine_library_philox_uniform_scale[] = {
    stack_ptx_encode_constant_f32(CURAND_2POW32_INV_HALF),
    stack_ptx_encode_ptx_instruction_cvt_rn_f32_u32,
    stack_ptx_encode_constant_f32(CURAND_2POW32_INV),
    stack_ptx_encode_ptx_instruction_fma_rn_ftz_f32,
    stack_ptx_encode_return
};

static const StackPtxInstruction routine_library_philox_box_muller[] = {
    stack_ptx_encode_ptx_instruction_cvt_rn_f32_u32,
    stack_ptx_encode_ptx_instruction_cvt_rn_f32_u32,

    stack_ptx_encode_store(STACK_PTX_STACK_TYPE_F32, 10),
    stack_ptx_encode_store(STACK_PTX_STACK_TYPE_F32, 11),

    // u = x * CURAND_2POW32_INV + (CURAND_2POW32_INV/2);
    stack_ptx_encode_constant_f32(CURAND_2POW32_INV_HALF),
    stack_ptx_encode_load(10),
    stack_ptx_encode_constant_f32(CURAND_2POW32_INV),
    stack_ptx_encode_ptx_instruction_fma_rn_ftz_f32,

    // s = sqrtf(-2.0f * logf(u));
    stack_ptx_encode_ptx_instruction_lg2_approx_ftz_f32,
    stack_ptx_encode_constant_f32(PTX_LN2 * (-2.0f)),
    stack_ptx_encode_ptx_instruction_mul_ftz_f32,
    stack_ptx_encode_ptx_instruction_sqrt_approx_ftz_f32,

    stack_ptx_encode_store(STACK_PTX_STACK_TYPE_F32, 10),

    // v = y * CURAND_2POW32_INV_2PI + (CURAND_2POW32_INV_2PI/2);
    stack_ptx_encode_constant_f32(CURAND_2POW32_INV_2PI_HALF),
    stack_ptx_encode_load(11),
    stack_ptx_encode_constant_f32(CURAND_2POW32_INV_2PI),
    stack_ptx_encode_ptx_instruction_fma_rn_ftz_f32,
    stack_ptx_encode_store(STACK_PTX_STACK_TYPE_F32, 11),

    stack_ptx_encode_load(11),
    stack_ptx_encode_ptx_instruction_cos_approx_ftz_f32,
    stack_ptx_encode_load(10),
    stack_ptx_encode_ptx_instruction_mul_ftz_f32,

    stack_ptx_encode_load(11),
    stack_ptx_encode_ptx_instruction_sin_approx_ftz_f32,
    stack_ptx_encode_load(10),
    stack_ptx_encode_ptx_instruction_mul_ftz_f32,

    stack_ptx_encode_return
};

static const StackPtxInstruction routine_library_philox_curand[] = {
    stack_ptx_encode_store(STACK_PTX_STACK_TYPE_PHILOX, PHILOX_CTR_X_IDX),
    stack_ptx_encode_store(STACK_PTX_STACK_TYPE_PHILOX, PHILOX_CTR_Y_IDX),
    stack_ptx_encode_store(STACK_PTX_STACK_TYPE_PHILOX, PHILOX_CTR_Z_IDX),
    stack_ptx_encode_store(STACK_PTX_STACK_TYPE_PHILOX, PHILOX_CTR_W_IDX),
    stack_ptx_encode_store(STACK_PTX_STACK_TYPE_PHILOX, PHILOX_KEY_X_IDX),
    stack_ptx_encode_store(STACK_PTX_STACK_TYPE_PHILOX, PHILOX_KEY_Y_IDX),

    stack_ptx_encode_load(PHILOX_KEY_Y_IDX),
    stack_ptx_encode_load(PHILOX_KEY_X_IDX),
    stack_ptx_encode_load(PHILOX_CTR_W_IDX),
    stack_ptx_encode_load(PHILOX_CTR_Z_IDX),
    stack_ptx_encode_load(PHILOX_CTR_Y_IDX),
    stack_ptx_encode_load(PHILOX_CTR_X_IDX),

    stack_ptx_encode_load(PHILOX_KEY_Y_IDX),
    stack_ptx_encode_load(PHILOX_KEY_X_IDX),
    stack_ptx_encode_load(PHILOX_CTR_W_IDX),
    stack_ptx_encode_load(PHILOX_CTR_Z_IDX),
    stack_ptx_encode_load(PHILOX_CTR_Y_IDX),
    stack_ptx_encode_load(PHILOX_CTR_X_IDX),

    stack_ptx_encode_routine(ROUTINE_LIBRARY_PHILOX_ROUND),
    stack_ptx_encode_routine(ROUTINE_LIBRARY_PHILOX_ROUND),
    stack_ptx_encode_routine(ROUTINE_LIBRARY_PHILOX_ROUND),
    stack_ptx_encode_routine(ROUTINE_LIBRARY_PHILOX_ROUND),
    stack_ptx_encode_routine(ROUTINE_LIBRARY_PHILOX_ROUND),
    stack_ptx_encode_routine(ROUTINE_LIBRARY_PHILOX_ROUND),
    stack_ptx_encode_routine(ROUTINE_LIBRARY_PHILOX_ROUND),
    stack_ptx_encode_routine(ROUTINE_LIBRARY_PHILOX_ROUND),
    stack_ptx_encode_routine(ROUTINE_LIBRARY_PHILOX_ROUND),
    stack_ptx_encode_routine(ROUTINE_LIBRARY_PHILOX_ROUND),

    stack_ptx_encode_ptx_instruction_mov_philox,
    stack_ptx_encode_ptx_instruction_mov_philox,
    stack_ptx_encode_ptx_instruction_mov_philox,
    stack_ptx_encode_ptx_instruction_mov_philox,

    stack_ptx_encode_meta_constant(2),
    stack_ptx_encode_meta_drop(STACK_PTX_STACK_TYPE_PHILOX),

    stack_ptx_encode_constant_philox(1),
    stack_ptx_encode_ptx_instruction_add_philox,
    stack_ptx_encode_return
};

static const StackPtxInstruction routine_library_philox_curand_uniform[] = {
    stack_ptx_encode_routine(ROUTINE_LIBRARY_PHILOX_CURAND),
    stack_ptx_encode_routine(ROUTINE_LIBRARY_PHILOX_UNIFORM_SCALE),
    stack_ptx_encode_routine(ROUTINE_LIBRARY_PHILOX_UNIFORM_SCALE),
    stack_ptx_encode_routine(ROUTINE_LIBRARY_PHILOX_UNIFORM_SCALE),
    stack_ptx_encode_routine(ROUTINE_LIBRARY_PHILOX_UNIFORM_SCALE),
    stack_ptx_encode_return
};

static const StackPtxInstruction routine_library_philox_curand_normal[] = {
    stack_ptx_encode_routine(ROUTINE_LIBRARY_PHILOX_CURAND),
    stack_ptx_encode_routine(ROUTINE_LIBRARY_PHILOX_BOX_MULLER),
    stack_ptx_encode_routine(ROUTINE_LIBRARY_PHILOX_BOX_MULLER),
    stack_ptx_encode_return
};

// Requires ptx_inject.h and REGISTER_PHILOX_* enum entries in the including unit.
static
PtxInjectResult
ptx_inject_philox_populate_registers(
    PtxInjectHandle ptx_inject,
    size_t inject_func_idx,
    StackPtxRegister* registers
) {
    PtxInjectResult result;
    result = ptx_inject_variable_info_by_name(ptx_inject, inject_func_idx, "philox_key_x", NULL, &registers[REGISTER_PHILOX_KEY_X].name, NULL, NULL, NULL);
    if (result != PTX_INJECT_SUCCESS) return result;

    result = ptx_inject_variable_info_by_name(ptx_inject, inject_func_idx, "philox_key_y", NULL, &registers[REGISTER_PHILOX_KEY_Y].name, NULL, NULL, NULL);
    if (result != PTX_INJECT_SUCCESS) return result;

    result = ptx_inject_variable_info_by_name(ptx_inject, inject_func_idx, "philox_ctr_x", NULL, &registers[REGISTER_PHILOX_CTR_X].name, NULL, NULL, NULL);
    if (result != PTX_INJECT_SUCCESS) return result;

    result = ptx_inject_variable_info_by_name(ptx_inject, inject_func_idx, "philox_ctr_y", NULL, &registers[REGISTER_PHILOX_CTR_Y].name, NULL, NULL, NULL);
    if (result != PTX_INJECT_SUCCESS) return result;

    result = ptx_inject_variable_info_by_name(ptx_inject, inject_func_idx, "philox_ctr_z", NULL, &registers[REGISTER_PHILOX_CTR_Z].name, NULL, NULL, NULL);
    if (result != PTX_INJECT_SUCCESS) return result;

    result = ptx_inject_variable_info_by_name(ptx_inject, inject_func_idx, "philox_ctr_w", NULL, &registers[REGISTER_PHILOX_CTR_W].name, NULL, NULL, NULL);
    if (result != PTX_INJECT_SUCCESS) return result;

    return result;
}
