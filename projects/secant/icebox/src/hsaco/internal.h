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
#ifndef SECANT_HSACO_INTERNAL_H_INCLUDED
#define SECANT_HSACO_INTERNAL_H_INCLUDED
#include "secant_hsaco.h"


#include <elf.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>

#define _SECANT_HSACO_WORKSPACE_ALIGNMENT 16u
#define _SECANT_HSACO_MAX_WAVES 16u
#define _SECANT_HSACO_MAX_REGISTERS 256u
#define _SECANT_HSACO_NAME_BYTES 256u
#define _SECANT_HSACO_ROUTINE_DEPTH 8u
#define _SECANT_HSACO_STACK_DEPTH SECANT_AST_MAX_STACK_DEPTH
#define _SECANT_HSACO_FIRST_MARKER_BITS 0x7fc0ffeeu
#define _SECANT_HSACO_TRAP_WORD 0xbf900002u
#define _SECANT_HSACO_NOP_WORD 0xbf800000u
#define _SECANT_HSACO_BRANCH_WORD 0xbfa00000u
#define _SECANT_HSACO_DELAY_ALU_MASK 0xffff0000u
#define _SECANT_HSACO_DELAY_ALU_WORD 0xbf870000u
#define _SECANT_HSACO_VOPD_ADD_MASK 0xffff0000u
#define _SECANT_HSACO_VOPD_ADD_WORD 0xc9080000u
#define _SECANT_HSACO_GLOBAL_STORE_B32_WORD 0xee06807cu
#define _SECANT_HSACO_COMPUTE_PGM_RSRC1_OFFSET 0x30u
#define _SECANT_HSACO_COMPUTE_PGM_RSRC1_VGPRS_MASK 0x3fu
#define _SECANT_HSACO_AMDGPU_METADATA_NOTE 32u
#define _SECANT_HSACO_MSGPACK_FIXINT 1u
#define _SECANT_HSACO_MSGPACK_UINT8 2u
#define _SECANT_HSACO_MSGPACK_UINT16 3u
#define _SECANT_HSACO_MSGPACK_UINT32 4u
#define _SECANT_HSACO_MSGPACK_UINT64 5u
#define _SECANT_HSACO_MSGPACK_MAX_DEPTH 64u

#define _SECANT_HSACO_V_ADD_F32 0x06000000u
#define _SECANT_HSACO_V_SUB_F32 0x08000000u
#define _SECANT_HSACO_V_MUL_F32 0x10000000u
#define _SECANT_HSACO_V_MIN_F32 0x2a000000u
#define _SECANT_HSACO_V_MAX_F32 0x2c000000u
#define _SECANT_HSACO_V_AND_B32 0x36000000u
#define _SECANT_HSACO_V_XOR_B32 0x3a000000u
#define _SECANT_HSACO_V_FMA_F32 0xd6130000u
#define _SECANT_HSACO_VOP1 0x7e000000u
#define _SECANT_HSACO_VOP1_MOV 0x01u
#define _SECANT_HSACO_VOP1_EXP 0x25u
#define _SECANT_HSACO_VOP1_LOG 0x27u
#define _SECANT_HSACO_VOP1_RCP 0x2au
#define _SECANT_HSACO_VOP1_RSQ 0x2eu
#define _SECANT_HSACO_VOP1_SQRT 0x33u
#define _SECANT_HSACO_VOP1_SIN 0x35u
#define _SECANT_HSACO_VOP1_COS 0x36u
#define _SECANT_HSACO_VGPR_SOURCE 0x100u
#define _SECANT_HSACO_LITERAL_SOURCE 0xffu

#define _SECANT_HSACO_ERROR_RET(ans) do { SecantResult _secant_hsaco_result = (ans); return _secant_hsaco_result; } while (0)
#define _SECANT_HSACO_CHECK_RET(ans) do { SecantResult _secant_hsaco_check_result = (ans); if (_secant_hsaco_check_result != SECANT_SUCCESS) { _SECANT_HSACO_ERROR_RET(_secant_hsaco_check_result); } } while (0)

typedef enum _SecantHsacoShape {
    _SECANT_HSACO_SHAPE_MATERIALIZE = 1,
    _SECANT_HSACO_SHAPE_SSE = 2,
    _SECANT_HSACO_SHAPE_DYNAMIC_CONSTANT_SSE = 3
} _SecantHsacoShape;

typedef struct _SecantHsacoSite {
    size_t load_fence_file_offset;
    size_t start_file_offset;
    size_t end_file_offset;
    size_t patch_size;
    size_t num_input_regs;
    uint8_t input_regs[_SECANT_HSACO_MAX_REGISTERS];
    size_t num_target_regs;
    uint8_t target_regs[_SECANT_HSACO_MAX_REGISTERS];
    size_t num_output_regs;
    uint8_t output_regs[_SECANT_HSACO_MAX_REGISTERS];
    size_t num_available_regs;
    uint8_t available_regs[_SECANT_HSACO_MAX_REGISTERS];
} _SecantHsacoSite;

typedef struct _SecantHsacoKernel {
    char name[_SECANT_HSACO_NAME_BYTES];
    uint32_t register_count;
    uint32_t allocated_registers;
    size_t compute_pgm_rsrc1_file_offset;
    uint32_t compute_pgm_rsrc1;
    size_t num_vgpr_symbol_value_file_offset;
    size_t metadata_vgpr_file_offset;
    uint8_t metadata_vgpr_encoding;
    _SecantHsacoSite site;
} _SecantHsacoKernel;

typedef struct _SecantHsacoTemplate {
    _SecantHsacoShape shape;
    size_t hsaco_size;
    _SecantHsacoKernel* kernels;
    size_t num_kernels;
    size_t asts_per_kernel;
    size_t num_inputs;
    size_t num_input_constants;
    size_t num_targets;
    size_t tile_rows;
    size_t threads_per_block;
    size_t patch_capacity_instructions;
    uint32_t gfx_arch;
} _SecantHsacoTemplate;

struct SecantHsacoPlan {
    _SecantHsacoTemplate template_data;
};

static inline int
_secant_hsaco_checked_add(size_t lhs, size_t rhs, size_t* result_ret) {
    if (rhs > SIZE_MAX - lhs) {
        return 0;
    }
    *result_ret = lhs + rhs;
    return 1;
}

static inline int
_secant_hsaco_checked_mul(size_t lhs, size_t rhs, size_t* result_ret) {
    if (lhs != 0u && rhs > SIZE_MAX / lhs) {
        return 0;
    }
    *result_ret = lhs * rhs;
    return 1;
}

static inline int
_secant_hsaco_range_ok(size_t size, size_t offset, size_t count) {
    return offset <= size && count <= size - offset;
}

static inline uint32_t
_secant_hsaco_read_u32(const unsigned char* data) {
    return (uint32_t)data[0] |
        ((uint32_t)data[1] << 8u) |
        ((uint32_t)data[2] << 16u) |
        ((uint32_t)data[3] << 24u);
}

static inline void
_secant_hsaco_write_u32(unsigned char* data, uint32_t value) {
    data[0] = (unsigned char)(value & 0xffu);
    data[1] = (unsigned char)((value >> 8u) & 0xffu);
    data[2] = (unsigned char)((value >> 16u) & 0xffu);
    data[3] = (unsigned char)((value >> 24u) & 0xffu);
}

static inline void
_secant_hsaco_write_u64(unsigned char* data, uint64_t value) {
    _secant_hsaco_write_u32(data, (uint32_t)value);
    _secant_hsaco_write_u32(data + 4u, (uint32_t)(value >> 32u));
}

static inline uint16_t
_secant_hsaco_read_be_u16(const unsigned char* data) {
    return ((uint16_t)data[0] << 8u) |
        (uint16_t)data[1];
}

static inline uint32_t
_secant_hsaco_read_be_u32(const unsigned char* data) {
    return ((uint32_t)data[0] << 24u) |
        ((uint32_t)data[1] << 16u) |
        ((uint32_t)data[2] << 8u) |
        (uint32_t)data[3];
}

static inline uint64_t
_secant_hsaco_read_be_u64(const unsigned char* data) {
    return ((uint64_t)_secant_hsaco_read_be_u32(data) << 32u) |
        _secant_hsaco_read_be_u32(data + 4u);
}

static inline void
_secant_hsaco_write_be_u16(unsigned char* data, uint16_t value) {
    data[0] = (unsigned char)(value >> 8u);
    data[1] = (unsigned char)(value & 0xffu);
}

static inline void
_secant_hsaco_write_be_u32(unsigned char* data, uint32_t value) {
    data[0] = (unsigned char)(value >> 24u);
    data[1] = (unsigned char)((value >> 16u) & 0xffu);
    data[2] = (unsigned char)((value >> 8u) & 0xffu);
    data[3] = (unsigned char)(value & 0xffu);
}

static inline void
_secant_hsaco_write_be_u64(unsigned char* data, uint64_t value) {
    _secant_hsaco_write_be_u32(data, (uint32_t)(value >> 32u));
    _secant_hsaco_write_be_u32(data + 4u, (uint32_t)value);
}
#endif /* SECANT_HSACO_INTERNAL_H_INCLUDED */
