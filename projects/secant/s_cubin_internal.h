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
#ifndef SECANT_CUBIN_INTERNAL_H_INCLUDED
#define SECANT_CUBIN_INTERNAL_H_INCLUDED

#include "secant.h"

#define _SECANT_CUBIN_WORKSPACE_ALIGNMENT 16u
#define _SECANT_CUBIN_SSE_MAX_WARPS 16u
#define _SECANT_CUBIN_MAX_REGISTERS 256u
#define _SECANT_CUBIN_NAME_BYTES 256u
#define _SECANT_CUBIN_INSTRUCTION_BYTES 16u
#define _SECANT_CUBIN_ROUTINE_DEPTH 8u
#define _SECANT_CUBIN_STACK_DEPTH SECANT_AST_MAX_STACK_DEPTH
#define _SECANT_CUBIN_BARRIER_SLOTS 6u
#define _SECANT_CUBIN_REGISTER_RZ 255u
#define _SECANT_CUBIN_REGISTER_PAD 2u
#define _SECANT_CUBIN_FIRST_MARKER_BITS 0x7fc0ffeeu

#define _SECANT_CUBIN_BPT_WORD0 0x000000040000795cull
#define _SECANT_CUBIN_NOP_WORD0 0x0000000000007918ull
#define _SECANT_CUBIN_NOP_WORD1 0x000fc00000000000ull
#define _SECANT_CUBIN_OPCODE_MOV 0x7202u
#define _SECANT_CUBIN_OPCODE_FSEL 0x7208u
#define _SECANT_CUBIN_OPCODE_FMNMX_REG 0x7209u
#define _SECANT_CUBIN_OPCODE_FSETP 0x720bu
#define _SECANT_CUBIN_OPCODE_FMUL_REG 0x7220u
#define _SECANT_CUBIN_OPCODE_FADD_REG 0x7221u
#define _SECANT_CUBIN_OPCODE_FFMA_REG 0x7223u
#define _SECANT_CUBIN_OPCODE_FADD_IMM 0x7421u
#define _SECANT_CUBIN_OPCODE_FMUL_IMM 0x7820u
#define _SECANT_CUBIN_OPCODE_MUFU 0x7308u
#define _SECANT_CUBIN_OPCODE_BRA 0x7947u
#define _SECANT_CUBIN_OPCODE_STS_RZ 0x7388u
#define _SECANT_CUBIN_OPCODE_STS_UR 0x7988u
#define _SECANT_CUBIN_OPCODE_UMOV 0x7c82u

#define _SECANT_CUBIN_CONTROL_ALU_HI 0xfcu
#define _SECANT_CUBIN_CONTROL_ALU_STALL 0xcu
#define _SECANT_CUBIN_CONTROL_BARRIER_BASE 0xe2u
#define _SECANT_CUBIN_CONTROL_MUFU_STALL 0x6u
#define _SECANT_CUBIN_REST_FADD_IMM 0x0000010000ull
#define _SECANT_CUBIN_REST_FADD_ABS 0x0000010200ull
#define _SECANT_CUBIN_REST_FMUL 0x0000410000ull
#define _SECANT_CUBIN_REST_FMUL_RZ 0x000040c000ull
#define _SECANT_CUBIN_REST_FMNMX_MIN 0x0003810000ull
#define _SECANT_CUBIN_REST_FMNMX_MAX 0x0007810000ull
#define _SECANT_CUBIN_REST_FFMA 0x0000010000ull
#define _SECANT_CUBIN_REST_FSETP_GE 0x0003f06000ull
#define _SECANT_CUBIN_REST_FSETP_GTU 0x0003f0c000ull
#define _SECANT_CUBIN_REST_FSETP_GEU 0x0003f0e000ull
#define _SECANT_CUBIN_REST_FSEL_NEGATE_LHS 0x0000000100ull
#define _SECANT_CUBIN_REST_FSEL_INVERT_PREDICATE 0x0004000000ull
#define _SECANT_CUBIN_REST_MOV 0x0000000f00ull
#define _SECANT_CUBIN_REST_MUFU_COS 0x0000000000ull
#define _SECANT_CUBIN_REST_MUFU_SIN 0x0000000400ull
#define _SECANT_CUBIN_REST_MUFU_EX2 0x0000000800ull
#define _SECANT_CUBIN_REST_MUFU_LG2 0x0000000c00ull
#define _SECANT_CUBIN_REST_MUFU_RCP 0x0000001000ull
#define _SECANT_CUBIN_REST_MUFU_RSQ 0x0000001400ull
#define _SECANT_CUBIN_REST_MUFU_SQRT 0x0000002000ull
#define _SECANT_CUBIN_REST_MUFU_TANH 0x0000002400ull
#define _SECANT_CUBIN_REST_BRA 0x0003800000ull
#define _SECANT_CUBIN_NEG_ONE_BITS 0xbf800000u
#define _SECANT_CUBIN_SIN_COS_SCALE_BITS 0x3e22f983u
#define _SECANT_CUBIN_LN_TWO_BITS 0x3f317218u
#define _SECANT_CUBIN_LOG2_E_BITS 0x3fb8aa3bu

#define _SECANT_CUBIN_NVINFO_FORMAT_U32 0x04u
#define _SECANT_CUBIN_NVINFO_ATTR_REGCOUNT 0x2fu
#define _SECANT_CUBIN_NVINFO_ATTR_SIZE 8u

#define _SECANT_CUBIN_ERROR_RET(ans) do { \
    SecantResult _secant_cubin_result = (ans); \
    return _secant_cubin_result; \
} while (0)
#define _SECANT_CUBIN_CHECK_RET(ans) do { \
    SecantResult _secant_cubin_check_result = (ans); \
    if (_secant_cubin_check_result != SECANT_SUCCESS) { \
        _SECANT_CUBIN_ERROR_RET(_secant_cubin_check_result); \
    } \
} while (0)

typedef enum _SecantCubinShape {
    _SECANT_CUBIN_SHAPE_MATERIALIZE = 1,
    _SECANT_CUBIN_SHAPE_SSE = 2,
    _SECANT_CUBIN_SHAPE_AFFINE_STATS = 5,
    _SECANT_CUBIN_SHAPE_GRAM_STATS = 6,
    _SECANT_CUBIN_SHAPE_TOGGLE_SSE = 10
} _SecantCubinShape;

typedef struct _SecantCubinSassInstruction {
    uint64_t word0;
    uint64_t word1;
} _SecantCubinSassInstruction;

typedef struct _SecantCubinSite {
    size_t load_fence_file_offset;
    size_t start_file_offset;
    size_t num_instructions;
    uint32_t incoming_wait_mask;
    size_t num_input_regs;
    uint8_t input_regs[_SECANT_CUBIN_MAX_REGISTERS];
    size_t num_target_regs;
    uint8_t target_regs[_SECANT_CUBIN_MAX_REGISTERS];
    size_t num_output_regs;
    uint8_t output_regs[_SECANT_CUBIN_MAX_REGISTERS];
    uint8_t permutation_reg;
    uint8_t predicate_reg;
    _SecantCubinSassInstruction toggle_test;
    size_t num_available_regs;
    uint8_t available_regs[_SECANT_CUBIN_MAX_REGISTERS];
} _SecantCubinSite;

typedef struct _SecantCubinKernel {
    uint32_t register_count;
    size_t first_register_count;
    size_t num_register_counts;
    _SecantCubinSite site;
} _SecantCubinKernel;

struct SecantCubinPlan {
    _SecantCubinShape shape;
    size_t cubin_size;
    _SecantCubinKernel* kernels;
    size_t num_kernels;
    size_t* register_count_file_offsets;
    size_t asts_per_kernel;
    size_t num_input_registers;
    size_t num_constant_registers;
    size_t num_targets;
    size_t tile_rows;
    size_t threads_per_block;
    size_t patch_capacity_instructions;
    uint32_t compute_capability_major;
};

SecantResult _secant_cubin_materialize_source_generate(
    const SecantCubinMaterializeRecipe* recipe,
    char* output,
    size_t output_size,
    size_t* required_size_ret
);

SecantResult _secant_cubin_sse_source_generate(
    const SecantCubinSSERecipe* recipe,
    char* output,
    size_t output_size,
    size_t* required_size_ret
);

SecantResult _secant_cubin_affine_stats_source_generate(
    const SecantCubinAffineStatsRecipe* recipe,
    char* output,
    size_t output_size,
    size_t* required_size_ret
);

SecantResult _secant_cubin_gram_stats_source_generate(
    const SecantCubinGramStatsRecipe* recipe,
    char* output,
    size_t output_size,
    size_t* required_size_ret
);

SecantResult _secant_cubin_toggle_sse_source_generate(const SecantCubinToggleSSERecipe*, char*, size_t, size_t*);
SecantResult _secant_cubin_toggle_sse_recipe_validate(const SecantCubinToggleSSERecipe*);

SecantResult _secant_cubin_recipe_header_validate(
    const SecantCubinRecipeHeader* recipe
);

SecantResult _secant_cubin_recipe_validate(
    const SecantCubinRecipeHeader* recipe
);

SecantResult _secant_cubin_materialize_recipe_validate(
    const SecantCubinMaterializeRecipe* recipe
);

SecantResult _secant_cubin_sse_recipe_validate(
    const SecantCubinSSERecipe* recipe
);

SecantResult _secant_cubin_affine_stats_recipe_validate(
    const SecantCubinAffineStatsRecipe* recipe
);

SecantResult _secant_cubin_gram_stats_recipe_validate(
    const SecantCubinGramStatsRecipe* recipe
);

static inline int
_secant_cubin_checked_add(size_t lhs, size_t rhs, size_t* result_ret) {
    if (rhs > SIZE_MAX - lhs) {
        return 0;
    }
    *result_ret = lhs + rhs;
    return 1;
}

static inline int
_secant_cubin_checked_mul(size_t lhs, size_t rhs, size_t* result_ret) {
    if (lhs != 0u && rhs > SIZE_MAX / lhs) {
        return 0;
    }
    *result_ret = lhs * rhs;
    return 1;
}

static inline int
_secant_cubin_range_ok(size_t size, size_t offset, size_t count) {
    return offset <= size && count <= size - offset;
}

static inline uint16_t
_secant_cubin_read_u16(const unsigned char* data) {
    return (uint16_t)data[0] | ((uint16_t)data[1] << 8);
}

static inline uint32_t
_secant_cubin_read_u32(const unsigned char* data) {
    return (uint32_t)data[0] |
        ((uint32_t)data[1] << 8) |
        ((uint32_t)data[2] << 16) |
        ((uint32_t)data[3] << 24);
}

static inline uint64_t
_secant_cubin_read_u64(const unsigned char* data) {
    return (uint64_t)_secant_cubin_read_u32(data) |
        ((uint64_t)_secant_cubin_read_u32(data + 4u) << 32);
}

static inline void
_secant_cubin_write_u32(unsigned char* data, uint32_t value) {
    data[0] = (unsigned char)(value & 0xffu);
    data[1] = (unsigned char)((value >> 8) & 0xffu);
    data[2] = (unsigned char)((value >> 16) & 0xffu);
    data[3] = (unsigned char)((value >> 24) & 0xffu);
}

static inline void
_secant_cubin_write_u64(unsigned char* data, uint64_t value) {
    _secant_cubin_write_u32(data, (uint32_t)(value & UINT32_MAX));
    _secant_cubin_write_u32(data + 4u, (uint32_t)(value >> 32));
}

static inline void
_secant_cubin_write_instruction(unsigned char* data, _SecantCubinSassInstruction instruction) {
    _secant_cubin_write_u64(data, instruction.word0);
    _secant_cubin_write_u64(data + 8u, instruction.word1);
}

static inline int
_secant_cubin_arch_supported(uint32_t major, uint32_t minor) {
    return (major == 8u && (minor == 0u || minor == 6u || minor == 9u)) ||
        (major == 9u && minor == 0u) ||
        (major == 10u && minor == 0u) ||
        (major == 12u && minor == 0u);
}

#endif /* SECANT_CUBIN_INTERNAL_H_INCLUDED */
