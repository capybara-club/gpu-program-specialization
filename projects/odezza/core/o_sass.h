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
#ifndef O_SASS_H
#define O_SASS_H

#include "odezza.h"
#include <limits.h>

/* Internal C99 writer for the validated SM89/SM90/SM120 instruction subset.
 * No AST, register allocator, CUBIN inspection, allocation, or CUDA calls.
 * Scheduling policy is deliberately fixed: 12-cycle ALU/predicate stalls;
 * MUFU writes a dependency barrier with a 6-cycle issue stall. Callers own
 * register lifetimes, barrier allocation, and all required dependency waits.
 */
#define O_SASS_INSTRUCTION_BYTES 16u
#define O_SASS_RZ 255u
#define O_SASS_WAIT_MASK 0x3fu
#define O_SASS_BARRIER_SLOTS 6u
#define O_SASS_ALU_STALL 12u

typedef struct OSassInstruction {
    uint64_t word0;
    uint64_t word1;
} OSassInstruction;

/* Encoders take wide operands so invalid values are rejected before narrowing.
 * An unsuccessful encoding contains no instruction and cannot be emitted. */
typedef struct OSassEncoding {
    uint64_t word0;
    uint64_t word1;
    OdezzaResult status;
} OSassEncoding;

typedef enum OSassMultiplyMode {
    O_SASS_MULTIPLY_NORMAL,
    O_SASS_MULTIPLY_RZ
} OSassMultiplyMode;

typedef enum OSassMufu {
    O_SASS_MUFU_COS, O_SASS_MUFU_SIN, O_SASS_MUFU_EX2, O_SASS_MUFU_LG2,
    O_SASS_MUFU_RCP, O_SASS_MUFU_RSQ, O_SASS_MUFU_SQRT, O_SASS_MUFU_TANH
} OSassMufu;

typedef struct OSassWriter {
    OSassInstruction *instructions;
    size_t capacity;
    size_t count;
    uint32_t pending_wait_mask;
} OSassWriter;

static inline int o_sass_architecture_supported(uint32_t architecture) {
    return architecture == 89u || architecture == 90u || architecture == 120u;
}

static inline OSassEncoding o_sass_error(OdezzaResult status) {
    OSassEncoding result = {0u, 0u, status};
    return result;
}

/* Private helpers: operands and control fields must already be validated. */
static inline uint64_t o_sass_alu_control(uint32_t wait, uint64_t rest) {
    return ((uint64_t)wait << 52u) | UINT64_C(0x000fcc0000000000) | rest;
}

static inline OSassEncoding o_sass_words(uint64_t word0, uint64_t word1) {
    OSassEncoding result = {word0, word1, ODEZZA_SUCCESS};
    return result;
}

static inline int o_sass_operands_valid(uint32_t destination, uint32_t source, uint32_t wait) {
    return destination < O_SASS_RZ && source <= O_SASS_RZ && wait <= O_SASS_WAIT_MASK;
}

static inline OSassEncoding o_sass_nop(uint32_t wait) {
    if (wait > O_SASS_WAIT_MASK) return o_sass_error(ODEZZA_ERROR_INVALID_ARGUMENT);
    return o_sass_words(UINT64_C(0x7918), ((uint64_t)wait << 52u) | UINT64_C(0x000fc00000000000));
}

static inline OSassEncoding o_sass_mov(uint32_t destination, uint32_t source, uint32_t wait) {
    if (!o_sass_operands_valid(destination, source, wait)) return o_sass_error(ODEZZA_ERROR_INVALID_ARGUMENT);
    return o_sass_words(((uint64_t)source << 32u) | ((uint64_t)destination << 16u) | 0x7202u,
                        o_sass_alu_control(wait, UINT64_C(0xf00)));
}

static inline OSassEncoding o_sass_fadd_register(uint32_t destination, uint32_t lhs, uint32_t rhs, uint32_t wait) {
    if (!o_sass_operands_valid(destination, lhs, wait) || rhs > O_SASS_RZ) return o_sass_error(ODEZZA_ERROR_INVALID_ARGUMENT);
    return o_sass_words(((uint64_t)rhs << 32u) | ((uint64_t)lhs << 24u) | ((uint64_t)destination << 16u) | 0x7221u,
                        o_sass_alu_control(wait, 0u));
}

static inline OSassEncoding o_sass_fadd_immediate(uint32_t destination, uint32_t source, uint32_t bits, uint32_t wait) {
    if (!o_sass_operands_valid(destination, source, wait)) return o_sass_error(ODEZZA_ERROR_INVALID_ARGUMENT);
    return o_sass_words(((uint64_t)bits << 32u) | ((uint64_t)source << 24u) | ((uint64_t)destination << 16u) | 0x7421u,
                        o_sass_alu_control(wait, UINT64_C(0x10000)));
}

static inline OSassEncoding o_sass_fmul_register(uint32_t destination, uint32_t lhs, uint32_t rhs, uint32_t wait) {
    if (!o_sass_operands_valid(destination, lhs, wait) || rhs > O_SASS_RZ) return o_sass_error(ODEZZA_ERROR_INVALID_ARGUMENT);
    return o_sass_words(((uint64_t)rhs << 32u) | ((uint64_t)lhs << 24u) | ((uint64_t)destination << 16u) | 0x7220u,
                        o_sass_alu_control(wait, UINT64_C(0x410000)));
}

static inline OSassEncoding o_sass_fmul_immediate(uint32_t destination, uint32_t source, uint32_t bits, uint32_t wait, OSassMultiplyMode mode) {
    if (!o_sass_operands_valid(destination, source, wait) || (mode != O_SASS_MULTIPLY_NORMAL && mode != O_SASS_MULTIPLY_RZ))
        return o_sass_error(ODEZZA_ERROR_INVALID_ARGUMENT);
    return o_sass_words(((uint64_t)bits << 32u) | ((uint64_t)source << 24u) | ((uint64_t)destination << 16u) | 0x7820u,
                        o_sass_alu_control(wait, mode == O_SASS_MULTIPLY_RZ ? UINT64_C(0x40c000) : UINT64_C(0x410000)));
}

static inline OSassEncoding o_sass_fsel(uint32_t destination, uint32_t true_value, uint32_t false_value, uint32_t predicate) {
    if (!o_sass_operands_valid(destination, true_value, 0u) || false_value > O_SASS_RZ || predicate >= 7u)
        return o_sass_error(ODEZZA_ERROR_INVALID_ARGUMENT);
    return o_sass_words(((uint64_t)false_value << 32u) | ((uint64_t)true_value << 24u) | ((uint64_t)destination << 16u) | 0x7208u,
                        o_sass_alu_control(0u, (uint64_t)predicate << 23u));
}

/* Preserve the compiler-observed predicate encoding, after verifying its form.
 * Only the permutation register, bit mask, incoming waits, and stall change. */
static inline OSassEncoding o_sass_toggle_test(OSassInstruction prototype, uint32_t permutation, uint32_t predicate, uint32_t bit) {
    uint64_t rest = prototype.word1 & ((UINT64_C(1) << 40u) - 1u);
    if (permutation >= O_SASS_RZ || predicate >= 7u || bit >= 32u) return o_sass_error(ODEZZA_ERROR_INVALID_ARGUMENT);
    if ((prototype.word0 & UINT64_C(0xffffff)) != UINT64_C(0xff7812) ||
        ((rest >> 17u) & 7u) != predicate || (rest & ~(UINT64_C(7) << 17u)) != UINT64_C(0x780c0ff))
        return o_sass_error(ODEZZA_ERROR_FORMAT);
    return o_sass_words((prototype.word0 & UINT64_C(0xffffff)) | ((uint64_t)permutation << 24u) | ((uint64_t)(UINT32_C(1) << bit) << 32u),
                        (prototype.word1 & ~((UINT64_C(0x3f) << 52u) | (UINT64_C(0xf) << 40u))) | ((uint64_t)O_SASS_ALU_STALL << 40u));
}

static inline OSassEncoding o_sass_absolute(uint32_t destination, uint32_t source, uint32_t wait) {
    if (!o_sass_operands_valid(destination, source, wait)) return o_sass_error(ODEZZA_ERROR_INVALID_ARGUMENT);
    return o_sass_words((UINT64_C(0x800000ff) << 32u) | ((uint64_t)source << 24u) | ((uint64_t)destination << 16u) | 0x7221u,
                        o_sass_alu_control(wait, UINT64_C(0x10200)));
}

static inline OSassEncoding o_sass_fmnmx(uint32_t destination, uint32_t lhs, uint32_t rhs, int maximum, uint32_t wait) {
    if (!o_sass_operands_valid(destination, lhs, wait) || rhs > O_SASS_RZ || (maximum != 0 && maximum != 1))
        return o_sass_error(ODEZZA_ERROR_INVALID_ARGUMENT);
    return o_sass_words(((uint64_t)rhs << 32u) | ((uint64_t)lhs << 24u) | ((uint64_t)destination << 16u) | 0x7209u,
                        o_sass_alu_control(wait, maximum ? UINT64_C(0x7810000) : UINT64_C(0x3810000)));
}

static inline OSassEncoding o_sass_ffma(uint32_t destination, uint32_t lhs, uint32_t rhs, uint32_t addend, uint32_t wait) {
    if (!o_sass_operands_valid(destination, lhs, wait) || rhs > O_SASS_RZ || addend > O_SASS_RZ)
        return o_sass_error(ODEZZA_ERROR_INVALID_ARGUMENT);
    return o_sass_words(((uint64_t)rhs << 32u) | ((uint64_t)lhs << 24u) | ((uint64_t)destination << 16u) | 0x7223u,
                        o_sass_alu_control(wait, UINT64_C(0x10000) | addend));
}

static inline OSassEncoding o_sass_mufu(uint32_t destination, uint32_t source, uint32_t wait, uint32_t barrier, OSassMufu function) {
    static const uint32_t rest[] = {0x0000u, 0x0400u, 0x0800u, 0x0c00u, 0x1000u, 0x1400u, 0x2000u, 0x2400u};
    if (!o_sass_operands_valid(destination, source, wait) || barrier >= O_SASS_BARRIER_SLOTS || (unsigned)function > O_SASS_MUFU_TANH)
        return o_sass_error(ODEZZA_ERROR_INVALID_ARGUMENT);
    return o_sass_words(((uint64_t)source << 32u) | ((uint64_t)destination << 16u) | 0x7308u,
                        ((uint64_t)wait << 52u) | ((UINT64_C(0xe2) + 4u * barrier) << 44u) | (UINT64_C(6) << 40u) | rest[function]);
}

/* Forward distance is measured from this instruction, in 16-byte instructions.
 * This subset deliberately accepts only signed-32-bit positive byte offsets.
 * The old writer checked 64-bit multiplication but could truncate the encoded
 * displacement. No backward branches or larger displacement forms are claimed. */
static inline OSassEncoding o_sass_branch(size_t distance, uint32_t architecture) {
    uint64_t target;
    if (!o_sass_architecture_supported(architecture)) return o_sass_error(ODEZZA_ERROR_UNSUPPORTED);
    if (distance == 0u) return o_sass_error(ODEZZA_ERROR_INVALID_ARGUMENT);
    if (distance - 1u > INT32_MAX / O_SASS_INSTRUCTION_BYTES) return o_sass_error(ODEZZA_ERROR_OVERFLOW);
    target = (uint64_t)(distance - 1u) * O_SASS_INSTRUCTION_BYTES;
    if (architecture == 89u) return o_sass_words((target << 32u) | 0x7947u, o_sass_alu_control(0u, UINT64_C(0x3800000)));
    target /= 4u;
    return o_sass_words(((target & 0xffu) << 16u) | ((target & ~UINT64_C(0xff)) << 26u) | 0x7947u,
                        o_sass_alu_control(0u, UINT64_C(0x3800000)));
}

static inline OdezzaResult o_sass_instruction(OSassEncoding encoding, OSassInstruction *output) {
    if (output == NULL) return ODEZZA_ERROR_INVALID_ARGUMENT;
    if (encoding.status != ODEZZA_SUCCESS) return encoding.status;
    output->word0 = encoding.word0;
    output->word1 = encoding.word1;
    return ODEZZA_SUCCESS;
}

static inline OdezzaResult o_sass_writer_init(OSassWriter *writer, OSassInstruction *instructions, size_t capacity, uint32_t incoming_waits) {
    if (writer == NULL || (instructions == NULL && capacity != 0u) || ((uintptr_t)instructions & 7u) != 0u || incoming_waits > O_SASS_WAIT_MASK)
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    if (capacity > SIZE_MAX / sizeof(*instructions)) return ODEZZA_ERROR_OVERFLOW;
    writer->instructions = instructions;
    writer->capacity = capacity;
    writer->count = 0u;
    writer->pending_wait_mask = incoming_waits;
    return ODEZZA_SUCCESS;
}

/* Failed emission preserves the buffer, count, and pending dependency waits. */
static inline OdezzaResult o_sass_writer_emit(OSassWriter *writer, OSassEncoding encoding) {
    OSassInstruction instruction;
    OdezzaResult result;
    if (writer == NULL || writer->pending_wait_mask > O_SASS_WAIT_MASK) return ODEZZA_ERROR_INVALID_ARGUMENT;
    result = o_sass_instruction(encoding, &instruction);
    if (result != ODEZZA_SUCCESS) return result;
    if (writer->count >= writer->capacity) return ODEZZA_ERROR_SPECIALIZATION_CAPACITY;
    if (writer->instructions == NULL) return ODEZZA_ERROR_INVALID_ARGUMENT;
    instruction.word1 |= (uint64_t)writer->pending_wait_mask << 52u;
    writer->instructions[writer->count++] = instruction;
    writer->pending_wait_mask = 0u;
    return ODEZZA_SUCCESS;
}

/* Serialize already encoded words; this does not validate arbitrary SASS.
 * Offsets need no native alignment. Bounds are checked before either word is
 * written, and little-endian output is independent of host byte order. */
static inline OdezzaResult o_sass_store(void *buffer, size_t size, size_t offset, OSassInstruction instruction) {
    unsigned char *bytes = (unsigned char *)buffer;
    size_t index;
    if (buffer == NULL) return ODEZZA_ERROR_INVALID_ARGUMENT;
    if (offset > size || O_SASS_INSTRUCTION_BYTES > size - offset) return ODEZZA_ERROR_INSUFFICIENT_BUFFER;
    for (index = 0u; index < 8u; ++index) {
        bytes[offset + index] = (unsigned char)(instruction.word0 >> (8u * index));
        bytes[offset + 8u + index] = (unsigned char)(instruction.word1 >> (8u * index));
    }
    return ODEZZA_SUCCESS;
}

#endif
