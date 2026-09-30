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
#ifndef CUSR_SASS_INSPECT_H_INCLUDED
#define CUSR_SASS_INSPECT_H_INCLUDED

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
#define CUSR_SASS_INSPECT_PUBLIC_DEF extern "C"
#else
#define CUSR_SASS_INSPECT_PUBLIC_DEF
#endif

#define CUSR_SASS_INSPECT_NAME_BYTES 256u
#define CUSR_SASS_INSPECT_INPUTS 8u
#define CUSR_SASS_INSPECT_MAX_OUTPUTS 64u
#define CUSR_SASS_INSPECT_MAX_SITE_INSTRUCTIONS 8192u
#define CUSR_SASS_INSPECT_SASS_INSTRUCTION_BYTES 16u
#define CUSR_SASS_INSPECT_BPT_WORD0 0x000000040000795cull
#define CUSR_SASS_INSPECT_MARKER_STRIDE 128u
#define CUSR_SASS_INSPECT_MAX_PRE_MARKER_SETUP 8u
#define CUSR_SASS_INSPECT_TARGET_MARKER_OFFSET 8u
#define CUSR_SASS_INSPECT_SSE_MARKER_OFFSET 16u
#define CUSR_SASS_INSPECT_PAD_INSTRUCTIONS_PER_AST 64u

typedef enum CusrSassInspectResult {
    CUSR_SASS_INSPECT_SUCCESS = 0,
    CUSR_SASS_INSPECT_ERROR_INVALID_VALUE = 1,
    CUSR_SASS_INSPECT_ERROR_OVERFLOW = 2,
    CUSR_SASS_INSPECT_ERROR_BAD_ELF = 3,
    CUSR_SASS_INSPECT_ERROR_FUNCTION_NOT_FOUND = 4,
    CUSR_SASS_INSPECT_ERROR_BAD_FUNCTION = 5,
    CUSR_SASS_INSPECT_ERROR_REGCOUNT_NOT_FOUND = 6,
    CUSR_SASS_INSPECT_ERROR_SITE_NOT_FOUND = 7,
    CUSR_SASS_INSPECT_ERROR_BAD_SITE = 8,
    CUSR_SASS_INSPECT_ERROR_INSUFFICIENT_WORKSPACE = 9
} CusrSassInspectResult;

typedef struct CusrSassInspectRegcountRecord {
    size_t kernel_index;
    size_t symbol_index;
    const char* section_name;
    size_t tag_file_offset;
    size_t value_file_offset;
    uint32_t value;
} CusrSassInspectRegcountRecord;

typedef struct CusrSassInspectKernel {
    char name[CUSR_SASS_INSPECT_NAME_BYTES];
    size_t kernel_index;
    size_t symbol_index;
    const char* text_section_name;
    uint64_t start_address;
    uint64_t end_address;
    size_t start_file_offset;
    size_t end_file_offset;
    size_t first_instruction;
    size_t num_instructions;
    size_t first_regcount_record;
    size_t num_regcount_records;
    size_t first_site;
    size_t num_sites;
} CusrSassInspectKernel;

typedef struct CusrSassInspectSite {
    size_t kernel_index;
    size_t occurrence_index;
    uint32_t marker;
    size_t start_instruction;
    size_t end_instruction;
    size_t start_file_offset;
    size_t end_file_offset;
    uint64_t start_address;
    uint64_t end_address;
    uint8_t input_regs[CUSR_SASS_INSPECT_INPUTS];
    uint8_t target_reg;
    uint8_t output_regs[CUSR_SASS_INSPECT_MAX_OUTPUTS];
    size_t num_output_regs;
    uint8_t available_regs[256u];
    size_t num_available_regs;
    uint32_t incoming_wait_mask;
} CusrSassInspectSite;

typedef struct CusrSassInspectHandle {
    const unsigned char* cubin;
    size_t cubin_size;
    uint32_t sass_arch;
    size_t num_kernels;
    size_t ast_capacity;
    uint32_t first_marker_bits;
    size_t expected_occurrences;
    CusrSassInspectKernel* kernels;
    size_t num_sites;
    CusrSassInspectSite* sites;
    size_t num_regcount_records;
    CusrSassInspectRegcountRecord* regcount_records;
    void* workspace;
    size_t workspace_size;
} CusrSassInspectHandle;

/*
 * Inspection recognizes the tile-static patch-site ABI: eight inputs, one
 * target, ast_capacity SSE accumulators, and one contiguous BPT pad. A zero
 * expected_occurrences accepts any positive number of compiler-created copies.
 * function_names must contain one exact ELF symbol per kernel, ordered by the
 * kernel index used to derive that kernel's marker range.
 * Function names are consumed during inspection and copied into the kernel
 * records. The handle points into the caller-owned cubin and workspace; both
 * must outlive the handle.
 */

CUSR_SASS_INSPECT_PUBLIC_DEF
const char*
cusr_sass_inspect_result_to_string(CusrSassInspectResult result);

CUSR_SASS_INSPECT_PUBLIC_DEF
CusrSassInspectResult
cusr_sass_inspect_workspace_size(
    const void* cubin,
    size_t cubin_size,
    const char* const* function_names,
    size_t num_kernels,
    size_t ast_capacity,
    uint32_t first_marker_bits,
    size_t expected_occurrences,
    size_t* workspace_size_ret
);

CUSR_SASS_INSPECT_PUBLIC_DEF
CusrSassInspectResult
cusr_sass_inspect(
    const void* cubin,
    size_t cubin_size,
    const char* const* function_names,
    size_t num_kernels,
    size_t ast_capacity,
    uint32_t first_marker_bits,
    size_t expected_occurrences,
    void* workspace,
    size_t workspace_size,
    CusrSassInspectHandle* handle_ret
);

CUSR_SASS_INSPECT_PUBLIC_DEF
void
cusr_sass_inspect_print(
    FILE* stream,
    const char* cubin_label,
    const CusrSassInspectHandle* handle
);

CUSR_SASS_INSPECT_PUBLIC_DEF
void
cusr_sass_inspect_print_json(FILE* stream, const CusrSassInspectHandle* handle);

#endif /* CUSR_SASS_INSPECT_H_INCLUDED */

#ifdef CUSR_SASS_INSPECT_IMPLEMENTATION
#ifndef CUSR_SASS_INSPECT_IMPLEMENTATION_ONCE
#define CUSR_SASS_INSPECT_IMPLEMENTATION_ONCE

#include <elf.h>

#include <inttypes.h>
#include <string.h>

#define CUSR_SASS_INSPECT_WORD_BYTES 8u
#define CUSR_SASS_INSPECT_WORKSPACE_ALIGNMENT 16u

#define CUSR_SASS_INSPECT_OPCODE_FADD_IMM 0x7421u
#define CUSR_SASS_INSPECT_OPCODE_FMUL_IMM 0x7820u
#define CUSR_SASS_INSPECT_OPCODE_FFMA_IMM 0x7423u
#define CUSR_SASS_INSPECT_OPCODE_ULEA 0x7291u
#define CUSR_SASS_INSPECT_OPCODE_UMOV 0x7882u
#define CUSR_SASS_INSPECT_OPCODE_S2UR 0x79c3u
#define CUSR_SASS_INSPECT_OPCODE_STS_REG 0x7388u
#define CUSR_SASS_INSPECT_OPCODE_STS_UR 0x7988u

#define CUSR_SASS_INSPECT_NVINFO_FORMAT_U32 0x04u
#define CUSR_SASS_INSPECT_NVINFO_ATTR_MAXREG_COUNT 0x2fu
#define CUSR_SASS_INSPECT_NVINFO_ATTR_SIZE_U32_PAIR 8u

#define _CUSR_SASS_INSPECT_ERROR_RET(ans) \
    do { \
        CusrSassInspectResult cusr_sass_inspect_result = (ans); \
        return cusr_sass_inspect_result; \
    } while (0)

#define _CUSR_SASS_INSPECT_CHECK_RET(ans) \
    do { \
        CusrSassInspectResult cusr_sass_inspect_check_ret = (ans); \
        if (cusr_sass_inspect_check_ret != CUSR_SASS_INSPECT_SUCCESS) { \
            _CUSR_SASS_INSPECT_ERROR_RET(cusr_sass_inspect_check_ret); \
        } \
    } while (0)

typedef struct CusrSassInspectElf {
    const unsigned char* data;
    size_t size;
    const Elf64_Ehdr* ehdr;
    const Elf64_Shdr* shdrs;
    const char* shstrtab;
    size_t shstrtab_size;
    const Elf64_Sym* symtab;
    size_t num_symbols;
    const char* strtab;
    size_t strtab_size;
} CusrSassInspectElf;

typedef struct CusrSassInspectFunction {
    char name[CUSR_SASS_INSPECT_NAME_BYTES];
    size_t symbol_index;
    const Elf64_Shdr* text_section;
    const char* text_section_name;
    uint64_t start_address;
    uint64_t end_address;
    size_t start_file_offset;
    size_t end_file_offset;
    size_t first_instruction;
    size_t num_instructions;
} CusrSassInspectFunction;

typedef struct CusrSassInspectCounts {
    size_t num_regcount_records;
    size_t num_sites;
} CusrSassInspectCounts;

typedef struct CusrSassInspectArena {
    unsigned char* base;
    size_t size;
    size_t offset;
} CusrSassInspectArena;

static uint16_t
cusr_sass_inspect_read_u16(const unsigned char* data)
{
    return (uint16_t)data[0] | ((uint16_t)data[1] << 8);
}

static uint32_t
cusr_sass_inspect_read_u32(const unsigned char* data)
{
    return
        (uint32_t)data[0] |
        ((uint32_t)data[1] << 8) |
        ((uint32_t)data[2] << 16) |
        ((uint32_t)data[3] << 24);
}

static uint64_t
cusr_sass_inspect_read_u64(const unsigned char* data)
{
    return
        (uint64_t)cusr_sass_inspect_read_u32(data) |
        ((uint64_t)cusr_sass_inspect_read_u32(data + 4u) << 32);
}

static uint32_t
cusr_sass_inspect_arch_from_flags(uint32_t flags)
{
    uint32_t arch = flags & 0xffu;

    if (arch >= 30u) {
        return arch;
    }

    arch = (flags >> 8) & 0xffu;
    if (arch >= 30u) {
        return arch;
    }

    return 0u;
}

static int
cusr_sass_inspect_range_ok(size_t size, size_t offset, size_t count)
{
    if (offset > size) {
        return 0;
    }

    if (count > size - offset) {
        return 0;
    }

    return 1;
}

static int
cusr_sass_inspect_checked_add(size_t a, size_t b, size_t* out)
{
    if (a > SIZE_MAX - b) {
        return 0;
    }

    *out = a + b;
    return 1;
}

static int
cusr_sass_inspect_checked_mul(size_t a, size_t b, size_t* out)
{
    if (a != 0u && b > SIZE_MAX / a) {
        return 0;
    }

    *out = a * b;
    return 1;
}

static size_t
cusr_sass_inspect_align_up(size_t value, size_t alignment)
{
    return (value + alignment - 1u) & ~(alignment - 1u);
}

static int
cusr_sass_inspect_add_aligned(size_t* total, size_t bytes)
{
    size_t aligned_total;
    size_t next_total;

    aligned_total = cusr_sass_inspect_align_up(*total, CUSR_SASS_INSPECT_WORKSPACE_ALIGNMENT);
    if (aligned_total < *total) {
        return 0;
    }

    if (!cusr_sass_inspect_checked_add(aligned_total, bytes, &next_total)) {
        return 0;
    }

    *total = next_total;
    return 1;
}

static void*
cusr_sass_inspect_arena_alloc(CusrSassInspectArena* arena, size_t bytes)
{
    uintptr_t base_address;
    uintptr_t current_address;
    uintptr_t aligned_address;
    size_t aligned_offset;
    size_t next_offset;

    base_address = (uintptr_t)arena->base;
    current_address = base_address + arena->offset;
    aligned_address = (current_address + CUSR_SASS_INSPECT_WORKSPACE_ALIGNMENT - 1u) & ~(uintptr_t)(CUSR_SASS_INSPECT_WORKSPACE_ALIGNMENT - 1u);
    aligned_offset = (size_t)(aligned_address - base_address);

    if (aligned_offset < arena->offset) {
        return NULL;
    }

    if (!cusr_sass_inspect_checked_add(aligned_offset, bytes, &next_offset)) {
        return NULL;
    }

    if (next_offset > arena->size) {
        return NULL;
    }

    arena->offset = next_offset;
    return arena->base + aligned_offset;
}

static const char*
cusr_sass_inspect_section_name(const CusrSassInspectElf* elf, const Elf64_Shdr* section)
{
    if (section->sh_name >= elf->shstrtab_size) {
        return NULL;
    }

    return elf->shstrtab + section->sh_name;
}

static const char*
cusr_sass_inspect_string_at(const char* strings, size_t strings_size, size_t offset)
{
    if (offset >= strings_size) {
        return NULL;
    }

    return strings + offset;
}

static CusrSassInspectResult
cusr_sass_inspect_validate_args(
    const void* cubin,
    size_t cubin_size,
    const char* const* function_names,
    size_t num_kernels,
    size_t ast_capacity)
{
    size_t kernel_index;

    if (cubin == NULL || cubin_size == 0u || function_names == NULL) {
        _CUSR_SASS_INSPECT_ERROR_RET(CUSR_SASS_INSPECT_ERROR_INVALID_VALUE);
    }

    if (num_kernels == 0u ||
        ast_capacity == 0u ||
        ast_capacity > CUSR_SASS_INSPECT_MAX_OUTPUTS) {
        _CUSR_SASS_INSPECT_ERROR_RET(CUSR_SASS_INSPECT_ERROR_INVALID_VALUE);
    }

    for (kernel_index = 0u; kernel_index < num_kernels; ++kernel_index) {
        if (function_names[kernel_index] == NULL || function_names[kernel_index][0] == '\0') {
            _CUSR_SASS_INSPECT_ERROR_RET(CUSR_SASS_INSPECT_ERROR_INVALID_VALUE);
        }
    }

    return CUSR_SASS_INSPECT_SUCCESS;
}

static CusrSassInspectResult
cusr_sass_inspect_parse_elf(const void* cubin, size_t cubin_size, CusrSassInspectElf* elf)
{
    size_t section_table_size;
    const Elf64_Shdr* shstr_section;
    const Elf64_Shdr* symtab_section = NULL;
    const Elf64_Shdr* strtab_section;
    size_t i;

    memset(elf, 0, sizeof(*elf));
    elf->data = (const unsigned char*)cubin;
    elf->size = cubin_size;

    if (!cusr_sass_inspect_range_ok(cubin_size, 0u, sizeof(Elf64_Ehdr))) {
        _CUSR_SASS_INSPECT_ERROR_RET(CUSR_SASS_INSPECT_ERROR_BAD_ELF);
    }

    elf->ehdr = (const Elf64_Ehdr*)elf->data;
    if (memcmp(elf->ehdr->e_ident, ELFMAG, SELFMAG) != 0 ||
        elf->ehdr->e_ident[EI_CLASS] != ELFCLASS64 ||
        elf->ehdr->e_ident[EI_DATA] != ELFDATA2LSB ||
        elf->ehdr->e_shentsize != sizeof(Elf64_Shdr) ||
        elf->ehdr->e_shnum == 0u ||
        elf->ehdr->e_shstrndx >= elf->ehdr->e_shnum) {
        _CUSR_SASS_INSPECT_ERROR_RET(CUSR_SASS_INSPECT_ERROR_BAD_ELF);
    }

    if (!cusr_sass_inspect_checked_mul((size_t)elf->ehdr->e_shnum, sizeof(Elf64_Shdr), &section_table_size) ||
        !cusr_sass_inspect_range_ok(cubin_size, (size_t)elf->ehdr->e_shoff, section_table_size)) {
        _CUSR_SASS_INSPECT_ERROR_RET(CUSR_SASS_INSPECT_ERROR_BAD_ELF);
    }

    elf->shdrs = (const Elf64_Shdr*)(elf->data + elf->ehdr->e_shoff);
    shstr_section = elf->shdrs + elf->ehdr->e_shstrndx;
    if (!cusr_sass_inspect_range_ok(cubin_size, (size_t)shstr_section->sh_offset, (size_t)shstr_section->sh_size)) {
        _CUSR_SASS_INSPECT_ERROR_RET(CUSR_SASS_INSPECT_ERROR_BAD_ELF);
    }

    elf->shstrtab = (const char*)elf->data + shstr_section->sh_offset;
    elf->shstrtab_size = (size_t)shstr_section->sh_size;

    for (i = 0u; i < (size_t)elf->ehdr->e_shnum; ++i) {
        if (elf->shdrs[i].sh_type == SHT_SYMTAB) {
            symtab_section = elf->shdrs + i;
            break;
        }
    }

    if (symtab_section == NULL ||
        symtab_section->sh_entsize != sizeof(Elf64_Sym) ||
        symtab_section->sh_link >= elf->ehdr->e_shnum) {
        _CUSR_SASS_INSPECT_ERROR_RET(CUSR_SASS_INSPECT_ERROR_BAD_ELF);
    }

    strtab_section = elf->shdrs + symtab_section->sh_link;
    if (!cusr_sass_inspect_range_ok(cubin_size, (size_t)symtab_section->sh_offset, (size_t)symtab_section->sh_size) ||
        !cusr_sass_inspect_range_ok(cubin_size, (size_t)strtab_section->sh_offset, (size_t)strtab_section->sh_size)) {
        _CUSR_SASS_INSPECT_ERROR_RET(CUSR_SASS_INSPECT_ERROR_BAD_ELF);
    }

    elf->symtab = (const Elf64_Sym*)(elf->data + symtab_section->sh_offset);
    elf->num_symbols = (size_t)(symtab_section->sh_size / symtab_section->sh_entsize);
    elf->strtab = (const char*)elf->data + strtab_section->sh_offset;
    elf->strtab_size = (size_t)strtab_section->sh_size;

    return CUSR_SASS_INSPECT_SUCCESS;
}

static CusrSassInspectResult
cusr_sass_inspect_find_function(const CusrSassInspectElf* elf, const char* name, CusrSassInspectFunction* function)
{
    size_t i;

    memset(function, 0, sizeof(*function));

    for (i = 0u; i < elf->num_symbols; ++i) {
        const Elf64_Sym* symbol = elf->symtab + i;
        const char* symbol_name = cusr_sass_inspect_string_at(elf->strtab, elf->strtab_size, symbol->st_name);

        if (symbol_name != NULL &&
            strcmp(symbol_name, name) == 0 &&
            ELF64_ST_TYPE(symbol->st_info) == STT_FUNC) {
            const Elf64_Shdr* text_section;
            const char* section_name;
            size_t symbol_value = (size_t)symbol->st_value;
            size_t symbol_size = (size_t)symbol->st_size;

            if (symbol->st_shndx == SHN_UNDEF || symbol->st_shndx >= elf->ehdr->e_shnum) {
                _CUSR_SASS_INSPECT_ERROR_RET(CUSR_SASS_INSPECT_ERROR_BAD_FUNCTION);
            }

            text_section = elf->shdrs + symbol->st_shndx;
            section_name = cusr_sass_inspect_section_name(elf, text_section);
            if (section_name == NULL) {
                _CUSR_SASS_INSPECT_ERROR_RET(CUSR_SASS_INSPECT_ERROR_BAD_FUNCTION);
            }

            if (symbol_size == 0u || (symbol_size % CUSR_SASS_INSPECT_SASS_INSTRUCTION_BYTES) != 0u) {
                _CUSR_SASS_INSPECT_ERROR_RET(CUSR_SASS_INSPECT_ERROR_BAD_FUNCTION);
            }

            if (symbol_value > (size_t)text_section->sh_size ||
                symbol_size > (size_t)text_section->sh_size - symbol_value ||
                !cusr_sass_inspect_range_ok(elf->size, (size_t)text_section->sh_offset + symbol_value, symbol_size)) {
                _CUSR_SASS_INSPECT_ERROR_RET(CUSR_SASS_INSPECT_ERROR_BAD_FUNCTION);
            }

            if ((((size_t)text_section->sh_offset + symbol_value) % CUSR_SASS_INSPECT_SASS_INSTRUCTION_BYTES) != 0u) {
                _CUSR_SASS_INSPECT_ERROR_RET(CUSR_SASS_INSPECT_ERROR_BAD_FUNCTION);
            }

            snprintf(function->name, sizeof(function->name), "%s", name);
            function->symbol_index = i;
            function->text_section = text_section;
            function->text_section_name = section_name;
            function->start_address = (uint64_t)text_section->sh_addr + symbol->st_value;
            function->end_address = function->start_address + symbol->st_size;
            function->start_file_offset = (size_t)text_section->sh_offset + symbol_value;
            function->end_file_offset = function->start_file_offset + symbol_size;
            function->first_instruction = function->start_file_offset / CUSR_SASS_INSPECT_SASS_INSTRUCTION_BYTES;
            function->num_instructions = symbol_size / CUSR_SASS_INSPECT_SASS_INSTRUCTION_BYTES;
            return CUSR_SASS_INSPECT_SUCCESS;
        }
    }

    _CUSR_SASS_INSPECT_ERROR_RET(CUSR_SASS_INSPECT_ERROR_FUNCTION_NOT_FOUND);
}

static uint64_t
cusr_sass_inspect_instruction_word0(const CusrSassInspectElf* elf, size_t instruction_file_offset)
{
    return cusr_sass_inspect_read_u64(elf->data + instruction_file_offset);
}

static uint64_t
cusr_sass_inspect_instruction_word1(const CusrSassInspectElf* elf, size_t instruction_file_offset)
{
    return cusr_sass_inspect_read_u64(elf->data + instruction_file_offset + CUSR_SASS_INSPECT_WORD_BYTES);
}

static uint32_t
cusr_sass_inspect_opcode(uint64_t word0)
{
    return (uint32_t)(word0 & 0xffffu);
}

static uint8_t
cusr_sass_inspect_dst_reg(uint64_t word0)
{
    return (uint8_t)((word0 >> 16) & 0xffu);
}

static uint8_t
cusr_sass_inspect_src0_reg(uint64_t word0)
{
    return (uint8_t)((word0 >> 24) & 0xffu);
}

static uint32_t
cusr_sass_inspect_imm_bits(uint64_t word0)
{
    return (uint32_t)((word0 >> 32) & 0xffffffffu);
}

static uint32_t
cusr_sass_inspect_wait_mask(uint64_t word1)
{
    return (uint32_t)((word1 >> 52) & 0xfffu);
}

static int
cusr_sass_inspect_is_bpt(uint64_t word0)
{
    return word0 == CUSR_SASS_INSPECT_BPT_WORD0;
}

static int
cusr_sass_inspect_is_sts(uint64_t word0)
{
    const uint32_t opcode = cusr_sass_inspect_opcode(word0);
    return opcode == CUSR_SASS_INSPECT_OPCODE_STS_REG ||
           opcode == CUSR_SASS_INSPECT_OPCODE_STS_UR;
}

static int
cusr_sass_inspect_is_pre_bpt_anchor_setup(uint64_t word0)
{
    const uint32_t opcode = cusr_sass_inspect_opcode(word0);
    return opcode == CUSR_SASS_INSPECT_OPCODE_ULEA ||
           opcode == CUSR_SASS_INSPECT_OPCODE_UMOV ||
           opcode == CUSR_SASS_INSPECT_OPCODE_S2UR;
}

static void
cusr_sass_inspect_add_available_reg(CusrSassInspectSite* site, uint8_t reg)
{
    size_t i;

    for (i = 0u; i < site->num_available_regs; ++i) {
        if (site->available_regs[i] == reg) {
            return;
        }
    }

    if (site->num_available_regs < 256u) {
        site->available_regs[site->num_available_regs] = reg;
        site->num_available_regs += 1u;
    }
}

static int
cusr_sass_inspect_site_reg_is_reserved(const CusrSassInspectSite* site, uint8_t reg)
{
    size_t i;

    if (reg == site->target_reg || reg == UINT8_MAX) {
        return 1;
    }
    for (i = 0u; i < CUSR_SASS_INSPECT_INPUTS; ++i) {
        if (reg == site->input_regs[i]) {
            return 1;
        }
    }
    for (i = 0u; i < site->num_output_regs; ++i) {
        if (reg == site->output_regs[i]) {
            return 1;
        }
    }

    return 0;
}

static void
cusr_sass_inspect_remove_reserved_available_regs(CusrSassInspectSite* site)
{
    size_t read_idx;
    size_t write_idx = 0u;

    for (read_idx = 0u; read_idx < site->num_available_regs; ++read_idx) {
        const uint8_t reg = site->available_regs[read_idx];
        if (!cusr_sass_inspect_site_reg_is_reserved(site, reg)) {
            site->available_regs[write_idx++] = reg;
        }
    }
    site->num_available_regs = write_idx;
}

static CusrSassInspectResult
cusr_sass_inspect_find_site_start(
    const CusrSassInspectElf* elf,
    size_t function_first_instruction,
    size_t marker_instruction,
    size_t* site_start_ret)
{
    size_t i;

    if (marker_instruction <= function_first_instruction) {
        _CUSR_SASS_INSPECT_ERROR_RET(CUSR_SASS_INSPECT_ERROR_BAD_SITE);
    }

    i = marker_instruction;
    while (i > function_first_instruction) {
        size_t instruction_file_offset;
        uint64_t word0;

        i -= 1u;
        instruction_file_offset = i * CUSR_SASS_INSPECT_SASS_INSTRUCTION_BYTES;
        word0 = cusr_sass_inspect_instruction_word0(elf, instruction_file_offset);

        if (cusr_sass_inspect_is_bpt(word0)) {
            *site_start_ret = i;
            _CUSR_SASS_INSPECT_ERROR_RET(CUSR_SASS_INSPECT_SUCCESS);
        }

        if (marker_instruction - i > CUSR_SASS_INSPECT_MAX_PRE_MARKER_SETUP ||
            !cusr_sass_inspect_is_pre_bpt_anchor_setup(word0)) {
            _CUSR_SASS_INSPECT_ERROR_RET(CUSR_SASS_INSPECT_ERROR_BAD_SITE);
        }
    }

    _CUSR_SASS_INSPECT_ERROR_RET(CUSR_SASS_INSPECT_ERROR_BAD_SITE);
}

static CusrSassInspectResult
cusr_sass_inspect_scan_one_site_occurrence(
    const CusrSassInspectElf* elf,
    const CusrSassInspectFunction* function,
    size_t kernel_index,
    uint32_t marker,
    size_t ast_capacity,
    size_t found_instruction,
    size_t occurrence_index,
    CusrSassInspectSite* site_ret)
{
    size_t function_first_instruction = function->start_file_offset / CUSR_SASS_INSPECT_SASS_INSTRUCTION_BYTES;
    size_t function_last_instruction = function_first_instruction + function->num_instructions;
    CusrSassInspectSite site;
    size_t site_start_instruction;
    size_t last_role_instruction = 0u;
    size_t site_end_instruction;
    size_t store_count = 0u;
    size_t trailing_bpt_count = 0u;
    size_t expected_trailing_bpt_count;
    uint8_t input_seen[CUSR_SASS_INSPECT_INPUTS] = { 0u };
    uint8_t output_seen[CUSR_SASS_INSPECT_MAX_OUTPUTS] = { 0u };
    uint8_t target_seen = 0u;
    size_t i;

    memset(&site, 0, sizeof(site));
    site.kernel_index = kernel_index;
    site.occurrence_index = occurrence_index;
    site.marker = marker;
    site.num_output_regs = ast_capacity;
    expected_trailing_bpt_count = ast_capacity * CUSR_SASS_INSPECT_PAD_INSTRUCTIONS_PER_AST + 1u;

    if (found_instruction < function_first_instruction ||
        found_instruction >= function_last_instruction) {
        _CUSR_SASS_INSPECT_ERROR_RET(CUSR_SASS_INSPECT_ERROR_BAD_SITE);
    }

    _CUSR_SASS_INSPECT_CHECK_RET(cusr_sass_inspect_find_site_start(
        elf,
        function_first_instruction,
        found_instruction,
        &site_start_instruction
    ));

    site.start_instruction = site_start_instruction;
    site.start_file_offset = site_start_instruction * CUSR_SASS_INSPECT_SASS_INSTRUCTION_BYTES;
    site.start_address = function->start_address + (uint64_t)(site.start_file_offset - function->start_file_offset);

    for (i = site_start_instruction;
         i < function_last_instruction && i < site_start_instruction + CUSR_SASS_INSPECT_MAX_SITE_INSTRUCTIONS;
         ++i) {
        size_t instruction_file_offset = i * CUSR_SASS_INSPECT_SASS_INSTRUCTION_BYTES;
        uint64_t word0 = cusr_sass_inspect_instruction_word0(elf, instruction_file_offset);
        uint32_t opcode = cusr_sass_inspect_opcode(word0);

        if (opcode == CUSR_SASS_INSPECT_OPCODE_FADD_IMM) {
            const uint32_t immediate = cusr_sass_inspect_imm_bits(word0);

            if (immediate >= marker && immediate - marker < CUSR_SASS_INSPECT_MARKER_STRIDE) {
                const uint32_t role = immediate - marker;
                const uint8_t dst = cusr_sass_inspect_dst_reg(word0);
                const uint8_t src = cusr_sass_inspect_src0_reg(word0);
                const uint64_t word1 = cusr_sass_inspect_instruction_word1(elf, instruction_file_offset);

                if (role < CUSR_SASS_INSPECT_INPUTS) {
                    if (input_seen[role]) {
                        _CUSR_SASS_INSPECT_ERROR_RET(CUSR_SASS_INSPECT_ERROR_BAD_SITE);
                    }
                    input_seen[role] = 1u;
                    site.input_regs[role] = src;
                    site.incoming_wait_mask |= cusr_sass_inspect_wait_mask(word1);
                    cusr_sass_inspect_add_available_reg(&site, dst);
                } else if (role == CUSR_SASS_INSPECT_TARGET_MARKER_OFFSET) {
                    if (target_seen) {
                        _CUSR_SASS_INSPECT_ERROR_RET(CUSR_SASS_INSPECT_ERROR_BAD_SITE);
                    }
                    target_seen = 1u;
                    site.target_reg = src;
                    site.incoming_wait_mask |= cusr_sass_inspect_wait_mask(word1);
                    cusr_sass_inspect_add_available_reg(&site, dst);
                } else if (role >= CUSR_SASS_INSPECT_SSE_MARKER_OFFSET &&
                           role < CUSR_SASS_INSPECT_SSE_MARKER_OFFSET + ast_capacity) {
                    const size_t output_idx = role - CUSR_SASS_INSPECT_SSE_MARKER_OFFSET;
                    if (output_seen[output_idx] || dst != src) {
                        _CUSR_SASS_INSPECT_ERROR_RET(CUSR_SASS_INSPECT_ERROR_BAD_SITE);
                    }
                    output_seen[output_idx] = 1u;
                    site.output_regs[output_idx] = dst;
                } else {
                    _CUSR_SASS_INSPECT_ERROR_RET(CUSR_SASS_INSPECT_ERROR_BAD_SITE);
                }

                last_role_instruction = i;
                trailing_bpt_count = 0u;
                continue;
            }
        }

        if (cusr_sass_inspect_is_sts(word0)) {
            store_count += 1u;
            trailing_bpt_count = 0u;
            continue;
        }

        if (cusr_sass_inspect_is_bpt(word0)) {
            if (last_role_instruction != 0u && i > last_role_instruction) {
                trailing_bpt_count += 1u;
            }
            continue;
        }

        if (cusr_sass_inspect_is_pre_bpt_anchor_setup(word0) &&
            store_count < CUSR_SASS_INSPECT_INPUTS + 1u) {
            trailing_bpt_count = 0u;
            continue;
        }

        if (last_role_instruction != 0u && trailing_bpt_count != 0u) {
            break;
        }

        _CUSR_SASS_INSPECT_ERROR_RET(CUSR_SASS_INSPECT_ERROR_BAD_SITE);
    }

    if (store_count != CUSR_SASS_INSPECT_INPUTS + 1u ||
        !target_seen ||
        trailing_bpt_count != expected_trailing_bpt_count) {
        _CUSR_SASS_INSPECT_ERROR_RET(CUSR_SASS_INSPECT_ERROR_BAD_SITE);
    }
    site_end_instruction = i;

    for (i = 0u; i < CUSR_SASS_INSPECT_INPUTS; ++i) {
        if (!input_seen[i]) {
            _CUSR_SASS_INSPECT_ERROR_RET(CUSR_SASS_INSPECT_ERROR_BAD_SITE);
        }
    }
    for (i = 0u; i < ast_capacity; ++i) {
        if (!output_seen[i]) {
            _CUSR_SASS_INSPECT_ERROR_RET(CUSR_SASS_INSPECT_ERROR_BAD_SITE);
        }
    }

    for (i = 0u; i < CUSR_SASS_INSPECT_INPUTS; ++i) {
        size_t j;
        for (j = 0u; j < i; ++j) {
            if (site.input_regs[i] == site.input_regs[j]) {
                _CUSR_SASS_INSPECT_ERROR_RET(CUSR_SASS_INSPECT_ERROR_BAD_SITE);
            }
        }
        for (j = 0u; j < ast_capacity; ++j) {
            if (site.input_regs[i] == site.output_regs[j]) {
                _CUSR_SASS_INSPECT_ERROR_RET(CUSR_SASS_INSPECT_ERROR_BAD_SITE);
            }
        }
    }
    for (i = 0u; i < ast_capacity; ++i) {
        size_t j;
        if (site.output_regs[i] == site.target_reg) {
            _CUSR_SASS_INSPECT_ERROR_RET(CUSR_SASS_INSPECT_ERROR_BAD_SITE);
        }
        for (j = 0u; j < i; ++j) {
            if (site.output_regs[i] == site.output_regs[j]) {
                _CUSR_SASS_INSPECT_ERROR_RET(CUSR_SASS_INSPECT_ERROR_BAD_SITE);
            }
        }
    }

    cusr_sass_inspect_remove_reserved_available_regs(&site);

    site.end_instruction = site_end_instruction;
    site.end_file_offset = site_end_instruction * CUSR_SASS_INSPECT_SASS_INSTRUCTION_BYTES;
    site.end_address = function->start_address + (uint64_t)(site.end_file_offset - function->start_file_offset);

    if (site.end_instruction <= site.start_instruction ||
        site.end_instruction > function_last_instruction ||
        site.end_instruction - site.start_instruction > CUSR_SASS_INSPECT_MAX_SITE_INSTRUCTIONS) {
        _CUSR_SASS_INSPECT_ERROR_RET(CUSR_SASS_INSPECT_ERROR_BAD_SITE);
    }

    if (site.end_instruction == site.start_instruction + CUSR_SASS_INSPECT_MAX_SITE_INSTRUCTIONS &&
        site.end_instruction < function_last_instruction) {
        const size_t next_file_offset = site.end_instruction * CUSR_SASS_INSPECT_SASS_INSTRUCTION_BYTES;
        if (cusr_sass_inspect_is_bpt(cusr_sass_inspect_instruction_word0(elf, next_file_offset))) {
            _CUSR_SASS_INSPECT_ERROR_RET(CUSR_SASS_INSPECT_ERROR_BAD_SITE);
        }
    }

    if (site_ret != NULL) {
        *site_ret = site;
    }

    return CUSR_SASS_INSPECT_SUCCESS;
}

static CusrSassInspectResult
cusr_sass_inspect_collect_regcounts(
    const CusrSassInspectElf* elf,
    size_t kernel_index,
    size_t symbol_index,
    CusrSassInspectRegcountRecord* records,
    size_t records_capacity,
    size_t* records_count_ret,
    size_t* records_for_kernel_ret)
{
    size_t section_index;
    size_t records_for_kernel = 0u;
    size_t records_count = *records_count_ret;

    for (section_index = 0u; section_index < (size_t)elf->ehdr->e_shnum; ++section_index) {
        const Elf64_Shdr* section = elf->shdrs + section_index;
        const char* section_name = cusr_sass_inspect_section_name(elf, section);
        size_t section_offset;

        if (section_name == NULL || strstr(section_name, ".nv.info") == NULL) {
            continue;
        }

        if (!cusr_sass_inspect_range_ok(elf->size, (size_t)section->sh_offset, (size_t)section->sh_size)) {
            _CUSR_SASS_INSPECT_ERROR_RET(CUSR_SASS_INSPECT_ERROR_BAD_ELF);
        }

        for (section_offset = 0u; section_offset + 12u <= (size_t)section->sh_size; section_offset += 4u) {
            size_t file_offset = (size_t)section->sh_offset + section_offset;
            const unsigned char* bytes = elf->data + file_offset;
            uint32_t payload_symbol_index;

            if (bytes[0] != CUSR_SASS_INSPECT_NVINFO_FORMAT_U32 ||
                bytes[1] != CUSR_SASS_INSPECT_NVINFO_ATTR_MAXREG_COUNT ||
                cusr_sass_inspect_read_u16(bytes + 2u) != CUSR_SASS_INSPECT_NVINFO_ATTR_SIZE_U32_PAIR) {
                continue;
            }

            payload_symbol_index = cusr_sass_inspect_read_u32(bytes + 4u);
            if (payload_symbol_index != symbol_index) {
                continue;
            }

            if (records != NULL) {
                CusrSassInspectRegcountRecord* record;

                if (records_count >= records_capacity) {
                    _CUSR_SASS_INSPECT_ERROR_RET(CUSR_SASS_INSPECT_ERROR_INSUFFICIENT_WORKSPACE);
                }

                record = records + records_count;
                record->kernel_index = kernel_index;
                record->symbol_index = symbol_index;
                record->section_name = section_name;
                record->tag_file_offset = file_offset;
                record->value_file_offset = file_offset + 8u;
                record->value = cusr_sass_inspect_read_u32(bytes + 8u);
            }

            records_count += 1u;
            records_for_kernel += 1u;
        }
    }

    if (records_for_kernel == 0u) {
        _CUSR_SASS_INSPECT_ERROR_RET(CUSR_SASS_INSPECT_ERROR_REGCOUNT_NOT_FOUND);
    }

    *records_count_ret = records_count;
    *records_for_kernel_ret = records_for_kernel;
    return CUSR_SASS_INSPECT_SUCCESS;
}

static CusrSassInspectResult
cusr_sass_inspect_scan_site(
    const CusrSassInspectElf* elf,
    const CusrSassInspectFunction* function,
    size_t kernel_index,
    uint32_t marker,
    size_t ast_capacity,
    size_t expected_occurrences,
    CusrSassInspectSite* sites,
    size_t sites_capacity,
    size_t* sites_count_ret,
    size_t* occurrences_ret)
{
    size_t first_instruction = function->start_file_offset / CUSR_SASS_INSPECT_SASS_INSTRUCTION_BYTES;
    size_t last_instruction = first_instruction + function->num_instructions;
    size_t search_instruction = first_instruction;
    size_t occurrence_index = 0u;
    size_t sites_count = *sites_count_ret;

    while (search_instruction < last_instruction) {
        size_t found_instruction = SIZE_MAX;
        size_t i;
        CusrSassInspectSite site;

        for (i = search_instruction; i < last_instruction; ++i) {
            size_t instruction_file_offset = i * CUSR_SASS_INSPECT_SASS_INSTRUCTION_BYTES;
            uint64_t word0 = cusr_sass_inspect_instruction_word0(elf, instruction_file_offset);

            if (cusr_sass_inspect_opcode(word0) == CUSR_SASS_INSPECT_OPCODE_FADD_IMM &&
                cusr_sass_inspect_imm_bits(word0) == marker) {
                found_instruction = i;
                break;
            }
        }

        if (found_instruction == SIZE_MAX) {
            break;
        }

        _CUSR_SASS_INSPECT_CHECK_RET(cusr_sass_inspect_scan_one_site_occurrence(
            elf,
            function,
            kernel_index,
            marker,
            ast_capacity,
            found_instruction,
            occurrence_index,
            &site
        ));

        if (sites != NULL) {
            if (sites_count >= sites_capacity) {
                _CUSR_SASS_INSPECT_ERROR_RET(CUSR_SASS_INSPECT_ERROR_INSUFFICIENT_WORKSPACE);
            }

            sites[sites_count] = site;
        }

        sites_count += 1u;
        occurrence_index += 1u;
        search_instruction = site.end_instruction;
    }

    if (occurrence_index == 0u) {
        _CUSR_SASS_INSPECT_ERROR_RET(CUSR_SASS_INSPECT_ERROR_SITE_NOT_FOUND);
    }

    if (expected_occurrences != 0u && occurrence_index != expected_occurrences) {
        _CUSR_SASS_INSPECT_ERROR_RET(CUSR_SASS_INSPECT_ERROR_BAD_SITE);
    }

    *sites_count_ret = sites_count;
    *occurrences_ret = occurrence_index;
    return CUSR_SASS_INSPECT_SUCCESS;
}

static CusrSassInspectResult
cusr_sass_inspect_scan_kernel(
    const CusrSassInspectElf* elf,
    const char* function_name,
    size_t kernel_index,
    size_t ast_capacity,
    uint32_t first_marker_bits,
    size_t expected_occurrences,
    CusrSassInspectKernel* kernels,
    CusrSassInspectRegcountRecord* regcount_records,
    size_t regcount_capacity,
    size_t* regcount_count_ret,
    CusrSassInspectSite* sites,
    size_t sites_capacity,
    size_t* sites_count_ret)
{
    CusrSassInspectFunction function;
    size_t regcount_first = *regcount_count_ret;
    size_t regcount_for_kernel = 0u;
    size_t site_first = *sites_count_ret;
    uint32_t marker;
    size_t marker_offset;
    size_t occurrences = 0u;

    _CUSR_SASS_INSPECT_CHECK_RET(cusr_sass_inspect_find_function(elf, function_name, &function));
    _CUSR_SASS_INSPECT_CHECK_RET(cusr_sass_inspect_collect_regcounts(
        elf,
        kernel_index,
        function.symbol_index,
        regcount_records,
        regcount_capacity,
        regcount_count_ret,
        &regcount_for_kernel
    ));

    if (!cusr_sass_inspect_checked_mul(kernel_index, CUSR_SASS_INSPECT_MARKER_STRIDE, &marker_offset) ||
        marker_offset > (size_t)UINT32_MAX - (size_t)first_marker_bits) {
        _CUSR_SASS_INSPECT_ERROR_RET(CUSR_SASS_INSPECT_ERROR_OVERFLOW);
    }

    marker = first_marker_bits + (uint32_t)marker_offset;
    _CUSR_SASS_INSPECT_CHECK_RET(cusr_sass_inspect_scan_site(
        elf,
        &function,
        kernel_index,
        marker,
        ast_capacity,
        expected_occurrences,
        sites,
        sites_capacity,
        sites_count_ret,
        &occurrences
    ));

    if (strlen(function.name) + 1u > CUSR_SASS_INSPECT_NAME_BYTES) {
        _CUSR_SASS_INSPECT_ERROR_RET(CUSR_SASS_INSPECT_ERROR_BAD_FUNCTION);
    }

    if (kernels != NULL) {
        CusrSassInspectKernel* kernel = kernels + kernel_index;
        const size_t function_name_size = strlen(function.name) + 1u;

        memset(kernel, 0, sizeof(*kernel));
        memcpy(kernel->name, function.name, function_name_size);
        kernel->kernel_index = kernel_index;
        kernel->symbol_index = function.symbol_index;
        kernel->text_section_name = function.text_section_name;
        kernel->start_address = function.start_address;
        kernel->end_address = function.end_address;
        kernel->start_file_offset = function.start_file_offset;
        kernel->end_file_offset = function.end_file_offset;
        kernel->first_instruction = function.first_instruction;
        kernel->num_instructions = function.num_instructions;
        kernel->first_regcount_record = regcount_first;
        kernel->num_regcount_records = regcount_for_kernel;
        kernel->first_site = site_first;
        kernel->num_sites = *sites_count_ret - site_first;
    }

    return CUSR_SASS_INSPECT_SUCCESS;
}

static CusrSassInspectResult
cusr_sass_inspect_count(
    const CusrSassInspectElf* elf,
    const char* const* function_names,
    size_t num_kernels,
    size_t ast_capacity,
    uint32_t first_marker_bits,
    size_t expected_occurrences,
    CusrSassInspectCounts* counts)
{
    size_t kernel_index;
    size_t regcount_count = 0u;
    size_t sites_count = 0u;

    memset(counts, 0, sizeof(*counts));

    for (kernel_index = 0u; kernel_index < num_kernels; ++kernel_index) {
        _CUSR_SASS_INSPECT_CHECK_RET(cusr_sass_inspect_scan_kernel(
            elf,
            function_names[kernel_index],
            kernel_index,
            ast_capacity,
            first_marker_bits,
            expected_occurrences,
            NULL,
            NULL,
            0u,
            &regcount_count,
            NULL,
            0u,
            &sites_count
        ));
    }

    counts->num_regcount_records = regcount_count;
    counts->num_sites = sites_count;
    return CUSR_SASS_INSPECT_SUCCESS;
}

CUSR_SASS_INSPECT_PUBLIC_DEF
const char*
cusr_sass_inspect_result_to_string(CusrSassInspectResult result)
{
    switch (result) {
        case CUSR_SASS_INSPECT_SUCCESS:
            return "CUSR_SASS_INSPECT_SUCCESS";
        case CUSR_SASS_INSPECT_ERROR_INVALID_VALUE:
            return "CUSR_SASS_INSPECT_ERROR_INVALID_VALUE";
        case CUSR_SASS_INSPECT_ERROR_OVERFLOW:
            return "CUSR_SASS_INSPECT_ERROR_OVERFLOW";
        case CUSR_SASS_INSPECT_ERROR_BAD_ELF:
            return "CUSR_SASS_INSPECT_ERROR_BAD_ELF";
        case CUSR_SASS_INSPECT_ERROR_FUNCTION_NOT_FOUND:
            return "CUSR_SASS_INSPECT_ERROR_FUNCTION_NOT_FOUND";
        case CUSR_SASS_INSPECT_ERROR_BAD_FUNCTION:
            return "CUSR_SASS_INSPECT_ERROR_BAD_FUNCTION";
        case CUSR_SASS_INSPECT_ERROR_REGCOUNT_NOT_FOUND:
            return "CUSR_SASS_INSPECT_ERROR_REGCOUNT_NOT_FOUND";
        case CUSR_SASS_INSPECT_ERROR_SITE_NOT_FOUND:
            return "CUSR_SASS_INSPECT_ERROR_SITE_NOT_FOUND";
        case CUSR_SASS_INSPECT_ERROR_BAD_SITE:
            return "CUSR_SASS_INSPECT_ERROR_BAD_SITE";
        case CUSR_SASS_INSPECT_ERROR_INSUFFICIENT_WORKSPACE:
            return "CUSR_SASS_INSPECT_ERROR_INSUFFICIENT_WORKSPACE";
    }

    return "CUSR_SASS_INSPECT_ERROR_UNKNOWN";
}

CUSR_SASS_INSPECT_PUBLIC_DEF
CusrSassInspectResult
cusr_sass_inspect_workspace_size(
    const void* cubin,
    size_t cubin_size,
    const char* const* function_names,
    size_t num_kernels,
    size_t ast_capacity,
    uint32_t first_marker_bits,
    size_t expected_occurrences,
    size_t* workspace_size_ret)
{
    CusrSassInspectElf elf;
    CusrSassInspectCounts counts;
    size_t bytes = CUSR_SASS_INSPECT_WORKSPACE_ALIGNMENT;
    size_t array_bytes;

    if (workspace_size_ret == NULL) {
        _CUSR_SASS_INSPECT_ERROR_RET(CUSR_SASS_INSPECT_ERROR_INVALID_VALUE);
    }
    *workspace_size_ret = 0u;

    _CUSR_SASS_INSPECT_CHECK_RET(cusr_sass_inspect_validate_args(
        cubin,
        cubin_size,
        function_names,
        num_kernels,
        ast_capacity
    ));
    _CUSR_SASS_INSPECT_CHECK_RET(cusr_sass_inspect_parse_elf(cubin, cubin_size, &elf));
    _CUSR_SASS_INSPECT_CHECK_RET(cusr_sass_inspect_count(
        &elf,
        function_names,
        num_kernels,
        ast_capacity,
        first_marker_bits,
        expected_occurrences,
        &counts
    ));

    if (!cusr_sass_inspect_checked_mul(num_kernels, sizeof(CusrSassInspectKernel), &array_bytes) ||
        !cusr_sass_inspect_add_aligned(&bytes, array_bytes) ||
        !cusr_sass_inspect_checked_mul(counts.num_regcount_records, sizeof(CusrSassInspectRegcountRecord), &array_bytes) ||
        !cusr_sass_inspect_add_aligned(&bytes, array_bytes) ||
        !cusr_sass_inspect_checked_mul(counts.num_sites, sizeof(CusrSassInspectSite), &array_bytes) ||
        !cusr_sass_inspect_add_aligned(&bytes, array_bytes)) {
        _CUSR_SASS_INSPECT_ERROR_RET(CUSR_SASS_INSPECT_ERROR_OVERFLOW);
    }

    *workspace_size_ret = bytes;
    return CUSR_SASS_INSPECT_SUCCESS;
}

CUSR_SASS_INSPECT_PUBLIC_DEF
CusrSassInspectResult
cusr_sass_inspect(
    const void* cubin,
    size_t cubin_size,
    const char* const* function_names,
    size_t num_kernels,
    size_t ast_capacity,
    uint32_t first_marker_bits,
    size_t expected_occurrences,
    void* workspace,
    size_t workspace_size,
    CusrSassInspectHandle* handle_ret)
{
    CusrSassInspectElf elf;
    CusrSassInspectCounts counts;
    CusrSassInspectArena arena;
    CusrSassInspectKernel* kernels;
    CusrSassInspectRegcountRecord* regcount_records;
    CusrSassInspectSite* sites;
    size_t regcount_count = 0u;
    size_t sites_count = 0u;
    size_t kernel_index;
    size_t array_bytes;

    if (handle_ret == NULL || workspace == NULL) {
        _CUSR_SASS_INSPECT_ERROR_RET(CUSR_SASS_INSPECT_ERROR_INVALID_VALUE);
    }

    memset(handle_ret, 0, sizeof(*handle_ret));

    _CUSR_SASS_INSPECT_CHECK_RET(cusr_sass_inspect_validate_args(
        cubin,
        cubin_size,
        function_names,
        num_kernels,
        ast_capacity
    ));
    _CUSR_SASS_INSPECT_CHECK_RET(cusr_sass_inspect_parse_elf(cubin, cubin_size, &elf));
    _CUSR_SASS_INSPECT_CHECK_RET(cusr_sass_inspect_count(
        &elf,
        function_names,
        num_kernels,
        ast_capacity,
        first_marker_bits,
        expected_occurrences,
        &counts
    ));

    arena.base = (unsigned char*)workspace;
    arena.size = workspace_size;
    arena.offset = 0u;

    if (!cusr_sass_inspect_checked_mul(num_kernels, sizeof(CusrSassInspectKernel), &array_bytes)) {
        _CUSR_SASS_INSPECT_ERROR_RET(CUSR_SASS_INSPECT_ERROR_OVERFLOW);
    }
    kernels = (CusrSassInspectKernel*)cusr_sass_inspect_arena_alloc(&arena, array_bytes);

    if (!cusr_sass_inspect_checked_mul(counts.num_regcount_records, sizeof(CusrSassInspectRegcountRecord), &array_bytes)) {
        _CUSR_SASS_INSPECT_ERROR_RET(CUSR_SASS_INSPECT_ERROR_OVERFLOW);
    }
    regcount_records = (CusrSassInspectRegcountRecord*)cusr_sass_inspect_arena_alloc(&arena, array_bytes);

    if (!cusr_sass_inspect_checked_mul(counts.num_sites, sizeof(CusrSassInspectSite), &array_bytes)) {
        _CUSR_SASS_INSPECT_ERROR_RET(CUSR_SASS_INSPECT_ERROR_OVERFLOW);
    }
    sites = (CusrSassInspectSite*)cusr_sass_inspect_arena_alloc(&arena, array_bytes);

    if (kernels == NULL || regcount_records == NULL || sites == NULL) {
        _CUSR_SASS_INSPECT_ERROR_RET(CUSR_SASS_INSPECT_ERROR_INSUFFICIENT_WORKSPACE);
    }

    memset(kernels, 0, num_kernels * sizeof(*kernels));
    memset(regcount_records, 0, counts.num_regcount_records * sizeof(*regcount_records));
    memset(sites, 0, counts.num_sites * sizeof(*sites));

    for (kernel_index = 0u; kernel_index < num_kernels; ++kernel_index) {
        _CUSR_SASS_INSPECT_CHECK_RET(cusr_sass_inspect_scan_kernel(
            &elf,
            function_names[kernel_index],
            kernel_index,
            ast_capacity,
            first_marker_bits,
            expected_occurrences,
            kernels,
            regcount_records,
            counts.num_regcount_records,
            &regcount_count,
            sites,
            counts.num_sites,
            &sites_count
        ));
    }

    handle_ret->cubin = (const unsigned char*)cubin;
    handle_ret->cubin_size = cubin_size;
    handle_ret->sass_arch = cusr_sass_inspect_arch_from_flags(elf.ehdr->e_flags);
    handle_ret->num_kernels = num_kernels;
    handle_ret->ast_capacity = ast_capacity;
    handle_ret->first_marker_bits = first_marker_bits;
    handle_ret->expected_occurrences = expected_occurrences;
    handle_ret->kernels = kernels;
    handle_ret->num_sites = sites_count;
    handle_ret->sites = sites;
    handle_ret->num_regcount_records = regcount_count;
    handle_ret->regcount_records = regcount_records;
    handle_ret->workspace = workspace;
    handle_ret->workspace_size = workspace_size;

    return CUSR_SASS_INSPECT_SUCCESS;
}

static void
cusr_sass_inspect_print_regs(FILE* stream, const uint8_t* regs, size_t count)
{
    size_t i;

    fprintf(stream, "{");
    for (i = 0u; i < count; ++i) {
        fprintf(stream, "%sR%u", i == 0u ? "" : ",", (unsigned)regs[i]);
    }
    fprintf(stream, "}");
}

static void
cusr_sass_inspect_print_json_string(FILE* stream, const char* text)
{
    const unsigned char* cursor = (const unsigned char*)text;

    fputc('"', stream);
    while (*cursor != '\0') {
        switch (*cursor) {
            case '"': fputs("\\\"", stream); break;
            case '\\': fputs("\\\\", stream); break;
            case '\b': fputs("\\b", stream); break;
            case '\f': fputs("\\f", stream); break;
            case '\n': fputs("\\n", stream); break;
            case '\r': fputs("\\r", stream); break;
            case '\t': fputs("\\t", stream); break;
            default:
                if (*cursor < 0x20u) {
                    fprintf(stream, "\\u%04x", (unsigned)*cursor);
                } else {
                    fputc((int)*cursor, stream);
                }
                break;
        }
        cursor += 1u;
    }
    fputc('"', stream);
}

static void
cusr_sass_inspect_print_json_regs(FILE* stream, const uint8_t* regs, size_t count)
{
    size_t i;

    fputc('[', stream);
    for (i = 0u; i < count; ++i) {
        fprintf(stream, "%s%u", i == 0u ? "" : ",", (unsigned)regs[i]);
    }
    fputc(']', stream);
}

CUSR_SASS_INSPECT_PUBLIC_DEF
void
cusr_sass_inspect_print(
    FILE* stream,
    const char* cubin_label,
    const CusrSassInspectHandle* handle)
{
    size_t kernel_index;

    if (stream == NULL || handle == NULL) {
        return;
    }

    fprintf(
        stream,
        "cubin=%s sass_arch=sm_%u kernels=%zu ast_capacity=%zu first_marker=0x%08" PRIx32 " inputs=%u\n",
        cubin_label == NULL ? "<memory>" : cubin_label,
        (unsigned)handle->sass_arch,
        handle->num_kernels,
        handle->ast_capacity,
        handle->first_marker_bits,
        (unsigned)CUSR_SASS_INSPECT_INPUTS
    );

    if (handle->expected_occurrences != 0u) {
        fprintf(stream, "expected_site_occurrences_per_kernel=%zu\n", handle->expected_occurrences);
    }

    for (kernel_index = 0u; kernel_index < handle->num_kernels; ++kernel_index) {
        const CusrSassInspectKernel* kernel = handle->kernels + kernel_index;
        size_t i;

        fprintf(
            stream,
            "kernel[%zu] %s: symbol_index=%zu section=%s code=[0x%08" PRIx64 ",0x%08" PRIx64 ") file=[0x%zx,0x%zx)\n",
            kernel->kernel_index,
            kernel->name,
            kernel->symbol_index,
            kernel->text_section_name,
            kernel->start_address,
            kernel->end_address,
            kernel->start_file_offset,
            kernel->end_file_offset
        );

        for (i = 0u; i < kernel->num_regcount_records; ++i) {
            const CusrSassInspectRegcountRecord* record = handle->regcount_records + kernel->first_regcount_record + i;

            fprintf(
                stream,
                "    regcount[%zu]: section=%s tag_file=0x%zx value_file=0x%zx value=%" PRIu32 "\n",
                i,
                record->section_name,
                record->tag_file_offset,
                record->value_file_offset,
                record->value
            );
        }

        for (i = 0u; i < kernel->num_sites; ++i) {
            const CusrSassInspectSite* site = handle->sites + kernel->first_site + i;

            fprintf(
                stream,
                "    site occurrence[%zu]: marker=0x%08" PRIx32 " code=[0x%08" PRIx64 ",0x%08" PRIx64 ") file=[0x%zx,0x%zx)\n",
                site->occurrence_index,
                site->marker,
                site->start_address,
                site->end_address,
                site->start_file_offset,
                site->end_file_offset
            );
            fprintf(stream, "      input_regs=");
            cusr_sass_inspect_print_regs(stream, site->input_regs, CUSR_SASS_INSPECT_INPUTS);
            fprintf(stream, " target_reg=R%u incoming_wait_mask=0x%03" PRIx32 "\n", (unsigned)site->target_reg, site->incoming_wait_mask);
            fprintf(stream, "      output_regs=");
            cusr_sass_inspect_print_regs(stream, site->output_regs, site->num_output_regs);
            fprintf(stream, " available_regs=");
            cusr_sass_inspect_print_regs(stream, site->available_regs, site->num_available_regs);
            fprintf(stream, "\n");
        }
    }
}

CUSR_SASS_INSPECT_PUBLIC_DEF
void
cusr_sass_inspect_print_json(FILE* stream, const CusrSassInspectHandle* handle)
{
    size_t kernel_index;

    if (stream == NULL || handle == NULL) {
        return;
    }

    fprintf(
        stream,
        "{\"cubin_size\":%zu,\"sass_arch\":%" PRIu32 ",\"num_kernels\":%zu,\"ast_capacity\":%zu,"
        "\"first_marker_bits\":%" PRIu32 ",\"expected_occurrences\":%zu,\"num_sites\":%zu,"
        "\"num_regcount_records\":%zu,\"kernels\":[",
        handle->cubin_size,
        handle->sass_arch,
        handle->num_kernels,
        handle->ast_capacity,
        handle->first_marker_bits,
        handle->expected_occurrences,
        handle->num_sites,
        handle->num_regcount_records
    );

    for (kernel_index = 0u; kernel_index < handle->num_kernels; ++kernel_index) {
        const CusrSassInspectKernel* kernel = handle->kernels + kernel_index;
        size_t i;

        fprintf(stream, "%s{\"name\":", kernel_index == 0u ? "" : ",");
        cusr_sass_inspect_print_json_string(stream, kernel->name);
        fprintf(stream, ",\"kernel_index\":%zu,\"symbol_index\":%zu,\"text_section_name\":", kernel->kernel_index, kernel->symbol_index);
        cusr_sass_inspect_print_json_string(stream, kernel->text_section_name);
        fprintf(
            stream,
            ",\"start_address\":%" PRIu64 ",\"end_address\":%" PRIu64 ",\"start_file_offset\":%zu,"
            "\"end_file_offset\":%zu,\"first_instruction\":%zu,\"num_instructions\":%zu,"
            "\"first_regcount_record\":%zu,\"num_regcount_records\":%zu,\"first_site\":%zu,\"num_sites\":%zu,"
            "\"regcount_records\":[",
            kernel->start_address,
            kernel->end_address,
            kernel->start_file_offset,
            kernel->end_file_offset,
            kernel->first_instruction,
            kernel->num_instructions,
            kernel->first_regcount_record,
            kernel->num_regcount_records,
            kernel->first_site,
            kernel->num_sites
        );

        for (i = 0u; i < kernel->num_regcount_records; ++i) {
            const CusrSassInspectRegcountRecord* record = handle->regcount_records + kernel->first_regcount_record + i;

            fprintf(stream, "%s{\"kernel_index\":%zu,\"symbol_index\":%zu,\"section_name\":", i == 0u ? "" : ",", record->kernel_index, record->symbol_index);
            cusr_sass_inspect_print_json_string(stream, record->section_name);
            fprintf(
                stream,
                ",\"tag_file_offset\":%zu,\"value_file_offset\":%zu,\"value\":%" PRIu32 "}",
                record->tag_file_offset,
                record->value_file_offset,
                record->value
            );
        }

        fputs("],\"sites\":[", stream);
        for (i = 0u; i < kernel->num_sites; ++i) {
            const CusrSassInspectSite* site = handle->sites + kernel->first_site + i;

            fprintf(
                stream,
                "%s{\"kernel_index\":%zu,\"occurrence_index\":%zu,\"marker\":%" PRIu32 ","
                "\"start_instruction\":%zu,\"end_instruction\":%zu,\"start_file_offset\":%zu,"
                "\"end_file_offset\":%zu,\"start_address\":%" PRIu64 ",\"end_address\":%" PRIu64 ",\"input_regs\":",
                i == 0u ? "" : ",",
                site->kernel_index,
                site->occurrence_index,
                site->marker,
                site->start_instruction,
                site->end_instruction,
                site->start_file_offset,
                site->end_file_offset,
                site->start_address,
                site->end_address
            );
            cusr_sass_inspect_print_json_regs(stream, site->input_regs, CUSR_SASS_INSPECT_INPUTS);
            fprintf(stream, ",\"target_reg\":%u,\"output_regs\":", (unsigned)site->target_reg);
            cusr_sass_inspect_print_json_regs(stream, site->output_regs, site->num_output_regs);
            fputs(",\"available_regs\":", stream);
            cusr_sass_inspect_print_json_regs(stream, site->available_regs, site->num_available_regs);
            fprintf(stream, ",\"incoming_wait_mask\":%" PRIu32 "}", site->incoming_wait_mask);
        }
        fputs("]}", stream);
    }

    fputs("]}\n", stream);
}

#endif /* CUSR_SASS_INSPECT_IMPLEMENTATION_ONCE */
#endif /* CUSR_SASS_INSPECT_IMPLEMENTATION */
