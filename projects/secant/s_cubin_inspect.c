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
#include "s_cubin_internal.h"

#include "s_elf_internal.h"
#include <stdio.h>
#include <string.h>

static SecantResult
_secant_cubin_shape_num_outputs(
    _SecantCubinShape shape,
    size_t asts_per_kernel,
    size_t num_targets,
    size_t* num_outputs_ret
) {
    size_t num_outputs;

    if (num_outputs_ret == NULL) {
        _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    if (shape == _SECANT_CUBIN_SHAPE_MATERIALIZE ||
        shape == _SECANT_CUBIN_SHAPE_GRAM_STATS) {
        *num_outputs_ret = asts_per_kernel;
        return SECANT_SUCCESS;
    }
    if (num_targets == 0u) {
        _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    if (shape == _SECANT_CUBIN_SHAPE_SSE ||
        shape == _SECANT_CUBIN_SHAPE_TOGGLE_SSE) {
        if (!_secant_cubin_checked_mul(asts_per_kernel, num_targets, &num_outputs)) {
            _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_OVERFLOW);
        }
        *num_outputs_ret = num_outputs;
        return SECANT_SUCCESS;
    }
    if (shape == _SECANT_CUBIN_SHAPE_AFFINE_STATS) {
        size_t num_cross_outputs;

        if (!_secant_cubin_checked_mul(asts_per_kernel, 2u, &num_outputs) ||
            !_secant_cubin_checked_mul(asts_per_kernel, num_targets, &num_cross_outputs) ||
            !_secant_cubin_checked_add(num_outputs, num_cross_outputs, &num_outputs)) {
            _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_OVERFLOW);
        }
        *num_outputs_ret = num_outputs;
        return SECANT_SUCCESS;
    }
    _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_UNSUPPORTED_SHAPE);
}

static SecantResult
_secant_cubin_validate_inspect_args(
    _SecantCubinShape shape,
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    size_t num_targets,
    size_t patch_capacity_instructions
) {
    size_t num_outputs;
    size_t marker_count;
    size_t total_markers;

    if (num_kernels == 0u || asts_per_kernel == 0u ||
        asts_per_kernel > _SECANT_CUBIN_MAX_REGISTERS ||
        (num_inputs == 0u && shape != _SECANT_CUBIN_SHAPE_TOGGLE_SSE) ||
        num_inputs > SECANT_AST_MAX_INPUTS ||
        patch_capacity_instructions == 0u) {
        _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    if (shape == _SECANT_CUBIN_SHAPE_MATERIALIZE ||
        shape == _SECANT_CUBIN_SHAPE_GRAM_STATS) {
        num_targets = 0u;
    }
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_shape_num_outputs(
        shape,
        asts_per_kernel,
        num_targets,
        &num_outputs));
    if (!_secant_cubin_checked_add(num_inputs, num_targets, &marker_count) ||
        !_secant_cubin_checked_add(marker_count, num_outputs, &marker_count) ||
        marker_count > _SECANT_CUBIN_MAX_REGISTERS ||
        !_secant_cubin_checked_mul(num_kernels, marker_count, &total_markers) ||
        total_markers - 1u >
            0x7fffffffu - _SECANT_CUBIN_FIRST_MARKER_BITS) {
        _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    return SECANT_SUCCESS;
}
typedef struct _SecantCubinElf {
    const unsigned char* data;
    size_t size;
    const Elf64_Ehdr* header;
    const Elf64_Shdr* sections;
    const char* section_names;
    size_t section_names_size;
} _SecantCubinElf;

typedef struct _SecantCubinSymbol {
    const Elf64_Sym* symbol;
    size_t symbol_index;
    size_t data_file_offset;
} _SecantCubinSymbol;

static const char*
_secant_cubin_section_name(
    const _SecantCubinElf* elf,
    const Elf64_Shdr* section
) {
    const char* name;

    if (section->sh_name >= elf->section_names_size) {
        return NULL;
    }
    name = elf->section_names + section->sh_name;
    return memchr(
        name,
        '\0',
        elf->section_names_size - section->sh_name) != NULL
        ? name
        : NULL;
}

static SecantResult
_secant_cubin_parse_elf(
    const void* cubin,
    size_t cubin_size,
    _SecantCubinElf* elf_ret
) {
    const unsigned char* data = (const unsigned char*)cubin;
    const Elf64_Ehdr* header;
    const Elf64_Shdr* section_names;
    size_t section_table_size;

    if (cubin == NULL || elf_ret == NULL || !_secant_cubin_range_ok(cubin_size, 0u, sizeof(*header))) {
        _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    if ((uintptr_t)data % sizeof(uint64_t) != 0u) {
        _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    memset(elf_ret, 0, sizeof(*elf_ret));
    header = (const Elf64_Ehdr*)data;
    if (memcmp(header->e_ident, ELFMAG, SELFMAG) != 0 ||
        header->e_ident[EI_CLASS] != ELFCLASS64 ||
        header->e_ident[EI_DATA] != ELFDATA2LSB ||
        header->e_machine != EM_CUDA ||
        header->e_shentsize != sizeof(Elf64_Shdr) ||
        header->e_shnum == 0u ||
        header->e_shstrndx >= header->e_shnum ||
        header->e_shoff % sizeof(uint64_t) != 0u ||
        !_secant_cubin_checked_mul(
            (size_t)header->e_shnum,
            sizeof(Elf64_Shdr),
            &section_table_size) ||
        !_secant_cubin_range_ok(
            cubin_size,
            (size_t)header->e_shoff,
            section_table_size)) {
        _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_PARSE_FAILED);
    }
    elf_ret->data = data;
    elf_ret->size = cubin_size;
    elf_ret->header = header;
    elf_ret->sections = (const Elf64_Shdr*)(data + header->e_shoff);
    section_names = elf_ret->sections + header->e_shstrndx;
    if (!_secant_cubin_range_ok(
            cubin_size,
            (size_t)section_names->sh_offset,
            (size_t)section_names->sh_size)) {
        memset(elf_ret, 0, sizeof(*elf_ret));
        _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_PARSE_FAILED);
    }
    elf_ret->section_names = (const char*)data + (size_t)section_names->sh_offset;
    elf_ret->section_names_size = (size_t)section_names->sh_size;
    return SECANT_SUCCESS;
}

static SecantResult
_secant_cubin_find_function(
    const _SecantCubinElf* elf,
    const char* function_name,
    _SecantCubinSymbol* function_ret
) {
    size_t pass;

    if (elf == NULL || function_name == NULL || function_name[0] == '\0' || function_ret == NULL) {
        _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    memset(function_ret, 0, sizeof(*function_ret));
    for (pass = 0u; pass < 2u; ++pass) {
        const uint32_t requested_type = pass == 0u ? SHT_SYMTAB : SHT_DYNSYM;
        size_t section_idx;

        for (section_idx = 0u;
             section_idx < elf->header->e_shnum;
             ++section_idx) {
            const Elf64_Shdr* symbol_table = elf->sections + section_idx;
            const Elf64_Shdr* strings;
            const Elf64_Sym* symbols;
            const char* string_data;
            size_t num_symbols;
            size_t symbol_idx;

            if (symbol_table->sh_type != requested_type) {
                continue;
            }
            if (symbol_table->sh_entsize != sizeof(Elf64_Sym) ||
                symbol_table->sh_size % symbol_table->sh_entsize != 0u ||
                symbol_table->sh_offset % sizeof(uint64_t) != 0u ||
                symbol_table->sh_link >= elf->header->e_shnum ||
                !_secant_cubin_range_ok(
                    elf->size,
                    (size_t)symbol_table->sh_offset,
                    (size_t)symbol_table->sh_size)) {
                _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_PARSE_FAILED);
            }
            strings = elf->sections + symbol_table->sh_link;
            if (!_secant_cubin_range_ok(
                    elf->size,
                    (size_t)strings->sh_offset,
                    (size_t)strings->sh_size)) {
                _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_PARSE_FAILED);
            }
            symbols = (const Elf64_Sym*)(
                elf->data + (size_t)symbol_table->sh_offset);
            string_data = (const char*)elf->data + (size_t)strings->sh_offset;
            num_symbols = (size_t)(symbol_table->sh_size / symbol_table->sh_entsize);
            for (symbol_idx = 0u; symbol_idx < num_symbols; ++symbol_idx) {
                const Elf64_Sym* symbol = symbols + symbol_idx;
                const char* name;
                const Elf64_Shdr* section;
                const char* section_name;
                uint64_t section_offset;
                size_t file_offset;

                if (symbol->st_name >= strings->sh_size) {
                    _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_PARSE_FAILED);
                }
                name = string_data + symbol->st_name;
                if (memchr(
                        name,
                        '\0',
                        (size_t)strings->sh_size - symbol->st_name) == NULL) {
                    _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_PARSE_FAILED);
                }
                if (strcmp(name, function_name) != 0 || ELF64_ST_TYPE(symbol->st_info) != STT_FUNC) {
                    continue;
                }
                if (symbol->st_shndx == SHN_UNDEF ||
                    symbol->st_shndx == SHN_ABS ||
                    symbol->st_shndx >= elf->header->e_shnum) {
                    _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_PARSE_FAILED);
                }
                section = elf->sections + symbol->st_shndx;
                section_name = _secant_cubin_section_name(elf, section);
                if (section_name == NULL ||
                    symbol->st_value < section->sh_addr ||
                    symbol->st_size == 0u) {
                    _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_PARSE_FAILED);
                }
                section_offset = symbol->st_value - section->sh_addr;
                if (section_offset > section->sh_size ||
                    symbol->st_size > section->sh_size - section_offset ||
                    section_offset > SIZE_MAX ||
                    section->sh_offset > SIZE_MAX ||
                    (size_t)section_offset >
                        SIZE_MAX - (size_t)section->sh_offset ||
                    symbol->st_size > SIZE_MAX) {
                    _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_PARSE_FAILED);
                }
                file_offset = (size_t)section->sh_offset + (size_t)section_offset;
                if (!_secant_cubin_range_ok(
                        elf->size,
                        file_offset,
                        (size_t)symbol->st_size) ||
                    file_offset % _SECANT_CUBIN_INSTRUCTION_BYTES != 0u ||
                    symbol->st_size % _SECANT_CUBIN_INSTRUCTION_BYTES != 0u) {
                    _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_PARSE_FAILED);
                }
                function_ret->symbol = symbol;
                function_ret->symbol_index = symbol_idx;
                function_ret->data_file_offset = file_offset;
                return SECANT_SUCCESS;
            }
        }
    }
    _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_SKELETON_NOT_FOUND);
}

static uint32_t
_secant_cubin_arch_from_flags(uint32_t flags) {
    uint32_t arch = flags & 0xffu;

    if (arch >= 30u) {
        return arch;
    }
    arch = (flags >> 8) & 0xffu;
    return arch >= 30u ? arch : 0u;
}

static uint32_t
_secant_cubin_opcode(uint64_t word0) {
    return (uint32_t)(word0 & 0xffffu);
}

static uint8_t
_secant_cubin_dst_reg(uint64_t word0) {
    return (uint8_t)((word0 >> 16) & 0xffu);
}

static uint8_t
_secant_cubin_src_reg(uint64_t word0) {
    return (uint8_t)((word0 >> 24) & 0xffu);
}

static uint32_t
_secant_cubin_immediate(uint64_t word0) {
    return (uint32_t)(word0 >> 32);
}

static uint32_t
_secant_cubin_wait_mask(uint64_t word1) {
    return (uint32_t)((word1 >> 52) & 0xfffu);
}

static int
_secant_cubin_is_bpt(uint64_t word0) {
    return word0 == _SECANT_CUBIN_BPT_WORD0;
}

static int
_secant_cubin_reg_in(const uint8_t* regs, size_t count, uint8_t reg) {
    size_t idx;

    for (idx = 0u; idx < count; ++idx) {
        if (regs[idx] == reg) {
            return 1;
        }
    }
    return 0;
}

static SecantResult
_secant_cubin_add_reg(uint8_t* regs, size_t* count, uint8_t reg) {
    if (_secant_cubin_reg_in(regs, *count, reg)) {
        return SECANT_SUCCESS;
    }
    if (*count >= _SECANT_CUBIN_MAX_REGISTERS) {
        _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_REGISTER_OVERFLOW);
    }
    regs[(*count)++] = reg;
    return SECANT_SUCCESS;
}

static void
_secant_cubin_remove_reserved_regs(_SecantCubinSite* site) {
    size_t read_idx;
    size_t write_idx = 0u;

    for (read_idx = 0u;
         read_idx < site->num_available_regs;
         ++read_idx) {
        const uint8_t reg = site->available_regs[read_idx];

        if (!_secant_cubin_reg_in(
                site->input_regs,
                site->num_input_regs,
                reg) &&
            !_secant_cubin_reg_in(
                site->target_regs,
                site->num_target_regs,
                reg) &&
            !_secant_cubin_reg_in(
                site->output_regs,
                site->num_output_regs,
                reg)) {
            site->available_regs[write_idx++] = reg;
        }
    }
    site->num_available_regs = write_idx;
}

static SecantResult
_secant_cubin_collect_register_counts(
    const _SecantCubinElf* elf,
    size_t symbol_index,
    size_t* record_file_offsets,
    size_t records_capacity,
    size_t* record_count,
    size_t* kernel_record_count_ret,
    uint32_t* register_count_ret
) {
    size_t kernel_record_count = 0u;
    uint32_t register_count = 0u;
    size_t section_idx;

    for (section_idx = 0u;
         section_idx < elf->header->e_shnum;
         ++section_idx) {
        const Elf64_Shdr* section = elf->sections + section_idx;
        const char* section_name = _secant_cubin_section_name(elf, section);
        size_t section_offset;

        if (section_name == NULL) {
            _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_PARSE_FAILED);
        }
        if (strstr(section_name, ".nv.info") == NULL) {
            continue;
        }
        if (!_secant_cubin_range_ok(
                elf->size,
                (size_t)section->sh_offset,
                (size_t)section->sh_size)) {
            _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_PARSE_FAILED);
        }
        for (section_offset = 0u;
             section_offset + 12u <= section->sh_size;
             section_offset += 4u) {
            const size_t file_offset = (size_t)section->sh_offset + section_offset;
            const unsigned char* bytes = elf->data + file_offset;
            const uint32_t value = _secant_cubin_read_u32(bytes + 8u);

            if (bytes[0] != _SECANT_CUBIN_NVINFO_FORMAT_U32 ||
                bytes[1] != _SECANT_CUBIN_NVINFO_ATTR_REGCOUNT ||
                _secant_cubin_read_u16(bytes + 2u) !=
                    _SECANT_CUBIN_NVINFO_ATTR_SIZE ||
                _secant_cubin_read_u32(bytes + 4u) != symbol_index) {
                continue;
            }
            if (kernel_record_count != 0u && value != register_count) {
                _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_PARSE_FAILED);
            }
            if (record_file_offsets != NULL) {
                if (*record_count >= records_capacity) {
                    _SECANT_CUBIN_ERROR_RET(
                        SECANT_ERROR_INSUFFICIENT_BUFFER);
                }
                record_file_offsets[*record_count] = file_offset + 8u;
            }
            register_count = value;
            *record_count += 1u;
            kernel_record_count += 1u;
        }
    }
    if (kernel_record_count == 0u) {
        _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_PARSE_FAILED);
    }
    *kernel_record_count_ret = kernel_record_count;
    *register_count_ret = register_count;
    return SECANT_SUCCESS;
}

static int
_secant_cubin_is_zero_umov(uint64_t word0, uint32_t arch) {
    return
        _secant_cubin_opcode(word0) == _SECANT_CUBIN_OPCODE_UMOV &&
        (arch < 90u ||
         ((word0 >> 32) & 0xffu) == (arch == 90u ? 0x3fu : 0xffu));
}

static SecantResult
_secant_cubin_scan_site(
    const _SecantCubinElf* elf,
    const _SecantCubinSymbol* function,
    uint32_t arch,
    uint32_t marker,
    _SecantCubinShape shape,
    size_t asts_per_kernel,
    size_t num_inputs,
    size_t num_targets,
    size_t patch_capacity_instructions,
    _SecantCubinSite* site_ret
) {
    const size_t function_start = function->data_file_offset;
    const size_t function_end = function_start + (size_t)function->symbol->st_size;
    _SecantCubinSite site;
    uint8_t marker_seen[_SECANT_CUBIN_MAX_REGISTERS] = { 0u };
    size_t num_outputs;
    size_t num_sources;
    size_t marker_count;
    size_t expected_reductions;
    size_t expected_bpts;
    size_t first_bpt = SIZE_MAX;
    size_t last_bpt = SIZE_MAX;
    size_t end_file_offset;
    size_t bpt_count = 0u;
    size_t reduction_count = 0u;
    size_t toggle_tests = 0u, toggle_selects = 0u;
    size_t keepalive_store_count = 0u;
    uint8_t output_source = UINT8_MAX;
    size_t offset;
    size_t marker_idx;

    if (shape == _SECANT_CUBIN_SHAPE_MATERIALIZE ||
        shape == _SECANT_CUBIN_SHAPE_GRAM_STATS) {
        num_targets = 0u;
    }
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_shape_num_outputs(
        shape,
        asts_per_kernel,
        num_targets,
        &num_outputs));
    if (!_secant_cubin_checked_add(num_inputs, num_targets, &num_sources) ||
        !_secant_cubin_checked_add(num_sources, num_outputs, &marker_count) ||
        marker_count > _SECANT_CUBIN_MAX_REGISTERS ||
        !_secant_cubin_checked_add(
            patch_capacity_instructions,
            3u,
            &expected_bpts)) {
        _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_OVERFLOW);
    }
    memset(&site, 0, sizeof(site));
    site.num_input_regs = num_inputs;
    site.num_target_regs = num_targets;
    site.num_output_regs = num_outputs;
    expected_reductions = (shape == _SECANT_CUBIN_SHAPE_MATERIALIZE ||
                           shape == _SECANT_CUBIN_SHAPE_GRAM_STATS)
        ? (num_inputs - 1u) + (asts_per_kernel + num_inputs - 1u)
        : 2u * num_sources - 1u;
    for (offset = function_start;
         offset < function_end;
         offset += _SECANT_CUBIN_INSTRUCTION_BYTES) {
        const uint64_t word0 = _secant_cubin_read_u64(elf->data + offset);

        if (_secant_cubin_is_bpt(word0)) {
            if (first_bpt == SIZE_MAX) {
                first_bpt = offset;
            }
            last_bpt = offset;
            bpt_count += 1u;
        }
    }
    if (first_bpt == SIZE_MAX || last_bpt == SIZE_MAX) {
        _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_SKELETON_NOT_FOUND);
    }
    if (bpt_count != expected_bpts || first_bpt + _SECANT_CUBIN_INSTRUCTION_BYTES > last_bpt) {
        _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_PARSE_FAILED);
    }
    for (offset = first_bpt;
         offset <= last_bpt;
         offset += _SECANT_CUBIN_INSTRUCTION_BYTES) {
        const uint64_t word0 = _secant_cubin_read_u64(elf->data + offset);
        const uint64_t word1 = _secant_cubin_read_u64(elf->data + offset + 8u);
        const uint32_t opcode = _secant_cubin_opcode(word0);
        const uint32_t immediate = _secant_cubin_immediate(word0);
        const int is_marker =
            opcode == _SECANT_CUBIN_OPCODE_FADD_IMM &&
            immediate >= marker &&
            immediate - marker < marker_count;

        if (is_marker) {
            marker_idx = immediate - marker;
            if (marker_seen[marker_idx]) {
                _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_PARSE_FAILED);
            }
            marker_seen[marker_idx] = 1u;
            if (marker_idx < num_sources) {
                const uint8_t source = _secant_cubin_src_reg(word0);

                if (marker_idx < num_inputs) {
                    site.input_regs[marker_idx] = source;
                } else {
                    site.target_regs[marker_idx - num_inputs] = source;
                }
                site.incoming_wait_mask |= _secant_cubin_wait_mask(word1);
                _SECANT_CUBIN_CHECK_RET(_secant_cubin_add_reg(
                    site.available_regs,
                    &site.num_available_regs,
                    _secant_cubin_dst_reg(word0)));
            } else {
                const size_t output_idx = marker_idx - num_sources;
                const uint8_t source = _secant_cubin_src_reg(word0);
                const uint8_t destination = _secant_cubin_dst_reg(word0);

                if (shape == _SECANT_CUBIN_SHAPE_MATERIALIZE ||
                    shape == _SECANT_CUBIN_SHAPE_GRAM_STATS) {
                    if (output_idx != 0u && source != output_source) {
                        _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_PARSE_FAILED);
                    }
                    output_source = source;
                } else if (source != destination) {
                    _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_PARSE_FAILED);
                }
                site.output_regs[output_idx] = destination;
            }
        } else if (shape == _SECANT_CUBIN_SHAPE_TOGGLE_SSE && opcode == 0x7812u) {
            uint64_t rest = word1 & ((UINT64_C(1) << 40) - 1);
            if ((word0 & 0xffffffu) != 0xff7812u || (word0 >> 32) != 1u ||
                (rest & ~(UINT64_C(7) << 17)) != UINT64_C(0x780c0ff) || toggle_tests++)
                return SECANT_ERROR_UNEXPECTED_INSTRUCTION;
            site.permutation_reg = (uint8_t)(word0 >> 24);
            site.predicate_reg = (uint8_t)((word1 >> 17) & 7);
            site.toggle_test.word0 = word0; site.toggle_test.word1 = word1;
            site.incoming_wait_mask |= _secant_cubin_wait_mask(word1);
        } else if (shape == _SECANT_CUBIN_SHAPE_TOGGLE_SSE &&
                   (opcode == _SECANT_CUBIN_OPCODE_FSEL || ((opcode & 0xfffu) == 0x221u && (opcode >> 12) < 7))) {
            uint64_t rest = word1 & ((UINT64_C(1) << 40) - 1);
            if (!toggle_tests || toggle_selects++) return SECANT_ERROR_UNEXPECTED_INSTRUCTION;
            if (opcode == _SECANT_CUBIN_OPCODE_FSEL) {
                if (((rest >> 23) & 7) != site.predicate_reg || (rest & ~(UINT64_C(7) << 23)))
                    return SECANT_ERROR_UNEXPECTED_INSTRUCTION;
            } else if ((opcode >> 12) != site.predicate_reg || rest != 0x10000u)
                return SECANT_ERROR_UNEXPECTED_INSTRUCTION;
            _SECANT_CUBIN_CHECK_RET(_secant_cubin_add_reg(site.available_regs, &site.num_available_regs, _secant_cubin_dst_reg(word0)));
        } else if (opcode == _SECANT_CUBIN_OPCODE_FADD_REG) {
            reduction_count += 1u;
            _SECANT_CUBIN_CHECK_RET(_secant_cubin_add_reg(
                site.available_regs,
                &site.num_available_regs,
                _secant_cubin_dst_reg(word0)));
        } else if (opcode == _SECANT_CUBIN_OPCODE_STS_RZ ||
                   opcode == _SECANT_CUBIN_OPCODE_STS_UR) {
            keepalive_store_count += 1u;
        } else if (!_secant_cubin_is_bpt(word0) &&
                   !_secant_cubin_is_zero_umov(word0, arch)) {
            _SECANT_CUBIN_ERROR_RET(
                SECANT_ERROR_UNEXPECTED_INSTRUCTION);
        }
    }
    if (reduction_count != expected_reductions || keepalive_store_count != 1u) {
        _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_PARSE_FAILED);
    }
    if (shape == _SECANT_CUBIN_SHAPE_TOGGLE_SSE &&
        (toggle_tests != 1 || toggle_selects != 1 || site.predicate_reg >= 7 || site.permutation_reg == 255))
        return SECANT_ERROR_PARSE_FAILED;
    for (marker_idx = 0u; marker_idx < marker_count; ++marker_idx) {
        if (!marker_seen[marker_idx]) {
            _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_PARSE_FAILED);
        }
    }
    for (marker_idx = 0u; marker_idx < num_inputs; ++marker_idx) {
        if (_secant_cubin_reg_in(
                site.input_regs,
                marker_idx,
                site.input_regs[marker_idx])) {
            _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_PARSE_FAILED);
        }
    }
    for (marker_idx = 0u; marker_idx < num_targets; ++marker_idx) {
        if (_secant_cubin_reg_in(
                site.target_regs,
                marker_idx,
                site.target_regs[marker_idx]) ||
            _secant_cubin_reg_in(
                site.input_regs,
                num_inputs,
                site.target_regs[marker_idx])) {
            _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_PARSE_FAILED);
        }
    }
    for (marker_idx = 0u; marker_idx < num_outputs; ++marker_idx) {
        if (_secant_cubin_reg_in(
                site.output_regs,
                marker_idx,
                site.output_regs[marker_idx]) ||
            _secant_cubin_reg_in(
                site.input_regs,
                num_inputs,
                site.output_regs[marker_idx]) ||
            _secant_cubin_reg_in(
                site.target_regs,
                num_targets,
                site.output_regs[marker_idx])) {
            _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_PARSE_FAILED);
        }
    }
    if (shape == _SECANT_CUBIN_SHAPE_TOGGLE_SSE &&
        (_secant_cubin_reg_in(site.input_regs, num_inputs, site.permutation_reg) ||
         _secant_cubin_reg_in(site.target_regs, num_targets, site.permutation_reg) ||
         _secant_cubin_reg_in(site.output_regs, num_outputs, site.permutation_reg)))
        return SECANT_ERROR_PARSE_FAILED;
    site.load_fence_file_offset = first_bpt;
    site.start_file_offset = first_bpt + _SECANT_CUBIN_INSTRUCTION_BYTES;
    end_file_offset = last_bpt + _SECANT_CUBIN_INSTRUCTION_BYTES;
    if (end_file_offset >= function_end || end_file_offset <= site.start_file_offset) {
        _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_PARSE_FAILED);
    }
    site.num_instructions = (end_file_offset - site.start_file_offset) / _SECANT_CUBIN_INSTRUCTION_BYTES;
    _secant_cubin_remove_reserved_regs(&site);
    *site_ret = site;
    return SECANT_SUCCESS;
}

static SecantResult
_secant_cubin_format_kernel_name(
    _SecantCubinShape shape,
    size_t kernel_idx,
    char* name,
    size_t name_size
) {
    const char* pattern;
    int bytes;

    if (shape == _SECANT_CUBIN_SHAPE_MATERIALIZE) {
        pattern = "secant_cubin_materialize_%03zu";
    } else if (shape == _SECANT_CUBIN_SHAPE_SSE) {
        pattern = "secant_cubin_sse_%03zu";
    } else if (shape == _SECANT_CUBIN_SHAPE_TOGGLE_SSE) {
        pattern = "secant_cubin_toggle_sse_%03zu";
} else if (shape == _SECANT_CUBIN_SHAPE_AFFINE_STATS) {
        pattern = "secant_cubin_affine_stats_%03zu";
    } else if (shape == _SECANT_CUBIN_SHAPE_GRAM_STATS) {
        pattern = "secant_cubin_gram_stats_%03zu";
} else {
        _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    bytes = snprintf(name, name_size, pattern, kernel_idx);

    if (bytes < 0 || (size_t)bytes >= name_size) {
        _SECANT_CUBIN_ERROR_RET(
            bytes < 0
                ? SECANT_ERROR_FORMAT
                : SECANT_ERROR_OVERFLOW);
    }
    return SECANT_SUCCESS;
}

static SecantResult
_secant_cubin_scan_kernel(
    const _SecantCubinElf* elf,
    _SecantCubinShape shape,
    size_t kernel_idx,
    size_t asts_per_kernel,
    size_t num_inputs,
    size_t num_targets,
    size_t patch_capacity_instructions,
    size_t* record_file_offsets,
    size_t records_capacity,
    size_t* record_count,
    _SecantCubinKernel* kernel_ret
) {
    _SecantCubinSymbol function;
    _SecantCubinKernel kernel;
    char function_name[_SECANT_CUBIN_NAME_BYTES];
    size_t num_outputs;
    size_t marker_count;
    size_t marker_offset;
    size_t kernel_record_count;
    uint32_t arch = _secant_cubin_arch_from_flags(elf->header->e_flags);

    memset(&kernel, 0, sizeof(kernel));
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_format_kernel_name(
        shape,
        kernel_idx,
        function_name,
        sizeof(function_name)));
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_find_function(
        elf,
        function_name,
        &function));
    if (shape == _SECANT_CUBIN_SHAPE_MATERIALIZE ||
        shape == _SECANT_CUBIN_SHAPE_GRAM_STATS) {
        num_targets = 0u;
    }
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_shape_num_outputs(
        shape,
        asts_per_kernel,
        num_targets,
        &num_outputs));
    if (!_secant_cubin_checked_add(
            num_inputs,
            num_targets,
            &marker_count) ||
        !_secant_cubin_checked_add(marker_count, num_outputs, &marker_count) ||
        !_secant_cubin_checked_mul(
            kernel_idx,
            marker_count,
            &marker_offset) ||
        marker_offset >
            UINT32_MAX - _SECANT_CUBIN_FIRST_MARKER_BITS) {
        _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_OVERFLOW);
    }
    kernel.first_register_count = *record_count;
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_collect_register_counts(
        elf,
        function.symbol_index,
        record_file_offsets,
        records_capacity,
        record_count,
        &kernel_record_count,
        &kernel.register_count));
    kernel.num_register_counts = kernel_record_count;
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_scan_site(
        elf,
        &function,
        arch,
        _SECANT_CUBIN_FIRST_MARKER_BITS + (uint32_t)marker_offset,
        shape,
        asts_per_kernel,
        num_inputs,
        num_targets,
        patch_capacity_instructions,
        &kernel.site));
    if (kernel_ret != NULL) {
        *kernel_ret = kernel;
    }
    return SECANT_SUCCESS;
}

static SecantResult
_secant_cubin_scan_template(
    SecantCubinPlan* plan,
    const void* cubin,
    size_t cubin_size,
    _SecantCubinKernel* kernels,
    size_t* register_count_file_offsets,
    size_t register_counts_capacity,
    size_t* num_register_counts_ret,
    size_t* sass_capacity_ret
) {
    _SecantCubinElf elf;
    size_t record_count = 0u;
    size_t sass_capacity = 0u;
    size_t kernel_idx;
    uint32_t arch;
    uint32_t minor;

    if (plan == NULL || cubin == NULL || cubin_size == 0u || num_register_counts_ret == NULL || sass_capacity_ret == NULL) {
        _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_parse_elf(
        cubin,
        cubin_size,
        &elf));
    arch = _secant_cubin_arch_from_flags(elf.header->e_flags);
    plan->compute_capability_major = arch / 10u;
    minor = arch % 10u;
    if (!_secant_cubin_arch_supported(plan->compute_capability_major, minor)) {
        _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_UNSUPPORTED_ARCHITECTURE);
    }
    for (kernel_idx = 0u; kernel_idx < plan->num_kernels; ++kernel_idx) {
        _SecantCubinKernel kernel;

        _SECANT_CUBIN_CHECK_RET(_secant_cubin_scan_kernel(
            &elf,
            plan->shape,
            kernel_idx,
            plan->asts_per_kernel,
            plan->num_input_registers,
            plan->num_targets,
            plan->patch_capacity_instructions,
            register_count_file_offsets,
            register_counts_capacity,
            &record_count,
            &kernel));
        if (kernels != NULL) {
            kernels[kernel_idx] = kernel;
        }
        if (kernel.site.num_instructions > sass_capacity) {
            sass_capacity = kernel.site.num_instructions;
        }
    }
    if (record_count == 0u || sass_capacity == 0u) {
        _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_PARSE_FAILED);
    }
    *num_register_counts_ret = record_count;
    *sass_capacity_ret = sass_capacity;
    return SECANT_SUCCESS;
}

static int
_secant_cubin_checked_align(
    size_t value,
    size_t alignment,
    size_t* result_ret
) {
    const size_t mask = alignment - 1u;

    if (alignment == 0u || (alignment & mask) != 0u || value > SIZE_MAX - mask) {
        return 0;
    }
    *result_ret = (value + mask) & ~mask;
    return 1;
}

static int
_secant_cubin_workspace_layout(
    size_t num_kernels,
    size_t num_register_counts,
    size_t* kernels_offset_ret,
    size_t* register_counts_offset_ret,
    size_t* payload_size_ret
) {
    size_t kernels_bytes;
    size_t register_counts_bytes;
    size_t offset = sizeof(struct SecantCubinPlan);

    if (!_secant_cubin_checked_mul(
            num_kernels,
            sizeof(_SecantCubinKernel),
            &kernels_bytes) ||
        !_secant_cubin_checked_mul(
            num_register_counts,
            sizeof(size_t),
            &register_counts_bytes) ||
        !_secant_cubin_checked_align(
            offset,
            _SECANT_CUBIN_WORKSPACE_ALIGNMENT,
            &offset)) {
        return 0;
    }
    *kernels_offset_ret = offset;
    if (!_secant_cubin_checked_add(offset, kernels_bytes, &offset) ||
        !_secant_cubin_checked_align(
            offset,
            _SECANT_CUBIN_WORKSPACE_ALIGNMENT,
            &offset)) {
        return 0;
    }
    *register_counts_offset_ret = offset;
    if (!_secant_cubin_checked_add(
            offset,
            register_counts_bytes,
            &offset) ||
        !_secant_cubin_checked_align(
            offset,
            _SECANT_CUBIN_WORKSPACE_ALIGNMENT,
            &offset)) {
        return 0;
    }
    *payload_size_ret = offset;
    return 1;
}

static SecantResult
_secant_cubin_inspect(
    _SecantCubinShape shape,
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    size_t num_input_constants,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    size_t patch_capacity_instructions,
    const void* cubin,
    size_t cubin_size,
    void* workspace,
    size_t workspace_size,
    size_t* workspace_size_ret,
    SecantCubinPlan** plan_ret
) {
    SecantCubinPlan measured_plan;
    SecantCubinPlan* plan;
    size_t expected_num_register_counts;
    size_t num_register_counts;
    size_t sass_capacity;
    size_t kernels_offset;
    size_t register_counts_offset;
    size_t payload_size;
    size_t required_size;
    uintptr_t workspace_address;
    uintptr_t aligned_address;
    unsigned char* aligned_workspace;

    if (cubin == NULL || cubin_size == 0u ||
        workspace_size_ret == NULL || plan_ret == NULL ||
        (workspace == NULL && workspace_size != 0u)) {
        _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    *plan_ret = NULL;
    *workspace_size_ret = 0u;
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_validate_inspect_args(
        shape,
        num_kernels,
        asts_per_kernel,
        num_inputs,
        num_targets,
        patch_capacity_instructions));
    if (shape == _SECANT_CUBIN_SHAPE_MATERIALIZE) {
        if (num_targets != 0u || tile_rows != 0u ||
            threads_per_block != 0u) {
            _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_UNSUPPORTED_SHAPE);
        }
    } else if ((shape == _SECANT_CUBIN_SHAPE_SSE ||
                shape == _SECANT_CUBIN_SHAPE_AFFINE_STATS ||
                shape == _SECANT_CUBIN_SHAPE_GRAM_STATS) &&
        (num_targets == 0u ||
         tile_rows == 0u ||
         threads_per_block < 32u ||
         threads_per_block > tile_rows ||
         threads_per_block > _SECANT_CUBIN_SSE_MAX_WARPS * 32u ||
         threads_per_block % 32u != 0u)) {
        _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_UNSUPPORTED_SHAPE);
    } else if (shape == _SECANT_CUBIN_SHAPE_TOGGLE_SSE) {
        if (num_input_constants > num_inputs || !num_targets || !tile_rows ||
            !threads_per_block || threads_per_block > 1024 || threads_per_block % 32)
            return SECANT_ERROR_UNSUPPORTED_SHAPE;
    }
    if (shape != _SECANT_CUBIN_SHAPE_TOGGLE_SSE && num_input_constants) return SECANT_ERROR_UNSUPPORTED_SHAPE;
    memset(&measured_plan, 0, sizeof(measured_plan));
    measured_plan.shape = shape;
    measured_plan.cubin_size = cubin_size;
    measured_plan.num_kernels = num_kernels;
    measured_plan.asts_per_kernel = asts_per_kernel;
    measured_plan.num_input_registers = num_inputs;
    measured_plan.num_constant_registers = num_input_constants;
    measured_plan.num_targets = num_targets;
    measured_plan.tile_rows = tile_rows;
    measured_plan.threads_per_block = threads_per_block;
    measured_plan.patch_capacity_instructions = patch_capacity_instructions;
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_scan_template(
        &measured_plan,
        cubin,
        cubin_size,
        NULL,
        NULL,
        0u,
        &expected_num_register_counts,
        &sass_capacity));
    if (!_secant_cubin_workspace_layout(
            num_kernels,
            expected_num_register_counts,
            &kernels_offset,
            &register_counts_offset,
            &payload_size) ||
        !_secant_cubin_checked_add(
            payload_size,
            _SECANT_CUBIN_WORKSPACE_ALIGNMENT - 1u,
            &required_size)) {
        _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_OVERFLOW);
    }
    *workspace_size_ret = required_size;
    if (workspace == NULL) {
        return SECANT_SUCCESS;
    }
    if (workspace_size < required_size) {
        _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_INSUFFICIENT_BUFFER);
    }
    workspace_address = (uintptr_t)workspace;
    if (workspace_address > UINTPTR_MAX - (_SECANT_CUBIN_WORKSPACE_ALIGNMENT - 1u)) {
        _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_OVERFLOW);
    }
    aligned_address =
        (workspace_address + _SECANT_CUBIN_WORKSPACE_ALIGNMENT - 1u) &
        ~(uintptr_t)(_SECANT_CUBIN_WORKSPACE_ALIGNMENT - 1u);
    aligned_workspace = (unsigned char*)aligned_address;
    memset(aligned_workspace, 0, payload_size);
    plan = (SecantCubinPlan*)aligned_workspace;
    *plan = measured_plan;
    plan->kernels = (_SecantCubinKernel*)(aligned_workspace + kernels_offset);
    plan->register_count_file_offsets = (size_t*)(aligned_workspace + register_counts_offset);
    _SECANT_CUBIN_CHECK_RET(_secant_cubin_scan_template(
        plan,
        cubin,
        cubin_size,
        plan->kernels,
        plan->register_count_file_offsets,
        expected_num_register_counts,
        &num_register_counts,
        &sass_capacity));
    if (num_register_counts != expected_num_register_counts) {
        _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_INVALID_STATE);
    }
    *plan_ret = plan;
    return SECANT_SUCCESS;
}

static SecantResult
_secant_cubin_materialize_inspect(
    const SecantCubinMaterializeRecipe* recipe,
    const void* cubin,
    size_t cubin_size,
    void* plan_storage,
    size_t plan_storage_size,
    size_t* required_plan_storage_ret,
    SecantCubinPlan** plan_ret
) {
    if (recipe == NULL) {
        _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    _SECANT_CUBIN_ERROR_RET(_secant_cubin_inspect(
        _SECANT_CUBIN_SHAPE_MATERIALIZE,
        recipe->num_kernels,
        recipe->asts_per_kernel,
        recipe->num_inputs,
        0u,
        0u,
        0u,
        0u,
        recipe->patch_capacity_instructions,
        cubin,
        cubin_size,
        plan_storage,
        plan_storage_size,
        required_plan_storage_ret,
        plan_ret));
}

static SecantResult
_secant_cubin_sse_inspect(
    const SecantCubinSSERecipe* recipe,
    const void* cubin,
    size_t cubin_size,
    void* plan_storage,
    size_t plan_storage_size,
    size_t* required_plan_storage_ret,
    SecantCubinPlan** plan_ret
) {
    if (recipe == NULL) {
        _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    _SECANT_CUBIN_ERROR_RET(_secant_cubin_inspect(
        _SECANT_CUBIN_SHAPE_SSE,
        recipe->num_kernels,
        recipe->asts_per_kernel,
        recipe->num_inputs,
        0u,
        recipe->num_targets,
        recipe->tile_rows,
        recipe->threads_per_block,
        recipe->patch_capacity_instructions,
        cubin,
        cubin_size,
        plan_storage,
        plan_storage_size,
        required_plan_storage_ret,
        plan_ret));
}

static SecantResult
_secant_cubin_affine_stats_inspect(
    const SecantCubinAffineStatsRecipe* recipe,
    const void* cubin,
    size_t cubin_size,
    void* plan_storage,
    size_t plan_storage_size,
    size_t* required_plan_storage_ret,
    SecantCubinPlan** plan_ret
) {
    if (recipe == NULL) {
        _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    _SECANT_CUBIN_ERROR_RET(_secant_cubin_inspect(
        _SECANT_CUBIN_SHAPE_AFFINE_STATS,
        recipe->num_kernels,
        recipe->asts_per_kernel,
        recipe->num_inputs,
        0u,
        recipe->num_targets,
        recipe->tile_rows,
        recipe->threads_per_block,
        recipe->patch_capacity_instructions,
        cubin,
        cubin_size,
        plan_storage,
        plan_storage_size,
        required_plan_storage_ret,
        plan_ret));
}

static SecantResult
_secant_cubin_gram_stats_inspect(
    const SecantCubinGramStatsRecipe* recipe,
    const void* cubin,
    size_t cubin_size,
    void* plan_storage,
    size_t plan_storage_size,
    size_t* required_plan_storage_ret,
    SecantCubinPlan** plan_ret
) {
    if (recipe == NULL) {
        _SECANT_CUBIN_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    _SECANT_CUBIN_ERROR_RET(_secant_cubin_inspect(
        _SECANT_CUBIN_SHAPE_GRAM_STATS,
        recipe->num_kernels,
        recipe->asts_per_kernel,
        recipe->num_inputs,
        0u,
        recipe->num_targets,
        recipe->tile_rows,
        recipe->threads_per_block,
        recipe->patch_capacity_instructions,
        cubin,
        cubin_size,
        plan_storage,
        plan_storage_size,
        required_plan_storage_ret,
        plan_ret));
}

static SecantResult _secant_cubin_toggle_sse_inspect(const SecantCubinToggleSSERecipe *r,
    const void *cubin, size_t cubin_size, void *storage, size_t storage_size, size_t *required, SecantCubinPlan **plan) {
    return _secant_cubin_inspect(_SECANT_CUBIN_SHAPE_TOGGLE_SSE, r->num_kernels, r->asts_per_kernel,
        r->num_inputs + r->num_constants, r->num_constants, r->num_targets, r->tile_rows,
        r->threads_per_block, r->patch_capacity_instructions, cubin, cubin_size, storage, storage_size, required, plan);
}

static SecantResult
_secant_cubin_recipe_inspect(
    const SecantCubinRecipeHeader* recipe,
    const void* cubin,
    size_t cubin_size,
    void* plan_storage,
    size_t plan_storage_size,
    size_t* required_plan_storage_ret,
    SecantCubinPlan** plan_ret
) {
    SecantResult result = _secant_cubin_recipe_validate(recipe);

    if (result != SECANT_SUCCESS) {
        return result;
    }
    switch (recipe->shape) {
        case SECANT_KERNEL_SHAPE_STATIC_MATERIALIZE_F32:
            return _secant_cubin_materialize_inspect(
                (const SecantCubinMaterializeRecipe*)(const void*)recipe,
                cubin,
                cubin_size,
                plan_storage,
                plan_storage_size,
                required_plan_storage_ret,
                plan_ret);
        case SECANT_KERNEL_SHAPE_STATIC_SSE_F32:
            return _secant_cubin_sse_inspect(
                (const SecantCubinSSERecipe*)(const void*)recipe,
                cubin,
                cubin_size,
                plan_storage,
                plan_storage_size,
                required_plan_storage_ret,
                plan_ret);
        case SECANT_KERNEL_SHAPE_STATIC_AFFINE_STATS_F32:
            return _secant_cubin_affine_stats_inspect(
                (const SecantCubinAffineStatsRecipe*)(const void*)recipe,
                cubin,
                cubin_size,
                plan_storage,
                plan_storage_size,
                required_plan_storage_ret,
                plan_ret);
        case SECANT_KERNEL_SHAPE_STATIC_GRAM_STATS_F32:
            return _secant_cubin_gram_stats_inspect(
                (const SecantCubinGramStatsRecipe*)(const void*)recipe,
                cubin,
                cubin_size,
                plan_storage,
                plan_storage_size,
                required_plan_storage_ret,
                plan_ret);
        case SECANT_KERNEL_SHAPE_TOGGLE_SSE_F32:
            return _secant_cubin_toggle_sse_inspect((const SecantCubinToggleSSERecipe*)recipe, cubin,
                cubin_size, plan_storage, plan_storage_size, required_plan_storage_ret, plan_ret);
        default:
            return SECANT_ERROR_UNSUPPORTED_SHAPE;
    }
}

SecantResult
secant_cubin_plan_storage_size(
    const SecantCubinRecipeHeader* recipe,
    const void* cubin,
    size_t cubin_size,
    size_t* required_storage_size_ret
) {
    SecantCubinPlan* plan = NULL;

    if (required_storage_size_ret == NULL) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    *required_storage_size_ret = 0u;
    return _secant_cubin_recipe_inspect(
        recipe,
        cubin,
        cubin_size,
        NULL,
        0u,
        required_storage_size_ret,
        &plan);
}

SecantResult
secant_cubin_plan_init(
    const SecantCubinRecipeHeader* recipe,
    const void* cubin,
    size_t cubin_size,
    void* plan_storage,
    size_t plan_storage_size,
    SecantCubinPlan** plan_ret
) {
    size_t required_storage_size;

    if (plan_ret == NULL) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    *plan_ret = NULL;
    if (plan_storage == NULL || plan_storage_size == 0u) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    return _secant_cubin_recipe_inspect(
        recipe,
        cubin,
        cubin_size,
        plan_storage,
        plan_storage_size,
        &required_storage_size,
        plan_ret);
}

SecantResult
secant_cubin_plan_info_get(
    const SecantCubinPlan* plan,
    SecantCubinPlanInfo* info_ret
) {
    SecantKernelShape shape;

    if (plan == NULL || info_ret == NULL) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    if ((size_t)info_ret->struct_size < sizeof(*info_ret)) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    if (info_ret->version != SECANT_CUBIN_PLAN_INFO_VERSION_1) {
        return SECANT_ERROR_UNSUPPORTED_VERSION;
    }
    if (info_ret->flags != 0u) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    switch (plan->shape) {
        case _SECANT_CUBIN_SHAPE_MATERIALIZE:
            shape = SECANT_KERNEL_SHAPE_STATIC_MATERIALIZE_F32;
            break;
        case _SECANT_CUBIN_SHAPE_SSE:
            shape = SECANT_KERNEL_SHAPE_STATIC_SSE_F32;
            break;
        case _SECANT_CUBIN_SHAPE_AFFINE_STATS:
            shape = SECANT_KERNEL_SHAPE_STATIC_AFFINE_STATS_F32;
            break;
        case _SECANT_CUBIN_SHAPE_GRAM_STATS:
            shape = SECANT_KERNEL_SHAPE_STATIC_GRAM_STATS_F32;
            break;
        case _SECANT_CUBIN_SHAPE_TOGGLE_SSE:
            shape = SECANT_KERNEL_SHAPE_TOGGLE_SSE_F32; break;
        default:
            return SECANT_ERROR_UNSUPPORTED_SHAPE;
    }
    info_ret->shape = shape;
    info_ret->cubin_size = plan->cubin_size;
    return SECANT_SUCCESS;
}

SecantResult
secant_cubin_plan_recipe_get(
    const SecantCubinPlan* plan,
    SecantCubinRecipeHeader* recipe_ret
) {
    SecantCubinPlanInfo info = secant_cubin_plan_info_init();
    SecantResult result;

    if (plan == NULL || recipe_ret == NULL) {
        return SECANT_ERROR_INVALID_VALUE;
    }
    result = _secant_cubin_recipe_header_validate(recipe_ret);
    if (result != SECANT_SUCCESS) {
        return result;
    }
    result = secant_cubin_plan_info_get(plan, &info);
    if (result != SECANT_SUCCESS) {
        return result;
    }
    if (info.shape != recipe_ret->shape) {
        return SECANT_ERROR_UNSUPPORTED_SHAPE;
    }
    switch (recipe_ret->shape) {
        case SECANT_KERNEL_SHAPE_STATIC_MATERIALIZE_F32: {
            SecantCubinMaterializeRecipe* recipe = (SecantCubinMaterializeRecipe*)(void*)recipe_ret;

            recipe->num_kernels = plan->num_kernels;
            recipe->asts_per_kernel = plan->asts_per_kernel;
            recipe->num_inputs = plan->num_input_registers;
            recipe->patch_capacity_instructions = plan->patch_capacity_instructions;
            return SECANT_SUCCESS;
        }
        case SECANT_KERNEL_SHAPE_STATIC_SSE_F32: {
            SecantCubinSSERecipe* recipe = (SecantCubinSSERecipe*)(void*)recipe_ret;

            recipe->num_kernels = plan->num_kernels;
            recipe->asts_per_kernel = plan->asts_per_kernel;
            recipe->num_inputs = plan->num_input_registers;
            recipe->num_targets = plan->num_targets;
            recipe->tile_rows = plan->tile_rows;
            recipe->threads_per_block = plan->threads_per_block;
            recipe->patch_capacity_instructions = plan->patch_capacity_instructions;
            return SECANT_SUCCESS;
        }
        case SECANT_KERNEL_SHAPE_STATIC_AFFINE_STATS_F32: {
            SecantCubinAffineStatsRecipe* recipe = (SecantCubinAffineStatsRecipe*)(void*)recipe_ret;

            recipe->num_kernels = plan->num_kernels;
            recipe->asts_per_kernel = plan->asts_per_kernel;
            recipe->num_inputs = plan->num_input_registers;
            recipe->num_targets = plan->num_targets;
            recipe->tile_rows = plan->tile_rows;
            recipe->threads_per_block = plan->threads_per_block;
            recipe->patch_capacity_instructions = plan->patch_capacity_instructions;
            return SECANT_SUCCESS;
        }
        case SECANT_KERNEL_SHAPE_STATIC_GRAM_STATS_F32: {
            SecantCubinGramStatsRecipe* recipe = (SecantCubinGramStatsRecipe*)(void*)recipe_ret;

            recipe->num_kernels = plan->num_kernels;
            recipe->asts_per_kernel = plan->asts_per_kernel;
            recipe->num_inputs = plan->num_input_registers;
            recipe->num_targets = plan->num_targets;
            recipe->tile_rows = plan->tile_rows;
            recipe->threads_per_block = plan->threads_per_block;
            recipe->patch_capacity_instructions = plan->patch_capacity_instructions;
            return SECANT_SUCCESS;
        }
        case SECANT_KERNEL_SHAPE_TOGGLE_SSE_F32: {
            SecantCubinToggleSSERecipe *r = (SecantCubinToggleSSERecipe*)recipe_ret;
            r->num_kernels = plan->num_kernels;
            r->asts_per_kernel = plan->asts_per_kernel;
            r->num_inputs = plan->num_input_registers - plan->num_constant_registers;
            r->num_constants = plan->num_constant_registers;
            r->num_targets = plan->num_targets;
            r->tile_rows = plan->tile_rows;
            r->threads_per_block = plan->threads_per_block;
            r->patch_capacity_instructions = plan->patch_capacity_instructions;
            return SECANT_SUCCESS;
        }
        default:
            return SECANT_ERROR_UNSUPPORTED_SHAPE;
    }
}

#undef _SECANT_CUBIN_CHECK_RET
#undef _SECANT_CUBIN_ERROR_RET
