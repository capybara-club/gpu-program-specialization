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
#include "o_odezza_internal.h"
#include <string.h>
#define O_TRY O_RETURN_IF_ERROR
#define O_ELF_SECTION_BYTES 64u
#define O_ELF_SYMBOL_BYTES 24u
#define O_INSTRUCTION_BYTES 16u
OdezzaResult o_size_add(size_t left, size_t right, size_t *value_ret) {
    if (value_ret == NULL)
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    if (left > SIZE_MAX - right)
        return ODEZZA_ERROR_OVERFLOW;
    *value_ret = left + right;
    return ODEZZA_SUCCESS;
}

OdezzaResult o_size_multiply(size_t left, size_t right, size_t *value_ret) {
    if (value_ret == NULL)
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    if (right != 0u && left > SIZE_MAX / right)
        return ODEZZA_ERROR_OVERFLOW;
    *value_ret = left * right;
    return ODEZZA_SUCCESS;
}

OdezzaResult o_check_range(size_t data_size, size_t offset, size_t range_size) {
    if (offset > data_size || range_size > data_size - offset)
        return ODEZZA_ERROR_FORMAT;
    return ODEZZA_SUCCESS;
}

OdezzaResult o_read_u16(const unsigned char *data, size_t size, size_t offset, uint16_t *value_ret) {
    if (data == NULL || value_ret == NULL)
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    O_TRY(o_check_range(size, offset, 2u));
    *value_ret = (uint16_t)data[offset] | ((uint16_t)data[offset + 1u] << 8u);
    return ODEZZA_SUCCESS;
}

OdezzaResult o_read_u32(const unsigned char *data, size_t size, size_t offset, uint32_t *value_ret) {
    if (data == NULL || value_ret == NULL)
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    O_TRY(o_check_range(size, offset, 4u));
    *value_ret = (uint32_t)data[offset] | ((uint32_t)data[offset + 1u] << 8u) |
                 ((uint32_t)data[offset + 2u] << 16u) | ((uint32_t)data[offset + 3u] << 24u);
    return ODEZZA_SUCCESS;
}

OdezzaResult o_read_u64(const unsigned char *data, size_t size, size_t offset, uint64_t *value_ret) {
    uint64_t value = 0u;
    size_t index;
    if (data == NULL || value_ret == NULL)
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    O_TRY(o_check_range(size, offset, 8u));
    for (index = 0u; index < 8u; ++index)
        value |= (uint64_t)data[offset + index] << (index * 8u);
    *value_ret = value;
    return ODEZZA_SUCCESS;
}

OdezzaResult o_u64_to_size(uint64_t value, size_t *value_ret) {
    if (value_ret == NULL)
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    if (value > SIZE_MAX)
        return ODEZZA_ERROR_OVERFLOW;
    *value_ret = (size_t)value;
    return ODEZZA_SUCCESS;
}

OdezzaResult o_elf_section(const OElf *elf, size_t index, OElfSection *section_ret) {
    size_t offset;
    uint64_t value;
    if (elf == NULL || section_ret == NULL || index >= elf->section_count)
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    O_TRY(o_size_multiply(index, O_ELF_SECTION_BYTES, &offset));
    O_TRY(o_size_add(elf->section_table_offset, offset, &offset));
    O_TRY(o_read_u32(elf->data, elf->size, offset, &section_ret->name_offset));
    O_TRY(o_read_u32(elf->data, elf->size, offset + 4u, &section_ret->kind));
    O_TRY(o_read_u64(elf->data, elf->size, offset + 8u, &section_ret->flags));
    O_TRY(o_read_u64(elf->data, elf->size, offset + 16u, &section_ret->address));
    O_TRY(o_read_u64(elf->data, elf->size, offset + 24u, &value));
    O_TRY(o_u64_to_size(value, &section_ret->offset));
    O_TRY(o_read_u64(elf->data, elf->size, offset + 32u, &value));
    O_TRY(o_u64_to_size(value, &section_ret->size));
    O_TRY(o_read_u32(elf->data, elf->size, offset + 40u, &section_ret->link));
    O_TRY(o_read_u32(elf->data, elf->size, offset + 44u, &section_ret->info));
    O_TRY(o_read_u64(elf->data, elf->size, offset + 48u, &section_ret->alignment));
    O_TRY(o_read_u64(elf->data, elf->size, offset + 56u, &value));
    O_TRY(o_u64_to_size(value, &section_ret->entry_size));
    return ODEZZA_SUCCESS;
}

OdezzaResult o_elf_init(const void *data, size_t size, OElf *elf_ret) {
    const unsigned char *bytes = (const unsigned char *)data;
    uint16_t machine;
    uint16_t section_entry_size;
    uint64_t section_table_offset;
    size_t section_table_bytes;
    OElfSection names;

    if (bytes == NULL || elf_ret == NULL)
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    if (size < 64u || bytes[0] != 0x7fu || bytes[1] != 'E' || bytes[2] != 'L' || bytes[3] != 'F')
        return ODEZZA_ERROR_FORMAT;
    if (bytes[4] != 2u || bytes[5] != 1u)
        return ODEZZA_ERROR_UNSUPPORTED;
    O_TRY(o_read_u16(bytes, size, 18u, &machine));
    if (machine != 190u)
        return ODEZZA_ERROR_FORMAT;
    memset(elf_ret, 0, sizeof(*elf_ret));
    elf_ret->data = bytes;
    elf_ret->size = size;
    O_TRY(o_read_u32(bytes, size, 48u, &elf_ret->flags));
    O_TRY(o_read_u64(bytes, size, 40u, &section_table_offset));
    O_TRY(o_u64_to_size(section_table_offset, &elf_ret->section_table_offset));
    O_TRY(o_read_u16(bytes, size, 58u, &section_entry_size));
    O_TRY(o_read_u16(bytes, size, 60u, &elf_ret->section_count));
    O_TRY(o_read_u16(bytes, size, 62u, &elf_ret->names_section_index));
    if (section_entry_size != O_ELF_SECTION_BYTES || elf_ret->section_count == 0u ||
        elf_ret->names_section_index >= elf_ret->section_count)
        return ODEZZA_ERROR_FORMAT;
    O_TRY(o_size_multiply(elf_ret->section_count, O_ELF_SECTION_BYTES, &section_table_bytes));
    O_TRY(o_check_range(size, elf_ret->section_table_offset, section_table_bytes));
    O_TRY(o_elf_section(elf_ret, elf_ret->names_section_index, &names));
    O_TRY(o_check_range(size, names.offset, names.size));
    elf_ret->names_offset = names.offset;
    elf_ret->names_size = names.size;
    return ODEZZA_SUCCESS;
}

OdezzaResult o_elf_string(const OElf *elf, size_t table_offset, size_t table_size, uint32_t string_offset,
                          const unsigned char **string_ret, size_t *length_ret) {
    size_t index;
    if (elf == NULL || string_ret == NULL || length_ret == NULL || string_offset >= table_size)
        return ODEZZA_ERROR_FORMAT;
    O_TRY(o_check_range(elf->size, table_offset, table_size));
    for (index = string_offset; index < table_size; ++index) {
        if (elf->data[table_offset + index] == 0u) {
            *string_ret = elf->data + table_offset + string_offset;
            *length_ret = index - string_offset;
            return ODEZZA_SUCCESS;
        }
    }
    return ODEZZA_ERROR_FORMAT;
}

OdezzaResult o_section_name_matches(const OElf *elf, const OElfSection *section, const char *name,
                                    int *matches_ret) {
    const unsigned char *actual;
    size_t actual_length;
    size_t wanted_length;
    if (elf == NULL || section == NULL || name == NULL || matches_ret == NULL)
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    O_TRY(
        o_elf_string(elf, elf->names_offset, elf->names_size, section->name_offset, &actual, &actual_length));
    wanted_length = strlen(name);
    *matches_ret = actual_length == wanted_length && memcmp(actual, name, wanted_length) == 0;
    return ODEZZA_SUCCESS;
}

OdezzaResult o_section_name_contains(const OElf *elf, const OElfSection *section, const char *needle,
                                     int *contains_ret) {
    const unsigned char *actual;
    size_t actual_length;
    size_t needle_length;
    size_t index;
    if (elf == NULL || section == NULL || needle == NULL || contains_ret == NULL)
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    O_TRY(
        o_elf_string(elf, elf->names_offset, elf->names_size, section->name_offset, &actual, &actual_length));
    needle_length = strlen(needle);
    *contains_ret = 0;
    if (needle_length > actual_length)
        return ODEZZA_SUCCESS;
    for (index = 0u; index + needle_length <= actual_length; ++index) {
        if (memcmp(actual + index, needle, needle_length) == 0) {
            *contains_ret = 1;
            return ODEZZA_SUCCESS;
        }
    }
    return ODEZZA_SUCCESS;
}

OdezzaResult o_section_name_starts_with(const OElf *elf, const OElfSection *section, const char *prefix,
                                        int *matches_ret) {
    const unsigned char *actual;
    size_t actual_length;
    size_t prefix_length;
    if (elf == NULL || section == NULL || prefix == NULL || matches_ret == NULL)
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    O_TRY(
        o_elf_string(elf, elf->names_offset, elf->names_size, section->name_offset, &actual, &actual_length));
    prefix_length = strlen(prefix);
    *matches_ret = actual_length >= prefix_length && memcmp(actual, prefix, prefix_length) == 0;
    return ODEZZA_SUCCESS;
}

OdezzaResult o_symbol_name_matches(const OElf *elf, const OElfSection *strings, uint32_t name_offset,
                                   const char *name, int *matches_ret) {
    const unsigned char *actual;
    size_t actual_length;
    size_t wanted_length;
    if (elf == NULL || strings == NULL || name == NULL || matches_ret == NULL)
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    O_TRY(o_elf_string(elf, strings->offset, strings->size, name_offset, &actual, &actual_length));
    wanted_length = strlen(name);
    *matches_ret = actual_length == wanted_length && memcmp(actual, name, wanted_length) == 0;
    return ODEZZA_SUCCESS;
}

OdezzaResult o_find_function(const OElf *elf, const char *name, OFunction *function_ret) {
    static const uint32_t table_kinds[] = {2u, 11u};
    size_t kind_index;
    size_t section_index;
    if (elf == NULL || name == NULL || function_ret == NULL)
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    for (kind_index = 0u; kind_index < 2u; ++kind_index) {
        for (section_index = 0u; section_index < elf->section_count; ++section_index) {
            OElfSection table;
            OElfSection strings;
            size_t symbol_count;
            size_t symbol_index;
            O_TRY(o_elf_section(elf, section_index, &table));
            if (table.kind != table_kinds[kind_index])
                continue;
            if (table.entry_size != O_ELF_SYMBOL_BYTES || table.size % O_ELF_SYMBOL_BYTES != 0u ||
                table.link >= elf->section_count)
                return ODEZZA_ERROR_FORMAT;
            O_TRY(o_check_range(elf->size, table.offset, table.size));
            O_TRY(o_elf_section(elf, table.link, &strings));
            O_TRY(o_check_range(elf->size, strings.offset, strings.size));
            symbol_count = table.size / O_ELF_SYMBOL_BYTES;
            for (symbol_index = 0u; symbol_index < symbol_count; ++symbol_index) {
                size_t offset = table.offset + symbol_index * O_ELF_SYMBOL_BYTES;
                uint32_t name_offset;
                uint16_t section_number;
                uint64_t value;
                uint64_t symbol_size;
                int matches;
                OElfSection code;
                size_t delta;
                O_TRY(o_read_u32(elf->data, elf->size, offset, &name_offset));
                if ((elf->data[offset + 4u] & 0x0fu) != 2u)
                    continue;
                O_TRY(o_symbol_name_matches(elf, &strings, name_offset, name, &matches));
                if (!matches)
                    continue;
                O_TRY(o_read_u16(elf->data, elf->size, offset + 6u, &section_number));
                O_TRY(o_read_u64(elf->data, elf->size, offset + 8u, &value));
                O_TRY(o_read_u64(elf->data, elf->size, offset + 16u, &symbol_size));
                if (section_number == 0u || section_number >= elf->section_count || symbol_size == 0u ||
                    symbol_index > UINT32_MAX)
                    return ODEZZA_ERROR_FORMAT;
                O_TRY(o_elf_section(elf, section_number, &code));
                if (value < code.address || value - code.address > SIZE_MAX)
                    return ODEZZA_ERROR_FORMAT;
                delta = (size_t)(value - code.address);
                O_TRY(o_size_add(code.offset, delta, &function_ret->file_offset));
                O_TRY(o_u64_to_size(symbol_size, &function_ret->size));
                O_TRY(o_check_range(elf->size, function_ret->file_offset, function_ret->size));
                if (function_ret->file_offset % O_INSTRUCTION_BYTES != 0u ||
                    function_ret->size % O_INSTRUCTION_BYTES != 0u)
                    return ODEZZA_ERROR_FORMAT;
                function_ret->symbol_index = (uint32_t)symbol_index;
                return ODEZZA_SUCCESS;
            }
        }
    }
    return ODEZZA_ERROR_FORMAT;
}

OdezzaResult o_find_data_symbol(const OElf *elf, const char *name, size_t expected_size,
                                ODataSymbol *symbol_ret) {
    static const uint32_t table_kinds[] = {2u, 11u};
    int found = 0;
    size_t kind_index;
    size_t section_index;
    if (elf == NULL || name == NULL || symbol_ret == NULL)
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    for (kind_index = 0u; kind_index < 2u; ++kind_index) {
        for (section_index = 0u; section_index < elf->section_count; ++section_index) {
            OElfSection table;
            OElfSection strings;
            size_t symbol_count;
            size_t symbol_index;
            O_TRY(o_elf_section(elf, section_index, &table));
            if (table.kind != table_kinds[kind_index])
                continue;
            if (table.entry_size != O_ELF_SYMBOL_BYTES || table.size % O_ELF_SYMBOL_BYTES != 0u ||
                table.link >= elf->section_count)
                return ODEZZA_ERROR_FORMAT;
            O_TRY(o_check_range(elf->size, table.offset, table.size));
            O_TRY(o_elf_section(elf, table.link, &strings));
            O_TRY(o_check_range(elf->size, strings.offset, strings.size));
            symbol_count = table.size / O_ELF_SYMBOL_BYTES;
            for (symbol_index = 0u; symbol_index < symbol_count; ++symbol_index) {
                size_t offset = table.offset + symbol_index * O_ELF_SYMBOL_BYTES;
                uint32_t name_offset;
                uint16_t section_number;
                uint64_t value;
                uint64_t symbol_size;
                int matches;
                OElfSection data_section;
                size_t candidate_offset;
                size_t candidate_size;
                size_t delta;
                O_TRY(o_read_u32(elf->data, elf->size, offset, &name_offset));
                if ((elf->data[offset + 4u] & 0x0fu) != 1u)
                    continue;
                O_TRY(o_symbol_name_matches(elf, &strings, name_offset, name, &matches));
                if (!matches)
                    continue;
                O_TRY(o_read_u16(elf->data, elf->size, offset + 6u, &section_number));
                O_TRY(o_read_u64(elf->data, elf->size, offset + 8u, &value));
                O_TRY(o_read_u64(elf->data, elf->size, offset + 16u, &symbol_size));
                if (section_number == 0u || section_number >= elf->section_count || symbol_size == 0u)
                    return ODEZZA_ERROR_FORMAT;
                O_TRY(o_elf_section(elf, section_number, &data_section));
                if (value < data_section.address || value - data_section.address > SIZE_MAX)
                    return ODEZZA_ERROR_FORMAT;
                delta = (size_t)(value - data_section.address);
                O_TRY(o_size_add(data_section.offset, delta, &candidate_offset));
                O_TRY(o_u64_to_size(symbol_size, &candidate_size));
                O_TRY(o_check_range(elf->size, candidate_offset, candidate_size));
                if (!found) {
                    symbol_ret->file_offset = candidate_offset;
                    symbol_ret->size = candidate_size;
                    found = 1;
                } else if (symbol_ret->file_offset != candidate_offset ||
                           symbol_ret->size != candidate_size) {
                    return ODEZZA_ERROR_FORMAT;
                }
            }
        }
    }
    if (!found || symbol_ret->size != expected_size)
        return ODEZZA_ERROR_FORMAT;
    return ODEZZA_SUCCESS;
}
