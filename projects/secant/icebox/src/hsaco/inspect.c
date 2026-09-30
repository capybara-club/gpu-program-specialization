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
#include "internal.h"

typedef struct _SecantHsacoElf {
    const unsigned char* data;
    size_t size;
    const Elf64_Ehdr* header;
    const Elf64_Shdr* sections;
    const char* section_names;
    size_t section_names_size;
} _SecantHsacoElf;

typedef struct _SecantHsacoSymbol {
    const Elf64_Sym* symbol;
    const Elf64_Shdr* section;
    const char* section_name;
    size_t symbol_index;
    size_t symbol_file_offset;
    size_t data_file_offset;
} _SecantHsacoSymbol;

static const char*
_secant_hsaco_section_name(
    const _SecantHsacoElf* elf,
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
_secant_hsaco_parse_elf(
    const void* hsaco,
    size_t hsaco_size,
    _SecantHsacoElf* elf_ret
) {
    const unsigned char* data = (const unsigned char*)hsaco;
    const Elf64_Ehdr* header;
    const Elf64_Shdr* section_names;
    size_t section_table_size;

    if (hsaco == NULL || elf_ret == NULL ||
        !_secant_hsaco_range_ok(hsaco_size, 0u, sizeof(*header))) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    if ((uintptr_t)data % sizeof(uint64_t) != 0u) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    memset(elf_ret, 0, sizeof(*elf_ret));
    header = (const Elf64_Ehdr*)data;
    if (memcmp(header->e_ident, ELFMAG, SELFMAG) != 0 ||
        header->e_ident[EI_CLASS] != ELFCLASS64 ||
        header->e_ident[EI_DATA] != ELFDATA2LSB ||
        header->e_machine != EM_AMDGPU ||
        header->e_shentsize != sizeof(Elf64_Shdr) ||
        header->e_shnum == 0u ||
        header->e_shstrndx >= header->e_shnum ||
        header->e_shoff % sizeof(uint64_t) != 0u ||
        !_secant_hsaco_checked_mul(
            (size_t)header->e_shnum,
            sizeof(Elf64_Shdr),
            &section_table_size) ||
        !_secant_hsaco_range_ok(
            hsaco_size,
            (size_t)header->e_shoff,
            section_table_size)) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_PARSE_FAILED);
    }
    elf_ret->data = data;
    elf_ret->size = hsaco_size;
    elf_ret->header = header;
    elf_ret->sections =
        (const Elf64_Shdr*)(data + (size_t)header->e_shoff);
    section_names = elf_ret->sections + header->e_shstrndx;
    if (!_secant_hsaco_range_ok(
            hsaco_size,
            (size_t)section_names->sh_offset,
            (size_t)section_names->sh_size)) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_PARSE_FAILED);
    }
    elf_ret->section_names =
        (const char*)data + (size_t)section_names->sh_offset;
    elf_ret->section_names_size = (size_t)section_names->sh_size;
    return SECANT_SUCCESS;
}

static int
_secant_hsaco_msgpack_take(
    size_t size,
    size_t* offset,
    size_t count
) {
    if (offset == NULL ||
        !_secant_hsaco_range_ok(size, *offset, count)) {
        return 0;
    }
    *offset += count;
    return 1;
}

static SecantResult
_secant_hsaco_msgpack_read_map(
    const unsigned char* data,
    size_t size,
    size_t* offset,
    uint32_t* count_ret
) {
    uint8_t prefix;

    if (data == NULL || offset == NULL || count_ret == NULL ||
        !_secant_hsaco_msgpack_take(size, offset, 1u)) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_PARSE_FAILED);
    }
    prefix = data[*offset - 1u];
    if ((prefix & 0xf0u) == 0x80u) {
        *count_ret = prefix & 0x0fu;
    } else if (prefix == 0xdeu) {
        if (!_secant_hsaco_msgpack_take(size, offset, 2u)) {
            _SECANT_HSACO_ERROR_RET(
                SECANT_ERROR_PARSE_FAILED);
        }
        *count_ret = _secant_hsaco_read_be_u16(
            data + *offset - 2u);
    } else if (prefix == 0xdfu) {
        if (!_secant_hsaco_msgpack_take(size, offset, 4u)) {
            _SECANT_HSACO_ERROR_RET(
                SECANT_ERROR_PARSE_FAILED);
        }
        *count_ret = _secant_hsaco_read_be_u32(
            data + *offset - 4u);
    } else {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_PARSE_FAILED);
    }
    return SECANT_SUCCESS;
}

static SecantResult
_secant_hsaco_msgpack_read_array(
    const unsigned char* data,
    size_t size,
    size_t* offset,
    uint32_t* count_ret
) {
    uint8_t prefix;

    if (data == NULL || offset == NULL || count_ret == NULL ||
        !_secant_hsaco_msgpack_take(size, offset, 1u)) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_PARSE_FAILED);
    }
    prefix = data[*offset - 1u];
    if ((prefix & 0xf0u) == 0x90u) {
        *count_ret = prefix & 0x0fu;
    } else if (prefix == 0xdcu) {
        if (!_secant_hsaco_msgpack_take(size, offset, 2u)) {
            _SECANT_HSACO_ERROR_RET(
                SECANT_ERROR_PARSE_FAILED);
        }
        *count_ret = _secant_hsaco_read_be_u16(
            data + *offset - 2u);
    } else if (prefix == 0xddu) {
        if (!_secant_hsaco_msgpack_take(size, offset, 4u)) {
            _SECANT_HSACO_ERROR_RET(
                SECANT_ERROR_PARSE_FAILED);
        }
        *count_ret = _secant_hsaco_read_be_u32(
            data + *offset - 4u);
    } else {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_PARSE_FAILED);
    }
    return SECANT_SUCCESS;
}

static SecantResult
_secant_hsaco_msgpack_read_string(
    const unsigned char* data,
    size_t size,
    size_t* offset,
    const unsigned char** string_ret,
    size_t* string_size_ret
) {
    uint8_t prefix;
    size_t string_size;

    if (data == NULL || offset == NULL || string_ret == NULL ||
        string_size_ret == NULL ||
        !_secant_hsaco_msgpack_take(size, offset, 1u)) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_PARSE_FAILED);
    }
    prefix = data[*offset - 1u];
    if ((prefix & 0xe0u) == 0xa0u) {
        string_size = prefix & 0x1fu;
    } else if (prefix == 0xd9u) {
        if (!_secant_hsaco_msgpack_take(size, offset, 1u)) {
            _SECANT_HSACO_ERROR_RET(
                SECANT_ERROR_PARSE_FAILED);
        }
        string_size = data[*offset - 1u];
    } else if (prefix == 0xdau) {
        if (!_secant_hsaco_msgpack_take(size, offset, 2u)) {
            _SECANT_HSACO_ERROR_RET(
                SECANT_ERROR_PARSE_FAILED);
        }
        string_size = _secant_hsaco_read_be_u16(
            data + *offset - 2u);
    } else if (prefix == 0xdbu) {
        if (!_secant_hsaco_msgpack_take(size, offset, 4u)) {
            _SECANT_HSACO_ERROR_RET(
                SECANT_ERROR_PARSE_FAILED);
        }
        string_size = _secant_hsaco_read_be_u32(
            data + *offset - 4u);
    } else {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_PARSE_FAILED);
    }
    if (!_secant_hsaco_range_ok(size, *offset, string_size)) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_PARSE_FAILED);
    }
    *string_ret = data + *offset;
    *string_size_ret = string_size;
    *offset += string_size;
    return SECANT_SUCCESS;
}

static SecantResult
_secant_hsaco_msgpack_read_uint(
    const unsigned char* data,
    size_t size,
    size_t* offset,
    uint64_t* value_ret,
    size_t* value_offset_ret,
    uint8_t* encoding_ret
) {
    const size_t value_offset = offset != NULL ? *offset : 0u;
    uint8_t prefix;

    if (data == NULL || offset == NULL || value_ret == NULL ||
        value_offset_ret == NULL || encoding_ret == NULL ||
        !_secant_hsaco_msgpack_take(size, offset, 1u)) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_PARSE_FAILED);
    }
    prefix = data[value_offset];
    if (prefix <= 0x7fu) {
        *value_ret = prefix;
        *encoding_ret = _SECANT_HSACO_MSGPACK_FIXINT;
    } else if (prefix == 0xccu) {
        if (!_secant_hsaco_msgpack_take(size, offset, 1u)) {
            _SECANT_HSACO_ERROR_RET(
                SECANT_ERROR_PARSE_FAILED);
        }
        *value_ret = data[*offset - 1u];
        *encoding_ret = _SECANT_HSACO_MSGPACK_UINT8;
    } else if (prefix == 0xcdu) {
        if (!_secant_hsaco_msgpack_take(size, offset, 2u)) {
            _SECANT_HSACO_ERROR_RET(
                SECANT_ERROR_PARSE_FAILED);
        }
        *value_ret = _secant_hsaco_read_be_u16(
            data + *offset - 2u);
        *encoding_ret = _SECANT_HSACO_MSGPACK_UINT16;
    } else if (prefix == 0xceu) {
        if (!_secant_hsaco_msgpack_take(size, offset, 4u)) {
            _SECANT_HSACO_ERROR_RET(
                SECANT_ERROR_PARSE_FAILED);
        }
        *value_ret = _secant_hsaco_read_be_u32(
            data + *offset - 4u);
        *encoding_ret = _SECANT_HSACO_MSGPACK_UINT32;
    } else if (prefix == 0xcfu) {
        if (!_secant_hsaco_msgpack_take(size, offset, 8u)) {
            _SECANT_HSACO_ERROR_RET(
                SECANT_ERROR_PARSE_FAILED);
        }
        *value_ret = _secant_hsaco_read_be_u64(
            data + *offset - 8u);
        *encoding_ret = _SECANT_HSACO_MSGPACK_UINT64;
    } else {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_PARSE_FAILED);
    }
    *value_offset_ret = value_offset;
    return SECANT_SUCCESS;
}

static SecantResult
_secant_hsaco_msgpack_skip(
    const unsigned char* data,
    size_t size,
    size_t* offset,
    uint32_t depth
) {
    uint8_t prefix;
    uint32_t count = 0u;
    size_t bytes = 0u;
    uint32_t idx;

    if (data == NULL || offset == NULL ||
        depth >= _SECANT_HSACO_MSGPACK_MAX_DEPTH ||
        !_secant_hsaco_msgpack_take(size, offset, 1u)) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_PARSE_FAILED);
    }
    prefix = data[*offset - 1u];
    if (prefix <= 0x7fu || prefix >= 0xe0u ||
        prefix == 0xc0u || prefix == 0xc2u || prefix == 0xc3u) {
        return SECANT_SUCCESS;
    }
    if ((prefix & 0xe0u) == 0xa0u) {
        bytes = prefix & 0x1fu;
    } else if ((prefix & 0xf0u) == 0x90u) {
        count = prefix & 0x0fu;
        for (idx = 0u; idx < count; ++idx) {
            _SECANT_HSACO_CHECK_RET(_secant_hsaco_msgpack_skip(
                data,
                size,
                offset,
                depth + 1u));
        }
        return SECANT_SUCCESS;
    } else if ((prefix & 0xf0u) == 0x80u) {
        count = prefix & 0x0fu;
        for (idx = 0u; idx < count * 2u; ++idx) {
            _SECANT_HSACO_CHECK_RET(_secant_hsaco_msgpack_skip(
                data,
                size,
                offset,
                depth + 1u));
        }
        return SECANT_SUCCESS;
    } else {
        switch (prefix) {
            case 0xc4u:
            case 0xd9u:
                if (!_secant_hsaco_msgpack_take(
                        size,
                        offset,
                        1u)) {
                    _SECANT_HSACO_ERROR_RET(
                        SECANT_ERROR_PARSE_FAILED);
                }
                bytes = data[*offset - 1u];
                break;
            case 0xc5u:
            case 0xdau:
                if (!_secant_hsaco_msgpack_take(
                        size,
                        offset,
                        2u)) {
                    _SECANT_HSACO_ERROR_RET(
                        SECANT_ERROR_PARSE_FAILED);
                }
                bytes = _secant_hsaco_read_be_u16(
                    data + *offset - 2u);
                break;
            case 0xc6u:
            case 0xdbu:
                if (!_secant_hsaco_msgpack_take(
                        size,
                        offset,
                        4u)) {
                    _SECANT_HSACO_ERROR_RET(
                        SECANT_ERROR_PARSE_FAILED);
                }
                bytes = _secant_hsaco_read_be_u32(
                    data + *offset - 4u);
                break;
            case 0xc7u:
                if (!_secant_hsaco_msgpack_take(
                        size,
                        offset,
                        1u)) {
                    _SECANT_HSACO_ERROR_RET(
                        SECANT_ERROR_PARSE_FAILED);
                }
                bytes = (size_t)data[*offset - 1u] + 1u;
                break;
            case 0xc8u:
                if (!_secant_hsaco_msgpack_take(
                        size,
                        offset,
                        2u)) {
                    _SECANT_HSACO_ERROR_RET(
                        SECANT_ERROR_PARSE_FAILED);
                }
                bytes = (size_t)_secant_hsaco_read_be_u16(
                    data + *offset - 2u) + 1u;
                break;
            case 0xc9u:
                if (!_secant_hsaco_msgpack_take(
                        size,
                        offset,
                        4u)) {
                    _SECANT_HSACO_ERROR_RET(
                        SECANT_ERROR_PARSE_FAILED);
                }
                bytes = (size_t)_secant_hsaco_read_be_u32(
                    data + *offset - 4u) + 1u;
                break;
            case 0xcau:
            case 0xceu:
            case 0xd2u:
                bytes = 4u;
                break;
            case 0xcbu:
            case 0xcfu:
            case 0xd3u:
                bytes = 8u;
                break;
            case 0xccu:
            case 0xd0u:
                bytes = 1u;
                break;
            case 0xcdu:
            case 0xd1u:
                bytes = 2u;
                break;
            case 0xd4u:
                bytes = 2u;
                break;
            case 0xd5u:
                bytes = 3u;
                break;
            case 0xd6u:
                bytes = 5u;
                break;
            case 0xd7u:
                bytes = 9u;
                break;
            case 0xd8u:
                bytes = 17u;
                break;
            case 0xdcu:
            case 0xddu: {
                const size_t count_bytes =
                    prefix == 0xdcu ? 2u : 4u;

                if (!_secant_hsaco_msgpack_take(
                        size,
                        offset,
                        count_bytes)) {
                    _SECANT_HSACO_ERROR_RET(
                        SECANT_ERROR_PARSE_FAILED);
                }
                count = count_bytes == 2u
                    ? _secant_hsaco_read_be_u16(
                        data + *offset - count_bytes)
                    : _secant_hsaco_read_be_u32(
                        data + *offset - count_bytes);
                for (idx = 0u; idx < count; ++idx) {
                    _SECANT_HSACO_CHECK_RET(
                        _secant_hsaco_msgpack_skip(
                            data,
                            size,
                            offset,
                            depth + 1u));
                }
                return SECANT_SUCCESS;
            }
            case 0xdeu:
            case 0xdfu: {
                const size_t count_bytes =
                    prefix == 0xdeu ? 2u : 4u;

                if (!_secant_hsaco_msgpack_take(
                        size,
                        offset,
                        count_bytes)) {
                    _SECANT_HSACO_ERROR_RET(
                        SECANT_ERROR_PARSE_FAILED);
                }
                count = count_bytes == 2u
                    ? _secant_hsaco_read_be_u16(
                        data + *offset - count_bytes)
                    : _secant_hsaco_read_be_u32(
                        data + *offset - count_bytes);
                if (count > UINT32_MAX / 2u) {
                    _SECANT_HSACO_ERROR_RET(
                        SECANT_ERROR_PARSE_FAILED);
                }
                for (idx = 0u; idx < count * 2u; ++idx) {
                    _SECANT_HSACO_CHECK_RET(
                        _secant_hsaco_msgpack_skip(
                            data,
                            size,
                            offset,
                            depth + 1u));
                }
                return SECANT_SUCCESS;
            }
            default:
                _SECANT_HSACO_ERROR_RET(
                    SECANT_ERROR_PARSE_FAILED);
        }
    }
    if (!_secant_hsaco_msgpack_take(size, offset, bytes)) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_PARSE_FAILED);
    }
    return SECANT_SUCCESS;
}

static int
_secant_hsaco_msgpack_key_is(
    const unsigned char* key,
    size_t key_size,
    const char* expected
) {
    const size_t expected_size = strlen(expected);

    return key_size == expected_size &&
        memcmp(key, expected, expected_size) == 0;
}

static SecantResult
_secant_hsaco_msgpack_parse_kernel(
    const unsigned char* data,
    size_t size,
    size_t file_offset,
    size_t* offset,
    _SecantHsacoKernel* kernels,
    size_t num_kernels
) {
    const unsigned char* name = NULL;
    size_t name_size = 0u;
    size_t vgpr_offset = 0u;
    uint64_t vgpr_count = 0u;
    uint8_t vgpr_encoding = 0u;
    uint32_t count;
    uint32_t idx;
    size_t kernel_idx;

    _SECANT_HSACO_CHECK_RET(_secant_hsaco_msgpack_read_map(
        data,
        size,
        offset,
        &count));
    for (idx = 0u; idx < count; ++idx) {
        const unsigned char* key;
        size_t key_size;

        _SECANT_HSACO_CHECK_RET(_secant_hsaco_msgpack_read_string(
            data,
            size,
            offset,
            &key,
            &key_size));
        if (_secant_hsaco_msgpack_key_is(
                key,
                key_size,
                ".name")) {
            if (name != NULL) {
                _SECANT_HSACO_ERROR_RET(
                    SECANT_ERROR_PARSE_FAILED);
            }
            _SECANT_HSACO_CHECK_RET(
                _secant_hsaco_msgpack_read_string(
                    data,
                    size,
                    offset,
                    &name,
                    &name_size));
        } else if (_secant_hsaco_msgpack_key_is(
                key,
                key_size,
                ".vgpr_count")) {
            if (vgpr_encoding != 0u) {
                _SECANT_HSACO_ERROR_RET(
                    SECANT_ERROR_PARSE_FAILED);
            }
            _SECANT_HSACO_CHECK_RET(
                _secant_hsaco_msgpack_read_uint(
                    data,
                    size,
                    offset,
                    &vgpr_count,
                    &vgpr_offset,
                    &vgpr_encoding));
        } else {
            _SECANT_HSACO_CHECK_RET(_secant_hsaco_msgpack_skip(
                data,
                size,
                offset,
                0u));
        }
    }
    if (name == NULL || vgpr_encoding == 0u ||
        vgpr_count == 0u || vgpr_count > UINT32_MAX) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_PARSE_FAILED);
    }
    for (kernel_idx = 0u;
         kernel_idx < num_kernels;
         ++kernel_idx) {
        const size_t expected_name_size =
            strlen(kernels[kernel_idx].name);

        if (name_size != expected_name_size ||
            memcmp(
                name,
                kernels[kernel_idx].name,
                name_size) != 0) {
            continue;
        }
        if (kernels[kernel_idx].metadata_vgpr_file_offset !=
                SIZE_MAX ||
            vgpr_count != kernels[kernel_idx].register_count ||
            file_offset > SIZE_MAX - vgpr_offset) {
            _SECANT_HSACO_ERROR_RET(
                SECANT_ERROR_PARSE_FAILED);
        }
        kernels[kernel_idx].metadata_vgpr_file_offset =
            file_offset + vgpr_offset;
        kernels[kernel_idx].metadata_vgpr_encoding =
            vgpr_encoding;
        return SECANT_SUCCESS;
    }
    return SECANT_SUCCESS;
}

static SecantResult
_secant_hsaco_msgpack_parse_metadata(
    const unsigned char* data,
    size_t size,
    size_t file_offset,
    _SecantHsacoKernel* kernels,
    size_t num_kernels
) {
    size_t offset = 0u;
    uint32_t count;
    uint32_t idx;
    int found_kernels = 0;

    _SECANT_HSACO_CHECK_RET(_secant_hsaco_msgpack_read_map(
        data,
        size,
        &offset,
        &count));
    for (idx = 0u; idx < count; ++idx) {
        const unsigned char* key;
        size_t key_size;

        _SECANT_HSACO_CHECK_RET(_secant_hsaco_msgpack_read_string(
            data,
            size,
            &offset,
            &key,
            &key_size));
        if (_secant_hsaco_msgpack_key_is(
                key,
                key_size,
                "amdhsa.kernels")) {
            uint32_t num_entries;
            uint32_t entry_idx;

            if (found_kernels) {
                _SECANT_HSACO_ERROR_RET(
                    SECANT_ERROR_PARSE_FAILED);
            }
            found_kernels = 1;
            _SECANT_HSACO_CHECK_RET(
                _secant_hsaco_msgpack_read_array(
                    data,
                    size,
                    &offset,
                    &num_entries));
            for (entry_idx = 0u;
                 entry_idx < num_entries;
                 ++entry_idx) {
                _SECANT_HSACO_CHECK_RET(
                    _secant_hsaco_msgpack_parse_kernel(
                        data,
                        size,
                        file_offset,
                        &offset,
                        kernels,
                        num_kernels));
            }
        } else {
            _SECANT_HSACO_CHECK_RET(_secant_hsaco_msgpack_skip(
                data,
                size,
                &offset,
                0u));
        }
    }
    if (!found_kernels || offset != size) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_PARSE_FAILED);
    }
    for (idx = 0u; idx < num_kernels; ++idx) {
        if (kernels[idx].metadata_vgpr_file_offset == SIZE_MAX ||
            kernels[idx].metadata_vgpr_encoding == 0u) {
            _SECANT_HSACO_ERROR_RET(
                SECANT_ERROR_PARSE_FAILED);
        }
    }
    return SECANT_SUCCESS;
}

static SecantResult
_secant_hsaco_find_metadata(
    const _SecantHsacoElf* elf,
    _SecantHsacoKernel* kernels,
    size_t num_kernels
) {
    size_t section_idx;
    int found = 0;

    for (section_idx = 0u;
         section_idx < elf->header->e_shnum;
         ++section_idx) {
        const Elf64_Shdr* section = elf->sections + section_idx;
        size_t offset;
        size_t end;

        if (section->sh_type != SHT_NOTE) {
            continue;
        }
        if (!_secant_hsaco_range_ok(
                elf->size,
                (size_t)section->sh_offset,
                (size_t)section->sh_size)) {
            _SECANT_HSACO_ERROR_RET(SECANT_ERROR_PARSE_FAILED);
        }
        offset = (size_t)section->sh_offset;
        end = offset + (size_t)section->sh_size;
        while (offset < end) {
            Elf64_Nhdr header;
            size_t name_offset;
            size_t descriptor_offset;
            size_t next_offset;

            if (!_secant_hsaco_range_ok(
                    end,
                    offset,
                    sizeof(header))) {
                _SECANT_HSACO_ERROR_RET(
                    SECANT_ERROR_PARSE_FAILED);
            }
            memcpy(&header, elf->data + offset, sizeof(header));
            name_offset = offset + sizeof(header);
            descriptor_offset =
                name_offset +
                (((size_t)header.n_namesz + 3u) & ~3u);
            next_offset =
                descriptor_offset +
                (((size_t)header.n_descsz + 3u) & ~3u);
            if (descriptor_offset < name_offset ||
                next_offset < descriptor_offset ||
                next_offset > end ||
                !_secant_hsaco_range_ok(
                    end,
                    name_offset,
                    header.n_namesz) ||
                !_secant_hsaco_range_ok(
                    end,
                    descriptor_offset,
                    header.n_descsz)) {
                _SECANT_HSACO_ERROR_RET(
                    SECANT_ERROR_PARSE_FAILED);
            }
            if (header.n_type ==
                    _SECANT_HSACO_AMDGPU_METADATA_NOTE &&
                header.n_namesz == 7u &&
                memcmp(
                    elf->data + name_offset,
                    "AMDGPU",
                    7u) == 0) {
                if (found) {
                    _SECANT_HSACO_ERROR_RET(
                        SECANT_ERROR_PARSE_FAILED);
                }
                found = 1;
                _SECANT_HSACO_CHECK_RET(
                    _secant_hsaco_msgpack_parse_metadata(
                        elf->data + descriptor_offset,
                        header.n_descsz,
                        descriptor_offset,
                        kernels,
                        num_kernels));
            }
            offset = next_offset;
        }
    }
    _SECANT_HSACO_ERROR_RET(
        found
            ? SECANT_SUCCESS
            : SECANT_ERROR_PARSE_FAILED);
}

static SecantResult
_secant_hsaco_find_symbol(
    const _SecantHsacoElf* elf,
    const char* name,
    int symbol_type,
    _SecantHsacoSymbol* symbol_ret
) {
    size_t pass;

    if (elf == NULL || name == NULL || name[0] == '\0' ||
        symbol_ret == NULL) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    memset(symbol_ret, 0, sizeof(*symbol_ret));
    for (pass = 0u; pass < 2u; ++pass) {
        const uint32_t requested_type =
            pass == 0u ? SHT_SYMTAB : SHT_DYNSYM;
        size_t section_idx;

        for (section_idx = 0u;
             section_idx < elf->header->e_shnum;
             ++section_idx) {
            const Elf64_Shdr* symbol_table =
                elf->sections + section_idx;
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
                !_secant_hsaco_range_ok(
                    elf->size,
                    (size_t)symbol_table->sh_offset,
                    (size_t)symbol_table->sh_size)) {
                _SECANT_HSACO_ERROR_RET(
                    SECANT_ERROR_PARSE_FAILED);
            }
            strings = elf->sections + symbol_table->sh_link;
            if (!_secant_hsaco_range_ok(
                    elf->size,
                    (size_t)strings->sh_offset,
                    (size_t)strings->sh_size)) {
                _SECANT_HSACO_ERROR_RET(
                    SECANT_ERROR_PARSE_FAILED);
            }
            symbols = (const Elf64_Sym*)(
                elf->data + (size_t)symbol_table->sh_offset);
            string_data =
                (const char*)elf->data + (size_t)strings->sh_offset;
            num_symbols =
                (size_t)(symbol_table->sh_size /
                    symbol_table->sh_entsize);
            for (symbol_idx = 0u;
                 symbol_idx < num_symbols;
                 ++symbol_idx) {
                const Elf64_Sym* symbol = symbols + symbol_idx;
                const char* symbol_name;
                const Elf64_Shdr* section;
                const char* section_name;
                uint64_t section_offset;
                size_t file_offset;

                if (symbol->st_name >= strings->sh_size) {
                    _SECANT_HSACO_ERROR_RET(
                        SECANT_ERROR_PARSE_FAILED);
                }
                symbol_name = string_data + symbol->st_name;
                if (memchr(
                        symbol_name,
                        '\0',
                        (size_t)strings->sh_size -
                            symbol->st_name) == NULL) {
                    _SECANT_HSACO_ERROR_RET(
                        SECANT_ERROR_PARSE_FAILED);
                }
                if (strcmp(symbol_name, name) != 0 ||
                    (symbol_type >= 0 &&
                     ELF64_ST_TYPE(symbol->st_info) != symbol_type)) {
                    continue;
                }
                if (symbol->st_shndx == SHN_ABS) {
                    symbol_ret->symbol = symbol;
                    symbol_ret->symbol_index = symbol_idx;
                    symbol_ret->symbol_file_offset =
                        (size_t)symbol_table->sh_offset +
                        symbol_idx * sizeof(*symbol);
                    return SECANT_SUCCESS;
                }
                if (symbol->st_shndx == SHN_UNDEF ||
                    symbol->st_shndx >= elf->header->e_shnum) {
                    _SECANT_HSACO_ERROR_RET(
                        SECANT_ERROR_PARSE_FAILED);
                }
                section = elf->sections + symbol->st_shndx;
                section_name =
                    _secant_hsaco_section_name(elf, section);
                if (section_name == NULL ||
                    symbol->st_value < section->sh_addr) {
                    _SECANT_HSACO_ERROR_RET(
                        SECANT_ERROR_PARSE_FAILED);
                }
                section_offset =
                    symbol->st_value - section->sh_addr;
                if (section_offset > section->sh_size ||
                    symbol->st_size >
                        section->sh_size - section_offset ||
                    section->sh_offset > SIZE_MAX - section_offset) {
                    _SECANT_HSACO_ERROR_RET(
                        SECANT_ERROR_PARSE_FAILED);
                }
                file_offset =
                    (size_t)section->sh_offset +
                    (size_t)section_offset;
                if (!_secant_hsaco_range_ok(
                        elf->size,
                        file_offset,
                        (size_t)symbol->st_size)) {
                    _SECANT_HSACO_ERROR_RET(
                        SECANT_ERROR_PARSE_FAILED);
                }
                symbol_ret->symbol = symbol;
                symbol_ret->section = section;
                symbol_ret->section_name = section_name;
                symbol_ret->symbol_index = symbol_idx;
                symbol_ret->symbol_file_offset =
                    (size_t)symbol_table->sh_offset +
                    symbol_idx * sizeof(*symbol);
                symbol_ret->data_file_offset = file_offset;
                return SECANT_SUCCESS;
            }
        }
    }
    _SECANT_HSACO_ERROR_RET(SECANT_ERROR_SKELETON_NOT_FOUND);
}

static uint32_t
_secant_hsaco_arch_from_flags(uint32_t flags) {
    switch (flags & 0xffu) {
        case 0x48u:
            return 1200u;
        case 0x4eu:
            return 1201u;
        default:
            return 0u;
    }
}

static int
_secant_hsaco_is_trap(uint32_t word) {
    return word == _SECANT_HSACO_TRAP_WORD;
}

static int
_secant_hsaco_is_literal_add(uint32_t word) {
    return ((word >> 25u) & 0x3fu) == 3u &&
        (word & 0x1ffu) == _SECANT_HSACO_LITERAL_SOURCE;
}

static int
_secant_hsaco_is_register_add(uint32_t word) {
    return ((word >> 25u) & 0x3fu) == 3u &&
        (word & 0x1ffu) != _SECANT_HSACO_LITERAL_SOURCE;
}

static uint8_t
_secant_hsaco_vop2_dst(uint32_t word) {
    return (uint8_t)((word >> 17u) & 0xffu);
}

static uint8_t
_secant_hsaco_vop2_src1(uint32_t word) {
    return (uint8_t)((word >> 9u) & 0xffu);
}

static void
_secant_hsaco_add_reg(
    uint8_t* registers,
    size_t* count,
    uint8_t reg
) {
    size_t idx;

    for (idx = 0u; idx < *count; ++idx) {
        if (registers[idx] == reg) {
            return;
        }
    }
    if (*count < _SECANT_HSACO_MAX_REGISTERS) {
        registers[(*count)++] = reg;
    }
}

static int
_secant_hsaco_reg_in(
    const uint8_t* registers,
    size_t count,
    uint8_t reg
) {
    size_t idx;

    for (idx = 0u; idx < count; ++idx) {
        if (registers[idx] == reg) {
            return 1;
        }
    }
    return 0;
}

static void
_secant_hsaco_remove_reserved_regs(_SecantHsacoSite* site) {
    size_t read_idx;
    size_t write_idx = 0u;

    for (read_idx = 0u;
         read_idx < site->num_available_regs;
         ++read_idx) {
        const uint8_t reg = site->available_regs[read_idx];

        if (!_secant_hsaco_reg_in(
                site->input_regs,
                site->num_input_regs,
                reg) &&
            !_secant_hsaco_reg_in(
                site->target_regs,
                site->num_target_regs,
                reg) &&
            !_secant_hsaco_reg_in(
                site->output_regs,
                site->num_output_regs,
                reg)) {
            site->available_regs[write_idx++] = reg;
        }
    }
    site->num_available_regs = write_idx;
}

static SecantResult
_secant_hsaco_format_kernel_name(
    _SecantHsacoShape shape,
    size_t kernel_idx,
    char* name,
    size_t name_size
) {
    const char* pattern;
    int bytes;

    if (shape == _SECANT_HSACO_SHAPE_MATERIALIZE) {
        pattern = "secant_hsaco_materialize_%03zu";
    } else if (shape == _SECANT_HSACO_SHAPE_SSE) {
        pattern = "secant_hsaco_sse_%03zu";
    } else if (shape == _SECANT_HSACO_SHAPE_DYNAMIC_CONSTANT_SSE) {
        pattern = "secant_hsaco_dynamic_constant_sse_%03zu";
    } else {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    bytes = snprintf(name, name_size, pattern, kernel_idx);

    if (bytes < 0 || (size_t)bytes >= name_size) {
        _SECANT_HSACO_ERROR_RET(
            bytes < 0
                ? SECANT_ERROR_FORMAT
                : SECANT_ERROR_OVERFLOW);
    }
    return SECANT_SUCCESS;
}

static SecantResult
_secant_hsaco_register_metadata(
    const _SecantHsacoElf* elf,
    const char* function_name,
    _SecantHsacoKernel* kernel
) {
    char symbol_name[_SECANT_HSACO_NAME_BYTES + 32u];
    _SecantHsacoSymbol descriptor;
    _SecantHsacoSymbol num_vgpr;
    int bytes;
    uint32_t encoded_vgprs;

    bytes = snprintf(
        symbol_name,
        sizeof(symbol_name),
        "%s.kd",
        function_name);
    if (bytes < 0 || (size_t)bytes >= sizeof(symbol_name)) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_FORMAT);
    }
    _SECANT_HSACO_CHECK_RET(_secant_hsaco_find_symbol(
        elf,
        symbol_name,
        STT_OBJECT,
        &descriptor));
    if (descriptor.symbol->st_size <
            _SECANT_HSACO_COMPUTE_PGM_RSRC1_OFFSET +
                sizeof(uint32_t) ||
        !_secant_hsaco_range_ok(
            elf->size,
            descriptor.data_file_offset +
                _SECANT_HSACO_COMPUTE_PGM_RSRC1_OFFSET,
            sizeof(uint32_t))) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_PARSE_FAILED);
    }
    bytes = snprintf(
        symbol_name,
        sizeof(symbol_name),
        "%s.num_vgpr",
        function_name);
    if (bytes < 0 || (size_t)bytes >= sizeof(symbol_name)) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_FORMAT);
    }
    _SECANT_HSACO_CHECK_RET(_secant_hsaco_find_symbol(
        elf,
        symbol_name,
        -1,
        &num_vgpr));
    if (num_vgpr.symbol->st_shndx != SHN_ABS ||
        num_vgpr.symbol->st_value == 0u ||
        num_vgpr.symbol->st_value >
            _SECANT_HSACO_MAX_REGISTERS ||
        !_secant_hsaco_range_ok(
            elf->size,
            num_vgpr.symbol_file_offset,
            sizeof(Elf64_Sym))) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_PARSE_FAILED);
    }
    kernel->compute_pgm_rsrc1_file_offset =
        descriptor.data_file_offset +
        _SECANT_HSACO_COMPUTE_PGM_RSRC1_OFFSET;
    kernel->compute_pgm_rsrc1 = _secant_hsaco_read_u32(
        elf->data + kernel->compute_pgm_rsrc1_file_offset);
    encoded_vgprs =
        kernel->compute_pgm_rsrc1 &
        _SECANT_HSACO_COMPUTE_PGM_RSRC1_VGPRS_MASK;
    kernel->allocated_registers = (encoded_vgprs + 1u) * 8u;
    kernel->num_vgpr_symbol_value_file_offset =
        num_vgpr.symbol_file_offset + offsetof(Elf64_Sym, st_value);
    kernel->register_count = (uint32_t)num_vgpr.symbol->st_value;
    if (kernel->register_count > kernel->allocated_registers) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_PARSE_FAILED);
    }
    return SECANT_SUCCESS;
}

static SecantResult
_secant_hsaco_scan_site(
    const _SecantHsacoElf* elf,
    const _SecantHsacoSymbol* function,
    _SecantHsacoShape shape,
    uint32_t marker,
    size_t asts_per_kernel,
    size_t num_inputs,
    size_t num_targets,
    size_t patch_capacity_instructions,
    _SecantHsacoSite* site_ret
) {
    const size_t function_start = function->data_file_offset;
    const size_t function_end =
        function_start + (size_t)function->symbol->st_size;
    _SecantHsacoSite site;
    uint8_t marker_seen[_SECANT_HSACO_MAX_REGISTERS] = { 0u };
    size_t num_sources;
    size_t num_outputs;
    size_t marker_count;
    size_t expected_pad_traps;
    size_t first_marker = SIZE_MAX;
    size_t pad_start = SIZE_MAX;
    size_t pad_end = SIZE_MAX;
    size_t offset;
    size_t marker_idx;

    if (!_secant_hsaco_checked_add(
            num_inputs,
            num_targets,
            &num_sources)) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_OVERFLOW);
    }
    if (shape == _SECANT_HSACO_SHAPE_MATERIALIZE) {
        num_outputs = asts_per_kernel;
    } else if (shape == _SECANT_HSACO_SHAPE_SSE ||
               shape == _SECANT_HSACO_SHAPE_DYNAMIC_CONSTANT_SSE) {
        if (!_secant_hsaco_checked_mul(
                asts_per_kernel,
                num_targets,
                &num_outputs)) {
            _SECANT_HSACO_ERROR_RET(SECANT_ERROR_OVERFLOW);
        }
    } else {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    if (!_secant_hsaco_checked_add(
            num_sources,
            num_outputs,
            &marker_count) ||
        marker_count > _SECANT_HSACO_MAX_REGISTERS) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_OVERFLOW);
    }
    expected_pad_traps = patch_capacity_instructions;
    memset(&site, 0, sizeof(site));
    site.num_input_regs = num_inputs;
    site.num_target_regs = num_targets;
    site.num_output_regs = num_outputs;

    for (offset = function_start + 4u;
         offset + 4u <= function_end;
         offset += 4u) {
        const uint32_t word =
            _secant_hsaco_read_u32(elf->data + offset - 4u);
        const uint32_t literal =
            _secant_hsaco_read_u32(elf->data + offset);

        if (!_secant_hsaco_is_literal_add(word) ||
            literal < marker ||
            literal - marker >= marker_count) {
            continue;
        }
        marker_idx = literal - marker;
        if (marker_seen[marker_idx]) {
            _SECANT_HSACO_ERROR_RET(SECANT_ERROR_PARSE_FAILED);
        }
        marker_seen[marker_idx] = 1u;
        if (first_marker == SIZE_MAX) {
            first_marker = offset - 4u;
        }
        if (marker_idx < num_sources) {
            const uint8_t source = _secant_hsaco_vop2_src1(word);

            if (marker_idx < num_inputs) {
                site.input_regs[marker_idx] = source;
            } else {
                site.target_regs[marker_idx - num_inputs] = source;
            }
            _secant_hsaco_add_reg(
                site.available_regs,
                &site.num_available_regs,
                _secant_hsaco_vop2_dst(word));
        } else {
            const size_t output_idx = marker_idx - num_sources;
            const uint8_t source = _secant_hsaco_vop2_src1(word);
            const uint8_t destination =
                _secant_hsaco_vop2_dst(word);

            if ((shape == _SECANT_HSACO_SHAPE_SSE ||
                 shape == _SECANT_HSACO_SHAPE_DYNAMIC_CONSTANT_SSE) &&
                source != destination) {
                _SECANT_HSACO_ERROR_RET(
                    SECANT_ERROR_PARSE_FAILED);
            }
            site.output_regs[output_idx] = destination;
        }
        offset += 4u;
    }
    if (first_marker == SIZE_MAX ||
        first_marker < function_start + 4u ||
        !_secant_hsaco_is_trap(_secant_hsaco_read_u32(
            elf->data + first_marker - 4u))) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_SKELETON_NOT_FOUND);
    }
    for (marker_idx = 0u; marker_idx < marker_count; ++marker_idx) {
        if (!marker_seen[marker_idx]) {
            _SECANT_HSACO_ERROR_RET(SECANT_ERROR_PARSE_FAILED);
        }
    }

    for (offset = first_marker;
         offset + 4u <= function_end;
         offset += 4u) {
        size_t run_end;
        size_t run_count = 0u;

        if (!_secant_hsaco_is_trap(
                _secant_hsaco_read_u32(elf->data + offset))) {
            continue;
        }
        run_end = offset;
        while (run_end + 4u <= function_end &&
               _secant_hsaco_is_trap(
                   _secant_hsaco_read_u32(elf->data + run_end))) {
            run_count += 1u;
            run_end += 4u;
        }
        if (run_count == expected_pad_traps) {
            if (pad_start != SIZE_MAX) {
                _SECANT_HSACO_ERROR_RET(
                    SECANT_ERROR_PARSE_FAILED);
            }
            pad_start = offset;
            pad_end = run_end;
        }
        offset = run_end - 4u;
    }
    if (pad_start == SIZE_MAX || pad_end == SIZE_MAX ||
        pad_start <= first_marker || pad_end >= function_end) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_PARSE_FAILED);
    }

    for (offset = first_marker; offset < pad_start;) {
        const uint32_t word =
            _secant_hsaco_read_u32(elf->data + offset);

        if (_secant_hsaco_is_trap(word) ||
            (word & _SECANT_HSACO_DELAY_ALU_MASK) ==
                _SECANT_HSACO_DELAY_ALU_WORD) {
            offset += 4u;
        } else if ((word & _SECANT_HSACO_VOPD_ADD_MASK) ==
                   _SECANT_HSACO_VOPD_ADD_WORD) {
            uint32_t word1;
            uint8_t first_dst;
            uint8_t second_dst;

            if (offset + 8u > pad_start) {
                _SECANT_HSACO_ERROR_RET(
                    SECANT_ERROR_UNEXPECTED_INSTRUCTION);
            }
            word1 = _secant_hsaco_read_u32(
                elf->data + offset + 4u);
            first_dst = (uint8_t)(word1 >> 24u);
            second_dst = (uint8_t)(
                ((word1 >> 16u) & 0xfeu) |
                ((first_dst & 1u) ^ 1u));
            _secant_hsaco_add_reg(
                site.available_regs,
                &site.num_available_regs,
                first_dst);
            _secant_hsaco_add_reg(
                site.available_regs,
                &site.num_available_regs,
                second_dst);
            offset += 8u;
        } else if (_secant_hsaco_is_literal_add(word)) {
            if (offset + 8u > pad_start) {
                _SECANT_HSACO_ERROR_RET(
                    SECANT_ERROR_UNEXPECTED_INSTRUCTION);
            }
            offset += 8u;
        } else if (_secant_hsaco_is_register_add(word)) {
            _secant_hsaco_add_reg(
                site.available_regs,
                &site.num_available_regs,
                _secant_hsaco_vop2_dst(word));
            offset += 4u;
        } else {
            _SECANT_HSACO_ERROR_RET(
                SECANT_ERROR_UNEXPECTED_INSTRUCTION);
        }
    }
    for (offset = pad_end;
         offset + 4u <= function_end;
         offset += 4u) {
        if (_secant_hsaco_is_trap(
                _secant_hsaco_read_u32(elf->data + offset))) {
            _SECANT_HSACO_ERROR_RET(SECANT_ERROR_PARSE_FAILED);
        }
    }
    for (marker_idx = 0u; marker_idx < num_inputs; ++marker_idx) {
        if (_secant_hsaco_reg_in(
                site.input_regs,
                marker_idx,
                site.input_regs[marker_idx])) {
            _SECANT_HSACO_ERROR_RET(SECANT_ERROR_PARSE_FAILED);
        }
    }
    for (marker_idx = 0u; marker_idx < num_targets; ++marker_idx) {
        if (_secant_hsaco_reg_in(
                site.target_regs,
                marker_idx,
                site.target_regs[marker_idx]) ||
            _secant_hsaco_reg_in(
                site.input_regs,
                num_inputs,
                site.target_regs[marker_idx])) {
            _SECANT_HSACO_ERROR_RET(SECANT_ERROR_PARSE_FAILED);
        }
    }
    for (marker_idx = 0u; marker_idx < num_outputs; ++marker_idx) {
        if (_secant_hsaco_reg_in(
                site.output_regs,
                marker_idx,
                site.output_regs[marker_idx]) ||
            _secant_hsaco_reg_in(
                site.input_regs,
                num_inputs,
                site.output_regs[marker_idx]) ||
            _secant_hsaco_reg_in(
                site.target_regs,
                num_targets,
                site.output_regs[marker_idx])) {
            _SECANT_HSACO_ERROR_RET(SECANT_ERROR_PARSE_FAILED);
        }
    }
    site.load_fence_file_offset = first_marker - 4u;
    site.start_file_offset = first_marker;
    site.end_file_offset = pad_end;
    site.patch_size = pad_end - first_marker;
    _secant_hsaco_remove_reserved_regs(&site);
    *site_ret = site;
    return SECANT_SUCCESS;
}

static SecantResult
_secant_hsaco_scan_kernel(
    const _SecantHsacoElf* elf,
    _SecantHsacoShape shape,
    size_t kernel_idx,
    size_t asts_per_kernel,
    size_t num_inputs,
    size_t num_targets,
    size_t patch_capacity_instructions,
    _SecantHsacoKernel* kernel_ret
) {
    _SecantHsacoSymbol function;
    _SecantHsacoKernel kernel;
    size_t num_outputs;
    size_t marker_count;
    size_t marker_offset;

    memset(&kernel, 0, sizeof(kernel));
    kernel.metadata_vgpr_file_offset = SIZE_MAX;
    _SECANT_HSACO_CHECK_RET(_secant_hsaco_format_kernel_name(
        shape,
        kernel_idx,
        kernel.name,
        sizeof(kernel.name)));
    _SECANT_HSACO_CHECK_RET(_secant_hsaco_find_symbol(
        elf,
        kernel.name,
        STT_FUNC,
        &function));
    if (function.symbol->st_size == 0u ||
        (function.data_file_offset & 3u) != 0u ||
        (function.symbol->st_size & 3u) != 0u) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_PARSE_FAILED);
    }
    if (shape == _SECANT_HSACO_SHAPE_MATERIALIZE) {
        num_outputs = asts_per_kernel;
        num_targets = 0u;
    } else if (!_secant_hsaco_checked_mul(
            asts_per_kernel,
            num_targets,
            &num_outputs)) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_OVERFLOW);
    }
    if (!_secant_hsaco_checked_add(
            num_inputs,
            num_targets,
            &marker_count) ||
        !_secant_hsaco_checked_add(
            marker_count,
            num_outputs,
            &marker_count) ||
        !_secant_hsaco_checked_mul(
            kernel_idx,
            marker_count,
            &marker_offset) ||
        marker_offset >
            UINT32_MAX - _SECANT_HSACO_FIRST_MARKER_BITS) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_OVERFLOW);
    }
    _SECANT_HSACO_CHECK_RET(_secant_hsaco_register_metadata(
        elf,
        kernel.name,
        &kernel));
    _SECANT_HSACO_CHECK_RET(_secant_hsaco_scan_site(
        elf,
        &function,
        shape,
        _SECANT_HSACO_FIRST_MARKER_BITS +
            (uint32_t)marker_offset,
        asts_per_kernel,
        num_inputs,
        num_targets,
        patch_capacity_instructions,
        &kernel.site));
    *kernel_ret = kernel;
    return SECANT_SUCCESS;
}

static SecantResult
_secant_hsaco_scan_template(
    _SecantHsacoTemplate* template_data,
    const void* hsaco,
    size_t hsaco_size,
    _SecantHsacoKernel* kernels,
    size_t* isa_capacity_ret
) {
    _SecantHsacoElf elf;
    size_t max_patch_size = 0u;
    size_t kernel_idx;
    uint32_t gfx_arch;

    _SECANT_HSACO_CHECK_RET(_secant_hsaco_parse_elf(
        hsaco,
        hsaco_size,
        &elf));
    gfx_arch = _secant_hsaco_arch_from_flags(elf.header->e_flags);
    if (gfx_arch != 1200u && gfx_arch != 1201u) {
        _SECANT_HSACO_ERROR_RET(
            SECANT_ERROR_UNSUPPORTED_ARCHITECTURE);
    }
    template_data->gfx_arch = gfx_arch;
    for (kernel_idx = 0u;
         kernel_idx < template_data->num_kernels;
         ++kernel_idx) {
        _SecantHsacoKernel kernel;

        _SECANT_HSACO_CHECK_RET(_secant_hsaco_scan_kernel(
            &elf,
            template_data->shape,
            kernel_idx,
            template_data->asts_per_kernel,
            template_data->num_inputs,
            template_data->num_targets,
            template_data->patch_capacity_instructions,
            &kernel));
        _SECANT_HSACO_CHECK_RET(_secant_hsaco_find_metadata(
            &elf,
            &kernel,
            1u));
        if (kernels != NULL) {
            kernels[kernel_idx] = kernel;
        }
        if (kernel.site.patch_size > max_patch_size) {
            max_patch_size = kernel.site.patch_size;
        }
    }
    if (max_patch_size == 0u) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_PARSE_FAILED);
    }
    *isa_capacity_ret = max_patch_size;
    return SECANT_SUCCESS;
}
static SecantResult
_secant_hsaco_validate_inspect_args(
    _SecantHsacoShape shape,
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
        asts_per_kernel > _SECANT_HSACO_MAX_REGISTERS ||
        num_inputs == 0u || num_inputs > _SECANT_HSACO_MAX_REGISTERS ||
        patch_capacity_instructions == 0u) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    if (shape == _SECANT_HSACO_SHAPE_MATERIALIZE) {
        num_outputs = asts_per_kernel;
        num_targets = 0u;
    } else if (shape == _SECANT_HSACO_SHAPE_SSE ||
               shape == _SECANT_HSACO_SHAPE_DYNAMIC_CONSTANT_SSE) {
        if (num_targets == 0u ||
            !_secant_hsaco_checked_mul(
                asts_per_kernel,
                num_targets,
                &num_outputs)) {
            _SECANT_HSACO_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
        }
    } else {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    if (!_secant_hsaco_checked_add(
            num_inputs,
            num_targets,
            &marker_count) ||
        !_secant_hsaco_checked_add(
            marker_count,
            num_outputs,
            &marker_count) ||
        marker_count > _SECANT_HSACO_MAX_REGISTERS ||
        !_secant_hsaco_checked_mul(
            num_kernels,
            marker_count,
            &total_markers) ||
        total_markers - 1u >
            0x7fffffffu - _SECANT_HSACO_FIRST_MARKER_BITS) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    return SECANT_SUCCESS;
}

static int
_secant_hsaco_checked_align(
    size_t value,
    size_t alignment,
    size_t* result_ret
) {
    const size_t mask = alignment - 1u;

    if (alignment == 0u || (alignment & mask) != 0u ||
        value > SIZE_MAX - mask) {
        return 0;
    }
    *result_ret = (value + mask) & ~mask;
    return 1;
}

static int
_secant_hsaco_workspace_layout(
    size_t num_kernels,
    size_t* kernels_offset_ret,
    size_t* payload_size_ret
) {
    size_t kernels_bytes;
    size_t offset = sizeof(struct SecantHsacoPlan);

    if (!_secant_hsaco_checked_mul(
            num_kernels,
            sizeof(_SecantHsacoKernel),
            &kernels_bytes) ||
        !_secant_hsaco_checked_align(
            offset,
            _SECANT_HSACO_WORKSPACE_ALIGNMENT,
            &offset)) {
        return 0;
    }
    *kernels_offset_ret = offset;
    if (!_secant_hsaco_checked_add(offset, kernels_bytes, &offset) ||
        !_secant_hsaco_checked_align(
            offset,
            _SECANT_HSACO_WORKSPACE_ALIGNMENT,
            &offset)) {
        return 0;
    }
    *payload_size_ret = offset;
    return 1;
}

static SecantResult
_secant_hsaco_inspect(
    _SecantHsacoShape shape,
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    size_t num_input_constants,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    size_t patch_capacity_instructions,
    const void* hsaco,
    size_t hsaco_size,
    void* workspace,
    size_t workspace_size,
    size_t* workspace_size_ret,
    SecantHsacoPlan** plan_ret
) {
    _SecantHsacoTemplate measured_template;
    SecantHsacoPlan* plan;
    size_t isa_capacity;
    size_t kernels_offset;
    size_t payload_size;
    size_t required_size;
    uintptr_t workspace_address;
    uintptr_t aligned_address;
    unsigned char* aligned_workspace;

    if (hsaco == NULL || hsaco_size == 0u ||
        workspace_size_ret == NULL || plan_ret == NULL ||
        (workspace == NULL && workspace_size != 0u)) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    *plan_ret = NULL;
    *workspace_size_ret = 0u;
    _SECANT_HSACO_CHECK_RET(_secant_hsaco_validate_inspect_args(
        shape,
        num_kernels,
        asts_per_kernel,
        num_inputs,
        num_targets,
        patch_capacity_instructions));
    if (shape == _SECANT_HSACO_SHAPE_MATERIALIZE) {
        if (num_targets != 0u || tile_rows != 0u ||
            threads_per_block != 0u) {
            _SECANT_HSACO_ERROR_RET(
                SECANT_ERROR_UNSUPPORTED_SHAPE);
        }
    } else if (shape == _SECANT_HSACO_SHAPE_SSE &&
        (num_targets == 0u ||
         tile_rows == 0u ||
         threads_per_block < 64u ||
         threads_per_block > tile_rows ||
         threads_per_block > _SECANT_HSACO_MAX_WAVES * 64u ||
         threads_per_block % 64u != 0u)) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_UNSUPPORTED_SHAPE);
    } else if (shape == _SECANT_HSACO_SHAPE_DYNAMIC_CONSTANT_SSE &&
        (num_input_constants == 0u ||
         num_targets == 0u ||
         tile_rows == 0u ||
         threads_per_block == 0u ||
         threads_per_block > 1024u)) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_UNSUPPORTED_SHAPE);
    } else if (shape != _SECANT_HSACO_SHAPE_SSE &&
        shape != _SECANT_HSACO_SHAPE_DYNAMIC_CONSTANT_SSE &&
        shape != _SECANT_HSACO_SHAPE_MATERIALIZE) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_UNSUPPORTED_SHAPE);
    }
    if (shape != _SECANT_HSACO_SHAPE_DYNAMIC_CONSTANT_SSE &&
        num_input_constants != 0u) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_UNSUPPORTED_SHAPE);
    }
    memset(&measured_template, 0, sizeof(measured_template));
    measured_template.shape = shape;
    measured_template.hsaco_size = hsaco_size;
    measured_template.num_kernels = num_kernels;
    measured_template.asts_per_kernel = asts_per_kernel;
    measured_template.num_inputs = num_inputs;
    measured_template.num_input_constants = num_input_constants;
    measured_template.num_targets = num_targets;
    measured_template.tile_rows = tile_rows;
    measured_template.threads_per_block = threads_per_block;
    measured_template.patch_capacity_instructions =
        patch_capacity_instructions;
    _SECANT_HSACO_CHECK_RET(_secant_hsaco_scan_template(
        &measured_template,
        hsaco,
        hsaco_size,
        NULL,
        &isa_capacity));
    if (!_secant_hsaco_workspace_layout(
            num_kernels,
            &kernels_offset,
            &payload_size) ||
        !_secant_hsaco_checked_add(
            payload_size,
            _SECANT_HSACO_WORKSPACE_ALIGNMENT - 1u,
            &required_size)) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_OVERFLOW);
    }
    *workspace_size_ret = required_size;
    if (workspace == NULL) {
        return SECANT_SUCCESS;
    }
    if (workspace_size < required_size) {
        _SECANT_HSACO_ERROR_RET(
            SECANT_ERROR_INSUFFICIENT_BUFFER);
    }
    workspace_address = (uintptr_t)workspace;
    if (workspace_address >
        UINTPTR_MAX - (_SECANT_HSACO_WORKSPACE_ALIGNMENT - 1u)) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_OVERFLOW);
    }
    aligned_address =
        (workspace_address + _SECANT_HSACO_WORKSPACE_ALIGNMENT - 1u) &
        ~(uintptr_t)(_SECANT_HSACO_WORKSPACE_ALIGNMENT - 1u);
    aligned_workspace = (unsigned char*)aligned_address;
    memset(aligned_workspace, 0, payload_size);
    plan = (SecantHsacoPlan*)aligned_workspace;
    plan->template_data = measured_template;
    plan->template_data.kernels =
        (_SecantHsacoKernel*)(aligned_workspace + kernels_offset);
    _SECANT_HSACO_CHECK_RET(_secant_hsaco_scan_template(
        &plan->template_data,
        hsaco,
        hsaco_size,
        plan->template_data.kernels,
        &isa_capacity));
    *plan_ret = plan;
    return SECANT_SUCCESS;
}

SecantResult
secant_hsaco_materialize_inspect(
    const SecantHsacoMaterializeRecipe* recipe,
    const void* hsaco,
    size_t hsaco_size,
    void* plan_storage,
    size_t plan_storage_size,
    size_t* required_plan_storage_ret,
    SecantHsacoPlan** plan_ret
) {
    if (recipe == NULL) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    _SECANT_HSACO_ERROR_RET(_secant_hsaco_inspect(
        _SECANT_HSACO_SHAPE_MATERIALIZE,
        recipe->num_kernels,
        recipe->asts_per_kernel,
        recipe->num_inputs,
        0u,
        0u,
        0u,
        0u,
        recipe->patch_capacity_instructions,
        hsaco,
        hsaco_size,
        plan_storage,
        plan_storage_size,
        required_plan_storage_ret,
        plan_ret));
}

SecantResult
secant_hsaco_sse_inspect(
    const SecantHsacoSSERecipe* recipe,
    const void* hsaco,
    size_t hsaco_size,
    void* plan_storage,
    size_t plan_storage_size,
    size_t* required_plan_storage_ret,
    SecantHsacoPlan** plan_ret
) {
    if (recipe == NULL ||
        (recipe->reduction_mode !=
             SECANT_SSE_REDUCTION_MODE_ATOMIC &&
         recipe->reduction_mode !=
             SECANT_SSE_REDUCTION_MODE_WORKSPACE)) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    _SECANT_HSACO_ERROR_RET(_secant_hsaco_inspect(
        _SECANT_HSACO_SHAPE_SSE,
        recipe->num_kernels,
        recipe->asts_per_kernel,
        recipe->num_inputs,
        0u,
        recipe->num_targets,
        recipe->tile_rows,
        recipe->threads_per_block,
        recipe->patch_capacity_instructions,
        hsaco,
        hsaco_size,
        plan_storage,
        plan_storage_size,
        required_plan_storage_ret,
        plan_ret));
}

SecantResult
secant_hsaco_dynamic_constant_sse_inspect(
    const SecantHsacoDynamicConstantSSERecipe* recipe,
    const void* hsaco,
    size_t hsaco_size,
    void* plan_storage,
    size_t plan_storage_size,
    size_t* required_plan_storage_ret,
    SecantHsacoPlan** plan_ret
) {
    size_t num_inputs;

    if (recipe == NULL ||
        (recipe->reduction_mode !=
             SECANT_SSE_REDUCTION_MODE_ATOMIC &&
         recipe->reduction_mode !=
             SECANT_SSE_REDUCTION_MODE_WORKSPACE) ||
        !_secant_hsaco_checked_add(
            recipe->num_input_columns,
            recipe->num_input_constants,
            &num_inputs)) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    _SECANT_HSACO_ERROR_RET(_secant_hsaco_inspect(
        _SECANT_HSACO_SHAPE_DYNAMIC_CONSTANT_SSE,
        recipe->num_kernels,
        recipe->asts_per_kernel,
        num_inputs,
        recipe->num_input_constants,
        recipe->num_targets,
        recipe->tile_rows,
        recipe->threads_per_block,
        recipe->patch_capacity_instructions,
        hsaco,
        hsaco_size,
        plan_storage,
        plan_storage_size,
        required_plan_storage_ret,
        plan_ret));
}

SecantResult
secant_hsaco_materialize_plan_info_get(
    const SecantHsacoPlan* plan,
    size_t* hsaco_size_ret,
    size_t* num_kernels_ret,
    size_t* asts_per_kernel_ret,
    size_t* num_inputs_ret
) {
    if (plan == NULL ||
        plan->template_data.shape != _SECANT_HSACO_SHAPE_MATERIALIZE ||
        hsaco_size_ret == NULL || num_kernels_ret == NULL ||
        asts_per_kernel_ret == NULL || num_inputs_ret == NULL) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    *hsaco_size_ret = plan->template_data.hsaco_size;
    *num_kernels_ret = plan->template_data.num_kernels;
    *asts_per_kernel_ret = plan->template_data.asts_per_kernel;
    *num_inputs_ret = plan->template_data.num_inputs;
    return SECANT_SUCCESS;
}

SecantResult
secant_hsaco_dynamic_constant_sse_plan_info_get(
    const SecantHsacoPlan* plan,
    size_t* hsaco_size_ret,
    size_t* num_kernels_ret,
    size_t* asts_per_kernel_ret,
    size_t* num_input_columns_ret,
    size_t* num_input_constants_ret,
    size_t* num_targets_ret,
    size_t* tile_rows_ret,
    size_t* threads_per_block_ret
) {
    if (plan == NULL ||
        plan->template_data.shape !=
            _SECANT_HSACO_SHAPE_DYNAMIC_CONSTANT_SSE ||
        hsaco_size_ret == NULL || num_kernels_ret == NULL ||
        asts_per_kernel_ret == NULL ||
        num_input_columns_ret == NULL ||
        num_input_constants_ret == NULL ||
        num_targets_ret == NULL || tile_rows_ret == NULL ||
        threads_per_block_ret == NULL ||
        plan->template_data.num_input_constants >
            plan->template_data.num_inputs) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    *hsaco_size_ret = plan->template_data.hsaco_size;
    *num_kernels_ret = plan->template_data.num_kernels;
    *asts_per_kernel_ret = plan->template_data.asts_per_kernel;
    *num_input_columns_ret =
        plan->template_data.num_inputs -
        plan->template_data.num_input_constants;
    *num_input_constants_ret =
        plan->template_data.num_input_constants;
    *num_targets_ret = plan->template_data.num_targets;
    *tile_rows_ret = plan->template_data.tile_rows;
    *threads_per_block_ret = plan->template_data.threads_per_block;
    return SECANT_SUCCESS;
}

SecantResult
secant_hsaco_sse_plan_info_get(
    const SecantHsacoPlan* plan,
    size_t* hsaco_size_ret,
    size_t* num_kernels_ret,
    size_t* asts_per_kernel_ret,
    size_t* num_inputs_ret,
    size_t* num_targets_ret,
    size_t* tile_rows_ret,
    size_t* threads_per_block_ret
) {
    if (plan == NULL ||
        plan->template_data.shape != _SECANT_HSACO_SHAPE_SSE ||
        hsaco_size_ret == NULL || num_kernels_ret == NULL ||
        asts_per_kernel_ret == NULL || num_inputs_ret == NULL ||
        num_targets_ret == NULL || tile_rows_ret == NULL ||
        threads_per_block_ret == NULL) {
        _SECANT_HSACO_ERROR_RET(SECANT_ERROR_INVALID_VALUE);
    }
    *hsaco_size_ret = plan->template_data.hsaco_size;
    *num_kernels_ret = plan->template_data.num_kernels;
    *asts_per_kernel_ret = plan->template_data.asts_per_kernel;
    *num_inputs_ret = plan->template_data.num_inputs;
    *num_targets_ret = plan->template_data.num_targets;
    *tile_rows_ret = plan->template_data.tile_rows;
    *threads_per_block_ret = plan->template_data.threads_per_block;
    return SECANT_SUCCESS;
}

#undef _SECANT_HSACO_CHECK_RET
#undef _SECANT_HSACO_ERROR_RET
