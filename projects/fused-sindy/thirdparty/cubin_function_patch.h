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
 * SPDX-FileCopyrightText: Copyright (c) 2026 Charles Durham
 * SPDX-License-Identifier: MIT
 *
 * cubin_function_patch.h
 *
 * Single-header C99 API for replacing reserved CUDA function bodies in a fully
 * linked cubin/ELF with self-contained function bodies from nvPTXCompiler
 * --compile-only RDC ELF output.
 *
 * Contract:
 * - The handle stores a pointer to the linked cubin template; the caller keeps
 *   that cubin alive and immutable for the handle lifetime.
 * - The handle is immutable after create. Multiple threads may call patch
 *   routines through the same handle when each call uses its own output cubin
 *   buffer.
 * - The replacement function must have the same symbol name, fit inside the
 *   reserved linked function body, target the same ELF machine, and have no
 *   relocations targeting the replacement function text except for supported
 *   CUDA constant-bank relocations into matching reserved side sections.
 * - The linked cubin metadata is preserved. Build the reserved template with
 *   worst-case register/local/stack requirements for the functions being
 *   replaced.
 * - On cubins with NVIDIA cap/merc text mirror sections, the patcher also
 *   patches the matching .nv.capmerc.text.<symbol> section. Metadata remains
 *   the template metadata.
 * - No allocation is performed internally. Create uses caller-provided handle
 *   memory. Patch writes the new cubin into caller-provided output memory.
 *
 * Define CUBIN_FUNCTION_PATCH_IMPLEMENTATION in exactly one translation unit.
 */
#ifndef CUBIN_FUNCTION_PATCH_H_INCLUDE
#define CUBIN_FUNCTION_PATCH_H_INCLUDE

#define CUBIN_FUNCTION_PATCH_VERSION_MAJOR 1 //!< Cubin Function Patch major version.
#define CUBIN_FUNCTION_PATCH_VERSION_MINOR 0 //!< Cubin Function Patch minor version.
#define CUBIN_FUNCTION_PATCH_VERSION_PATCH 1 //!< Cubin Function Patch patch version.

#ifndef CUBIN_FUNCTION_PATCH_VERSION_STRING
#define CUBIN_FUNCTION_PATCH_VERSION_STRING "1.0.1"
#endif

#define CUBIN_FUNCTION_PATCH_VERSION \
    (CUBIN_FUNCTION_PATCH_VERSION_MAJOR * 10000 + \
     CUBIN_FUNCTION_PATCH_VERSION_MINOR * 100 + \
     CUBIN_FUNCTION_PATCH_VERSION_PATCH)

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#define CUBIN_FUNCTION_PATCH_PUBLIC_DEC extern "C"
#define CUBIN_FUNCTION_PATCH_PUBLIC_DEF extern "C"
#else
#define CUBIN_FUNCTION_PATCH_PUBLIC_DEC extern
#define CUBIN_FUNCTION_PATCH_PUBLIC_DEF
#endif

/**
 * Result code returned by cubin_function_patch API calls.
 */
typedef enum {
    CUBIN_FUNCTION_PATCH_SUCCESS = 0,
    CUBIN_FUNCTION_PATCH_ERROR_INVALID_VALUE = 1,
    CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF = 2,
    CUBIN_FUNCTION_PATCH_ERROR_WORKSPACE_EXHAUSTED = 3,
    CUBIN_FUNCTION_PATCH_ERROR_SYMBOL_NOT_FOUND = 4,
    CUBIN_FUNCTION_PATCH_ERROR_REPLACEMENT_TOO_LARGE = 5,
    CUBIN_FUNCTION_PATCH_ERROR_RELOCATION_UNSUPPORTED = 6,
    CUBIN_FUNCTION_PATCH_ERROR_MACHINE_MISMATCH = 7,
    CUBIN_FUNCTION_PATCH_ERROR_AMBIGUOUS_SYMBOL = 8,
    CUBIN_FUNCTION_PATCH_ERROR_TAIL_POLICY = 9
} CubinFunctionPatchResult;

/**
 * Policy for bytes after a smaller replacement function body.
 */
typedef enum {
    CUBIN_FUNCTION_PATCH_TAIL_LEAVE_UNCHANGED = 0,
    CUBIN_FUNCTION_PATCH_TAIL_REQUIRE_EXACT_SIZE = 1
} CubinFunctionPatchTailPolicy;

/**
 * Immutable patch handle created from a fully linked cubin template.
 *
 * The handle does not own the linked cubin bytes. The caller must keep the
 * linked cubin alive and immutable until cubin_function_patch_destroy is called.
 * After creation, the same handle can be used concurrently by multiple threads
 * as long as each thread writes to a distinct output cubin buffer.
 */
typedef struct CubinFunctionPatchHandle CubinFunctionPatchHandle;

/**
 * Optional per-symbol patch report.
 *
 * symbol_name points into the handle and remains valid for the handle lifetime.
 * reserved_size is the linked template function-body size. replacement_size is
 * the size copied from the replacement RDC ELF.
 */
typedef struct {
    const char* symbol_name;
    size_t reserved_size;
    size_t replacement_size;
} CubinFunctionPatchReport;

/**
 * Return a stable string for a CubinFunctionPatchResult value.
 */
CUBIN_FUNCTION_PATCH_PUBLIC_DEC const char* cubin_function_patch_result_to_string(CubinFunctionPatchResult result);

/**
 * Measure caller-provided memory required for a patch handle.
 *
 * symbol_names is the list of function symbols that will be patchable. The same
 * list must be passed to cubin_function_patch_create. The returned byte count
 * includes storage for copied symbol names.
 */
CUBIN_FUNCTION_PATCH_PUBLIC_DEC CubinFunctionPatchResult
cubin_function_patch_handle_size(
    const char* const* symbol_names,
    size_t num_symbols,
    size_t* bytes_out
);

/**
 * Create a patch handle by scanning a fully linked cubin template once.
 *
 * linked_cubin must be a fully linked CUDA cubin/ELF containing each requested
 * symbol as an STT_FUNC symbol in an executable section. handle_memory is caller
 * owned and must be at least the size returned by cubin_function_patch_handle_size.
 *
 * The function records each symbol's linked text offset and reserved size. It
 * does not copy linked_cubin; keep linked_cubin alive for the handle lifetime.
 */
CUBIN_FUNCTION_PATCH_PUBLIC_DEC CubinFunctionPatchResult
cubin_function_patch_create(
    const void* linked_cubin,
    size_t linked_cubin_bytes,
    const char* const* symbol_names,
    size_t num_symbols,
    void* handle_memory,
    size_t handle_memory_size,
    CubinFunctionPatchHandle** handle_out
);

/**
 * Destroy a handle created by cubin_function_patch_create.
 *
 * This currently performs no deallocation because handle memory is caller-owned.
 * It exists to keep the API symmetric if future handle cleanup is needed.
 */
CUBIN_FUNCTION_PATCH_PUBLIC_DEC void cubin_function_patch_destroy(CubinFunctionPatchHandle* handle);

/**
 * Return the number of patchable function sites stored in the handle.
 */
CUBIN_FUNCTION_PATCH_PUBLIC_DEC size_t cubin_function_patch_num_sites(const CubinFunctionPatchHandle* handle);

/**
 * Return the symbol name for a patch site by index, or NULL for an invalid index.
 *
 * The returned pointer is owned by the handle and remains valid for the handle
 * lifetime.
 */
CUBIN_FUNCTION_PATCH_PUBLIC_DEC const char* cubin_function_patch_site_symbol(const CubinFunctionPatchHandle* handle, size_t site_index);

/**
 * Return the reserved linked function-body size for a patch site.
 *
 * This is useful for checking whether a worst-case template is large enough for
 * expected replacement functions.
 */
CUBIN_FUNCTION_PATCH_PUBLIC_DEC CubinFunctionPatchResult
cubin_function_patch_site_reserved_size(
    const CubinFunctionPatchHandle* handle,
    size_t site_index,
    size_t* bytes_out
);

/**
 * Return the number of bytes required for an output cubin buffer.
 *
 * The output cubin has the same byte size as the linked template cubin.
 */
CUBIN_FUNCTION_PATCH_PUBLIC_DEC CubinFunctionPatchResult
cubin_function_patch_output_size(
    const CubinFunctionPatchHandle* handle,
    size_t* bytes_out
);

/**
 * Copy the linked template cubin into an output cubin buffer.
 *
 * Use this when patching multiple independent replacement RDC objects into the
 * same output cubin with the in-place patch calls. For the common case where
 * one replacement RDC provides all functions, use cubin_function_patch_apply_all.
 */
CUBIN_FUNCTION_PATCH_PUBLIC_DEC CubinFunctionPatchResult
cubin_function_patch_begin(
    const CubinFunctionPatchHandle* handle,
    void* output_cubin,
    size_t output_cubin_capacity,
    size_t* output_cubin_bytes_out
);

/**
 * Patch one function in an existing output cubin buffer.
 *
 * output_cubin should usually be initialized by cubin_function_patch_begin.
 * replacement_rdc_elf must be nvPTXCompiler --compile-only output containing a
 * same-named self-contained STT_FUNC body. The replacement is rejected if it is
 * larger than the reserved body, targets a different ELF machine, or has a
 * relocation targeting the replacement function text.
 *
 * If the replacement is smaller than the reserved body, bytes after copied
 * replacement representations are left as they were in the template. The
 * reserved template must therefore use safe tail padding and the replacement
 * must not branch into that tail. Use cubin_function_patch_apply_one_in_place_ex with
 * CUBIN_FUNCTION_PATCH_TAIL_REQUIRE_EXACT_SIZE to reject smaller replacements.
 */
CUBIN_FUNCTION_PATCH_PUBLIC_DEC CubinFunctionPatchResult
cubin_function_patch_apply_one_in_place(
    const CubinFunctionPatchHandle* handle,
    void* output_cubin,
    size_t output_cubin_bytes,
    const char* symbol_name,
    const void* replacement_rdc_elf,
    size_t replacement_rdc_elf_bytes,
    CubinFunctionPatchReport* report_out
);

/**
 * Patch one function in an existing output cubin buffer with an explicit tail policy.
 */
CUBIN_FUNCTION_PATCH_PUBLIC_DEC CubinFunctionPatchResult
cubin_function_patch_apply_one_in_place_ex(
    const CubinFunctionPatchHandle* handle,
    void* output_cubin,
    size_t output_cubin_bytes,
    const char* symbol_name,
    const void* replacement_rdc_elf,
    size_t replacement_rdc_elf_bytes,
    CubinFunctionPatchTailPolicy tail_policy,
    CubinFunctionPatchReport* report_out
);

/**
 * Patch every site in the handle from one replacement RDC ELF.
 *
 * The replacement RDC must contain every symbol in the handle. reports may be
 * NULL. If reports is not NULL, report_count must be at least the handle site
 * count.
 *
 * On failure, patches before the failing site may already have been applied.
 */
CUBIN_FUNCTION_PATCH_PUBLIC_DEC CubinFunctionPatchResult
cubin_function_patch_apply_all_in_place(
    const CubinFunctionPatchHandle* handle,
    void* output_cubin,
    size_t output_cubin_bytes,
    const void* replacement_rdc_elf,
    size_t replacement_rdc_elf_bytes,
    CubinFunctionPatchReport* reports,
    size_t report_count
);

/**
 * Patch every site in the handle from one replacement RDC ELF with an explicit tail policy.
 */
CUBIN_FUNCTION_PATCH_PUBLIC_DEC CubinFunctionPatchResult
cubin_function_patch_apply_all_in_place_ex(
    const CubinFunctionPatchHandle* handle,
    void* output_cubin,
    size_t output_cubin_bytes,
    const void* replacement_rdc_elf,
    size_t replacement_rdc_elf_bytes,
    CubinFunctionPatchTailPolicy tail_policy,
    CubinFunctionPatchReport* reports,
    size_t report_count
);

/**
 * Copy the template cubin and patch one function into the output buffer.
 *
 * This is the one-shot equivalent of cubin_function_patch_begin followed by
 * cubin_function_patch_apply_one_in_place.
 *
 * The output buffer receives a copy of the template before patching. If patching
 * fails, the buffer may contain an unpatched template cubin.
 */
CUBIN_FUNCTION_PATCH_PUBLIC_DEC CubinFunctionPatchResult
cubin_function_patch_apply_one(
    const CubinFunctionPatchHandle* handle,
    const char* symbol_name,
    const void* replacement_rdc_elf,
    size_t replacement_rdc_elf_bytes,
    void* output_cubin,
    size_t output_cubin_capacity,
    size_t* output_cubin_bytes_out,
    CubinFunctionPatchReport* report_out
);

/**
 * Copy the template cubin and patch one function with an explicit tail policy.
 */
CUBIN_FUNCTION_PATCH_PUBLIC_DEC CubinFunctionPatchResult
cubin_function_patch_apply_one_ex(
    const CubinFunctionPatchHandle* handle,
    const char* symbol_name,
    const void* replacement_rdc_elf,
    size_t replacement_rdc_elf_bytes,
    CubinFunctionPatchTailPolicy tail_policy,
    void* output_cubin,
    size_t output_cubin_capacity,
    size_t* output_cubin_bytes_out,
    CubinFunctionPatchReport* report_out
);

/**
 * Copy the template cubin and patch every site from one replacement RDC ELF.
 *
 * This is the simplest hot-path call when one replacement object contains every
 * function body that should change.
 *
 * The output buffer receives a copy of the template before patching. If patching
 * fails, the buffer may contain a partially patched cubin.
 */
CUBIN_FUNCTION_PATCH_PUBLIC_DEC CubinFunctionPatchResult
cubin_function_patch_apply_all(
    const CubinFunctionPatchHandle* handle,
    const void* replacement_rdc_elf,
    size_t replacement_rdc_elf_bytes,
    void* output_cubin,
    size_t output_cubin_capacity,
    size_t* output_cubin_bytes_out,
    CubinFunctionPatchReport* reports,
    size_t report_count
);

/**
 * Copy the template cubin and patch every site with an explicit tail policy.
 */
CUBIN_FUNCTION_PATCH_PUBLIC_DEC CubinFunctionPatchResult
cubin_function_patch_apply_all_ex(
    const CubinFunctionPatchHandle* handle,
    const void* replacement_rdc_elf,
    size_t replacement_rdc_elf_bytes,
    CubinFunctionPatchTailPolicy tail_policy,
    void* output_cubin,
    size_t output_cubin_capacity,
    size_t* output_cubin_bytes_out,
    CubinFunctionPatchReport* reports,
    size_t report_count
);

#endif /* CUBIN_FUNCTION_PATCH_H_INCLUDE */

#ifdef CUBIN_FUNCTION_PATCH_IMPLEMENTATION
#ifndef CUBIN_FUNCTION_PATCH_IMPLEMENTATION_ONCE
#define CUBIN_FUNCTION_PATCH_IMPLEMENTATION_ONCE

#include <elf.h>
#include <string.h>

typedef struct {
    size_t linked_offset;
    size_t reserved_size;
    size_t section_index;
    uint64_t section_offset;
    int kind;
} CubinFunctionPatchSideSection;

enum {
    CUBIN_FUNCTION_PATCH_SIDE_SECTION_INFO = 1,
    CUBIN_FUNCTION_PATCH_SIDE_SECTION_CONSTANT2 = 2,
    CUBIN_FUNCTION_PATCH_MAX_SIDE_SECTIONS = 4
};

typedef struct {
    char* symbol_name;
    size_t symbol_index;
    size_t symbol_table_index;
    size_t linked_offset;
    size_t reserved_size;
    size_t section_index;
    uint64_t section_offset;
    size_t capmerc_linked_offset;
    size_t capmerc_reserved_size;
    size_t capmerc_section_index;
    uint64_t capmerc_section_offset;
    size_t num_side_sections;
    CubinFunctionPatchSideSection side_sections[CUBIN_FUNCTION_PATCH_MAX_SIDE_SECTIONS];
    int has_capmerc;
} CubinFunctionPatchSite;

struct CubinFunctionPatchHandle {
    const unsigned char* linked_cubin;
    size_t linked_cubin_bytes;
    uint16_t elf_machine;
    uint32_t elf_flags;
    uint32_t sm_version;
    size_t num_sites;
    CubinFunctionPatchSite sites[1];
};

typedef struct {
    size_t file_offset;
    size_t size;
    size_t section_index;
    uint64_t section_offset;
} CubinFunctionPatchSpan;

#ifndef CUBIN_FUNCTION_PATCH_SHT_NV_CAPMERC_TEXT
#define CUBIN_FUNCTION_PATCH_SHT_NV_CAPMERC_TEXT ((Elf64_Word)(SHT_LOPROC + 0x16u))
#endif
#ifndef CUBIN_FUNCTION_PATCH_SHT_NV_MERC_RELA
#define CUBIN_FUNCTION_PATCH_SHT_NV_MERC_RELA ((Elf64_Word)(SHT_LOPROC + 0x82u))
#endif
#ifndef CUBIN_FUNCTION_PATCH_SHT_CUDA_INFO
#define CUBIN_FUNCTION_PATCH_SHT_CUDA_INFO ((Elf64_Word)(SHT_LOPROC + 0x0u))
#endif
#ifndef CUBIN_FUNCTION_PATCH_SHT_CUDA_CONSTANT_B2
#define CUBIN_FUNCTION_PATCH_SHT_CUDA_CONSTANT_B2 ((Elf64_Word)(SHT_LOPROC + 0x66u))
#endif
#ifndef CUBIN_FUNCTION_PATCH_R_CUDA_CONST_BANK
#define CUBIN_FUNCTION_PATCH_R_CUDA_CONST_BANK 0x42u
#endif

#define CUBIN_FUNCTION_PATCH_EIFMT_SVAL 0x04u
#define CUBIN_FUNCTION_PATCH_EIATTR_FRAME_SIZE 0x11u
#define CUBIN_FUNCTION_PATCH_EIATTR_MIN_STACK_SIZE 0x12u
#define CUBIN_FUNCTION_PATCH_EIATTR_MAX_STACK_SIZE 0x23u
#define CUBIN_FUNCTION_PATCH_EIATTR_REGCOUNT 0x2fu

static size_t
cubin_function_patch_strnlen(
    const char* s,
    size_t max_len
) {
    size_t i;
    if (s == NULL) return 0u;
    for (i = 0u; i < max_len; ++i) {
        if (s[i] == '\0') return i;
    }
    return max_len;
}

static size_t
cubin_function_patch_strlen(
    const char* s
) {
    size_t n = 0u;
    if (s == NULL) return 0u;
    while (s[n] != '\0') n++;
    return n;
}

static CubinFunctionPatchResult
cubin_function_patch_align_up_size(
    size_t value,
    size_t align,
    size_t* out
) {
    size_t mask;
    size_t aligned;
    if (out == NULL || align == 0u || (align & (align - 1u)) != 0u) {
        return CUBIN_FUNCTION_PATCH_ERROR_INVALID_VALUE;
    }
    mask = align - 1u;
    if (value > (size_t)-1 - mask) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_VALUE;
    aligned = (value + mask) & ~mask;
    if (aligned < value) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_VALUE;
    *out = aligned;
    return CUBIN_FUNCTION_PATCH_SUCCESS;
}

static CubinFunctionPatchResult
cubin_function_patch_add_size(
    size_t a,
    size_t b,
    size_t* out
) {
    if (out == NULL) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_VALUE;
    if (b > (size_t)-1 - a) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_VALUE;
    *out = a + b;
    return CUBIN_FUNCTION_PATCH_SUCCESS;
}

static CubinFunctionPatchResult
cubin_function_patch_mul_size(
    size_t a,
    size_t b,
    size_t* out
) {
    if (out == NULL) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_VALUE;
    if (a != 0u && b > (size_t)-1 / a) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_VALUE;
    *out = a * b;
    return CUBIN_FUNCTION_PATCH_SUCCESS;
}

static CubinFunctionPatchResult
cubin_function_patch_u64_to_size(
    uint64_t value,
    size_t* out
) {
    if (out == NULL) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_VALUE;
    if ((uint64_t)(size_t)value != value) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
    *out = (size_t)value;
    return CUBIN_FUNCTION_PATCH_SUCCESS;
}

static int
cubin_function_patch_range_ok(
    size_t offset,
    size_t bytes,
    size_t total
) {
    if (offset > total) return 0;
    if (bytes > total - offset) return 0;
    return 1;
}

static CubinFunctionPatchResult
cubin_function_patch_check_symbol_names(
    const char* const* symbol_names,
    size_t num_symbols
) {
    size_t i;
    size_t j;
    if (symbol_names == NULL || num_symbols == 0u) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_VALUE;
    for (i = 0u; i < num_symbols; ++i) {
        if (symbol_names[i] == NULL || symbol_names[i][0] == '\0') {
            return CUBIN_FUNCTION_PATCH_ERROR_INVALID_VALUE;
        }
        for (j = 0u; j < i; ++j) {
            if (strcmp(symbol_names[i], symbol_names[j]) == 0) {
                return CUBIN_FUNCTION_PATCH_ERROR_AMBIGUOUS_SYMBOL;
            }
        }
    }
    return CUBIN_FUNCTION_PATCH_SUCCESS;
}

static int
cubin_function_patch_read(
    const void* data,
    size_t bytes,
    size_t offset,
    void* out,
    size_t out_bytes
) {
    if (data == NULL || out == NULL) return 0;
    if (!cubin_function_patch_range_ok(offset, out_bytes, bytes)) return 0;
    memcpy(out, (const unsigned char*)data + offset, out_bytes);
    return 1;
}

static uint16_t
cubin_function_patch_read_u16_unaligned(
    const unsigned char* p
) {
    uint16_t value;
    memcpy(&value, p, sizeof(value));
    return value;
}

static uint32_t
cubin_function_patch_read_u32_unaligned(
    const unsigned char* p
) {
    uint32_t value;
    memcpy(&value, p, sizeof(value));
    return value;
}

static void
cubin_function_patch_write_u32_unaligned(
    unsigned char* p,
    uint32_t value
) {
    memcpy(p, &value, sizeof(value));
}

static uint32_t
cubin_function_patch_sm_from_elf_flags(
    uint32_t flags
) {
    uint32_t sm = (flags >> 8u) & 0xffu;
    if (sm >= 30u) return sm;
    sm = flags & 0xffu;
    if (sm >= 30u) return sm;
    sm = (flags >> 16u) & 0xffu;
    if (sm >= 30u) return sm;
    return 0u;
}

static CubinFunctionPatchResult
cubin_function_patch_read_ehdr(
    const void* elf_data,
    size_t elf_bytes,
    Elf64_Ehdr* eh_out
) {
    Elf64_Ehdr eh;
    size_t section_offset;
    size_t section_bytes;
    CubinFunctionPatchResult result;
    if (!cubin_function_patch_read(elf_data, elf_bytes, 0u, &eh, sizeof(eh))) {
        return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
    }
    if (memcmp(eh.e_ident, ELFMAG, SELFMAG) != 0) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
    if (eh.e_ident[EI_CLASS] != ELFCLASS64) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
    if (eh.e_ident[EI_DATA] != ELFDATA2LSB) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
    if (eh.e_shentsize != sizeof(Elf64_Shdr)) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
    if (eh.e_shnum == 0u) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
    result = cubin_function_patch_u64_to_size(eh.e_shoff, &section_offset);
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
    result = cubin_function_patch_mul_size((size_t)eh.e_shnum, sizeof(Elf64_Shdr), &section_bytes);
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
    if (!cubin_function_patch_range_ok(section_offset, section_bytes, elf_bytes)) {
        return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
    }
    *eh_out = eh;
    return CUBIN_FUNCTION_PATCH_SUCCESS;
}

static CubinFunctionPatchResult
cubin_function_patch_read_shdr(
    const void* elf_data,
    size_t elf_bytes,
    const Elf64_Ehdr* eh,
    size_t section_index,
    Elf64_Shdr* sh_out
) {
    size_t offset;
    size_t section_table_offset;
    size_t section_delta;
    CubinFunctionPatchResult result;
    if (eh == NULL || sh_out == NULL || section_index >= eh->e_shnum) {
        return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
    }
    result = cubin_function_patch_u64_to_size(eh->e_shoff, &section_table_offset);
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
    result = cubin_function_patch_mul_size(section_index, sizeof(Elf64_Shdr), &section_delta);
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
    result = cubin_function_patch_add_size(section_table_offset, section_delta, &offset);
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
    if (!cubin_function_patch_read(elf_data, elf_bytes, offset, sh_out, sizeof(*sh_out))) {
        return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
    }
    return CUBIN_FUNCTION_PATCH_SUCCESS;
}

static CubinFunctionPatchResult
cubin_function_patch_read_sym(
    const void* elf_data,
    size_t elf_bytes,
    const Elf64_Shdr* symtab,
    size_t symbol_index,
    Elf64_Sym* sym_out
) {
    size_t offset;
    size_t table_offset;
    size_t symbol_delta;
    CubinFunctionPatchResult result;
    if (symtab == NULL || sym_out == NULL || symtab->sh_entsize != sizeof(Elf64_Sym)) {
        return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
    }
    result = cubin_function_patch_u64_to_size(symtab->sh_offset, &table_offset);
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
    result = cubin_function_patch_mul_size(symbol_index, sizeof(Elf64_Sym), &symbol_delta);
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
    result = cubin_function_patch_add_size(table_offset, symbol_delta, &offset);
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
    if (!cubin_function_patch_read(elf_data, elf_bytes, offset, sym_out, sizeof(*sym_out))) {
        return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
    }
    return CUBIN_FUNCTION_PATCH_SUCCESS;
}

static int
cubin_function_patch_symbol_name_equals(
    const void* elf_data,
    size_t elf_bytes,
    const Elf64_Shdr* strtab,
    Elf64_Word name_offset,
    const char* name
) {
    const char* symbol_name;
    size_t remaining;
    size_t strtab_offset;
    size_t strtab_size;
    CubinFunctionPatchResult result;

    if (strtab == NULL || name == NULL) return 0;
    if (name_offset >= strtab->sh_size) return 0;
    result = cubin_function_patch_u64_to_size(strtab->sh_offset, &strtab_offset);
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return 0;
    result = cubin_function_patch_u64_to_size(strtab->sh_size, &strtab_size);
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return 0;
    if (!cubin_function_patch_range_ok(strtab_offset, strtab_size, elf_bytes)) return 0;
    symbol_name = (const char*)elf_data + strtab_offset + (size_t)name_offset;
    remaining = strtab_size - (size_t)name_offset;
    return cubin_function_patch_strnlen(symbol_name, remaining) < remaining && strcmp(symbol_name, name) == 0;
}

static int
cubin_function_patch_section_name_matches_prefix_symbol(
    const void* elf_data,
    size_t elf_bytes,
    const Elf64_Ehdr* eh,
    size_t section_index,
    const char* prefix,
    const char* symbol_name
) {
    Elf64_Shdr shstrtab;
    Elf64_Shdr section;
    const char* section_name;
    size_t section_name_len;
    size_t prefix_len;
    size_t symbol_len;
    size_t shstrtab_offset;
    size_t shstrtab_size;
    size_t remaining;
    CubinFunctionPatchResult result;

    if (elf_data == NULL || eh == NULL || prefix == NULL || symbol_name == NULL) return 0;
    if (eh->e_shstrndx == SHN_UNDEF || eh->e_shstrndx >= eh->e_shnum) return 0;
    result = cubin_function_patch_read_shdr(elf_data, elf_bytes, eh, eh->e_shstrndx, &shstrtab);
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS || shstrtab.sh_type != SHT_STRTAB) return 0;
    result = cubin_function_patch_read_shdr(elf_data, elf_bytes, eh, section_index, &section);
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return 0;
    if (section.sh_name >= shstrtab.sh_size) return 0;
    result = cubin_function_patch_u64_to_size(shstrtab.sh_offset, &shstrtab_offset);
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return 0;
    result = cubin_function_patch_u64_to_size(shstrtab.sh_size, &shstrtab_size);
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return 0;
    if (!cubin_function_patch_range_ok(shstrtab_offset, shstrtab_size, elf_bytes)) return 0;
    section_name = (const char*)elf_data + shstrtab_offset + (size_t)section.sh_name;
    remaining = shstrtab_size - (size_t)section.sh_name;
    section_name_len = cubin_function_patch_strnlen(section_name, remaining);
    if (section_name_len >= remaining) return 0;
    prefix_len = cubin_function_patch_strlen(prefix);
    symbol_len = cubin_function_patch_strlen(symbol_name);
    if (section_name_len != prefix_len + symbol_len) return 0;
    if (memcmp(section_name, prefix, prefix_len) != 0) return 0;
    if (memcmp(section_name + prefix_len, symbol_name, symbol_len) != 0) return 0;
    return 1;
}

static int
cubin_function_patch_section_name_starts_with(
    const void* elf_data,
    size_t elf_bytes,
    const Elf64_Ehdr* eh,
    size_t section_index,
    const char* prefix
) {
    Elf64_Shdr shstrtab;
    Elf64_Shdr section;
    const char* section_name;
    size_t section_name_len;
    size_t prefix_len;
    size_t shstrtab_offset;
    size_t shstrtab_size;
    size_t remaining;
    CubinFunctionPatchResult result;

    if (elf_data == NULL || eh == NULL || prefix == NULL) return 0;
    if (eh->e_shstrndx == SHN_UNDEF || eh->e_shstrndx >= eh->e_shnum) return 0;
    result = cubin_function_patch_read_shdr(elf_data, elf_bytes, eh, eh->e_shstrndx, &shstrtab);
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS || shstrtab.sh_type != SHT_STRTAB) return 0;
    result = cubin_function_patch_read_shdr(elf_data, elf_bytes, eh, section_index, &section);
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return 0;
    if (section.sh_name >= shstrtab.sh_size) return 0;
    result = cubin_function_patch_u64_to_size(shstrtab.sh_offset, &shstrtab_offset);
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return 0;
    result = cubin_function_patch_u64_to_size(shstrtab.sh_size, &shstrtab_size);
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return 0;
    if (!cubin_function_patch_range_ok(shstrtab_offset, shstrtab_size, elf_bytes)) return 0;
    section_name = (const char*)elf_data + shstrtab_offset + (size_t)section.sh_name;
    remaining = shstrtab_size - (size_t)section.sh_name;
    section_name_len = cubin_function_patch_strnlen(section_name, remaining);
    if (section_name_len >= remaining) return 0;
    prefix_len = cubin_function_patch_strlen(prefix);
    if (section_name_len < prefix_len) return 0;
    return memcmp(section_name, prefix, prefix_len) == 0;
}

static int
cubin_function_patch_section_name_equals(
    const void* elf_data,
    size_t elf_bytes,
    const Elf64_Ehdr* eh,
    size_t section_index,
    const char* expected_name
) {
    Elf64_Shdr shstrtab;
    Elf64_Shdr section;
    const char* section_name;
    size_t section_name_len;
    size_t expected_len;
    size_t shstrtab_offset;
    size_t shstrtab_size;
    size_t remaining;
    CubinFunctionPatchResult result;

    if (elf_data == NULL || eh == NULL || expected_name == NULL) return 0;
    if (eh->e_shstrndx == SHN_UNDEF || eh->e_shstrndx >= eh->e_shnum) return 0;
    result = cubin_function_patch_read_shdr(elf_data, elf_bytes, eh, eh->e_shstrndx, &shstrtab);
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS || shstrtab.sh_type != SHT_STRTAB) return 0;
    result = cubin_function_patch_read_shdr(elf_data, elf_bytes, eh, section_index, &section);
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return 0;
    if (section.sh_name >= shstrtab.sh_size) return 0;
    result = cubin_function_patch_u64_to_size(shstrtab.sh_offset, &shstrtab_offset);
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return 0;
    result = cubin_function_patch_u64_to_size(shstrtab.sh_size, &shstrtab_size);
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return 0;
    if (!cubin_function_patch_range_ok(shstrtab_offset, shstrtab_size, elf_bytes)) return 0;
    section_name = (const char*)elf_data + shstrtab_offset + (size_t)section.sh_name;
    remaining = shstrtab_size - (size_t)section.sh_name;
    section_name_len = cubin_function_patch_strnlen(section_name, remaining);
    if (section_name_len >= remaining) return 0;
    expected_len = cubin_function_patch_strlen(expected_name);
    if (section_name_len != expected_len) return 0;
    return memcmp(section_name, expected_name, expected_len) == 0;
}

static int
cubin_function_patch_parse_decimal_suffix(
    const char* text,
    size_t* value_out
) {
    const char* p;
    const char* start;
    size_t value = 0u;

    if (text == NULL || text[0] == '\0' || value_out == NULL) return 0;
    p = text + cubin_function_patch_strlen(text);
    start = p;
    while (start != text && start[-1] >= '0' && start[-1] <= '9') {
        --start;
    }
    if (start == p) return 0;
    while (start != p) {
        size_t digit = (size_t)(*start - '0');
        if (value > ((size_t)-1 - digit) / 10u) return 0;
        value = value * 10u + digit;
        ++start;
    }
    *value_out = value;
    return 1;
}

static int
cubin_function_patch_section_name_has_decimal_suffix(
    const void* elf_data,
    size_t elf_bytes,
    const Elf64_Ehdr* eh,
    size_t section_index,
    const char* prefix,
    size_t requested_suffix
) {
    Elf64_Shdr shstrtab;
    Elf64_Shdr section;
    const char* section_name;
    size_t section_name_len;
    size_t prefix_len;
    size_t suffix = 0u;
    size_t shstrtab_offset;
    size_t shstrtab_size;
    size_t remaining;
    CubinFunctionPatchResult result;

    if (elf_data == NULL || eh == NULL || prefix == NULL) return 0;
    if (eh->e_shstrndx == SHN_UNDEF || eh->e_shstrndx >= eh->e_shnum) return 0;
    result = cubin_function_patch_read_shdr(elf_data, elf_bytes, eh, eh->e_shstrndx, &shstrtab);
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS || shstrtab.sh_type != SHT_STRTAB) return 0;
    result = cubin_function_patch_read_shdr(elf_data, elf_bytes, eh, section_index, &section);
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return 0;
    if (section.sh_name >= shstrtab.sh_size) return 0;
    result = cubin_function_patch_u64_to_size(shstrtab.sh_offset, &shstrtab_offset);
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return 0;
    result = cubin_function_patch_u64_to_size(shstrtab.sh_size, &shstrtab_size);
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return 0;
    if (!cubin_function_patch_range_ok(shstrtab_offset, shstrtab_size, elf_bytes)) return 0;
    section_name = (const char*)elf_data + shstrtab_offset + (size_t)section.sh_name;
    remaining = shstrtab_size - (size_t)section.sh_name;
    section_name_len = cubin_function_patch_strnlen(section_name, remaining);
    if (section_name_len >= remaining) return 0;
    prefix_len = cubin_function_patch_strlen(prefix);
    if (section_name_len < prefix_len) return 0;
    if (memcmp(section_name, prefix, prefix_len) != 0) return 0;
    if (!cubin_function_patch_parse_decimal_suffix(section_name, &suffix)) return 0;
    return suffix == requested_suffix;
}

static CubinFunctionPatchResult
cubin_function_patch_find_capmerc_span(
    const void* elf_data,
    size_t elf_bytes,
    const char* symbol_name,
    CubinFunctionPatchSpan* span_out
) {
    static const char prefix[] = ".nv.capmerc.text.";
    Elf64_Ehdr eh;
    CubinFunctionPatchResult result;
    size_t i;
    size_t match_count = 0u;
    CubinFunctionPatchSpan matched_span;

    if (symbol_name == NULL || symbol_name[0] == '\0' || span_out == NULL) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_VALUE;
    memset(span_out, 0, sizeof(*span_out));
    memset(&matched_span, 0, sizeof(matched_span));
    result = cubin_function_patch_read_ehdr(elf_data, elf_bytes, &eh);
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;

    for (i = 0u; i < eh.e_shnum; ++i) {
        Elf64_Shdr section;
        size_t section_offset;
        size_t section_size;

        result = cubin_function_patch_read_shdr(elf_data, elf_bytes, &eh, i, &section);
        if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
        if (section.sh_type != CUBIN_FUNCTION_PATCH_SHT_NV_CAPMERC_TEXT) continue;
        if (!cubin_function_patch_section_name_matches_prefix_symbol(elf_data, elf_bytes, &eh, i, prefix, symbol_name)) continue;
        result = cubin_function_patch_u64_to_size(section.sh_offset, &section_offset);
        if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
        result = cubin_function_patch_u64_to_size(section.sh_size, &section_size);
        if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
        if (!cubin_function_patch_range_ok(section_offset, section_size, elf_bytes)) {
            return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
        }
        matched_span.file_offset = section_offset;
        matched_span.size = section_size;
        matched_span.section_index = i;
        matched_span.section_offset = 0u;
        match_count++;
    }

    if (match_count == 0u) return CUBIN_FUNCTION_PATCH_ERROR_SYMBOL_NOT_FOUND;
    if (match_count != 1u) return CUBIN_FUNCTION_PATCH_ERROR_AMBIGUOUS_SYMBOL;
    *span_out = matched_span;
    return CUBIN_FUNCTION_PATCH_SUCCESS;
}

static const char*
cubin_function_patch_side_section_prefix(
    int kind
) {
    switch (kind) {
        case CUBIN_FUNCTION_PATCH_SIDE_SECTION_INFO:
            return ".nv.info.";
        case CUBIN_FUNCTION_PATCH_SIDE_SECTION_CONSTANT2:
            return ".nv.constant2.";
    }
    return NULL;
}

static Elf64_Word
cubin_function_patch_side_section_type(
    int kind
) {
    switch (kind) {
        case CUBIN_FUNCTION_PATCH_SIDE_SECTION_INFO:
            return CUBIN_FUNCTION_PATCH_SHT_CUDA_INFO;
        case CUBIN_FUNCTION_PATCH_SIDE_SECTION_CONSTANT2:
            return CUBIN_FUNCTION_PATCH_SHT_CUDA_CONSTANT_B2;
    }
    return SHT_NULL;
}

static int
cubin_function_patch_section_type_matches_side_kind(
    Elf64_Word section_type,
    int kind
) {
    switch (kind) {
        case CUBIN_FUNCTION_PATCH_SIDE_SECTION_INFO:
            return section_type == CUBIN_FUNCTION_PATCH_SHT_CUDA_INFO;
        case CUBIN_FUNCTION_PATCH_SIDE_SECTION_CONSTANT2:
            return section_type == CUBIN_FUNCTION_PATCH_SHT_CUDA_CONSTANT_B2 ||
                section_type == SHT_PROGBITS;
    }
    return 0;
}

static int
cubin_function_patch_site_has_side_kind(
    const CubinFunctionPatchSite* site,
    int kind
) {
    size_t i;
    if (site == NULL) return 0;
    for (i = 0u; i < site->num_side_sections; ++i) {
        if (site->side_sections[i].kind == kind) return 1;
    }
    return 0;
}

static CubinFunctionPatchResult
cubin_function_patch_find_side_section_span(
    const void* elf_data,
    size_t elf_bytes,
    const char* symbol_name,
    int kind,
    CubinFunctionPatchSpan* span_out
) {
    Elf64_Ehdr eh;
    CubinFunctionPatchResult result;
    const char* prefix = cubin_function_patch_side_section_prefix(kind);
    Elf64_Word section_type = cubin_function_patch_side_section_type(kind);
    size_t i;
    size_t match_count = 0u;
    CubinFunctionPatchSpan matched_span;

    if (symbol_name == NULL || symbol_name[0] == '\0' || span_out == NULL ||
        prefix == NULL || section_type == SHT_NULL) {
        return CUBIN_FUNCTION_PATCH_ERROR_INVALID_VALUE;
    }
    memset(span_out, 0, sizeof(*span_out));
    memset(&matched_span, 0, sizeof(matched_span));
    result = cubin_function_patch_read_ehdr(elf_data, elf_bytes, &eh);
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;

    for (i = 0u; i < eh.e_shnum; ++i) {
        Elf64_Shdr section;
        size_t section_offset;
        size_t section_size;

        result = cubin_function_patch_read_shdr(elf_data, elf_bytes, &eh, i, &section);
        if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
        (void)section_type;
        if (!cubin_function_patch_section_type_matches_side_kind(section.sh_type, kind)) continue;
        if (!cubin_function_patch_section_name_matches_prefix_symbol(elf_data, elf_bytes, &eh, i, prefix, symbol_name)) continue;
        result = cubin_function_patch_u64_to_size(section.sh_offset, &section_offset);
        if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
        result = cubin_function_patch_u64_to_size(section.sh_size, &section_size);
        if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
        if (!cubin_function_patch_range_ok(section_offset, section_size, elf_bytes)) {
            return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
        }
        matched_span.file_offset = section_offset;
        matched_span.size = section_size;
        matched_span.section_index = i;
        matched_span.section_offset = 0u;
        match_count++;
    }

    if (match_count == 0u) return CUBIN_FUNCTION_PATCH_ERROR_SYMBOL_NOT_FOUND;
    if (match_count != 1u) return CUBIN_FUNCTION_PATCH_ERROR_AMBIGUOUS_SYMBOL;
    *span_out = matched_span;
    return CUBIN_FUNCTION_PATCH_SUCCESS;
}

static CubinFunctionPatchResult
cubin_function_patch_add_side_section(
    CubinFunctionPatchSite* site,
    const void* elf_data,
    size_t elf_bytes,
    const char* symbol_name,
    int kind
) {
    CubinFunctionPatchSpan span;
    CubinFunctionPatchResult result;
    CubinFunctionPatchSideSection* side;
    if (site == NULL) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_VALUE;
    result = cubin_function_patch_find_side_section_span(elf_data, elf_bytes, symbol_name, kind, &span);
    if (result == CUBIN_FUNCTION_PATCH_ERROR_SYMBOL_NOT_FOUND) return CUBIN_FUNCTION_PATCH_SUCCESS;
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
    if (site->num_side_sections >= CUBIN_FUNCTION_PATCH_MAX_SIDE_SECTIONS) {
        return CUBIN_FUNCTION_PATCH_ERROR_WORKSPACE_EXHAUSTED;
    }
    side = &site->side_sections[site->num_side_sections++];
    side->linked_offset = span.file_offset;
    side->reserved_size = span.size;
    side->section_index = span.section_index;
    side->section_offset = span.section_offset;
    side->kind = kind;
    return CUBIN_FUNCTION_PATCH_SUCCESS;
}

static CubinFunctionPatchResult
cubin_function_patch_find_suffix_side_section_by_prefix(
    const void* elf_data,
    size_t elf_bytes,
    int kind,
    size_t requested_suffix,
    CubinFunctionPatchSpan* span_out,
    size_t* total_matches_out
) {
    Elf64_Ehdr eh;
    CubinFunctionPatchResult result;
    const char* prefix = cubin_function_patch_side_section_prefix(kind);
    size_t i;
    size_t match_count = 0u;
    size_t suffix_match_count = 0u;
    CubinFunctionPatchSpan first_span;
    CubinFunctionPatchSpan matched_span;

    if (span_out == NULL || total_matches_out == NULL || prefix == NULL) {
        return CUBIN_FUNCTION_PATCH_ERROR_INVALID_VALUE;
    }
    memset(span_out, 0, sizeof(*span_out));
    memset(&first_span, 0, sizeof(first_span));
    memset(&matched_span, 0, sizeof(matched_span));
    *total_matches_out = 0u;
    result = cubin_function_patch_read_ehdr(elf_data, elf_bytes, &eh);
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;

    for (i = 0u; i < eh.e_shnum; ++i) {
        Elf64_Shdr section;
        size_t section_offset;
        size_t section_size;

        result = cubin_function_patch_read_shdr(elf_data, elf_bytes, &eh, i, &section);
        if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
        if (!cubin_function_patch_section_type_matches_side_kind(section.sh_type, kind)) continue;
        if (!cubin_function_patch_section_name_starts_with(elf_data, elf_bytes, &eh, i, prefix)) continue;
        result = cubin_function_patch_u64_to_size(section.sh_offset, &section_offset);
        if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
        result = cubin_function_patch_u64_to_size(section.sh_size, &section_size);
        if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
        if (!cubin_function_patch_range_ok(section_offset, section_size, elf_bytes)) {
            return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
        }
        if (match_count == 0u) {
            first_span.file_offset = section_offset;
            first_span.size = section_size;
            first_span.section_index = i;
            first_span.section_offset = 0u;
        }
        match_count++;
        if (!cubin_function_patch_section_name_has_decimal_suffix(
                elf_data,
                elf_bytes,
                &eh,
                i,
                prefix,
                requested_suffix)) {
            continue;
        }
        matched_span.file_offset = section_offset;
        matched_span.size = section_size;
        matched_span.section_index = i;
        matched_span.section_offset = 0u;
        suffix_match_count++;
    }

    *total_matches_out = match_count;
    if (suffix_match_count == 0u) {
        if (match_count == 1u && requested_suffix == 0u) {
            *span_out = first_span;
            return CUBIN_FUNCTION_PATCH_SUCCESS;
        }
        return CUBIN_FUNCTION_PATCH_ERROR_SYMBOL_NOT_FOUND;
    }
    if (suffix_match_count != 1u) return CUBIN_FUNCTION_PATCH_ERROR_AMBIGUOUS_SYMBOL;
    *span_out = matched_span;
    return CUBIN_FUNCTION_PATCH_SUCCESS;
}

static CubinFunctionPatchResult
cubin_function_patch_add_fallback_side_section(
    CubinFunctionPatchSite* site,
    const void* elf_data,
    size_t elf_bytes,
    int kind,
    size_t requested_idx,
    size_t expected_total
) {
    CubinFunctionPatchSpan span;
    CubinFunctionPatchResult result;
    CubinFunctionPatchSideSection* side;
    size_t total_matches = 0u;
    if (site == NULL || expected_total == 0u) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_VALUE;
    if (cubin_function_patch_site_has_side_kind(site, kind)) return CUBIN_FUNCTION_PATCH_SUCCESS;
    result = cubin_function_patch_find_suffix_side_section_by_prefix(
        elf_data,
        elf_bytes,
        kind,
        requested_idx,
        &span,
        &total_matches
    );
    if (result == CUBIN_FUNCTION_PATCH_ERROR_SYMBOL_NOT_FOUND) return CUBIN_FUNCTION_PATCH_SUCCESS;
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
    if (total_matches != expected_total) return CUBIN_FUNCTION_PATCH_SUCCESS;
    if (site->num_side_sections >= CUBIN_FUNCTION_PATCH_MAX_SIDE_SECTIONS) {
        return CUBIN_FUNCTION_PATCH_ERROR_WORKSPACE_EXHAUSTED;
    }
    side = &site->side_sections[site->num_side_sections++];
    side->linked_offset = span.file_offset;
    side->reserved_size = span.size;
    side->section_index = span.section_index;
    side->section_offset = span.section_offset;
    side->kind = kind;
    return CUBIN_FUNCTION_PATCH_SUCCESS;
}

static CubinFunctionPatchResult
cubin_function_patch_find_function_span(
    const void* elf_data,
    size_t elf_bytes,
    const char* symbol_name,
    CubinFunctionPatchSpan* span_out,
    size_t* symbol_index_out,
    size_t* symbol_table_index_out,
    uint16_t* machine_out
) {
    Elf64_Ehdr eh;
    CubinFunctionPatchResult result;
    size_t i;
    size_t match_count = 0u;
    size_t matched_symbol_index = 0u;
    size_t matched_symbol_table_index = 0u;
    CubinFunctionPatchSpan matched_span;

    if (symbol_name == NULL || span_out == NULL) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_VALUE;
    memset(span_out, 0, sizeof(*span_out));
    memset(&matched_span, 0, sizeof(matched_span));
    if (symbol_index_out != NULL) *symbol_index_out = 0u;
    if (symbol_table_index_out != NULL) *symbol_table_index_out = 0u;
    result = cubin_function_patch_read_ehdr(elf_data, elf_bytes, &eh);
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
    if (machine_out != NULL) *machine_out = eh.e_machine;

    for (i = 0u; i < eh.e_shnum; ++i) {
        Elf64_Shdr symtab;
        Elf64_Shdr strtab;
        size_t symtab_offset;
        size_t symtab_size;
        size_t num_symbols;
        size_t sym_idx;

        result = cubin_function_patch_read_shdr(elf_data, elf_bytes, &eh, i, &symtab);
        if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
        if (symtab.sh_type != SHT_SYMTAB && symtab.sh_type != SHT_DYNSYM) continue;
        if (symtab.sh_entsize != sizeof(Elf64_Sym)) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
        if (symtab.sh_size % symtab.sh_entsize != 0u) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
        if (symtab.sh_link >= eh.e_shnum) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
        result = cubin_function_patch_u64_to_size(symtab.sh_offset, &symtab_offset);
        if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
        result = cubin_function_patch_u64_to_size(symtab.sh_size, &symtab_size);
        if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
        if (!cubin_function_patch_range_ok(symtab_offset, symtab_size, elf_bytes)) {
            return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
        }
        result = cubin_function_patch_read_shdr(elf_data, elf_bytes, &eh, symtab.sh_link, &strtab);
        if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
        if (strtab.sh_type != SHT_STRTAB) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
        num_symbols = symtab_size / sizeof(Elf64_Sym);

        for (sym_idx = 0u; sym_idx < num_symbols; ++sym_idx) {
            Elf64_Sym sym;
            Elf64_Shdr sec;
            uint64_t in_section_offset;
            size_t sec_offset;
            size_t sec_size;
            size_t in_section_offset_size;
            size_t sym_size;
            size_t file_offset;

            result = cubin_function_patch_read_sym(elf_data, elf_bytes, &symtab, sym_idx, &sym);
            if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
            if (ELF64_ST_TYPE(sym.st_info) != STT_FUNC) continue;
            if (!cubin_function_patch_symbol_name_equals(elf_data, elf_bytes, &strtab, sym.st_name, symbol_name)) continue;
            if (sym.st_shndx == SHN_UNDEF || sym.st_shndx >= eh.e_shnum) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
            if (sym.st_size == 0u) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
            result = cubin_function_patch_read_shdr(elf_data, elf_bytes, &eh, sym.st_shndx, &sec);
            if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
            if ((sec.sh_flags & SHF_EXECINSTR) == 0u) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
            if (sec.sh_type != SHT_PROGBITS) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
            result = cubin_function_patch_u64_to_size(sec.sh_offset, &sec_offset);
            if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
            result = cubin_function_patch_u64_to_size(sec.sh_size, &sec_size);
            if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
            if (sec.sh_addr != 0u && sym.st_value >= sec.sh_addr) {
                in_section_offset = sym.st_value - sec.sh_addr;
            } else {
                in_section_offset = sym.st_value;
            }
            if (in_section_offset > sec.sh_size) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
            if (sym.st_size > sec.sh_size - in_section_offset) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
            result = cubin_function_patch_u64_to_size(in_section_offset, &in_section_offset_size);
            if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
            result = cubin_function_patch_u64_to_size(sym.st_size, &sym_size);
            if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
            result = cubin_function_patch_add_size(sec_offset, in_section_offset_size, &file_offset);
            if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
            if (!cubin_function_patch_range_ok(file_offset, sym_size, elf_bytes)) {
                return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
            }
            matched_span.file_offset = file_offset;
            matched_span.size = sym_size;
            matched_span.section_index = (size_t)sym.st_shndx;
            matched_span.section_offset = in_section_offset;
            matched_symbol_index = sym_idx;
            matched_symbol_table_index = i;
            match_count++;
        }
    }

    if (match_count == 0u) return CUBIN_FUNCTION_PATCH_ERROR_SYMBOL_NOT_FOUND;
    if (match_count != 1u) return CUBIN_FUNCTION_PATCH_ERROR_AMBIGUOUS_SYMBOL;
    *span_out = matched_span;
    if (symbol_index_out != NULL) *symbol_index_out = matched_symbol_index;
    if (symbol_table_index_out != NULL) *symbol_table_index_out = matched_symbol_table_index;
    return CUBIN_FUNCTION_PATCH_SUCCESS;
}

static int
cubin_function_patch_relocation_hits_span(
    uint64_t relocation_offset,
    const CubinFunctionPatchSpan* span
) {
    uint64_t span_end;
    if (span == NULL) return 0;
    if (relocation_offset < span->section_offset) return 0;
    if ((uint64_t)span->size > UINT64_MAX - span->section_offset) return 0;
    span_end = span->section_offset + (uint64_t)span->size;
    if (relocation_offset >= span_end) return 0;
    return 1;
}

static int
cubin_function_patch_relocation_target_kind(
    const void* elf_data,
    size_t elf_bytes,
    const Elf64_Ehdr* eh,
    size_t target_section_index,
    const char* symbol_name,
    const CubinFunctionPatchSite* site,
    int* kind_out
) {
    if (kind_out != NULL) *kind_out = 0;
    if (elf_data == NULL || eh == NULL || symbol_name == NULL || site == NULL) return 0;
    if (target_section_index >= eh->e_shnum) return 0;
    if (cubin_function_patch_site_has_side_kind(site, CUBIN_FUNCTION_PATCH_SIDE_SECTION_CONSTANT2) &&
        cubin_function_patch_section_name_matches_prefix_symbol(
            elf_data,
            elf_bytes,
            eh,
            target_section_index,
            cubin_function_patch_side_section_prefix(CUBIN_FUNCTION_PATCH_SIDE_SECTION_CONSTANT2),
            symbol_name)) {
        if (kind_out != NULL) *kind_out = CUBIN_FUNCTION_PATCH_SIDE_SECTION_CONSTANT2;
        return 1;
    }
    if (cubin_function_patch_site_has_side_kind(site, CUBIN_FUNCTION_PATCH_SIDE_SECTION_INFO) &&
        cubin_function_patch_section_name_matches_prefix_symbol(
            elf_data,
            elf_bytes,
            eh,
            target_section_index,
            cubin_function_patch_side_section_prefix(CUBIN_FUNCTION_PATCH_SIDE_SECTION_INFO),
            symbol_name)) {
        if (kind_out != NULL) *kind_out = CUBIN_FUNCTION_PATCH_SIDE_SECTION_INFO;
        return 1;
    }
    return 0;
}

static CubinFunctionPatchResult
cubin_function_patch_apply_cuda_const_relocation_42(
    void* output_cubin,
    size_t output_cubin_bytes,
    size_t output_offset,
    uint64_t symbol_value,
    int64_t addend
) {
    uint64_t word;
    int64_t immediate;
    int64_t scaled;
    uint64_t field;
    if (output_cubin == NULL || output_offset > output_cubin_bytes ||
        output_cubin_bytes - output_offset < sizeof(word)) {
        return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
    }
    immediate = (int64_t)symbol_value + addend;
    if ((immediate & 3) != 0) return CUBIN_FUNCTION_PATCH_ERROR_RELOCATION_UNSUPPORTED;
    scaled = immediate >> 2;
    if (scaled < -8192 || scaled > 8191) return CUBIN_FUNCTION_PATCH_ERROR_RELOCATION_UNSUPPORTED;
    memcpy(&word, (unsigned char*)output_cubin + output_offset, sizeof(word));
    /*
     * R_CUDA_CONST_BANK relocations for .nv.constant2 references encode both
     * the signed 14-bit scaled offset and the constant-bank selector in the
     * same 16-bit instruction field. nvPTXCompiler RDC text leaves the bank
     * selector unset; the real linker writes bank 2 as the high two bits.
     */
    field = (0x8000ull | ((uint64_t)scaled & 0x3fffull)) << 40;
    word = (word & ~(0xffffull << 40)) | field;
    memcpy((unsigned char*)output_cubin + output_offset, &word, sizeof(word));
    return CUBIN_FUNCTION_PATCH_SUCCESS;
}

static CubinFunctionPatchResult
cubin_function_patch_span_has_unsupported_relocations(
    const void* elf_data,
    size_t elf_bytes,
    const CubinFunctionPatchSpan* span,
    const char* symbol_name,
    const CubinFunctionPatchSite* site,
    int* has_unsupported_relocations_out
) {
    Elf64_Ehdr eh;
    CubinFunctionPatchResult result;
    size_t i;

    if (span == NULL || symbol_name == NULL || site == NULL || has_unsupported_relocations_out == NULL) {
        return CUBIN_FUNCTION_PATCH_ERROR_INVALID_VALUE;
    }
    *has_unsupported_relocations_out = 0;
    result = cubin_function_patch_read_ehdr(elf_data, elf_bytes, &eh);
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;

    for (i = 0u; i < eh.e_shnum; ++i) {
        Elf64_Shdr relsec;
        size_t relsec_offset;
        size_t relsec_size;
        size_t rel_entsize;
        size_t relocation_count;
        size_t rel_idx;
        Elf64_Shdr symtab;
        size_t symtab_size;

        result = cubin_function_patch_read_shdr(elf_data, elf_bytes, &eh, i, &relsec);
        if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
        if (relsec.sh_type != SHT_RELA && relsec.sh_type != SHT_REL &&
            relsec.sh_type != CUBIN_FUNCTION_PATCH_SHT_NV_MERC_RELA) {
            continue;
        }
        if (relsec.sh_info != span->section_index) continue;
        if (relsec.sh_link >= eh.e_shnum) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
        result = cubin_function_patch_read_shdr(elf_data, elf_bytes, &eh, relsec.sh_link, &symtab);
        if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
        if (symtab.sh_type != SHT_SYMTAB && symtab.sh_type != SHT_DYNSYM) {
            return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
        }
        if (symtab.sh_entsize != sizeof(Elf64_Sym)) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
        if (symtab.sh_size % symtab.sh_entsize != 0u) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
        result = cubin_function_patch_u64_to_size(symtab.sh_size, &symtab_size);
        if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
        if ((relsec.sh_type == SHT_RELA || relsec.sh_type == CUBIN_FUNCTION_PATCH_SHT_NV_MERC_RELA) &&
            relsec.sh_entsize != sizeof(Elf64_Rela)) {
            return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
        }
        if (relsec.sh_type == SHT_REL && relsec.sh_entsize != sizeof(Elf64_Rel)) {
            return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
        }
        if (relsec.sh_size % relsec.sh_entsize != 0u) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
        result = cubin_function_patch_u64_to_size(relsec.sh_offset, &relsec_offset);
        if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
        result = cubin_function_patch_u64_to_size(relsec.sh_size, &relsec_size);
        if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
        result = cubin_function_patch_u64_to_size(relsec.sh_entsize, &rel_entsize);
        if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
        if (!cubin_function_patch_range_ok(relsec_offset, relsec_size, elf_bytes)) {
            return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
        }
        relocation_count = relsec_size / rel_entsize;

        for (rel_idx = 0u; rel_idx < relocation_count; ++rel_idx) {
            uint64_t relocation_offset;
            uint64_t relocation_info;
            size_t relocation_symbol_index;
            size_t rel_delta;
            size_t rel_offset;
            result = cubin_function_patch_mul_size(rel_idx, rel_entsize, &rel_delta);
            if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
            result = cubin_function_patch_add_size(relsec_offset, rel_delta, &rel_offset);
            if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
            if (relsec.sh_type == SHT_RELA || relsec.sh_type == CUBIN_FUNCTION_PATCH_SHT_NV_MERC_RELA) {
                Elf64_Rela rela;
                if (!cubin_function_patch_read(elf_data, elf_bytes, rel_offset, &rela, sizeof(rela))) {
                    return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
                }
                relocation_offset = rela.r_offset;
                relocation_info = rela.r_info;
            } else {
                Elf64_Rel rel;
                if (!cubin_function_patch_read(elf_data, elf_bytes, rel_offset, &rel, sizeof(rel))) {
                    return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
                }
                relocation_offset = rel.r_offset;
                relocation_info = rel.r_info;
            }
            if (cubin_function_patch_relocation_hits_span(relocation_offset, span)) {
                Elf64_Sym sym;
                uint32_t relocation_type;
                int target_kind = 0;
                relocation_symbol_index = (size_t)ELF64_R_SYM(relocation_info);
                if (relocation_symbol_index >= symtab_size / sizeof(Elf64_Sym)) {
                    return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
                }
                result = cubin_function_patch_read_sym(elf_data, elf_bytes, &symtab, relocation_symbol_index, &sym);
                if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
                if (sym.st_shndx == SHN_UNDEF ||
                    !cubin_function_patch_relocation_target_kind(
                        elf_data,
                        elf_bytes,
                        &eh,
                        (size_t)sym.st_shndx,
                        symbol_name,
                        site,
                        &target_kind)) {
                    *has_unsupported_relocations_out = 1;
                    return CUBIN_FUNCTION_PATCH_SUCCESS;
                }
                relocation_type = (uint32_t)ELF64_R_TYPE(relocation_info);
                if (relsec.sh_type != SHT_RELA &&
                    relsec.sh_type != CUBIN_FUNCTION_PATCH_SHT_NV_MERC_RELA) {
                    *has_unsupported_relocations_out = 1;
                    return CUBIN_FUNCTION_PATCH_SUCCESS;
                }
                if (target_kind != CUBIN_FUNCTION_PATCH_SIDE_SECTION_CONSTANT2 ||
                    relocation_type != CUBIN_FUNCTION_PATCH_R_CUDA_CONST_BANK) {
                    *has_unsupported_relocations_out = 1;
                    return CUBIN_FUNCTION_PATCH_SUCCESS;
                }
            }
        }
    }

    return CUBIN_FUNCTION_PATCH_SUCCESS;
}

static CubinFunctionPatchResult
cubin_function_patch_apply_supported_relocations(
    const void* elf_data,
    size_t elf_bytes,
    const CubinFunctionPatchSpan* span,
    const char* symbol_name,
    const CubinFunctionPatchSite* site,
    void* output_cubin,
    size_t output_cubin_bytes,
    size_t output_span_offset
) {
    Elf64_Ehdr eh;
    CubinFunctionPatchResult result;
    size_t i;

    if (span == NULL || symbol_name == NULL || site == NULL || output_cubin == NULL) {
        return CUBIN_FUNCTION_PATCH_ERROR_INVALID_VALUE;
    }
    result = cubin_function_patch_read_ehdr(elf_data, elf_bytes, &eh);
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;

    for (i = 0u; i < eh.e_shnum; ++i) {
        Elf64_Shdr relsec;
        Elf64_Shdr symtab;
        size_t symtab_size;
        size_t relsec_offset;
        size_t relsec_size;
        size_t rel_entsize;
        size_t relocation_count;
        size_t rel_idx;

        result = cubin_function_patch_read_shdr(elf_data, elf_bytes, &eh, i, &relsec);
        if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
        if (relsec.sh_type != SHT_RELA && relsec.sh_type != CUBIN_FUNCTION_PATCH_SHT_NV_MERC_RELA) continue;
        if (relsec.sh_info != span->section_index) continue;
        if (relsec.sh_link >= eh.e_shnum) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
        result = cubin_function_patch_read_shdr(elf_data, elf_bytes, &eh, relsec.sh_link, &symtab);
        if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
        if (symtab.sh_type != SHT_SYMTAB && symtab.sh_type != SHT_DYNSYM) {
            return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
        }
        if (symtab.sh_entsize != sizeof(Elf64_Sym)) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
        if (symtab.sh_size % symtab.sh_entsize != 0u) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
        result = cubin_function_patch_u64_to_size(symtab.sh_size, &symtab_size);
        if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
        if (relsec.sh_entsize != sizeof(Elf64_Rela)) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
        if (relsec.sh_size % relsec.sh_entsize != 0u) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
        result = cubin_function_patch_u64_to_size(relsec.sh_offset, &relsec_offset);
        if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
        result = cubin_function_patch_u64_to_size(relsec.sh_size, &relsec_size);
        if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
        result = cubin_function_patch_u64_to_size(relsec.sh_entsize, &rel_entsize);
        if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
        if (!cubin_function_patch_range_ok(relsec_offset, relsec_size, elf_bytes)) {
            return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
        }
        relocation_count = relsec_size / rel_entsize;
        for (rel_idx = 0u; rel_idx < relocation_count; ++rel_idx) {
            Elf64_Rela rela;
            Elf64_Sym sym;
            size_t rel_delta;
            size_t rel_offset;
            size_t relocation_symbol_index;
            int target_kind = 0;
            uint32_t relocation_type;
            uint64_t relocation_delta_u64;
            size_t relocation_delta;
            size_t output_offset;

            result = cubin_function_patch_mul_size(rel_idx, rel_entsize, &rel_delta);
            if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
            result = cubin_function_patch_add_size(relsec_offset, rel_delta, &rel_offset);
            if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
            if (!cubin_function_patch_read(elf_data, elf_bytes, rel_offset, &rela, sizeof(rela))) {
                return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
            }
            if (!cubin_function_patch_relocation_hits_span(rela.r_offset, span)) continue;
            relocation_symbol_index = (size_t)ELF64_R_SYM(rela.r_info);
            if (relocation_symbol_index >= symtab_size / sizeof(Elf64_Sym)) {
                return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
            }
            result = cubin_function_patch_read_sym(elf_data, elf_bytes, &symtab, relocation_symbol_index, &sym);
            if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
            if (sym.st_shndx == SHN_UNDEF ||
                !cubin_function_patch_relocation_target_kind(
                    elf_data,
                    elf_bytes,
                    &eh,
                    (size_t)sym.st_shndx,
                    symbol_name,
                    site,
                    &target_kind)) {
                return CUBIN_FUNCTION_PATCH_ERROR_RELOCATION_UNSUPPORTED;
            }
            relocation_type = (uint32_t)ELF64_R_TYPE(rela.r_info);
            if (target_kind != CUBIN_FUNCTION_PATCH_SIDE_SECTION_CONSTANT2 ||
                relocation_type != CUBIN_FUNCTION_PATCH_R_CUDA_CONST_BANK) {
                return CUBIN_FUNCTION_PATCH_ERROR_RELOCATION_UNSUPPORTED;
            }
            relocation_delta_u64 = rela.r_offset - span->section_offset;
            result = cubin_function_patch_u64_to_size(relocation_delta_u64, &relocation_delta);
            if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
            result = cubin_function_patch_add_size(output_span_offset, relocation_delta, &output_offset);
            if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
            result = cubin_function_patch_apply_cuda_const_relocation_42(
                output_cubin,
                output_cubin_bytes,
                output_offset,
                sym.st_value,
                rela.r_addend
            );
            if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
        }
    }

    return CUBIN_FUNCTION_PATCH_SUCCESS;
}

typedef struct {
    uint32_t frame_size;
    uint32_t min_stack_size;
    uint32_t max_stack_size;
    uint32_t reg_count;
    int has_frame_size;
    int has_min_stack_size;
    int has_max_stack_size;
    int has_reg_count;
} CubinFunctionPatchGlobalInfoValues;

static int
cubin_function_patch_global_info_attr_patchable(
    unsigned char attr
) {
    return attr == CUBIN_FUNCTION_PATCH_EIATTR_FRAME_SIZE ||
        attr == CUBIN_FUNCTION_PATCH_EIATTR_MIN_STACK_SIZE ||
        attr == CUBIN_FUNCTION_PATCH_EIATTR_MAX_STACK_SIZE ||
        attr == CUBIN_FUNCTION_PATCH_EIATTR_REGCOUNT;
}

static CubinFunctionPatchResult
cubin_function_patch_find_global_info_span(
    const void* elf_data,
    size_t elf_bytes,
    CubinFunctionPatchSpan* span_out
) {
    Elf64_Ehdr eh;
    CubinFunctionPatchResult result;
    size_t i;
    size_t match_count = 0u;
    CubinFunctionPatchSpan matched_span;

    if (span_out == NULL) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_VALUE;
    memset(span_out, 0, sizeof(*span_out));
    memset(&matched_span, 0, sizeof(matched_span));
    result = cubin_function_patch_read_ehdr(elf_data, elf_bytes, &eh);
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;

    for (i = 0u; i < eh.e_shnum; ++i) {
        Elf64_Shdr section;
        size_t section_offset;
        size_t section_size;

        result = cubin_function_patch_read_shdr(elf_data, elf_bytes, &eh, i, &section);
        if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
        if (section.sh_type != CUBIN_FUNCTION_PATCH_SHT_CUDA_INFO) continue;
        if (!cubin_function_patch_section_name_equals(elf_data, elf_bytes, &eh, i, ".nv.info")) continue;
        result = cubin_function_patch_u64_to_size(section.sh_offset, &section_offset);
        if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
        result = cubin_function_patch_u64_to_size(section.sh_size, &section_size);
        if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
        if (!cubin_function_patch_range_ok(section_offset, section_size, elf_bytes)) {
            return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
        }
        matched_span.file_offset = section_offset;
        matched_span.size = section_size;
        matched_span.section_index = i;
        matched_span.section_offset = 0u;
        match_count++;
    }

    if (match_count == 0u) return CUBIN_FUNCTION_PATCH_ERROR_SYMBOL_NOT_FOUND;
    if (match_count != 1u) return CUBIN_FUNCTION_PATCH_ERROR_AMBIGUOUS_SYMBOL;
    *span_out = matched_span;
    return CUBIN_FUNCTION_PATCH_SUCCESS;
}

static CubinFunctionPatchResult
cubin_function_patch_info_next_offset(
    const unsigned char* info,
    size_t info_size,
    size_t offset,
    size_t* next_offset_out
) {
    size_t entry_size = 4u;
    unsigned char format;
    CubinFunctionPatchResult result;

    if (info == NULL || next_offset_out == NULL || offset > info_size || info_size - offset < 4u) {
        return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
    }
    format = info[offset];
    if (format == CUBIN_FUNCTION_PATCH_EIFMT_SVAL) {
        uint16_t payload_size = cubin_function_patch_read_u16_unaligned(info + offset + 2u);
        result = cubin_function_patch_add_size(4u, (size_t)payload_size, &entry_size);
        if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
        result = cubin_function_patch_align_up_size(entry_size, 4u, &entry_size);
        if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
    }
    if (entry_size > info_size - offset) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
    *next_offset_out = offset + entry_size;
    return CUBIN_FUNCTION_PATCH_SUCCESS;
}

static void
cubin_function_patch_global_info_values_set(
    CubinFunctionPatchGlobalInfoValues* values,
    unsigned char attr,
    uint32_t value
) {
    if (values == NULL) return;
    switch (attr) {
        case CUBIN_FUNCTION_PATCH_EIATTR_FRAME_SIZE:
            values->frame_size = value;
            values->has_frame_size = 1;
            break;
        case CUBIN_FUNCTION_PATCH_EIATTR_MIN_STACK_SIZE:
            values->min_stack_size = value;
            values->has_min_stack_size = 1;
            break;
        case CUBIN_FUNCTION_PATCH_EIATTR_MAX_STACK_SIZE:
            values->max_stack_size = value;
            values->has_max_stack_size = 1;
            break;
        case CUBIN_FUNCTION_PATCH_EIATTR_REGCOUNT:
            values->reg_count = value;
            values->has_reg_count = 1;
            break;
    }
}

static int
cubin_function_patch_global_info_values_has_attr(
    const CubinFunctionPatchGlobalInfoValues* values,
    unsigned char attr,
    uint32_t* value_out
) {
    if (values == NULL || value_out == NULL) return 0;
    switch (attr) {
        case CUBIN_FUNCTION_PATCH_EIATTR_FRAME_SIZE:
            if (!values->has_frame_size) return 0;
            *value_out = values->frame_size;
            return 1;
        case CUBIN_FUNCTION_PATCH_EIATTR_MIN_STACK_SIZE:
            if (!values->has_min_stack_size) return 0;
            *value_out = values->min_stack_size;
            return 1;
        case CUBIN_FUNCTION_PATCH_EIATTR_MAX_STACK_SIZE:
            if (!values->has_max_stack_size) return 0;
            *value_out = values->max_stack_size;
            return 1;
        case CUBIN_FUNCTION_PATCH_EIATTR_REGCOUNT:
            if (!values->has_reg_count) return 0;
            *value_out = values->reg_count;
            return 1;
    }
    return 0;
}

static CubinFunctionPatchResult
cubin_function_patch_collect_global_info_values(
    const void* replacement_rdc_elf,
    size_t replacement_rdc_elf_bytes,
    size_t replacement_symbol_index,
    CubinFunctionPatchGlobalInfoValues* values_out
) {
    CubinFunctionPatchSpan info_span;
    const unsigned char* info;
    size_t offset = 0u;
    CubinFunctionPatchResult result;

    if (values_out == NULL) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_VALUE;
    memset(values_out, 0, sizeof(*values_out));
    result = cubin_function_patch_find_global_info_span(
        replacement_rdc_elf,
        replacement_rdc_elf_bytes,
        &info_span
    );
    if (result == CUBIN_FUNCTION_PATCH_ERROR_SYMBOL_NOT_FOUND) return CUBIN_FUNCTION_PATCH_SUCCESS;
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
    info = (const unsigned char*)replacement_rdc_elf + info_span.file_offset;

    while (offset < info_span.size) {
        size_t next_offset;
        unsigned char format = info[offset];
        unsigned char attr = info[offset + 1u];
        uint16_t payload_size = 0u;

        result = cubin_function_patch_info_next_offset(info, info_span.size, offset, &next_offset);
        if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
        if (format == CUBIN_FUNCTION_PATCH_EIFMT_SVAL &&
            cubin_function_patch_global_info_attr_patchable(attr)) {
            payload_size = cubin_function_patch_read_u16_unaligned(info + offset + 2u);
            if (payload_size == 8u) {
                uint32_t symbol_index = cubin_function_patch_read_u32_unaligned(info + offset + 4u);
                uint32_t value = cubin_function_patch_read_u32_unaligned(info + offset + 8u);
                if ((size_t)symbol_index == replacement_symbol_index) {
                    cubin_function_patch_global_info_values_set(values_out, attr, value);
                }
            }
        }
        offset = next_offset;
    }

    return CUBIN_FUNCTION_PATCH_SUCCESS;
}

static CubinFunctionPatchResult
cubin_function_patch_apply_global_info_values(
    const CubinFunctionPatchHandle* handle,
    const CubinFunctionPatchSite* site,
    const CubinFunctionPatchGlobalInfoValues* values,
    void* output_cubin,
    size_t output_cubin_bytes
) {
    CubinFunctionPatchSpan info_span;
    unsigned char* info;
    size_t offset = 0u;
    int found_frame_size = 0;
    int found_min_stack_size = 0;
    int found_max_stack_size = 0;
    int found_reg_count = 0;
    CubinFunctionPatchResult result;

    if (handle == NULL || site == NULL || values == NULL || output_cubin == NULL) {
        return CUBIN_FUNCTION_PATCH_ERROR_INVALID_VALUE;
    }
    if (!values->has_frame_size && !values->has_min_stack_size &&
        !values->has_max_stack_size && !values->has_reg_count) {
        return CUBIN_FUNCTION_PATCH_SUCCESS;
    }
    result = cubin_function_patch_find_global_info_span(
        handle->linked_cubin,
        handle->linked_cubin_bytes,
        &info_span
    );
    if (result == CUBIN_FUNCTION_PATCH_ERROR_SYMBOL_NOT_FOUND) return CUBIN_FUNCTION_PATCH_SUCCESS;
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
    if (info_span.file_offset > output_cubin_bytes ||
        output_cubin_bytes - info_span.file_offset < info_span.size) {
        return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
    }
    info = (unsigned char*)output_cubin + info_span.file_offset;

    while (offset < info_span.size) {
        size_t next_offset;
        unsigned char format = info[offset];
        unsigned char attr = info[offset + 1u];
        uint16_t payload_size = 0u;

        result = cubin_function_patch_info_next_offset(info, info_span.size, offset, &next_offset);
        if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
        if (format == CUBIN_FUNCTION_PATCH_EIFMT_SVAL &&
            cubin_function_patch_global_info_attr_patchable(attr)) {
            payload_size = cubin_function_patch_read_u16_unaligned(info + offset + 2u);
            if (payload_size == 8u) {
                uint32_t symbol_index = cubin_function_patch_read_u32_unaligned(info + offset + 4u);
                uint32_t value = 0u;
                if (handle->sm_version == 89u &&
                    attr == CUBIN_FUNCTION_PATCH_EIATTR_REGCOUNT &&
                    values->has_reg_count) {
                    uint32_t current_value = cubin_function_patch_read_u32_unaligned(info + offset + 8u);
                    if (current_value < values->reg_count) {
                        cubin_function_patch_write_u32_unaligned(info + offset + 8u, values->reg_count);
                    }
                }
                if ((size_t)symbol_index == site->symbol_index &&
                    cubin_function_patch_global_info_values_has_attr(values, attr, &value)) {
                    cubin_function_patch_write_u32_unaligned(info + offset + 8u, value);
                    switch (attr) {
                        case CUBIN_FUNCTION_PATCH_EIATTR_FRAME_SIZE:
                            found_frame_size = 1;
                            break;
                        case CUBIN_FUNCTION_PATCH_EIATTR_MIN_STACK_SIZE:
                            found_min_stack_size = 1;
                            break;
                        case CUBIN_FUNCTION_PATCH_EIATTR_MAX_STACK_SIZE:
                            found_max_stack_size = 1;
                            break;
                        case CUBIN_FUNCTION_PATCH_EIATTR_REGCOUNT:
                            found_reg_count = 1;
                            break;
                    }
                }
            }
        }
        offset = next_offset;
    }

    if (values->has_reg_count && !found_reg_count) return CUBIN_FUNCTION_PATCH_ERROR_RELOCATION_UNSUPPORTED;
    if (values->has_frame_size && values->frame_size != 0u && !found_frame_size) {
        return CUBIN_FUNCTION_PATCH_ERROR_RELOCATION_UNSUPPORTED;
    }
    if (values->has_min_stack_size && values->min_stack_size != 0u && !found_min_stack_size) {
        return CUBIN_FUNCTION_PATCH_ERROR_RELOCATION_UNSUPPORTED;
    }
    if (values->has_max_stack_size && values->max_stack_size != 0u && !found_max_stack_size) {
        return CUBIN_FUNCTION_PATCH_ERROR_RELOCATION_UNSUPPORTED;
    }
    return CUBIN_FUNCTION_PATCH_SUCCESS;
}

static CubinFunctionPatchResult
cubin_function_patch_apply_global_info_from_replacement(
    const CubinFunctionPatchHandle* handle,
    const CubinFunctionPatchSite* site,
    const void* replacement_rdc_elf,
    size_t replacement_rdc_elf_bytes,
    size_t replacement_symbol_index,
    void* output_cubin,
    size_t output_cubin_bytes
) {
    CubinFunctionPatchGlobalInfoValues values;
    CubinFunctionPatchResult result;

    result = cubin_function_patch_collect_global_info_values(
        replacement_rdc_elf,
        replacement_rdc_elf_bytes,
        replacement_symbol_index,
        &values
    );
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
    return cubin_function_patch_apply_global_info_values(
        handle,
        site,
        &values,
        output_cubin,
        output_cubin_bytes
    );
}

static CubinFunctionPatchResult
cubin_function_patch_write_section_header(
    void* elf_data,
    size_t elf_bytes,
    const Elf64_Ehdr* eh,
    size_t section_index,
    const Elf64_Shdr* section
) {
    size_t section_table_offset;
    size_t section_delta;
    size_t section_offset;
    CubinFunctionPatchResult result;

    if (elf_data == NULL || eh == NULL || section == NULL || section_index >= eh->e_shnum) {
        return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
    }
    result = cubin_function_patch_u64_to_size(eh->e_shoff, &section_table_offset);
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
    result = cubin_function_patch_mul_size(section_index, sizeof(Elf64_Shdr), &section_delta);
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
    result = cubin_function_patch_add_size(section_table_offset, section_delta, &section_offset);
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
    if (!cubin_function_patch_range_ok(section_offset, sizeof(*section), elf_bytes)) {
        return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
    }
    memcpy((unsigned char*)elf_data + section_offset, section, sizeof(*section));
    return CUBIN_FUNCTION_PATCH_SUCCESS;
}

static CubinFunctionPatchResult
cubin_function_patch_write_sym(
    void* elf_data,
    size_t elf_bytes,
    const Elf64_Shdr* symtab,
    size_t symbol_index,
    const Elf64_Sym* sym
) {
    size_t table_offset;
    size_t symbol_delta;
    size_t symbol_offset;
    CubinFunctionPatchResult result;

    if (elf_data == NULL || symtab == NULL || sym == NULL || symtab->sh_entsize != sizeof(Elf64_Sym)) {
        return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
    }
    result = cubin_function_patch_u64_to_size(symtab->sh_offset, &table_offset);
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
    result = cubin_function_patch_mul_size(symbol_index, sizeof(Elf64_Sym), &symbol_delta);
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
    result = cubin_function_patch_add_size(table_offset, symbol_delta, &symbol_offset);
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
    if (!cubin_function_patch_range_ok(symbol_offset, sizeof(*sym), elf_bytes)) {
        return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
    }
    memcpy((unsigned char*)elf_data + symbol_offset, sym, sizeof(*sym));
    return CUBIN_FUNCTION_PATCH_SUCCESS;
}

static CubinFunctionPatchResult
cubin_function_patch_apply_function_elf_metadata(
    const CubinFunctionPatchSite* site,
    const void* replacement_rdc_elf,
    size_t replacement_rdc_elf_bytes,
    const CubinFunctionPatchSpan* replacement_span,
    void* output_cubin,
    size_t output_cubin_bytes
) {
    Elf64_Ehdr output_eh;
    Elf64_Shdr output_text_section;
    Elf64_Shdr output_symtab;
    Elf64_Sym output_symbol;
    Elf64_Ehdr replacement_eh;
    Elf64_Shdr replacement_text_section;
    CubinFunctionPatchResult result;

    if (site == NULL || replacement_span == NULL || output_cubin == NULL) {
        return CUBIN_FUNCTION_PATCH_ERROR_INVALID_VALUE;
    }
    result = cubin_function_patch_read_ehdr(output_cubin, output_cubin_bytes, &output_eh);
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
    result = cubin_function_patch_read_shdr(
        output_cubin,
        output_cubin_bytes,
        &output_eh,
        site->section_index,
        &output_text_section
    );
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
    result = cubin_function_patch_read_shdr(
        output_cubin,
        output_cubin_bytes,
        &output_eh,
        site->symbol_table_index,
        &output_symtab
    );
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
    result = cubin_function_patch_read_sym(
        output_cubin,
        output_cubin_bytes,
        &output_symtab,
        site->symbol_index,
        &output_symbol
    );
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
    result = cubin_function_patch_read_ehdr(replacement_rdc_elf, replacement_rdc_elf_bytes, &replacement_eh);
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
    result = cubin_function_patch_read_shdr(
        replacement_rdc_elf,
        replacement_rdc_elf_bytes,
        &replacement_eh,
        replacement_span->section_index,
        &replacement_text_section
    );
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
    if (site->symbol_index > 0x00ffffffu) return CUBIN_FUNCTION_PATCH_ERROR_RELOCATION_UNSUPPORTED;

    output_text_section.sh_size = (Elf64_Xword)replacement_span->size;
    output_text_section.sh_info =
        (replacement_text_section.sh_info & 0xff000000u) |
        (Elf64_Word)(site->symbol_index & 0x00ffffffu);
    result = cubin_function_patch_write_section_header(
        output_cubin,
        output_cubin_bytes,
        &output_eh,
        site->section_index,
        &output_text_section
    );
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;

    output_symbol.st_size = (Elf64_Xword)replacement_span->size;
    result = cubin_function_patch_write_sym(
        output_cubin,
        output_cubin_bytes,
        &output_symtab,
        site->symbol_index,
        &output_symbol
    );
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
    return CUBIN_FUNCTION_PATCH_SUCCESS;
}

static const CubinFunctionPatchSite*
cubin_function_patch_site_at(
    const CubinFunctionPatchHandle* handle,
    size_t site_index
) {
    if (handle == NULL || site_index >= handle->num_sites) return NULL;
    return &handle->sites[site_index];
}

static const CubinFunctionPatchSite*
cubin_function_patch_find_site(
    const CubinFunctionPatchHandle* handle,
    const char* symbol_name
) {
    size_t i;
    if (handle == NULL || symbol_name == NULL) return NULL;
    for (i = 0u; i < handle->num_sites; ++i) {
        if (strcmp(handle->sites[i].symbol_name, symbol_name) == 0) {
            return &handle->sites[i];
        }
    }
    return NULL;
}

CUBIN_FUNCTION_PATCH_PUBLIC_DEF
const char*
cubin_function_patch_result_to_string(
    CubinFunctionPatchResult result
) {
    switch (result) {
        case CUBIN_FUNCTION_PATCH_SUCCESS: return "CUBIN_FUNCTION_PATCH_SUCCESS";
        case CUBIN_FUNCTION_PATCH_ERROR_INVALID_VALUE: return "CUBIN_FUNCTION_PATCH_ERROR_INVALID_VALUE";
        case CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF: return "CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF";
        case CUBIN_FUNCTION_PATCH_ERROR_WORKSPACE_EXHAUSTED: return "CUBIN_FUNCTION_PATCH_ERROR_WORKSPACE_EXHAUSTED";
        case CUBIN_FUNCTION_PATCH_ERROR_SYMBOL_NOT_FOUND: return "CUBIN_FUNCTION_PATCH_ERROR_SYMBOL_NOT_FOUND";
        case CUBIN_FUNCTION_PATCH_ERROR_REPLACEMENT_TOO_LARGE: return "CUBIN_FUNCTION_PATCH_ERROR_REPLACEMENT_TOO_LARGE";
        case CUBIN_FUNCTION_PATCH_ERROR_RELOCATION_UNSUPPORTED: return "CUBIN_FUNCTION_PATCH_ERROR_RELOCATION_UNSUPPORTED";
        case CUBIN_FUNCTION_PATCH_ERROR_MACHINE_MISMATCH: return "CUBIN_FUNCTION_PATCH_ERROR_MACHINE_MISMATCH";
        case CUBIN_FUNCTION_PATCH_ERROR_AMBIGUOUS_SYMBOL: return "CUBIN_FUNCTION_PATCH_ERROR_AMBIGUOUS_SYMBOL";
        case CUBIN_FUNCTION_PATCH_ERROR_TAIL_POLICY: return "CUBIN_FUNCTION_PATCH_ERROR_TAIL_POLICY";
    }
    return "CUBIN_FUNCTION_PATCH_ERROR_UNKNOWN";
}

CUBIN_FUNCTION_PATCH_PUBLIC_DEF
CubinFunctionPatchResult
cubin_function_patch_handle_size(
    const char* const* symbol_names,
    size_t num_symbols,
    size_t* bytes_out
) {
    size_t total;
    size_t i;
    CubinFunctionPatchResult result;

    if (bytes_out == NULL) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_VALUE;
    result = cubin_function_patch_check_symbol_names(symbol_names, num_symbols);
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
    total = sizeof(CubinFunctionPatchHandle);
    if (num_symbols > 1u) {
        size_t extra_sites;
        result = cubin_function_patch_mul_size(num_symbols - 1u, sizeof(CubinFunctionPatchSite), &extra_sites);
        if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
        result = cubin_function_patch_add_size(total, extra_sites, &total);
        if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
    }
    result = cubin_function_patch_align_up_size(total, 8u, &total);
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
    for (i = 0u; i < num_symbols; ++i) {
        size_t name_bytes;
        name_bytes = cubin_function_patch_strlen(symbol_names[i]) + 1u;
        result = cubin_function_patch_add_size(total, name_bytes, &total);
        if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
    }
    *bytes_out = total;
    return CUBIN_FUNCTION_PATCH_SUCCESS;
}

CUBIN_FUNCTION_PATCH_PUBLIC_DEF
CubinFunctionPatchResult
cubin_function_patch_create(
    const void* linked_cubin,
    size_t linked_cubin_bytes,
    const char* const* symbol_names,
    size_t num_symbols,
    void* handle_memory,
    size_t handle_memory_size,
    CubinFunctionPatchHandle** handle_out
) {
    CubinFunctionPatchHandle* handle;
    unsigned char* p;
    unsigned char* end;
    size_t required = 0u;
    size_t header_bytes;
    size_t i;
    uint16_t machine = 0u;
    Elf64_Ehdr linked_eh;
    CubinFunctionPatchResult result;

    if (linked_cubin == NULL || linked_cubin_bytes == 0u || symbol_names == NULL ||
        num_symbols == 0u || handle_memory == NULL || handle_out == NULL) {
        return CUBIN_FUNCTION_PATCH_ERROR_INVALID_VALUE;
    }
    *handle_out = NULL;
    result = cubin_function_patch_handle_size(symbol_names, num_symbols, &required);
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
    if (handle_memory_size < required) return CUBIN_FUNCTION_PATCH_ERROR_WORKSPACE_EXHAUSTED;
    result = cubin_function_patch_read_ehdr(linked_cubin, linked_cubin_bytes, &linked_eh);
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;

    memset(handle_memory, 0, required);
    handle = (CubinFunctionPatchHandle*)handle_memory;
    header_bytes = sizeof(CubinFunctionPatchHandle);
    if (num_symbols > 1u) {
        size_t extra_sites;
        result = cubin_function_patch_mul_size(num_symbols - 1u, sizeof(CubinFunctionPatchSite), &extra_sites);
        if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
        result = cubin_function_patch_add_size(header_bytes, extra_sites, &header_bytes);
        if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
    }
    result = cubin_function_patch_align_up_size(header_bytes, 8u, &header_bytes);
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
    p = (unsigned char*)handle_memory + header_bytes;
    end = (unsigned char*)handle_memory + handle_memory_size;

    handle->linked_cubin = (const unsigned char*)linked_cubin;
    handle->linked_cubin_bytes = linked_cubin_bytes;
    handle->elf_flags = linked_eh.e_flags;
    handle->sm_version = cubin_function_patch_sm_from_elf_flags(linked_eh.e_flags);
    handle->num_sites = num_symbols;

    for (i = 0u; i < num_symbols; ++i) {
        CubinFunctionPatchSpan span;
        CubinFunctionPatchSpan capmerc_span;
        size_t name_bytes = cubin_function_patch_strlen(symbol_names[i]) + 1u;
        size_t fallback_side_idx = i;
        size_t symbol_suffix = 0u;
        result = cubin_function_patch_find_function_span(
            linked_cubin,
            linked_cubin_bytes,
            symbol_names[i],
            &span,
            &handle->sites[i].symbol_index,
            &handle->sites[i].symbol_table_index,
            i == 0u ? &machine : NULL
        );
        if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
        if ((size_t)(end - p) < name_bytes) return CUBIN_FUNCTION_PATCH_ERROR_WORKSPACE_EXHAUSTED;
        memcpy(p, symbol_names[i], name_bytes);
        handle->sites[i].symbol_name = (char*)p;
        handle->sites[i].linked_offset = span.file_offset;
        handle->sites[i].reserved_size = span.size;
        handle->sites[i].section_index = span.section_index;
        handle->sites[i].section_offset = span.section_offset;
        result = cubin_function_patch_find_capmerc_span(
            linked_cubin,
            linked_cubin_bytes,
            symbol_names[i],
            &capmerc_span
        );
        if (result == CUBIN_FUNCTION_PATCH_SUCCESS) {
            handle->sites[i].capmerc_linked_offset = capmerc_span.file_offset;
            handle->sites[i].capmerc_reserved_size = capmerc_span.size;
            handle->sites[i].capmerc_section_index = capmerc_span.section_index;
            handle->sites[i].capmerc_section_offset = capmerc_span.section_offset;
            handle->sites[i].has_capmerc = 1;
        } else if (result != CUBIN_FUNCTION_PATCH_ERROR_SYMBOL_NOT_FOUND) {
            return result;
        }
        result = cubin_function_patch_add_side_section(
            &handle->sites[i],
            linked_cubin,
            linked_cubin_bytes,
            symbol_names[i],
            CUBIN_FUNCTION_PATCH_SIDE_SECTION_INFO
        );
        if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
        result = cubin_function_patch_add_side_section(
            &handle->sites[i],
            linked_cubin,
            linked_cubin_bytes,
            symbol_names[i],
            CUBIN_FUNCTION_PATCH_SIDE_SECTION_CONSTANT2
        );
        if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
        if (cubin_function_patch_parse_decimal_suffix(symbol_names[i], &symbol_suffix) &&
            symbol_suffix < num_symbols) {
            fallback_side_idx = symbol_suffix;
        }
        result = cubin_function_patch_add_fallback_side_section(
            &handle->sites[i],
            linked_cubin,
            linked_cubin_bytes,
            CUBIN_FUNCTION_PATCH_SIDE_SECTION_CONSTANT2,
            fallback_side_idx,
            num_symbols
        );
        if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
        p += name_bytes;
    }

    handle->elf_machine = machine;
    *handle_out = handle;
    return CUBIN_FUNCTION_PATCH_SUCCESS;
}

CUBIN_FUNCTION_PATCH_PUBLIC_DEF
void
cubin_function_patch_destroy(
    CubinFunctionPatchHandle* handle
) {
    (void)handle;
}

CUBIN_FUNCTION_PATCH_PUBLIC_DEF
size_t
cubin_function_patch_num_sites(
    const CubinFunctionPatchHandle* handle
) {
    if (handle == NULL) return 0u;
    return handle->num_sites;
}

CUBIN_FUNCTION_PATCH_PUBLIC_DEF
const char*
cubin_function_patch_site_symbol(
    const CubinFunctionPatchHandle* handle,
    size_t site_index
) {
    const CubinFunctionPatchSite* site = cubin_function_patch_site_at(handle, site_index);
    if (site == NULL) return NULL;
    return site->symbol_name;
}

CUBIN_FUNCTION_PATCH_PUBLIC_DEF
CubinFunctionPatchResult
cubin_function_patch_site_reserved_size(
    const CubinFunctionPatchHandle* handle,
    size_t site_index,
    size_t* bytes_out
) {
    const CubinFunctionPatchSite* site;
    if (bytes_out == NULL) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_VALUE;
    site = cubin_function_patch_site_at(handle, site_index);
    if (site == NULL) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_VALUE;
    *bytes_out = site->reserved_size;
    return CUBIN_FUNCTION_PATCH_SUCCESS;
}

CUBIN_FUNCTION_PATCH_PUBLIC_DEF
CubinFunctionPatchResult
cubin_function_patch_output_size(
    const CubinFunctionPatchHandle* handle,
    size_t* bytes_out
) {
    if (handle == NULL || bytes_out == NULL) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_VALUE;
    *bytes_out = handle->linked_cubin_bytes;
    return CUBIN_FUNCTION_PATCH_SUCCESS;
}

CUBIN_FUNCTION_PATCH_PUBLIC_DEF
CubinFunctionPatchResult
cubin_function_patch_begin(
    const CubinFunctionPatchHandle* handle,
    void* output_cubin,
    size_t output_cubin_capacity,
    size_t* output_cubin_bytes_out
) {
    if (handle == NULL || output_cubin == NULL || output_cubin_bytes_out == NULL) {
        return CUBIN_FUNCTION_PATCH_ERROR_INVALID_VALUE;
    }
    if (output_cubin_capacity < handle->linked_cubin_bytes) {
        return CUBIN_FUNCTION_PATCH_ERROR_WORKSPACE_EXHAUSTED;
    }
    memcpy(output_cubin, handle->linked_cubin, handle->linked_cubin_bytes);
    *output_cubin_bytes_out = handle->linked_cubin_bytes;
    return CUBIN_FUNCTION_PATCH_SUCCESS;
}

CUBIN_FUNCTION_PATCH_PUBLIC_DEF
CubinFunctionPatchResult
cubin_function_patch_apply_one_in_place_ex(
    const CubinFunctionPatchHandle* handle,
    void* output_cubin,
    size_t output_cubin_bytes,
    const char* symbol_name,
    const void* replacement_rdc_elf,
    size_t replacement_rdc_elf_bytes,
    CubinFunctionPatchTailPolicy tail_policy,
    CubinFunctionPatchReport* report_out
) {
    const CubinFunctionPatchSite* site;
    CubinFunctionPatchSpan replacement_span;
    CubinFunctionPatchSpan replacement_capmerc_span;
    size_t replacement_symbol_index = 0u;
    uint16_t replacement_machine = 0u;
    int has_unsupported_relocations = 0;
    CubinFunctionPatchResult result;
    size_t side_idx;

    if (handle == NULL || output_cubin == NULL || output_cubin_bytes < handle->linked_cubin_bytes ||
        symbol_name == NULL || replacement_rdc_elf == NULL || replacement_rdc_elf_bytes == 0u) {
        return CUBIN_FUNCTION_PATCH_ERROR_INVALID_VALUE;
    }
    if (tail_policy != CUBIN_FUNCTION_PATCH_TAIL_LEAVE_UNCHANGED &&
        tail_policy != CUBIN_FUNCTION_PATCH_TAIL_REQUIRE_EXACT_SIZE) {
        return CUBIN_FUNCTION_PATCH_ERROR_INVALID_VALUE;
    }
    site = cubin_function_patch_find_site(handle, symbol_name);
    if (site == NULL) return CUBIN_FUNCTION_PATCH_ERROR_SYMBOL_NOT_FOUND;
    result = cubin_function_patch_find_function_span(
        replacement_rdc_elf,
        replacement_rdc_elf_bytes,
        symbol_name,
        &replacement_span,
        &replacement_symbol_index,
        NULL,
        &replacement_machine
    );
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
    if (replacement_machine != handle->elf_machine) return CUBIN_FUNCTION_PATCH_ERROR_MACHINE_MISMATCH;
    if (replacement_span.size > site->reserved_size) return CUBIN_FUNCTION_PATCH_ERROR_REPLACEMENT_TOO_LARGE;
    if (tail_policy == CUBIN_FUNCTION_PATCH_TAIL_REQUIRE_EXACT_SIZE &&
        replacement_span.size != site->reserved_size) {
        return CUBIN_FUNCTION_PATCH_ERROR_TAIL_POLICY;
    }
    result = cubin_function_patch_span_has_unsupported_relocations(
        replacement_rdc_elf,
        replacement_rdc_elf_bytes,
        &replacement_span,
        symbol_name,
        site,
        &has_unsupported_relocations
    );
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
    if (has_unsupported_relocations) return CUBIN_FUNCTION_PATCH_ERROR_RELOCATION_UNSUPPORTED;
    if (site->has_capmerc) {
        result = cubin_function_patch_find_capmerc_span(
            replacement_rdc_elf,
            replacement_rdc_elf_bytes,
            symbol_name,
            &replacement_capmerc_span
        );
        if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
        if (replacement_capmerc_span.size > site->capmerc_reserved_size) {
            return CUBIN_FUNCTION_PATCH_ERROR_REPLACEMENT_TOO_LARGE;
        }
        if (tail_policy == CUBIN_FUNCTION_PATCH_TAIL_REQUIRE_EXACT_SIZE &&
            replacement_capmerc_span.size != site->capmerc_reserved_size) {
            return CUBIN_FUNCTION_PATCH_ERROR_TAIL_POLICY;
        }
        result = cubin_function_patch_span_has_unsupported_relocations(
            replacement_rdc_elf,
            replacement_rdc_elf_bytes,
            &replacement_capmerc_span,
            symbol_name,
            site,
            &has_unsupported_relocations
        );
        if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
        if (has_unsupported_relocations) return CUBIN_FUNCTION_PATCH_ERROR_RELOCATION_UNSUPPORTED;
    }

    memcpy(
        (unsigned char*)output_cubin + site->linked_offset,
        (const unsigned char*)replacement_rdc_elf + replacement_span.file_offset,
        replacement_span.size
    );
    result = cubin_function_patch_apply_supported_relocations(
        replacement_rdc_elf,
        replacement_rdc_elf_bytes,
        &replacement_span,
        symbol_name,
        site,
        output_cubin,
        output_cubin_bytes,
        site->linked_offset
    );
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
    if (site->has_capmerc) {
        memcpy(
            (unsigned char*)output_cubin + site->capmerc_linked_offset,
            (const unsigned char*)replacement_rdc_elf + replacement_capmerc_span.file_offset,
            replacement_capmerc_span.size
        );
        result = cubin_function_patch_apply_supported_relocations(
            replacement_rdc_elf,
            replacement_rdc_elf_bytes,
            &replacement_capmerc_span,
            symbol_name,
            site,
            output_cubin,
            output_cubin_bytes,
            site->capmerc_linked_offset
        );
        if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
    }
    for (side_idx = 0u; side_idx < site->num_side_sections; ++side_idx) {
        const CubinFunctionPatchSideSection* side = &site->side_sections[side_idx];
        CubinFunctionPatchSpan replacement_side_span;
        if (side->linked_offset > output_cubin_bytes ||
            output_cubin_bytes - side->linked_offset < side->reserved_size) {
            return CUBIN_FUNCTION_PATCH_ERROR_INVALID_ELF;
        }
        result = cubin_function_patch_find_side_section_span(
            replacement_rdc_elf,
            replacement_rdc_elf_bytes,
            symbol_name,
            side->kind,
            &replacement_side_span
        );
        if (result == CUBIN_FUNCTION_PATCH_ERROR_SYMBOL_NOT_FOUND) continue;
        if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
        if (replacement_side_span.size > side->reserved_size) {
            return CUBIN_FUNCTION_PATCH_ERROR_REPLACEMENT_TOO_LARGE;
        }
        if (side->kind == CUBIN_FUNCTION_PATCH_SIDE_SECTION_INFO) {
            continue;
        }
        memset((unsigned char*)output_cubin + side->linked_offset, 0, side->reserved_size);
        memcpy(
            (unsigned char*)output_cubin + side->linked_offset,
            (const unsigned char*)replacement_rdc_elf + replacement_side_span.file_offset,
            replacement_side_span.size
        );
    }
    result = cubin_function_patch_apply_function_elf_metadata(
        site,
        replacement_rdc_elf,
        replacement_rdc_elf_bytes,
        &replacement_span,
        output_cubin,
        output_cubin_bytes
    );
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
    result = cubin_function_patch_apply_global_info_from_replacement(
        handle,
        site,
        replacement_rdc_elf,
        replacement_rdc_elf_bytes,
        replacement_symbol_index,
        output_cubin,
        output_cubin_bytes
    );
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
    if (report_out != NULL) {
        report_out->symbol_name = site->symbol_name;
        report_out->reserved_size = site->reserved_size;
        report_out->replacement_size = replacement_span.size;
    }
    return CUBIN_FUNCTION_PATCH_SUCCESS;
}

CUBIN_FUNCTION_PATCH_PUBLIC_DEF
CubinFunctionPatchResult
cubin_function_patch_apply_one_in_place(
    const CubinFunctionPatchHandle* handle,
    void* output_cubin,
    size_t output_cubin_bytes,
    const char* symbol_name,
    const void* replacement_rdc_elf,
    size_t replacement_rdc_elf_bytes,
    CubinFunctionPatchReport* report_out
) {
    return cubin_function_patch_apply_one_in_place_ex(
        handle,
        output_cubin,
        output_cubin_bytes,
        symbol_name,
        replacement_rdc_elf,
        replacement_rdc_elf_bytes,
        CUBIN_FUNCTION_PATCH_TAIL_LEAVE_UNCHANGED,
        report_out
    );
}

CUBIN_FUNCTION_PATCH_PUBLIC_DEF
CubinFunctionPatchResult
cubin_function_patch_apply_all_in_place_ex(
    const CubinFunctionPatchHandle* handle,
    void* output_cubin,
    size_t output_cubin_bytes,
    const void* replacement_rdc_elf,
    size_t replacement_rdc_elf_bytes,
    CubinFunctionPatchTailPolicy tail_policy,
    CubinFunctionPatchReport* reports,
    size_t report_count
) {
    size_t i;
    if (handle == NULL) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_VALUE;
    if (reports != NULL && report_count < handle->num_sites) return CUBIN_FUNCTION_PATCH_ERROR_INVALID_VALUE;
    for (i = 0u; i < handle->num_sites; ++i) {
        CubinFunctionPatchResult result = cubin_function_patch_apply_one_in_place_ex(
            handle,
            output_cubin,
            output_cubin_bytes,
            handle->sites[i].symbol_name,
            replacement_rdc_elf,
            replacement_rdc_elf_bytes,
            tail_policy,
            reports != NULL ? &reports[i] : NULL
        );
        if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
    }
    return CUBIN_FUNCTION_PATCH_SUCCESS;
}

CUBIN_FUNCTION_PATCH_PUBLIC_DEF
CubinFunctionPatchResult
cubin_function_patch_apply_all_in_place(
    const CubinFunctionPatchHandle* handle,
    void* output_cubin,
    size_t output_cubin_bytes,
    const void* replacement_rdc_elf,
    size_t replacement_rdc_elf_bytes,
    CubinFunctionPatchReport* reports,
    size_t report_count
) {
    return cubin_function_patch_apply_all_in_place_ex(
        handle,
        output_cubin,
        output_cubin_bytes,
        replacement_rdc_elf,
        replacement_rdc_elf_bytes,
        CUBIN_FUNCTION_PATCH_TAIL_LEAVE_UNCHANGED,
        reports,
        report_count
    );
}

CUBIN_FUNCTION_PATCH_PUBLIC_DEF
CubinFunctionPatchResult
cubin_function_patch_apply_one_ex(
    const CubinFunctionPatchHandle* handle,
    const char* symbol_name,
    const void* replacement_rdc_elf,
    size_t replacement_rdc_elf_bytes,
    CubinFunctionPatchTailPolicy tail_policy,
    void* output_cubin,
    size_t output_cubin_capacity,
    size_t* output_cubin_bytes_out,
    CubinFunctionPatchReport* report_out
) {
    CubinFunctionPatchResult result = cubin_function_patch_begin(
        handle,
        output_cubin,
        output_cubin_capacity,
        output_cubin_bytes_out
    );
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
    return cubin_function_patch_apply_one_in_place_ex(
        handle,
        output_cubin,
        *output_cubin_bytes_out,
        symbol_name,
        replacement_rdc_elf,
        replacement_rdc_elf_bytes,
        tail_policy,
        report_out
    );
}

CUBIN_FUNCTION_PATCH_PUBLIC_DEF
CubinFunctionPatchResult
cubin_function_patch_apply_one(
    const CubinFunctionPatchHandle* handle,
    const char* symbol_name,
    const void* replacement_rdc_elf,
    size_t replacement_rdc_elf_bytes,
    void* output_cubin,
    size_t output_cubin_capacity,
    size_t* output_cubin_bytes_out,
    CubinFunctionPatchReport* report_out
) {
    return cubin_function_patch_apply_one_ex(
        handle,
        symbol_name,
        replacement_rdc_elf,
        replacement_rdc_elf_bytes,
        CUBIN_FUNCTION_PATCH_TAIL_LEAVE_UNCHANGED,
        output_cubin,
        output_cubin_capacity,
        output_cubin_bytes_out,
        report_out
    );
}

CUBIN_FUNCTION_PATCH_PUBLIC_DEF
CubinFunctionPatchResult
cubin_function_patch_apply_all_ex(
    const CubinFunctionPatchHandle* handle,
    const void* replacement_rdc_elf,
    size_t replacement_rdc_elf_bytes,
    CubinFunctionPatchTailPolicy tail_policy,
    void* output_cubin,
    size_t output_cubin_capacity,
    size_t* output_cubin_bytes_out,
    CubinFunctionPatchReport* reports,
    size_t report_count
) {
    CubinFunctionPatchResult result = cubin_function_patch_begin(
        handle,
        output_cubin,
        output_cubin_capacity,
        output_cubin_bytes_out
    );
    if (result != CUBIN_FUNCTION_PATCH_SUCCESS) return result;
    return cubin_function_patch_apply_all_in_place_ex(
        handle,
        output_cubin,
        *output_cubin_bytes_out,
        replacement_rdc_elf,
        replacement_rdc_elf_bytes,
        tail_policy,
        reports,
        report_count
    );
}

CUBIN_FUNCTION_PATCH_PUBLIC_DEF
CubinFunctionPatchResult
cubin_function_patch_apply_all(
    const CubinFunctionPatchHandle* handle,
    const void* replacement_rdc_elf,
    size_t replacement_rdc_elf_bytes,
    void* output_cubin,
    size_t output_cubin_capacity,
    size_t* output_cubin_bytes_out,
    CubinFunctionPatchReport* reports,
    size_t report_count
) {
    return cubin_function_patch_apply_all_ex(
        handle,
        replacement_rdc_elf,
        replacement_rdc_elf_bytes,
        CUBIN_FUNCTION_PATCH_TAIL_LEAVE_UNCHANGED,
        output_cubin,
        output_cubin_capacity,
        output_cubin_bytes_out,
        reports,
        report_count
    );
}

#endif /* CUBIN_FUNCTION_PATCH_IMPLEMENTATION_ONCE */
#endif /* CUBIN_FUNCTION_PATCH_IMPLEMENTATION */
