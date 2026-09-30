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
#include <elf.h>
#include <hip/hip_runtime_api.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CUSR_AMD_START_SENTINEL 0xbf807fc0u
#define CUSR_AMD_END_SENTINEL 0xbf807fc1u
#define CUSR_AMD_S_NOP_0 0xbf800000u
#define CUSR_AMD_V_ADD_F32_BASE 0x06000000u
#define CUSR_AMD_V_MUL_F32_BASE 0x10000000u
#define CUSR_AMD_VOP2_SRC_VGPR_BASE 0x100u
#define CUSR_AMD_ANCHOR_1_BITS 0xf2u
#define CUSR_AMD_ANCHOR_2_BITS 0xf4u
#define CUSR_AMD_ANCHOR_4_BITS 0xf6u
#define CUSR_AMD_ISLAND_NOP_COUNT 32u

#define CUSR_AMD_CHECK_HIP(ans) \
    do { \
        hipError_t cusr_amd_hip_result = (ans); \
        if (cusr_amd_hip_result != hipSuccess) { \
            fprintf(stderr, "%s failed: %s\n", #ans, hipGetErrorString(cusr_amd_hip_result)); \
            return 0; \
        } \
    } while (0)

typedef enum CusrAmdProbeOp {
    CUSR_AMD_PROBE_OP_ADD = 0,
    CUSR_AMD_PROBE_OP_MUL = 1
} CusrAmdProbeOp;

typedef struct CusrAmdKernelCode {
    size_t file_offset;
    size_t size;
} CusrAmdKernelCode;

typedef struct CusrAmdRunResources {
    unsigned char* image;
    float* host_input0;
    float* host_input1;
    float* host_output;
    float* device_input0;
    float* device_input1;
    float* device_output;
    hipModule_t module;
} CusrAmdRunResources;

static uint32_t
cusr_amd_read_u32(const unsigned char* data)
{
    uint32_t value;
    memcpy(&value, data, sizeof(value));
    return value;
}

static void
cusr_amd_write_u32(unsigned char* data, uint32_t value)
{
    memcpy(data, &value, sizeof(value));
}

static int
cusr_amd_range_is_valid(size_t offset, size_t bytes, size_t size)
{
    return offset <= size && bytes <= size - offset;
}

static int
cusr_amd_find_kernel(
    const unsigned char* image,
    size_t image_size,
    const char* kernel_name,
    CusrAmdKernelCode* code_ret
)
{
    const Elf64_Ehdr* ehdr;
    const Elf64_Shdr* sections;
    size_t section_idx;

    if (image == NULL || kernel_name == NULL || code_ret == NULL || image_size < sizeof(Elf64_Ehdr)) return 0;
    ehdr = (const Elf64_Ehdr*)image;
    if (memcmp(ehdr->e_ident, ELFMAG, SELFMAG) != 0 ||
        ehdr->e_ident[EI_CLASS] != ELFCLASS64 ||
        ehdr->e_ident[EI_DATA] != ELFDATA2LSB ||
        ehdr->e_machine != EM_AMDGPU ||
        ehdr->e_shentsize != sizeof(Elf64_Shdr) ||
        ehdr->e_shnum == 0 ||
        !cusr_amd_range_is_valid(ehdr->e_shoff, (size_t)ehdr->e_shnum * sizeof(Elf64_Shdr), image_size)) return 0;

    sections = (const Elf64_Shdr*)(image + ehdr->e_shoff);

    for (section_idx = 0; section_idx < ehdr->e_shnum; ++section_idx) {
        const Elf64_Shdr* symbol_section = &sections[section_idx];
        const Elf64_Shdr* string_section;
        const Elf64_Sym* symbols;
        const char* strings;
        size_t num_symbols;
        size_t symbol_idx;

        if (symbol_section->sh_type != SHT_DYNSYM && symbol_section->sh_type != SHT_SYMTAB) continue;
        if (symbol_section->sh_link >= ehdr->e_shnum || symbol_section->sh_entsize != sizeof(Elf64_Sym)) return 0;
        string_section = &sections[symbol_section->sh_link];
        if (!cusr_amd_range_is_valid(symbol_section->sh_offset, symbol_section->sh_size, image_size) ||
            !cusr_amd_range_is_valid(string_section->sh_offset, string_section->sh_size, image_size)) return 0;

        symbols = (const Elf64_Sym*)(image + symbol_section->sh_offset);
        strings = (const char*)(image + string_section->sh_offset);
        num_symbols = symbol_section->sh_size / sizeof(Elf64_Sym);

        for (symbol_idx = 0; symbol_idx < num_symbols; ++symbol_idx) {
            const Elf64_Sym* symbol = &symbols[symbol_idx];
            const Elf64_Shdr* text_section;
            const char* name;
            size_t function_offset;

            if (symbol->st_name >= string_section->sh_size || symbol->st_shndx >= ehdr->e_shnum) continue;
            name = strings + symbol->st_name;
            if (strcmp(name, kernel_name) != 0 || ELF64_ST_TYPE(symbol->st_info) != STT_FUNC) continue;
            text_section = &sections[symbol->st_shndx];
            if (symbol->st_value < text_section->sh_addr) return 0;
            function_offset = text_section->sh_offset + (size_t)(symbol->st_value - text_section->sh_addr);
            if (!cusr_amd_range_is_valid(function_offset, symbol->st_size, image_size)) return 0;
            code_ret->file_offset = function_offset;
            code_ret->size = symbol->st_size;
            return 1;
        }
    }

    return 0;
}

static uint32_t
cusr_amd_vop2_source1(uint32_t word)
{
    return (word >> 9u) & 0xffu;
}

static uint32_t
cusr_amd_vop2_destination(uint32_t word)
{
    return (word >> 17u) & 0xffu;
}

static uint32_t
cusr_amd_encode_vop2(uint32_t opcode, uint32_t destination, uint32_t source0, uint32_t source1)
{
    return opcode |
        ((destination & 0xffu) << 17u) |
        ((source1 & 0xffu) << 9u) |
        (CUSR_AMD_VOP2_SRC_VGPR_BASE + (source0 & 0xffu));
}

static int
cusr_amd_anchor_matches(uint32_t word, uint32_t destination, uint32_t source1, uint32_t inline_constant)
{
    return (word & 0xfe000000u) == CUSR_AMD_V_ADD_F32_BASE &&
        cusr_amd_vop2_destination(word) == destination &&
        cusr_amd_vop2_source1(word) == source1 &&
        (word & 0x1ffu) == inline_constant;
}

static int
cusr_amd_patch_expression(
    unsigned char* image,
    size_t image_size,
    CusrAmdProbeOp op,
    uint32_t* input0_reg_ret,
    uint32_t* input1_reg_ret,
    uint32_t* result_reg_ret,
    size_t* island_offset_ret
)
{
    CusrAmdKernelCode code;
    unsigned char* function;
    size_t word_count;
    size_t word_idx;
    size_t start_idx = SIZE_MAX;
    size_t end_idx = SIZE_MAX;
    uint32_t anchor0;
    uint32_t anchor1;
    uint32_t result_anchor;
    uint32_t result_reg;
    uint32_t input0_reg;
    uint32_t input1_reg;
    uint32_t opcode;

    if (!cusr_amd_find_kernel(image, image_size, "cusr_amd_island_probe", &code) || code.size % sizeof(uint32_t) != 0) return 0;
    function = image + code.file_offset;
    word_count = code.size / sizeof(uint32_t);

    for (word_idx = 0; word_idx < word_count; ++word_idx) {
        const uint32_t word = cusr_amd_read_u32(function + word_idx * sizeof(uint32_t));
        if (word == CUSR_AMD_START_SENTINEL) {
            if (start_idx != SIZE_MAX) return 0;
            start_idx = word_idx;
        } else if (word == CUSR_AMD_END_SENTINEL) {
            if (end_idx != SIZE_MAX) return 0;
            end_idx = word_idx;
        }
    }

    if (start_idx == SIZE_MAX || end_idx == SIZE_MAX ||
        end_idx != start_idx + CUSR_AMD_ISLAND_NOP_COUNT + 4u) return 0;

    anchor0 = cusr_amd_read_u32(function + (start_idx + 1u) * sizeof(uint32_t));
    anchor1 = cusr_amd_read_u32(function + (start_idx + 2u) * sizeof(uint32_t));
    result_anchor = cusr_amd_read_u32(function + (end_idx - 1u) * sizeof(uint32_t));
    result_reg = cusr_amd_vop2_destination(anchor0);
    input0_reg = cusr_amd_vop2_source1(anchor0);
    input1_reg = cusr_amd_vop2_source1(anchor1);

    if (!cusr_amd_anchor_matches(anchor0, result_reg, input0_reg, CUSR_AMD_ANCHOR_1_BITS) ||
        !cusr_amd_anchor_matches(anchor1, result_reg, input1_reg, CUSR_AMD_ANCHOR_2_BITS) ||
        !cusr_amd_anchor_matches(result_anchor, result_reg, input0_reg, CUSR_AMD_ANCHOR_4_BITS)) return 0;

    for (word_idx = start_idx + 3u; word_idx < end_idx - 1u; ++word_idx) {
        if (cusr_amd_read_u32(function + word_idx * sizeof(uint32_t)) != CUSR_AMD_S_NOP_0) return 0;
    }

    opcode = op == CUSR_AMD_PROBE_OP_ADD ? CUSR_AMD_V_ADD_F32_BASE : CUSR_AMD_V_MUL_F32_BASE;
    cusr_amd_write_u32(
        function + (start_idx + 1u) * sizeof(uint32_t),
        cusr_amd_encode_vop2(opcode, result_reg, input0_reg, input1_reg)
    );
    for (word_idx = start_idx + 2u; word_idx < end_idx; ++word_idx) {
        cusr_amd_write_u32(function + word_idx * sizeof(uint32_t), CUSR_AMD_S_NOP_0);
    }

    *input0_reg_ret = input0_reg;
    *input1_reg_ret = input1_reg;
    *result_reg_ret = result_reg;
    *island_offset_ret = code.file_offset + start_idx * sizeof(uint32_t);
    return 1;
}

static int
cusr_amd_select_gfx1201(void)
{
    int count;
    int device;

    CUSR_AMD_CHECK_HIP(hipGetDeviceCount(&count));
    for (device = 0; device < count; ++device) {
        hipDeviceProp_t properties;
        CUSR_AMD_CHECK_HIP(hipGetDeviceProperties(&properties, device));
        if (strstr(properties.gcnArchName, "gfx1201") != NULL) {
            CUSR_AMD_CHECK_HIP(hipSetDevice(device));
            printf("device=%d name=%s arch=%s\n", device, properties.name, properties.gcnArchName);
            return 1;
        }
    }

    fprintf(stderr, "gfx1201 device not found\n");
    return 0;
}

static int
cusr_amd_run_variant_impl(
    const unsigned char* original_image,
    size_t image_size,
    CusrAmdProbeOp op,
    CusrAmdRunResources* resources
)
{
    enum { COUNT = 1024, THREADS = 256 };
    hipFunction_t function = NULL;
    size_t count = COUNT;
    uint32_t input0_reg;
    uint32_t input1_reg;
    uint32_t result_reg;
    size_t island_offset;
    size_t idx;

    resources->image = (unsigned char*)malloc(image_size);
    resources->host_input0 = (float*)malloc(COUNT * sizeof(float));
    resources->host_input1 = (float*)malloc(COUNT * sizeof(float));
    resources->host_output = (float*)malloc(COUNT * sizeof(float));
    if (resources->image == NULL || resources->host_input0 == NULL ||
        resources->host_input1 == NULL || resources->host_output == NULL) return 0;
    memcpy(resources->image, original_image, image_size);

    if (!cusr_amd_patch_expression(resources->image, image_size, op, &input0_reg, &input1_reg, &result_reg, &island_offset)) {
        fprintf(stderr, "failed to patch expression island\n");
        return 0;
    }

    for (idx = 0; idx < COUNT; ++idx) {
        resources->host_input0[idx] = (float)idx * 0.25f - 17.0f;
        resources->host_input1[idx] = (float)(idx % 37u) * 0.125f + 0.5f;
    }

    CUSR_AMD_CHECK_HIP(hipMalloc((void**)&resources->device_input0, COUNT * sizeof(float)));
    CUSR_AMD_CHECK_HIP(hipMalloc((void**)&resources->device_input1, COUNT * sizeof(float)));
    CUSR_AMD_CHECK_HIP(hipMalloc((void**)&resources->device_output, COUNT * sizeof(float)));
    CUSR_AMD_CHECK_HIP(hipMemcpy(resources->device_input0, resources->host_input0, COUNT * sizeof(float), hipMemcpyHostToDevice));
    CUSR_AMD_CHECK_HIP(hipMemcpy(resources->device_input1, resources->host_input1, COUNT * sizeof(float), hipMemcpyHostToDevice));
    CUSR_AMD_CHECK_HIP(hipModuleLoadData(&resources->module, resources->image));
    CUSR_AMD_CHECK_HIP(hipModuleGetFunction(&function, resources->module, "cusr_amd_island_probe"));

    {
        void* args[] = { &resources->device_input0, &resources->device_input1, &resources->device_output, &count };
        CUSR_AMD_CHECK_HIP(hipModuleLaunchKernel(
            function,
            (COUNT + THREADS - 1u) / THREADS,
            1u,
            1u,
            THREADS,
            1u,
            1u,
            0u,
            NULL,
            args,
            NULL
        ));
    }

    CUSR_AMD_CHECK_HIP(hipDeviceSynchronize());
    CUSR_AMD_CHECK_HIP(hipMemcpy(resources->host_output, resources->device_output, COUNT * sizeof(float), hipMemcpyDeviceToHost));

    for (idx = 0; idx < COUNT; ++idx) {
        const float expected = op == CUSR_AMD_PROBE_OP_ADD
            ? resources->host_input0[idx] + resources->host_input1[idx]
            : resources->host_input0[idx] * resources->host_input1[idx];
        if (resources->host_output[idx] != expected) {
            fprintf(stderr, "mismatch op=%s idx=%zu expected=%.9g actual=%.9g\n",
                op == CUSR_AMD_PROBE_OP_ADD ? "add" : "mul", idx, expected, resources->host_output[idx]);
            return 0;
        }
    }

    printf("patch op=%s island_file_offset=0x%zx input_regs={v%u,v%u} result_reg=v%u passed=yes\n",
        op == CUSR_AMD_PROBE_OP_ADD ? "add" : "mul",
        island_offset,
        input0_reg,
        input1_reg,
        result_reg);
    return 1;
}

static int
cusr_amd_run_variant(
    const unsigned char* original_image,
    size_t image_size,
    CusrAmdProbeOp op
)
{
    CusrAmdRunResources resources;
    int passed;

    memset(&resources, 0, sizeof(resources));
    passed = cusr_amd_run_variant_impl(original_image, image_size, op, &resources);
    if (resources.module != NULL) (void)hipModuleUnload(resources.module);
    if (resources.device_output != NULL) (void)hipFree(resources.device_output);
    if (resources.device_input1 != NULL) (void)hipFree(resources.device_input1);
    if (resources.device_input0 != NULL) (void)hipFree(resources.device_input0);
    free(resources.host_output);
    free(resources.host_input1);
    free(resources.host_input0);
    free(resources.image);
    return passed;
}

static int
cusr_amd_read_file(const char* path, unsigned char** data_ret, size_t* size_ret)
{
    FILE* file;
    long size;
    unsigned char* data;

    file = fopen(path, "rb");
    if (file == NULL) return 0;
    if (fseek(file, 0, SEEK_END) != 0 || (size = ftell(file)) < 0 || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return 0;
    }
    data = (unsigned char*)malloc((size_t)size);
    if (data == NULL || fread(data, 1u, (size_t)size, file) != (size_t)size) {
        free(data);
        fclose(file);
        return 0;
    }
    fclose(file);
    *data_ret = data;
    *size_ret = (size_t)size;
    return 1;
}

int
main(int argc, char** argv)
{
    unsigned char* image = NULL;
    size_t image_size = 0;
    int passed;

    if (argc != 2) {
        fprintf(stderr, "usage: %s <probe.hsaco>\n", argv[0]);
        return 2;
    }
    if (!cusr_amd_read_file(argv[1], &image, &image_size)) {
        fprintf(stderr, "failed to read %s\n", argv[1]);
        return 1;
    }
    if (!cusr_amd_select_gfx1201()) {
        free(image);
        return 1;
    }

    passed = cusr_amd_run_variant(image, image_size, CUSR_AMD_PROBE_OP_ADD) &&
        cusr_amd_run_variant(image, image_size, CUSR_AMD_PROBE_OP_MUL);
    free(image);
    return passed ? 0 : 1;
}
