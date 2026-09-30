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
#define CUSR_AST_SASS_IMPLEMENTATION
#define CUSR_AST_SASS_PATCH_IMPLEMENTATION

#include <cusr_ast_routines.h>
#include <cusr_ast_sass_patch.h>

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CUSR_CONTRACT_SITE_INSTRUCTIONS 64u
#define CUSR_CONTRACT_CUBIN_BYTES 2048u
#define CUSR_CONTRACT_REGCOUNT_OFFSET 1536u

#define CUSR_CONTRACT_CHECK(ans) \
    do { \
        if (!(ans)) { \
            fprintf(stderr, "contract check failed at %s:%d: %s\n", __FILE__, __LINE__, #ans); \
            return 0; \
        } \
    } while (0)

static const CusrAstInstruction cusr_contract_identity_routine[] = {
    cusr_ast_encode_routine_arg(0u),
    cusr_ast_encode_return
};

static const CusrAstInstruction* const cusr_contract_identity_routines[] = {
    cusr_contract_identity_routine
};

static uint32_t
cusr_contract_read_u32(const unsigned char* bytes)
{
    return
        (uint32_t)bytes[0] |
        ((uint32_t)bytes[1] << 8) |
        ((uint32_t)bytes[2] << 16) |
        ((uint32_t)bytes[3] << 24);
}

static void
cusr_contract_write_u32(unsigned char* bytes, uint32_t value)
{
    bytes[0] = (unsigned char)(value & 0xffu);
    bytes[1] = (unsigned char)((value >> 8) & 0xffu);
    bytes[2] = (unsigned char)((value >> 16) & 0xffu);
    bytes[3] = (unsigned char)((value >> 24) & 0xffu);
}

static int
cusr_contract_test_program_termination(void)
{
    const CusrAstInstruction missing_return_program[2] = {
        cusr_ast_encode_input(0u)
    };
    const uint8_t input_registers[] = { 2u };
    CusrSassInstruction sass[8];
    size_t sass_count = 0u;
    uint32_t register_count = 0u;

    CUSR_CONTRACT_CHECK(CUSR_AST_INSTRUCTION_TYPE_NONE == 0);
    CUSR_CONTRACT_CHECK(CUSR_AST_INSTRUCTION_TYPE_RETURN != 0);
    CUSR_CONTRACT_CHECK(cusr_ast_sass_generate(
        input_registers,
        1u,
        NULL,
        0u,
        10u,
        0u,
        30u,
        12u,
        0u,
        NULL,
        0u,
        missing_return_program,
        sass,
        8u,
        &sass_count,
        &register_count
    ) == CUSR_AST_SASS_ERROR_BAD_PROGRAM);

    return 1;
}

static int
cusr_contract_test_ast_prefix_only(void)
{
    const CusrAstInstruction program[] = {
        cusr_ast_encode_input(0u),
        cusr_ast_encode_return
    };
    const uint8_t input_registers[] = { 2u };
    CusrSassInstruction sass[8];
    const unsigned char* sass_bytes = (const unsigned char*)sass;
    size_t sass_count = 0u;
    uint32_t register_count = 0u;
    size_t i;

    memset(sass, 0x5a, sizeof(sass));
    CUSR_CONTRACT_CHECK(cusr_ast_sass_generate(
        input_registers,
        1u,
        NULL,
        0u,
        10u,
        0u,
        30u,
        12u,
        0u,
        NULL,
        0u,
        program,
        sass,
        8u,
        &sass_count,
        &register_count
    ) == CUSR_AST_SASS_SUCCESS);
    CUSR_CONTRACT_CHECK(sass_count == 1u);
    for (i = sass_count * sizeof(*sass); i < sizeof(sass); ++i) {
        CUSR_CONTRACT_CHECK(sass_bytes[i] == 0x5au);
    }

    return 1;
}

static int
cusr_contract_test_routine_registers(void)
{
    CusrAstInstruction safe_program[32];
    CusrAstInstruction direct_program[32];
    CusrAstInstruction identity_program[] = {
        cusr_ast_encode_input(0u),
        cusr_ast_encode_constant(1.0f),
        cusr_ast_encode_add,
        cusr_ast_encode_routine(0u, 1u),
        cusr_ast_encode_constant(2.0f),
        cusr_ast_encode_add,
        cusr_ast_encode_return
    };
    CusrAstInstruction ignored_mufu_arg_program[] = {
        cusr_ast_encode_input(0u),
        cusr_ast_encode_input(1u),
        cusr_ast_encode_sqrt,
        cusr_ast_encode_routine(0u, 2u),
        cusr_ast_encode_constant(1.0f),
        cusr_ast_encode_add,
        cusr_ast_encode_return
    };
    CusrSassInstruction sass[256];
    const uint8_t input_registers[] = { 2u };
    const uint8_t barrier_input_registers[] = { 2u, 3u };
    uint32_t safe_register_count = 0u;
    uint32_t direct_register_count = 0u;
    uint32_t identity_register_count = 0u;
    size_t sass_count = 0u;
    size_t i;

    CUSR_CONTRACT_CHECK(cusr_ast_encode_log.payload.idx != cusr_ast_encode_safe_log.payload.idx);
    CUSR_CONTRACT_CHECK(cusr_ast_encode_log10.payload.idx != cusr_ast_encode_safe_log10.payload.idx);
    CUSR_CONTRACT_CHECK(cusr_ast_encode_pow.payload.idx != cusr_ast_encode_safe_pow.payload.idx);

    safe_program[0] = cusr_ast_encode_input(0u);
    direct_program[0] = cusr_ast_encode_input(0u);
    for (i = 0u; i < 20u; ++i) {
        safe_program[i + 1u] = cusr_ast_encode_safe_sqrt;
        direct_program[i + 1u] = cusr_ast_encode_sqrt;
    }
    safe_program[21] = cusr_ast_encode_return;
    direct_program[21] = cusr_ast_encode_return;

    CUSR_CONTRACT_CHECK(cusr_ast_sass_generate(
        input_registers,
        1u,
        NULL,
        0u,
        10u,
        0u,
        30u,
        12u,
        0u,
        cusr_ast_routines,
        CUSR_AST_ROUTINES_COUNT,
        safe_program,
        sass,
        256u,
        &sass_count,
        &safe_register_count
    ) == CUSR_AST_SASS_SUCCESS);

    CUSR_CONTRACT_CHECK(cusr_ast_sass_generate(
        input_registers,
        1u,
        NULL,
        0u,
        10u,
        0u,
        30u,
        12u,
        0u,
        NULL,
        0u,
        direct_program,
        sass,
        256u,
        &sass_count,
        &direct_register_count
    ) == CUSR_AST_SASS_SUCCESS);

    CUSR_CONTRACT_CHECK(cusr_ast_sass_generate(
        input_registers,
        1u,
        NULL,
        0u,
        10u,
        0u,
        30u,
        12u,
        0u,
        cusr_contract_identity_routines,
        1u,
        identity_program,
        sass,
        256u,
        &sass_count,
        &identity_register_count
    ) == CUSR_AST_SASS_SUCCESS);

    CUSR_CONTRACT_CHECK(safe_register_count == 32u);
    CUSR_CONTRACT_CHECK(direct_register_count == 31u);
    CUSR_CONTRACT_CHECK(identity_register_count == 31u);

    CUSR_CONTRACT_CHECK(cusr_ast_sass_generate(
        barrier_input_registers,
        2u,
        NULL,
        0u,
        10u,
        0u,
        30u,
        12u,
        0u,
        cusr_contract_identity_routines,
        1u,
        ignored_mufu_arg_program,
        sass,
        256u,
        &sass_count,
        &identity_register_count
    ) == CUSR_AST_SASS_SUCCESS);
    CUSR_CONTRACT_CHECK((sass[0].word0 & 0xffffu) == CUSR_AST_SASS_OPCODE_MUFU);
    CUSR_CONTRACT_CHECK(((sass[1].word1 >> 52) & 0x1u) != 0u);

    return 1;
}

static void
cusr_contract_init_inspect(
    CusrSassInspectHandle* inspect,
    CusrSassInspectKernel* kernel,
    CusrSassInspectSite* site,
    CusrSassInspectRegcountRecord* regcount,
    unsigned char* cubin)
{
    size_t i;

    memset(inspect, 0, sizeof(*inspect));
    memset(kernel, 0, sizeof(*kernel));
    memset(site, 0, sizeof(*site));
    memset(regcount, 0, sizeof(*regcount));
    memset(cubin, 0, CUSR_CONTRACT_CUBIN_BYTES);
    memset(cubin, 0x5a, CUSR_CONTRACT_SITE_INSTRUCTIONS * CUSR_SASS_INSPECT_SASS_INSTRUCTION_BYTES);

    inspect->sass_arch = 120u;
    inspect->num_kernels = 1u;
    inspect->ast_capacity = 1u;
    inspect->kernels = kernel;
    inspect->num_sites = 1u;
    inspect->sites = site;
    inspect->num_regcount_records = 1u;
    inspect->regcount_records = regcount;

    kernel->kernel_index = 0u;
    kernel->first_regcount_record = 0u;
    kernel->num_regcount_records = 1u;
    kernel->first_site = 0u;
    kernel->num_sites = 1u;

    site->kernel_index = 0u;
    site->start_file_offset = 0u;
    site->end_file_offset = CUSR_CONTRACT_SITE_INSTRUCTIONS * CUSR_SASS_INSPECT_SASS_INSTRUCTION_BYTES;
    site->target_reg = 19u;
    site->output_regs[0] = 10u;
    site->num_output_regs = 1u;
    for (i = 0u; i < CUSR_SASS_INSPECT_INPUTS; ++i) {
        site->input_regs[i] = (uint8_t)(2u + i);
    }

    regcount->kernel_index = 0u;
    regcount->value_file_offset = CUSR_CONTRACT_REGCOUNT_OFFSET;
    regcount->value = 30u;
    cusr_contract_write_u32(cubin + CUSR_CONTRACT_REGCOUNT_OFFSET, 30u);
}

static int
cusr_contract_test_patcher(void)
{
    static const CusrAstInstruction heavy_program[] = {
        cusr_ast_encode_input(0u),
        cusr_ast_encode_constant(1.0f),
        cusr_ast_encode_add,
        cusr_ast_encode_return
    };
    static const CusrAstInstruction light_program[] = {
        cusr_ast_encode_input(0u),
        cusr_ast_encode_return
    };
    const CusrAstInstruction* extra_programs[] = { heavy_program, light_program };
    const CusrAstInstruction* heavy_programs[] = { heavy_program };
    const CusrAstInstruction* light_programs[] = { light_program };
    unsigned char cubin[CUSR_CONTRACT_CUBIN_BYTES];
    unsigned char before_light[CUSR_CONTRACT_CUBIN_BYTES];
    CusrSassInspectHandle inspect;
    CusrSassInspectKernel kernel;
    CusrSassInspectSite site;
    CusrSassInspectRegcountRecord regcount;
    CusrAstSassPatchStats stats;
    size_t i;
    size_t site_bytes = CUSR_CONTRACT_SITE_INSTRUCTIONS * CUSR_SASS_INSPECT_SASS_INSTRUCTION_BYTES;

    cusr_contract_init_inspect(&inspect, &kernel, &site, &regcount, cubin);

    CUSR_CONTRACT_CHECK(cusr_ast_sass_patch_cubin(
        &inspect, 12u, 0u, CUSR_AST_SASS_PATCH_EPILOGUE_SSE, NULL, 0u, extra_programs, 2u,
        cubin, sizeof(cubin), &stats
    ) == CUSR_AST_SASS_PATCH_ERROR_PROGRAM_COUNT);
    CUSR_CONTRACT_CHECK(cusr_contract_read_u32(cubin + CUSR_CONTRACT_REGCOUNT_OFFSET) == 30u);

    CUSR_CONTRACT_CHECK(cusr_ast_sass_patch_cubin(
        &inspect, 8u, 0u, CUSR_AST_SASS_PATCH_EPILOGUE_SSE, NULL, 0u, heavy_programs, 1u,
        cubin, sizeof(cubin), &stats
    ) == CUSR_AST_SASS_PATCH_ERROR_ARCH_MISMATCH);
    CUSR_CONTRACT_CHECK(cusr_contract_read_u32(cubin + CUSR_CONTRACT_REGCOUNT_OFFSET) == 30u);

    CUSR_CONTRACT_CHECK(cusr_ast_sass_patch_cubin(
        &inspect, 12u, 0u, CUSR_AST_SASS_PATCH_EPILOGUE_SSE, NULL, 0u, heavy_programs, 1u,
        cubin, sizeof(cubin), &stats
    ) == CUSR_AST_SASS_PATCH_SUCCESS);
    CUSR_CONTRACT_CHECK(stats.sites_patched == 1u);
    CUSR_CONTRACT_CHECK(stats.asts_patched == 1u);
    CUSR_CONTRACT_CHECK(stats.sass_bytes_written == stats.sass_instructions_written * CUSR_SASS_INSPECT_SASS_INSTRUCTION_BYTES);
    CUSR_CONTRACT_CHECK(stats.sass_bytes_written < site_bytes);
    for (i = stats.sass_bytes_written; i < site_bytes; ++i) {
        CUSR_CONTRACT_CHECK(cubin[i] == 0x5au);
    }
    CUSR_CONTRACT_CHECK(cusr_contract_read_u32(cubin + CUSR_CONTRACT_REGCOUNT_OFFSET) == 33u);

    memcpy(before_light, cubin, sizeof(cubin));

    CUSR_CONTRACT_CHECK(cusr_ast_sass_patch_cubin(
        &inspect, 12u, 0u, CUSR_AST_SASS_PATCH_EPILOGUE_SSE, NULL, 0u, light_programs, 1u,
        cubin, sizeof(cubin), &stats
    ) == CUSR_AST_SASS_PATCH_SUCCESS);
    CUSR_CONTRACT_CHECK(stats.sites_patched == 1u);
    CUSR_CONTRACT_CHECK(stats.asts_patched == 1u);
    CUSR_CONTRACT_CHECK(stats.sass_bytes_written == stats.sass_instructions_written * CUSR_SASS_INSPECT_SASS_INSTRUCTION_BYTES);
    CUSR_CONTRACT_CHECK(stats.sass_bytes_written < site_bytes);
    for (i = stats.sass_bytes_written; i < site_bytes; ++i) {
        CUSR_CONTRACT_CHECK(cubin[i] == before_light[i]);
    }
    CUSR_CONTRACT_CHECK(cusr_contract_read_u32(cubin + CUSR_CONTRACT_REGCOUNT_OFFSET) == 33u);
    return 1;
}

int
main(void)
{
    if (!cusr_contract_test_program_termination() ||
        !cusr_contract_test_ast_prefix_only() ||
        !cusr_contract_test_routine_registers() ||
        !cusr_contract_test_patcher()) {
        return 1;
    }

    printf("AST SASS contracts passed\n");
    return 0;
}
