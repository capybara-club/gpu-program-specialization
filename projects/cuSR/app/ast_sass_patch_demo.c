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
#include <cusr_ast_sass_patch.h>

#include <stdio.h>
#include <string.h>

#define CUSR_AST_SASS_PATCH_DEMO_CUBIN_BYTES 512u
#define CUSR_AST_SASS_PATCH_DEMO_REGCOUNT_OFFSET 32u
#define CUSR_AST_SASS_PATCH_DEMO_SITE_OFFSET 128u
#define CUSR_AST_SASS_PATCH_DEMO_SITE_INSTRUCTIONS 8u
#define CUSR_AST_SASS_PATCH_DEMO_PROGRAM_STRIDE 8u

static const CusrAstInstruction cusr_ast_sass_patch_demo_routine[] = {
    cusr_ast_encode_routine_arg(0u),
    cusr_ast_encode_routine_arg(1u),
    cusr_ast_encode_add,
    cusr_ast_encode_return
};

static const CusrAstInstruction* const cusr_ast_sass_patch_demo_routines[] = {
    cusr_ast_sass_patch_demo_routine
};

static const CusrAstInstruction cusr_ast_sass_patch_demo_programs[CUSR_AST_SASS_PATCH_DEMO_PROGRAM_STRIDE] = {
    cusr_ast_encode_input(0u),
    cusr_ast_encode_input(1u),
    cusr_ast_encode_routine(0u, 2u),
    cusr_ast_encode_return,
    cusr_ast_encode_return,
    cusr_ast_encode_return,
    cusr_ast_encode_return,
    cusr_ast_encode_return
};

static const CusrAstInstruction* const cusr_ast_sass_patch_demo_program_ptrs[] = {
    cusr_ast_sass_patch_demo_programs
};

static unsigned
cusr_ast_sass_patch_demo_read_u32(const unsigned char* bytes)
{
    return
        ((unsigned)bytes[0]) |
        ((unsigned)bytes[1] << 8) |
        ((unsigned)bytes[2] << 16) |
        ((unsigned)bytes[3] << 24);
}

static unsigned long long
cusr_ast_sass_patch_demo_read_u64(const unsigned char* bytes)
{
    return
        ((unsigned long long)bytes[0]) |
        ((unsigned long long)bytes[1] << 8) |
        ((unsigned long long)bytes[2] << 16) |
        ((unsigned long long)bytes[3] << 24) |
        ((unsigned long long)bytes[4] << 32) |
        ((unsigned long long)bytes[5] << 40) |
        ((unsigned long long)bytes[6] << 48) |
        ((unsigned long long)bytes[7] << 56);
}

int
main(void)
{
    unsigned char cubin[CUSR_AST_SASS_PATCH_DEMO_CUBIN_BYTES];
    CusrSassInspectKernel kernel;
    CusrSassInspectSite site;
    CusrSassInspectRegcountRecord regcount;
    CusrSassInspectHandle inspect;
    CusrAstSassPatchResult patch_result;

    memset(cubin, 0, sizeof(cubin));
    memset(&kernel, 0, sizeof(kernel));
    memset(&site, 0, sizeof(site));
    memset(&regcount, 0, sizeof(regcount));
    memset(&inspect, 0, sizeof(inspect));

    cubin[CUSR_AST_SASS_PATCH_DEMO_REGCOUNT_OFFSET + 0u] = 30u;
    cubin[CUSR_AST_SASS_PATCH_DEMO_REGCOUNT_OFFSET + 1u] = 0u;
    cubin[CUSR_AST_SASS_PATCH_DEMO_REGCOUNT_OFFSET + 2u] = 0u;
    cubin[CUSR_AST_SASS_PATCH_DEMO_REGCOUNT_OFFSET + 3u] = 0u;

    kernel.kernel_index = 0u;
    kernel.first_regcount_record = 0u;
    kernel.num_regcount_records = 1u;
    kernel.first_site = 0u;
    kernel.num_sites = 1u;

    regcount.kernel_index = 0u;
    regcount.value_file_offset = CUSR_AST_SASS_PATCH_DEMO_REGCOUNT_OFFSET;
    regcount.value = 30u;

    site.kernel_index = 0u;
    site.occurrence_index = 0u;
    site.start_file_offset = CUSR_AST_SASS_PATCH_DEMO_SITE_OFFSET;
    site.end_file_offset =
        CUSR_AST_SASS_PATCH_DEMO_SITE_OFFSET +
        CUSR_AST_SASS_PATCH_DEMO_SITE_INSTRUCTIONS * CUSR_SASS_INSPECT_SASS_INSTRUCTION_BYTES;
    site.input_regs[0] = 6u;
    site.input_regs[1] = 12u;
    site.input_regs[2] = 15u;
    site.input_regs[3] = 14u;
    site.input_regs[4] = 13u;
    site.input_regs[5] = 16u;
    site.input_regs[6] = 17u;
    site.input_regs[7] = 18u;
    site.target_reg = 19u;
    site.output_regs[0] = 10u;
    site.num_output_regs = 1u;
    site.incoming_wait_mask = 0x01cu;

    inspect.cubin = cubin;
    inspect.cubin_size = sizeof(cubin);
    inspect.sass_arch = 120u;
    inspect.num_kernels = 1u;
    inspect.ast_capacity = 1u;
    inspect.kernels = &kernel;
    inspect.num_sites = 1u;
    inspect.sites = &site;
    inspect.num_regcount_records = 1u;
    inspect.regcount_records = &regcount;

    patch_result = cusr_ast_sass_patch_cubin(
        &inspect,
        12u,
        0u,
        CUSR_AST_SASS_PATCH_EPILOGUE_SSE,
        cusr_ast_sass_patch_demo_routines,
        1u,
        cusr_ast_sass_patch_demo_program_ptrs,
        1u,
        cubin,
        sizeof(cubin),
        NULL
    );

    if (patch_result != CUSR_AST_SASS_PATCH_SUCCESS) {
        fprintf(
            stderr,
            "patch failed: %s\n",
            cusr_ast_sass_patch_result_to_string(patch_result)
        );
        return 1;
    }

    if (cusr_ast_sass_patch_demo_read_u64(cubin + CUSR_AST_SASS_PATCH_DEMO_SITE_OFFSET) == 0u) {
        fprintf(stderr, "site was not patched\n");
        return 1;
    }

    printf(
        "cubin_regcount=%u\n",
        cusr_ast_sass_patch_demo_read_u32(cubin + CUSR_AST_SASS_PATCH_DEMO_REGCOUNT_OFFSET)
    );

    return 0;
}
