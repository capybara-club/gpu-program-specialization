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
#include <stack_ptx_compiler.h>

#include <stack_ptx.h>
#include <stack_ptx_nng_client_descriptions.h>

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const char kDemoKernelPtx[] =
    ".version 7.0\n"
    ".target sm_52\n"
    ".address_size 64\n"
    "\n"
    ".visible .entry stack_ptx_nng_demo_kernel()\n"
    "{\n"
    "    {\n"
    "        .reg .f32 %_x0;\n"
    "        .reg .f32 %_x1;\n"
    "        .reg .f32 %_x2;\n"
    "\n"
    "        // Seed registers so the demo PTX is self-contained.\n"
    "        mov.f32 %_x0, 0f3f800000;\n"
    "        mov.f32 %_x1, 0f40000000;\n"
    "        mov.f32 %_x2, 0f00000000;\n"
    "\n"
    "        // PTX_INJECT_START func_0\n"
    "        // _x0 i f32 F32 x0\n"
    "        // _x1 i f32 F32 x1\n"
    "        // _x2 o f32 F32 y0\n"
    "        // PTX_INJECT_END\n"
    "    }\n"
    "    ret;\n"
    "}\n";

static void sleep_ms(int ms) {
    if (ms <= 0) {
        return;
    }
    struct timespec ts;
    ts.tv_sec = ms / 1000;
    ts.tv_nsec = (ms % 1000) * 1000000L;
    nanosleep(&ts, NULL);
}

int main(int argc, char** argv) {
    const char* addr = NULL;
    size_t num_jobs = 0;

    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (!arg || !arg[0]) {
            continue;
        }
        if ((strcmp(arg, "-h") == 0) || (strcmp(arg, "--help") == 0)) {
            fprintf(stderr, "usage: %s [addr] [--jobs N]\n", argv[0] ? argv[0] : "stack_ptx_nng_client");
            fprintf(stderr, "  addr may be tcp://HOST:PORT or HOST:PORT\n");
            return 0;
        }
        if (strcmp(arg, "--jobs") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "--jobs requires an argument\n");
                return 1;
            }
            char* end = NULL;
            unsigned long v = strtoul(argv[i + 1], &end, 10);
            if (!end || end == argv[i + 1] || *end != '\0' || v == 0) {
                fprintf(stderr, "invalid --jobs value: %s\n", argv[i + 1]);
                return 1;
            }
            num_jobs = (size_t)v;
            i += 1;
            continue;
        }
        if (arg[0] != '-' && !addr) {
            addr = arg;
            continue;
        }
        fprintf(stderr, "unknown argument: %s\n", arg);
        return 1;
    }

    if (addr && addr[0]) {
        const char* addr_to_set = addr;
        char normalized_addr[1024];
        if (strstr(addr, "://") == NULL) {
            const int n = snprintf(normalized_addr, sizeof(normalized_addr), "tcp://%s", addr);
            if (n < 0 || (size_t)n >= sizeof(normalized_addr)) {
                fprintf(stderr, "addr too long\n");
                return 1;
            }
            addr_to_set = normalized_addr;
        }
        if (setenv("STACK_PTX_NNG_ADDR", addr_to_set, 1) != 0) {
            fprintf(stderr, "setenv(STACK_PTX_NNG_ADDR) failed\n");
            return 1;
        }
    }

    StackPtxCompilerHandleConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.kernel_ptx = kDemoKernelPtx;
    cfg.kernel_name_format = "kernel_%06zu";
    cfg.kernel_num_kernels = 1;
    cfg.kernel_groups_per_kernel = 1;
    cfg.execution_limit = 100;
    cfg.max_results = 1;
    cfg.workspace_bytes = 1024 * 1024;
    cfg.stack_info = &stack_ptx_stack_info;
    cfg.compiler_info = NULL;
    cfg.inject_prefix = "func_";
    cfg.input_register_name = "x";
    cfg.output_register_prefix = "y";
    cfg.backend = STACK_PTX_COMPILER_BACKEND_NNG;
    cfg.requested_capabilities = num_jobs;

    StackPtxCompilerHandle* compiler = NULL;
    size_t capabilities = 0;
    size_t queue_slots = 0;
    StackPtxCompilerResult rc = stack_ptx_compiler_create(&cfg, &compiler, &capabilities, &queue_slots);
    if (rc != STACK_PTX_COMPILER_SUCCESS) {
        fprintf(stderr, "stack_ptx_compiler_create failed: %s\n", stack_ptx_compiler_result_to_string(rc));
        return 1;
    }

    fprintf(stderr, "capabilities=%zu queue_slots=%zu\n", capabilities, queue_slots);

    if (num_jobs == 0) {
        num_jobs = queue_slots;
        if (num_jobs == 0) {
            num_jobs = 1;
        }
        if (num_jobs > 32) {
            num_jobs = 32;
        }
    }

    enum { REGISTER_X0 = 0, REGISTER_X1 = 1 };

    static const StackPtxInstruction program[] = {
        stack_ptx_encode_input(REGISTER_X0),
        stack_ptx_encode_input(REGISTER_X1),
        stack_ptx_encode_ptx_instruction_add_ftz_f32,
        stack_ptx_encode_return,
    };

    StackPtxCompilerWork work;
    memset(&work, 0, sizeof(work));
    work.population = program;
    work.population_instructions = (size_t)(sizeof(program) / sizeof(program[0]));
    work.gene_length = work.population_instructions;
    work.module_idx = 0;
    work.job_id = 0;

    size_t submitted = 0;
    for (size_t i = 0; i < num_jobs; ++i) {
        work.job_id = (uint64_t)(i + 1);
        rc = stack_ptx_compiler_submit(compiler, &work);
        if (rc != STACK_PTX_COMPILER_SUCCESS) {
            fprintf(stderr, "submit job_id=%" PRIu64 " failed: %s\n", work.job_id, stack_ptx_compiler_result_to_string(rc));
            break;
        }
        submitted += 1;
    }
    if (submitted == 0) {
        stack_ptx_compiler_destroy(compiler);
        return 1;
    }

    size_t completed = 0;
    for (;;) {
        StackPtxCompilerOutput out;
        int has = 0;
        rc = stack_ptx_compiler_poll(compiler, &out, &has);
        if (rc != STACK_PTX_COMPILER_SUCCESS) {
            fprintf(stderr, "stack_ptx_compiler_poll failed: %s\n", stack_ptx_compiler_result_to_string(rc));
            break;
        }
        if (!has) {
            sleep_ms(10);
            continue;
        }

        fprintf(stderr, "job_id=%" PRIu64 " status=%s compile_ms=%.3f cubin_size=%zu\n", out.job_id,
            stack_ptx_compiler_result_to_string(out.status), out.compile_ms, out.cubin_size);

        if (out.cubin) {
            free(out.cubin);
        }
        completed += 1;
        if (completed >= submitted) {
            break;
        }
    }

    stack_ptx_compiler_destroy(compiler);
    return 0;
}
