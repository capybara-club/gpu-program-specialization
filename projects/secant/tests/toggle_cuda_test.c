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
#include "toggle_fixture.h"
#include <cuda.h>
#include <nvrtc.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define REQUIRE(c)                                                                                           \
    do {                                                                                                     \
        if (!(c)) {                                                                                          \
            fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #c);                                          \
            exit(1);                                                                                         \
        }                                                                                                    \
    } while (0)
#define CUDA(c)                                                                                              \
    do {                                                                                                     \
        CUresult e = (c);                                                                                    \
        if (e != CUDA_SUCCESS) {                                                                             \
            const char *name = NULL;                                                                         \
            cuGetErrorName(e, &name);                                                                        \
            fprintf(stderr, "line %d: %s: %s\n", __LINE__, #c, name ? name : "CUDA error");                  \
            exit(1);                                                                                         \
        }                                                                                                    \
    } while (0)
#define SEC(c)                                                                                               \
    do {                                                                                                     \
        SecantResult e = (c);                                                                                \
        if (e != SECANT_SUCCESS) {                                                                           \
            fprintf(stderr, "line %d: %s: %s\n", __LINE__, #c, secant_result_to_string(e));                  \
            exit(1);                                                                                         \
        }                                                                                                    \
    } while (0)
typedef struct TestRunner {
    void *cubin, *storage;
    size_t bytes;
    SecantCubinPlan *plan;
    SecantCubinRunner runner;
} TestRunner;
static TestRunner build(const SecantCubinRecipeHeader *recipe) {
    TestRunner out;
    char *source, *log;
    size_t size, logsize;
    nvrtcProgram program;
    nvrtcResult status;
    SecantCubinRunnerOptions options = secant_cubin_runner_options_init();
    char architecture[48];
    CUdevice device;
    int major, minor;
    const char *flags[] = {architecture, "--std=c++14", "--use_fast_math", "--fmad=false"};
    CUDA(cuCtxGetDevice(&device));
    CUDA(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    CUDA(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
    snprintf(architecture, sizeof(architecture), "--gpu-architecture=sm_%d%d", major, minor);
    memset(&out, 0, sizeof(out));
    SEC(secant_cubin_source_size(recipe, &size));
    source = (char *)malloc(size);
    REQUIRE(source);
    SEC(secant_cubin_source_write(recipe, source, size));
    REQUIRE(nvrtcCreateProgram(&program, source, "secant_toggle.cu", 0, NULL, NULL) == NVRTC_SUCCESS);
    status = nvrtcCompileProgram(program, 4, flags);
    REQUIRE(nvrtcGetProgramLogSize(program, &logsize) == NVRTC_SUCCESS);
    log = (char *)malloc(logsize);
    REQUIRE(log);
    REQUIRE(nvrtcGetProgramLog(program, log) == NVRTC_SUCCESS);
    if (status != NVRTC_SUCCESS)
        fprintf(stderr, "%s\n", log);
    REQUIRE(status == NVRTC_SUCCESS);
    free(log);
    REQUIRE(nvrtcGetCUBINSize(program, &out.bytes) == NVRTC_SUCCESS);
    out.cubin = malloc(out.bytes);
    REQUIRE(out.cubin);
    REQUIRE(nvrtcGetCUBIN(program, (char *)out.cubin) == NVRTC_SUCCESS);
    nvrtcDestroyProgram(&program);
    free(source);
    if (getenv("SECANT_TEST_DUMP_CUBIN")) {
        FILE *f = fopen(getenv("SECANT_TEST_DUMP_CUBIN"), "wb");
        REQUIRE(f);
        REQUIRE(fwrite(out.cubin, 1, out.bytes, f) == out.bytes);
        REQUIRE(fclose(f) == 0);
    }
    SEC(secant_cubin_plan_storage_size(recipe, out.cubin, out.bytes, &size));
    out.storage = malloc(size);
    REQUIRE(out.storage);
    SEC(secant_cubin_plan_init(recipe, out.cubin, out.bytes, out.storage, size, &out.plan));
    options.num_workers = 3;
    options.num_streams = 2;
    SEC(secant_cubin_runner_create(out.plan, out.cubin, out.bytes, &options, &out.runner));
    return out;
}
static void destroy(TestRunner *r) {
    SEC(secant_cubin_runner_destroy(r->runner));
    free(r->storage);
    free(r->cubin);
}
static void report_registers(const TestRunner *r, const SecantCubinToggleSSERecipe *recipe,
                             const SecantAstProgramSet *programs) {
    SecantAstProgramSet first = *programs;
    size_t capacity = recipe->num_kernels * recipe->asts_per_kernel, i;
    void *patched = malloc(r->bytes);
    CUmodule module;
    int max_registers = 0, max_local_bytes = 0;
    REQUIRE(patched);
    if (first.asts.count > capacity)
        first.asts.count = capacity;
    memcpy(patched, r->cubin, r->bytes);
    SEC(secant_cubin_specialize_into(r->plan, &first, patched, r->bytes));
    CUDA(cuModuleLoadData(&module, patched));
    for (i = 0; i * recipe->asts_per_kernel < first.asts.count; ++i) {
        char name[96];
        CUfunction function;
        int registers, local_bytes;
        snprintf(name, sizeof(name), "secant_cubin_toggle_sse_%03zu", i);
        CUDA(cuModuleGetFunction(&function, module, name));
        CUDA(cuFuncGetAttribute(&registers, CU_FUNC_ATTRIBUTE_NUM_REGS, function));
        CUDA(cuFuncGetAttribute(&local_bytes, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, function));
        if (registers > max_registers)
            max_registers = registers;
        if (local_bytes > max_local_bytes)
            max_local_bytes = local_bytes;
    }
    printf("specialized resources (packed=%zu, constants=%zu): max %d registers/thread, %d local bytes/thread\n",
           recipe->asts_per_kernel, recipe->num_constants, max_registers, max_local_bytes);
    CUDA(cuModuleUnload(module));
    free(patched);
}
static CUdeviceptr upload(const float *data, size_t count) {
    CUdeviceptr d;
    CUDA(cuMemAlloc(&d, count * sizeof(float)));
    CUDA(cuMemcpyHtoD(d, data, count * sizeof(float)));
    return d;
}
static void compare(const float *a, const float *b, size_t count) {
    size_t i;
    for (i = 0; i < count; ++i)
        if (!isfinite(a[i]) || fabsf(a[i] - b[i]) > 2e-4f * (1 + fabsf(b[i]))) {
            fprintf(stderr, "mismatch %zu: GPU %.9g CPU %.9g\n", i, a[i], b[i]);
            exit(1);
        }
}
static void toggle_test(size_t packed) {
    enum { rows = 257, nb = 13, ncfg = 104, ld = 109, na = 7, nt = 2, bank_elements = (nb - 1) * 3 + 2 };
    float input[3 * rows], targets[nt * rows], banks[bank_elements], expected[na * nt * ld], actual[na * nt * ld];
    CUdeviceptr di, dt, db, do1, do2;
    size_t i;
    int pass;
    SecantCubinToggleSSERecipe recipe = secant_cubin_toggle_sse_recipe_init();
    SecantCpuToggleSSERun cpu = secant_cpu_toggle_sse_run_init();
    SecantCubinToggleSSERun gpu = secant_cubin_toggle_sse_run_init();
    SecantRunnerStats stats = secant_runner_stats_init();
    TestRunner runner;
    for (i = 0; i < 3 * rows; ++i)
        input[i] = (float)i * .0013f - .3f;
    for (i = 0; i < nt * rows; ++i)
        targets[i] = (float)i * .0007f;
    for (i = 0; i < bank_elements; ++i)
        banks[i] = (float)i * .01f - .6f;
    for (i = 0; i < na * nt * ld; ++i)
        actual[i] = expected[i] = -123.f;
    di = upload(input, 3 * rows);
    dt = upload(targets, nt * rows);
    db = upload(banks, bank_elements);
    do1 = upload(actual, na * nt * ld);
    do2 = upload(actual, na * nt * ld);
    recipe.num_kernels = 3;
    recipe.asts_per_kernel = packed;
    recipe.num_inputs = 3;
    recipe.num_constants = 2;
    recipe.num_targets = nt;
    recipe.tile_rows = 31;
    recipe.threads_per_block = 128;
    recipe.patch_capacity_instructions = 256;
    runner = build(&recipe.header);
    cpu.programs.asts.items = toggle_asts;
    cpu.programs.asts.count = na;
    cpu.num_inputs = 3;
    cpu.num_constants = 2;
    cpu.num_targets = nt;
    cpu.num_rows = rows;
    cpu.num_banks = nb;
    cpu.toggle_bits = 3;
    cpu.input.data = input;
    cpu.input.num_elements = 3 * rows;
    cpu.input.leading_dimension = rows;
    cpu.targets.data = targets;
    cpu.targets.num_elements = nt * rows;
    cpu.targets.leading_dimension = rows;
    cpu.constants.data = banks;
    cpu.constants.num_elements = bank_elements;
    cpu.constants.bank_stride = 3;
    cpu.output.data = expected;
    cpu.output.num_elements = na * nt * ld;
    cpu.output.leading_dimension = ld;
    gpu.programs = cpu.programs;
    gpu.num_targets = nt;
    gpu.num_rows = rows;
    gpu.num_banks = nb;
    gpu.toggle_bits = 3;
    gpu.input.address = di;
    gpu.input.num_elements = 3 * rows;
    gpu.input.leading_dimension = rows;
    gpu.targets.address = dt;
    gpu.targets.num_elements = nt * rows;
    gpu.targets.leading_dimension = rows;
    gpu.constants.address = db;
    gpu.constants.num_elements = bank_elements;
    gpu.constants.bank_stride = 3;
    gpu.output.address = do1;
    gpu.output.num_elements = na * nt * ld;
    gpu.output.leading_dimension = ld;
    for (pass = 0; pass < 2; ++pass) {
        SecantCubinToggleSSERun second;
        const SecantCubinRunHeader *batch[2];
        if (pass) {
            for (i = 0; i < bank_elements; ++i)
                banks[i] += .137f;
            CUDA(cuMemcpyHtoD(db, banks, sizeof(banks)));
        }
        SEC(secant_cpu_run_toggle_sse(&cpu));
        second = gpu;
        second.output.address = do2;
        batch[0] = &gpu.header;
        batch[1] = &second.header;
        SEC(secant_cubin_runner_run_batch(runner.runner, batch, 2, &stats));
        REQUIRE(stats.modules_loaded == (7 + 3 * packed - 1) / (3 * packed) && stats.num_asts == 7);
        CUDA(cuMemcpyDtoH(actual, do1, sizeof(actual)));
        compare(actual, expected, na * nt * ld);
        CUDA(cuMemcpyDtoH(actual, do2, sizeof(actual)));
        compare(actual, expected, na * nt * ld);
        printf("toggle packed=%zu repeat=%d: 7 ASTs x 13 banks x 8 permutations x 257 rows x 2 targets; "
               "GPU %.6fs total %.6fs\n",
               packed, pass, stats.runtime_seconds, stats.total_seconds);
        second.output.address = di;
        REQUIRE(secant_cubin_runner_run_batch(runner.runner, batch, 2, &stats) ==
                SECANT_ERROR_INVALID_VALUE);
        REQUIRE(stats.modules_loaded == 0);
    }
    {
        float second_expected[na * nt * ld];
        SecantCpuToggleSSERun second_cpu = cpu;
        SecantCubinToggleSSERun second = gpu;
        const SecantCubinRunHeader *batch[2] = {&gpu.header, &second.header};
        for (i = 0; i < na * nt * ld; ++i)
            second_expected[i] = -123.f;
        CUDA(cuMemcpyHtoD(do2, second_expected, sizeof(second_expected)));
        second_cpu.num_rows = second.num_rows = 129;
        second_cpu.num_banks = second.num_banks = nb - 1;
        second_cpu.constants.data += 3;
        second.constants.address += 3 * sizeof(float);
        second_cpu.constants.num_elements -= 3;
        second.constants.num_elements -= 3;
        second_cpu.output.data = second_expected;
        second.output.address = do2;
        SEC(secant_cpu_run_toggle_sse(&second_cpu));
        SEC(secant_cubin_runner_run_batch(runner.runner, batch, 2, &stats));
        CUDA(cuMemcpyDtoH(actual, do1, sizeof(actual)));
        compare(actual, expected, na * nt * ld);
        CUDA(cuMemcpyDtoH(actual, do2, sizeof(actual)));
        compare(actual, second_expected, na * nt * ld);
    }
    for (gpu.header.version = 1; gpu.header.version < 3; ++gpu.header.version) {
        REQUIRE(secant_cubin_runner_run_toggle_sse(runner.runner, &gpu, &stats) ==
                SECANT_ERROR_UNSUPPORTED_VERSION);
        REQUIRE(stats.modules_loaded == 0);
    }
    gpu.header.version = SECANT_CUBIN_RUN_VERSION_3;
    gpu.toggle_bits = 2;
    REQUIRE(secant_cubin_runner_run_toggle_sse(runner.runner, &gpu, &stats) == SECANT_ERROR_BAD_PROGRAM);
    gpu.toggle_bits = 3;
    {
        const uint8_t bad[] = {COL(0), secant_ast_encode_add_f32, RET};
        const uint8_t *asts[] = {bad};
        gpu.programs.asts.items = asts;
        gpu.programs.asts.count = 1;
        REQUIRE(secant_cubin_runner_run_toggle_sse(runner.runner, &gpu, &stats) ==
                SECANT_ERROR_STACK_UNDERFLOW);
        gpu.programs = cpu.programs;
        SEC(secant_cubin_runner_run_toggle_sse(runner.runner, &gpu, &stats));
        CUDA(cuMemcpyDtoH(actual, do1, sizeof(actual)));
        compare(actual, expected, na * nt * ld);
    }
    destroy(&runner);
    CUDA(cuMemFree(di));
    CUDA(cuMemFree(dt));
    CUDA(cuMemFree(db));
    CUDA(cuMemFree(do1));
    CUDA(cuMemFree(do2));
}
static void static_test(void) {
    enum { rows = 257, na = 13, nt = 2 };
    size_t i;
    float input[3 * rows], targets[2 * rows], cpu_out[na * rows], gpu_out[na * rows];
    const uint8_t add[] = {COL(0), COL(1), secant_ast_encode_add_f32, RET};
    const uint8_t mul[] = {COL(1), COL(2), secant_ast_encode_mul_f32, RET};
    const uint8_t *asts[na];
    CUdeviceptr di, dt, dout;
    int shape;
    for (i = 0; i < 3 * rows; ++i)
        input[i] = (float)i * .003f - .5f;
    for (i = 0; i < 2 * rows; ++i)
        targets[i] = (float)i * .001f;
    for (i = 0; i < na; ++i)
        asts[i] = (i & 1) ? add : mul;
    memset(gpu_out, 0, sizeof(gpu_out));
    di = upload(input, 3 * rows);
    dt = upload(targets, 2 * rows);
    dout = upload(gpu_out, na * rows);
    for (shape = 0; shape < 4; ++shape) {
        TestRunner r;
        SecantAstProgramSet programs;
        size_t width, output_count = na;
        SecantCubinMaterializeRecipe mr = secant_cubin_materialize_recipe_init();
        SecantCubinSSERecipe sr = secant_cubin_sse_recipe_init();
        SecantCubinAffineStatsRecipe ar = secant_cubin_affine_stats_recipe_init();
        SecantCubinGramStatsRecipe gr = secant_cubin_gram_stats_recipe_init();
        SecantCubinMaterializeRun mg = secant_cubin_materialize_run_init();
        SecantCpuMaterializeRun mc = secant_cpu_materialize_run_init();
        SecantCubinSSERun sg = secant_cubin_sse_run_init();
        SecantCpuSSERun sc = secant_cpu_sse_run_init();
        SecantCubinAffineStatsRun ag = secant_cubin_affine_stats_run_init();
        SecantCpuAffineStatsRun ac = secant_cpu_affine_stats_run_init();
        SecantCubinGramStatsRun gg = secant_cubin_gram_stats_run_init();
        SecantCpuGramStatsRun gc = secant_cpu_gram_stats_run_init();
        memset(&programs, 0, sizeof(programs));
        programs.asts.items = asts;
        programs.asts.count = na;
        memset(cpu_out, 0, sizeof(cpu_out));
        memset(gpu_out, 0, sizeof(gpu_out));
        CUDA(cuMemcpyHtoD(dout, gpu_out, sizeof(gpu_out)));
#define RECIPE(v)                                                                                            \
    do {                                                                                                     \
        (v).num_kernels = 2;                                                                                 \
        (v).asts_per_kernel = 3;                                                                             \
        (v).num_inputs = 3;                                                                                  \
        (v).patch_capacity_instructions = 256;                                                               \
    } while (0)
#define SCORE(v)                                                                                             \
    do {                                                                                                     \
        RECIPE(v);                                                                                           \
        (v).num_targets = nt;                                                                                \
        (v).tile_rows = 128;                                                                                 \
        (v).threads_per_block = 128;                                                                         \
    } while (0)
        RECIPE(mr);
        SCORE(sr);
        SCORE(ar);
        SCORE(gr);
        if (shape == 0)
            width = rows;
        else if (shape == 1)
            width = nt;
        else if (shape == 2)
            width = 2 + nt;
        else {
            width = 3 * (1 + 3 + nt);
            output_count = (na + 2) / 3;
        }
#define RUN(g, c, outmember)                                                                                 \
    do {                                                                                                     \
        (g).programs = (c).programs = programs;                                                              \
        (g).num_rows = (c).num_rows = rows;                                                                  \
        (c).num_inputs = 3;                                                                                  \
        (g).input.address = di;                                                                              \
        (g).input.num_elements = (c).input.num_elements = 3 * rows;                                          \
        (g).input.leading_dimension = (c).input.leading_dimension = rows;                                    \
        (c).input.data = input;                                                                              \
        (g).outmember.address = dout;                                                                        \
        (c).outmember.data = cpu_out;                                                                        \
        (g).outmember.num_elements = (c).outmember.num_elements = na * rows;                                 \
        (g).outmember.leading_dimension = (c).outmember.leading_dimension = width;                           \
    } while (0)
#define TARGET(g, c)                                                                                         \
    do {                                                                                                     \
        (g).num_targets = (c).num_targets = nt;                                                              \
        (g).targets.address = dt;                                                                            \
        (c).targets.data = targets;                                                                          \
        (g).targets.num_elements = (c).targets.num_elements = nt * rows;                                     \
        (g).targets.leading_dimension = (c).targets.leading_dimension = rows;                                \
    } while (0)
        if (shape == 0) {
            RUN(mg, mc, output);
            r = build(&mr.header);
            SEC(secant_cpu_run(&mc.header));
            SEC(secant_cubin_runner_run(r.runner, &mg.header, NULL));
        } else if (shape == 1) {
            RUN(sg, sc, output);
            TARGET(sg, sc);
            r = build(&sr.header);
            SEC(secant_cpu_run(&sc.header));
            SEC(secant_cubin_runner_run(r.runner, &sg.header, NULL));
        } else if (shape == 2) {
            RUN(ag, ac, ast_stats);
            TARGET(ag, ac);
            r = build(&ar.header);
            SEC(secant_cpu_run(&ac.header));
            SEC(secant_cubin_runner_run(r.runner, &ag.header, NULL));
        } else {
            RUN(gg, gc, statistics);
            TARGET(gg, gc);
            gc.asts_per_cohort = 3;
            r = build(&gr.header);
            SEC(secant_cpu_run(&gc.header));
            SEC(secant_cubin_runner_run(r.runner, &gg.header, NULL));
        }
        CUDA(cuMemcpyDtoH(gpu_out, dout, sizeof(gpu_out)));
        compare(gpu_out, cpu_out, width * output_count);
        destroy(&r);
        printf("static shape %d passed\n", shape);
    }
    CUDA(cuMemFree(di));
    CUDA(cuMemFree(dt));
    CUDA(cuMemFree(dout));
}
static void edge_test(void) {
    float targets[9] = {0, .1f, .2f, .3f, .4f, .5f, .6f, .7f, .8f}, expected[2], actual[2],
          banks[2] = {.7f, -.3f};
    const uint8_t literal[] = {secant_ast_encode_constant_f32_bits(SECANT_F32_BITS_TWO), RET};
    const uint8_t *asts[] = {literal};
    CUdeviceptr dt = upload(targets, 9), dout = upload(banks, 2), db = upload(banks, 2);
    SecantCubinToggleSSERecipe recipe = secant_cubin_toggle_sse_recipe_init();
    SecantCpuToggleSSERun cpu = secant_cpu_toggle_sse_run_init();
    SecantCubinToggleSSERun gpu = secant_cubin_toggle_sse_run_init();
    TestRunner r;
    recipe.num_kernels = 1;
    recipe.num_targets = 1;
    recipe.tile_rows = 8;
    recipe.threads_per_block = 64;
    recipe.patch_capacity_instructions = 128;
    cpu.programs.asts.items = asts;
    cpu.programs.asts.count = 1;
    cpu.num_rows = 9;
    cpu.num_targets = 1;
    cpu.num_banks = 1;
    cpu.targets.data = targets;
    cpu.targets.num_elements = 9;
    cpu.targets.leading_dimension = 9;
    cpu.output.data = expected;
    cpu.output.num_elements = 2;
    cpu.output.leading_dimension = 2;
    gpu.programs = cpu.programs;
    gpu.num_rows = 9;
    gpu.num_targets = 1;
    gpu.num_banks = 1;
    gpu.targets.address = dt;
    gpu.targets.num_elements = 9;
    gpu.targets.leading_dimension = 9;
    gpu.output.address = dout;
    gpu.output.num_elements = 2;
    gpu.output.leading_dimension = 2;
    r = build(&recipe.header);
    SEC(secant_cpu_run_toggle_sse(&cpu));
    SEC(secant_cubin_runner_run_toggle_sse(r.runner, &gpu, NULL));
    CUDA(cuMemcpyDtoH(actual, dout, sizeof(actual)));
    compare(actual, expected, 1);
    destroy(&r);
    {
        const uint8_t routine[] = {BANK(0), BANK(1), TOG(0), RET};
        const uint8_t caller[] = {secant_ast_encode_routine_f32(0), RET};
        const uint8_t *routines[] = {routine};
        asts[0] = caller;
        recipe.num_constants = 2;
        cpu.num_constants = 2;
        cpu.toggle_bits = gpu.toggle_bits = 1;
        cpu.programs.routines.items = routines;
        cpu.programs.routines.count = 1;
        gpu.programs = cpu.programs;
        cpu.constants.data = banks;
        cpu.constants.num_elements = 2;
        cpu.constants.bank_stride = 2;
        gpu.constants.address = db;
        gpu.constants.num_elements = 2;
        gpu.constants.bank_stride = 2;
        r = build(&recipe.header);
        SEC(secant_cpu_run_toggle_sse(&cpu));
        SEC(secant_cubin_runner_run_toggle_sse(r.runner, &gpu, NULL));
        CUDA(cuMemcpyDtoH(actual, dout, sizeof(actual)));
        compare(actual, expected, 2);
        /* Full enumeration of bit 31 needs 2^32 outputs. Check its mask encoding
         * through specialization without allocating that enormous result matrix. */
        {
            const uint8_t high[] = {BANK(0), BANK(1), TOG(31), RET};
            const uint8_t *high_asts[] = {high};
            SecantAstProgramSet programs = cpu.programs;
            size_t i;
            int found = 0;
            unsigned char *patched = (unsigned char *)malloc(r.bytes);
            REQUIRE(patched);
            memcpy(patched, r.cubin, r.bytes);
            programs.asts.items = high_asts;
            SEC(secant_cubin_specialize_into(r.plan, &programs, patched, r.bytes));
            for (i = 0; i + 16 <= r.bytes; i += 4)
                if (patched[i] == 0x12 && patched[i + 1] == 0x78 && patched[i + 2] == 0xff &&
                    patched[i + 4] == 0 && patched[i + 5] == 0 && patched[i + 6] == 0 &&
                    patched[i + 7] == 0x80)
                    found = 1;
            REQUIRE(found);
            free(patched);
        }
        destroy(&r);
    }
    CUDA(cuMemFree(dt));
    CUDA(cuMemFree(dout));
    CUDA(cuMemFree(db));
    puts("zero columns/constants/toggles, routine propagation, bit 31 encoding passed");
}
static unsigned test_random(unsigned *state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}
static void random_test(size_t packed) {
    enum { na = 129, rows = 97, cols = 8, nconst = 8, banks_count = 5, bits = 8, configs = 1280 };
    uint8_t programs[na][512];
    const uint8_t *asts[na];
    float input[cols * rows], target[rows], banks[banks_count * nconst];
    float *expected, *actual;
    size_t a, i;
    unsigned seed = 424242;
    SecantCpuToggleSSERun cpu = secant_cpu_toggle_sse_run_init();
    SecantCubinToggleSSERun gpu = secant_cubin_toggle_sse_run_init();
    SecantCubinToggleSSERecipe recipe = secant_cubin_toggle_sse_recipe_init();
    SecantRunnerStats stats = secant_runner_stats_init();
    TestRunner r;
    CUdeviceptr di, dt, db, dout;
    for (a = 0; a < na; ++a) {
        size_t n = 0;
        unsigned term;
        asts[a] = programs[a];
        for (term = 0; term < 5; ++term) {
            unsigned four = (test_random(&seed) >> 8) & 1, leaf;
            for (leaf = 0; leaf < (four ? 4u : 2u); ++leaf) {
                unsigned pick = test_random(&seed);
                if ((pick & 7u) == 0) {
                    secant_ast_affine_bank_write(programs[a]+n, (uint8_t)((pick >> 8) % nconst),
                        (float)((int)(pick % 7)-3)*.25f, (float)((int)(pick % 13)-6)*.1f);
                    n += 10;
                    continue;
                }
                programs[a][n++] = ((pick >> 8) & 1) ? SECANT_AST_INSTRUCTION_TYPE_COLUMN_F32
                                                     : SECANT_AST_INSTRUCTION_TYPE_BANK_CONSTANT_F32;
                programs[a][n++] = (uint8_t)((pick >> 12) % 8);
            }
            programs[a][n++] =
                four ? SECANT_AST_INSTRUCTION_TYPE_TOGGLE4_F32 : SECANT_AST_INSTRUCTION_TYPE_TOGGLE2_F32;
            {
                unsigned bit = (test_random(&seed) >> 8) % bits;
                programs[a][n++] = (uint8_t)bit;
                if (four)
                    programs[a][n++] = (uint8_t)((bit + 1 + (test_random(&seed) >> 8) % 7) % bits);
            }
            if (term & 1)
                programs[a][n++] = SECANT_AST_INSTRUCTION_TYPE_SIN_F32;
            if (term)
                programs[a][n++] =
                    (term & 1) ? SECANT_AST_INSTRUCTION_TYPE_MUL_F32 : SECANT_AST_INSTRUCTION_TYPE_ADD_F32;
        }
        programs[a][n++] = SECANT_AST_INSTRUCTION_TYPE_RETURN_F32;
        REQUIRE(n < sizeof(programs[a]));
    }
    for (i = 0; i < cols * rows; ++i)
        input[i] = (float)((int)(test_random(&seed) % 1000) - 500) * .001f;
    for (i = 0; i < rows; ++i)
        target[i] = (float)i * .003f;
    for (i = 0; i < banks_count * nconst; ++i)
        banks[i] = (float)((int)(test_random(&seed) % 1000) - 500) * .001f;
    expected = (float *)calloc(na * configs, sizeof(float));
    actual = (float *)calloc(na * configs, sizeof(float));
    REQUIRE(expected && actual);
    di = upload(input, cols * rows);
    dt = upload(target, rows);
    db = upload(banks, banks_count * nconst);
    dout = upload(actual, na * configs);
    cpu.programs.asts.items = asts;
    cpu.programs.asts.count = na;
    cpu.num_inputs = cols;
    cpu.num_constants = nconst;
    cpu.num_targets = 1;
    cpu.num_rows = rows;
    cpu.num_banks = banks_count;
    cpu.toggle_bits = bits;
    cpu.input.data = input;
    cpu.input.num_elements = cols * rows;
    cpu.input.leading_dimension = rows;
    cpu.targets.data = target;
    cpu.targets.num_elements = rows;
    cpu.targets.leading_dimension = rows;
    cpu.constants.data = banks;
    cpu.constants.num_elements = banks_count * nconst;
    cpu.constants.bank_stride = nconst;
    cpu.output.data = expected;
    cpu.output.num_elements = na * configs;
    cpu.output.leading_dimension = configs;
    gpu.programs = cpu.programs;
    gpu.num_targets = 1;
    gpu.num_rows = rows;
    gpu.num_banks = banks_count;
    gpu.toggle_bits = bits;
    gpu.input.address = di;
    gpu.input.num_elements = cols * rows;
    gpu.input.leading_dimension = rows;
    gpu.targets.address = dt;
    gpu.targets.num_elements = rows;
    gpu.targets.leading_dimension = rows;
    gpu.constants.address = db;
    gpu.constants.num_elements = banks_count * nconst;
    gpu.constants.bank_stride = nconst;
    gpu.output.address = dout;
    gpu.output.num_elements = na * configs;
    gpu.output.leading_dimension = configs;
    recipe.num_kernels = 8;
    recipe.asts_per_kernel = packed;
    recipe.num_inputs = cols;
    recipe.num_constants = nconst;
    recipe.num_targets = 1;
    recipe.tile_rows = 32;
    recipe.threads_per_block = 128;
    recipe.patch_capacity_instructions = packed > 8 ? packed * 256 : 2048;
    r = build(&recipe.header);
    SEC(secant_cpu_run_toggle_sse(&cpu));
    SEC(secant_cubin_runner_run_toggle_sse(r.runner, &gpu, &stats));
    CUDA(cuMemcpyDtoH(actual, dout, na * configs * sizeof(float)));
    compare(actual, expected, na * configs);
    printf("random toggles (packed=%zu): %u configs / %u row evaluations; GPU %.6fs total %.6fs load %.6fs "
           "specialize(work) %.6fs\n",
           packed, na * configs, na * configs * rows, stats.runtime_seconds, stats.total_seconds,
           stats.module_load_seconds, stats.compile_work_seconds);
    report_registers(&r, &recipe, &gpu.programs);
    destroy(&r);
    CUDA(cuMemFree(di));
    CUDA(cuMemFree(dt));
    CUDA(cuMemFree(db));
    CUDA(cuMemFree(dout));
    free(actual);
    free(expected);
}
int main(void) {
    CUdevice device;
    CUcontext context;
    CUDA(cuInit(0));
    CUDA(cuDeviceGet(&device, 0));
    CUDA(cuDevicePrimaryCtxRetain(&context, device));
    CUDA(cuCtxSetCurrent(context));
    toggle_test(1);
    toggle_test(3);
    edge_test();
    random_test(1);
    random_test(8);
    random_test(32);
    static_test();
    CUDA(cuDevicePrimaryCtxRelease(device));
    puts("CUDA toggles and static regression passed");
    return 0;
}
