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
/* Build this same harness against 7c77c3e (-DSETTINGS_BASELINE=1) or Secant 0.3.
 * The fixture, reference evaluator, timers, and graph scheduler are identical.
 */
#include "secant.h"
#include <cuda.h>
#include <math.h>
#include <nvrtc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef SETTINGS_BASELINE
#define SETTINGS_BASELINE 0
#endif
#define CHECK(x)                                                                                             \
    do {                                                                                                     \
        if (!(x)) {                                                                                          \
            fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x);                                          \
            exit(1);                                                                                         \
        }                                                                                                    \
    } while (0)
#define SEC(x)                                                                                               \
    do {                                                                                                     \
        SecantResult e_ = (x);                                                                               \
        if (e_ != SECANT_SUCCESS) {                                                                          \
            fprintf(stderr, "%d: %s: %s\n", __LINE__, #x, secant_result_to_string(e_));                      \
            exit(1);                                                                                         \
        }                                                                                                    \
    } while (0)
#define CUDA(x)                                                                                              \
    do {                                                                                                     \
        CUresult e_ = (x);                                                                                   \
        if (e_ != CUDA_SUCCESS) {                                                                            \
            const char *n_ = NULL;                                                                           \
            cuGetErrorName(e_, &n_);                                                                         \
            fprintf(stderr, "%d: %s: %s\n", __LINE__, #x, n_);                                               \
            exit(1);                                                                                         \
        }                                                                                                    \
    } while (0)
#ifndef CHOICE_GROUPS
#define CHOICE_GROUPS 1
#endif
#if CHOICE_GROUPS < 1 || CHOICE_GROUPS > 4
#error CHOICE_GROUPS must be between 1 and 4
#endif
enum {
    COLS = 8,
    CONSTANTS = 8,
    SLOTS = 8 * CHOICE_GROUPS,
    BITS = 8,
    PERIOD = 64,
    MAX_ASTS = 256,
    MAX_CODE = 512
};
typedef struct Leaf {
    unsigned constant, index;
} Leaf;
typedef struct Choice {
    unsigned count, lo, hi;
    Leaf leaves[4];
} Choice;
/* A slot has the same meaning across ASTs. The ASTs combine those slots with
 * different arithmetic operators and leaf order. This is representable by both APIs. */
static const Choice choices[8] = {{2, 0, 0, {{0, 0}, {0, 1}}},
                                  {2, 1, 0, {{0, 2}, {1, 0}}},
                                  {2, 2, 0, {{1, 1}, {1, 2}}},
                                  {4, 3, 4, {{0, 3}, {0, 4}, {1, 3}, {1, 4}}},
                                  {4, 5, 6, {{0, 5}, {1, 5}, {0, 6}, {1, 6}}},
                                  {2, 7, 0, {{0, 7}, {1, 7}}},
                                  {2, 0, 0, {{1, 0}, {0, 0}}},
                                  {4, 2, 5, {{0, 1}, {1, 1}, {0, 2}, {1, 2}}}};
typedef struct Bench {
    size_t asts, packed, rows, banks, configs, kernels, tile, threads, repeats;
    unsigned transcendental;
    const char *prefix;
    float *input, *targets, *constants, *scores;
    double *reference;
    uint32_t *masks, *words;
    uint8_t code[MAX_ASTS][MAX_CODE];
    const uint8_t *asts_ptr[MAX_ASTS];
    SecantAstProgramSet programs;
    void *cubin, *plan_storage;
    size_t cubin_bytes;
    SecantCubinPlan *plan;
    CUmodule module;
    CUfunction *functions;
    CUdeviceptr di, dt, db, dm, dw, dout, dprobe;
    CUstream stream;
    CUevent start, stop;
    int regs_min, regs_max, local_max, active_blocks_min, sm_count;
    size_t shared_bytes;
    double prepare_seconds, nvrtc_seconds, specialize_seconds, load_seconds;
    double reference_error, prediction_error;
} Bench;
static double seconds(void) {
    struct timespec t;
    CHECK(clock_gettime(CLOCK_MONOTONIC, &t) == 0);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
static void *allocate(size_t n, size_t width) {
    void *p = calloc(n, width);
    CHECK(p);
    return p;
}
static unsigned hash(unsigned x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    return x ^ (x >> 16);
}
static float value(unsigned x) { return ((int)(hash(x) % 1024) - 512) * (1.f / 1024.f); }
static Choice choice(unsigned slot) {
    unsigned group = slot / 8, i;
    Choice c = choices[slot % 8];
    c.lo = (c.lo + group) % BITS;
    c.hi = (c.hi + group) % BITS;
    for (i = 0; i < c.count; ++i)
        c.leaves[i].index = (c.leaves[i].index + 2 * group) % COLS;
    return c;
}
static unsigned leaf_slot(size_t ast, unsigned first) {
    return (first + (unsigned)ast / 3) % 8 + ((unsigned)ast % CHOICE_GROUPS) * 8;
}
static Leaf chosen(unsigned slot, unsigned permutation) {
    Choice selected_choice = choice(slot);
    const Choice *c = &selected_choice;
    unsigned i = (permutation >> c->lo) & 1;
    if (c->count == 4)
        i |= ((permutation >> c->hi) & 1) << 1;
    return c->leaves[i];
}
static unsigned operation(size_t ast, unsigned node) {
    while (node--)
        ast /= 3;
    return (unsigned)(ast % 3);
}
static void emit_leaf(uint8_t *out, size_t *n, unsigned slot) {
#if SETTINGS_BASELINE
    out[(*n)++] = SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_CONSTANT_OR_COLUMN_INPUT_F32;
    out[(*n)++] = (uint8_t)slot;
#else
    Choice selected_choice = choice(slot);
    const Choice *c = &selected_choice;
    unsigned i;
    for (i = 0; i < c->count; ++i) {
        out[(*n)++] = c->leaves[i].constant ? SECANT_AST_INSTRUCTION_TYPE_BANK_CONSTANT_F32
                                            : SECANT_AST_INSTRUCTION_TYPE_COLUMN_F32;
        out[(*n)++] = (uint8_t)c->leaves[i].index;
    }
    out[(*n)++] =
        c->count == 4 ? SECANT_AST_INSTRUCTION_TYPE_TOGGLE4_F32 : SECANT_AST_INSTRUCTION_TYPE_TOGGLE2_F32;
    out[(*n)++] = (uint8_t)c->lo;
    if (c->count == 4)
        out[(*n)++] = (uint8_t)c->hi;
#endif
}
static void emit_tree(Bench *b, size_t ast, unsigned depth, unsigned first, unsigned *node, size_t *n) {
    uint8_t *out = b->code[ast];
    static const unsigned ops[] = {SECANT_AST_INSTRUCTION_TYPE_ADD_F32, SECANT_AST_INSTRUCTION_TYPE_SUB_F32,
                                   SECANT_AST_INSTRUCTION_TYPE_MUL_F32};
    if (!depth) {
        emit_leaf(out, n, leaf_slot(ast, first));
        return;
    }
    emit_tree(b, ast, depth - 1, first, node, n);
    emit_tree(b, ast, depth - 1, first + (1U << (depth - 1)), node, n);
    out[(*n)++] = (uint8_t)ops[operation(ast, (*node)++)];
    if (b->transcendental && depth == 1)
        out[(*n)++] = SECANT_AST_INSTRUCTION_TYPE_SIN_F32;
}
static float evaluate(const Bench *b, size_t ast, unsigned depth, unsigned first, unsigned *node,
                      const float *leaves) {
    float left, right, result;
    unsigned op;
    if (!depth)
        return leaves[leaf_slot(ast, first)];
    left = evaluate(b, ast, depth - 1, first, node, leaves);
    right = evaluate(b, ast, depth - 1, first + (1U << (depth - 1)), node, leaves);
    op = operation(ast, (*node)++);
    result = op == 0 ? left + right : (op == 1 ? left - right : left * right);
    return b->transcendental && depth == 1 ? sinf(result) : result;
}
static void reference(Bench *b) {
    size_t q, r, a, s;
    b->reference = allocate(b->asts * b->configs, sizeof(double));
    for (q = 0; q < b->configs; ++q) {
        Leaf selected[SLOTS];
        for (s = 0; s < SLOTS; ++s)
            selected[s] = chosen((unsigned)s, (unsigned)q);
        for (r = 0; r < PERIOD && r < b->rows; ++r) {
            float leaves[SLOTS];
            size_t multiplicity = b->rows / PERIOD + (r < b->rows % PERIOD);
            for (s = 0; s < SLOTS; ++s)
                leaves[s] = selected[s].constant ? b->constants[(q >> BITS) * CONSTANTS + selected[s].index]
                                                 : b->input[selected[s].index * b->rows + r];
            for (a = 0; a < b->asts; ++a) {
                unsigned node = 0;
                double error = (double)evaluate(b, a, 3, 0, &node, leaves) - b->targets[r];
                b->reference[a * b->configs + q] += error * error * (double)multiplicity;
            }
        }
    }
}
static CUdeviceptr upload(const void *data, size_t bytes) {
    CUdeviceptr ptr;
    CUDA(cuMemAlloc(&ptr, bytes));
    CUDA(cuMemcpyHtoD(ptr, data, bytes));
    return ptr;
}
static void prepare(Bench *b) {
    size_t a, c, r, q, s;
    float probes[2] = {0, 1};
    double begin = seconds();
    b->configs = b->banks << BITS;
    b->kernels = b->asts / b->packed;
    b->input = allocate(COLS * b->rows, sizeof(float));
    b->targets = allocate(b->rows, sizeof(float));
    b->constants = allocate(b->banks * CONSTANTS, sizeof(float));
    b->scores = allocate(b->asts * b->configs, sizeof(float));
    b->masks = allocate(b->configs, sizeof(uint32_t));
    b->words = allocate(b->configs * SLOTS, sizeof(uint32_t));
    for (c = 0; c < COLS; ++c)
        for (r = 0; r < b->rows; ++r)
            b->input[c * b->rows + r] = value((unsigned)(c * PERIOD + r % PERIOD + 17));
    for (r = 0; r < b->rows; ++r)
        b->targets[r] = value((unsigned)(3007 + r % PERIOD));
    for (c = 0; c < b->banks * CONSTANTS; ++c)
        b->constants[c] = value((unsigned)c + 7019);
    for (q = 0; q < b->configs; ++q)
        for (s = 0; s < SLOTS; ++s) {
            Leaf leaf = chosen((unsigned)s, (unsigned)q);
            if (leaf.constant)
                memcpy(b->words + q * SLOTS + s, b->constants + (q >> BITS) * CONSTANTS + leaf.index,
                       sizeof(float));
            else {
                b->masks[q] |= 1U << s;
                b->words[q * SLOTS + s] = leaf.index;
            }
        }
    for (a = 0; a < b->asts; ++a) {
        unsigned node = 0;
        size_t n = 0;
        emit_tree(b, a, 3, 0, &node, &n);
        b->code[a][n++] = SECANT_AST_INSTRUCTION_TYPE_RETURN_F32;
        CHECK(n < MAX_CODE);
        b->asts_ptr[a] = b->code[a];
    }
    b->programs.asts.items = b->asts_ptr;
    b->programs.asts.count = b->asts;
    b->di = upload(b->input, COLS * b->rows * sizeof(float));
    b->dt = upload(b->targets, b->rows * sizeof(float));
    b->dout = upload(b->scores, b->asts * b->configs * sizeof(float));
    b->dprobe = upload(probes, sizeof(probes));
#if SETTINGS_BASELINE
    b->dm = upload(b->masks, b->configs * sizeof(uint32_t));
    b->dw = upload(b->words, b->configs * SLOTS * sizeof(uint32_t));
#else
    b->db = upload(b->constants, b->banks * CONSTANTS * sizeof(float));
#endif
    b->prepare_seconds = seconds() - begin;
}
static void compile_template(Bench *b) {
    char *source, *log, arch[48];
    size_t bytes, log_bytes, storage;
    CUdevice device;
    int major, minor;
    nvrtcProgram program;
    nvrtcResult status;
    const char *flags[] = {arch, "--std=c++14", "--use_fast_math", "--fmad=false"};
    const SecantCubinRecipeHeader *header;
#if SETTINGS_BASELINE
    SecantCubinDynamicLeafSSERecipe recipe = secant_cubin_dynamic_leaf_sse_recipe_init();
    recipe.num_input_columns = COLS;
    recipe.num_static_input_columns = 0;
    recipe.num_dynamic_leaves = SLOTS;
#else
    SecantCubinToggleSSERecipe recipe = secant_cubin_toggle_sse_recipe_init();
    recipe.num_inputs = COLS;
    recipe.num_constants = CONSTANTS;
#endif
    double begin;
    recipe.num_kernels = b->kernels;
    recipe.asts_per_kernel = b->packed;
    recipe.num_targets = 1;
    recipe.tile_rows = b->tile;
    recipe.threads_per_block = b->threads;
    recipe.patch_capacity_instructions = 128 * b->packed;
    header = &recipe.header;
    SEC(secant_cubin_source_size(header, &bytes));
    source = allocate(bytes, 1);
    SEC(secant_cubin_source_write(header, source, bytes));
    CUDA(cuCtxGetDevice(&device));
    CUDA(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    CUDA(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
    CUDA(cuDeviceGetAttribute(&b->sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
    snprintf(arch, sizeof(arch), "--gpu-architecture=sm_%d%d", major, minor);
    CHECK(nvrtcCreateProgram(&program, source, "comparison.cu", 0, NULL, NULL) == NVRTC_SUCCESS);
    begin = seconds();
    status = nvrtcCompileProgram(program, 4, flags);
    b->nvrtc_seconds = seconds() - begin;
    CHECK(nvrtcGetProgramLogSize(program, &log_bytes) == NVRTC_SUCCESS);
    log = allocate(log_bytes, 1);
    CHECK(nvrtcGetProgramLog(program, log) == NVRTC_SUCCESS);
    if (status != NVRTC_SUCCESS)
        fprintf(stderr, "%s\n", log);
    CHECK(status == NVRTC_SUCCESS);
    free(log);
    free(source);
    CHECK(nvrtcGetCUBINSize(program, &b->cubin_bytes) == NVRTC_SUCCESS);
    b->cubin = allocate(b->cubin_bytes, 1);
    CHECK(nvrtcGetCUBIN(program, b->cubin) == NVRTC_SUCCESS);
    CHECK(nvrtcDestroyProgram(&program) == NVRTC_SUCCESS);
    SEC(secant_cubin_plan_storage_size(header, b->cubin, b->cubin_bytes, &storage));
    b->plan_storage = allocate(storage, 1);
    SEC(secant_cubin_plan_init(header, b->cubin, b->cubin_bytes, b->plan_storage, storage, &b->plan));
}
static void load_specialized(Bench *b) {
    void *patched = allocate(b->cubin_bytes, 1);
    size_t k;
    double begin;
    memcpy(patched, b->cubin, b->cubin_bytes);
    begin = seconds();
    SEC(secant_cubin_specialize_into(b->plan, &b->programs, patched, b->cubin_bytes));
    b->specialize_seconds = seconds() - begin;
    if (getenv("SECANT_COMPARE_DUMP_CUBIN")) {
        FILE *f = fopen(getenv("SECANT_COMPARE_DUMP_CUBIN"), "wb");
        CHECK(f);
        CHECK(fwrite(patched, 1, b->cubin_bytes, f) == b->cubin_bytes);
        CHECK(fclose(f) == 0);
    }
    begin = seconds();
    CUDA(cuModuleLoadData(&b->module, patched));
    b->functions = allocate(b->kernels, sizeof(CUfunction));
    b->regs_min = 9999;
    b->active_blocks_min = 9999;
    b->shared_bytes = SETTINGS_BASELINE ? ((COLS | 1) + 1) * b->tile * sizeof(float) : 0;
    for (k = 0; k < b->kernels; ++k) {
        char name[96];
        int regs, local, blocks;
        snprintf(name, sizeof(name), "%s_%03zu",
                 SETTINGS_BASELINE ? "secant_cubin_dynamic_leaf_sse" : "secant_cubin_toggle_sse", k);
        CUDA(cuModuleGetFunction(b->functions + k, b->module, name));
        CUDA(cuFuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, b->functions[k]));
        CUDA(cuFuncGetAttribute(&local, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, b->functions[k]));
        CUDA(cuOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, b->functions[k], (int)b->threads,
                                                         b->shared_bytes));
        if (regs < b->regs_min)
            b->regs_min = regs;
        if (regs > b->regs_max)
            b->regs_max = regs;
        if (local > b->local_max)
            b->local_max = local;
        if (blocks < b->active_blocks_min)
            b->active_blocks_min = blocks;
    }
    b->load_seconds = seconds() - begin;
    free(patched);
}
/* Independent kernel nodes remove per-function host submission gaps from the
 * execution-engine measurement. Both backends use this exact graph scheduler. */
static CUgraphExec make_graph(Bench *b, size_t rows, CUdeviceptr target) {
    CUgraph graph;
    CUgraphExec executable;
    size_t k;
    CUDA(cuGraphCreate(&graph, 0));
    for (k = 0; k < b->kernels; ++k) {
        CUDA_KERNEL_NODE_PARAMS p;
        CUgraphNode node;
        size_t cols = COLS, input_ld = b->rows, target_ld = b->rows, word_ld = SLOTS;
        size_t targets = 1, active = SETTINGS_BASELINE ? b->packed : b->asts, out_ld = b->configs,
               bank_stride = CONSTANTS;
        unsigned bits = BITS;
        CUdeviceptr output = b->dout + (SETTINGS_BASELINE ? k * b->packed * b->configs * sizeof(float) : 0);
#if SETTINGS_BASELINE
        void *args[] = {&b->di,  &cols,      &input_ld, &b->dm,  &b->dw,   &word_ld, &b->configs,
                        &target, &target_ld, &rows,     &active, &targets, &output,  &out_ld};
        (void)bank_stride;
        (void)bits;
#else
        void *args[] = {&b->di, &input_ld,   &target, &target_ld, &b->db,  &bank_stride,
                        &rows,  &b->configs, &bits,   &output,    &out_ld, &active};
        (void)cols;
        (void)word_ld;
        (void)targets;
#endif
        memset(&p, 0, sizeof(p));
        p.func = b->functions[k];
        p.gridDimX = (unsigned)((rows + b->tile - 1) / b->tile);
        p.gridDimY = SETTINGS_BASELINE ? 1 : (unsigned)((b->configs + b->threads - 1) / b->threads);
        p.gridDimZ = 1;
        p.blockDimX = (unsigned)b->threads;
        p.blockDimY = p.blockDimZ = 1;
        p.sharedMemBytes = (unsigned)b->shared_bytes;
        p.kernelParams = args;
        CUDA(cuGraphAddKernelNode(&node, graph, NULL, 0, &p));
    }
    CUDA(cuGraphInstantiateWithFlags(&executable, graph, 0));
    CUDA(cuGraphDestroy(graph));
    return executable;
}
static double graph_run(Bench *b, CUgraphExec graph) {
    float ms;
    CUDA(cuMemsetD32Async(b->dout, 0, b->asts * b->configs, b->stream));
    CUDA(cuEventRecord(b->start, b->stream));
    CUDA(cuGraphLaunch(graph, b->stream));
    CUDA(cuEventRecord(b->stop, b->stream));
    CUDA(cuEventSynchronize(b->stop));
    CUDA(cuEventElapsedTime(&ms, b->start, b->stop));
    return ms * 1e-3;
}
static void check_scores(Bench *b) {
    size_t i;
    CUDA(cuMemcpyDtoH(b->scores, b->dout, b->asts * b->configs * sizeof(float)));
    for (i = 0; i < b->asts * b->configs; ++i) {
        double err = fabs(b->scores[i] - b->reference[i]) / (1 + fabs(b->reference[i]));
        if (!isfinite(b->scores[i]) || err > 2e-5) {
            fprintf(stderr, "score mismatch %zu GPU %.9g reference %.12g scaled error %.9g\n", i,
                    b->scores[i], b->reference[i], err);
            exit(1);
        }
        if (err > b->reference_error)
            b->reference_error = err;
    }
}
static void check_predictions(Bench *b) {
    size_t q, a, s;
    float *zero = allocate(b->asts * b->configs, sizeof(float));
    CUgraphExec g0 = make_graph(b, 1, b->dprobe), g1 = make_graph(b, 1, b->dprobe + sizeof(float));
    (void)graph_run(b, g0);
    CUDA(cuMemcpyDtoH(zero, b->dout, b->asts * b->configs * sizeof(float)));
    (void)graph_run(b, g1);
    CUDA(cuMemcpyDtoH(b->scores, b->dout, b->asts * b->configs * sizeof(float)));
    for (q = 0; q < b->configs; ++q) {
        float leaves[SLOTS];
        for (s = 0; s < SLOTS; ++s) {
            Leaf leaf = chosen((unsigned)s, (unsigned)q);
            leaves[s] = leaf.constant ? b->constants[(q >> BITS) * CONSTANTS + leaf.index]
                                      : b->input[leaf.index * b->rows];
        }
        for (a = 0; a < b->asts; ++a) {
            unsigned node = 0;
            size_t i = a * b->configs + q;
            double expected = evaluate(b, a, 3, 0, &node, leaves);
            double prediction = ((double)zero[i] - b->scores[i] + 1) * .5;
            double error = fabs(prediction - expected) / (1 + fabs(expected));
            CHECK(isfinite(prediction) && error < 2e-5);
            if (error > b->prediction_error)
                b->prediction_error = error;
        }
    }
    CUDA(cuGraphExecDestroy(g0));
    CUDA(cuGraphExecDestroy(g1));
    free(zero);
}
static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
static double median(double *values, size_t n) {
    qsort(values, n, sizeof(*values), cmp_double);
    return values[n / 2];
}
static void pipeline(Bench *b, double *wall, double *device, double *load) {
    SecantCubinRunner runner;
    SecantCubinRunnerOptions options = secant_cubin_runner_options_init();
    double ws[3], ds[3], ls[3];
    size_t i;
#if SETTINGS_BASELINE
    SecantCubinDynamicLeafSSERun run = secant_cubin_dynamic_leaf_sse_run_init();
    run.num_input_columns = COLS;
    run.leaf_masks.address = b->dm;
    run.leaf_masks.num_elements = b->configs;
    run.leaf_words.address = b->dw;
    run.leaf_words.num_elements = b->configs * SLOTS;
    run.leaf_words.leading_dimension = SLOTS;
    run.num_settings = b->configs;
#else
    SecantCubinToggleSSERun run = secant_cubin_toggle_sse_run_init();
    run.constants.address = b->db;
    run.constants.num_elements = b->banks * CONSTANTS;
    run.constants.bank_stride = CONSTANTS;
    run.num_banks = b->banks;
    run.toggle_bits = BITS;
#endif
    run.programs = b->programs;
    run.input.address = b->di;
    run.input.num_elements = COLS * b->rows;
    run.input.leading_dimension = b->rows;
    run.targets.address = b->dt;
    run.targets.num_elements = b->rows;
    run.targets.leading_dimension = b->rows;
    run.num_rows = b->rows;
    run.num_targets = 1;
    run.output.address = b->dout;
    run.output.num_elements = b->asts * b->configs;
    run.output.leading_dimension = b->configs;
    options.num_workers = 3;
    options.num_streams = b->kernels < 8 ? b->kernels : 8;
    SEC(secant_cubin_runner_create(b->plan, b->cubin, b->cubin_bytes, &options, &runner));
    for (i = 0; i < 4; ++i) {
        SecantRunnerStats stats = secant_runner_stats_init();
        double begin = seconds();
        SEC(secant_cubin_runner_run(runner, &run.header, &stats));
        if (i) {
            ws[i - 1] = seconds() - begin;
            ds[i - 1] = stats.runtime_seconds;
            ls[i - 1] = stats.module_load_seconds;
        }
        check_scores(b);
    }
    SEC(secant_cubin_runner_destroy(runner));
    *wall = median(ws, 3);
    *device = median(ds, 3);
    *load = median(ls, 3);
}
int main(int argc, char **argv) {
    Bench b;
    CUdevice device;
    CUcontext context;
    CUgraphExec graph;
    double times[31], kernel_seconds, wall, device_seconds, load, validation_seconds, begin;
    size_t i;
    char path[2048];
    FILE *f;
    CHECK(argc == 10);
    memset(&b, 0, sizeof(b));
    b.packed = strtoul(argv[1], NULL, 10);
    b.rows = strtoul(argv[2], NULL, 10);
    b.banks = strtoul(argv[3], NULL, 10);
    b.asts = strtoul(argv[4], NULL, 10);
    b.tile = strtoul(argv[5], NULL, 10);
    b.threads = strtoul(argv[6], NULL, 10);
    b.transcendental = (unsigned)strtoul(argv[7], NULL, 10);
    b.repeats = strtoul(argv[8], NULL, 10);
    b.prefix = argv[9];
    CHECK(b.packed && b.packed <= 32 && b.asts && b.asts <= MAX_ASTS && b.asts % b.packed == 0);
    CHECK(b.rows >= PERIOD && b.rows <= 65536 && b.banks && b.banks <= 64 && b.tile && b.tile <= 512);
    CHECK(b.threads >= 32 && b.threads <= 512 && b.threads % 32 == 0 && b.repeats >= 3 && b.repeats <= 31 &&
          b.transcendental <= 1);
    CUDA(cuInit(0));
    CUDA(cuDeviceGet(&device, 0));
    CUDA(cuDevicePrimaryCtxRetain(&context, device));
    CUDA(cuCtxSetCurrent(context));
    CUDA(cuStreamCreate(&b.stream, CU_STREAM_NON_BLOCKING));
    CUDA(cuEventCreate(&b.start, 0));
    CUDA(cuEventCreate(&b.stop, 0));
    prepare(&b);
    compile_template(&b);
    load_specialized(&b);
    begin = seconds();
    reference(&b);
    check_predictions(&b);
    validation_seconds = seconds() - begin;
    graph = make_graph(&b, b.rows, b.dt);
    (void)graph_run(&b, graph);
    check_scores(&b);
    (void)graph_run(&b, graph);
    for (i = 0; i < b.repeats; ++i)
        times[i] = graph_run(&b, graph);
    check_scores(&b);
    kernel_seconds = median(times, b.repeats);
    CUDA(cuGraphExecDestroy(graph));
    CUDA(cuModuleUnload(b.module));
    pipeline(&b, &wall, &device_seconds, &load);
    CHECK(snprintf(path, sizeof(path), "%s.scores.f32", b.prefix) < (int)sizeof(path));
    f = fopen(path, "wb");
    CHECK(f);
    CHECK(fwrite(b.scores, sizeof(float), b.asts * b.configs, f) == b.asts * b.configs);
    CHECK(fclose(f) == 0);
    printf("{\"backend\":\"%s\",\"version\":\"%s\",\"asts\":%zu,\"packed\":%zu,\"rows\":%zu,\"banks\":%zu,"
           "\"configs_per_ast\":%zu,\"tile\":%zu,\"threads\":%zu,\"transcendental\":%u,\"samples\":%zu,\"sm_"
           "count\":%d,\"registers_min\":%d,\"registers_max\":%d,\"local_bytes_max\":%d,\"active_blocks_per_"
           "sm_min\":%d,\"kernel_seconds_median\":%.9g,\"kernel_seconds_min\":%.9g,\"kernel_seconds_max\":%."
           "9g,\"row_evals_per_second\":%.9g,\"pipeline_seconds_median\":%.9g,\"pipeline_device_seconds_"
           "median\":%.9g,\"pipeline_load_seconds_median\":%.9g,\"pipeline_row_evals_per_second\":%.9g,"
           "\"nvrtc_seconds\":%.9g,\"prepare_seconds\":%.9g,\"specialize_seconds\":%.9g,\"resident_load_"
           "seconds\":%.9g,\"validation_seconds\":%.9g,\"score_max_scaled_error\":%.9g,\"prediction_max_"
           "scaled_error\":%.9g,\"data_period\":%d,\"choice_groups\":%d}\n",
           SETTINGS_BASELINE ? "settings" : "toggles", SECANT_VERSION_STRING, b.asts, b.packed, b.rows,
           b.banks, b.configs, b.tile, b.threads, b.transcendental, b.repeats, b.sm_count, b.regs_min,
           b.regs_max, b.local_max, b.active_blocks_min, kernel_seconds, times[0], times[b.repeats - 1],
           (double)b.asts * b.configs * b.rows / kernel_seconds, wall, device_seconds, load,
           (double)b.asts * b.configs * b.rows / wall, b.nvrtc_seconds, b.prepare_seconds,
           b.specialize_seconds, b.load_seconds, validation_seconds, b.reference_error, b.prediction_error,
           PERIOD, CHOICE_GROUPS);
    CUDA(cuMemFree(b.di));
    CUDA(cuMemFree(b.dt));
    CUDA(cuMemFree(b.dout));
    CUDA(cuMemFree(b.dprobe));
    if (b.db)
        CUDA(cuMemFree(b.db));
    if (b.dm)
        CUDA(cuMemFree(b.dm));
    if (b.dw)
        CUDA(cuMemFree(b.dw));
    CUDA(cuEventDestroy(b.start));
    CUDA(cuEventDestroy(b.stop));
    CUDA(cuStreamDestroy(b.stream));
    CUDA(cuDevicePrimaryCtxRelease(device));
    free(b.input);
    free(b.targets);
    free(b.constants);
    free(b.scores);
    free(b.reference);
    free(b.masks);
    free(b.words);
    free(b.cubin);
    free(b.plan_storage);
    free(b.functions);
    return 0;
}
