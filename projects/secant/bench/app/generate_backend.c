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
#include "bench_ast.h"

#if defined(SECANT_GENERATE_CUDA)
#include "secant_cuda.h"
#elif defined(SECANT_GENERATE_PTX)
#include "secant_ptx.h"
#else
#error "SECANT_GENERATE_CUDA or SECANT_GENERATE_PTX must be selected"
#endif

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum GenerateShape {
    GENERATE_SHAPE_MATERIALIZE = 0,
    GENERATE_SHAPE_SSE = 1,
    GENERATE_SHAPE_DYNAMIC_CONSTANT_SSE = 2,
    GENERATE_SHAPE_PHILOX_DYNAMIC_LEAF_SELECT = 3
} GenerateShape;

typedef struct GenerateOptions {
    GenerateShape shape;
    size_t num_kernels;
    size_t asts_per_kernel;
    size_t num_input_columns;
    size_t num_static_input_columns;
    size_t num_input_constants;
    size_t num_dynamic_leaves;
    size_t num_targets;
    size_t tile_rows;
    size_t threads_per_block;
    size_t patch_instructions_per_ast;
    uint32_t seed;
    SecantBenchAstMode ast_mode;
    const char* output_path;
} GenerateOptions;

static const char*
generate_backend_name(void) {
#if defined(SECANT_GENERATE_CUDA)
    return "cuda";
#else
    return "ptx";
#endif
}

static void
generate_usage(const char* argv0) {
    fprintf(
        stderr,
        "usage: %s <materialize|sse|dynamic_constant_sse|philox_dynamic_leaf_select> [options]\n"
        "\n"
        "options:\n"
        "  --kernels N                    kernels (default: 2)\n"
        "  --asts N                       ASTs per kernel (default: 2)\n"
        "  --inputs N                     input columns (default: 4)\n"
        "  --static-inputs N              mixed-leaf static columns (default: 4)\n"
        "  --constants N                  dynamic constants (default: 3)\n"
        "  --leaves N                     dynamic leaves (default: 8)\n"
        "  --targets N                    SSE targets (default: 2)\n"
        "  --tile-rows N                  rows per tile (default: 128)\n"
        "  --threads N                    threads per block (default: 128)\n"
        "  --patch-instructions-per-ast N direct ISA reserve (default: 64)\n"
        "  --ast-mode simple|alu|mufu     expression corpus (default: simple)\n"
        "  --seed N                       expression seed (default: 1)\n"
        "  -o FILE                        output file (default: stdout)\n",
        argv0);
}

static int
generate_parse_size(const char* text, size_t* value_ret) {
    char* end = NULL;
    unsigned long long value;

    errno = 0;
    value = strtoull(text, &end, 10);
    if (errno != 0 || text == end || *end != '\0' || value == 0u || value > SIZE_MAX) {
        return 0;
    }
    *value_ret = (size_t)value;
    return 1;
}

static int
generate_parse_u32(const char* text, uint32_t* value_ret) {
    char* end = NULL;
    unsigned long value;

    errno = 0;
    value = strtoul(text, &end, 0);
    if (errno != 0 || text == end || *end != '\0' || value > UINT32_MAX) {
        return 0;
    }
    *value_ret = (uint32_t)value;
    return 1;
}

static int
generate_parse_mode(
    const char* text,
    SecantBenchAstMode* mode_ret
) {
    if (strcmp(text, "simple") == 0) {
        *mode_ret = SECANT_BENCH_AST_MODE_SIMPLE;
    } else if (strcmp(text, "alu") == 0) {
        *mode_ret = SECANT_BENCH_AST_MODE_ALU;
    } else if (strcmp(text, "mufu") == 0) {
        *mode_ret = SECANT_BENCH_AST_MODE_MUFU;
    } else {
        return 0;
    }
    return 1;
}

static int
generate_options_parse(
    int argc,
    char** argv,
    GenerateOptions* options
) {
    int arg_idx;

    if (options == NULL || argc < 2) {
        return 0;
    }
    memset(options, 0, sizeof(*options));
    options->num_kernels = 2u;
    options->asts_per_kernel = 2u;
    options->num_input_columns = 4u;
    options->num_static_input_columns = 4u;
    options->num_input_constants = 3u;
    options->num_dynamic_leaves = 8u;
    options->num_targets = 2u;
    options->tile_rows = 128u;
    options->threads_per_block = 128u;
    options->patch_instructions_per_ast = 64u;
    options->seed = 1u;
    options->ast_mode = SECANT_BENCH_AST_MODE_SIMPLE;

    if (strcmp(argv[1], "materialize") == 0) {
        options->shape = GENERATE_SHAPE_MATERIALIZE;
    } else if (strcmp(argv[1], "sse") == 0) {
        options->shape = GENERATE_SHAPE_SSE;
    } else if (strcmp(argv[1], "dynamic_constant_sse") == 0) {
        options->shape = GENERATE_SHAPE_DYNAMIC_CONSTANT_SSE;
    } else if (strcmp(argv[1], "philox_dynamic_leaf_select") == 0) {
#if defined(SECANT_GENERATE_CUDA)
        options->shape = GENERATE_SHAPE_PHILOX_DYNAMIC_LEAF_SELECT;
#else
        return 0;
#endif
    } else {
        return 0;
    }
    for (arg_idx = 2; arg_idx < argc; ++arg_idx) {
        size_t* size_value = NULL;

        if (strcmp(argv[arg_idx], "--kernels") == 0) {
            size_value = &options->num_kernels;
        } else if (strcmp(argv[arg_idx], "--asts") == 0) {
            size_value = &options->asts_per_kernel;
        } else if (strcmp(argv[arg_idx], "--inputs") == 0) {
            size_value = &options->num_input_columns;
        } else if (strcmp(argv[arg_idx], "--constants") == 0) {
            size_value = &options->num_input_constants;
        } else if (strcmp(argv[arg_idx], "--leaves") == 0) {
            size_value = &options->num_dynamic_leaves;
        } else if (strcmp(argv[arg_idx], "--targets") == 0) {
            size_value = &options->num_targets;
        } else if (strcmp(argv[arg_idx], "--tile-rows") == 0) {
            size_value = &options->tile_rows;
        } else if (strcmp(argv[arg_idx], "--threads") == 0) {
            size_value = &options->threads_per_block;
        } else if (strcmp(argv[arg_idx], "--patch-instructions-per-ast") == 0) {
            size_value = &options->patch_instructions_per_ast;
        }
        if (size_value != NULL) {
            if (++arg_idx >= argc || !generate_parse_size(argv[arg_idx], size_value)) {
                return 0;
            }
        } else if (strcmp(argv[arg_idx], "--static-inputs") == 0) {
            uint32_t static_inputs;

            if (++arg_idx >= argc || !generate_parse_u32(argv[arg_idx], &static_inputs)) {
                return 0;
            }
            options->num_static_input_columns = (size_t)static_inputs;
        } else if (strcmp(argv[arg_idx], "--seed") == 0) {
            if (++arg_idx >= argc || !generate_parse_u32(argv[arg_idx], &options->seed)) {
                return 0;
            }
        } else if (strcmp(argv[arg_idx], "--ast-mode") == 0) {
            if (++arg_idx >= argc || !generate_parse_mode(argv[arg_idx], &options->ast_mode)) {
                return 0;
            }
        } else if (strcmp(argv[arg_idx], "-o") == 0 || strcmp(argv[arg_idx], "--output") == 0) {
            if (++arg_idx >= argc || argv[arg_idx][0] == '\0') {
                return 0;
            }
            options->output_path = argv[arg_idx];
        } else {
            return 0;
        }
    }
    return options->num_kernels <= SIZE_MAX / options->asts_per_kernel &&
        options->asts_per_kernel <= SIZE_MAX / options->patch_instructions_per_ast;
}

static int
generate_source(
    const GenerateOptions* options,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    const SecantAstInstruction* const* asts,
    char* output,
    size_t output_size,
    size_t* required_size_ret
) {
#if defined(SECANT_GENERATE_CUDA)
    SecantCUDAResult result;

    if (options->shape == GENERATE_SHAPE_MATERIALIZE) {
        result = secant_cuda_materialize_source_generate(
            options->num_kernels,
            options->asts_per_kernel,
            options->num_input_columns,
            routines,
            num_routines,
            routine_names,
            asts,
            output,
            output_size,
            required_size_ret,
            NULL);
    } else if (options->shape == GENERATE_SHAPE_SSE) {
        result = secant_cuda_sse_source_generate(
            options->num_kernels,
            options->asts_per_kernel,
            options->num_input_columns,
            options->num_targets,
            options->tile_rows,
            options->threads_per_block,
            routines,
            num_routines,
            routine_names,
            asts,
            output,
            output_size,
            required_size_ret,
            NULL);
    } else if (options->shape == GENERATE_SHAPE_DYNAMIC_CONSTANT_SSE) {
        result = secant_cuda_dynamic_constant_sse_source_generate(
            options->num_kernels,
            options->asts_per_kernel,
            options->num_input_columns,
            options->num_input_constants,
            options->num_targets,
            options->tile_rows,
            options->threads_per_block,
            routines,
            num_routines,
            routine_names,
            asts,
            output,
            output_size,
            required_size_ret,
            NULL);
    } else {
        result = secant_cuda_philox_dynamic_leaf_select_source_generate(
            options->num_kernels,
            options->asts_per_kernel,
            options->num_input_columns,
            options->num_static_input_columns,
            options->num_dynamic_leaves,
            options->tile_rows,
            options->threads_per_block,
            routines,
            num_routines,
            routine_names,
            asts,
            output,
            output_size,
            required_size_ret,
            NULL);
    }
    if (result != SECANT_CUDA_SUCCESS) {
        fprintf(
            stderr,
            "generation failed: %s\n",
            secant_cuda_result_to_string(result));
        return 0;
    }
#elif defined(SECANT_GENERATE_PTX)
    SecantPTXResult result;

    (void)routines;
    (void)num_routines;
    (void)routine_names;
    (void)asts;
    if (options->shape == GENERATE_SHAPE_MATERIALIZE) {
        result = secant_ptx_materialize_source_generate(
            options->num_kernels,
            options->asts_per_kernel,
            options->num_input_columns,
            output,
            output_size,
            required_size_ret);
    } else if (options->shape == GENERATE_SHAPE_SSE) {
        result = secant_ptx_sse_source_generate(
            options->num_kernels,
            options->asts_per_kernel,
            options->num_input_columns,
            options->num_targets,
            options->tile_rows,
            options->threads_per_block,
            output,
            output_size,
            required_size_ret);
    } else if (options->shape == GENERATE_SHAPE_DYNAMIC_CONSTANT_SSE) {
        result = secant_ptx_dynamic_constant_sse_source_generate(
            options->num_kernels,
            options->asts_per_kernel,
            options->num_input_columns,
            options->num_input_constants,
            options->num_targets,
            options->tile_rows,
            options->threads_per_block,
            output,
            output_size,
            required_size_ret);
    } else {
        return 0;
    }
    if (result != SECANT_PTX_SUCCESS) {
        fprintf(
            stderr,
            "generation failed: %s\n",
            secant_ptx_result_to_string(result));
        return 0;
    }
#endif
    return 1;
}

static int
generate_output_write(
    const GenerateOptions* options,
    const char* source,
    size_t source_size
) {
    FILE* output = stdout;
    int success = 1;

    if (options->output_path != NULL) {
        output = fopen(options->output_path, "wb");
        if (output == NULL) {
            fprintf(
                stderr,
                "failed to open output: %s\n",
                options->output_path);
            return 0;
        }
    }
    if (fprintf(
            output,
            "// Generated by secant_generate_%s. Do not edit.\n\n",
            generate_backend_name()) < 0 ||
        fwrite(source, 1u, source_size - 1u, output) !=
            source_size - 1u) {
        fprintf(stderr, "failed to write generated source\n");
        success = 0;
    }
    if (options->output_path != NULL && fclose(output) != 0) {
        fprintf(
            stderr,
            "failed to close output: %s\n",
            options->output_path);
        success = 0;
    }
    return success;
}

int
main(int argc, char** argv) {
    GenerateOptions options;
    SecantAstInstruction* programs = NULL;
    const SecantAstInstruction** asts = NULL;
    const SecantAstInstruction* const* routines = NULL;
    const char* const* routine_names = NULL;
    char* source = NULL;
    size_t program_bytes;
    size_t pointer_bytes;
    size_t source_size = 0u;
    size_t rendered_size = 0u;
    size_t num_inputs;
    size_t num_routines = 0u;
    size_t max_dynamic_leaves_used = 0u;
    int success;

    if (!generate_options_parse(argc, argv, &options)) {
        generate_usage(argv[0]);
        return 1;
    }
    num_inputs = options.num_input_columns +
        (options.shape == GENERATE_SHAPE_DYNAMIC_CONSTANT_SSE
            ? options.num_input_constants
            : 0u);
    success = secant_bench_ast_storage_sizes(
        1u,
        options.num_kernels,
        options.asts_per_kernel,
        &program_bytes,
        &pointer_bytes);
    if (success) {
        programs = (SecantAstInstruction*)malloc(program_bytes);
        asts = (const SecantAstInstruction**)malloc(pointer_bytes);
        success = programs != NULL && asts != NULL;
    }
    if (success) {
        secant_bench_ast_fill(
            1u,
            options.num_kernels,
            options.asts_per_kernel,
            num_inputs,
            options.seed,
            options.ast_mode,
            SECANT_BENCH_CSE_DEFAULT,
            programs,
            asts);
        if (options.shape == GENERATE_SHAPE_DYNAMIC_CONSTANT_SSE) {
            success = secant_bench_ast_dynamic_constants_rewrite(
                options.num_kernels * options.asts_per_kernel,
                options.num_input_columns,
                options.num_input_constants,
                programs);
        } else if (options.shape == GENERATE_SHAPE_PHILOX_DYNAMIC_LEAF_SELECT) {
            success = secant_bench_ast_dynamic_leaves_rewrite(
                options.num_kernels * options.asts_per_kernel,
                options.num_dynamic_leaves,
                options.num_dynamic_leaves,
                options.num_static_input_columns != 0u,
                programs,
                &max_dynamic_leaves_used);
        }
    }
    if (success) {
        secant_bench_ast_get_routines(
            options.ast_mode,
            &routines,
            &num_routines,
            &routine_names);
        success = generate_source(
            &options,
            routines,
            num_routines,
            routine_names,
            asts,
            NULL,
            0u,
            &source_size);
    }
    if (success) {
        source = (char*)malloc(source_size);
        success = source != NULL;
    }
    if (success) {
        success = generate_source(
            &options,
            routines,
            num_routines,
            routine_names,
            asts,
            source,
            source_size,
            &rendered_size) &&
            rendered_size == source_size;
    }
    if (success) {
        success = generate_output_write(
            &options,
            source,
            source_size);
    }
    free(source);
    free(asts);
    free(programs);
    return success ? 0 : 1;
}
