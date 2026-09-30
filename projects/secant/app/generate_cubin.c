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
#include "secant.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/* Emits CUDA source; NVRTC compilation and cache ownership remain caller policy. */
int main(int argc, char **argv) {
    SecantCubinMaterializeRecipe m = secant_cubin_materialize_recipe_init();
    SecantCubinSSERecipe s = secant_cubin_sse_recipe_init();
    SecantCubinAffineStatsRecipe a = secant_cubin_affine_stats_recipe_init();
    SecantCubinGramStatsRecipe g = secant_cubin_gram_stats_recipe_init();
    SecantCubinToggleSSERecipe t = secant_cubin_toggle_sse_recipe_init();
    SecantCubinRecipeHeader *header;
    size_t kernels = 8, asts = 8, inputs = 4, constants = 4, targets = 1, tile = 128, threads = 128,
           patch = 1024, bytes;
    const char *output_path = NULL;
    int i;
    char *source;
    FILE *output;
    SecantResult result;
    if (argc < 2)
        goto usage;
    for (i = 2; i < argc; ++i) {
        size_t *value = NULL;
        char *end;
        unsigned long long parsed;
        if (++i >= argc)
            goto usage;
        if (!strcmp(argv[i - 1], "--output")) {
            output_path = argv[i];
            continue;
        }
        if (!strcmp(argv[i - 1], "--kernels"))
            value = &kernels;
        else if (!strcmp(argv[i - 1], "--asts-per-kernel"))
            value = &asts;
        else if (!strcmp(argv[i - 1], "--inputs"))
            value = &inputs;
        else if (!strcmp(argv[i - 1], "--constants"))
            value = &constants;
        else if (!strcmp(argv[i - 1], "--targets"))
            value = &targets;
        else if (!strcmp(argv[i - 1], "--tile-rows"))
            value = &tile;
        else if (!strcmp(argv[i - 1], "--threads"))
            value = &threads;
        else if (!strcmp(argv[i - 1], "--patch-instructions"))
            value = &patch;
        else
            goto usage;
        errno = 0;
        parsed = strtoull(argv[i], &end, 10);
        if (errno || argv[i][0] == '-' || end == argv[i] || *end || parsed > SIZE_MAX)
            goto usage;
        *value = (size_t)parsed;
    }
#define SET_COMMON(r)                                                                                        \
    do {                                                                                                     \
        (r).num_kernels = kernels;                                                                           \
        (r).asts_per_kernel = asts;                                                                          \
        (r).num_inputs = inputs;                                                                             \
        (r).patch_capacity_instructions = patch;                                                             \
    } while (0)
#define SET_SCORE(r)                                                                                         \
    do {                                                                                                     \
        SET_COMMON(r);                                                                                       \
        (r).num_targets = targets;                                                                           \
        (r).tile_rows = tile;                                                                                \
        (r).threads_per_block = threads;                                                                     \
    } while (0)
    SET_COMMON(m);
    SET_SCORE(s);
    SET_SCORE(a);
    SET_SCORE(g);
    t.num_kernels = kernels;
    t.asts_per_kernel = asts;
    t.num_inputs = inputs;
    t.num_constants = constants;
    t.num_targets = targets;
    t.tile_rows = tile;
    t.threads_per_block = threads;
    t.patch_capacity_instructions = patch;
    if (!strcmp(argv[1], "materialize"))
        header = &m.header;
    else if (!strcmp(argv[1], "sse"))
        header = &s.header;
    else if (!strcmp(argv[1], "affine_stats"))
        header = &a.header;
    else if (!strcmp(argv[1], "gram_stats"))
        header = &g.header;
    else if (!strcmp(argv[1], "toggle_sse"))
        header = &t.header;
    else
        goto usage;
    result = secant_cubin_source_size(header, &bytes);
    if (result != SECANT_SUCCESS) {
        fprintf(stderr, "%s\n", secant_result_to_string(result));
        return 1;
    }
    source = (char *)malloc(bytes);
    if (!source)
        return 1;
    result = secant_cubin_source_write(header, source, bytes);
    if (result != SECANT_SUCCESS) {
        free(source);
        return 1;
    }
    output = output_path ? fopen(output_path, "wb") : stdout;
    if (!output) {
        free(source);
        return 1;
    }
    i = fwrite(source, 1, bytes - 1, output) == bytes - 1 ? 0 : 1;
    if (output_path && fclose(output))
        i = 1;
    free(source);
    return i;
usage:
    fprintf(stderr,
            "usage: %s materialize|sse|affine_stats|gram_stats|toggle_sse [--kernels N] [--asts-per-kernel "
            "N] [--inputs N] [--constants N] [--targets N] [--tile-rows N] [--threads N] "
            "[--patch-instructions N] [--output path]\nToggle SSE packs the requested ASTs per kernel.\n",
            argv[0]);
    return 2;
}
