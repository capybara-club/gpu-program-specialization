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
#define _POSIX_C_SOURCE 200809L

#include "cubin_cache.h"

#include <sqlite3.h>

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void
secant_sr_cubin_cache_test_files_remove(const char* path) {
    char auxiliary_path[512];

    (void)remove(path);
    if (snprintf(auxiliary_path, sizeof(auxiliary_path), "%s-wal", path) > 0) {
        (void)remove(auxiliary_path);
    }
    if (snprintf(auxiliary_path, sizeof(auxiliary_path), "%s-shm", path) > 0) {
        (void)remove(auxiliary_path);
    }
}

static int
secant_sr_cubin_cache_test_v2_create(const char* path, const unsigned char* cubin, size_t cubin_size) {
    static const char schema[] =
        "CREATE TABLE cubin_template ("
        "id INTEGER PRIMARY KEY, source_sha256 BLOB NOT NULL, compute_capability_major INTEGER NOT NULL,"
        "compute_capability_minor INTEGER NOT NULL, nvrtc_major INTEGER NOT NULL, nvrtc_minor INTEGER NOT NULL,"
        "ptxas_opt_level INTEGER NOT NULL, nvrtc_no_cache INTEGER NOT NULL, cubin BLOB NOT NULL,"
        "compile_seconds REAL NOT NULL, created_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);"
        "PRAGMA user_version=2;";
    static const char insert_sql[] =
        "INSERT INTO cubin_template (source_sha256, compute_capability_major, compute_capability_minor, nvrtc_major,"
        "nvrtc_minor, ptxas_opt_level, nvrtc_no_cache, cubin, compile_seconds) "
        "VALUES (zeroblob(32), 12, 0, 12, 9, 1, 1, ?1, 1.25);";
    sqlite3* database = NULL;
    sqlite3_stmt* statement = NULL;
    int success = 0;

    if (cubin_size > (size_t)INT_MAX || sqlite3_open(path, &database) != SQLITE_OK ||
        sqlite3_exec(database, schema, NULL, NULL, NULL) != SQLITE_OK ||
        sqlite3_prepare_v2(database, insert_sql, -1, &statement, NULL) != SQLITE_OK ||
        sqlite3_bind_blob(statement, 1, cubin, (int)cubin_size, SQLITE_TRANSIENT) != SQLITE_OK ||
        sqlite3_step(statement) != SQLITE_DONE) {
        goto cleanup;
    }
    success = 1;

cleanup:
    if (statement != NULL) {
        (void)sqlite3_finalize(statement);
    }
    if (database != NULL) {
        (void)sqlite3_close(database);
    }
    return success;
}

static int
secant_sr_cubin_cache_test_schema_check(const char* path) {
    static const char sql[] =
        "SELECT (SELECT user_version FROM pragma_user_version), "
        "(SELECT count(*) FROM pragma_table_info('cubin_template') WHERE name='source_sha256'), "
        "(SELECT count(*) FROM cubin_template), "
        "(SELECT count(*) FROM cubin_artifact);";
    sqlite3* database = NULL;
    sqlite3_stmt* statement = NULL;
    int success = 0;

    if (sqlite3_open_v2(path, &database, SQLITE_OPEN_READONLY, NULL) != SQLITE_OK ||
        sqlite3_prepare_v2(database, sql, -1, &statement, NULL) != SQLITE_OK ||
        sqlite3_step(statement) != SQLITE_ROW) {
        goto cleanup;
    }
    success = sqlite3_column_int(statement, 0) == 3 &&
        sqlite3_column_int(statement, 1) == 0 &&
        sqlite3_column_int(statement, 2) == 5 &&
        sqlite3_column_int(statement, 3) == 1;

cleanup:
    if (statement != NULL) {
        (void)sqlite3_finalize(statement);
    }
    if (database != NULL) {
        (void)sqlite3_close(database);
    }
    return success;
}

int
main(void) {
    static const unsigned char cubin[] = {0x7fu, 0x45u, 0x4cu, 0x46u, 0x01u, 0x02u, 0x03u};
    SecantCubinSSERecipe recipe = secant_cubin_sse_recipe_init();
    SecantCubinSSERecipe changed_recipe;
    SecantCubinConstantOptimizerSSERecipe optimizer_recipe = secant_cubin_constant_optimizer_sse_recipe_init();
    SecantCubinPackedConstantOptimizerSSERecipe packed_optimizer_recipe =
        secant_cubin_packed_constant_optimizer_sse_recipe_init();
    SecantCubinDynamicLeafSSERecipe dynamic_recipe = secant_cubin_dynamic_leaf_sse_recipe_init();
    SecantCubinDynamicLeafSSERecipe mixed_recipe;
    char path[256];
    SecantSRCubinCache cache = {0};
    unsigned char* cached_cubin = NULL;
    size_t cached_cubin_size = 0u;
    double compile_seconds = 0.0;
    int hit = 0;
    int status = 1;

    recipe.num_kernels = 64u;
    recipe.asts_per_kernel = 32u;
    recipe.num_inputs = 8u;
    recipe.num_targets = 1u;
    recipe.tile_rows = 1024u;
    recipe.threads_per_block = 128u;
    recipe.patch_capacity_instructions = 2048u;
    changed_recipe = recipe;
    changed_recipe.tile_rows = 2048u;
    optimizer_recipe.num_kernels = 128u;
    optimizer_recipe.num_input_columns = 8u;
    optimizer_recipe.num_input_constants = 8u;
    optimizer_recipe.tile_rows = 128u;
    optimizer_recipe.threads_per_block = 128u;
    optimizer_recipe.patch_capacity_instructions = 64u;
    packed_optimizer_recipe.num_kernels = 64u;
    packed_optimizer_recipe.asts_per_kernel = 32u;
    packed_optimizer_recipe.num_input_columns = 8u;
    packed_optimizer_recipe.num_input_constants = 8u;
    packed_optimizer_recipe.tile_rows = 128u;
    packed_optimizer_recipe.threads_per_block = 128u;
    packed_optimizer_recipe.patch_capacity_instructions = 2048u;
    dynamic_recipe.num_kernels = 64u;
    dynamic_recipe.asts_per_kernel = 32u;
    dynamic_recipe.num_input_columns = 8u;
    dynamic_recipe.num_static_input_columns = 0u;
    dynamic_recipe.num_dynamic_leaves = 8u;
    dynamic_recipe.num_targets = 1u;
    dynamic_recipe.tile_rows = 64u;
    dynamic_recipe.threads_per_block = 128u;
    dynamic_recipe.patch_capacity_instructions = 64u;
    mixed_recipe = dynamic_recipe;
    mixed_recipe.num_static_input_columns = dynamic_recipe.num_input_columns;
    if (snprintf(path, sizeof(path), "secant_sr_cubin_cache_test_%ld.sqlite3", (long)getpid()) < 0) {
        return 1;
    }
    secant_sr_cubin_cache_test_files_remove(path);
    if (!secant_sr_cubin_cache_test_v2_create(path, cubin, sizeof(cubin)) ||
        !secant_sr_cubin_cache_open(path, &cache) ||
        !secant_sr_cubin_cache_lookup(
            &cache, &recipe.header, 12, 0, 12, 9, 1, 1,
            &cached_cubin, &cached_cubin_size, &compile_seconds, &hit) ||
        hit || cached_cubin != NULL ||
        !secant_sr_cubin_cache_store(
            &cache, &recipe.header, 12, 0, 12, 9, 1, 1, cubin, sizeof(cubin), 2.5) ||
        !secant_sr_cubin_cache_lookup(
            &cache, &changed_recipe.header, 12, 0, 12, 9, 1, 1,
            &cached_cubin, &cached_cubin_size, &compile_seconds, &hit) ||
        hit || cached_cubin != NULL) {
        goto cleanup;
    }
    secant_sr_cubin_cache_close(&cache);
    if (!secant_sr_cubin_cache_open(path, &cache) ||
        !secant_sr_cubin_cache_lookup(
            &cache, &recipe.header, 12, 0, 12, 9, 1, 1,
            &cached_cubin, &cached_cubin_size, &compile_seconds, &hit) ||
        !hit || cached_cubin_size != sizeof(cubin) || memcmp(cached_cubin, cubin, sizeof(cubin)) != 0 ||
        fabs(compile_seconds - 2.5) > 1.0e-12) {
        goto cleanup;
    }
    free(cached_cubin);
    cached_cubin = NULL;
    if (!secant_sr_cubin_cache_store(
            &cache, &optimizer_recipe.header, 12, 0, 12, 9, 1, 1, cubin, sizeof(cubin), 3.5) ||
        !secant_sr_cubin_cache_lookup(
            &cache,
            &optimizer_recipe.header,
            12,
            0,
            12,
            9,
            1,
            1,
            &cached_cubin,
            &cached_cubin_size,
            &compile_seconds,
            &hit) ||
        !hit || cached_cubin_size != sizeof(cubin) || memcmp(cached_cubin, cubin, sizeof(cubin)) != 0 ||
        fabs(compile_seconds - 3.5) > 1.0e-12) {
        goto cleanup;
    }
    free(cached_cubin);
    cached_cubin = NULL;
    if (!secant_sr_cubin_cache_store(
            &cache, &packed_optimizer_recipe.header, 12, 0, 12, 9, 1, 1, cubin, sizeof(cubin), 4.0) ||
        !secant_sr_cubin_cache_lookup(
            &cache,
            &packed_optimizer_recipe.header,
            12,
            0,
            12,
            9,
            1,
            1,
            &cached_cubin,
            &cached_cubin_size,
            &compile_seconds,
            &hit) ||
        !hit || cached_cubin_size != sizeof(cubin) || memcmp(cached_cubin, cubin, sizeof(cubin)) != 0 ||
        fabs(compile_seconds - 4.0) > 1.0e-12) {
        goto cleanup;
    }
    free(cached_cubin);
    cached_cubin = NULL;
    if (!secant_sr_cubin_cache_store(
            &cache, &dynamic_recipe.header, 12, 0, 12, 9, 1, 1, cubin, sizeof(cubin), 5.0) ||
        !secant_sr_cubin_cache_lookup(
            &cache,
            &mixed_recipe.header,
            12,
            0,
            12,
            9,
            1,
            1,
            &cached_cubin,
            &cached_cubin_size,
            &compile_seconds,
            &hit) ||
        hit || cached_cubin != NULL ||
        !secant_sr_cubin_cache_store(
            &cache, &mixed_recipe.header, 12, 0, 12, 9, 1, 1, cubin, sizeof(cubin), 5.5) ||
        !secant_sr_cubin_cache_lookup(
            &cache,
            &dynamic_recipe.header,
            12,
            0,
            12,
            9,
            1,
            1,
            &cached_cubin,
            &cached_cubin_size,
            &compile_seconds,
            &hit) ||
        !hit || fabs(compile_seconds - 5.0) > 1.0e-12) {
        goto cleanup;
    }
    free(cached_cubin);
    cached_cubin = NULL;
    if (!secant_sr_cubin_cache_lookup(
            &cache,
            &mixed_recipe.header,
            12,
            0,
            12,
            9,
            1,
            1,
            &cached_cubin,
            &cached_cubin_size,
            &compile_seconds,
            &hit) ||
        !hit || fabs(compile_seconds - 5.5) > 1.0e-12) {
        goto cleanup;
    }
    free(cached_cubin);
    cached_cubin = NULL;
    if (!secant_sr_cubin_cache_artifact_lookup(
            &cache, "mse_reducer", 1u, 12, 0, 12, 9, 1, 1,
            &cached_cubin, &cached_cubin_size, &compile_seconds, &hit) ||
        hit || cached_cubin != NULL ||
        !secant_sr_cubin_cache_artifact_store(
            &cache, "mse_reducer", 1u, 12, 0, 12, 9, 1, 1, cubin, sizeof(cubin), 4.5) ||
        !secant_sr_cubin_cache_artifact_lookup(
            &cache, "mse_reducer", 2u, 12, 0, 12, 9, 1, 1,
            &cached_cubin, &cached_cubin_size, &compile_seconds, &hit) ||
        hit || cached_cubin != NULL ||
        !secant_sr_cubin_cache_artifact_lookup(
            &cache, "mse_reducer", 1u, 12, 0, 12, 9, 1, 1,
            &cached_cubin, &cached_cubin_size, &compile_seconds, &hit) ||
        !hit || cached_cubin_size != sizeof(cubin) || memcmp(cached_cubin, cubin, sizeof(cubin)) != 0 ||
        fabs(compile_seconds - 4.5) > 1.0e-12) {
        goto cleanup;
    }
    free(cached_cubin);
    cached_cubin = NULL;
    secant_sr_cubin_cache_close(&cache);
    if (!secant_sr_cubin_cache_test_schema_check(path)) {
        goto cleanup_closed;
    }
    status = 0;
    goto cleanup_closed;

cleanup:
    free(cached_cubin);
    secant_sr_cubin_cache_close(&cache);
cleanup_closed:
    secant_sr_cubin_cache_test_files_remove(path);
    return status;
}
