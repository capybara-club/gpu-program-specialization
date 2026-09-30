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
#include "cubin_cache.h"

#include <sqlite3.h>

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SECANT_SR_CUBIN_CACHE_SCHEMA_VERSION 3

typedef struct SecantSRCubinCacheRecipeKey {
    uint32_t secant_version_major;
    uint32_t secant_version_minor;
    uint32_t secant_version_patch;
    uint32_t recipe_version;
    uint32_t recipe_flags;
    uint32_t shape;
    sqlite3_int64 num_kernels;
    sqlite3_int64 asts_per_kernel;
    sqlite3_int64 num_inputs;
    sqlite3_int64 num_input_columns;
    sqlite3_int64 num_input_constants;
    sqlite3_int64 num_dynamic_leaves;
    sqlite3_int64 num_targets;
    sqlite3_int64 tile_rows;
    sqlite3_int64 threads_per_block;
    sqlite3_int64 patch_capacity_instructions;
} SecantSRCubinCacheRecipeKey;

static const char secant_sr_cubin_cache_table_create_sql[] =
    "CREATE TABLE IF NOT EXISTS cubin_template ("
    "secant_version_major INTEGER NOT NULL,"
    "secant_version_minor INTEGER NOT NULL,"
    "secant_version_patch INTEGER NOT NULL,"
    "recipe_version INTEGER NOT NULL,"
    "recipe_flags INTEGER NOT NULL,"
    "shape INTEGER NOT NULL,"
    "num_kernels INTEGER NOT NULL,"
    "asts_per_kernel INTEGER NOT NULL,"
    "num_inputs INTEGER NOT NULL,"
    "num_input_columns INTEGER NOT NULL,"
    "num_input_constants INTEGER NOT NULL,"
    "num_dynamic_leaves INTEGER NOT NULL,"
    "num_targets INTEGER NOT NULL,"
    "tile_rows INTEGER NOT NULL,"
    "threads_per_block INTEGER NOT NULL,"
    "patch_capacity_instructions INTEGER NOT NULL,"
    "compute_capability_major INTEGER NOT NULL,"
    "compute_capability_minor INTEGER NOT NULL,"
    "nvrtc_major INTEGER NOT NULL,"
    "nvrtc_minor INTEGER NOT NULL,"
    "ptxas_opt_level INTEGER NOT NULL,"
    "nvrtc_no_cache INTEGER NOT NULL,"
    "cubin BLOB NOT NULL,"
    "compile_seconds REAL NOT NULL,"
    "created_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,"
    "PRIMARY KEY (secant_version_major, secant_version_minor, secant_version_patch, recipe_version, recipe_flags, "
    "shape, num_kernels, asts_per_kernel, num_inputs, num_input_columns, num_input_constants, num_dynamic_leaves, "
    "num_targets, tile_rows, threads_per_block, patch_capacity_instructions, compute_capability_major, "
    "compute_capability_minor, nvrtc_major, nvrtc_minor, ptxas_opt_level, nvrtc_no_cache)"
    ") WITHOUT ROWID;";

static const char secant_sr_cubin_cache_artifact_table_create_sql[] =
    "CREATE TABLE IF NOT EXISTS cubin_artifact ("
    "artifact_name TEXT NOT NULL,"
    "artifact_version INTEGER NOT NULL,"
    "compute_capability_major INTEGER NOT NULL,"
    "compute_capability_minor INTEGER NOT NULL,"
    "nvrtc_major INTEGER NOT NULL,"
    "nvrtc_minor INTEGER NOT NULL,"
    "ptxas_opt_level INTEGER NOT NULL,"
    "nvrtc_no_cache INTEGER NOT NULL,"
    "cubin BLOB NOT NULL,"
    "compile_seconds REAL NOT NULL,"
    "created_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,"
    "PRIMARY KEY (artifact_name, artifact_version, compute_capability_major, compute_capability_minor, nvrtc_major, "
    "nvrtc_minor, ptxas_opt_level, nvrtc_no_cache)"
    ") WITHOUT ROWID;";

static sqlite3*
secant_sr_cubin_cache_database_get(SecantSRCubinCache* cache) {
    return cache == NULL ? NULL : (sqlite3*)cache->database;
}

static int
secant_sr_cubin_cache_exec(sqlite3* database, const char* sql) {
    char* error = NULL;
    const int result = sqlite3_exec(database, sql, NULL, NULL, &error);

    if (result != SQLITE_OK) {
        fprintf(stderr, "CUBIN cache SQLite error: %s\n", error == NULL ? sqlite3_errmsg(database) : error);
        sqlite3_free(error);
        return 0;
    }
    return 1;
}

static int
secant_sr_cubin_cache_size_set(size_t value, sqlite3_int64* value_ret) {
#if SIZE_MAX > INT64_MAX
    if (value > (size_t)INT64_MAX) {
        return 0;
    }
#endif
    *value_ret = (sqlite3_int64)value;
    return 1;
}

static int
secant_sr_cubin_cache_recipe_key_get(
    const SecantCubinRecipeHeader* recipe,
    SecantSRCubinCacheRecipeKey* key_ret
) {
    if (recipe == NULL || key_ret == NULL || recipe->version != SECANT_CUBIN_RECIPE_VERSION_1 ||
        recipe->flags != 0u) {
        return 0;
    }
    memset(key_ret, 0, sizeof(*key_ret));
    key_ret->secant_version_major = SECANT_VERSION_MAJOR;
    key_ret->secant_version_minor = SECANT_VERSION_MINOR;
    key_ret->secant_version_patch = SECANT_VERSION_PATCH;
    key_ret->recipe_version = recipe->version;
    key_ret->recipe_flags = recipe->flags;
    key_ret->shape = recipe->shape;
    switch (recipe->shape) {
        case SECANT_KERNEL_SHAPE_STATIC_MATERIALIZE_F32: {
            const SecantCubinMaterializeRecipe* value;

            if (recipe->struct_size < sizeof(SecantCubinMaterializeRecipe)) {
                return 0;
            }
            value = (const SecantCubinMaterializeRecipe*)(const void*)recipe;
            return secant_sr_cubin_cache_size_set(value->num_kernels, &key_ret->num_kernels) &&
                secant_sr_cubin_cache_size_set(value->asts_per_kernel, &key_ret->asts_per_kernel) &&
                secant_sr_cubin_cache_size_set(value->num_inputs, &key_ret->num_inputs) &&
                secant_sr_cubin_cache_size_set(
                    value->patch_capacity_instructions, &key_ret->patch_capacity_instructions);
        }
        case SECANT_KERNEL_SHAPE_STATIC_SSE_F32: {
            const SecantCubinSSERecipe* value;

            if (recipe->struct_size < sizeof(SecantCubinSSERecipe)) {
                return 0;
            }
            value = (const SecantCubinSSERecipe*)(const void*)recipe;
            return secant_sr_cubin_cache_size_set(value->num_kernels, &key_ret->num_kernels) &&
                secant_sr_cubin_cache_size_set(value->asts_per_kernel, &key_ret->asts_per_kernel) &&
                secant_sr_cubin_cache_size_set(value->num_inputs, &key_ret->num_inputs) &&
                secant_sr_cubin_cache_size_set(value->num_targets, &key_ret->num_targets) &&
                secant_sr_cubin_cache_size_set(value->tile_rows, &key_ret->tile_rows) &&
                secant_sr_cubin_cache_size_set(value->threads_per_block, &key_ret->threads_per_block) &&
                secant_sr_cubin_cache_size_set(
                    value->patch_capacity_instructions, &key_ret->patch_capacity_instructions);
        }
        case SECANT_KERNEL_SHAPE_STATIC_AFFINE_STATS_F32: {
            const SecantCubinAffineStatsRecipe* value;

            if (recipe->struct_size < sizeof(SecantCubinAffineStatsRecipe)) {
                return 0;
            }
            value = (const SecantCubinAffineStatsRecipe*)(const void*)recipe;
            return secant_sr_cubin_cache_size_set(value->num_kernels, &key_ret->num_kernels) &&
                secant_sr_cubin_cache_size_set(value->asts_per_kernel, &key_ret->asts_per_kernel) &&
                secant_sr_cubin_cache_size_set(value->num_inputs, &key_ret->num_inputs) &&
                secant_sr_cubin_cache_size_set(value->num_targets, &key_ret->num_targets) &&
                secant_sr_cubin_cache_size_set(value->tile_rows, &key_ret->tile_rows) &&
                secant_sr_cubin_cache_size_set(value->threads_per_block, &key_ret->threads_per_block) &&
                secant_sr_cubin_cache_size_set(
                    value->patch_capacity_instructions, &key_ret->patch_capacity_instructions);
        }
        case SECANT_KERNEL_SHAPE_STATIC_GRAM_STATS_F32: {
            const SecantCubinGramStatsRecipe* value;

            if (recipe->struct_size < sizeof(SecantCubinGramStatsRecipe)) {
                return 0;
            }
            value = (const SecantCubinGramStatsRecipe*)(const void*)recipe;
            return secant_sr_cubin_cache_size_set(value->num_kernels, &key_ret->num_kernels) &&
                secant_sr_cubin_cache_size_set(value->asts_per_kernel, &key_ret->asts_per_kernel) &&
                secant_sr_cubin_cache_size_set(value->num_inputs, &key_ret->num_inputs) &&
                secant_sr_cubin_cache_size_set(value->num_targets, &key_ret->num_targets) &&
                secant_sr_cubin_cache_size_set(value->tile_rows, &key_ret->tile_rows) &&
                secant_sr_cubin_cache_size_set(value->threads_per_block, &key_ret->threads_per_block) &&
                secant_sr_cubin_cache_size_set(
                    value->patch_capacity_instructions, &key_ret->patch_capacity_instructions);
        }
        case SECANT_KERNEL_SHAPE_DYNAMIC_CONSTANT_SSE_F32: {
            const SecantCubinDynamicConstantSSERecipe* value;

            if (recipe->struct_size < sizeof(SecantCubinDynamicConstantSSERecipe)) {
                return 0;
            }
            value = (const SecantCubinDynamicConstantSSERecipe*)(const void*)recipe;
            return secant_sr_cubin_cache_size_set(value->num_kernels, &key_ret->num_kernels) &&
                secant_sr_cubin_cache_size_set(value->asts_per_kernel, &key_ret->asts_per_kernel) &&
                secant_sr_cubin_cache_size_set(value->num_input_columns, &key_ret->num_input_columns) &&
                secant_sr_cubin_cache_size_set(value->num_input_constants, &key_ret->num_input_constants) &&
                secant_sr_cubin_cache_size_set(value->num_targets, &key_ret->num_targets) &&
                secant_sr_cubin_cache_size_set(value->tile_rows, &key_ret->tile_rows) &&
                secant_sr_cubin_cache_size_set(value->threads_per_block, &key_ret->threads_per_block) &&
                secant_sr_cubin_cache_size_set(
                    value->patch_capacity_instructions, &key_ret->patch_capacity_instructions);
        }
        case SECANT_KERNEL_SHAPE_DYNAMIC_LEAF_SSE_F32: {
            const SecantCubinDynamicLeafSSERecipe* value;

            if (recipe->struct_size < sizeof(SecantCubinDynamicLeafSSERecipe)) {
                return 0;
            }
            value = (const SecantCubinDynamicLeafSSERecipe*)(const void*)recipe;
            return secant_sr_cubin_cache_size_set(value->num_kernels, &key_ret->num_kernels) &&
                secant_sr_cubin_cache_size_set(value->asts_per_kernel, &key_ret->asts_per_kernel) &&
                secant_sr_cubin_cache_size_set(value->num_static_input_columns, &key_ret->num_inputs) &&
                secant_sr_cubin_cache_size_set(value->num_input_columns, &key_ret->num_input_columns) &&
                secant_sr_cubin_cache_size_set(value->num_dynamic_leaves, &key_ret->num_dynamic_leaves) &&
                secant_sr_cubin_cache_size_set(value->num_targets, &key_ret->num_targets) &&
                secant_sr_cubin_cache_size_set(value->tile_rows, &key_ret->tile_rows) &&
                secant_sr_cubin_cache_size_set(value->threads_per_block, &key_ret->threads_per_block) &&
                secant_sr_cubin_cache_size_set(
                    value->patch_capacity_instructions, &key_ret->patch_capacity_instructions);
        }
        case SECANT_KERNEL_SHAPE_CONSTANT_OPTIMIZER_SSE_F32: {
            const SecantCubinConstantOptimizerSSERecipe* value;

            if (recipe->struct_size < sizeof(SecantCubinConstantOptimizerSSERecipe)) {
                return 0;
            }
            value = (const SecantCubinConstantOptimizerSSERecipe*)(const void*)recipe;
            return secant_sr_cubin_cache_size_set(value->num_kernels, &key_ret->num_kernels) &&
                secant_sr_cubin_cache_size_set(value->num_input_columns, &key_ret->num_input_columns) &&
                secant_sr_cubin_cache_size_set(value->num_input_constants, &key_ret->num_input_constants) &&
                secant_sr_cubin_cache_size_set(value->tile_rows, &key_ret->tile_rows) &&
                secant_sr_cubin_cache_size_set(value->threads_per_block, &key_ret->threads_per_block) &&
                secant_sr_cubin_cache_size_set(
                    value->patch_capacity_instructions, &key_ret->patch_capacity_instructions);
        }
        case SECANT_KERNEL_SHAPE_PACKED_CONSTANT_OPTIMIZER_SSE_F32: {
            const SecantCubinPackedConstantOptimizerSSERecipe* value;

            if (recipe->struct_size < sizeof(SecantCubinPackedConstantOptimizerSSERecipe)) {
                return 0;
            }
            value = (const SecantCubinPackedConstantOptimizerSSERecipe*)(const void*)recipe;
            return secant_sr_cubin_cache_size_set(value->num_kernels, &key_ret->num_kernels) &&
                secant_sr_cubin_cache_size_set(value->asts_per_kernel, &key_ret->asts_per_kernel) &&
                secant_sr_cubin_cache_size_set(value->num_input_columns, &key_ret->num_input_columns) &&
                secant_sr_cubin_cache_size_set(value->num_input_constants, &key_ret->num_input_constants) &&
                secant_sr_cubin_cache_size_set(value->tile_rows, &key_ret->tile_rows) &&
                secant_sr_cubin_cache_size_set(value->threads_per_block, &key_ret->threads_per_block) &&
                secant_sr_cubin_cache_size_set(
                    value->patch_capacity_instructions, &key_ret->patch_capacity_instructions);
        }
        default:
            return 0;
    }
}

static int
secant_sr_cubin_cache_key_bind(
    sqlite3_stmt* statement,
    const SecantSRCubinCacheRecipeKey* key,
    int compute_capability_major,
    int compute_capability_minor,
    int nvrtc_major,
    int nvrtc_minor,
    int ptxas_opt_level,
    int nvrtc_no_cache
) {
    return sqlite3_bind_int64(statement, 1, (sqlite3_int64)key->secant_version_major) == SQLITE_OK &&
        sqlite3_bind_int64(statement, 2, (sqlite3_int64)key->secant_version_minor) == SQLITE_OK &&
        sqlite3_bind_int64(statement, 3, (sqlite3_int64)key->secant_version_patch) == SQLITE_OK &&
        sqlite3_bind_int64(statement, 4, (sqlite3_int64)key->recipe_version) == SQLITE_OK &&
        sqlite3_bind_int64(statement, 5, (sqlite3_int64)key->recipe_flags) == SQLITE_OK &&
        sqlite3_bind_int64(statement, 6, (sqlite3_int64)key->shape) == SQLITE_OK &&
        sqlite3_bind_int64(statement, 7, key->num_kernels) == SQLITE_OK &&
        sqlite3_bind_int64(statement, 8, key->asts_per_kernel) == SQLITE_OK &&
        sqlite3_bind_int64(statement, 9, key->num_inputs) == SQLITE_OK &&
        sqlite3_bind_int64(statement, 10, key->num_input_columns) == SQLITE_OK &&
        sqlite3_bind_int64(statement, 11, key->num_input_constants) == SQLITE_OK &&
        sqlite3_bind_int64(statement, 12, key->num_dynamic_leaves) == SQLITE_OK &&
        sqlite3_bind_int64(statement, 13, key->num_targets) == SQLITE_OK &&
        sqlite3_bind_int64(statement, 14, key->tile_rows) == SQLITE_OK &&
        sqlite3_bind_int64(statement, 15, key->threads_per_block) == SQLITE_OK &&
        sqlite3_bind_int64(statement, 16, key->patch_capacity_instructions) == SQLITE_OK &&
        sqlite3_bind_int(statement, 17, compute_capability_major) == SQLITE_OK &&
        sqlite3_bind_int(statement, 18, compute_capability_minor) == SQLITE_OK &&
        sqlite3_bind_int(statement, 19, nvrtc_major) == SQLITE_OK &&
        sqlite3_bind_int(statement, 20, nvrtc_minor) == SQLITE_OK &&
        sqlite3_bind_int(statement, 21, ptxas_opt_level) == SQLITE_OK &&
        sqlite3_bind_int(statement, 22, nvrtc_no_cache) == SQLITE_OK;
}

static int
secant_sr_cubin_cache_schema_reset(sqlite3* database) {
    int success = 0;

    if (!secant_sr_cubin_cache_exec(database, "BEGIN IMMEDIATE;") ||
        !secant_sr_cubin_cache_exec(database, "DROP TABLE IF EXISTS cubin_template;") ||
        !secant_sr_cubin_cache_exec(database, "DROP TABLE IF EXISTS cubin_artifact;") ||
        !secant_sr_cubin_cache_exec(database, secant_sr_cubin_cache_table_create_sql) ||
        !secant_sr_cubin_cache_exec(database, secant_sr_cubin_cache_artifact_table_create_sql) ||
        !secant_sr_cubin_cache_exec(database, "PRAGMA user_version=3;") ||
        !secant_sr_cubin_cache_exec(database, "COMMIT;")) {
        (void)sqlite3_exec(database, "ROLLBACK;", NULL, NULL, NULL);
        return 0;
    }
    success = 1;
    return success;
}

int
secant_sr_cubin_cache_open(const char* path, SecantSRCubinCache* cache) {
    sqlite3* database = NULL;
    sqlite3_stmt* statement = NULL;
    int schema_version = 0;
    int reset = 0;

    if (path == NULL || path[0] == '\0' || cache == NULL) {
        return 0;
    }
    memset(cache, 0, sizeof(*cache));
    if (sqlite3_open_v2(
            path,
            &database,
            SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX,
            NULL) != SQLITE_OK) {
        fprintf(stderr, "CUBIN cache open failed: %s\n", database == NULL ? "unknown error" : sqlite3_errmsg(database));
        if (database != NULL) {
            (void)sqlite3_close(database);
        }
        return 0;
    }
    (void)sqlite3_busy_timeout(database, 30000);
    if (!secant_sr_cubin_cache_exec(database, "PRAGMA journal_mode=WAL;") ||
        !secant_sr_cubin_cache_exec(database, "PRAGMA synchronous=NORMAL;") ||
        sqlite3_prepare_v2(database, "PRAGMA user_version;", -1, &statement, NULL) != SQLITE_OK ||
        sqlite3_step(statement) != SQLITE_ROW) {
        fprintf(stderr, "CUBIN cache initialization failed: %s\n", sqlite3_errmsg(database));
        if (statement != NULL) {
            (void)sqlite3_finalize(statement);
        }
        (void)sqlite3_close(database);
        return 0;
    }
    schema_version = sqlite3_column_int(statement, 0);
    (void)sqlite3_finalize(statement);
    statement = NULL;
    if (schema_version == 1 || schema_version == 2) {
        if (!secant_sr_cubin_cache_schema_reset(database)) {
            fprintf(stderr, "CUBIN cache schema reset failed: %s\n", sqlite3_errmsg(database));
            (void)sqlite3_close(database);
            return 0;
        }
        schema_version = SECANT_SR_CUBIN_CACHE_SCHEMA_VERSION;
        reset = 1;
    }
    if (schema_version != 0 && schema_version != SECANT_SR_CUBIN_CACHE_SCHEMA_VERSION) {
        fprintf(stderr, "unsupported CUBIN cache schema version: %d\n", schema_version);
        (void)sqlite3_close(database);
        return 0;
    }
    if (!secant_sr_cubin_cache_exec(database, secant_sr_cubin_cache_table_create_sql) ||
        !secant_sr_cubin_cache_exec(database, secant_sr_cubin_cache_artifact_table_create_sql) ||
        (schema_version == 0 && !secant_sr_cubin_cache_exec(database, "PRAGMA user_version=3;"))) {
        (void)sqlite3_close(database);
        return 0;
    }
    if (reset) {
        (void)secant_sr_cubin_cache_exec(database, "PRAGMA wal_checkpoint(TRUNCATE);");
        (void)secant_sr_cubin_cache_exec(database, "VACUUM;");
    }
    cache->database = database;
    return 1;
}

int
secant_sr_cubin_cache_lookup(
    SecantSRCubinCache* cache,
    const SecantCubinRecipeHeader* recipe,
    int compute_capability_major,
    int compute_capability_minor,
    int nvrtc_major,
    int nvrtc_minor,
    int ptxas_opt_level,
    int nvrtc_no_cache,
    unsigned char** cubin_ret,
    size_t* cubin_size_ret,
    double* compile_seconds_ret,
    int* hit_ret
) {
    static const char sql[] =
        "SELECT cubin, compile_seconds FROM cubin_template WHERE "
        "secant_version_major=?1 AND secant_version_minor=?2 AND secant_version_patch=?3 AND "
        "recipe_version=?4 AND recipe_flags=?5 AND shape=?6 AND num_kernels=?7 AND asts_per_kernel=?8 AND "
        "num_inputs=?9 AND num_input_columns=?10 AND num_input_constants=?11 AND num_dynamic_leaves=?12 AND "
        "num_targets=?13 AND tile_rows=?14 AND threads_per_block=?15 AND patch_capacity_instructions=?16 AND "
        "compute_capability_major=?17 AND compute_capability_minor=?18 AND nvrtc_major=?19 AND nvrtc_minor=?20 AND "
        "ptxas_opt_level=?21 AND nvrtc_no_cache=?22;";
    sqlite3* database = secant_sr_cubin_cache_database_get(cache);
    sqlite3_stmt* statement = NULL;
    SecantSRCubinCacheRecipeKey key;
    int step_result;

    if (database == NULL || !secant_sr_cubin_cache_recipe_key_get(recipe, &key) || cubin_ret == NULL ||
        cubin_size_ret == NULL || compile_seconds_ret == NULL || hit_ret == NULL) {
        return 0;
    }
    *cubin_ret = NULL;
    *cubin_size_ret = 0u;
    *compile_seconds_ret = 0.0;
    *hit_ret = 0;
    if (sqlite3_prepare_v2(database, sql, -1, &statement, NULL) != SQLITE_OK ||
        !secant_sr_cubin_cache_key_bind(
            statement,
            &key,
            compute_capability_major,
            compute_capability_minor,
            nvrtc_major,
            nvrtc_minor,
            ptxas_opt_level,
            nvrtc_no_cache)) {
        fprintf(stderr, "CUBIN cache lookup setup failed: %s\n", sqlite3_errmsg(database));
        if (statement != NULL) {
            (void)sqlite3_finalize(statement);
        }
        return 0;
    }
    step_result = sqlite3_step(statement);
    if (step_result == SQLITE_ROW) {
        const void* stored_cubin = sqlite3_column_blob(statement, 0);
        const int stored_cubin_size = sqlite3_column_bytes(statement, 0);
        const double stored_compile_seconds = sqlite3_column_double(statement, 1);
        unsigned char* cubin;

        if (stored_cubin == NULL || stored_cubin_size <= 0 || stored_compile_seconds < 0.0) {
            (void)sqlite3_finalize(statement);
            return 0;
        }
        cubin = malloc((size_t)stored_cubin_size);
        if (cubin == NULL) {
            (void)sqlite3_finalize(statement);
            return 0;
        }
        memcpy(cubin, stored_cubin, (size_t)stored_cubin_size);
        *cubin_ret = cubin;
        *cubin_size_ret = (size_t)stored_cubin_size;
        *compile_seconds_ret = stored_compile_seconds;
        *hit_ret = 1;
    } else if (step_result != SQLITE_DONE) {
        fprintf(stderr, "CUBIN cache lookup failed: %s\n", sqlite3_errmsg(database));
        (void)sqlite3_finalize(statement);
        return 0;
    }
    (void)sqlite3_finalize(statement);
    return 1;
}

int
secant_sr_cubin_cache_store(
    SecantSRCubinCache* cache,
    const SecantCubinRecipeHeader* recipe,
    int compute_capability_major,
    int compute_capability_minor,
    int nvrtc_major,
    int nvrtc_minor,
    int ptxas_opt_level,
    int nvrtc_no_cache,
    const unsigned char* cubin,
    size_t cubin_size,
    double compile_seconds
) {
    static const char sql[] =
        "INSERT OR REPLACE INTO cubin_template ("
        "secant_version_major, secant_version_minor, secant_version_patch, recipe_version, recipe_flags, shape, "
        "num_kernels, asts_per_kernel, num_inputs, num_input_columns, num_input_constants, num_dynamic_leaves, "
        "num_targets, tile_rows, threads_per_block, patch_capacity_instructions, compute_capability_major, "
        "compute_capability_minor, nvrtc_major, nvrtc_minor, ptxas_opt_level, nvrtc_no_cache, cubin, compile_seconds"
        ") VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, "
        "?20, ?21, ?22, ?23, ?24);";
    sqlite3* database = secant_sr_cubin_cache_database_get(cache);
    sqlite3_stmt* statement = NULL;
    SecantSRCubinCacheRecipeKey key;
    int result;

    if (database == NULL || !secant_sr_cubin_cache_recipe_key_get(recipe, &key) || cubin == NULL || cubin_size == 0u ||
        compile_seconds < 0.0 || cubin_size > (size_t)INT_MAX) {
        return 0;
    }
    if (sqlite3_prepare_v2(database, sql, -1, &statement, NULL) != SQLITE_OK ||
        !secant_sr_cubin_cache_key_bind(
            statement,
            &key,
            compute_capability_major,
            compute_capability_minor,
            nvrtc_major,
            nvrtc_minor,
            ptxas_opt_level,
            nvrtc_no_cache) ||
        sqlite3_bind_blob(statement, 23, cubin, (int)cubin_size, SQLITE_TRANSIENT) != SQLITE_OK ||
        sqlite3_bind_double(statement, 24, compile_seconds) != SQLITE_OK) {
        fprintf(stderr, "CUBIN cache store setup failed: %s\n", sqlite3_errmsg(database));
        if (statement != NULL) {
            (void)sqlite3_finalize(statement);
        }
        return 0;
    }
    result = sqlite3_step(statement);
    if (result != SQLITE_DONE) {
        fprintf(stderr, "CUBIN cache store failed: %s\n", sqlite3_errmsg(database));
        (void)sqlite3_finalize(statement);
        return 0;
    }
    (void)sqlite3_finalize(statement);
    return 1;
}

int
secant_sr_cubin_cache_artifact_lookup(
    SecantSRCubinCache* cache,
    const char* artifact_name,
    uint32_t artifact_version,
    int compute_capability_major,
    int compute_capability_minor,
    int nvrtc_major,
    int nvrtc_minor,
    int ptxas_opt_level,
    int nvrtc_no_cache,
    unsigned char** cubin_ret,
    size_t* cubin_size_ret,
    double* compile_seconds_ret,
    int* hit_ret
) {
    static const char sql[] =
        "SELECT cubin, compile_seconds FROM cubin_artifact WHERE artifact_name=?1 AND artifact_version=?2 AND "
        "compute_capability_major=?3 AND compute_capability_minor=?4 AND nvrtc_major=?5 AND nvrtc_minor=?6 AND "
        "ptxas_opt_level=?7 AND nvrtc_no_cache=?8;";
    sqlite3* database = secant_sr_cubin_cache_database_get(cache);
    sqlite3_stmt* statement = NULL;
    int step_result;

    if (database == NULL || artifact_name == NULL || artifact_name[0] == '\0' || artifact_version == 0u ||
        cubin_ret == NULL || cubin_size_ret == NULL || compile_seconds_ret == NULL || hit_ret == NULL) {
        return 0;
    }
    *cubin_ret = NULL;
    *cubin_size_ret = 0u;
    *compile_seconds_ret = 0.0;
    *hit_ret = 0;
    if (sqlite3_prepare_v2(database, sql, -1, &statement, NULL) != SQLITE_OK ||
        sqlite3_bind_text(statement, 1, artifact_name, -1, SQLITE_STATIC) != SQLITE_OK ||
        sqlite3_bind_int64(statement, 2, (sqlite3_int64)artifact_version) != SQLITE_OK ||
        sqlite3_bind_int(statement, 3, compute_capability_major) != SQLITE_OK ||
        sqlite3_bind_int(statement, 4, compute_capability_minor) != SQLITE_OK ||
        sqlite3_bind_int(statement, 5, nvrtc_major) != SQLITE_OK ||
        sqlite3_bind_int(statement, 6, nvrtc_minor) != SQLITE_OK ||
        sqlite3_bind_int(statement, 7, ptxas_opt_level) != SQLITE_OK ||
        sqlite3_bind_int(statement, 8, nvrtc_no_cache) != SQLITE_OK) {
        fprintf(stderr, "CUBIN artifact cache lookup setup failed: %s\n", sqlite3_errmsg(database));
        if (statement != NULL) {
            (void)sqlite3_finalize(statement);
        }
        return 0;
    }
    step_result = sqlite3_step(statement);
    if (step_result == SQLITE_ROW) {
        const void* stored_cubin = sqlite3_column_blob(statement, 0);
        const int stored_cubin_size = sqlite3_column_bytes(statement, 0);
        const double stored_compile_seconds = sqlite3_column_double(statement, 1);
        unsigned char* cubin;

        if (stored_cubin == NULL || stored_cubin_size <= 0 || stored_compile_seconds < 0.0) {
            (void)sqlite3_finalize(statement);
            return 0;
        }
        cubin = malloc((size_t)stored_cubin_size);
        if (cubin == NULL) {
            (void)sqlite3_finalize(statement);
            return 0;
        }
        memcpy(cubin, stored_cubin, (size_t)stored_cubin_size);
        *cubin_ret = cubin;
        *cubin_size_ret = (size_t)stored_cubin_size;
        *compile_seconds_ret = stored_compile_seconds;
        *hit_ret = 1;
    } else if (step_result != SQLITE_DONE) {
        fprintf(stderr, "CUBIN artifact cache lookup failed: %s\n", sqlite3_errmsg(database));
        (void)sqlite3_finalize(statement);
        return 0;
    }
    (void)sqlite3_finalize(statement);
    return 1;
}

int
secant_sr_cubin_cache_artifact_store(
    SecantSRCubinCache* cache,
    const char* artifact_name,
    uint32_t artifact_version,
    int compute_capability_major,
    int compute_capability_minor,
    int nvrtc_major,
    int nvrtc_minor,
    int ptxas_opt_level,
    int nvrtc_no_cache,
    const unsigned char* cubin,
    size_t cubin_size,
    double compile_seconds
) {
    static const char sql[] =
        "INSERT OR REPLACE INTO cubin_artifact (artifact_name, artifact_version, compute_capability_major, "
        "compute_capability_minor, nvrtc_major, nvrtc_minor, ptxas_opt_level, nvrtc_no_cache, cubin, "
        "compile_seconds) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10);";
    sqlite3* database = secant_sr_cubin_cache_database_get(cache);
    sqlite3_stmt* statement = NULL;
    int result;

    if (database == NULL || artifact_name == NULL || artifact_name[0] == '\0' || artifact_version == 0u ||
        cubin == NULL || cubin_size == 0u || cubin_size > (size_t)INT_MAX || compile_seconds < 0.0) {
        return 0;
    }
    if (sqlite3_prepare_v2(database, sql, -1, &statement, NULL) != SQLITE_OK ||
        sqlite3_bind_text(statement, 1, artifact_name, -1, SQLITE_STATIC) != SQLITE_OK ||
        sqlite3_bind_int64(statement, 2, (sqlite3_int64)artifact_version) != SQLITE_OK ||
        sqlite3_bind_int(statement, 3, compute_capability_major) != SQLITE_OK ||
        sqlite3_bind_int(statement, 4, compute_capability_minor) != SQLITE_OK ||
        sqlite3_bind_int(statement, 5, nvrtc_major) != SQLITE_OK ||
        sqlite3_bind_int(statement, 6, nvrtc_minor) != SQLITE_OK ||
        sqlite3_bind_int(statement, 7, ptxas_opt_level) != SQLITE_OK ||
        sqlite3_bind_int(statement, 8, nvrtc_no_cache) != SQLITE_OK ||
        sqlite3_bind_blob(statement, 9, cubin, (int)cubin_size, SQLITE_TRANSIENT) != SQLITE_OK ||
        sqlite3_bind_double(statement, 10, compile_seconds) != SQLITE_OK) {
        fprintf(stderr, "CUBIN artifact cache store setup failed: %s\n", sqlite3_errmsg(database));
        if (statement != NULL) {
            (void)sqlite3_finalize(statement);
        }
        return 0;
    }
    result = sqlite3_step(statement);
    if (result != SQLITE_DONE) {
        fprintf(stderr, "CUBIN artifact cache store failed: %s\n", sqlite3_errmsg(database));
        (void)sqlite3_finalize(statement);
        return 0;
    }
    (void)sqlite3_finalize(statement);
    return 1;
}

void
secant_sr_cubin_cache_close(SecantSRCubinCache* cache) {
    sqlite3* database = secant_sr_cubin_cache_database_get(cache);

    if (database != NULL) {
        (void)sqlite3_close(database);
    }
    if (cache != NULL) {
        memset(cache, 0, sizeof(*cache));
    }
}
