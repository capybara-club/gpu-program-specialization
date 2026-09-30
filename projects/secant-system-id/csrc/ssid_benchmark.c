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
#include "ssid_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct ssid_raw_benchmark_shared {
    const ssid_template *template_value;
    const ssid_genome_batch *batch;
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    uint32_t ready_count;
    int start;
    int status;
    char error[512];
} ssid_raw_benchmark_shared;

typedef struct ssid_raw_benchmark_worker {
    ssid_raw_benchmark_shared *shared;
    pthread_t thread;
    uint64_t module_count;
} ssid_raw_benchmark_worker;

static void ssid_raw_benchmark_fail(ssid_raw_benchmark_shared *shared, int status, const char *error) {
    pthread_mutex_lock(&shared->mutex);
    if (shared->status == SSID_OK) {
        shared->status = status;
        snprintf(shared->error, sizeof(shared->error), "%s", error == NULL ? "raw specialization benchmark failed" : error);
    }
    pthread_mutex_unlock(&shared->mutex);
}

static void *ssid_raw_benchmark_worker_main(void *argument) {
    ssid_raw_benchmark_worker *worker = (ssid_raw_benchmark_worker *)argument;
    ssid_raw_benchmark_shared *shared = worker->shared;
    uint8_t *image = (uint8_t *)malloc(shared->template_value->cubin_byte_count);
    uint64_t index;
    pthread_mutex_lock(&shared->mutex);
    shared->ready_count += 1;
    pthread_cond_broadcast(&shared->condition);
    while (!shared->start) pthread_cond_wait(&shared->condition, &shared->mutex);
    pthread_mutex_unlock(&shared->mutex);
    if (image == NULL) {
        ssid_raw_benchmark_fail(shared, SSID_OUT_OF_MEMORY, "could not allocate a worker-owned benchmark CUBIN");
        return NULL;
    }
    for (index = 0; index < worker->module_count; ++index) {
        int status = ssid_specialize_module(shared->template_value, shared->batch, image, shared->template_value->cubin_byte_count, NULL);
        if (status != SSID_OK) {
            ssid_raw_benchmark_fail(shared, status, ssid_last_error());
            break;
        }
    }
    free(image);
    return NULL;
}

int ssid_benchmark_specialization_raw(const ssid_template *template_value, const ssid_genome_batch *batch, uint32_t thread_count, uint64_t module_count, ssid_raw_benchmark_result *result) {
    ssid_raw_benchmark_shared shared;
    ssid_raw_benchmark_worker *workers;
    uint32_t created = 0;
    uint32_t index;
    double started;
    double elapsed;
    int status;
    if (result != NULL) memset(result, 0, sizeof(*result));
    if (template_value == NULL || batch == NULL || result == NULL || thread_count == 0 || module_count == 0) {
        ssid_set_error("invalid raw specialization benchmark configuration");
        return SSID_INVALID_ARGUMENT;
    }
    if ((status = ssid_validate_batch(template_value, batch)) != SSID_OK) return status;
    workers = (ssid_raw_benchmark_worker *)calloc(thread_count, sizeof(*workers));
    if (workers == NULL) return SSID_OUT_OF_MEMORY;
    memset(&shared, 0, sizeof(shared));
    shared.template_value = template_value;
    shared.batch = batch;
    shared.status = SSID_OK;
    pthread_mutex_init(&shared.mutex, NULL);
    pthread_cond_init(&shared.condition, NULL);
    for (index = 0; index < thread_count; ++index) {
        workers[index].shared = &shared;
        workers[index].module_count = module_count / thread_count + (index < module_count % thread_count ? 1u : 0u);
        if (pthread_create(&workers[index].thread, NULL, ssid_raw_benchmark_worker_main, &workers[index]) != 0) {
            ssid_set_error("could not create raw specialization benchmark worker %u", index);
            status = SSID_INTERNAL_ERROR;
            break;
        }
        created += 1;
    }
    pthread_mutex_lock(&shared.mutex);
    while (shared.ready_count < created) pthread_cond_wait(&shared.condition, &shared.mutex);
    started = ssid_monotonic_seconds();
    shared.start = 1;
    pthread_cond_broadcast(&shared.condition);
    pthread_mutex_unlock(&shared.mutex);
    for (index = 0; index < created; ++index) pthread_join(workers[index].thread, NULL);
    elapsed = ssid_monotonic_seconds() - started;
    if (created != thread_count && status == SSID_OK) status = SSID_INTERNAL_ERROR;
    if (status == SSID_OK) status = shared.status;
    if (status == SSID_OK) {
        result->thread_count = thread_count;
        result->module_count = module_count;
        result->genome_count = module_count * batch->genome_count;
        result->ast_count = module_count * batch->ast_count;
        result->cubin_bytes_copied = module_count * template_value->cubin_byte_count;
        result->wall_seconds = elapsed;
        result->modules_per_second = module_count / elapsed;
        result->genomes_per_second = result->genome_count / elapsed;
        result->asts_per_second = result->ast_count / elapsed;
        result->cubin_gib_per_second = result->cubin_bytes_copied / elapsed / (1024.0 * 1024.0 * 1024.0);
    } else if (shared.error[0] != '\0') {
        ssid_set_error("%s", shared.error);
    }
    pthread_cond_destroy(&shared.condition);
    pthread_mutex_destroy(&shared.mutex);
    free(workers);
    return status;
}
