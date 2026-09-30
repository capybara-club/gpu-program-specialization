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
#define _GNU_SOURCE

#include <cuda.h>

#include <dirent.h>
#include <errno.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* CUDA 13 maps cuCtxCreate to the four-argument v4 entry point and hides the
 * older declaration, but the v2 ABI remains exported by current and older
 * drivers.  Using it keeps this probe buildable with the CUDA 12 toolkits
 * commonly installed on Grace Hopper machines. */
extern CUresult CUDAAPI cuCtxCreate_v2(
    CUcontext* context_ret,
    unsigned int flags,
    CUdevice device);

typedef enum ContextMode {
    CONTEXT_MODE_SHARED = 0,
    CONTEXT_MODE_SEPARATE = 1
} ContextMode;

typedef enum EnumerationMode {
    ENUMERATION_NONE = 0,
    ENUMERATION_HANDLES = 1,
    ENUMERATION_NAMES = 2
} EnumerationMode;

typedef struct CubinImage {
    char* path;
    unsigned char* data;
    size_t size;
} CubinImage;

typedef struct BenchmarkState BenchmarkState;

typedef struct LoaderWorker {
    BenchmarkState* state;
    size_t index;
    CUcontext context;
    pthread_t thread;
    CUmodule module;
    CUfunction* functions;
    double driver_load_seconds;
    double enumeration_seconds;
    double unload_seconds;
} LoaderWorker;

struct BenchmarkState {
    CubinImage* images;
    size_t image_count;
    size_t loader_count;
    size_t wave;
    size_t function_capacity;
    EnumerationMode enumeration_mode;
    pthread_barrier_t start_barrier;
    pthread_barrier_t loaded_barrier;
    pthread_barrier_t unload_start_barrier;
    pthread_barrier_t unloaded_barrier;
    atomic_int stop;
    atomic_int failed;
};

typedef struct Options {
    CubinImage* images;
    size_t image_count;
    size_t image_capacity;
    size_t loaders;
    size_t waves;
    size_t warmup_waves;
    int device_ordinal;
    ContextMode context_mode;
    EnumerationMode enumeration_mode;
} Options;

static double now_seconds(void) {
    struct timespec value;

    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        perror("clock_gettime");
        exit(1);
    }
    return (double)value.tv_sec + (double)value.tv_nsec * 1.0e-9;
}

static int size_parse(const char* text, size_t* value_ret) {
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

static int int_parse(const char* text, int* value_ret) {
    char* end = NULL;
    long value;

    errno = 0;
    value = strtol(text, &end, 10);
    if (errno != 0 || text == end || *end != '\0' || value < 0 || value > INT32_MAX) {
        return 0;
    }
    *value_ret = (int)value;
    return 1;
}

static int has_cubin_suffix(const char* name) {
    const size_t length = strlen(name);
    static const char suffix[] = ".cubin";

    return length >= sizeof(suffix) - 1u &&
        strcmp(name + length - (sizeof(suffix) - 1u), suffix) == 0;
}

static int image_path_compare(const void* left, const void* right) {
    const CubinImage* a = (const CubinImage*)left;
    const CubinImage* b = (const CubinImage*)right;

    return strcmp(a->path, b->path);
}

static int options_add_image_path(Options* options, const char* path) {
    CubinImage* grown;
    char* copy;

    if (options->image_count == options->image_capacity) {
        size_t capacity = options->image_capacity == 0u ? 16u : options->image_capacity * 2u;

        if (capacity < options->image_capacity || capacity > SIZE_MAX / sizeof(*grown)) {
            return 0;
        }
        grown = (CubinImage*)realloc(options->images, capacity * sizeof(*grown));
        if (grown == NULL) {
            return 0;
        }
        memset(
            grown + options->image_capacity,
            0,
            (capacity - options->image_capacity) * sizeof(*grown));
        options->images = grown;
        options->image_capacity = capacity;
    }
    copy = strdup(path);
    if (copy == NULL) {
        return 0;
    }
    options->images[options->image_count].path = copy;
    options->image_count += 1u;
    return 1;
}

static int options_add_directory(Options* options, const char* path) {
    DIR* directory = opendir(path);
    struct dirent* entry;
    int okay = 1;

    if (directory == NULL) {
        fprintf(stderr, "cannot open CUBIN directory %s: %s\n", path, strerror(errno));
        return 0;
    }
    while ((entry = readdir(directory)) != NULL) {
        char* full_path;
        size_t required;

        if (!has_cubin_suffix(entry->d_name)) {
            continue;
        }
        required = strlen(path) + 1u + strlen(entry->d_name) + 1u;
        full_path = (char*)malloc(required);
        if (full_path == NULL) {
            okay = 0;
            break;
        }
        if (snprintf(full_path, required, "%s/%s", path, entry->d_name) < 0 ||
            !options_add_image_path(options, full_path)) {
            okay = 0;
        }
        free(full_path);
        if (!okay) {
            break;
        }
    }
    if (closedir(directory) != 0) {
        okay = 0;
    }
    return okay;
}

static int image_read(CubinImage* image) {
    FILE* file = fopen(image->path, "rb");
    long length;

    if (file == NULL) {
        fprintf(stderr, "cannot open %s: %s\n", image->path, strerror(errno));
        return 0;
    }
    if (fseek(file, 0, SEEK_END) != 0 || (length = ftell(file)) <= 0 ||
        fseek(file, 0, SEEK_SET) != 0) {
        fprintf(stderr, "cannot size %s\n", image->path);
        fclose(file);
        return 0;
    }
    image->data = (unsigned char*)calloc((size_t)length + 16u, 1u);
    if (image->data == NULL ||
        fread(image->data, 1u, (size_t)length, file) != (size_t)length) {
        fprintf(stderr, "cannot read %s\n", image->path);
        free(image->data);
        image->data = NULL;
        fclose(file);
        return 0;
    }
    image->size = (size_t)length;
    if (fclose(file) != 0) {
        return 0;
    }
    return 1;
}

static void options_destroy(Options* options) {
    size_t index;

    for (index = 0u; index < options->image_count; ++index) {
        free(options->images[index].data);
        free(options->images[index].path);
    }
    free(options->images);
    memset(options, 0, sizeof(*options));
}

static void usage(const char* program) {
    fprintf(
        stderr,
        "usage: %s (--cubin FILE | --cubin-directory DIR) [options]\n"
        "  --loaders N                 persistent loader threads (default: 1)\n"
        "  --contexts shared|separate (default: shared)\n"
        "  --enumeration none|handles|names (default: names)\n"
        "  --waves N                   measured waves (default: 31)\n"
        "  --warmup-waves N            untimed waves (default: 3)\n"
        "  --device N                  CUDA device ordinal (default: 0)\n",
        program);
}

static int options_parse(int argc, char** argv, Options* options) {
    int arg_index;

    memset(options, 0, sizeof(*options));
    options->loaders = 1u;
    options->waves = 31u;
    options->warmup_waves = 3u;
    options->context_mode = CONTEXT_MODE_SHARED;
    options->enumeration_mode = ENUMERATION_NAMES;
    for (arg_index = 1; arg_index < argc; ++arg_index) {
        const char* argument = argv[arg_index];
        const char* value;

        if (strcmp(argument, "--help") == 0) {
            usage(argv[0]);
            return 0;
        }
        if (++arg_index >= argc) {
            usage(argv[0]);
            return -1;
        }
        value = argv[arg_index];
        if (strcmp(argument, "--cubin") == 0) {
            if (!options_add_image_path(options, value)) {
                return -1;
            }
        } else if (strcmp(argument, "--cubin-directory") == 0) {
            if (!options_add_directory(options, value)) {
                return -1;
            }
        } else if (strcmp(argument, "--loaders") == 0) {
            if (!size_parse(value, &options->loaders)) {
                return -1;
            }
        } else if (strcmp(argument, "--waves") == 0) {
            if (!size_parse(value, &options->waves)) {
                return -1;
            }
        } else if (strcmp(argument, "--warmup-waves") == 0) {
            if (!size_parse(value, &options->warmup_waves)) {
                return -1;
            }
        } else if (strcmp(argument, "--device") == 0) {
            if (!int_parse(value, &options->device_ordinal)) {
                return -1;
            }
        } else if (strcmp(argument, "--contexts") == 0) {
            if (strcmp(value, "shared") == 0) {
                options->context_mode = CONTEXT_MODE_SHARED;
            } else if (strcmp(value, "separate") == 0) {
                options->context_mode = CONTEXT_MODE_SEPARATE;
            } else {
                return -1;
            }
        } else if (strcmp(argument, "--enumeration") == 0) {
            if (strcmp(value, "none") == 0) {
                options->enumeration_mode = ENUMERATION_NONE;
            } else if (strcmp(value, "handles") == 0) {
                options->enumeration_mode = ENUMERATION_HANDLES;
            } else if (strcmp(value, "names") == 0) {
                options->enumeration_mode = ENUMERATION_NAMES;
            } else {
                return -1;
            }
        } else {
            usage(argv[0]);
            return -1;
        }
    }
    if (options->image_count == 0u || options->loaders > 64u) {
        usage(argv[0]);
        return -1;
    }
    qsort(options->images, options->image_count, sizeof(*options->images), image_path_compare);
    for (size_t image_index = 0u; image_index < options->image_count; ++image_index) {
        if (!image_read(&options->images[image_index])) {
            return -1;
        }
    }
    return 1;
}

static int cuda_check(CUresult result, const char* operation) {
    const char* name = NULL;
    const char* description = NULL;

    if (result == CUDA_SUCCESS) {
        return 1;
    }
    (void)cuGetErrorName(result, &name);
    (void)cuGetErrorString(result, &description);
    fprintf(
        stderr,
        "%s failed: %s (%s)\n",
        operation,
        name != NULL ? name : "unknown",
        description != NULL ? description : "no description");
    return 0;
}

static int barrier_wait(pthread_barrier_t* barrier) {
    const int result = pthread_barrier_wait(barrier);

    return result == 0 || result == PTHREAD_BARRIER_SERIAL_THREAD;
}

static void* loader_worker_main(void* argument) {
    LoaderWorker* worker = (LoaderWorker*)argument;
    BenchmarkState* state = worker->state;

    if (!cuda_check(cuCtxSetCurrent(worker->context), "cuCtxSetCurrent")) {
        atomic_store(&state->failed, 1);
    }
    for (;;) {
        CubinImage* image;
        double begin;
        unsigned int function_count = 0u;

        if (!barrier_wait(&state->start_barrier)) {
            atomic_store(&state->failed, 1);
            break;
        }
        if (atomic_load(&state->stop)) {
            break;
        }
        worker->module = NULL;
        worker->driver_load_seconds = 0.0;
        worker->enumeration_seconds = 0.0;
        worker->unload_seconds = 0.0;
        image = &state->images[
            (state->wave * state->loader_count + worker->index) % state->image_count];

        begin = now_seconds();
        if (!cuda_check(
                cuModuleLoadData(&worker->module, image->data),
                "cuModuleLoadData")) {
            atomic_store(&state->failed, 1);
        }
        worker->driver_load_seconds = now_seconds() - begin;

        begin = now_seconds();
        if (worker->module != NULL && state->enumeration_mode != ENUMERATION_NONE) {
            if (!cuda_check(
                    cuModuleGetFunctionCount(&function_count, worker->module),
                    "cuModuleGetFunctionCount") ||
                function_count > state->function_capacity ||
                !cuda_check(
                    cuModuleEnumerateFunctions(
                        worker->functions,
                        function_count,
                        worker->module),
                    "cuModuleEnumerateFunctions")) {
                atomic_store(&state->failed, 1);
            } else if (state->enumeration_mode == ENUMERATION_NAMES) {
                for (unsigned int function_index = 0u;
                     function_index < function_count;
                     ++function_index) {
                    const char* name = NULL;

                    if (!cuda_check(
                            cuFuncGetName(&name, worker->functions[function_index]),
                            "cuFuncGetName") ||
                        name == NULL || name[0] == '\0') {
                        atomic_store(&state->failed, 1);
                        break;
                    }
                }
            }
        }
        worker->enumeration_seconds = now_seconds() - begin;
        if (!barrier_wait(&state->loaded_barrier) ||
            !barrier_wait(&state->unload_start_barrier)) {
            atomic_store(&state->failed, 1);
            break;
        }

        begin = now_seconds();
        if (worker->module != NULL &&
            !cuda_check(cuModuleUnload(worker->module), "cuModuleUnload")) {
            atomic_store(&state->failed, 1);
        }
        worker->module = NULL;
        worker->unload_seconds = now_seconds() - begin;
        if (!barrier_wait(&state->unloaded_barrier)) {
            atomic_store(&state->failed, 1);
            break;
        }
    }
    (void)cuCtxSetCurrent(NULL);
    return NULL;
}

static int double_compare(const void* left, const void* right) {
    const double a = *(const double*)left;
    const double b = *(const double*)right;

    return (a > b) - (a < b);
}

static double percentile(double* values, size_t count, double fraction) {
    size_t index;

    qsort(values, count, sizeof(*values), double_compare);
    index = (size_t)(fraction * (double)(count - 1u));
    return values[index];
}

static const char* context_mode_name(ContextMode mode) {
    return mode == CONTEXT_MODE_SHARED ? "shared" : "separate";
}

static const char* enumeration_mode_name(EnumerationMode mode) {
    if (mode == ENUMERATION_NONE) {
        return "none";
    }
    return mode == ENUMERATION_HANDLES ? "handles" : "names";
}

int main(int argc, char** argv) {
    Options options;
    BenchmarkState state;
    LoaderWorker* workers = NULL;
    CUcontext* contexts = NULL;
    CUcontext shared_context = NULL;
    CUdevice device;
    CUmoduleLoadingMode loading_mode;
    unsigned int function_count = 0u;
    double* load_wave_seconds = NULL;
    double* unload_wave_seconds = NULL;
    double driver_load_sum = 0.0;
    double enumeration_sum = 0.0;
    double unload_sum = 0.0;
    double load_wall_sum = 0.0;
    double unload_wall_sum = 0.0;
    size_t image_min = SIZE_MAX;
    size_t image_max = 0u;
    int parse_result;
    int result = 1;

    parse_result = options_parse(argc, argv, &options);
    if (parse_result <= 0) {
        options_destroy(&options);
        return parse_result == 0 ? 0 : 2;
    }
    if (setenv("CUDA_MODULE_LOADING", "EAGER", 1) != 0 ||
        !cuda_check(cuInit(0u), "cuInit") ||
        !cuda_check(cuDeviceGet(&device, options.device_ordinal), "cuDeviceGet") ||
        !cuda_check(cuModuleGetLoadingMode(&loading_mode), "cuModuleGetLoadingMode") ||
        loading_mode != CU_MODULE_EAGER_LOADING) {
        fprintf(stderr, "CUDA eager module loading is required\n");
        goto cleanup;
    }

    workers = (LoaderWorker*)calloc(options.loaders, sizeof(*workers));
    contexts = (CUcontext*)calloc(options.loaders, sizeof(*contexts));
    load_wave_seconds = (double*)calloc(options.waves, sizeof(*load_wave_seconds));
    unload_wave_seconds = (double*)calloc(options.waves, sizeof(*unload_wave_seconds));
    if (workers == NULL || contexts == NULL ||
        load_wave_seconds == NULL || unload_wave_seconds == NULL) {
        goto cleanup;
    }

    if (options.context_mode == CONTEXT_MODE_SHARED) {
        if (!cuda_check(
                cuDevicePrimaryCtxRetain(&shared_context, device),
                "cuDevicePrimaryCtxRetain")) {
            goto cleanup;
        }
        for (size_t index = 0u; index < options.loaders; ++index) {
            contexts[index] = shared_context;
        }
    } else {
        for (size_t index = 0u; index < options.loaders; ++index) {
            if (!cuda_check(
                    cuCtxCreate_v2(&contexts[index], CU_CTX_SCHED_AUTO, device),
                    "cuCtxCreate") ||
                !cuda_check(cuCtxSetCurrent(NULL), "cuCtxSetCurrent(NULL)")) {
                goto cleanup;
            }
        }
    }

    if (!cuda_check(cuCtxSetCurrent(contexts[0]), "cuCtxSetCurrent(probe)")) {
        goto cleanup;
    }
    {
        CUmodule probe = NULL;

        if (!cuda_check(cuModuleLoadData(&probe, options.images[0].data), "probe cuModuleLoadData") ||
            !cuda_check(cuModuleGetFunctionCount(&function_count, probe), "probe cuModuleGetFunctionCount") ||
            !cuda_check(cuModuleUnload(probe), "probe cuModuleUnload")) {
            goto cleanup;
        }
    }
    if (!cuda_check(cuCtxSetCurrent(NULL), "cuCtxSetCurrent(NULL)")) {
        goto cleanup;
    }

    memset(&state, 0, sizeof(state));
    state.images = options.images;
    state.image_count = options.image_count;
    state.loader_count = options.loaders;
    state.function_capacity = function_count;
    state.enumeration_mode = options.enumeration_mode;
    atomic_init(&state.stop, 0);
    atomic_init(&state.failed, 0);
    if (pthread_barrier_init(&state.start_barrier, NULL, (unsigned int)options.loaders + 1u) != 0 ||
        pthread_barrier_init(&state.loaded_barrier, NULL, (unsigned int)options.loaders + 1u) != 0 ||
        pthread_barrier_init(&state.unload_start_barrier, NULL, (unsigned int)options.loaders + 1u) != 0 ||
        pthread_barrier_init(&state.unloaded_barrier, NULL, (unsigned int)options.loaders + 1u) != 0) {
        fprintf(stderr, "pthread_barrier_init failed\n");
        goto cleanup;
    }

    for (size_t index = 0u; index < options.loaders; ++index) {
        workers[index].state = &state;
        workers[index].index = index;
        workers[index].context = contexts[index];
        if (function_count > 0u) {
            workers[index].functions = (CUfunction*)calloc(function_count, sizeof(CUfunction));
            if (workers[index].functions == NULL) {
                atomic_store(&state.failed, 1);
                goto allocation_cleanup;
            }
        }
    }
    for (size_t index = 0u; index < options.loaders; ++index) {
        if (pthread_create(
                &workers[index].thread,
                NULL,
                loader_worker_main,
                &workers[index]) != 0) {
            fprintf(stderr, "pthread_create failed\n");
            /* A partially formed barrier group cannot be recovered safely. */
            exit(1);
        }
    }

    for (size_t wave = 0u; wave < options.warmup_waves + options.waves; ++wave) {
        double begin;
        double load_wall;
        double unload_wall;

        state.wave = wave;
        begin = now_seconds();
        if (!barrier_wait(&state.start_barrier) ||
            !barrier_wait(&state.loaded_barrier)) {
            atomic_store(&state.failed, 1);
            break;
        }
        load_wall = now_seconds() - begin;
        begin = now_seconds();
        if (!barrier_wait(&state.unload_start_barrier) ||
            !barrier_wait(&state.unloaded_barrier)) {
            atomic_store(&state.failed, 1);
            break;
        }
        unload_wall = now_seconds() - begin;
        if (atomic_load(&state.failed)) {
            break;
        }
        if (wave >= options.warmup_waves) {
            const size_t measured_wave = wave - options.warmup_waves;

            load_wave_seconds[measured_wave] = load_wall;
            unload_wave_seconds[measured_wave] = unload_wall;
            load_wall_sum += load_wall;
            unload_wall_sum += unload_wall;
            for (size_t index = 0u; index < options.loaders; ++index) {
                driver_load_sum += workers[index].driver_load_seconds;
                enumeration_sum += workers[index].enumeration_seconds;
                unload_sum += workers[index].unload_seconds;
            }
        }
    }

    atomic_store(&state.stop, 1);
    (void)barrier_wait(&state.start_barrier);
    for (size_t index = 0u; index < options.loaders; ++index) {
        (void)pthread_join(workers[index].thread, NULL);
        free(workers[index].functions);
        workers[index].functions = NULL;
    }
    if (atomic_load(&state.failed)) {
        goto barrier_cleanup;
    }

    for (size_t image_index = 0u; image_index < options.image_count; ++image_index) {
        if (options.images[image_index].size < image_min) {
            image_min = options.images[image_index].size;
        }
        if (options.images[image_index].size > image_max) {
            image_max = options.images[image_index].size;
        }
    }
    {
        const double modules = (double)(options.loaders * options.waves);
        const double load_median_ms = percentile(load_wave_seconds, options.waves, 0.5) * 1000.0;
        const double load_p95_ms = percentile(load_wave_seconds, options.waves, 0.95) * 1000.0;
        const double unload_median_ms = percentile(unload_wave_seconds, options.waves, 0.5) * 1000.0;

        printf("{\n");
        printf("  \"schema\": \"secant.cuda_module_load_scaling.v1\",\n");
        printf("  \"contexts\": \"%s\",\n", context_mode_name(options.context_mode));
        printf("  \"enumeration\": \"%s\",\n", enumeration_mode_name(options.enumeration_mode));
        printf("  \"loaders\": %zu,\n", options.loaders);
        printf("  \"waves\": %zu,\n", options.waves);
        printf("  \"warmup_waves\": %zu,\n", options.warmup_waves);
        printf("  \"images\": %zu,\n", options.image_count);
        printf("  \"image_bytes_min\": %zu,\n", image_min);
        printf("  \"image_bytes_max\": %zu,\n", image_max);
        printf("  \"functions_per_image\": %u,\n", function_count);
        printf("  \"modules\": %.0f,\n", modules);
        printf("  \"load_enumerate_wall_seconds\": %.9f,\n", load_wall_sum);
        printf("  \"driver_load_sum_seconds\": %.9f,\n", driver_load_sum);
        printf("  \"enumeration_sum_seconds\": %.9f,\n", enumeration_sum);
        printf("  \"unload_wall_seconds\": %.9f,\n", unload_wall_sum);
        printf("  \"unload_sum_seconds\": %.9f,\n", unload_sum);
        printf("  \"load_enumerate_modules_per_second\": %.3f,\n", modules / load_wall_sum);
        printf("  \"lifecycle_modules_per_second\": %.3f,\n", modules / (load_wall_sum + unload_wall_sum));
        printf("  \"driver_load_mean_ms\": %.6f,\n", driver_load_sum * 1000.0 / modules);
        printf("  \"enumeration_mean_ms\": %.6f,\n", enumeration_sum * 1000.0 / modules);
        printf("  \"unload_mean_ms\": %.6f,\n", unload_sum * 1000.0 / modules);
        printf("  \"load_wave_median_ms\": %.6f,\n", load_median_ms);
        printf("  \"load_wave_p95_ms\": %.6f,\n", load_p95_ms);
        printf("  \"unload_wave_median_ms\": %.6f\n", unload_median_ms);
        printf("}\n");
    }
    result = 0;

barrier_cleanup:
    (void)pthread_barrier_destroy(&state.unloaded_barrier);
    (void)pthread_barrier_destroy(&state.unload_start_barrier);
    (void)pthread_barrier_destroy(&state.loaded_barrier);
    (void)pthread_barrier_destroy(&state.start_barrier);
    goto cleanup;

allocation_cleanup:
    for (size_t index = 0u; index < options.loaders; ++index) {
        free(workers[index].functions);
        workers[index].functions = NULL;
    }
    (void)pthread_barrier_destroy(&state.unloaded_barrier);
    (void)pthread_barrier_destroy(&state.unload_start_barrier);
    (void)pthread_barrier_destroy(&state.loaded_barrier);
    (void)pthread_barrier_destroy(&state.start_barrier);

cleanup:
    (void)cuCtxSetCurrent(NULL);
    if (contexts != NULL) {
        if (options.context_mode == CONTEXT_MODE_SEPARATE) {
            for (size_t index = 0u; index < options.loaders; ++index) {
                if (contexts[index] != NULL) {
                    (void)cuCtxDestroy(contexts[index]);
                }
            }
        } else if (shared_context != NULL) {
            (void)cuDevicePrimaryCtxRelease(device);
        }
    }
    free(unload_wave_seconds);
    free(load_wave_seconds);
    free(contexts);
    free(workers);
    options_destroy(&options);
    return result;
}
