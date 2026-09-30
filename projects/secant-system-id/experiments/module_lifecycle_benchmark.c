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

#include <dlfcn.h>
#include <errno.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef int CUresult;
typedef int CUdevice;
typedef uint64_t CUdeviceptr;
typedef struct CUctx_st *CUcontext;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
typedef struct CUstream_st *CUstream;
typedef struct CUevent_st *CUevent;

enum {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_NOT_READY = 600,
    CU_STREAM_NON_BLOCKING = 1,
    CU_EVENT_DISABLE_TIMING = 2
};

typedef struct cuda_api {
    void *library;
    CUresult (*cuInit)(unsigned int);
    CUresult (*cuDeviceGet)(CUdevice *, int);
    CUresult (*cuCtxCreate_v2)(CUcontext *, unsigned int, CUdevice);
    CUresult (*cuCtxDestroy_v2)(CUcontext);
    CUresult (*cuCtxSetCurrent)(CUcontext);
    CUresult (*cuModuleLoadData)(CUmodule *, const void *);
    CUresult (*cuModuleUnload)(CUmodule);
    CUresult (*cuModuleGetFunction)(CUfunction *, CUmodule, const char *);
    CUresult (*cuLaunchKernel)(CUfunction, unsigned int, unsigned int, unsigned int,
        unsigned int, unsigned int, unsigned int, unsigned int, CUstream,
        void **, void **);
    CUresult (*cuStreamCreate)(CUstream *, unsigned int);
    CUresult (*cuStreamWaitEvent)(CUstream, CUevent, unsigned int);
    CUresult (*cuStreamSynchronize)(CUstream);
    CUresult (*cuStreamDestroy_v2)(CUstream);
    CUresult (*cuEventCreate)(CUevent *, unsigned int);
    CUresult (*cuEventRecord)(CUevent, CUstream);
    CUresult (*cuEventQuery)(CUevent);
    CUresult (*cuEventDestroy_v2)(CUevent);
    CUresult (*cuMemAlloc_v2)(CUdeviceptr *, size_t);
    CUresult (*cuMemFree_v2)(CUdeviceptr);
    CUresult (*cuMemcpyHtoD_v2)(CUdeviceptr, const void *, size_t);
    CUresult (*cuMemcpyDtoH_v2)(void *, CUdeviceptr, size_t);
    CUresult (*cuGetErrorString)(CUresult, const char **);
} cuda_api;

typedef struct benchmark benchmark;

typedef struct worker {
    benchmark *owner;
    pthread_t thread;
    uint32_t index;
    uint8_t *image;
    uint32_t ticket;
    int image_released;
} worker;

typedef struct inflight_module {
    CUmodule module;
    CUevent event;
    CUevent *dependency_events;
    uint32_t dependency_event_count;
    uint32_t ticket;
    int active;
} inflight_module;

typedef struct timings {
    double image_copy_seconds;
    double module_load_seconds;
    double function_lookup_seconds;
    double event_create_seconds;
    double launch_record_seconds;
    double event_query_seconds;
    double event_destroy_seconds;
    double module_unload_seconds;
    uint64_t event_queries;
} timings;

struct benchmark {
    const uint8_t *template_image;
    size_t image_size;
    size_t nonce_offset;
    uint64_t nonce_magic;
    uint32_t module_count;
    uint32_t next_ticket;
    uint32_t ready_count;
    uint32_t ready_head;
    uint32_t ready_tail;
    uint32_t worker_count;
    uint32_t workers_ready;
    uint32_t *ready_workers;
    worker *workers;
    int unique_images;
    int start_workers;
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    pthread_cond_t worker_condition;
    timings measured;
};

typedef struct options {
    const char *cubin_path;
    const char *kernel_name;
    uint64_t nonce_magic;
    size_t nonce_offset;
    uint32_t modules;
    uint32_t warmup_modules;
    uint32_t kernels;
    uint32_t workers;
    uint32_t streams;
    uint32_t loaded_modules;
    uint32_t blocks;
    uint32_t threads;
    uint32_t poll_microseconds;
    uint64_t wait_clocks;
    int device;
    int unique_images;
} options;

static double monotonic_seconds(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec + (double)value.tv_nsec * 1.0e-9;
}

static uint64_t mix64(uint64_t value) {
    value += UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

static uint64_t module_nonce(const benchmark *state, uint64_t ticket) {
    uint64_t value = state->nonce_magic ^ mix64(ticket + UINT64_C(0xd1b54a32d192ed03));
    if (value == state->nonce_magic) value ^= UINT64_C(0x0101010101010101);
    return value;
}

static void patch_image(const benchmark *state, uint8_t *image, uint64_t ticket) {
    uint64_t value;
    memcpy(image, state->template_image, state->image_size);
    if (!state->unique_images) return;
    value = module_nonce(state, ticket);
    memcpy(image + state->nonce_offset, &value, sizeof(value));
}

static uint64_t fnv1a64(const uint8_t *data, size_t size) {
    uint64_t hash = UINT64_C(1469598103934665603);
    size_t index;
    for (index = 0; index < size; ++index) {
        hash ^= data[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static int parse_u32(const char *text, uint32_t *value) {
    char *end = NULL;
    unsigned long parsed;
    errno = 0;
    parsed = strtoul(text, &end, 0);
    if (errno != 0 || end == text || *end != '\0' || parsed == 0 || parsed > UINT32_MAX) return 0;
    *value = (uint32_t)parsed;
    return 1;
}

static int parse_size(const char *text, size_t *value) {
    char *end = NULL;
    unsigned long long parsed;
    errno = 0;
    parsed = strtoull(text, &end, 0);
    if (errno != 0 || end == text || *end != '\0' || parsed > SIZE_MAX) return 0;
    *value = (size_t)parsed;
    return 1;
}

static int parse_u64(const char *text, uint64_t *value) {
    char *end = NULL;
    unsigned long long parsed;
    errno = 0;
    parsed = strtoull(text, &end, 0);
    if (errno != 0 || end == text || *end != '\0') return 0;
    *value = (uint64_t)parsed;
    return 1;
}

static void usage(const char *program) {
    fprintf(stderr,
        "usage: %s --cubin FILE --nonce-offset N --nonce-magic N [options]\n"
        "  --kernel NAME          default secant_module_lifecycle_kernel\n"
        "  --modules N            timed modules, default 1024\n"
        "  --warmup-modules N     unique untimed warmups, default 8\n"
        "  --kernels N            entry points per module, default 1\n"
        "  --workers N            worker-owned image copies, default 4\n"
        "  --streams N            nonblocking streams, default 4\n"
        "  --loaded-modules N     maximum in-flight modules, default 8\n"
        "  --blocks N             launch blocks/module, default 1\n"
        "  --threads N            launch threads/block, default 32\n"
        "  --wait-clocks N        per-kernel clock64 busy wait, default 0\n"
        "  --poll-us N            completion poll sleep, default 0\n"
        "  --device N             CUDA device ordinal, default 0\n"
        "  --identity MODE        unique or identical, default unique\n",
        program);
}

static int parse_options(int argc, char **argv, options *result) {
    int index;
    memset(result, 0, sizeof(*result));
    result->kernel_name = "secant_module_lifecycle_kernel";
    result->modules = 1024;
    result->warmup_modules = 8;
    result->kernels = 1;
    result->workers = 4;
    result->streams = 4;
    result->loaded_modules = 8;
    result->blocks = 1;
    result->threads = 32;
    result->unique_images = 1;
    for (index = 1; index < argc; ++index) {
        const char *name = argv[index];
        const char *value;
        if (index + 1 >= argc) return 0;
        value = argv[++index];
        if (strcmp(name, "--cubin") == 0) result->cubin_path = value;
        else if (strcmp(name, "--kernel") == 0) result->kernel_name = value;
        else if (strcmp(name, "--nonce-offset") == 0) {
            if (!parse_size(value, &result->nonce_offset)) return 0;
        } else if (strcmp(name, "--nonce-magic") == 0) {
            if (!parse_u64(value, &result->nonce_magic)) return 0;
        } else if (strcmp(name, "--modules") == 0) {
            if (!parse_u32(value, &result->modules)) return 0;
        } else if (strcmp(name, "--warmup-modules") == 0) {
            if (!parse_u32(value, &result->warmup_modules)) return 0;
        } else if (strcmp(name, "--kernels") == 0) {
            if (!parse_u32(value, &result->kernels)) return 0;
        } else if (strcmp(name, "--workers") == 0) {
            if (!parse_u32(value, &result->workers)) return 0;
        } else if (strcmp(name, "--streams") == 0) {
            if (!parse_u32(value, &result->streams)) return 0;
        } else if (strcmp(name, "--loaded-modules") == 0) {
            if (!parse_u32(value, &result->loaded_modules)) return 0;
        } else if (strcmp(name, "--blocks") == 0) {
            if (!parse_u32(value, &result->blocks)) return 0;
        } else if (strcmp(name, "--threads") == 0) {
            if (!parse_u32(value, &result->threads)) return 0;
        } else if (strcmp(name, "--wait-clocks") == 0) {
            if (!parse_u64(value, &result->wait_clocks)) return 0;
        } else if (strcmp(name, "--poll-us") == 0) {
            size_t parsed;
            if (!parse_size(value, &parsed) || parsed > UINT32_MAX) return 0;
            result->poll_microseconds = (uint32_t)parsed;
        } else if (strcmp(name, "--device") == 0) {
            char *end = NULL;
            long parsed = strtol(value, &end, 0);
            if (end == value || *end != '\0' || parsed < 0 || parsed > INT32_MAX) return 0;
            result->device = (int)parsed;
        } else if (strcmp(name, "--identity") == 0) {
            if (strcmp(value, "unique") == 0) result->unique_images = 1;
            else if (strcmp(value, "identical") == 0) result->unique_images = 0;
            else return 0;
        } else return 0;
    }
    return result->cubin_path != NULL && result->nonce_magic != 0 &&
        result->threads <= 1024;
}

static int load_cuda_symbol(void *library, void **target, const char *name) {
    *target = dlsym(library, name);
    if (*target == NULL) {
        fprintf(stderr, "missing CUDA symbol %s: %s\n", name, dlerror());
        return 0;
    }
    return 1;
}

#define LOAD_CUDA(api, name) load_cuda_symbol((api)->library, (void **)&(api)->name, #name)

static int open_cuda(cuda_api *api) {
    memset(api, 0, sizeof(*api));
    api->library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (api->library == NULL) {
        fprintf(stderr, "could not load libcuda.so.1: %s\n", dlerror());
        return 0;
    }
    return LOAD_CUDA(api, cuInit) && LOAD_CUDA(api, cuDeviceGet) &&
        LOAD_CUDA(api, cuCtxCreate_v2) && LOAD_CUDA(api, cuCtxDestroy_v2) &&
        LOAD_CUDA(api, cuCtxSetCurrent) && LOAD_CUDA(api, cuModuleLoadData) &&
        LOAD_CUDA(api, cuModuleUnload) && LOAD_CUDA(api, cuModuleGetFunction) &&
        LOAD_CUDA(api, cuLaunchKernel) && LOAD_CUDA(api, cuStreamCreate) &&
        LOAD_CUDA(api, cuStreamWaitEvent) && LOAD_CUDA(api, cuStreamSynchronize) &&
        LOAD_CUDA(api, cuStreamDestroy_v2) &&
        LOAD_CUDA(api, cuEventCreate) && LOAD_CUDA(api, cuEventRecord) &&
        LOAD_CUDA(api, cuEventQuery) && LOAD_CUDA(api, cuEventDestroy_v2) &&
        LOAD_CUDA(api, cuMemAlloc_v2) && LOAD_CUDA(api, cuMemFree_v2) &&
        LOAD_CUDA(api, cuMemcpyHtoD_v2) && LOAD_CUDA(api, cuMemcpyDtoH_v2) &&
        LOAD_CUDA(api, cuGetErrorString);
}

static int cuda_ok(cuda_api *api, CUresult result, const char *operation) {
    const char *message = NULL;
    if (result == CUDA_SUCCESS) return 1;
    if (api->cuGetErrorString != NULL) api->cuGetErrorString(result, &message);
    fprintf(stderr, "%s failed with CUDA error %d%s%s\n", operation, result,
        message == NULL ? "" : ": ", message == NULL ? "" : message);
    return 0;
}

static uint8_t *read_file(const char *path, size_t *size_out) {
    FILE *file = fopen(path, "rb");
    long length;
    uint8_t *data;
    if (file == NULL) return NULL;
    if (fseek(file, 0, SEEK_END) != 0 || (length = ftell(file)) <= 0 ||
        fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return NULL;
    }
    data = (uint8_t *)malloc((size_t)length);
    if (data == NULL || fread(data, 1, (size_t)length, file) != (size_t)length) {
        free(data);
        fclose(file);
        return NULL;
    }
    fclose(file);
    *size_out = (size_t)length;
    return data;
}

static void *worker_main(void *argument) {
    worker *current = (worker *)argument;
    benchmark *state = current->owner;
    pthread_mutex_lock(&state->mutex);
    state->workers_ready += 1;
    pthread_cond_broadcast(&state->condition);
    while (!state->start_workers) pthread_cond_wait(&state->worker_condition, &state->mutex);
    pthread_mutex_unlock(&state->mutex);
    for (;;) {
        double started;
        uint32_t ticket;
        pthread_mutex_lock(&state->mutex);
        if (state->next_ticket >= state->module_count) {
            pthread_mutex_unlock(&state->mutex);
            return NULL;
        }
        ticket = state->next_ticket++;
        pthread_mutex_unlock(&state->mutex);
        started = monotonic_seconds();
        patch_image(state, current->image, ticket);
        pthread_mutex_lock(&state->mutex);
        state->measured.image_copy_seconds += monotonic_seconds() - started;
        current->ticket = ticket;
        current->image_released = 0;
        state->ready_workers[state->ready_tail] = current->index;
        state->ready_tail = (state->ready_tail + 1u) % state->worker_count;
        state->ready_count += 1;
        pthread_cond_broadcast(&state->condition);
        while (!current->image_released) pthread_cond_wait(&state->worker_condition, &state->mutex);
        pthread_mutex_unlock(&state->mutex);
    }
}

static void release_worker(benchmark *state, worker *current) {
    pthread_mutex_lock(&state->mutex);
    current->image_released = 1;
    pthread_cond_broadcast(&state->worker_condition);
    pthread_mutex_unlock(&state->mutex);
}

static int format_kernel_name(
    const options *config, uint32_t index, char *name, size_t size) {
    int length;
    if (index >= config->kernels) return 0;
    length = config->kernels == 1
        ? snprintf(name, size, "%s", config->kernel_name)
        : snprintf(name, size, "%s_%03u", config->kernel_name, index);
    return length > 0 && (size_t)length < size;
}

static int verify_module(cuda_api *api, CUcontext context, CUstream stream,
    const options *config, const benchmark *state, CUdeviceptr gate_device,
    CUdeviceptr sink_device, uint8_t *scratch, double *seconds_out) {
    CUmodule module = NULL;
    CUfunction function = NULL;
    uint32_t gate = 2;
    uint64_t sink = 0;
    uint64_t expected;
    uint64_t wait_clocks = 0;
    CUdeviceptr gate_argument = gate_device;
    CUdeviceptr sink_argument = sink_device;
    void *arguments[] = {&gate_argument, &sink_argument, &wait_clocks};
    char kernel_name[256];
    double started = monotonic_seconds();
    patch_image(state, scratch, UINT64_C(0xffffffffffff0000));
    expected = state->unique_images
        ? module_nonce(state, UINT64_C(0xffffffffffff0000)) : state->nonce_magic;
    if (!format_kernel_name(config, 0, kernel_name, sizeof(kernel_name))) {
        fprintf(stderr, "kernel name is too long\n");
        return 0;
    }
    if (!cuda_ok(api, api->cuCtxSetCurrent(context), "cuCtxSetCurrent(verify)") ||
        !cuda_ok(api, api->cuMemcpyHtoD_v2(gate_device, &gate, sizeof(gate)), "cuMemcpyHtoD(verify gate)") ||
        !cuda_ok(api, api->cuMemcpyHtoD_v2(sink_device, &sink, sizeof(sink)), "cuMemcpyHtoD(verify sink)") ||
        !cuda_ok(api, api->cuModuleLoadData(&module, scratch), "cuModuleLoadData(verify)") ||
        !cuda_ok(api, api->cuModuleGetFunction(&function, module, kernel_name), "cuModuleGetFunction(verify)") ||
        !cuda_ok(api, api->cuLaunchKernel(function, 1, 1, 1, config->threads, 1, 1,
            0, stream, arguments, NULL), "cuLaunchKernel(verify)") ||
        !cuda_ok(api, api->cuStreamSynchronize(stream), "cuStreamSynchronize(verify)") ||
        !cuda_ok(api, api->cuMemcpyDtoH_v2(&sink, sink_device, sizeof(sink)), "cuMemcpyDtoH(verify sink)")) {
        if (module != NULL) api->cuModuleUnload(module);
        return 0;
    }
    if (sink != expected) {
        fprintf(stderr, "module identity verification failed: expected 0x%016" PRIx64
            ", observed 0x%016" PRIx64 "\n", expected, sink);
        api->cuModuleUnload(module);
        return 0;
    }
    if (!cuda_ok(api, api->cuModuleUnload(module), "cuModuleUnload(verify)")) return 0;
    *seconds_out = monotonic_seconds() - started;
    gate = 1;
    return cuda_ok(api, api->cuMemcpyHtoD_v2(gate_device, &gate, sizeof(gate)), "cuMemcpyHtoD(early-exit gate)");
}

static int warmup_modules(cuda_api *api, CUcontext context, CUstream stream,
    const options *config, const benchmark *state, CUdeviceptr gate_device,
    CUdeviceptr sink_device, uint8_t *scratch) {
    uint32_t index;
    CUdeviceptr gate_argument = gate_device;
    CUdeviceptr sink_argument = sink_device;
    uint64_t wait_clocks = config->wait_clocks;
    void *arguments[] = {&gate_argument, &sink_argument, &wait_clocks};
    for (index = 0; index < config->warmup_modules; ++index) {
        CUmodule module = NULL;
        uint32_t kernel_index;
        uint64_t ticket = UINT64_C(0xffffffff00000000) + index;
        patch_image(state, scratch, ticket);
        if (!cuda_ok(api, api->cuCtxSetCurrent(context), "cuCtxSetCurrent(warmup)") ||
            !cuda_ok(api, api->cuModuleLoadData(&module, scratch), "cuModuleLoadData(warmup)")) {
            if (module != NULL) api->cuModuleUnload(module);
            return 0;
        }
        for (kernel_index = 0; kernel_index < config->kernels; ++kernel_index) {
            CUfunction function = NULL;
            char kernel_name[256];
            if (!format_kernel_name(config, kernel_index, kernel_name, sizeof(kernel_name)) ||
                !cuda_ok(api, api->cuModuleGetFunction(&function, module, kernel_name), "cuModuleGetFunction(warmup)") ||
                !cuda_ok(api, api->cuLaunchKernel(function, config->blocks, 1, 1,
                    config->threads, 1, 1, 0, stream, arguments, NULL), "cuLaunchKernel(warmup)")) {
                api->cuModuleUnload(module);
                return 0;
            }
        }
        if (!cuda_ok(api, api->cuStreamSynchronize(stream), "cuStreamSynchronize(warmup)") ||
            !cuda_ok(api, api->cuModuleUnload(module), "cuModuleUnload(warmup)")) return 0;
    }
    return 1;
}

static void poll_sleep(uint32_t microseconds) {
    struct timespec delay;
    if (microseconds == 0) return;
    delay.tv_sec = microseconds / 1000000u;
    delay.tv_nsec = (long)(microseconds % 1000000u) * 1000L;
    nanosleep(&delay, NULL);
}

static int run_timed_pipeline(cuda_api *api, CUstream *streams,
    const options *config, benchmark *state, CUdeviceptr gate_device,
    CUdeviceptr sink_device, double *wall_seconds) {
    inflight_module *inflight;
    CUevent *dependency_storage;
    CUfunction *functions;
    uint32_t dependency_capacity = config->kernels < config->streams
        ? config->kernels : config->streams;
    uint32_t inflight_count = 0;
    uint32_t completed = 0;
    uint32_t next_stream = 0;
    uint32_t index;
    uint32_t created_workers = 0;
    int ok = 1;
    double wall_started;
    CUdeviceptr gate_argument = gate_device;
    CUdeviceptr sink_argument = sink_device;
    uint64_t wait_clocks = config->wait_clocks;
    void *arguments[] = {&gate_argument, &sink_argument, &wait_clocks};
    inflight = (inflight_module *)calloc(config->loaded_modules, sizeof(*inflight));
    dependency_storage = (CUevent *)calloc(
        (size_t)config->loaded_modules * dependency_capacity,
        sizeof(*dependency_storage));
    functions = (CUfunction *)calloc(config->kernels, sizeof(*functions));
    if (inflight == NULL || dependency_storage == NULL || functions == NULL) {
        free(inflight);
        free(dependency_storage);
        free(functions);
        return 0;
    }
    for (index = 0; index < config->loaded_modules; ++index) {
        inflight[index].dependency_events =
            dependency_storage + (size_t)index * dependency_capacity;
    }
    for (index = 0; index < state->worker_count; ++index) {
        if (pthread_create(&state->workers[index].thread, NULL, worker_main,
                &state->workers[index]) != 0) {
            fprintf(stderr, "could not create worker %u\n", index);
            ok = 0;
            break;
        }
        created_workers += 1;
    }
    if (!ok) goto finish;
    pthread_mutex_lock(&state->mutex);
    while (state->workers_ready != state->worker_count) pthread_cond_wait(&state->condition, &state->mutex);
    wall_started = monotonic_seconds();
    state->start_workers = 1;
    pthread_cond_broadcast(&state->worker_condition);
    pthread_mutex_unlock(&state->mutex);

    while (completed < config->modules && ok) {
        int progressed = 0;
        for (index = 0; index < config->loaded_modules; ++index) {
            CUresult result;
            double started;
            if (!inflight[index].active) continue;
            started = monotonic_seconds();
            result = api->cuEventQuery(inflight[index].event);
            state->measured.event_query_seconds += monotonic_seconds() - started;
            state->measured.event_queries += 1;
            if (result == CUDA_ERROR_NOT_READY) continue;
            if (!cuda_ok(api, result, "cuEventQuery")) {
                ok = 0;
                break;
            }
            started = monotonic_seconds();
            if (!cuda_ok(api, api->cuEventDestroy_v2(inflight[index].event), "cuEventDestroy")) {
                ok = 0;
                break;
            }
            state->measured.event_destroy_seconds += monotonic_seconds() - started;
            started = monotonic_seconds();
            {
                uint32_t dependency_index;
                for (dependency_index = 0;
                     dependency_index < inflight[index].dependency_event_count;
                     ++dependency_index) {
                    if (!cuda_ok(api, api->cuEventDestroy_v2(
                            inflight[index].dependency_events[dependency_index]),
                            "cuEventDestroy(dependency)")) {
                        ok = 0;
                        break;
                    }
                    inflight[index].dependency_events[dependency_index] = NULL;
                }
            }
            state->measured.event_destroy_seconds += monotonic_seconds() - started;
            if (!ok) break;
            started = monotonic_seconds();
            if (!cuda_ok(api, api->cuModuleUnload(inflight[index].module), "cuModuleUnload")) {
                ok = 0;
                break;
            }
            state->measured.module_unload_seconds += monotonic_seconds() - started;
            inflight[index].module = NULL;
            inflight[index].event = NULL;
            inflight[index].dependency_event_count = 0;
            inflight[index].ticket = 0;
            inflight[index].active = 0;
            inflight_count -= 1;
            completed += 1;
            progressed = 1;
        }
        while (ok && inflight_count < config->loaded_modules) {
            worker *current;
            uint32_t worker_index;
            uint32_t slot;
            CUmodule module = NULL;
            CUevent event = NULL;
            CUstream completion_stream;
            uint32_t module_stream_base;
            uint32_t kernel_index;
            uint32_t used_stream_count;
            uint32_t created_dependency_events = 0;
            double started;
            pthread_mutex_lock(&state->mutex);
            if (state->ready_count == 0) {
                pthread_mutex_unlock(&state->mutex);
                break;
            }
            worker_index = state->ready_workers[state->ready_head];
            state->ready_head = (state->ready_head + 1u) % state->worker_count;
            state->ready_count -= 1;
            pthread_mutex_unlock(&state->mutex);
            current = &state->workers[worker_index];
            for (slot = 0; slot < config->loaded_modules && inflight[slot].active; ++slot) {}
            if (slot == config->loaded_modules) {
                fprintf(stderr, "internal error: no free in-flight slot\n");
                ok = 0;
                break;
            }
            started = monotonic_seconds();
            if (!cuda_ok(api, api->cuModuleLoadData(&module, current->image), "cuModuleLoadData")) {
                release_worker(state, current);
                ok = 0;
                break;
            }
            state->measured.module_load_seconds += monotonic_seconds() - started;
            release_worker(state, current);
            started = monotonic_seconds();
            for (kernel_index = 0; kernel_index < config->kernels; ++kernel_index) {
                char kernel_name[256];
                if (!format_kernel_name(config, kernel_index, kernel_name, sizeof(kernel_name)) ||
                    !cuda_ok(api, api->cuModuleGetFunction(
                        &functions[kernel_index], module, kernel_name),
                        "cuModuleGetFunction")) {
                    ok = 0;
                    break;
                }
            }
            state->measured.function_lookup_seconds += monotonic_seconds() - started;
            if (!ok) {
                api->cuModuleUnload(module);
                break;
            }
            started = monotonic_seconds();
            if (!cuda_ok(api, api->cuEventCreate(&event, CU_EVENT_DISABLE_TIMING), "cuEventCreate")) {
                api->cuModuleUnload(module);
                ok = 0;
                break;
            }
            used_stream_count = dependency_capacity;
            for (kernel_index = 0; kernel_index < used_stream_count; ++kernel_index) {
                if (!cuda_ok(api, api->cuEventCreate(
                        &inflight[slot].dependency_events[kernel_index],
                        CU_EVENT_DISABLE_TIMING),
                        "cuEventCreate(dependency)")) {
                    ok = 0;
                    break;
                }
                created_dependency_events += 1;
            }
            state->measured.event_create_seconds += monotonic_seconds() - started;
            if (!ok) {
                for (kernel_index = 0; kernel_index < created_dependency_events; ++kernel_index) {
                    api->cuEventDestroy_v2(inflight[slot].dependency_events[kernel_index]);
                    inflight[slot].dependency_events[kernel_index] = NULL;
                }
                api->cuEventDestroy_v2(event);
                api->cuModuleUnload(module);
                break;
            }
            module_stream_base = next_stream;
            next_stream = (next_stream + 1u) % config->streams;
            completion_stream = streams[module_stream_base];
            started = monotonic_seconds();
            for (kernel_index = 0; kernel_index < config->kernels; ++kernel_index) {
                CUstream stream = streams[
                    (module_stream_base + kernel_index) % config->streams];
                if (!cuda_ok(api, api->cuLaunchKernel(
                        functions[kernel_index], config->blocks, 1, 1,
                        config->threads, 1, 1, 0, stream, arguments, NULL),
                        "cuLaunchKernel")) {
                    ok = 0;
                    break;
                }
            }
            for (kernel_index = 0; ok && kernel_index < used_stream_count; ++kernel_index) {
                CUstream stream = streams[
                    (module_stream_base + kernel_index) % config->streams];
                if (!cuda_ok(api, api->cuEventRecord(
                        inflight[slot].dependency_events[kernel_index], stream),
                        "cuEventRecord(dependency)") ||
                    !cuda_ok(api, api->cuStreamWaitEvent(
                        completion_stream,
                        inflight[slot].dependency_events[kernel_index], 0),
                        "cuStreamWaitEvent")) {
                    ok = 0;
                    break;
                }
            }
            if (ok && !cuda_ok(api, api->cuEventRecord(event, completion_stream),
                    "cuEventRecord")) ok = 0;
            if (!ok) {
                uint32_t dependency_index;
                for (dependency_index = 0;
                     dependency_index < created_dependency_events;
                     ++dependency_index) {
                    api->cuEventDestroy_v2(
                        inflight[slot].dependency_events[dependency_index]);
                    inflight[slot].dependency_events[dependency_index] = NULL;
                }
                api->cuEventDestroy_v2(event);
                api->cuModuleUnload(module);
                break;
            }
            state->measured.launch_record_seconds += monotonic_seconds() - started;
            inflight[slot].module = module;
            inflight[slot].event = event;
            inflight[slot].dependency_event_count = used_stream_count;
            inflight[slot].ticket = current->ticket;
            inflight[slot].active = 1;
            inflight_count += 1;
            progressed = 1;
        }
        if (!progressed && ok) poll_sleep(config->poll_microseconds);
    }
    *wall_seconds = monotonic_seconds() - wall_started;

finish:
    pthread_mutex_lock(&state->mutex);
    for (index = 0; index < state->worker_count; ++index) state->workers[index].image_released = 1;
    state->start_workers = 1;
    pthread_cond_broadcast(&state->worker_condition);
    pthread_mutex_unlock(&state->mutex);
    for (index = 0; index < created_workers; ++index) pthread_join(state->workers[index].thread, NULL);
    for (index = 0; index < config->loaded_modules; ++index) {
        if (!inflight[index].active) continue;
        api->cuStreamSynchronize(streams[index % config->streams]);
        api->cuEventDestroy_v2(inflight[index].event);
        {
            uint32_t dependency_index;
            for (dependency_index = 0;
                 dependency_index < inflight[index].dependency_event_count;
                 ++dependency_index) {
                api->cuEventDestroy_v2(
                    inflight[index].dependency_events[dependency_index]);
            }
        }
        api->cuModuleUnload(inflight[index].module);
    }
    free(inflight);
    free(dependency_storage);
    free(functions);
    return ok && completed == config->modules;
}

int main(int argc, char **argv) {
    options config;
    benchmark state;
    cuda_api cuda;
    uint8_t *image = NULL;
    uint8_t *scratch = NULL;
    size_t image_size = 0;
    uint64_t observed_magic = 0;
    uint64_t identical_hash;
    uint64_t unique_hash;
    CUdevice device;
    CUcontext context = NULL;
    CUstream *streams = NULL;
    CUdeviceptr gate_device = 0;
    CUdeviceptr sink_device = 0;
    double verification_seconds = 0.0;
    double wall_seconds = 0.0;
    uint32_t index;
    int ok = 0;

    memset(&state, 0, sizeof(state));
    memset(&cuda, 0, sizeof(cuda));
    if (!parse_options(argc, argv, &config)) {
        usage(argv[0]);
        return 2;
    }
    image = read_file(config.cubin_path, &image_size);
    if (image == NULL) {
        fprintf(stderr, "could not read %s\n", config.cubin_path);
        goto cleanup;
    }
    if (config.nonce_offset > image_size || image_size - config.nonce_offset < sizeof(uint64_t)) {
        fprintf(stderr, "nonce offset lies outside the CUBIN\n");
        goto cleanup;
    }
    memcpy(&observed_magic, image + config.nonce_offset, sizeof(observed_magic));
    if (observed_magic != config.nonce_magic) {
        fprintf(stderr, "nonce marker mismatch at offset %zu\n", config.nonce_offset);
        goto cleanup;
    }
    state.template_image = image;
    state.image_size = image_size;
    state.nonce_offset = config.nonce_offset;
    state.nonce_magic = config.nonce_magic;
    state.module_count = config.modules;
    state.worker_count = config.workers;
    state.unique_images = config.unique_images;
    pthread_mutex_init(&state.mutex, NULL);
    pthread_cond_init(&state.condition, NULL);
    pthread_cond_init(&state.worker_condition, NULL);
    state.ready_workers = (uint32_t *)calloc(config.workers, sizeof(uint32_t));
    state.workers = (worker *)calloc(config.workers, sizeof(worker));
    scratch = (uint8_t *)malloc(image_size);
    if (state.ready_workers == NULL || state.workers == NULL || scratch == NULL) goto cleanup_state;
    for (index = 0; index < config.workers; ++index) {
        state.workers[index].owner = &state;
        state.workers[index].index = index;
        state.workers[index].image = (uint8_t *)malloc(image_size);
        if (state.workers[index].image == NULL) goto cleanup_state;
    }
    identical_hash = fnv1a64(image, image_size);
    patch_image(&state, scratch, 0);
    unique_hash = fnv1a64(scratch, image_size);

    if (!open_cuda(&cuda) || !cuda_ok(&cuda, cuda.cuInit(0), "cuInit") ||
        !cuda_ok(&cuda, cuda.cuDeviceGet(&device, config.device), "cuDeviceGet") ||
        !cuda_ok(&cuda, cuda.cuCtxCreate_v2(&context, 0, device), "cuCtxCreate") ||
        !cuda_ok(&cuda, cuda.cuCtxSetCurrent(context), "cuCtxSetCurrent")) goto cleanup_state;
    streams = (CUstream *)calloc(config.streams, sizeof(CUstream));
    if (streams == NULL) goto cleanup_cuda;
    for (index = 0; index < config.streams; ++index) {
        if (!cuda_ok(&cuda, cuda.cuStreamCreate(&streams[index], CU_STREAM_NON_BLOCKING), "cuStreamCreate")) goto cleanup_cuda;
    }
    if (!cuda_ok(&cuda, cuda.cuMemAlloc_v2(&gate_device, sizeof(uint32_t)), "cuMemAlloc(gate)") ||
        !cuda_ok(&cuda, cuda.cuMemAlloc_v2(&sink_device, sizeof(uint64_t)), "cuMemAlloc(sink)")) goto cleanup_cuda;
    if (!verify_module(&cuda, context, streams[0], &config, &state, gate_device,
            sink_device, scratch, &verification_seconds) ||
        !warmup_modules(&cuda, context, streams[0], &config, &state, gate_device,
            sink_device, scratch) ||
        !run_timed_pipeline(&cuda, streams, &config, &state, gate_device,
            sink_device, &wall_seconds)) goto cleanup_cuda;

    printf("{\n");
    printf("  \"schema\": \"secant.module_lifecycle_benchmark.v1\",\n");
    printf("  \"cubin_path\": \"%s\",\n", config.cubin_path);
    printf("  \"cubin_bytes\": %zu,\n", image_size);
    printf("  \"kernel_name\": \"%s\",\n", config.kernel_name);
    printf("  \"kernel_count\": %u,\n", config.kernels);
    printf("  \"identity_mode\": \"%s\",\n", config.unique_images ? "byte-distinct-code-identical" : "identical-image");
    printf("  \"nonce_offset\": %zu,\n", config.nonce_offset);
    printf("  \"template_hash_fnv1a64\": \"%016" PRIx64 "\",\n", identical_hash);
    printf("  \"first_timed_image_hash_fnv1a64\": \"%016" PRIx64 "\",\n", unique_hash);
    printf("  \"identity_verified_on_gpu\": true,\n");
    printf("  \"verification_seconds\": %.9f,\n", verification_seconds);
    printf("  \"warmup_modules\": %u,\n", config.warmup_modules);
    printf("  \"timed_modules\": %u,\n", config.modules);
    printf("  \"workers\": %u,\n", config.workers);
    printf("  \"streams\": %u,\n", config.streams);
    printf("  \"maximum_loaded_modules\": %u,\n", config.loaded_modules);
    printf("  \"maximum_queued_function_launches\": %" PRIu64 ",\n",
        (uint64_t)config.loaded_modules * config.kernels);
    printf("  \"maximum_queued_blocks\": %" PRIu64 ",\n",
        (uint64_t)config.loaded_modules * config.kernels * config.blocks);
    printf("  \"dependency_streams_per_module\": %u,\n",
        config.kernels < config.streams ? config.kernels : config.streams);
    printf("  \"blocks_per_launch\": %u,\n", config.blocks);
    printf("  \"threads_per_block\": %u,\n", config.threads);
    printf("  \"wait_clocks_per_kernel\": %" PRIu64 ",\n", config.wait_clocks);
    printf("  \"completion_poll_microseconds\": %u,\n", config.poll_microseconds);
    printf("  \"wall_seconds\": %.9f,\n", wall_seconds);
    printf("  \"modules_per_second\": %.6f,\n", config.modules / wall_seconds);
    printf("  \"microseconds_per_module\": %.6f,\n", wall_seconds * 1.0e6 / config.modules);
    printf("  \"mean_worker_copy_microseconds\": %.6f,\n", state.measured.image_copy_seconds * 1.0e6 / config.modules);
    printf("  \"mean_module_load_microseconds\": %.6f,\n", state.measured.module_load_seconds * 1.0e6 / config.modules);
    printf("  \"mean_function_lookup_microseconds\": %.6f,\n", state.measured.function_lookup_seconds * 1.0e6 / config.modules);
    printf("  \"mean_event_create_microseconds\": %.6f,\n", state.measured.event_create_seconds * 1.0e6 / config.modules);
    printf("  \"mean_launch_and_record_microseconds\": %.6f,\n", state.measured.launch_record_seconds * 1.0e6 / config.modules);
    printf("  \"mean_event_query_microseconds\": %.6f,\n", state.measured.event_query_seconds * 1.0e6 / state.measured.event_queries);
    printf("  \"event_queries\": %" PRIu64 ",\n", state.measured.event_queries);
    printf("  \"mean_event_destroy_microseconds\": %.6f,\n", state.measured.event_destroy_seconds * 1.0e6 / config.modules);
    printf("  \"mean_module_unload_microseconds\": %.6f,\n", state.measured.module_unload_seconds * 1.0e6 / config.modules);
    printf("  \"material_deviations\": [\n");
    if (config.unique_images) {
        printf("    \"Unique mode patches a GPU-verified constant-data nonce; module bytes differ but SASS instructions are identical.\",\n");
    } else {
        printf("    \"Identical-image mode deliberately reuses exactly the same CUBIN and is a cache-favorable control, not a production-equivalent result.\",\n");
    }
    if (config.wait_clocks == 0) {
        printf("    \"The kernel performs a volatile global-memory gate load and exits before the BRKPT padding; it measures lifecycle throughput, not useful GPU arithmetic.\",\n");
    } else {
        printf("    \"The kernel performs a clock64 busy wait before exiting; this keeps a warp resident but does not reproduce RK4 arithmetic, memory traffic, or register pressure.\",\n");
    }
    printf("    \"Resident buffers and CUDA context setup are outside the timed region; worker image copy, load, lookup, launch, event handling, and unload are inside.\"\n");
    printf("  ]\n");
    printf("}\n");
    ok = 1;

cleanup_cuda:
    if (gate_device != 0) cuda.cuMemFree_v2(gate_device);
    if (sink_device != 0) cuda.cuMemFree_v2(sink_device);
    if (streams != NULL) {
        for (index = 0; index < config.streams; ++index) {
            if (streams[index] != NULL) cuda.cuStreamDestroy_v2(streams[index]);
        }
    }
    free(streams);
    if (context != NULL) cuda.cuCtxDestroy_v2(context);
    if (cuda.library != NULL) dlclose(cuda.library);
cleanup_state:
    if (state.workers != NULL) {
        for (index = 0; index < config.workers; ++index) free(state.workers[index].image);
    }
    free(state.workers);
    free(state.ready_workers);
    free(scratch);
    pthread_cond_destroy(&state.worker_condition);
    pthread_cond_destroy(&state.condition);
    pthread_mutex_destroy(&state.mutex);
cleanup:
    free(image);
    return ok ? 0 : 1;
}
