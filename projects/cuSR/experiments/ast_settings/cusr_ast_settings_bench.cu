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
#include "cusr_ast_settings_kernels.cuh"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#define CUSR_EXPERIMENT_CUDA_RET(ans) \
    do { \
        const cudaError_t cusr_experiment_cuda_result = (ans); \
        if (cusr_experiment_cuda_result != cudaSuccess) { \
            std::fprintf(stderr, "%s failed: %s\n", #ans, cudaGetErrorString(cusr_experiment_cuda_result)); \
            return false; \
        } \
    } while (0)

struct CusrExperimentConfig {
    size_t rows = 1u << 20;
    uint32_t columns = 32u;
    uint32_t settings = 4096u;
    int asts = 8;
    int expression = CUSR_AST_SETTINGS_COMPLEX_ALU;
    int iterations = 10;
};

template <typename T>
class CusrDeviceBuffer {
public:
    CusrDeviceBuffer() = default;
    CusrDeviceBuffer(const CusrDeviceBuffer&) = delete;
    CusrDeviceBuffer& operator=(const CusrDeviceBuffer&) = delete;

    ~CusrDeviceBuffer()
    {
        if (data_ != nullptr) cudaFree(data_);
    }

    bool allocate(size_t count)
    {
        if (data_ != nullptr) return false;
        count_ = count;
        return count == 0u || cudaMalloc((void**)&data_, count * sizeof(T)) == cudaSuccess;
    }

    T* data() { return data_; }
    const T* data() const { return data_; }
    size_t count() const { return count_; }
    size_t bytes() const { return count_ * sizeof(T); }

private:
    T* data_ = nullptr;
    size_t count_ = 0u;
};

class CusrCudaEvent {
public:
    CusrCudaEvent() = default;
    CusrCudaEvent(const CusrCudaEvent&) = delete;
    CusrCudaEvent& operator=(const CusrCudaEvent&) = delete;

    ~CusrCudaEvent()
    {
        if (event_ != nullptr) cudaEventDestroy(event_);
    }

    bool create() { return cudaEventCreate(&event_) == cudaSuccess; }
    cudaEvent_t get() const { return event_; }

private:
    cudaEvent_t event_ = nullptr;
};

struct CusrExperimentData {
    CusrDeviceBuffer<float> X;
    CusrDeviceBuffer<float> target;
    CusrDeviceBuffer<uint8_t> masks;
    CusrDeviceBuffer<uint32_t> words;
};

static bool cusr_experiment_checked_mul(size_t lhs, size_t rhs, size_t* result)
{
    if (lhs != 0u && rhs > std::numeric_limits<size_t>::max() / lhs) return false;
    *result = lhs * rhs;
    return true;
}

static uint32_t cusr_experiment_hash32(uint32_t value)
{
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value;
}

static float cusr_experiment_random_float(uint32_t value)
{
    const uint32_t bits = cusr_experiment_hash32(value);
    return ((float)(bits & 0x00ffffffu) * (1.0f / 16777215.0f) - 0.5f) * 1.5f;
}

static uint32_t cusr_experiment_float_bits(float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static const char* cusr_experiment_expression_name(int expression)
{
    switch (expression) {
        case CUSR_AST_SETTINGS_SIMPLE_ALU: return "simple_alu";
        case CUSR_AST_SETTINGS_COMPLEX_ALU: return "complex_alu";
        case CUSR_AST_SETTINGS_LIGHT_MUFU: return "light_mufu";
        case CUSR_AST_SETTINGS_HEAVY_MUFU: return "heavy_mufu";
    }

    return "unknown";
}

static bool cusr_experiment_parse_u64(const char* text, uint64_t* value)
{
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);

    if (text[0] == '\0' || end == nullptr || end[0] != '\0') return false;
    *value = (uint64_t)parsed;
    return true;
}

static void cusr_experiment_usage(const char* program)
{
    std::fprintf(stderr,
        "usage: %s [options]\n\n"
        "  --asts 4|8|16|32\n"
        "  --expr simple-alu|complex-alu|light-mufu|heavy-mufu|all\n"
        "  --rows N\n"
        "  --columns N\n"
        "  --settings N\n"
        "  --iters N\n",
        program);
}

static bool cusr_experiment_parse_expression(const char* text, int* expression)
{
    if (std::strcmp(text, "simple-alu") == 0) *expression = CUSR_AST_SETTINGS_SIMPLE_ALU;
    else if (std::strcmp(text, "complex-alu") == 0) *expression = CUSR_AST_SETTINGS_COMPLEX_ALU;
    else if (std::strcmp(text, "light-mufu") == 0) *expression = CUSR_AST_SETTINGS_LIGHT_MUFU;
    else if (std::strcmp(text, "heavy-mufu") == 0) *expression = CUSR_AST_SETTINGS_HEAVY_MUFU;
    else if (std::strcmp(text, "all") == 0) *expression = -1;
    else return false;
    return true;
}

static bool cusr_experiment_parse_args(int argc, char** argv, CusrExperimentConfig* config)
{
    for (int arg = 1; arg < argc; ++arg) {
        const char* name = argv[arg];

        if (std::strcmp(name, "--help") == 0) {
            cusr_experiment_usage(argv[0]);
            std::exit(0);
        }

        if (arg + 1 >= argc) return false;
        const char* value_text = argv[++arg];
        uint64_t value = 0u;

        if (std::strcmp(name, "--expr") == 0) {
            if (!cusr_experiment_parse_expression(value_text, &config->expression)) return false;
            continue;
        }

        if (!cusr_experiment_parse_u64(value_text, &value)) return false;

        if (std::strcmp(name, "--asts") == 0) config->asts = (int)value;
        else if (std::strcmp(name, "--rows") == 0) config->rows = (size_t)value;
        else if (std::strcmp(name, "--columns") == 0) config->columns = (uint32_t)value;
        else if (std::strcmp(name, "--settings") == 0) config->settings = (uint32_t)value;
        else if (std::strcmp(name, "--iters") == 0) config->iterations = (int)value;
        else return false;
    }

    const bool valid_asts = config->asts == 4 || config->asts == 8 || config->asts == 16 || config->asts == 32;
    return valid_asts && config->rows != 0u && config->columns != 0u && config->columns <= 32u &&
        config->settings != 0u && config->iterations > 0;
}

static bool cusr_experiment_make_data(const CusrExperimentConfig& config, CusrExperimentData* data)
{
    size_t x_count = 0u;
    size_t words_count = 0u;

    if (!cusr_experiment_checked_mul(config.rows, config.columns, &x_count) ||
        !cusr_experiment_checked_mul(config.settings, 8u, &words_count)) return false;

    std::vector<float> host_x(x_count);
    std::vector<float> host_target(config.rows);
    std::vector<uint8_t> host_masks(config.settings);
    std::vector<uint32_t> host_words(words_count);

    for (uint32_t column = 0u; column < config.columns; ++column) {
        for (size_t row = 0u; row < config.rows; ++row) {
            host_x[(size_t)column * config.rows + row] = cusr_experiment_random_float((uint32_t)row ^ (column + 1u) * 0x9e3779b9u);
        }
    }

    for (size_t row = 0u; row < config.rows; ++row) host_target[row] = cusr_experiment_random_float((uint32_t)row ^ 0xa511e9b3u) * 0.5f;

    for (uint32_t setting = 0u; setting < config.settings; ++setting) {
        uint8_t mask = 0u;

        for (uint32_t leaf = 0u; leaf < 8u; ++leaf) {
            const uint32_t hash = cusr_experiment_hash32(setting * 0x85ebca6bu + leaf * 0xc2b2ae35u);

            if ((hash & 3u) != 0u) {
                mask = (uint8_t)(mask | (uint8_t)(1u << leaf));
                host_words[(size_t)setting * 8u + leaf] = hash % config.columns;
            } else {
                host_words[(size_t)setting * 8u + leaf] = cusr_experiment_float_bits(cusr_experiment_random_float(hash) * 0.75f);
            }
        }

        host_masks[setting] = mask;
    }

    if (!data->X.allocate(x_count) || !data->target.allocate(config.rows) || !data->masks.allocate(config.settings) || !data->words.allocate(words_count)) {
        std::fprintf(stderr, "device input allocation failed\n");
        return false;
    }

    CUSR_EXPERIMENT_CUDA_RET(cudaMemcpy(data->X.data(), host_x.data(), data->X.bytes(), cudaMemcpyHostToDevice));
    CUSR_EXPERIMENT_CUDA_RET(cudaMemcpy(data->target.data(), host_target.data(), data->target.bytes(), cudaMemcpyHostToDevice));
    CUSR_EXPERIMENT_CUDA_RET(cudaMemcpy(data->masks.data(), host_masks.data(), data->masks.bytes(), cudaMemcpyHostToDevice));
    CUSR_EXPERIMENT_CUDA_RET(cudaMemcpy(data->words.data(), host_words.data(), data->words.bytes(), cudaMemcpyHostToDevice));
    return true;
}

template <typename Launch>
static bool cusr_experiment_time_gpu(Launch launch, int warmups, int iterations, double* seconds)
{
    CusrCudaEvent start;
    CusrCudaEvent stop;
    float milliseconds = 0.0f;

    if (!start.create() || !stop.create()) return false;

    for (int iteration = 0; iteration < warmups; ++iteration) CUSR_EXPERIMENT_CUDA_RET(launch());
    CUSR_EXPERIMENT_CUDA_RET(cudaDeviceSynchronize());
    CUSR_EXPERIMENT_CUDA_RET(cudaEventRecord(start.get()));
    for (int iteration = 0; iteration < iterations; ++iteration) CUSR_EXPERIMENT_CUDA_RET(launch());
    CUSR_EXPERIMENT_CUDA_RET(cudaEventRecord(stop.get()));
    CUSR_EXPERIMENT_CUDA_RET(cudaEventSynchronize(stop.get()));
    CUSR_EXPERIMENT_CUDA_RET(cudaEventElapsedTime(&milliseconds, start.get(), stop.get()));
    *seconds = (double)milliseconds * 1.0e-3 / iterations;
    return true;
}

template <int Expression, int NumAsts>
static cudaError_t cusr_experiment_launch_row_tiles(
    const CusrExperimentConfig& config,
    const CusrExperimentData& data,
    size_t rows,
    float* output)
{
    const size_t num_tiles = (rows + CUSR_AST_SETTINGS_TILE_ROWS - 1u) / CUSR_AST_SETTINGS_TILE_ROWS;
    const uint32_t shared_stride = config.columns | 1u;
    const size_t shared_bytes = ((size_t)CUSR_AST_SETTINGS_TILE_ROWS * shared_stride + CUSR_AST_SETTINGS_TILE_ROWS) * sizeof(float);

    cusr_ast_settings_row_tile_kernel<Expression, NumAsts><<<dim3((unsigned)num_tiles), CUSR_AST_SETTINGS_THREADS, shared_bytes>>>(
        data.X.data(), data.target.data(), rows, config.columns, config.rows, data.masks.data(), data.words.data(), 8u,
        config.settings, output);
    return cudaGetLastError();
}

static void cusr_experiment_report(const char* name, double seconds, double row_evaluations)
{
    std::printf("runtime path=%-22s ms=%9.3f row_evals_per_second=%.3e\n", name, seconds * 1.0e3, row_evaluations / seconds);
}

template <int Expression, int NumAsts>
static bool cusr_experiment_run_typed(const CusrExperimentConfig& config)
{
    CusrExperimentData data;
    cudaFuncAttributes row_attributes = {};
    const size_t output_count = (size_t)NumAsts * config.settings;

    if (!cusr_experiment_make_data(config, &data)) return false;
    CUSR_EXPERIMENT_CUDA_RET(cudaFuncGetAttributes(&row_attributes, cusr_ast_settings_row_tile_kernel<Expression, NumAsts>));

    CusrDeviceBuffer<float> output;
    if (!output.allocate(output_count)) return false;

    size_t free_bytes = 0u;
    size_t total_bytes = 0u;
    CUSR_EXPERIMENT_CUDA_RET(cudaMemGetInfo(&free_bytes, &total_bytes));

    std::printf("\nexpression=%s asts=%d settings=%u rows=%zu columns=%u\n",
        cusr_experiment_expression_name(Expression), NumAsts, config.settings, config.rows, config.columns);
    std::printf("registers=%d\n", row_attributes.numRegs);
    std::printf("memory final_output=%.3f_MiB device_free=%.3f_GiB\n",
        (double)output.bytes() / (double)(1ull << 20),
        (double)free_bytes / (double)(1ull << 30));

    const double row_evaluations = (double)config.rows * config.settings * NumAsts;
    double seconds = 0.0;

    const auto atomic = [&]() {
        cudaError_t result = cudaMemsetAsync(output.data(), 0, output.bytes());
        if (result != cudaSuccess) return result;
        return cusr_experiment_launch_row_tiles<Expression, NumAsts>(config, data, config.rows, output.data());
    };

    if (!cusr_experiment_time_gpu(atomic, 2, config.iterations, &seconds)) return false;
    cusr_experiment_report("row_atomic", seconds, row_evaluations);

    return true;
}

template <int NumAsts>
static bool cusr_experiment_dispatch_expression(const CusrExperimentConfig& config)
{
    if (config.expression == -1) {
        return cusr_experiment_run_typed<CUSR_AST_SETTINGS_SIMPLE_ALU, NumAsts>(config) &&
            cusr_experiment_run_typed<CUSR_AST_SETTINGS_COMPLEX_ALU, NumAsts>(config) &&
            cusr_experiment_run_typed<CUSR_AST_SETTINGS_LIGHT_MUFU, NumAsts>(config) &&
            cusr_experiment_run_typed<CUSR_AST_SETTINGS_HEAVY_MUFU, NumAsts>(config);
    }

    switch (config.expression) {
        case CUSR_AST_SETTINGS_SIMPLE_ALU: return cusr_experiment_run_typed<CUSR_AST_SETTINGS_SIMPLE_ALU, NumAsts>(config);
        case CUSR_AST_SETTINGS_COMPLEX_ALU: return cusr_experiment_run_typed<CUSR_AST_SETTINGS_COMPLEX_ALU, NumAsts>(config);
        case CUSR_AST_SETTINGS_LIGHT_MUFU: return cusr_experiment_run_typed<CUSR_AST_SETTINGS_LIGHT_MUFU, NumAsts>(config);
        case CUSR_AST_SETTINGS_HEAVY_MUFU: return cusr_experiment_run_typed<CUSR_AST_SETTINGS_HEAVY_MUFU, NumAsts>(config);
    }

    return false;
}

int main(int argc, char** argv)
{
    CusrExperimentConfig config;

    if (!cusr_experiment_parse_args(argc, argv, &config)) {
        cusr_experiment_usage(argv[0]);
        return 1;
    }

    bool success = false;
    switch (config.asts) {
        case 4: success = cusr_experiment_dispatch_expression<4>(config); break;
        case 8: success = cusr_experiment_dispatch_expression<8>(config); break;
        case 16: success = cusr_experiment_dispatch_expression<16>(config); break;
        case 32: success = cusr_experiment_dispatch_expression<32>(config); break;
    }

    return success ? 0 : 1;
}

#undef CUSR_EXPERIMENT_CUDA_RET
