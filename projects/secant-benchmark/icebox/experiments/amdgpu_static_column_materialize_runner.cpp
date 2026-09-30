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
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

#define HIP_CHECK(call) \
    do { \
        const hipError_t hip_check_result = (call); \
        if (hip_check_result != hipSuccess) { \
            std::fprintf(stderr, "%s failed: %s\n", #call, hipGetErrorString(hip_check_result)); \
            return 1; \
        } \
    } while (0)

static int run_materialize_probe(const char* code_object_path) {
    const size_t num_inputs = 8u;
    const size_t num_rows = 4096u;
    const size_t input_leading_dimension = num_rows;
    const unsigned int block_size = 128u;
    const unsigned int grid_size = (unsigned int)((num_rows + block_size - 1u) / block_size);

    std::vector<float> input(num_inputs * input_leading_dimension);
    std::vector<float> output(num_rows, 0.0f);
    for (size_t input_index = 0u; input_index < num_inputs; ++input_index) {
        for (size_t row = 0u; row < num_rows; ++row) {
            input[input_index * input_leading_dimension + row] =
                (float)(input_index + 1u) * 0.25f + (float)row * 0.0001f;
        }
    }

    HIP_CHECK(hipSetDevice(0));

    hipModule_t module = nullptr;
    hipFunction_t function = nullptr;
    HIP_CHECK(hipModuleLoad(&module, code_object_path));
    HIP_CHECK(hipModuleGetFunction(
        &function,
        module,
        "secant_amdgpu_static_column_materialize_probe"));

    int num_registers = 0;
    HIP_CHECK(hipFuncGetAttribute(&num_registers, HIP_FUNC_ATTRIBUTE_NUM_REGS, function));

    float* input_device = nullptr;
    float* output_device = nullptr;
    HIP_CHECK(hipMalloc((void**)&input_device, input.size() * sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&output_device, output.size() * sizeof(float)));
    HIP_CHECK(hipMemcpy(
        input_device,
        input.data(),
        input.size() * sizeof(float),
        hipMemcpyHostToDevice));

    const float* input_argument = input_device;
    float* output_argument = output_device;
    void* arguments[] = {
        &input_argument,
        (void*)&input_leading_dimension,
        (void*)&num_rows,
        &output_argument
    };

    HIP_CHECK(hipModuleLaunchKernel(
        function,
        grid_size,
        1u,
        1u,
        block_size,
        1u,
        1u,
        0u,
        nullptr,
        arguments,
        nullptr));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(
        output.data(),
        output_device,
        output.size() * sizeof(float),
        hipMemcpyDeviceToHost));

    float max_abs_error = 0.0f;
    for (size_t row = 0u; row < num_rows; ++row) {
        float expected = 0.0f;
        for (size_t input_index = 0u; input_index < num_inputs; ++input_index) {
            expected += input[input_index * input_leading_dimension + row];
        }
        max_abs_error = std::fmax(max_abs_error, std::fabs(output[row] - expected));
    }

    HIP_CHECK(hipFree(output_device));
    HIP_CHECK(hipFree(input_device));
    HIP_CHECK(hipModuleUnload(module));

    std::printf(
        "rows=%zu registers=%d max_abs_error=%.9g sample=%.9g\n",
        num_rows,
        num_registers,
        max_abs_error,
        output[0]);
    return max_abs_error <= 1.0e-5f ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <patched.hsaco>\n", argv[0]);
        return 1;
    }
    return run_materialize_probe(argv[1]);
}
