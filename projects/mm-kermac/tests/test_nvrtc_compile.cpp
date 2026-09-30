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
#include <iostream>

#include <kermac.hpp>
#include <check_result_helper.h>

#include <string.h>

static const size_t NUM_BYTES = 1ull << 20;

using namespace kermac;

static bool has_cuda_device() {
    if (cuInit(0) != CUDA_SUCCESS) {
        return false;
    }
    int count = 0;
    if (cuDeviceGetCount(&count) != CUDA_SUCCESS) {
        return false;
    }
    return count > 0;
}

int
main() {
    if (!has_cuda_device()) {
        std::cerr << "SKIP: no CUDA device available\n";
        return 77;
    }

    try {
        Kermac kermac(0);

        void* host_mem = host_alloc(NUM_BYTES);
        {
            HostStackAllocator hsa(host_mem, NUM_BYTES);

            static const char* program_source =
                "extern \"C\" __global__ void kermac_nvrtc_test_kernel() {}\n";

            static const char* options[] = {
                "--std=c++14"
            };

            NvrtcCompiler compiler(
                kermac,
                hsa,
                "kermac_nvrtc_test.cu",
                program_source,
                /*headers=*/nullptr,
                /*include_names=*/nullptr,
                /*num_headers=*/0,
                options,
                sizeof(options) / sizeof(*options),
                /*verbose=*/false
            );

            size_t ptx_size = compiler.ptx(nullptr, 0);
            ASSERT(ptx_size > 1);

            HostRawBuffer<char> ptx_buffer(hsa, static_cast<int64_t>(ptx_size));
            size_t written = compiler.ptx(ptx_buffer.data(), ptx_size);
            ASSERT(written == ptx_size);

            char* ptx_data = ptx_buffer.data();
            ptx_data[ptx_size - 1] = '\0';
            ASSERT(ptx_data[0] != '\0');
            ASSERT(strstr(ptx_data, "kermac_nvrtc_test_kernel") != NULL);
        }

        host_free(host_mem);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << "\n";
        return 1;
    }
}
