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
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>

#include <kermac.hpp>

using namespace kermac;

namespace {

constexpr int kDefaultEdgeItems = 4;
constexpr size_t kDefaultSeed = 1234;

bool has_cuda_device() {
    if (cuInit(0) != CUDA_SUCCESS) {
        return false;
    }
    int count = 0;
    if (cuDeviceGetCount(&count) != CUDA_SUCCESS) {
        return false;
    }
    return count > 0;
}

bool parse_dim(const char* text, int64_t* out) {
    if (!text || !*text) {
        return false;
    }
    char* end = nullptr;
    errno = 0;
    long long value = std::strtoll(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        return false;
    }
    if (value <= 0 || value > std::numeric_limits<int64_t>::max()) {
        return false;
    }
    *out = static_cast<int64_t>(value);
    return true;
}

void usage(const char* prog) {
    std::fprintf(stderr, "usage: %s [d0 [d1 [d2 [d3]]]]\n", prog);
    std::fprintf(stderr, "  default shape: 4 4\n");
}

size_t required_device_bytes(
    Kermac& handle,
    const Extent4& extent,
    size_t seed,
    CUstream stream
) {
    DeviceStackAllocator dsa_dry(nullptr, 0);
    {
        DeviceTensor<float> tensor(dsa_dry, extent);
    }
    return dsa_dry.get().largest_total_offset;
}

size_t required_host_bytes(
    const Extent4& extent
) {
    HostStackAllocator hsa_dry(nullptr, 0);
    {
        HostTensor<float> tensor(hsa_dry, extent);
    }
    return hsa_dry.get().largest_total_offset;
}

struct HostMemDeleter {
    void operator()(void* ptr) const noexcept {
        if (ptr) {
            (void)::kermac_host_free(ptr);
        }
    }
};

struct DeviceMemDeleter {
    KermacHandle handle = nullptr;
    void operator()(void* ptr) const noexcept {
        if (ptr && handle) {
            (void)::kermac_device_free(handle, ptr);
        }
    }
};

} // namespace

int main(int argc, char** argv) {
    int64_t dims[4] = {4, 4, 0, 0};

    if (argc > 1) {
        int dim_count = argc - 1;
        if (dim_count > 4) {
            usage(argv[0]);
            return 1;
        }
        for (int i = 0; i < dim_count; ++i) {
            if (!parse_dim(argv[i + 1], &dims[i])) {
                usage(argv[0]);
                return 1;
            }
        }
        for (int i = dim_count; i < 4; ++i) {
            dims[i] = 0;
        }
    }

    if (!has_cuda_device()) {
        std::fprintf(stderr, "No CUDA device available.\n");
        return 1;
    }

    try {
        Kermac kermac(0);
        CUstream stream = 0;
        Extent4 extent = {dims[0], dims[1], dims[2], dims[3]};

        size_t device_bytes = required_device_bytes(kermac, extent, kDefaultSeed, stream);
        size_t host_bytes = required_host_bytes(extent);

        void* host_ptr = nullptr;
        KERMAC_THROW_IF_ERROR(::kermac_host_alloc(&host_ptr, host_bytes));
        std::unique_ptr<void, HostMemDeleter> host_mem(host_ptr);

        void* device_ptr = nullptr;
        KERMAC_THROW_IF_ERROR(::kermac_device_alloc(kermac.get(), &device_ptr, device_bytes));
        std::unique_ptr<void, DeviceMemDeleter> device_mem(device_ptr, DeviceMemDeleter{kermac.get()});

        HostStackAllocator hsa(host_mem.get(), host_bytes);
        DeviceStackAllocator dsa(device_mem.get(), device_bytes);

        DeviceTensor<float> tensor(dsa, extent);
        rng(kermac, tensor, RNGType::KERMAC_RNG_TYPE_UNIFORM, 1.0f, 0.0f, stream);

        tensor.print(hsa, kDefaultEdgeItems, stream, "random tensor ");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }

    return 0;
}
