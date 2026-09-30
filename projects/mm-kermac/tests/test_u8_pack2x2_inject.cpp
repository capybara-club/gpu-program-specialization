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
#include <cstdint>
#include <cstdio>
#include <exception>
#include <vector>

#include <cuda.h>

#include <kermac.hpp>
#include <kermac_stack_ptx_descriptions.h>
#include <ptx_inject.h>
#include <check_result_helper.h>

#define INCBIN_SILENCE_BITCODE_WARNING
#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#define INCBIN_PREFIX g_
#include <incbin.h>

INCTXT(u8_pack2x2_16_annotated_ptx, PTX_U8_PACK2X2_16_INJECT);
INCTXT(u8_pack2x2_28_annotated_ptx, PTX_U8_PACK2X2_28_INJECT);
INCTXT(u8_pack2x2_32_annotated_ptx, PTX_U8_PACK2X2_32_INJECT);

using namespace kermac;

constexpr std::size_t NUM_SECONDARY_STREAMS = 0;
constexpr int64_t kNumBatches = 16;

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

struct PackedConvKernel {
    CUmodule module = nullptr;
    CUfunction function = nullptr;

    PackedConvKernel() = default;
    PackedConvKernel(const PackedConvKernel&) = delete;
    PackedConvKernel& operator=(const PackedConvKernel&) = delete;

    PackedConvKernel(PackedConvKernel&& other) noexcept
        : module(other.module)
        , function(other.function)
    {
        other.module = nullptr;
        other.function = nullptr;
    }

    PackedConvKernel& operator=(PackedConvKernel&& other) noexcept {
        if (this != &other) {
            if (module != nullptr) {
                cuModuleUnload(module);
            }
            module = other.module;
            function = other.function;
            other.module = nullptr;
            other.function = nullptr;
        }
        return *this;
    }

    ~PackedConvKernel() noexcept {
        if (module != nullptr) {
            cuModuleUnload(module);
        }
    }
};

static PackedConvKernel
create_packed_conv_kernel(
    Kermac& handle,
    HostStackAllocator& hsa,
    const char* annotated_ptx,
    const char* function_name,
    uint32_t add_constant
) {
    PackedConvKernel kernel;

    PtxInjectHandle ptx_inject = nullptr;
    ptxInjectCheck(ptx_inject_create(&ptx_inject, annotated_ptx));

    size_t num_injects = 0;
    ptxInjectCheck(ptx_inject_num_injects(ptx_inject, &num_injects));
    ASSERT(num_injects == 1);

    size_t inject_idx = 0;
    ptxInjectCheck(
        ptx_inject_inject_info_by_name(
            ptx_inject,
            "transform",
            &inject_idx,
            nullptr,
            nullptr
        )
    );

    enum Register {
        REGISTER_IN_U32,
        REGISTER_OUT_U32,
        REGISTER_NUM_ENUMS
    };

    StackPtxRegister registers[REGISTER_NUM_ENUMS] = {
        { nullptr, STACK_PTX_STACK_TYPE_U32 },
        { nullptr, STACK_PTX_STACK_TYPE_U32 }
    };

    ptxInjectCheck(
        ptx_inject_variable_info_by_name(
            ptx_inject,
            inject_idx,
            "in_u32",
            nullptr,
            &registers[REGISTER_IN_U32].name,
            nullptr,
            nullptr,
            nullptr
        )
    );

    ptxInjectCheck(
        ptx_inject_variable_info_by_name(
            ptx_inject,
            inject_idx,
            "out_u32",
            nullptr,
            &registers[REGISTER_OUT_U32].name,
            nullptr,
            nullptr,
            nullptr
        )
    );

    StackPtxInstruction instructions[] = {
        stack_ptx_encode_input(REGISTER_IN_U32),
        stack_ptx_encode_constant_u32(add_constant),
        stack_ptx_encode_ptx_instruction_add_u32,
        stack_ptx_encode_return
    };

    const StackPtxInstruction* instruction_stubs[] = { instructions };
    const size_t requests[] = { REGISTER_OUT_U32 };
    const size_t* request_stubs[] = { requests };
    const size_t request_stub_sizes[] = { 1 };

    static const int execution_limit = 32;
    StackPtxCompilerInfo compiler_info = {
        100,
        20,
        128,
        4,
        16
    };

    stack_ptx_inject_compile(
        handle,
        hsa,
        ptx_inject,
        &compiler_info,
        &stack_ptx_stack_info,
        execution_limit,
        registers,
        REGISTER_NUM_ENUMS,
        instruction_stubs,
        request_stubs,
        request_stub_sizes,
        1,
        &kernel.module
    );

    cuCheck(cuModuleGetFunction(&kernel.function, kernel.module, function_name));
    ptxInjectCheck(ptx_inject_destroy(ptx_inject));

    return kernel;
}

template <int IMAGE_N>
static void
fill_synthetic_images(
    HostTensor<uint8_t>& h_images
) {
    constexpr int64_t kImageElems = (int64_t)IMAGE_N * (int64_t)IMAGE_N;

    ASSERT(h_images.extent()[0] == kImageElems);
    ASSERT(h_images.extent()[1] == kNumBatches);

    uint8_t* ptr = h_images.ptr();
    const int64_t batch_stride = h_images.raw().stride[1];

    for (int64_t batch = 0; batch < kNumBatches; ++batch) {
        for (int64_t idx = 0; idx < kImageElems; ++idx) {
            uint32_t x = 0x9E3779B9u ^ (uint32_t)(batch * 0x85EBCA6Bu) ^ (uint32_t)(idx * 0xC2B2AE35u);
            x ^= x >> 16;
            x *= 0x7FEB352Du;
            x ^= x >> 15;
            x *= 0x846CA68Bu;
            x ^= x >> 16;
            ptr[batch * batch_stride + idx] = (uint8_t)(x & 0xFFu);
        }
    }
}

template <int IMAGE_N>
static void
compute_expected_tiles(
    const HostTensor<uint8_t>& h_images,
    uint32_t add_constant,
    std::vector<uint32_t>& expected
) {
    constexpr int64_t kImageElems = (int64_t)IMAGE_N * (int64_t)IMAGE_N;
    constexpr int64_t kOutSide = (int64_t)IMAGE_N - 1;
    constexpr int64_t kOutElems = kOutSide * kOutSide;

    ASSERT(h_images.extent()[0] == kImageElems);
    ASSERT(h_images.extent()[1] == kNumBatches);

    const uint8_t* ptr = h_images.ptr();
    const int64_t batch_stride = h_images.raw().stride[1];

    expected.resize((size_t)(kNumBatches * kOutElems));

    for (int64_t batch = 0; batch < kNumBatches; ++batch) {
        const uint8_t* image = ptr + batch * batch_stride;
        for (int64_t row = 0; row < kOutSide; ++row) {
            for (int64_t col = 0; col < kOutSide; ++col) {
                const int64_t out_idx = row * kOutSide + col;
                const int64_t tl = row * (int64_t)IMAGE_N + col;

                const uint32_t p00 = (uint32_t)image[tl];
                const uint32_t p01 = (uint32_t)image[tl + 1];
                const uint32_t p10 = (uint32_t)image[tl + (int64_t)IMAGE_N];
                const uint32_t p11 = (uint32_t)image[tl + (int64_t)IMAGE_N + 1];

                const uint32_t packed = p00 | (p01 << 8) | (p10 << 16) | (p11 << 24);
                expected[(size_t)(batch * kOutElems + out_idx)] = packed + add_constant;
            }
        }
    }
}

template <int IMAGE_N>
static void
assert_matches_expected(
    const HostTensor<uint32_t>& h_output,
    const std::vector<uint32_t>& expected,
    uint32_t add_constant
) {
    constexpr int64_t kOutSide = (int64_t)IMAGE_N - 1;
    constexpr int64_t kOutElems = kOutSide * kOutSide;

    ASSERT(h_output.extent()[0] == kOutElems);
    ASSERT(h_output.extent()[1] == kNumBatches);

    const uint32_t* out_ptr = h_output.ptr();
    const int64_t out_batch_stride = h_output.raw().stride[1];

    for (int64_t batch = 0; batch < kNumBatches; ++batch) {
        for (int64_t idx = 0; idx < kOutElems; ++idx) {
            const uint32_t got = out_ptr[batch * out_batch_stride + idx];
            const uint32_t exp = expected[(size_t)(batch * kOutElems + idx)];
            if (got != exp) {
                const int64_t row = idx / kOutSide;
                const int64_t col = idx - row * kOutSide;
                std::fprintf(
                    stderr,
                    "Mismatch IMAGE_N=%d add=%u batch=%lld row=%lld col=%lld got=%u exp=%u\n",
                    IMAGE_N,
                    add_constant,
                    (long long)batch,
                    (long long)row,
                    (long long)col,
                    got,
                    exp
                );
                ASSERT(false);
            }
        }
    }
}

static void
launch_kernel(
    const PackedConvKernel& kernel,
    int64_t num_batches,
    DeviceTensor<uint8_t>& d_images,
    DeviceTensor<uint32_t>& d_output,
    CUstream stream
) {
    bool in_is_dry = d_images.raw().memory.stack_allocator->is_dry;
    bool out_is_dry = d_output.raw().memory.stack_allocator->is_dry;
    if (in_is_dry && out_is_dry) {
        return;
    }
    ASSERT(!in_is_dry && !out_is_dry);

    uint8_t* in_ptr = d_images.ptr();
    uint32_t* out_ptr = d_output.ptr();

    int64_t in_batch_stride = d_images.raw().stride[1];
    int64_t out_batch_stride = d_output.raw().stride[1];

    void* args[] = {
        (void*)&num_batches,
        (void*)&in_ptr,
        (void*)&in_batch_stride,
        (void*)&out_ptr,
        (void*)&out_batch_stride
    };

    cuCheck(
        cuLaunchKernel(
            kernel.function,
            1, 1, 1,
            32, 1, 1,
            0,
            stream,
            args,
            nullptr
        )
    );
}

template <int IMAGE_N>
static void
run_size_case(
    Kermac& handle,
    HostStackAllocator& hsa,
    DeviceStackAllocator& dsa,
    CUstream stream,
    const char* annotated_ptx,
    const char* kernel_function_name
) {
    constexpr int64_t kImageElems = (int64_t)IMAGE_N * (int64_t)IMAGE_N;
    constexpr int64_t kOutElems = ((int64_t)IMAGE_N - 1) * ((int64_t)IMAGE_N - 1);

    HostTensor<uint8_t> h_images(hsa, kImageElems, kNumBatches);
    DeviceTensor<uint8_t> d_images(dsa, kImageElems, kNumBatches);
    DeviceTensor<uint32_t> d_output(dsa, kOutElems, kNumBatches);

    fill_synthetic_images<IMAGE_N>(h_images);

    if (dsa.get().is_dry) {
        return;
    }

    HostTensor<uint32_t> h_output(hsa, kOutElems, kNumBatches);

    d_images.copy_from(h_images, stream);

    for (uint32_t add_constant : { 0u, 1u }) {
        PackedConvKernel kernel = create_packed_conv_kernel(
            handle,
            hsa,
            annotated_ptx,
            kernel_function_name,
            add_constant
        );

        launch_kernel(kernel, kNumBatches, d_images, d_output, stream);

        h_output.copy_from(d_output, stream);
        cuCheck(cuStreamSynchronize(stream));

        std::vector<uint32_t> expected;
        compute_expected_tiles<IMAGE_N>(h_images, add_constant, expected);
        assert_matches_expected<IMAGE_N>(h_output, expected, add_constant);
    }
}

static void
run_all_cases(
    Kermac& handle,
    HostStackAllocator& hsa,
    DeviceStackAllocator& dsa,
    CUstream stream
) {
    run_size_case<16>(
        handle,
        hsa,
        dsa,
        stream,
        (const char*)g_u8_pack2x2_16_annotated_ptx_data,
        "u8_pack2x2_overlap_16x16_kernel"
    );

    run_size_case<28>(
        handle,
        hsa,
        dsa,
        stream,
        (const char*)g_u8_pack2x2_28_annotated_ptx_data,
        "u8_pack2x2_overlap_28x28_kernel"
    );

    run_size_case<32>(
        handle,
        hsa,
        dsa,
        stream,
        (const char*)g_u8_pack2x2_32_annotated_ptx_data,
        "u8_pack2x2_overlap_32x32_kernel"
    );
}

int
main() {
    if (!has_cuda_device()) {
        std::fprintf(stderr, "SKIP: no CUDA device available\n");
        return 77;
    }

    try {
        Kermac handle(NUM_SECONDARY_STREAMS);

        CUstream stream;
        cuCheck(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

        const size_t host_bytes = 1ull << 27;
        void* host_memory_ptr = host_alloc(host_bytes);
        HostStackAllocator hsa(host_memory_ptr, host_bytes);

        size_t device_bytes = 0;
        {
            DeviceStackAllocator dsa_dry(nullptr, 0);
            run_all_cases(handle, hsa, dsa_dry, stream);
            device_bytes = (size_t)dsa_dry.get().largest_total_offset;
        }

        void* device_memory_ptr = device_alloc(handle, device_bytes);
        {
            DeviceStackAllocator dsa_real(device_memory_ptr, device_bytes);
            run_all_cases(handle, hsa, dsa_real, stream);
        }

        device_free(handle, device_memory_ptr);
        host_free(host_memory_ptr);

        cuCheck(cuStreamDestroy(stream));
        return 0;
    }
    catch (const kermac::Error& e) {
        std::fprintf(stderr, "Kermac error: %s\n", e.what());
        return 1;
    }
    catch (const std::exception& e) {
        std::fprintf(stderr, "Exception: %s\n", e.what());
        return 1;
    }
}
