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
// fmnist_krr_semiring_u8_inject.cpp

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#include <cuda.h>

#include <kermac.hpp>
#include <kermac_cpu.hpp>
#include <kermac_stack_ptx_descriptions.h>
#include <ptx_inject.h>
#include <check_result_helper.h>

#define INCBIN_SILENCE_BITCODE_WARNING
#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#define INCBIN_PREFIX g_
#include <incbin.h>

INCTXT(fmnist_u8_to_f32_annotated_ptx, PTX_FMNIST_U8_INJECT);

using namespace kermac;

constexpr std::size_t NUM_SECONDARY_STREAMS = 128;

static const MatrixPackedType train_packed_type = KERMAC_MATRIX_PACKED_TYPE_LOWER_TRIANGLE;
static const MatrixPackedType solve_packed_type = KERMAC_MATRIX_PACKED_TYPE_LOWER_TRIANGLE;

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

struct U8ToF32Kernel {
    CUmodule module = nullptr;
    CUfunction function = nullptr;

    U8ToF32Kernel() = default;
    U8ToF32Kernel(const U8ToF32Kernel&) = delete;
    U8ToF32Kernel& operator=(const U8ToF32Kernel&) = delete;

    U8ToF32Kernel(U8ToF32Kernel&& other) noexcept
        : module(other.module)
        , function(other.function)
    {
        other.module = nullptr;
        other.function = nullptr;
    }

    U8ToF32Kernel& operator=(U8ToF32Kernel&& other) noexcept {
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

    ~U8ToF32Kernel() noexcept {
        if (module != nullptr) {
            cuModuleUnload(module);
        }
    }
};

static U8ToF32Kernel
create_u8_to_f32_kernel(
    Kermac& handle,
    HostStackAllocator& hsa
) {
    U8ToF32Kernel kernel;

    PtxInjectHandle ptx_inject = nullptr;
    ptxInjectCheck(
        ptx_inject_create(&ptx_inject, g_fmnist_u8_to_f32_annotated_ptx_data)
    );

    size_t num_injects = 0;
    ptxInjectCheck(ptx_inject_num_injects(ptx_inject, &num_injects));
    ASSERT(num_injects == 1);

    size_t inject_idx = 0;
    ptxInjectCheck(
        ptx_inject_inject_info_by_name(
            ptx_inject,
            "convert",
            &inject_idx,
            nullptr,
            nullptr
        )
    );

    enum Register {
        REGISTER_IN_U32,
        REGISTER_OUT_F32,
        REGISTER_NUM_ENUMS
    };

    StackPtxRegister registers[REGISTER_NUM_ENUMS] = {
        { nullptr, STACK_PTX_STACK_TYPE_U32 },
        { nullptr, STACK_PTX_STACK_TYPE_F32 }
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
            "out_f32",
            nullptr,
            &registers[REGISTER_OUT_F32].name,
            nullptr,
            nullptr,
            nullptr
        )
    );

    static const float k_u8_to_f32_scale = 1.0f / 255.0f;
    static const StackPtxInstruction instructions[] = {
        stack_ptx_encode_input(REGISTER_IN_U32),
        stack_ptx_encode_ptx_instruction_cvt_rn_f32_u32,
        stack_ptx_encode_constant_f32(k_u8_to_f32_scale),
        stack_ptx_encode_ptx_instruction_mul_ftz_f32,
        stack_ptx_encode_return
    };

    static const StackPtxInstruction* instruction_stubs[] = { instructions };
    static const size_t requests[] = { REGISTER_OUT_F32 };
    static const size_t* request_stubs[] = { requests };
    static const size_t request_stub_sizes[] = { 1 };
    static const int execution_limit = 100;

    StackPtxCompilerInfo compiler_info = {
        100, // max_ast_size
        20,  // max_ast_to_visit_stack_depth
        128, // stack_size
        4,   // max_frame_depth
        16   // store_size
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

    cuCheck(cuModuleGetFunction(&kernel.function, kernel.module, "u8_to_f32_kernel"));
    ptxInjectCheck(ptx_inject_destroy(ptx_inject));

    return kernel;
}

static void
labels_u8_to_onehot(
    HostTensor<uint8_t>& labels_u8,
    HostTensor<float>& labels_f32,
    int64_t num_labels
) {
    int64_t num_samples = labels_u8.extent()[0];
    int64_t ld = labels_f32.raw().stride[1];

    float* out = labels_f32.ptr();
    std::fill(out, out + ld * num_labels, 0.0f);

    const uint8_t* in = labels_u8.ptr();
    for (int64_t i = 0; i < num_samples; ++i) {
        uint8_t label = in[i];
        ASSERT(label < num_labels);
        out[(int64_t)label * ld + i] = 1.0f;
    }
}

static void
u8_to_f32_run(
    const U8ToF32Kernel& kernel,
    DeviceTensor<uint8_t>& in,
    DeviceTensor<float>& out,
    CUstream stream
) {
    bool in_is_dry = in.raw().memory.stack_allocator->is_dry;
    bool out_is_dry = out.raw().memory.stack_allocator->is_dry;
    if (in_is_dry && out_is_dry) {
        return;
    }
    ASSERT(!in_is_dry && !out_is_dry);

    if (in.extent()[0] != out.extent()[0] || in.extent()[1] != out.extent()[1]) {
        ASSERT(false);
    }

    int64_t M = out.extent()[0];
    int64_t N = out.extent()[1];
    int64_t L = out.raw().num_modes == 2 ? 1 : out.extent()[2];
    int64_t L_in = in.raw().num_modes == 2 ? 1 : in.extent()[2];
    ASSERT(L == L_in);

    uint8_t* in_ptr = in.ptr();
    float* out_ptr = out.ptr();

    int64_t ld_in = in.raw().stride[1];
    int64_t ld_out = out.raw().stride[1];
    int64_t batch_stride_in = in.raw().stride[2];
    int64_t batch_stride_out = out.raw().stride[2];

    void* args[] = {
        (void*)&M,
        (void*)&N,
        (void*)&in_ptr, (void*)&ld_in, (void*)&batch_stride_in,
        (void*)&out_ptr, (void*)&ld_out, (void*)&batch_stride_out
    };

    static const unsigned int BLOCK_SIZE_X = 32;
    static const unsigned int BLOCK_SIZE_Y = 8;

    unsigned int num_blocks_M = (unsigned int)((M + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X);
    unsigned int num_blocks_N = (unsigned int)((N + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);

    cuCheck(
        cuLaunchKernel(
            kernel.function,
            num_blocks_M, num_blocks_N, (unsigned int)L,
            BLOCK_SIZE_X, BLOCK_SIZE_Y, 1,
            0, stream,
            args, nullptr
        )
    );
}

static void
func(
    Kermac& handle,
    HostStackAllocator& hsa,
    DeviceStackAllocator& dsa,
    Semiring& train_semiring,
    Semiring& test_semiring,
    const U8ToF32Kernel& u8_kernel,
    CUstream stream
) {
    bool is_dry = dsa.get().is_dry;

    const int64_t max_num_train = 60000;
    const int64_t max_num_test = 10000;
    const int64_t num_batches = 1;

    TensorCoreMode tcm = KERMAC_TENSOR_CORE_MODE_F32;

    int64_t num_train, num_test, num_dims, num_labels;

    fmnist_dims(
        max_num_train,
        max_num_test,
        &num_train,
        &num_test,
        &num_dims,
        &num_labels
    );

    HostTensor<uint8_t> x_train_u8(hsa, num_train, num_dims);
    HostTensor<uint8_t> y_train_u8(hsa, num_train);
    HostTensor<uint8_t> x_test_u8(hsa, num_test, num_dims);
    HostTensor<uint8_t> y_test_u8(hsa, num_test);

    fmnist_load_u8(x_train_u8, y_train_u8, x_test_u8, y_test_u8);

    HostTensor<float> y_train(hsa, num_train, num_labels);
    HostTensor<float> y_test(hsa, num_test, num_labels);
    labels_u8_to_onehot(y_train_u8, y_train, num_labels);
    labels_u8_to_onehot(y_test_u8, y_test, num_labels);

    DeviceTensor<float> d_solution(dsa, num_train, num_labels, num_batches);
    DeviceTensor<uint8_t> d_x_train_u8(dsa, num_train, num_dims, num_batches);
    DeviceTensor<uint8_t> d_x_test_u8(dsa, num_test, num_dims, num_batches);
    DeviceTensor<float> d_x_train(dsa, num_train, num_dims, num_batches);
    DeviceTensor<float> d_x_test(dsa, num_test, num_dims, num_batches);
    DeviceTensor<float> d_y_train(dsa, num_train, num_labels, 1);
    DeviceTensor<float> d_y_test(dsa, num_test, num_labels, 1);

    d_x_train_u8.copy_from(x_train_u8, stream);
    d_x_test_u8.copy_from(x_test_u8, stream);
    d_y_train.copy_from(y_train, stream);
    d_y_test.copy_from(y_test, stream);

    d_solution.copy_from(d_y_train, stream);

    u8_to_f32_run(u8_kernel, d_x_train_u8, d_x_train, stream);
    u8_to_f32_run(u8_kernel, d_x_test_u8, d_x_test, stream);

    {
        DeviceTensor<float> d_kernel_matrix(dsa, num_train, num_train, num_batches);

        if (!is_dry) {
            cuCheck(cuStreamSynchronize(stream));
        }

        train_semiring.run(d_x_train, d_x_train, d_kernel_matrix, stream);

        DeviceTensor<int32_t> d_factor_info(dsa, num_batches);
        DeviceTensor<int32_t> d_solve_info (dsa, num_batches);

        solve(
            handle,
            solve_packed_type,
            dsa,
            d_kernel_matrix,
            d_solution,
            d_factor_info,
            d_solve_info,
            stream
        );

        HostTensor<int32_t> h_factor_info(hsa, d_factor_info.extent());
        HostTensor<int32_t> h_solve_info(hsa, d_solve_info.extent());

        h_factor_info.copy_from(d_factor_info, stream);
        h_solve_info.copy_from(d_solve_info, stream);

        if (!is_dry) {
            int32_t* h_factor_info_ptr = h_factor_info.ptr();
            int32_t* h_solve_info_ptr = h_solve_info.ptr();
            for (int64_t i = 0; i < num_batches; i++) {
                ASSERT(h_factor_info_ptr[i] == 0);
                ASSERT(h_solve_info_ptr[i] == 0);
            }
        }
    }

    {
        DeviceTensor<float> d_kernel_matrix(dsa, num_train, num_test, num_batches);

        test_semiring.run(d_x_train, d_x_test, d_kernel_matrix, stream);

        DeviceTensor<float> d_preds(dsa, num_test, num_labels, num_batches);

        contraction(
            handle,
            dsa,
            tcm,
            1.0f,
            d_kernel_matrix, "mnl",
            d_solution,      "mcl",
            0.0f,
            d_preds,         "ncl",
            d_preds,         "ncl",
            stream
        );

        DeviceTensor<float> mse_tensor(dsa, num_batches);
        DeviceTensor<float> acc_tensor(dsa, num_batches);

        HostTensor<float> h_mse_tensor(hsa, mse_tensor.extent());
        HostTensor<float> h_acc_tensor(hsa, acc_tensor.extent());

        mse_accuracy(
            handle,
            dsa,
            d_preds,
            d_y_test,
            mse_tensor,
            acc_tensor,
            stream
        );

        if (!is_dry) {
            h_mse_tensor.copy_from(mse_tensor, stream);
            h_acc_tensor.copy_from(acc_tensor, stream);

            float* h_mse_tensor_ptr = h_mse_tensor.ptr();
            float* h_acc_tensor_ptr = h_acc_tensor.ptr();

            for (int64_t i = 0; i < num_batches; i++) {
                ASSERT(h_mse_tensor_ptr[i] < 0.017f);
                ASSERT(h_acc_tensor_ptr[i] > 0.9f);
            }
        }

        mse(
            handle,
            dsa,
            d_preds,
            d_y_test,
            mse_tensor,
            stream
        );

        if (!is_dry) {
            h_mse_tensor.copy_from(mse_tensor, stream);

            float* h_mse_tensor_ptr = h_mse_tensor.ptr();

            for (int64_t i = 0; i < num_batches; i++) {
                ASSERT(h_mse_tensor_ptr[i] < 0.017f);
            }
        }
    }
}

int
main() {
    if (!has_cuda_device()) {
        std::fprintf(stderr, "SKIP: no CUDA device available\n");
        return 77;
    }
    try {
        CUstream stream;

        Kermac handle(NUM_SECONDARY_STREAMS);

        cuCheck(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

        static const float bandwidth   = 10.0f;
        static const float regularizer = 1e-3f;
        static const float epsilon     = 1e-5f;

        size_t host_bytes = 1ull << 30;
        void* host_memory_ptr = host_alloc(host_bytes);
        HostStackAllocator hsa(host_memory_ptr, host_bytes);

        Semiring train_laplace_l2_symm_semiring =
            Semiring::laplace_l2_symm(
                handle, hsa,
                train_packed_type,
                bandwidth,
                regularizer,
                epsilon
            );
        Semiring test_laplace_l2_semiring = Semiring::laplace_l2(handle, hsa, bandwidth);

        U8ToF32Kernel u8_kernel = create_u8_to_f32_kernel(handle, hsa);

        std::size_t device_bytes = 0;

        {
            DeviceStackAllocator dsa_dry(nullptr, 0);

            func(handle, hsa, dsa_dry, train_laplace_l2_symm_semiring, test_laplace_l2_semiring, u8_kernel, stream);

            device_bytes = static_cast<std::size_t>(dsa_dry.get().largest_total_offset);
        }

        CUdeviceptr device_memory_ptr = (CUdeviceptr)device_alloc(handle, device_bytes);
        {
            DeviceStackAllocator dsa_real(
                reinterpret_cast<void*>(device_memory_ptr),
                device_bytes
            );

            func(handle, hsa, dsa_real, train_laplace_l2_symm_semiring, test_laplace_l2_semiring, u8_kernel, stream);
        }

        device_free(handle, (void*)device_memory_ptr);
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
