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
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <string>

#include "kermac.h"

namespace kermac {

using Result        = KermacResult;
using HandleRaw     = KermacHandle;
using TensorRaw     = KermacTensor;
using StackAllocRaw = KermacStackAllocator;
using DataType      = KermacDataType;
using MemorySpace   = KermacMemorySpace;
using RNGType       = KermacRNGType;
using MatrixPackedType = KermacMatrixPackedType;
using TensorCoreMode   = KermacTensorCoreMode;
using SemiringRaw = ::KermacSemiring;
using SemiringGradientRaw = ::KermacSemiringGradient;
using AgopRaw = ::KermacAgop;
using AgopBackend = KermacAgopBackend;
using AgopOutput = KermacAgopOutput;
using ElementwiseRaw = ::KermacElementwise;
using CompilerRaw = ::KermacCompilerHandle;
using NvrtcCompilerRaw = ::KermacNvrtcCompilerHandle;
using Extent4 = KermacExtent;

class Error : public std::runtime_error {
public:
    Error(
        Result result, 
        const std::string& msg
    ) : std::runtime_error(msg), 
        result_(result) 
    {}

    Result result() const noexcept { return result_; }

private:
    Result result_;
};

inline 
const char* 
result_to_string(
    Result r
) noexcept {
    return ::kermac_result_to_string(r);
}

namespace detail {

inline 
void 
throw_if_error(
    Result result,
    const char* expr,
    const char* file,
    int line
) {
    if (result == KERMAC_SUCCESS) {
        return;
    }

    const char* msg = ::kermac_result_to_string(result);
    std::string full;
    full.reserve(256);

    full += "kermac call failed: ";
    full += expr;
    full += " -> ";
    full += (msg ? msg : "<unknown>");
    full += " (code ";
    full += std::to_string(static_cast<int>(result));
    full += ") at ";
    full += file;
    full += ":";
    full += std::to_string(line);

    throw Error(result, full);
}

inline 
void 
throw_if_cuda_error(
    CUresult cuda_result,
    const char* expr,
    const char* file,
    int line
) {
    if (cuda_result == CUDA_SUCCESS) {
        return;
    }

    const char* err_name  = nullptr;
    const char* err_str   = nullptr;

    CUresult name_res = cuGetErrorName(cuda_result, &err_name);
    CUresult str_res  = cuGetErrorString(cuda_result, &err_str);

    std::string msg;
    msg.reserve(256);

    msg += "CUDA call failed: ";
    msg += expr;
    msg += " -> ";

    if (name_res == CUDA_SUCCESS && err_name) {
        msg += err_name;
    } else {
        msg += "<unknown CUDA error name>";
    }

    msg += " (code ";
    msg += std::to_string(static_cast<int>(cuda_result));
    msg += ")";

    if (str_res == CUDA_SUCCESS && err_str) {
        msg += " : ";
        msg += err_str;
    }

    msg += " at ";
    msg += file;
    msg += ":";
    msg += std::to_string(line);

    // Lens the CUDA error into a single KermacResult enum
    // while preserving all CUDA details in the message.
    throw Error(KERMAC_ERROR_CUDA, msg);
}

}

#define KERMAC_THROW_IF_ERROR(expr) \
    ::kermac::detail::throw_if_error((expr), #expr, __FILE__, __LINE__)

template <typename T>
struct DataTypeTraits {
    static constexpr bool valid = false;
    static constexpr DataType value = static_cast<DataType>(-1);
};

// Fundamental types
template <>
struct DataTypeTraits<float> {
    static constexpr bool valid = true;
    static constexpr DataType value = KERMAC_DATA_TYPE_FLOAT;
};

template <>
struct DataTypeTraits<double> {
    static constexpr bool valid = true;
    static constexpr DataType value = KERMAC_DATA_TYPE_DOUBLE;
};

template <>
struct DataTypeTraits<int32_t> {
    static constexpr bool valid = true;
    static constexpr DataType value = KERMAC_DATA_TYPE_INT;
};

template <>
struct DataTypeTraits<uint32_t> {
    static constexpr bool valid = true;
    static constexpr DataType value = KERMAC_DATA_TYPE_UINT;
};

template <>
struct DataTypeTraits<uint8_t> {
    static constexpr bool valid = true;
    static constexpr DataType value = KERMAC_DATA_TYPE_BYTE;
};

template <>
struct DataTypeTraits<char> {
    static constexpr bool valid = true;
    static constexpr DataType value = KERMAC_DATA_TYPE_BYTE;
};

// Generic pointer support
template <typename T>
struct DataTypeTraits<T*> {
    static constexpr bool valid = true;
    static constexpr DataType value = KERMAC_DATA_TYPE_POINTER;
};

template <typename T>
struct DataTypeTraits<const T*> : DataTypeTraits<T*> {};

class Kermac {
public:
    Kermac(
        size_t num_secondary_streams
    ) {
        Result r = ::kermac_create(&handle_);
        KERMAC_THROW_IF_ERROR(r);
        valid_ = true;

        num_secondary_streams_ = num_secondary_streams;
        if (num_secondary_streams_ > 0) {
            secondary_streams_ = static_cast<CUstream*>(
                std::malloc(num_secondary_streams_ * sizeof(CUstream))
            );
            if (!secondary_streams_) {
                num_secondary_streams_ = 0;
                (void)::kermac_destroy(handle_);
                valid_ = false;
                throw std::bad_alloc();
            }
        }

        for (size_t i = 0; i < num_secondary_streams_; i++) {
            CUresult cuda_result = cuStreamCreate(&secondary_streams_[i], CU_STREAM_NON_BLOCKING);
            if (cuda_result != CUDA_SUCCESS) {
                for (size_t j = 0; j < i; ++j) {
                    (void)cuStreamDestroy(secondary_streams_[j]);
                }
                std::free(secondary_streams_);
                secondary_streams_       = nullptr;
                num_secondary_streams_   = 0;
                (void)::kermac_destroy(handle_);
                valid_ = false;

                ::kermac::detail::throw_if_cuda_error(
                    cuda_result,
                    "cuStreamCreate(&secondary_streams_[i], CU_STREAM_NON_BLOCKING)",
                    __FILE__,
                    __LINE__
                );
            }
        }
    }

    ~Kermac() noexcept {
        if (valid_) {
            for (size_t i = 0; i < num_secondary_streams_; i++) {
                (void)cuStreamDestroy(secondary_streams_[i]);
            }
            std::free(secondary_streams_);
            secondary_streams_ = nullptr;
            num_secondary_streams_ = 0;
            (void)::kermac_destroy(handle_);
            valid_ = false;
        }
    }

    Kermac(const Kermac&)            = delete;
    Kermac& operator=(const Kermac&) = delete;
    Kermac(Kermac&&)                 = delete;
    Kermac& operator=(Kermac&&)      = delete;

    HandleRaw get() const noexcept { return handle_; }
    
    CUstream* secondary_streams_ = nullptr;
    size_t num_secondary_streams_ = 0;

private:
    HandleRaw handle_{};
    bool      valid_ = false;
};

inline
void* 
host_alloc(
    std::size_t num_bytes
) {
    void* ptr;
    KERMAC_THROW_IF_ERROR(
        ::kermac_host_alloc(&ptr, num_bytes)
    );
    return ptr;
}

inline
void*
host_alloc_pinned(
    std::size_t num_bytes
) {
    void* ptr;
    KERMAC_THROW_IF_ERROR(
        ::kermac_host_alloc_pinned(&ptr, num_bytes)
    );
    return ptr;
}

inline
void 
host_free(
    void* ptr
) {
    KERMAC_THROW_IF_ERROR(
        ::kermac_host_free(ptr)
    );
}

inline
void
host_free_pinned(
    void* ptr
) {
    KERMAC_THROW_IF_ERROR(
        ::kermac_host_free_pinned(ptr)
    );
}

inline
void*
device_alloc(
    Kermac& handle, 
    std::size_t num_bytes
) {
    void* ptr;
    KERMAC_THROW_IF_ERROR(
        ::kermac_device_alloc(handle.get(), &ptr, num_bytes)
    );
    return ptr;
}

inline
void 
device_free(
    Kermac& handle, 
    void* ptr
) {
    KERMAC_THROW_IF_ERROR(
        ::kermac_device_free(handle.get(), ptr)
    );
}

template <MemorySpace Space>
class StackAllocator {
public:
    StackAllocator(
        void* memory, 
        std::size_t num_bytes
    ) {
        Result r = ::kermac_stack_allocator_create(
            &alloc_,
            Space,
            memory,
            num_bytes
        );
        KERMAC_THROW_IF_ERROR(r);
        valid_ = true;
    }

    ~StackAllocator() noexcept {
        if (valid_) {
            (void)::kermac_stack_allocator_destroy(alloc_);
        }
    }

    StackAllocator(const StackAllocator&)            = delete;
    StackAllocator& operator=(const StackAllocator&) = delete;
    StackAllocator(StackAllocator&&)                 = delete;
    StackAllocator& operator=(StackAllocator&&)      = delete;

    StackAllocRaw        get()      const noexcept { return alloc_; }
    StackAllocRaw*       get_ptr()        noexcept { return &alloc_; }
    const StackAllocRaw* get_ptr()  const noexcept { return &alloc_; }

private:
    StackAllocRaw alloc_{};
    bool          valid_ = false;
};

using DeviceStackAllocator = StackAllocator<KERMAC_MEMORY_SPACE_DEVICE>;
using HostStackAllocator   = StackAllocator<KERMAC_MEMORY_SPACE_HOST>;

template <typename T, MemorySpace Space>
class Tensor {
public:
    using value_type = T;
    using Extent = Extent4;

private:
    static constexpr DataType dtype() {
        static_assert(
            DataTypeTraits<T>::valid,
            "No DataTypeTraits specialization for this T. "
            "Either add one, or use RawBuffer<T> if you just want bytes."
        );
        return DataTypeTraits<T>::value;
    }

public:
    Tensor() = delete;

    Tensor(
        StackAllocator<Space>& allocator,
        std::int64_t dim0,
        std::int64_t dim1 = 0,
        std::int64_t dim2 = 0,
        std::int64_t dim3 = 0
    ) {
        std::int64_t extents[4] = { dim0, dim1, dim2, dim3 };

        Result r = ::kermac_tensor_create(
            &tensor_,
            dtype(),           // <--- uses DataTypeTraits<T>
            extents,
            allocator.get_ptr()
        );
        KERMAC_THROW_IF_ERROR(r);
        valid_  = true;
        is_view_ = false;
    }

    Tensor(
        StackAllocator<Space>& allocator,
        const Extent& ext
    ) {
        Result r = ::kermac_tensor_create(
            &tensor_,
            dtype(),          // <--- uses DataTypeTraits<T>
            ext,              // int64_t[4]
            allocator.get_ptr()
        );
        KERMAC_THROW_IF_ERROR(r);
        valid_   = true;
        is_view_ = false;
    }

    Tensor(
        const Tensor& base,
        std::uint32_t mode,
        std::size_t slice
    ) {
        Result r = ::kermac_tensor_view_create(
            base.tensor_,
            &tensor_,
            mode,
            slice
        );
        KERMAC_THROW_IF_ERROR(r);
        valid_  = true;
        is_view_ = true;
    }

    ~Tensor() noexcept {
        if (!valid_) {
            return;
        }
        if (is_view_) {
            (void)::kermac_tensor_view_destroy(tensor_);
        } else {
            (void)::kermac_tensor_destroy(tensor_);
        }
    }

    Tensor(const Tensor&)            = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&&)                 = delete;
    Tensor& operator=(Tensor&&)      = delete;

    const Extent& extent() const noexcept {
        // tensor_.extent is already int64_t[4], so this is a direct ref
        return reinterpret_cast<const Extent&>(tensor_.extent);
    }

    const std::int64_t* extent_data() const noexcept {
        return tensor_.extent;
    }

    TensorRaw raw() const noexcept { return tensor_; }

    T* ptr() {
        void* raw_ptr = nullptr;
        KERMAC_THROW_IF_ERROR(
            kermac_memory_pointer(tensor_.memory, &raw_ptr)
        );
        return static_cast<T*>(raw_ptr);
    }

    const T* ptr() const {
        void* raw_ptr = nullptr;
        KERMAC_THROW_IF_ERROR(
            kermac_memory_pointer(tensor_.memory, &raw_ptr)
        );
        return static_cast<const T*>(raw_ptr);
    }

    template <MemorySpace SrcSpace>
    void copy_from(
        const Tensor<T, SrcSpace>& src, 
        CUstream stream
    ) {
        KERMAC_THROW_IF_ERROR(
            ::kermac_tensor_copy(src.raw(), tensor_, stream)
        );
    }

    void print(
        HostStackAllocator& hsa,
        int edge_items,
        CUstream stream,
        const char* str = NULL
    ) {
        static_assert(
            Space == KERMAC_MEMORY_SPACE_DEVICE,
            "Tensor<T,Space>::print(HostStackAllocator&,int,CUstream) "
            "is only valid for device tensors"
        );

        if (!raw().memory.stack_allocator->is_dry) {   
            if (str != NULL) {
                printf("%s", str);
            }
            printf("(");
            for (uint32_t i = 0; i < tensor_.num_modes; i++) {
                printf("%zi%s", tensor_.extent[i], i == tensor_.num_modes - 1 ? "" : ",");
            }
            printf("):\n");
        }
        KERMAC_THROW_IF_ERROR(
            ::kermac_tensor_print(
                tensor_,
                hsa.get_ptr(),
                edge_items,
                stream
            )
        );
    }

    void print(
        int edge_items, 
        const char* str = NULL
    ) {
        static_assert(
            Space == KERMAC_MEMORY_SPACE_HOST,
            "Tensor<T,Space>::print(int) "
            "is only valid for host tensors"
        );

        if (str != NULL) {
            printf("%s", str);
        }
        printf("(");
        for (int i = 0; i < tensor_.num_modes; i++) {
            printf("%zi%s", tensor_.extent[i], i == tensor_.num_modes - 1 ? "" : ",");
        }
        printf("):\n");
        KERMAC_THROW_IF_ERROR(
            ::kermac_tensor_print(
                tensor_,
                /*hsa=*/nullptr,
                edge_items,
                /*stream=*/0
            )
        );
    }

private:
    TensorRaw tensor_{};
    bool      valid_  = false;
    bool      is_view_ = false;
};

template <typename T>
using DeviceTensor = Tensor<T, KERMAC_MEMORY_SPACE_DEVICE>;

template <typename T>
using HostTensor   = Tensor<T, KERMAC_MEMORY_SPACE_HOST>;

template <typename T, MemorySpace Space>
class RawBuffer {
public:
    using value_type = T;
    using ByteTensor = Tensor<uint8_t, Space>;

    RawBuffer(StackAllocator<Space>& allocator,
              std::int64_t count)
        : bytes_(allocator,
                 count * static_cast<std::int64_t>(sizeof(T)))
    {}

    T* data() {
        return reinterpret_cast<T*>(bytes_.ptr());
    }

    const T* data() const {
        return reinterpret_cast<const T*>(bytes_.ptr());
    }

    // If you need to pass the underlying tensor somewhere:
    ByteTensor& bytes() { return bytes_; }
    const ByteTensor& bytes() const { return bytes_; }

private:
    ByteTensor bytes_;
};

template <typename T>
using HostRawBuffer   = RawBuffer<T, KERMAC_MEMORY_SPACE_HOST>;

template <typename T>
using DeviceRawBuffer = RawBuffer<T, KERMAC_MEMORY_SPACE_DEVICE>;

inline
void 
rng(
    Kermac& handle,
    DeviceTensor<float>& device_tensor,
    RNGType rng_type,
    float scale,
    float shift,
    CUstream stream
) {
    Result r = ::kermac_tensor_rng_f32(
        handle.get(),
        device_tensor.raw(),
        rng_type,
        scale,
        shift,
        stream
    );
    KERMAC_THROW_IF_ERROR(r);
}

inline
void 
rng(
    Kermac& handle,
    DeviceTensor<uint32_t>& device_tensor,
    CUstream stream
) {
    Result r = ::kermac_tensor_rng_u32(
        handle.get(),
        device_tensor.raw(),
        stream
    );
    KERMAC_THROW_IF_ERROR(r);
}

inline
void
solve(
    Kermac& handle,
    MatrixPackedType packed_type,
    DeviceStackAllocator& dsa,
    DeviceTensor<float>& a,
    DeviceTensor<float>& b,
    DeviceTensor<int32_t>& factor_info,
    DeviceTensor<int32_t>& solve_info,
    CUstream primary_stream
) {
    CUstream* secondary_streams = handle.secondary_streams_;
    size_t num_secondary_streams = handle.num_secondary_streams_;
    KERMAC_THROW_IF_ERROR(
        ::kermac_solve(
            handle.get(),
            packed_type,
            dsa.get_ptr(),
            a.raw(),
            b.raw(),
            factor_info.raw(),
            solve_info.raw(),
            primary_stream,
            secondary_streams,
            num_secondary_streams
        )
    );
}

inline
void
syev_batched(
    Kermac& handle,
    MatrixPackedType packed_type,
    DeviceStackAllocator& dsa,
    DeviceTensor<float>& a,
    DeviceTensor<float>& w,
    DeviceTensor<int32_t>& info,
    CUstream stream
) {
    KERMAC_THROW_IF_ERROR(
        ::kermac_syev_batched(
            handle.get(),
            packed_type,
            dsa.get_ptr(),
            a.raw(),
            w.raw(),
            info.raw(),
            stream
        )
    );
}

inline
void
permute(
    Kermac& handle,
    float alpha,
    DeviceTensor<float>& a, const char* modes_a,
    DeviceTensor<float>& b, const char* modes_b,
    CUstream stream
) {
    KERMAC_THROW_IF_ERROR(
        ::kermac_permute(
            handle.get(),
            alpha,
            a.raw(), modes_a,
            b.raw(), modes_b,
            stream
        )
    );
}

inline
void
contraction(
    Kermac& handle,
    DeviceStackAllocator& dsa,
    TensorCoreMode tcm,
    float alpha,
    DeviceTensor<float>& a, const char* modes_a,
    DeviceTensor<float>& b, const char* modes_b,
    float beta,
    DeviceTensor<float>& c, const char* modes_c,
    DeviceTensor<float>& d, const char* modes_d,
    CUstream stream
) {
    KERMAC_THROW_IF_ERROR(
        ::kermac_contraction(
            handle.get(),
            dsa.get_ptr(),
            tcm,
            alpha,
            a.raw(), modes_a,
            b.raw(), modes_b,
            beta,
            c.raw(), modes_c,
            d.raw(), modes_d,
            stream
        )
    );
}

inline
void 
contraction_trinary(
    Kermac& handle,
    DeviceStackAllocator& dsa,
    TensorCoreMode tcm,
    float alpha,
    DeviceTensor<float>& a, const char* modes_a,
    DeviceTensor<float>& b, const char* modes_b,
    DeviceTensor<float>& c, const char* modes_c,
    float beta,
    DeviceTensor<float>& d, const char* modes_d,
    DeviceTensor<float>& e, const char* modes_e,
    CUstream stream
) {
    KERMAC_THROW_IF_ERROR(
        ::kermac_contraction_trinary(
            handle.get(),
            dsa.get_ptr(),
            tcm,
            alpha,
            a.raw(), modes_a,
            b.raw(), modes_b,
            c.raw(), modes_c,
            beta,
            d.raw(), modes_d,
            e.raw(), modes_e,
            stream
        )
    );
}

inline
void
cutensor_gradient_norm_l2(
    Kermac& handle,
    DeviceStackAllocator& dsa,
    TensorCoreMode tcm,
    float bandwidth,
    DeviceTensor<float>& kernel_matrix,
    DeviceTensor<float>& data_n,
    DeviceTensor<float>& solution,
    DeviceTensor<float>& data_m,
    DeviceTensor<float>& gradient,
    CUstream stream
) {
    KERMAC_THROW_IF_ERROR(
        ::kermac_cutensor_gradient_norm_l2(
            handle.get(),
            dsa.get_ptr(),
            tcm,
            bandwidth,
            kernel_matrix.raw(),
            data_n.raw(),
            solution.raw(),
            data_m.raw(),
            gradient.raw(),
            stream
        )
    );
}

inline
void
mse(
    Kermac& handle,
    DeviceStackAllocator& dsa,
    DeviceTensor<float>& a,
    DeviceTensor<float>& b,
    DeviceTensor<float>& mse_tensor,
    CUstream stream
) {
    KERMAC_THROW_IF_ERROR(
        ::kermac_mse(
            handle.get(),
            dsa.get_ptr(),
            a.raw(),
            b.raw(),
            mse_tensor.raw(),
            stream
        )
    );
}

inline
void 
mse_accuracy(
    Kermac& handle,
    DeviceStackAllocator& dsa,
    DeviceTensor<float>& a,
    DeviceTensor<float>& b,
    DeviceTensor<float>& mse_tensor,
    DeviceTensor<float>& accuracy_tensor,
    CUstream stream
) {
    KERMAC_THROW_IF_ERROR(
        ::kermac_mse_accuracy(
            handle.get(),
            dsa.get_ptr(),
            a.raw(),
            b.raw(),
            mse_tensor.raw(),
            accuracy_tensor.raw(),
            stream
        )
    );
}

inline
void
logdet_norm_norm_h2(
    Kermac& handle,
    float lambda_reg,
    DeviceTensor<float>& factored_matrix,
    DeviceTensor<float>& alpha,
    DeviceTensor<float>& y,
    DeviceTensor<float>& logdet_norm,
    DeviceTensor<float>& norm_H2,
    CUstream stream
) {
    KERMAC_THROW_IF_ERROR(
        ::kermac_logdet_norm_norm_h2(
            handle.get(),
            lambda_reg,
            factored_matrix.raw(),
            alpha.raw(),
            y.raw(),
            logdet_norm.raw(),
            norm_H2.raw(),
            stream
        )
    );
}

inline
void
row_stats(
    Kermac& handle,
    DeviceTensor<float>& a,
    DeviceTensor<float>& mean,
    DeviceTensor<float>& stdev,
    CUstream stream
) {
    KERMAC_THROW_IF_ERROR(
        ::kermac_row_stats(
            handle.get(),
            a.raw(),
            mean.raw(),
            stdev.raw(),
            stream
        )
    );
}

inline
void
apply_row_stats(
    Kermac& handle,
    DeviceTensor<float>& a,
    DeviceTensor<float>& mean,
    DeviceTensor<float>& stdev,
    uint32_t flags,
    CUstream stream
) {
    KERMAC_THROW_IF_ERROR(
        ::kermac_apply_row_stats(
            handle.get(),
            a.raw(),
            mean.raw(),
            stdev.raw(),
            flags,
            stream
        )
    );
}

inline
void
stdev_rows(
    Kermac& handle,
    DeviceTensor<float>& a,
    DeviceTensor<float>& stdev,
    CUstream stream
) {
    KERMAC_THROW_IF_ERROR(
        ::kermac_stdev_rows(
            handle.get(),
            a.raw(),
            stdev.raw(),
            stream
        )
    );
}

inline
void
apply_stdev_rows(
    Kermac& handle,
    DeviceTensor<float>& a,
    DeviceTensor<float>& stdev,
    CUstream stream
) {
    KERMAC_THROW_IF_ERROR(
        ::kermac_apply_stdev_rows(
            handle.get(),
            a.raw(),
            stdev.raw(),
            stream
        )
    );
}

// Disabled legacy AGOP wrapper; superseded by Agop API.
#if 0
inline
void 
agop(
    Kermac& handle,
    TensorCoreMode tcm,
    DeviceStackAllocator& dsa,
    DeviceTensor<float>& kernel_matrix_grad,
    DeviceTensor<float>& solution,
    DeviceTensor<float>& x,
    bool x_is_transposed,
    DeviceTensor<float>& feature_matrix,
    float bandwidth,
    CUstream stream
) {
    KERMAC_THROW_IF_ERROR(
        ::kermac_agop(
            handle.get(),
            tcm,
            dsa.get_ptr(),
            kernel_matrix_grad.raw(),
            solution.raw(),
            x.raw(),
            x_is_transposed,
            feature_matrix.raw(),
            bandwidth,
            stream
        )
    );
}
#endif

class Semiring {
public:
    Semiring() = delete;

    // Movable, non-copyable RAII wrapper
    Semiring(
        Semiring&& other
    ) noexcept
        : semiring_(other.semiring_)
        , valid_(other.valid_)
    {
        other.valid_    = false;
        other.semiring_ = SemiringRaw{};
    }

    Semiring(const Semiring&)            = delete;
    Semiring& operator=(const Semiring&) = delete;

    ~Semiring() noexcept {
        destroy();
    }

    static Semiring gemm(
        Kermac& handle,
        HostStackAllocator& hsa
    ) {
        SemiringRaw raw{};
        KERMAC_THROW_IF_ERROR(
            ::kermac_semiring_gemm_create(
                handle.get(), 
                hsa.get_ptr(), 
                &raw
            )
        );
        return Semiring(raw);
    }

    static Semiring norm_l2(
        Kermac& handle,
        HostStackAllocator& hsa
    ) {
        SemiringRaw raw{};
        KERMAC_THROW_IF_ERROR(
            ::kermac_semiring_norm_l2_create(
                handle.get(), 
                hsa.get_ptr(), 
                &raw
            )
        );
        return Semiring(raw);
    }

    static Semiring norm_l2_symm(
        Kermac& handle,
        HostStackAllocator& hsa,
        MatrixPackedType packed_type
    ) {
        SemiringRaw raw{};
        KERMAC_THROW_IF_ERROR(
            ::kermac_semiring_norm_l2_symm_create(
                handle.get(), 
                hsa.get_ptr(), 
                &raw,
                packed_type
            )
        );
        return Semiring(raw);
    }

    static Semiring laplace_l2(
        Kermac& handle,
        HostStackAllocator& hsa,
        float bandwidth
    ) {
        SemiringRaw raw{};
        KERMAC_THROW_IF_ERROR(
            ::kermac_semiring_laplace_l2_create(
                handle.get(), 
                hsa.get_ptr(), 
                &raw,
                bandwidth
            )
        );
        return Semiring(raw);
    }

    static Semiring laplace_l2_symm(
        Kermac& handle, 
        HostStackAllocator& hsa,
        MatrixPackedType packed_type,
        float bandwidth,
        float regularizer,
        float epsilon
    ) {
        SemiringRaw raw{};
        KERMAC_THROW_IF_ERROR(
            ::kermac_semiring_laplace_l2_symm_create(
                handle.get(), 
                hsa.get_ptr(),
                &raw,
                packed_type,
                bandwidth,
                regularizer,
                epsilon
            )
        );
        return Semiring(raw);
    }

    static Semiring laplace_l2_symm_copy_grad(
        Kermac& handle,
        HostStackAllocator& hsa,
        MatrixPackedType packed_type,
        float bandwidth,
        float regularizer,
        float epsilon
    ) {
        SemiringRaw raw{};
        KERMAC_THROW_IF_ERROR(
            ::kermac_semiring_laplace_l2_symm_copy_grad_create(
                handle.get(),
                hsa.get_ptr(),
                &raw,
                packed_type,
                bandwidth,
                regularizer,
                epsilon
            )
        );
        return Semiring(raw);
    }

    void run(
        DeviceTensor<float>& a,
        DeviceTensor<float>& b,
        DeviceTensor<float>& c,
        CUstream stream
    ) {
        KERMAC_THROW_IF_ERROR(
            ::kermac_semiring_run(
                semiring_,
                a.raw(),
                b.raw(),
                c.raw(),
                stream
            )
        );
    }

    SemiringRaw raw() const noexcept { return semiring_; }

private:
    explicit Semiring(
        SemiringRaw raw
    ) noexcept
        : semiring_(raw)
        , valid_(true)
    {}

    void destroy() noexcept {
        if (valid_) {
            (void)::kermac_semiring_destroy(semiring_);
            valid_ = false;
        }
    }

    SemiringRaw semiring_{};
    bool valid_ = false;
};

class SemiringGradient {
public:
    SemiringGradient() = delete;

    SemiringGradient(
        SemiringGradient&& other
    ) noexcept
        : semiring_(other.semiring_)
        , valid_(other.valid_)
    {
        other.valid_ = false;
        other.semiring_ = SemiringGradientRaw{};
    }

    SemiringGradient(const SemiringGradient&)            = delete;
    SemiringGradient& operator=(const SemiringGradient&) = delete;

    ~SemiringGradient() noexcept {
        destroy();
    }

    static SemiringGradient norm_l2(
        Kermac& handle,
        HostStackAllocator& hsa
    ) {
        SemiringGradientRaw raw{};
        KERMAC_THROW_IF_ERROR(
            ::kermac_semiring_gradient_norm_l2_create(
                handle.get(),
                hsa.get_ptr(),
                &raw
            )
        );
        return SemiringGradient(raw);
    }

    void run(
        DeviceTensor<float>& kernel_matrix,
        DeviceTensor<float>& data_n,
        DeviceTensor<float>& solution,
        DeviceTensor<float>& data_m,
        DeviceTensor<float>& gradient,
        float alpha,
        float beta,
        CUstream stream
    ) {
        KERMAC_THROW_IF_ERROR(
            ::kermac_semiring_gradient_run(
                semiring_,
                kernel_matrix.raw(),
                data_n.raw(),
                solution.raw(),
                data_m.raw(),
                gradient.raw(),
                alpha,
                beta,
                stream
            )
        );
    }

    SemiringGradientRaw raw() const noexcept { return semiring_; }

private:
    explicit SemiringGradient(
        SemiringGradientRaw raw
    ) noexcept
        : semiring_(raw)
        , valid_(true)
    {}

    void destroy() noexcept {
        if (valid_) {
            (void)::kermac_semiring_gradient_destroy(semiring_);
            valid_ = false;
        }
    }

    SemiringGradientRaw semiring_{};
    bool valid_ = false;
};

class Agop {
public:
    Agop() = delete;

    Agop(
        Agop&& other
    ) noexcept
        : agop_(other.agop_)
        , valid_(other.valid_)
    {
        other.valid_ = false;
        other.agop_ = AgopRaw{};
    }

    Agop(const Agop&)            = delete;
    Agop& operator=(const Agop&) = delete;

    ~Agop() noexcept {
        destroy();
    }

    static Agop create(
        Kermac& handle,
        HostStackAllocator& hsa
    ) {
        AgopRaw raw{};
        KERMAC_THROW_IF_ERROR(
            ::kermac_agop_create(
                handle.get(),
                hsa.get_ptr(),
                &raw
            )
        );
        return Agop(raw);
    }

    void run(
        Kermac& handle,
        DeviceStackAllocator& dsa,
        AgopBackend backend,
        AgopOutput output,
        float bandwidth,
        DeviceTensor<float>& kernel_matrix,
        DeviceTensor<float>& data_n,
        DeviceTensor<float>& solution,
        DeviceTensor<float>& data_m,
        DeviceTensor<float>& feature_matrix,
        CUstream stream
    ) {
        KERMAC_THROW_IF_ERROR(
            ::kermac_agop_run(
                handle.get(),
                dsa.get_ptr(),
                agop_,
                backend,
                output,
                bandwidth,
                kernel_matrix.raw(),
                data_n.raw(),
                solution.raw(),
                data_m.raw(),
                feature_matrix.raw(),
                stream
            )
        );
    }

    AgopRaw raw() const noexcept { return agop_; }

private:
    explicit Agop(
        AgopRaw raw
    ) noexcept
        : agop_(raw)
        , valid_(true)
    {}

    void destroy() noexcept {
        if (valid_) {
            (void)::kermac_agop_destroy(agop_);
            valid_ = false;
        }
    }

    AgopRaw agop_{};
    bool valid_ = false;
};

class Elementwise {
public:
    Elementwise() = delete;

    Elementwise(
        Elementwise&& other
    ) noexcept
        : elementwise_(other.elementwise_)
        , valid_(other.valid_)
    {
        other.valid_      = false;
        other.elementwise_ = ElementwiseRaw{};
    }

    Elementwise(const Elementwise&)            = delete;
    Elementwise& operator=(const Elementwise&) = delete;

    ~Elementwise() noexcept {
        destroy();
    }

    static Elementwise laplace(
        Kermac& handle,
        HostStackAllocator& hsa,
        float bandwidth
    ) {
        ElementwiseRaw raw{};
        KERMAC_THROW_IF_ERROR(
            ::kermac_elementwise_laplace_create(
                handle.get(), 
                hsa.get_ptr(),
                &raw,
                bandwidth
            )
        );
        return Elementwise(raw);
    }

    static Elementwise laplace_symm(
        Kermac& handle,
        HostStackAllocator& hsa,
        MatrixPackedType packed_type,
        float bandwidth,
        float regularizer,
        float epsilon
    ) {
        ElementwiseRaw raw{};
        KERMAC_THROW_IF_ERROR(
            ::kermac_elementwise_laplace_symm_create(
                handle.get(), 
                hsa.get_ptr(),
                &raw,
                packed_type,
                bandwidth,
                regularizer,
                epsilon
            )
        );
        return Elementwise(raw);
    }

    void run(
        DeviceTensor<float>& a,
        DeviceTensor<float>& c,
        CUstream stream
    ) {
        KERMAC_THROW_IF_ERROR(
            ::kermac_elementwise_run(
                elementwise_,
                a.raw(),
                c.raw(),
                stream
            )
        );
    }

    ElementwiseRaw raw() const noexcept { return elementwise_; }

private:
    explicit Elementwise(
        ElementwiseRaw raw
    ) noexcept
        : elementwise_(raw)
        , valid_(true)
    {}

    void destroy() noexcept {
        if (valid_) {
            (void)::kermac_elementwise_destroy(elementwise_);
            valid_ = false;
        }
    }

    ElementwiseRaw elementwise_{};
    bool           valid_ = false;
};

class LegacyHandle {
public:
    /// Construct a legacy handle bound to an existing core Handle.
    explicit LegacyHandle(
        Kermac& core
    ) {
        KermacResult result = ::kermac_legacy_create(core.get(), &legacy_);
        KERMAC_THROW_IF_ERROR(result);
        owns_ = true;
    }

    /// You can also construct from a raw KermacHandle if needed.
    explicit LegacyHandle(KermacHandle core_raw) {
        KermacResult result = ::kermac_legacy_create(core_raw, &legacy_);
        KERMAC_THROW_IF_ERROR(result);
        owns_ = true;
    }

    ~LegacyHandle() {
        destroy_if_owned();
    }

    LegacyHandle(const LegacyHandle&) = delete;
    LegacyHandle& operator=(const LegacyHandle&) = delete;

    LegacyHandle(LegacyHandle&& other) noexcept
        : legacy_(other.legacy_), owns_(other.owns_) {
        other.owns_ = false;
    }

    LegacyHandle& operator=(LegacyHandle&& other) noexcept {
        if (this != &other) {
            destroy_if_owned();
            legacy_ = other.legacy_;
            owns_   = other.owns_;
            other.owns_ = false;
        }
        return *this;
    }

    /// Access the underlying C handle (for calling C APIs).
    KermacLegacyHandle& raw() noexcept { return legacy_; }
    const KermacLegacyHandle& raw() const noexcept { return legacy_; }

private:
    void destroy_if_owned() noexcept {
        if (owns_) {
            // We cannot throw from a destructor; ignore the result.
            (void)::kermac_legacy_destroy(legacy_);
            owns_ = false;
        }
    }

    KermacLegacyHandle legacy_{};  // owns CUmodule / CUfunctions
    bool owns_ = false;
};

inline
void 
copy_diagonal(
    LegacyHandle& legacy,
    DeviceTensor<float>& symmetric_tensor,
    DeviceTensor<float>& diagonal,
    CUstream stream
) {
    KERMAC_THROW_IF_ERROR(
        ::kermac_legacy_copy_diagonal(
            legacy.raw(),
            symmetric_tensor.raw(),
            diagonal.raw(),
            stream
        )
    );
}

inline 
void 
copy_triangle(
    LegacyHandle& legacy,
    MatrixPackedType packed_type,
    DeviceTensor<float>& symmetric_tensor,
    float diagonal_value,
    CUstream stream
) {
    KERMAC_THROW_IF_ERROR(
        ::kermac_legacy_copy_triangle(
            legacy.raw(),
            packed_type,
            symmetric_tensor.raw(),
            diagonal_value,
            stream
        )
    );
}

inline
void 
laplace(
    LegacyHandle& legacy,
    DeviceTensor<float>& tensor,
    DeviceTensor<float>& norm_x,
    DeviceTensor<float>& norm_y,
    float bandwidth,
    float epsilon,
    CUstream stream
) {
    KERMAC_THROW_IF_ERROR(
        ::kermac_legacy_laplace(
            legacy.raw(),
            tensor.raw(),
            norm_x.raw(),
            norm_y.raw(),
            bandwidth,
            epsilon,
            stream
        )
    );
}

inline
void
laplace_symmetric(
    LegacyHandle& legacy,
    MatrixPackedType packed_type,
    DeviceStackAllocator& dsa,
    DeviceTensor<float>& tensor,
    float bandwidth,
    float regularization,
    float epsilon,
    CUstream stream
) {
    KERMAC_THROW_IF_ERROR(
        ::kermac_legacy_laplace_symmetric(
            legacy.raw(),
            packed_type,
            dsa.get_ptr(),
            tensor.raw(),
            bandwidth,
            regularization,
            epsilon,
            stream
        )
    );
}

inline
void
solve_compute_array(
    LegacyHandle& legacy,
    DeviceTensor<float>& a,
    DeviceTensor<float*>& a_array,
    CUstream stream
) {
    KERMAC_THROW_IF_ERROR(
        ::kermac_legacy_solve_compute_array(
            legacy.raw(),
            a.raw(),
            a_array.raw(),
            stream
        )
    );
}

inline
void
solve(
    Kermac& kermac,
    MatrixPackedType packed_type,
    DeviceTensor<float>& a,
    DeviceTensor<float>& b,
    DeviceTensor<float*>& a_array,
    DeviceTensor<float*>& b_array,
    DeviceTensor<int>& factor_info,
    DeviceTensor<int>& solve_info,
    CUstream stream
) {
    KERMAC_THROW_IF_ERROR(
        ::kermac_legacy_solve(
            kermac.get(),
            packed_type,
            a.raw(),
            b.raw(),
            a_array.raw(),
            b_array.raw(),
            factor_info.raw(),
            solve_info.raw(),
            stream
        )
    );
}

inline
void
stack_ptx_inject_compile(
    Kermac& kermac,
    HostStackAllocator& hsa,
    PtxInjectHandle ptx_inject,
    const StackPtxCompilerInfo* compiler_info,
    const StackPtxStackInfo* stack_info,
    size_t execution_limit,
    const StackPtxRegister* registers,
    size_t num_registers,
    const StackPtxInstruction* const* stack_ptx_instruction_stubs,
    const size_t** request_stubs,
    const size_t* request_stub_sizes,
    size_t num_stubs,
    CUmodule* module
) {
    KERMAC_THROW_IF_ERROR(
        ::kermac_stack_ptx_inject_compile(
            kermac.get(),
            hsa.get_ptr(),
            ptx_inject,
            compiler_info,
            stack_info,
            execution_limit,
            registers,
            num_registers,
            stack_ptx_instruction_stubs,
            request_stubs,
            request_stub_sizes,
            num_stubs,
            module
        )
    );
}

inline
void
stack_ptx_inject_compile_with_routines(
    Kermac& kermac,
    HostStackAllocator& hsa,
    PtxInjectHandle ptx_inject,
    const StackPtxCompilerInfo* compiler_info,
    const StackPtxStackInfo* stack_info,
    size_t execution_limit,
    const StackPtxRegister* registers,
    size_t num_registers,
    const StackPtxInstruction* const* stack_ptx_instruction_stubs,
    const StackPtxInstruction* const* routines,
    size_t num_routines,
    const size_t** request_stubs,
    const size_t* request_stub_sizes,
    size_t num_stubs,
    CUmodule* module
) {
    KERMAC_THROW_IF_ERROR(
        ::kermac_stack_ptx_inject_compile_with_routines(
            kermac.get(),
            hsa.get_ptr(),
            ptx_inject,
            compiler_info,
            stack_info,
            execution_limit,
            registers,
            num_registers,
            stack_ptx_instruction_stubs,
            routines,
            num_routines,
            request_stubs,
            request_stub_sizes,
            num_stubs,
            module
        )
    );
}

inline
void
tensor_broadcast_copy(
    Kermac& kermac, 
    DeviceTensor<float>& src, 
    DeviceTensor<float>& dst, 
    CUstream stream
) {
    KERMAC_THROW_IF_ERROR(
        ::kermac_tensor_broadcast_copy(
            kermac.get(),
            src.raw(),
            dst.raw(),
            stream
        )
    );
}

class Compiler {
public:
    Compiler() = delete;

    Compiler(
        Kermac& handle,
        HostStackAllocator& hsa,
        PtxInjectHandle ptx_inject,
        const StackPtxCompilerInfo* compiler_info,
        const StackPtxStackInfo* stack_info,
        size_t execution_limit,
        const StackPtxRegister* registers,
        size_t num_registers,
        const StackPtxInstruction* const* stack_ptx_instruction_stubs,
        const size_t** request_stubs,
        const size_t* request_stub_sizes,
        size_t num_stubs,
        bool verbose = false
    ) {
        CompilerRaw raw{};
        KERMAC_THROW_IF_ERROR(
            ::kermac_compiler_create(
                &raw,
                handle.get(),
                hsa.get_ptr(),
                ptx_inject,
                compiler_info,
                stack_info,
                execution_limit,
                registers,
                num_registers,
                stack_ptx_instruction_stubs,
                request_stubs,
                request_stub_sizes,
                num_stubs,
                verbose
            )
        );
        compiler_ = raw;
        valid_ = true;
    }

    ~Compiler() noexcept {
        destroy();
    }

    Compiler(const Compiler&)            = delete;
    Compiler& operator=(const Compiler&) = delete;

    CompilerRaw raw() const noexcept { return compiler_; }

    size_t cubin(
        void*  buffer,
        size_t buffer_size
    ) {
        size_t written = 0;
        KERMAC_THROW_IF_ERROR(
            ::kermac_compiler_cubin(
                compiler_,
                buffer,
                buffer_size,
                &written
            )
        );
        return written;
    }

private:
    void destroy() noexcept {
        if (valid_) {
            (void)::kermac_compiler_destroy(compiler_);
            valid_ = false;
            compiler_ = CompilerRaw{};
        }
    }

    CompilerRaw compiler_{};
    bool        valid_ = false;
};

class NvrtcCompiler {
public:
    NvrtcCompiler() = delete;

    NvrtcCompiler(
        Kermac& handle,
        HostStackAllocator& hsa,
        const char* program_name,
        const char* program_source,
        const char* const* headers,
        const char* const* include_names,
        size_t num_headers,
        const char* const* options,
        size_t num_options,
        bool verbose = false
    ) {
        NvrtcCompilerRaw raw{};
        KERMAC_THROW_IF_ERROR(
            ::kermac_nvrtc_compiler_create(
                &raw,
                handle.get(),
                hsa.get_ptr(),
                program_name,
                program_source,
                headers,
                include_names,
                num_headers,
                options,
                num_options,
                verbose
            )
        );
        compiler_ = raw;
        valid_ = true;
    }

    ~NvrtcCompiler() noexcept {
        destroy();
    }

    NvrtcCompiler(const NvrtcCompiler&)            = delete;
    NvrtcCompiler& operator=(const NvrtcCompiler&) = delete;

    NvrtcCompilerRaw raw() const noexcept { return compiler_; }

    size_t ptx(
        void* buffer,
        size_t buffer_size
    ) {
        size_t written = 0;
        KERMAC_THROW_IF_ERROR(
            ::kermac_nvrtc_compiler_ptx(
                compiler_,
                buffer,
                buffer_size,
                &written
            )
        );
        return written;
    }

private:
    void destroy() noexcept {
        if (valid_) {
            (void)::kermac_nvrtc_compiler_destroy(compiler_);
            valid_ = false;
            compiler_ = NvrtcCompilerRaw{};
        }
    }

    NvrtcCompilerRaw compiler_{};
    bool             valid_ = false;
};


} // namespace kermac
