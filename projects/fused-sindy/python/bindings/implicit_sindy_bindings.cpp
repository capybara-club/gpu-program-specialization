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
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "implicit_sindy.h"

#include <cuda.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace {

enum {
    ISPY_AST_UNARY_OPS = BINARY_AST_NUM_UNARY_OPS,
    ISPY_AST_BINARY_OPS = BINARY_AST_NUM_BINARY_OPS,
    ISPY_FEATURES_PER_KERNEL = 32,
    ISPY_MAX_KERNELS_PER_MODULE = 64
};

void
check_cuda(
    CUresult result,
    const char* label
) {
    const char* name = nullptr;
    const char* text = nullptr;
    std::string message;
    if (result == CUDA_SUCCESS) return;
    (void)cuGetErrorName(result, &name);
    (void)cuGetErrorString(result, &text);
    message = label != nullptr ? label : "CUDA Driver API";
    message += " failed";
    if (name != nullptr) {
        message += ": ";
        message += name;
    }
    if (text != nullptr) {
        message += " (";
        message += text;
        message += ")";
    }
    throw std::runtime_error(message);
}

void
check_result(
    ImplicitSindyResult result
) {
    if (result != IMPLICIT_SINDY_SUCCESS) {
        throw std::runtime_error(implicit_sindy_result_to_string(result));
    }
}

void
check_solve_result(
    ImplicitFeatureRidgeSolveResult result
) {
    if (result != IMPLICIT_FEATURE_RIDGE_SOLVE_SUCCESS) {
        throw std::runtime_error(implicit_feature_ridge_solve_result_to_string(result));
    }
}

size_t
kernels_per_module_count(
    int kernels_per_module
) {
    if (kernels_per_module < 1 || kernels_per_module > ISPY_MAX_KERNELS_PER_MODULE) {
        throw std::invalid_argument("kernels_per_module must be in [1, 64]");
    }
    return (size_t)kernels_per_module;
}

int32_t
read_i32(
    const std::string& bytes,
    size_t idx
) {
    int32_t value = 0;
    std::memcpy(&value, bytes.data() + idx * sizeof(value), sizeof(value));
    return value;
}

std::vector<BinaryAST>
decode_asts(
    const std::string& unary_bytes,
    const std::string& binary_bytes,
    size_t num_asts
) {
    std::vector<BinaryAST> asts;
    size_t i;
    size_t j;

    if (num_asts == 0u) {
        throw std::invalid_argument("num_asts must be positive");
    }
    if (unary_bytes.size() != num_asts * ISPY_AST_UNARY_OPS * sizeof(int32_t)) {
        throw std::invalid_argument("unary_bytes has the wrong size");
    }
    if (binary_bytes.size() != num_asts * ISPY_AST_BINARY_OPS * sizeof(int32_t)) {
        throw std::invalid_argument("binary_bytes has the wrong size");
    }

    asts.resize(num_asts);
    for (i = 0u; i < num_asts; ++i) {
        for (j = 0u; j < ISPY_AST_UNARY_OPS; ++j) {
            int32_t value = read_i32(unary_bytes, i * ISPY_AST_UNARY_OPS + j);
            if (value < 0 || value >= BINARY_AST_UNARY_NUM_ENUMS) {
                throw std::invalid_argument("unary op is out of range");
            }
            asts[i].unary[j] = (BinaryAstUnaryOp)value;
        }
        for (j = 0u; j < ISPY_AST_BINARY_OPS; ++j) {
            int32_t value = read_i32(binary_bytes, i * ISPY_AST_BINARY_OPS + j);
            if (value < 0 || value >= BINARY_AST_BINARY_NUM_ENUMS) {
                throw std::invalid_argument("binary op is out of range");
            }
            asts[i].binary[j] = (BinaryAstBinaryOp)value;
        }
    }
    return asts;
}

std::string
bytes_to_string(
    nb::bytes value,
    const char* label
) {
    char* data = nullptr;
    Py_ssize_t size = 0;
    if (PyBytes_AsStringAndSize(value.ptr(), &data, &size) != 0 || data == nullptr || size < 0) {
        throw std::invalid_argument(label != nullptr ? label : "bytes argument is invalid");
    }
    return std::string(data, (size_t)size);
}

CUcontext
current_context()
{
    CUcontext context = nullptr;
    check_cuda(cuInit(0), "cuInit");
    check_cuda(cuCtxGetCurrent(&context), "cuCtxGetCurrent");
    if (context == nullptr) {
        throw std::runtime_error("no current CUDA context; initialize torch.cuda before calling FusedSINDy");
    }
    return context;
}

class PushedContext {
public:
    explicit PushedContext(
        CUcontext context
    ) {
        if (context == nullptr) {
            throw std::runtime_error("cannot push a null CUDA context");
        }
        check_cuda(cuCtxPushCurrent(context), "cuCtxPushCurrent");
        pushed_ = true;
    }

    PushedContext(const PushedContext&) = delete;
    PushedContext& operator=(const PushedContext&) = delete;

    ~PushedContext() {
        if (pushed_) {
            CUcontext popped = nullptr;
            (void)cuCtxPopCurrent(&popped);
        }
    }

private:
    bool pushed_ = false;
};

void
cleanup_gram_modules(
    std::vector<ImplicitSindyGramModule*>& gram_modules
) {
    size_t i;
    for (i = 0u; i < gram_modules.size(); ++i) {
        if (gram_modules[i] != nullptr) {
            (void)implicit_sindy_gram_module_destroy(gram_modules[i]);
            gram_modules[i] = nullptr;
        }
    }
}

void
cleanup_columns_modules(
    std::vector<ImplicitSindyColumnsModule*>& columns_modules
) {
    size_t i;
    for (i = 0u; i < columns_modules.size(); ++i) {
        if (columns_modules[i] != nullptr) {
            (void)implicit_sindy_columns_module_destroy(columns_modules[i]);
            columns_modules[i] = nullptr;
        }
    }
}

void
cleanup_single_ast_column_modules(
    std::vector<ImplicitSindyAstColumnModule*>& column_modules
) {
    size_t i;
    for (i = 0u; i < column_modules.size(); ++i) {
        if (column_modules[i] != nullptr) {
            (void)implicit_sindy_ast_column_module_destroy(column_modules[i]);
            column_modules[i] = nullptr;
        }
    }
}

class AstCompilerGuard {
public:
    explicit AstCompilerGuard(
        ImplicitSindyAstCompiler* compiler
    ) : compiler_(compiler) {}

    AstCompilerGuard(const AstCompilerGuard&) = delete;
    AstCompilerGuard& operator=(const AstCompilerGuard&) = delete;

    ~AstCompilerGuard() {
        if (compiler_ != nullptr) {
            (void)implicit_sindy_ast_compiler_destroy(compiler_);
        }
    }

    ImplicitSindyAstCompiler*
    get() const {
        return compiler_;
    }

private:
    ImplicitSindyAstCompiler* compiler_ = nullptr;
};

class AstColumnCompilerGuard {
public:
    explicit AstColumnCompilerGuard(
        ImplicitSindyAstColumnCompiler* compiler
    ) : compiler_(compiler) {}

    AstColumnCompilerGuard(const AstColumnCompilerGuard&) = delete;
    AstColumnCompilerGuard& operator=(const AstColumnCompilerGuard&) = delete;

    ~AstColumnCompilerGuard() {
        if (compiler_ != nullptr) {
            (void)implicit_sindy_ast_column_compiler_destroy(compiler_);
        }
    }

    ImplicitSindyAstColumnCompiler*
    get() const {
        return compiler_;
    }

private:
    ImplicitSindyAstColumnCompiler* compiler_ = nullptr;
};

class GramKernelSet {
public:
    GramKernelSet(
        nb::bytes unary_bytes_obj,
        nb::bytes binary_bytes_obj,
        nb::bytes gram_template_ptx_obj,
        size_t num_asts,
        int kernels_per_module,
        size_t worker_count,
        size_t scratch_bytes_per_worker,
        const std::vector<std::string>& nvptx_options
    ) {
        std::string unary_bytes = bytes_to_string(unary_bytes_obj, "unary_bytes");
        std::string binary_bytes = bytes_to_string(binary_bytes_obj, "binary_bytes");
        std::string gram_template_ptx = bytes_to_string(gram_template_ptx_obj, "gram_template_ptx");
        std::vector<BinaryAST> asts = decode_asts(unary_bytes, binary_bytes, num_asts);
        std::vector<const char*> option_ptrs;
        std::vector<void*> cubins;
        std::vector<size_t> cubin_sizes;
        std::vector<void*> modules;
        std::vector<ImplicitSindyGramModule*> gram_modules;
        std::vector<unsigned char> workspace;
        ImplicitSindyAstCompiler* compiler = nullptr;
        CUdevice device = 0;
        int sm_major = 0;
        int sm_minor = 0;
        size_t cubin_count = 0u;
        size_t workspace_bytes = 0u;
        size_t i;

        kernels_per_module_ = kernels_per_module_count(kernels_per_module);
        num_asts_ = num_asts;
        num_cohorts_ = (num_asts + ISPY_FEATURES_PER_KERNEL - 1u) / ISPY_FEATURES_PER_KERNEL;
        context_ = current_context();
        check_cuda(cuCtxGetDevice(&device), "cuCtxGetDevice");
        check_cuda(cuDeviceGetAttribute(&sm_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device), "cuDeviceGetAttribute major");
        check_cuda(cuDeviceGetAttribute(&sm_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device), "cuDeviceGetAttribute minor");

        option_ptrs.reserve(nvptx_options.size());
        for (i = 0u; i < nvptx_options.size(); ++i) {
            option_ptrs.push_back(nvptx_options[i].c_str());
        }

        {
            nb::gil_scoped_release release;
            check_result(implicit_sindy_ast_compiler_create_with_gram_ptx(
                kernels_per_module_,
                (unsigned int)sm_major,
                (unsigned int)sm_minor,
                gram_template_ptx.data(),
                gram_template_ptx.size(),
                option_ptrs.empty() ? nullptr : option_ptrs.data(),
                option_ptrs.size(),
                &compiler
            ));
            AstCompilerGuard compiler_guard(compiler);

            cubin_count = implicit_sindy_ast_cubin_count(num_asts, kernels_per_module_);
            if (cubin_count == 0u) {
                throw std::runtime_error("implicit_sindy_ast_cubin_count returned zero");
            }
            cubins.assign(cubin_count, nullptr);
            cubin_sizes.assign(cubin_count, 0u);
            modules.assign(cubin_count, nullptr);
            gram_modules.assign(cubin_count, nullptr);

            check_result(implicit_sindy_ast_compile_workspace_size(
                compiler_guard.get(),
                num_asts,
                worker_count,
                scratch_bytes_per_worker,
                &workspace_bytes
            ));
            workspace.resize(workspace_bytes);

            try {
                check_result(implicit_sindy_ast_compile_cubins(
                    compiler_guard.get(),
                    asts.data(),
                    asts.size(),
                    sizeof(asts[0]),
                    worker_count,
                    workspace.data(),
                    workspace.size(),
                    cubins.data(),
                    cubin_sizes.data()
                ));
                check_result(implicit_sindy_load_cubin_modules(
                    cubins.data(),
                    cubin_sizes.data(),
                    cubin_count,
                    modules.data()
                ));
                for (i = 0u; i < cubin_count; ++i) {
                    check_result(implicit_sindy_gram_module_create(
                        modules[i],
                        kernels_per_module_,
                        &gram_modules[i]
                    ));
                }
            } catch (...) {
                cleanup_gram_modules(gram_modules);
                if (!modules.empty()) {
                    (void)implicit_sindy_unload_modules(modules.data(), modules.size());
                }
                if (!cubins.empty()) {
                    implicit_sindy_free_cubins(cubins.data(), cubin_sizes.data(), cubins.size());
                }
                throw;
            }
        }

        cubins_ = std::move(cubins);
        cubin_sizes_ = std::move(cubin_sizes);
        modules_ = std::move(modules);
        gram_modules_ = std::move(gram_modules);
    }

    GramKernelSet(const GramKernelSet&) = delete;
    GramKernelSet& operator=(const GramKernelSet&) = delete;

    ~GramKernelSet() {
        try {
            close();
        } catch (...) {
        }
    }

    void
    close() {
        ImplicitSindyResult unload_result = IMPLICIT_SINDY_SUCCESS;

        if (closed_) return;
        {
            PushedContext pushed(context_);
            cleanup_gram_modules(gram_modules_);
            if (!modules_.empty()) {
                unload_result = implicit_sindy_unload_modules(modules_.data(), modules_.size());
            }
        }
        if (!cubins_.empty()) {
            implicit_sindy_free_cubins(cubins_.data(), cubin_sizes_.data(), cubins_.size());
        }
        gram_modules_.clear();
        modules_.clear();
        cubins_.clear();
        cubin_sizes_.clear();
        closed_ = true;
        check_result(unload_result);
    }

    size_t
    num_cubins() const {
        return modules_.size();
    }

    size_t
    num_cohorts() const {
        return num_cohorts_;
    }

    size_t
    kernels_per_module() const {
        return kernels_per_module_;
    }

    void
    launch_raw(
        size_t cohort_index,
        uint64_t stream,
        int64_t num_settings,
        uint64_t primitive_features,
        int64_t row_count,
        int64_t primitive_feature_stride,
        int64_t num_primitive_features,
        uint64_t targets,
        int64_t target_rhs_stride,
        int64_t num_target_rhs,
        uint64_t leaf_masks,
        uint64_t leaf_words,
        int64_t leaf_words_feature_stride,
        uint64_t gram,
        int64_t gram_col_stride,
        uint64_t x_sum,
        uint64_t xty,
        int64_t xty_rhs_stride,
        uint64_t y_sum,
        uint64_t yy
    ) {
        CUcontext context = current_context();
        size_t module_index;
        size_t kernel_index;

        if (closed_) {
            throw std::runtime_error("GramKernelSet is closed");
        }
        if (context != context_) {
            throw std::runtime_error("current CUDA context does not match the context used to compile the gram kernels");
        }
        if (cohort_index >= num_cohorts_) {
            throw std::out_of_range("cohort_index is out of range");
        }

        module_index = cohort_index / kernels_per_module_;
        kernel_index = cohort_index % kernels_per_module_;
        if (module_index >= gram_modules_.size()) {
            throw std::out_of_range("module index is out of range");
        }

        nb::gil_scoped_release release;
        check_result(implicit_sindy_gram_launch(
            gram_modules_[module_index],
            kernel_index,
            reinterpret_cast<void*>(static_cast<uintptr_t>(stream)),
            num_settings,
            reinterpret_cast<const float*>(static_cast<uintptr_t>(primitive_features)),
            row_count,
            primitive_feature_stride,
            num_primitive_features,
            reinterpret_cast<const float*>(static_cast<uintptr_t>(targets)),
            target_rhs_stride,
            num_target_rhs,
            reinterpret_cast<const int32_t*>(static_cast<uintptr_t>(leaf_masks)),
            reinterpret_cast<const int32_t*>(static_cast<uintptr_t>(leaf_words)),
            leaf_words_feature_stride,
            reinterpret_cast<float*>(static_cast<uintptr_t>(gram)),
            gram_col_stride,
            reinterpret_cast<float*>(static_cast<uintptr_t>(x_sum)),
            reinterpret_cast<float*>(static_cast<uintptr_t>(xty)),
            xty_rhs_stride,
            reinterpret_cast<float*>(static_cast<uintptr_t>(y_sum)),
            reinterpret_cast<float*>(static_cast<uintptr_t>(yy))
        ));
    }

private:
    CUcontext context_ = nullptr;
    bool closed_ = false;
    size_t kernels_per_module_ = 0u;
    size_t num_asts_ = 0u;
    size_t num_cohorts_ = 0u;
    std::vector<void*> cubins_;
    std::vector<size_t> cubin_sizes_;
    std::vector<void*> modules_;
    std::vector<ImplicitSindyGramModule*> gram_modules_;
};

class ColumnsKernelSet {
public:
    ColumnsKernelSet(
        nb::bytes unary_bytes_obj,
        nb::bytes binary_bytes_obj,
        size_t num_asts,
        int kernels_per_module,
        size_t worker_count,
        size_t scratch_bytes_per_worker,
        const std::vector<std::string>& nvptx_options
    ) {
        std::string unary_bytes = bytes_to_string(unary_bytes_obj, "unary_bytes");
        std::string binary_bytes = bytes_to_string(binary_bytes_obj, "binary_bytes");
        std::vector<BinaryAST> asts = decode_asts(unary_bytes, binary_bytes, num_asts);
        std::vector<const char*> option_ptrs;
        std::vector<void*> cubins;
        std::vector<size_t> cubin_sizes;
        std::vector<void*> modules;
        std::vector<ImplicitSindyColumnsModule*> columns_modules;
        std::vector<unsigned char> workspace;
        ImplicitSindyAstCompiler* compiler = nullptr;
        CUdevice device = 0;
        int sm_major = 0;
        int sm_minor = 0;
        size_t cubin_count = 0u;
        size_t workspace_bytes = 0u;
        size_t i;

        kernels_per_module_ = kernels_per_module_count(kernels_per_module);
        num_asts_ = num_asts;
        num_cohorts_ = (num_asts + ISPY_FEATURES_PER_KERNEL - 1u) / ISPY_FEATURES_PER_KERNEL;
        context_ = current_context();
        check_cuda(cuCtxGetDevice(&device), "cuCtxGetDevice");
        check_cuda(cuDeviceGetAttribute(&sm_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device), "cuDeviceGetAttribute major");
        check_cuda(cuDeviceGetAttribute(&sm_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device), "cuDeviceGetAttribute minor");

        option_ptrs.reserve(nvptx_options.size());
        for (i = 0u; i < nvptx_options.size(); ++i) {
            option_ptrs.push_back(nvptx_options[i].c_str());
        }

        {
            nb::gil_scoped_release release;
            check_result(implicit_sindy_ast_compiler_create_columns(
                kernels_per_module_,
                (unsigned int)sm_major,
                (unsigned int)sm_minor,
                option_ptrs.empty() ? nullptr : option_ptrs.data(),
                option_ptrs.size(),
                &compiler
            ));
            AstCompilerGuard compiler_guard(compiler);

            cubin_count = implicit_sindy_ast_cubin_count(num_asts, kernels_per_module_);
            if (cubin_count == 0u) {
                throw std::runtime_error("implicit_sindy_ast_cubin_count returned zero");
            }
            cubins.assign(cubin_count, nullptr);
            cubin_sizes.assign(cubin_count, 0u);
            modules.assign(cubin_count, nullptr);
            columns_modules.assign(cubin_count, nullptr);

            check_result(implicit_sindy_ast_compile_workspace_size(
                compiler_guard.get(),
                num_asts,
                worker_count,
                scratch_bytes_per_worker,
                &workspace_bytes
            ));
            workspace.resize(workspace_bytes);

            try {
                check_result(implicit_sindy_ast_compile_cubins(
                    compiler_guard.get(),
                    asts.data(),
                    asts.size(),
                    sizeof(asts[0]),
                    worker_count,
                    workspace.data(),
                    workspace.size(),
                    cubins.data(),
                    cubin_sizes.data()
                ));
                check_result(implicit_sindy_load_cubin_modules(
                    cubins.data(),
                    cubin_sizes.data(),
                    cubin_count,
                    modules.data()
                ));
                for (i = 0u; i < cubin_count; ++i) {
                    check_result(implicit_sindy_columns_module_create(
                        modules[i],
                        kernels_per_module_,
                        &columns_modules[i]
                    ));
                }
            } catch (...) {
                cleanup_columns_modules(columns_modules);
                if (!modules.empty()) {
                    (void)implicit_sindy_unload_modules(modules.data(), modules.size());
                }
                if (!cubins.empty()) {
                    implicit_sindy_free_cubins(cubins.data(), cubin_sizes.data(), cubins.size());
                }
                throw;
            }
        }

        cubins_ = std::move(cubins);
        cubin_sizes_ = std::move(cubin_sizes);
        modules_ = std::move(modules);
        columns_modules_ = std::move(columns_modules);
    }

    ColumnsKernelSet(const ColumnsKernelSet&) = delete;
    ColumnsKernelSet& operator=(const ColumnsKernelSet&) = delete;

    ~ColumnsKernelSet() {
        try {
            close();
        } catch (...) {
        }
    }

    void
    close() {
        ImplicitSindyResult unload_result = IMPLICIT_SINDY_SUCCESS;

        if (closed_) return;
        {
            PushedContext pushed(context_);
            cleanup_columns_modules(columns_modules_);
            if (!modules_.empty()) {
                unload_result = implicit_sindy_unload_modules(modules_.data(), modules_.size());
            }
        }
        if (!cubins_.empty()) {
            implicit_sindy_free_cubins(cubins_.data(), cubin_sizes_.data(), cubins_.size());
        }
        columns_modules_.clear();
        modules_.clear();
        cubins_.clear();
        cubin_sizes_.clear();
        closed_ = true;
        check_result(unload_result);
    }

    size_t
    num_cubins() const {
        return modules_.size();
    }

    size_t
    num_cohorts() const {
        return num_cohorts_;
    }

    size_t
    kernels_per_module() const {
        return kernels_per_module_;
    }

    void
    launch_raw(
        size_t cohort_index,
        uint64_t stream,
        int64_t num_settings,
        uint64_t primitive_features,
        int64_t row_count,
        int64_t primitive_feature_stride,
        int64_t num_primitive_features,
        uint64_t leaf_masks,
        uint64_t leaf_words,
        int64_t leaf_words_feature_stride,
        uint64_t output,
        int64_t output_setting_stride,
        int64_t output_feature_stride
    ) {
        CUcontext context = current_context();
        size_t module_index;
        size_t kernel_index;

        if (closed_) {
            throw std::runtime_error("ColumnsKernelSet is closed");
        }
        if (context != context_) {
            throw std::runtime_error("current CUDA context does not match the context used to compile the column kernels");
        }
        if (cohort_index >= num_cohorts_) {
            throw std::out_of_range("cohort_index is out of range");
        }

        module_index = cohort_index / kernels_per_module_;
        kernel_index = cohort_index % kernels_per_module_;
        if (module_index >= columns_modules_.size()) {
            throw std::out_of_range("module index is out of range");
        }

        nb::gil_scoped_release release;
        check_result(implicit_sindy_columns_launch(
            columns_modules_[module_index],
            kernel_index,
            reinterpret_cast<void*>(static_cast<uintptr_t>(stream)),
            num_settings,
            reinterpret_cast<const float*>(static_cast<uintptr_t>(primitive_features)),
            row_count,
            primitive_feature_stride,
            num_primitive_features,
            reinterpret_cast<const int32_t*>(static_cast<uintptr_t>(leaf_masks)),
            reinterpret_cast<const int32_t*>(static_cast<uintptr_t>(leaf_words)),
            leaf_words_feature_stride,
            reinterpret_cast<float*>(static_cast<uintptr_t>(output)),
            output_setting_stride,
            output_feature_stride
        ));
    }

private:
    CUcontext context_ = nullptr;
    bool closed_ = false;
    size_t kernels_per_module_ = 0u;
    size_t num_asts_ = 0u;
    size_t num_cohorts_ = 0u;
    std::vector<void*> cubins_;
    std::vector<size_t> cubin_sizes_;
    std::vector<void*> modules_;
    std::vector<ImplicitSindyColumnsModule*> columns_modules_;
};

class SingleAstColumnKernelSet {
public:
    SingleAstColumnKernelSet(
        nb::bytes unary_bytes_obj,
        nb::bytes binary_bytes_obj,
        size_t num_asts,
        int kernels_per_module,
        size_t worker_count,
        size_t scratch_bytes_per_worker,
        const std::vector<std::string>& nvptx_options
    ) {
        std::string unary_bytes = bytes_to_string(unary_bytes_obj, "unary_bytes");
        std::string binary_bytes = bytes_to_string(binary_bytes_obj, "binary_bytes");
        std::vector<BinaryAST> asts = decode_asts(unary_bytes, binary_bytes, num_asts);
        std::vector<const char*> option_ptrs;
        std::vector<void*> cubins;
        std::vector<size_t> cubin_sizes;
        std::vector<void*> modules;
        std::vector<ImplicitSindyAstColumnModule*> column_modules;
        std::vector<unsigned char> workspace;
        ImplicitSindyAstColumnCompiler* compiler = nullptr;
        CUdevice device = 0;
        int sm_major = 0;
        int sm_minor = 0;
        size_t cubin_count = 0u;
        size_t workspace_bytes = 0u;
        size_t i;

        kernels_per_module_ = kernels_per_module_count(kernels_per_module);
        num_asts_ = num_asts;
        context_ = current_context();
        check_cuda(cuCtxGetDevice(&device), "cuCtxGetDevice");
        check_cuda(cuDeviceGetAttribute(&sm_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device), "cuDeviceGetAttribute major");
        check_cuda(cuDeviceGetAttribute(&sm_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device), "cuDeviceGetAttribute minor");

        option_ptrs.reserve(nvptx_options.size());
        for (i = 0u; i < nvptx_options.size(); ++i) {
            option_ptrs.push_back(nvptx_options[i].c_str());
        }

        {
            nb::gil_scoped_release release;
            check_result(implicit_sindy_ast_column_compiler_create(
                kernels_per_module_,
                (unsigned int)sm_major,
                (unsigned int)sm_minor,
                option_ptrs.empty() ? nullptr : option_ptrs.data(),
                option_ptrs.size(),
                &compiler
            ));
            AstColumnCompilerGuard compiler_guard(compiler);

            cubin_count = implicit_sindy_ast_column_cubin_count(num_asts, kernels_per_module_);
            if (cubin_count == 0u) {
                throw std::runtime_error("implicit_sindy_ast_column_cubin_count returned zero");
            }
            cubins.assign(cubin_count, nullptr);
            cubin_sizes.assign(cubin_count, 0u);
            modules.assign(cubin_count, nullptr);
            column_modules.assign(cubin_count, nullptr);

            check_result(implicit_sindy_ast_column_compile_workspace_size(
                compiler_guard.get(),
                num_asts,
                worker_count,
                scratch_bytes_per_worker,
                &workspace_bytes
            ));
            workspace.resize(workspace_bytes);

            try {
                check_result(implicit_sindy_ast_column_compile_cubins(
                    compiler_guard.get(),
                    asts.data(),
                    asts.size(),
                    sizeof(asts[0]),
                    worker_count,
                    workspace.data(),
                    workspace.size(),
                    cubins.data(),
                    cubin_sizes.data()
                ));
                check_result(implicit_sindy_load_cubin_modules(
                    cubins.data(),
                    cubin_sizes.data(),
                    cubin_count,
                    modules.data()
                ));
                for (i = 0u; i < cubin_count; ++i) {
                    check_result(implicit_sindy_ast_column_module_create(
                        modules[i],
                        kernels_per_module_,
                        &column_modules[i]
                    ));
                }
            } catch (...) {
                cleanup_single_ast_column_modules(column_modules);
                if (!modules.empty()) {
                    (void)implicit_sindy_unload_modules(modules.data(), modules.size());
                }
                if (!cubins.empty()) {
                    implicit_sindy_free_cubins(cubins.data(), cubin_sizes.data(), cubins.size());
                }
                throw;
            }
        }

        cubins_ = std::move(cubins);
        cubin_sizes_ = std::move(cubin_sizes);
        modules_ = std::move(modules);
        column_modules_ = std::move(column_modules);
    }

    SingleAstColumnKernelSet(const SingleAstColumnKernelSet&) = delete;
    SingleAstColumnKernelSet& operator=(const SingleAstColumnKernelSet&) = delete;

    ~SingleAstColumnKernelSet() {
        try {
            close();
        } catch (...) {
        }
    }

    void
    close() {
        ImplicitSindyResult unload_result = IMPLICIT_SINDY_SUCCESS;

        if (closed_) return;
        {
            PushedContext pushed(context_);
            cleanup_single_ast_column_modules(column_modules_);
            if (!modules_.empty()) {
                unload_result = implicit_sindy_unload_modules(modules_.data(), modules_.size());
            }
        }
        if (!cubins_.empty()) {
            implicit_sindy_free_cubins(cubins_.data(), cubin_sizes_.data(), cubins_.size());
        }
        column_modules_.clear();
        modules_.clear();
        cubins_.clear();
        cubin_sizes_.clear();
        closed_ = true;
        check_result(unload_result);
    }

    size_t
    num_cubins() const {
        return modules_.size();
    }

    size_t
    num_asts() const {
        return num_asts_;
    }

    size_t
    kernels_per_module() const {
        return kernels_per_module_;
    }

    void
    launch_raw(
        size_t ast_index,
        int64_t feature_index,
        uint64_t stream,
        int64_t num_settings,
        uint64_t primitive_features,
        int64_t row_count,
        int64_t primitive_feature_stride,
        int64_t num_primitive_features,
        uint64_t leaf_masks,
        uint64_t leaf_words,
        int64_t leaf_words_feature_stride,
        uint64_t output,
        int64_t output_setting_stride
    ) {
        CUcontext context = current_context();
        size_t module_index;
        size_t kernel_index;

        if (closed_) {
            throw std::runtime_error("SingleAstColumnKernelSet is closed");
        }
        if (context != context_) {
            throw std::runtime_error("current CUDA context does not match the context used to compile the single-AST column kernels");
        }
        if (ast_index >= num_asts_) {
            throw std::out_of_range("ast_index is out of range");
        }

        module_index = ast_index / kernels_per_module_;
        kernel_index = ast_index % kernels_per_module_;
        if (module_index >= column_modules_.size()) {
            throw std::out_of_range("module index is out of range");
        }

        nb::gil_scoped_release release;
        check_result(implicit_sindy_ast_column_launch(
            column_modules_[module_index],
            kernel_index,
            reinterpret_cast<void*>(static_cast<uintptr_t>(stream)),
            feature_index,
            num_settings,
            reinterpret_cast<const float*>(static_cast<uintptr_t>(primitive_features)),
            row_count,
            primitive_feature_stride,
            num_primitive_features,
            reinterpret_cast<const int32_t*>(static_cast<uintptr_t>(leaf_masks)),
            reinterpret_cast<const int32_t*>(static_cast<uintptr_t>(leaf_words)),
            leaf_words_feature_stride,
            reinterpret_cast<float*>(static_cast<uintptr_t>(output)),
            output_setting_stride
        ));
    }

private:
    CUcontext context_ = nullptr;
    bool closed_ = false;
    size_t kernels_per_module_ = 0u;
    size_t num_asts_ = 0u;
    std::vector<void*> cubins_;
    std::vector<size_t> cubin_sizes_;
    std::vector<void*> modules_;
    std::vector<ImplicitSindyAstColumnModule*> column_modules_;
};

class RidgeSolver {
public:
    RidgeSolver() {
        context_ = current_context();
        nb::gil_scoped_release release;
        check_solve_result(implicit_feature_ridge_solve_create(&solve_));
    }

    RidgeSolver(const RidgeSolver&) = delete;
    RidgeSolver& operator=(const RidgeSolver&) = delete;

    ~RidgeSolver() {
        try {
            close();
        } catch (...) {
        }
    }

    void
    close() {
        ImplicitFeatureRidgeSolve* solve;
        ImplicitFeatureRidgeSolveResult result = IMPLICIT_FEATURE_RIDGE_SOLVE_SUCCESS;

        if (closed_) return;
        if (solve_ != nullptr) {
            solve = solve_;
            solve_ = nullptr;
            PushedContext pushed(context_);
            result = implicit_feature_ridge_solve_destroy(solve);
        }
        closed_ = true;
        check_solve_result(result);
    }

    void
    solve_posv_raw(
        uint64_t stream,
        int64_t row_count,
        int64_t num_settings,
        int64_t num_rhs,
        int64_t num_sweeps,
        float scale_epsilon,
        uint64_t gram,
        int64_t gram_col_stride,
        int64_t gram_setting_stride,
        uint64_t x_sum,
        int64_t x_sum_setting_stride,
        uint64_t xty,
        int64_t xty_rhs_stride,
        int64_t xty_setting_stride,
        uint64_t y_sum,
        int64_t y_sum_rhs_stride,
        int64_t y_sum_setting_stride,
        uint64_t alphas,
        int64_t alphas_stride,
        uint64_t x_mean,
        int64_t x_mean_setting_stride,
        uint64_t x_scale,
        int64_t x_scale_setting_stride,
        uint64_t y_mean,
        int64_t y_mean_rhs_stride,
        int64_t y_mean_setting_stride,
        uint64_t beta_standardized,
        int64_t beta_rhs_stride,
        int64_t beta_setting_stride,
        int64_t beta_sweep_stride,
        uint64_t solve_info,
        int64_t solve_info_sweep_stride,
        int64_t solve_info_setting_stride
    ) {
        check_ready();
        nb::gil_scoped_release release;
        check_solve_result(implicit_feature_ridge_solve_posv_sweep(
            solve_,
            reinterpret_cast<void*>(static_cast<uintptr_t>(stream)),
            row_count,
            num_settings,
            num_rhs,
            num_sweeps,
            scale_epsilon,
            reinterpret_cast<const float*>(static_cast<uintptr_t>(gram)),
            gram_col_stride,
            gram_setting_stride,
            reinterpret_cast<const float*>(static_cast<uintptr_t>(x_sum)),
            x_sum_setting_stride,
            reinterpret_cast<const float*>(static_cast<uintptr_t>(xty)),
            xty_rhs_stride,
            xty_setting_stride,
            reinterpret_cast<const float*>(static_cast<uintptr_t>(y_sum)),
            y_sum_rhs_stride,
            y_sum_setting_stride,
            reinterpret_cast<const float*>(static_cast<uintptr_t>(alphas)),
            alphas_stride,
            reinterpret_cast<float*>(static_cast<uintptr_t>(x_mean)),
            x_mean_setting_stride,
            reinterpret_cast<float*>(static_cast<uintptr_t>(x_scale)),
            x_scale_setting_stride,
            reinterpret_cast<float*>(static_cast<uintptr_t>(y_mean)),
            y_mean_rhs_stride,
            y_mean_setting_stride,
            reinterpret_cast<float*>(static_cast<uintptr_t>(beta_standardized)),
            beta_rhs_stride,
            beta_setting_stride,
            beta_sweep_stride,
            reinterpret_cast<int32_t*>(static_cast<uintptr_t>(solve_info)),
            solve_info_sweep_stride,
            solve_info_setting_stride
        ));
    }

    void
    solve_stlsq_raw(
        uint64_t stream,
        int64_t row_count,
        int64_t num_settings,
        int64_t num_rhs,
        int64_t num_sweeps,
        float scale_epsilon,
        uint64_t gram,
        int64_t gram_col_stride,
        int64_t gram_setting_stride,
        uint64_t x_sum,
        int64_t x_sum_setting_stride,
        uint64_t xty,
        int64_t xty_rhs_stride,
        int64_t xty_setting_stride,
        uint64_t y_sum,
        int64_t y_sum_rhs_stride,
        int64_t y_sum_setting_stride,
        uint64_t alphas,
        int64_t alphas_stride,
        uint64_t thresholds,
        int64_t thresholds_stride,
        uint64_t x_mean,
        int64_t x_mean_setting_stride,
        uint64_t x_scale,
        int64_t x_scale_setting_stride,
        uint64_t y_mean,
        int64_t y_mean_rhs_stride,
        int64_t y_mean_setting_stride,
        uint64_t beta_standardized,
        int64_t beta_rhs_stride,
        int64_t beta_setting_stride,
        int64_t beta_sweep_stride,
        uint64_t active_masks,
        int64_t active_mask_rhs_stride,
        int64_t active_mask_setting_stride,
        int64_t active_mask_sweep_stride,
        uint64_t active_counts,
        int64_t active_count_rhs_stride,
        int64_t active_count_setting_stride,
        int64_t active_count_sweep_stride,
        uint64_t iteration_counts,
        int64_t iteration_count_rhs_stride,
        int64_t iteration_count_setting_stride,
        int64_t iteration_count_sweep_stride,
        uint64_t solve_info,
        int64_t solve_info_sweep_stride,
        int64_t solve_info_setting_stride
    ) {
        check_ready();
        nb::gil_scoped_release release;
        check_solve_result(implicit_feature_ridge_solve_stlsq_sweep(
            solve_,
            reinterpret_cast<void*>(static_cast<uintptr_t>(stream)),
            row_count,
            num_settings,
            num_rhs,
            num_sweeps,
            scale_epsilon,
            reinterpret_cast<const float*>(static_cast<uintptr_t>(gram)),
            gram_col_stride,
            gram_setting_stride,
            reinterpret_cast<const float*>(static_cast<uintptr_t>(x_sum)),
            x_sum_setting_stride,
            reinterpret_cast<const float*>(static_cast<uintptr_t>(xty)),
            xty_rhs_stride,
            xty_setting_stride,
            reinterpret_cast<const float*>(static_cast<uintptr_t>(y_sum)),
            y_sum_rhs_stride,
            y_sum_setting_stride,
            reinterpret_cast<const float*>(static_cast<uintptr_t>(alphas)),
            alphas_stride,
            reinterpret_cast<const float*>(static_cast<uintptr_t>(thresholds)),
            thresholds_stride,
            reinterpret_cast<float*>(static_cast<uintptr_t>(x_mean)),
            x_mean_setting_stride,
            reinterpret_cast<float*>(static_cast<uintptr_t>(x_scale)),
            x_scale_setting_stride,
            reinterpret_cast<float*>(static_cast<uintptr_t>(y_mean)),
            y_mean_rhs_stride,
            y_mean_setting_stride,
            reinterpret_cast<float*>(static_cast<uintptr_t>(beta_standardized)),
            beta_rhs_stride,
            beta_setting_stride,
            beta_sweep_stride,
            reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(active_masks)),
            active_mask_rhs_stride,
            active_mask_setting_stride,
            active_mask_sweep_stride,
            reinterpret_cast<int32_t*>(static_cast<uintptr_t>(active_counts)),
            active_count_rhs_stride,
            active_count_setting_stride,
            active_count_sweep_stride,
            reinterpret_cast<int32_t*>(static_cast<uintptr_t>(iteration_counts)),
            iteration_count_rhs_stride,
            iteration_count_setting_stride,
            iteration_count_sweep_stride,
            reinterpret_cast<int32_t*>(static_cast<uintptr_t>(solve_info)),
            solve_info_sweep_stride,
            solve_info_setting_stride
        ));
    }

    void
    score_validation_mse_raw(
        uint64_t stream,
        int64_t validation_row_count,
        int64_t num_settings,
        int64_t num_rhs,
        int64_t num_sweeps,
        uint64_t validation_gram,
        int64_t validation_gram_col_stride,
        int64_t validation_gram_setting_stride,
        uint64_t validation_x_sum,
        int64_t validation_x_sum_setting_stride,
        uint64_t validation_xty,
        int64_t validation_xty_rhs_stride,
        int64_t validation_xty_setting_stride,
        uint64_t validation_y_sum,
        int64_t validation_y_sum_rhs_stride,
        int64_t validation_y_sum_setting_stride,
        uint64_t validation_yy,
        int64_t validation_yy_rhs_stride,
        int64_t validation_yy_setting_stride,
        uint64_t beta_standardized,
        int64_t beta_rhs_stride,
        int64_t beta_setting_stride,
        int64_t beta_sweep_stride,
        uint64_t x_mean,
        int64_t x_mean_setting_stride,
        uint64_t x_scale,
        int64_t x_scale_setting_stride,
        uint64_t y_mean,
        int64_t y_mean_rhs_stride,
        int64_t y_mean_setting_stride,
        uint64_t solve_info,
        int64_t solve_info_sweep_stride,
        int64_t solve_info_setting_stride,
        uint64_t mse,
        int64_t mse_rhs_stride,
        int64_t mse_setting_stride,
        int64_t mse_sweep_stride
    ) {
        check_ready();
        nb::gil_scoped_release release;
        check_solve_result(implicit_feature_ridge_score_validation_mse(
            solve_,
            reinterpret_cast<void*>(static_cast<uintptr_t>(stream)),
            validation_row_count,
            num_settings,
            num_rhs,
            num_sweeps,
            reinterpret_cast<const float*>(static_cast<uintptr_t>(validation_gram)),
            validation_gram_col_stride,
            validation_gram_setting_stride,
            reinterpret_cast<const float*>(static_cast<uintptr_t>(validation_x_sum)),
            validation_x_sum_setting_stride,
            reinterpret_cast<const float*>(static_cast<uintptr_t>(validation_xty)),
            validation_xty_rhs_stride,
            validation_xty_setting_stride,
            reinterpret_cast<const float*>(static_cast<uintptr_t>(validation_y_sum)),
            validation_y_sum_rhs_stride,
            validation_y_sum_setting_stride,
            reinterpret_cast<const float*>(static_cast<uintptr_t>(validation_yy)),
            validation_yy_rhs_stride,
            validation_yy_setting_stride,
            reinterpret_cast<const float*>(static_cast<uintptr_t>(beta_standardized)),
            beta_rhs_stride,
            beta_setting_stride,
            beta_sweep_stride,
            reinterpret_cast<const float*>(static_cast<uintptr_t>(x_mean)),
            x_mean_setting_stride,
            reinterpret_cast<const float*>(static_cast<uintptr_t>(x_scale)),
            x_scale_setting_stride,
            reinterpret_cast<const float*>(static_cast<uintptr_t>(y_mean)),
            y_mean_rhs_stride,
            y_mean_setting_stride,
            reinterpret_cast<const int32_t*>(static_cast<uintptr_t>(solve_info)),
            solve_info_sweep_stride,
            solve_info_setting_stride,
            reinterpret_cast<float*>(static_cast<uintptr_t>(mse)),
            mse_rhs_stride,
            mse_setting_stride,
            mse_sweep_stride
        ));
    }

private:
    void
    check_ready() {
        CUcontext context = current_context();
        if (closed_ || solve_ == nullptr) {
            throw std::runtime_error("RidgeSolver is closed");
        }
        if (context != context_) {
            throw std::runtime_error("current CUDA context does not match the context used to create the solver");
        }
    }

    CUcontext context_ = nullptr;
    bool closed_ = false;
    ImplicitFeatureRidgeSolve* solve_ = nullptr;
};

}  // namespace

NB_MODULE(_implicit_sindy, m) {
    m.attr("BINARY_AST_NUM_INPUTS") = (int)BINARY_AST_NUM_INPUTS;
    m.attr("BINARY_AST_NUM_UNARY_OPS") = (int)BINARY_AST_NUM_UNARY_OPS;
    m.attr("BINARY_AST_NUM_BINARY_OPS") = (int)BINARY_AST_NUM_BINARY_OPS;
    m.attr("BINARY_AST_UNARY_IDENTITY") = (int)BINARY_AST_UNARY_IDENTITY;
    m.attr("BINARY_AST_UNARY_SQUARE_F32") = (int)BINARY_AST_UNARY_SQUARE_F32;
    m.attr("BINARY_AST_UNARY_CUBE_F32") = (int)BINARY_AST_UNARY_CUBE_F32;
    m.attr("BINARY_AST_UNARY_NEG_FTZ_F32") = (int)BINARY_AST_UNARY_NEG_FTZ_F32;
    m.attr("BINARY_AST_UNARY_ABS_FTZ_F32") = (int)BINARY_AST_UNARY_ABS_FTZ_F32;
    m.attr("BINARY_AST_UNARY_RCP_APPROX_FTZ_F32") = (int)BINARY_AST_UNARY_RCP_APPROX_FTZ_F32;
    m.attr("BINARY_AST_UNARY_SQRT_APPROX_FTZ_F32") = (int)BINARY_AST_UNARY_SQRT_APPROX_FTZ_F32;
    m.attr("BINARY_AST_UNARY_RSQRT_APPROX_FTZ_F32") = (int)BINARY_AST_UNARY_RSQRT_APPROX_FTZ_F32;
    m.attr("BINARY_AST_UNARY_SIN_APPROX_FTZ_F32") = (int)BINARY_AST_UNARY_SIN_APPROX_FTZ_F32;
    m.attr("BINARY_AST_UNARY_COS_APPROX_FTZ_F32") = (int)BINARY_AST_UNARY_COS_APPROX_FTZ_F32;
    m.attr("BINARY_AST_UNARY_EX2_APPROX_FTZ_F32") = (int)BINARY_AST_UNARY_EX2_APPROX_FTZ_F32;
    m.attr("BINARY_AST_UNARY_EXP_APPROX_FTZ_F32") = (int)BINARY_AST_UNARY_EXP_APPROX_FTZ_F32;
    m.attr("BINARY_AST_UNARY_LOG2_APPROX_FTZ_F32") = (int)BINARY_AST_UNARY_LOG2_APPROX_FTZ_F32;
    m.attr("BINARY_AST_UNARY_LOG10_APPROX_FTZ_F32") = (int)BINARY_AST_UNARY_LOG10_APPROX_FTZ_F32;
    m.attr("BINARY_AST_UNARY_SAFE_RCP_F32") = (int)BINARY_AST_UNARY_SAFE_RCP_F32;
    m.attr("BINARY_AST_UNARY_SAFE_SQRT_F32") = (int)BINARY_AST_UNARY_SAFE_SQRT_F32;
    m.attr("BINARY_AST_UNARY_SAFE_RSQRT_F32") = (int)BINARY_AST_UNARY_SAFE_RSQRT_F32;
    m.attr("BINARY_AST_UNARY_SAFE_EX2_F32") = (int)BINARY_AST_UNARY_SAFE_EX2_F32;
    m.attr("BINARY_AST_UNARY_SAFE_EXP_F32") = (int)BINARY_AST_UNARY_SAFE_EXP_F32;
    m.attr("BINARY_AST_UNARY_SAFE_LOG2_F32") = (int)BINARY_AST_UNARY_SAFE_LOG2_F32;
    m.attr("BINARY_AST_UNARY_SAFE_LOG10_F32") = (int)BINARY_AST_UNARY_SAFE_LOG10_F32;
    m.attr("BINARY_AST_UNARY_ZERO_F32") = (int)BINARY_AST_UNARY_ZERO_F32;
    m.attr("BINARY_AST_BINARY_ADD_FTZ_F32") = (int)BINARY_AST_BINARY_ADD_FTZ_F32;
    m.attr("BINARY_AST_BINARY_SUB_FTZ_F32") = (int)BINARY_AST_BINARY_SUB_FTZ_F32;
    m.attr("BINARY_AST_BINARY_MUL_FTZ_F32") = (int)BINARY_AST_BINARY_MUL_FTZ_F32;
    m.attr("BINARY_AST_BINARY_KEEP_LEFT") = (int)BINARY_AST_BINARY_KEEP_LEFT;
    m.attr("BINARY_AST_BINARY_KEEP_RIGHT") = (int)BINARY_AST_BINARY_KEEP_RIGHT;
    m.attr("BINARY_AST_BINARY_DIV_APPROX_FTZ_F32") = (int)BINARY_AST_BINARY_DIV_APPROX_FTZ_F32;
    m.attr("BINARY_AST_BINARY_MIN_FTZ_F32") = (int)BINARY_AST_BINARY_MIN_FTZ_F32;
    m.attr("BINARY_AST_BINARY_MAX_FTZ_F32") = (int)BINARY_AST_BINARY_MAX_FTZ_F32;
    m.attr("BINARY_AST_BINARY_SAFE_DIV_F32") = (int)BINARY_AST_BINARY_SAFE_DIV_F32;
    m.attr("KERNELS_PER_MODULE_1") = 1;
    m.attr("KERNELS_PER_MODULE_4") = 4;

    nb::class_<GramKernelSet>(m, "GramKernelSet")
        .def(
            nb::init<
                nb::bytes,
                nb::bytes,
                nb::bytes,
                size_t,
                int,
                size_t,
                size_t,
                const std::vector<std::string>&
            >(),
            nb::arg("unary_bytes"),
            nb::arg("binary_bytes"),
            nb::arg("gram_template_ptx"),
            nb::arg("num_asts"),
            nb::arg("kernels_per_module"),
            nb::arg("worker_count"),
            nb::arg("scratch_bytes_per_worker"),
            nb::arg("nvptx_options") = std::vector<std::string>()
        )
        .def_prop_ro("num_cubins", &GramKernelSet::num_cubins)
        .def_prop_ro("num_cohorts", &GramKernelSet::num_cohorts)
        .def_prop_ro("kernels_per_module", &GramKernelSet::kernels_per_module)
        .def("close", &GramKernelSet::close)
        .def(
            "launch_raw",
            &GramKernelSet::launch_raw,
            nb::arg("cohort_index"),
            nb::arg("stream"),
            nb::arg("num_settings"),
            nb::arg("primitive_features"),
            nb::arg("row_count"),
            nb::arg("primitive_feature_stride"),
            nb::arg("num_primitive_features"),
            nb::arg("targets"),
            nb::arg("target_rhs_stride"),
            nb::arg("num_target_rhs"),
            nb::arg("leaf_masks"),
            nb::arg("leaf_words"),
            nb::arg("leaf_words_feature_stride"),
            nb::arg("gram"),
            nb::arg("gram_col_stride"),
            nb::arg("x_sum"),
            nb::arg("xty"),
            nb::arg("xty_rhs_stride"),
            nb::arg("y_sum"),
            nb::arg("yy")
        );

    nb::class_<ColumnsKernelSet>(m, "ColumnsKernelSet")
        .def(
            nb::init<
                nb::bytes,
                nb::bytes,
                size_t,
                int,
                size_t,
                size_t,
                const std::vector<std::string>&
            >(),
            nb::arg("unary_bytes"),
            nb::arg("binary_bytes"),
            nb::arg("num_asts"),
            nb::arg("kernels_per_module"),
            nb::arg("worker_count"),
            nb::arg("scratch_bytes_per_worker"),
            nb::arg("nvptx_options") = std::vector<std::string>()
        )
        .def_prop_ro("num_cubins", &ColumnsKernelSet::num_cubins)
        .def_prop_ro("num_cohorts", &ColumnsKernelSet::num_cohorts)
        .def_prop_ro("kernels_per_module", &ColumnsKernelSet::kernels_per_module)
        .def("close", &ColumnsKernelSet::close)
        .def(
            "launch_raw",
            &ColumnsKernelSet::launch_raw,
            nb::arg("cohort_index"),
            nb::arg("stream"),
            nb::arg("num_settings"),
            nb::arg("primitive_features"),
            nb::arg("row_count"),
            nb::arg("primitive_feature_stride"),
            nb::arg("num_primitive_features"),
            nb::arg("leaf_masks"),
            nb::arg("leaf_words"),
            nb::arg("leaf_words_feature_stride"),
            nb::arg("output"),
            nb::arg("output_setting_stride"),
            nb::arg("output_feature_stride")
        );

    nb::class_<SingleAstColumnKernelSet>(m, "SingleAstColumnKernelSet")
        .def(
            nb::init<
                nb::bytes,
                nb::bytes,
                size_t,
                int,
                size_t,
                size_t,
                const std::vector<std::string>&
            >(),
            nb::arg("unary_bytes"),
            nb::arg("binary_bytes"),
            nb::arg("num_asts"),
            nb::arg("kernels_per_module"),
            nb::arg("worker_count"),
            nb::arg("scratch_bytes_per_worker"),
            nb::arg("nvptx_options") = std::vector<std::string>()
        )
        .def_prop_ro("num_cubins", &SingleAstColumnKernelSet::num_cubins)
        .def_prop_ro("num_asts", &SingleAstColumnKernelSet::num_asts)
        .def_prop_ro("kernels_per_module", &SingleAstColumnKernelSet::kernels_per_module)
        .def("close", &SingleAstColumnKernelSet::close)
        .def(
            "launch_raw",
            &SingleAstColumnKernelSet::launch_raw,
            nb::arg("ast_index"),
            nb::arg("feature_index"),
            nb::arg("stream"),
            nb::arg("num_settings"),
            nb::arg("primitive_features"),
            nb::arg("row_count"),
            nb::arg("primitive_feature_stride"),
            nb::arg("num_primitive_features"),
            nb::arg("leaf_masks"),
            nb::arg("leaf_words"),
            nb::arg("leaf_words_feature_stride"),
            nb::arg("output"),
            nb::arg("output_setting_stride")
        );

    nb::class_<RidgeSolver>(m, "RidgeSolver")
        .def(nb::init<>())
        .def("close", &RidgeSolver::close)
        .def(
            "solve_posv_raw",
            &RidgeSolver::solve_posv_raw,
            nb::arg("stream"),
            nb::arg("row_count"),
            nb::arg("num_settings"),
            nb::arg("num_rhs"),
            nb::arg("num_sweeps"),
            nb::arg("scale_epsilon"),
            nb::arg("gram"),
            nb::arg("gram_col_stride"),
            nb::arg("gram_setting_stride"),
            nb::arg("x_sum"),
            nb::arg("x_sum_setting_stride"),
            nb::arg("xty"),
            nb::arg("xty_rhs_stride"),
            nb::arg("xty_setting_stride"),
            nb::arg("y_sum"),
            nb::arg("y_sum_rhs_stride"),
            nb::arg("y_sum_setting_stride"),
            nb::arg("alphas"),
            nb::arg("alphas_stride"),
            nb::arg("x_mean"),
            nb::arg("x_mean_setting_stride"),
            nb::arg("x_scale"),
            nb::arg("x_scale_setting_stride"),
            nb::arg("y_mean"),
            nb::arg("y_mean_rhs_stride"),
            nb::arg("y_mean_setting_stride"),
            nb::arg("beta_standardized"),
            nb::arg("beta_rhs_stride"),
            nb::arg("beta_setting_stride"),
            nb::arg("beta_sweep_stride"),
            nb::arg("solve_info"),
            nb::arg("solve_info_sweep_stride"),
            nb::arg("solve_info_setting_stride")
        )
        .def(
            "solve_stlsq_raw",
            &RidgeSolver::solve_stlsq_raw,
            nb::arg("stream"),
            nb::arg("row_count"),
            nb::arg("num_settings"),
            nb::arg("num_rhs"),
            nb::arg("num_sweeps"),
            nb::arg("scale_epsilon"),
            nb::arg("gram"),
            nb::arg("gram_col_stride"),
            nb::arg("gram_setting_stride"),
            nb::arg("x_sum"),
            nb::arg("x_sum_setting_stride"),
            nb::arg("xty"),
            nb::arg("xty_rhs_stride"),
            nb::arg("xty_setting_stride"),
            nb::arg("y_sum"),
            nb::arg("y_sum_rhs_stride"),
            nb::arg("y_sum_setting_stride"),
            nb::arg("alphas"),
            nb::arg("alphas_stride"),
            nb::arg("thresholds"),
            nb::arg("thresholds_stride"),
            nb::arg("x_mean"),
            nb::arg("x_mean_setting_stride"),
            nb::arg("x_scale"),
            nb::arg("x_scale_setting_stride"),
            nb::arg("y_mean"),
            nb::arg("y_mean_rhs_stride"),
            nb::arg("y_mean_setting_stride"),
            nb::arg("beta_standardized"),
            nb::arg("beta_rhs_stride"),
            nb::arg("beta_setting_stride"),
            nb::arg("beta_sweep_stride"),
            nb::arg("active_masks"),
            nb::arg("active_mask_rhs_stride"),
            nb::arg("active_mask_setting_stride"),
            nb::arg("active_mask_sweep_stride"),
            nb::arg("active_counts"),
            nb::arg("active_count_rhs_stride"),
            nb::arg("active_count_setting_stride"),
            nb::arg("active_count_sweep_stride"),
            nb::arg("iteration_counts"),
            nb::arg("iteration_count_rhs_stride"),
            nb::arg("iteration_count_setting_stride"),
            nb::arg("iteration_count_sweep_stride"),
            nb::arg("solve_info"),
            nb::arg("solve_info_sweep_stride"),
            nb::arg("solve_info_setting_stride")
        )
        .def(
            "score_validation_mse_raw",
            &RidgeSolver::score_validation_mse_raw,
            nb::arg("stream"),
            nb::arg("validation_row_count"),
            nb::arg("num_settings"),
            nb::arg("num_rhs"),
            nb::arg("num_sweeps"),
            nb::arg("validation_gram"),
            nb::arg("validation_gram_col_stride"),
            nb::arg("validation_gram_setting_stride"),
            nb::arg("validation_x_sum"),
            nb::arg("validation_x_sum_setting_stride"),
            nb::arg("validation_xty"),
            nb::arg("validation_xty_rhs_stride"),
            nb::arg("validation_xty_setting_stride"),
            nb::arg("validation_y_sum"),
            nb::arg("validation_y_sum_rhs_stride"),
            nb::arg("validation_y_sum_setting_stride"),
            nb::arg("validation_yy"),
            nb::arg("validation_yy_rhs_stride"),
            nb::arg("validation_yy_setting_stride"),
            nb::arg("beta_standardized"),
            nb::arg("beta_rhs_stride"),
            nb::arg("beta_setting_stride"),
            nb::arg("beta_sweep_stride"),
            nb::arg("x_mean"),
            nb::arg("x_mean_setting_stride"),
            nb::arg("x_scale"),
            nb::arg("x_scale_setting_stride"),
            nb::arg("y_mean"),
            nb::arg("y_mean_rhs_stride"),
            nb::arg("y_mean_setting_stride"),
            nb::arg("solve_info"),
            nb::arg("solve_info_sweep_stride"),
            nb::arg("solve_info_setting_stride"),
            nb::arg("mse"),
            nb::arg("mse_rhs_stride"),
            nb::arg("mse_setting_stride"),
            nb::arg("mse_sweep_stride")
        );
}
