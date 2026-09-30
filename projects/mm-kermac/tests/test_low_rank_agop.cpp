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
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <kermac.hpp>
#include <check_result_helper.h>

using namespace kermac;

static const size_t NUM_BYTES = 1ull << 26;
static constexpr float kTol = 1e-2f;

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

struct TensorInfo {
    int64_t rows;
    int64_t cols;
};

struct VectorInfo {
    int64_t len;
};

static TensorInfo load_tensor_info(const std::string& base_path) {
    const std::string path = base_path + ".desc";
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open " << path << "\n";
        ASSERT(false);
    }

    file.seekg(0, std::ios::end);
    const std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (size < 0 || (size % static_cast<std::streamsize>(sizeof(int64_t))) != 0) {
        std::cerr << "Invalid desc size for " << path << "\n";
        ASSERT(false);
    }

    const size_t count = static_cast<size_t>(size / sizeof(int64_t));
    if (count != 3) {
        std::cerr << "Expected 2D tensor desc for " << path << "\n";
        ASSERT(false);
    }

    std::vector<int64_t> desc(count);
    if (!file.read(reinterpret_cast<char*>(desc.data()), size)) {
        std::cerr << "Failed to read " << path << "\n";
        ASSERT(false);
    }

    if (desc[0] != static_cast<int64_t>(sizeof(float))) {
        std::cerr << "Unexpected dtype size in " << path << "\n";
        ASSERT(false);
    }

    return TensorInfo{desc[1], desc[2]};
}

static VectorInfo load_vector_info(const std::string& base_path) {
    const std::string path = base_path + ".desc";
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open " << path << "\n";
        ASSERT(false);
    }

    file.seekg(0, std::ios::end);
    const std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (size < 0 || (size % static_cast<std::streamsize>(sizeof(int64_t))) != 0) {
        std::cerr << "Invalid desc size for " << path << "\n";
        ASSERT(false);
    }

    const size_t count = static_cast<size_t>(size / sizeof(int64_t));
    if (count != 2) {
        std::cerr << "Expected 1D tensor desc for " << path << "\n";
        ASSERT(false);
    }

    std::vector<int64_t> desc(count);
    if (!file.read(reinterpret_cast<char*>(desc.data()), size)) {
        std::cerr << "Failed to read " << path << "\n";
        ASSERT(false);
    }

    if (desc[0] != static_cast<int64_t>(sizeof(float))) {
        std::cerr << "Unexpected dtype size in " << path << "\n";
        ASSERT(false);
    }

    return VectorInfo{desc[1]};
}

static void load_matrix_data(
    const std::string& base_path,
    int64_t rows,
    int64_t cols,
    HostTensor<float>& dst
) {
    const std::string path = base_path + ".dat";
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open " << path << "\n";
        ASSERT(false);
    }

    file.seekg(0, std::ios::end);
    const std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    const std::streamsize expected_size =
        static_cast<std::streamsize>(rows * cols * sizeof(float));
    if (size != expected_size) {
        std::cerr << "Unexpected data size for " << path << "\n";
        ASSERT(false);
    }

    std::vector<float> data(static_cast<size_t>(rows * cols));
    if (!file.read(reinterpret_cast<char*>(data.data()), expected_size)) {
        std::cerr << "Failed to read " << path << "\n";
        ASSERT(false);
    }

    const KermacTensor raw = dst.raw();
    ASSERT(raw.extent[0] == rows);
    ASSERT(raw.extent[1] == cols);

    float* ptr = dst.ptr();
    const int64_t ld = raw.stride[1];
    for (int64_t col = 0; col < cols; ++col) {
        const float* src = data.data() + col * rows;
        float* dst_col = ptr + col * ld;
        std::memcpy(dst_col, src, static_cast<size_t>(rows * sizeof(float)));
    }
}

static void load_vector_data(
    const std::string& base_path,
    int64_t len,
    HostTensor<float>& dst
) {
    const std::string path = base_path + ".dat";
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open " << path << "\n";
        ASSERT(false);
    }

    file.seekg(0, std::ios::end);
    const std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    const std::streamsize expected_size = static_cast<std::streamsize>(len * sizeof(float));
    if (size != expected_size) {
        std::cerr << "Unexpected data size for " << path << "\n";
        ASSERT(false);
    }

    std::vector<float> data(static_cast<size_t>(len));
    if (!file.read(reinterpret_cast<char*>(data.data()), expected_size)) {
        std::cerr << "Failed to read " << path << "\n";
        ASSERT(false);
    }

    const KermacTensor raw = dst.raw();
    ASSERT(raw.num_modes == 1);
    ASSERT(raw.extent[0] == len);

    std::memcpy(dst.ptr(), data.data(), static_cast<size_t>(len * sizeof(float)));
}

static void check_close(
    const char* label,
    const HostTensor<float>& got,
    const HostTensor<float>& expected,
    float tol
) {
    const KermacTensor g = got.raw();
    const KermacTensor e = expected.raw();

    ASSERT(g.num_modes == 2);
    ASSERT(e.num_modes == 2);
    ASSERT(g.extent[0] == e.extent[0]);
    ASSERT(g.extent[1] == e.extent[1]);

    const int64_t rows = g.extent[0];
    const int64_t cols = g.extent[1];
    const int64_t g_ld = g.stride[1];
    const int64_t e_ld = e.stride[1];
    const float* g_ptr = got.ptr();
    const float* e_ptr = expected.ptr();

    float max_diff = 0.0f;
    int64_t max_r = 0;
    int64_t max_c = 0;
    float max_g = 0.0f;
    float max_e = 0.0f;

    for (int64_t c = 0; c < cols; ++c) {
        for (int64_t r = 0; r < rows; ++r) {
            const int64_t g_idx = c * g_ld + r;
            const int64_t e_idx = c * e_ld + r;
            const float diff = std::fabs(g_ptr[g_idx] - e_ptr[e_idx]);
            if (diff > max_diff) {
                max_diff = diff;
                max_r = r;
                max_c = c;
                max_g = g_ptr[g_idx];
                max_e = e_ptr[e_idx];
            }
        }
    }

    if (max_diff > tol) {
        std::cerr << "FAIL: " << label
                  << " max_diff=" << max_diff
                  << " at (r=" << max_r
                  << ", c=" << max_c
                  << "), got=" << max_g
                  << ", expected=" << max_e << "\n";
        ASSERT(false);
    }
}

static void check_close_vector(
    const char* label,
    const HostTensor<float>& got,
    const HostTensor<float>& expected,
    float tol
) {
    const KermacTensor g = got.raw();
    const KermacTensor e = expected.raw();

    ASSERT(g.num_modes == 1);
    ASSERT(e.num_modes == 1);
    ASSERT(g.extent[0] == e.extent[0]);

    const int64_t len = g.extent[0];
    const float* g_ptr = got.ptr();
    const float* e_ptr = expected.ptr();

    float max_diff = 0.0f;
    int64_t max_idx = 0;
    float max_g = 0.0f;
    float max_e = 0.0f;

    for (int64_t i = 0; i < len; ++i) {
        const float diff = std::fabs(g_ptr[i] - e_ptr[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_idx = i;
            max_g = g_ptr[i];
            max_e = e_ptr[i];
        }
    }

    if (max_diff > tol) {
        std::cerr << "FAIL: " << label
                  << " max_diff=" << max_diff
                  << " at i=" << max_idx
                  << ", got=" << max_g
                  << ", expected=" << max_e << "\n";
        ASSERT(false);
    }
}

static void compute_diag(
    const HostTensor<float>& full,
    HostTensor<float>& diag
) {
    const KermacTensor f = full.raw();
    const KermacTensor d = diag.raw();

    ASSERT(f.num_modes == 2);
    ASSERT(d.num_modes == 1);
    ASSERT(f.extent[0] == f.extent[1]);
    ASSERT(d.extent[0] == f.extent[0]);

    const float* f_ptr = full.ptr();
    float* d_ptr = diag.ptr();
    const int64_t ld = f.stride[1];

    for (int64_t i = 0; i < d.extent[0]; ++i) {
        d_ptr[i] = f_ptr[i + i * ld];
    }
}

static void compute_reference_agop(
    const HostTensor<float>& kernel,
    const HostTensor<float>& solution,
    const HostTensor<float>& features,
    float bandwidth,
    HostTensor<float>& out
) {
    ASSERT(bandwidth > 0.0f);

    const KermacTensor k = kernel.raw();
    const KermacTensor s = solution.raw();
    const KermacTensor x = features.raw();
    const KermacTensor o = out.raw();

    ASSERT(k.num_modes == 2);
    ASSERT(s.num_modes == 2);
    ASSERT(x.num_modes == 2);
    ASSERT(o.num_modes == 2);
    ASSERT(k.extent[0] == k.extent[1]);
    ASSERT(k.extent[0] == x.extent[0]);
    ASSERT(s.extent[0] == x.extent[0]);
    ASSERT(o.extent[0] == x.extent[1]);
    ASSERT(o.extent[1] == x.extent[1]);

    const int64_t M = k.extent[0];
    const int64_t N = k.extent[1];
    const int64_t D = x.extent[1];
    const int64_t C = s.extent[1];

    const float* k_ptr = kernel.ptr();
    const float* s_ptr = solution.ptr();
    const float* x_ptr = features.ptr();
    float* o_ptr = out.ptr();

    const int64_t k_ld = k.stride[1];
    const int64_t s_ld = s.stride[1];
    const int64_t x_ld = x.stride[1];
    const int64_t o_ld = o.stride[1];

    std::memset(o_ptr, 0, static_cast<size_t>(o_ld * D) * sizeof(float));

    const float scale = 1.0f / bandwidth;
    std::vector<float> g(static_cast<size_t>(D));

    for (int64_t m = 0; m < M; ++m) {
        for (int64_t c = 0; c < C; ++c) {
            float sum_nc = 0.0f;
            for (int64_t n = 0; n < N; ++n) {
                const float kmn = k_ptr[n * k_ld + m];
                const float snc = s_ptr[c * s_ld + n];
                sum_nc += kmn * snc;
            }

            for (int64_t d = 0; d < D; ++d) {
                float term_n = 0.0f;
                for (int64_t n = 0; n < N; ++n) {
                    const float kmn = k_ptr[n * k_ld + m];
                    const float snc = s_ptr[c * s_ld + n];
                    const float xnd = x_ptr[d * x_ld + n];
                    term_n += kmn * snc * xnd;
                }
                const float xmd = x_ptr[d * x_ld + m];
                g[static_cast<size_t>(d)] = scale * (xmd * sum_nc - term_n);
            }

            for (int64_t e = 0; e < D; ++e) {
                const float g_e = g[static_cast<size_t>(e)];
                float* o_col = o_ptr + e * o_ld;
                for (int64_t d = 0; d < D; ++d) {
                    o_col[d] += g[static_cast<size_t>(d)] * g_e;
                }
            }
        }
    }

    const float inv_m = 1.0f / static_cast<float>(M);
    for (int64_t e = 0; e < D; ++e) {
        float* o_col = o_ptr + e * o_ld;
        for (int64_t d = 0; d < D; ++d) {
            o_col[d] *= inv_m;
        }
    }
}

int main() {
    if (!has_cuda_device()) {
        std::cerr << "SKIP: no CUDA device available\n";
        return 77;
    }

#ifndef KERMAC_LOW_RANK_DATA_DIR
    std::cerr << "SKIP: KERMAC_LOW_RANK_DATA_DIR not defined\n";
    return 77;
#endif

    const std::string data_dir = KERMAC_LOW_RANK_DATA_DIR;
    const std::string kernel_path = data_dir + "/kernel_matrix";
    const std::string solution_path = data_dir + "/solution";
    const std::string features_path = data_dir + "/features";
    const std::string feature_matrix_path = data_dir + "/feature_matrix";
    const std::string feature_diag_path = data_dir + "/feature_matrix_diag";

    const TensorInfo kernel_info = load_tensor_info(kernel_path);
    const TensorInfo solution_info = load_tensor_info(solution_path);
    const TensorInfo features_info = load_tensor_info(features_path);
    const TensorInfo feature_matrix_info = load_tensor_info(feature_matrix_path);
    const VectorInfo feature_diag_info = load_vector_info(feature_diag_path);

    ASSERT(kernel_info.rows == kernel_info.cols);
    ASSERT(kernel_info.rows == solution_info.rows);
    ASSERT(kernel_info.rows == features_info.rows);
    ASSERT(features_info.cols == feature_matrix_info.rows);
    ASSERT(feature_matrix_info.rows == feature_matrix_info.cols);
    ASSERT(feature_diag_info.len == features_info.cols);

    const int64_t num_dims = features_info.cols;

    try {
        Kermac kermac(2);
        CUstream stream;
        cuCheck(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

        void* host_mem = host_alloc(NUM_BYTES);
        void* device_mem = device_alloc(kermac, NUM_BYTES);

        {
            HostStackAllocator hsa(host_mem, NUM_BYTES);
            DeviceStackAllocator dsa(device_mem, NUM_BYTES);

            HostTensor<float> h_kernel(hsa, kernel_info.rows, kernel_info.cols);
            HostTensor<float> h_solution(hsa, solution_info.rows, solution_info.cols);
            HostTensor<float> h_features(hsa, features_info.rows, features_info.cols);
            HostTensor<float> h_feature_matrix(hsa, feature_matrix_info.rows, feature_matrix_info.cols);
            HostTensor<float> h_feature_diag(hsa, feature_diag_info.len);

            load_matrix_data(kernel_path, kernel_info.rows, kernel_info.cols, h_kernel);
            load_matrix_data(solution_path, solution_info.rows, solution_info.cols, h_solution);
            load_matrix_data(features_path, features_info.rows, features_info.cols, h_features);
            load_matrix_data(
                feature_matrix_path,
                feature_matrix_info.rows,
                feature_matrix_info.cols,
                h_feature_matrix
            );
            load_vector_data(feature_diag_path, feature_diag_info.len, h_feature_diag);

            HostTensor<float> h_expected_agop(hsa, num_dims, num_dims);
            compute_reference_agop(
                h_kernel,
                h_solution,
                h_features,
                KERMAC_LOW_RANK_BANDWIDTH,
                h_expected_agop
            );
            check_close("low_rank_agop_data", h_expected_agop, h_feature_matrix, kTol);

            HostTensor<float> h_expected_diag(hsa, num_dims);
            compute_diag(h_feature_matrix, h_expected_diag);
            check_close_vector("low_rank_agop_diag_data", h_expected_diag, h_feature_diag, kTol);

            DeviceTensor<float> d_kernel(dsa, kernel_info.rows, kernel_info.cols);
            DeviceTensor<float> d_solution(dsa, solution_info.rows, solution_info.cols);
            DeviceTensor<float> d_features(dsa, features_info.rows, features_info.cols);

            d_kernel.copy_from(h_kernel, stream);
            d_solution.copy_from(h_solution, stream);
            d_features.copy_from(h_features, stream);

            Agop agop = Agop::create(kermac, hsa);

            DeviceTensor<float> d_agop_full(dsa, num_dims, num_dims);
            HostTensor<float> h_agop_full(hsa, num_dims, num_dims);

            agop.run(
                kermac, dsa,
                AgopBackend::KERMAC_AGOP_BACKEND_CUTENSOR_F32,
                AgopOutput::KERMAC_AGOP_OUTPUT_FULL,
                KERMAC_LOW_RANK_BANDWIDTH,
                d_kernel,
                d_features,
                d_solution,
                d_features,
                d_agop_full,
                stream
            );
            h_agop_full.copy_from(d_agop_full, stream);
            cuCheck(cuStreamSynchronize(stream));
            check_close("low_rank_agop_cutensor", h_agop_full, h_feature_matrix, kTol);

            agop.run(
                kermac, dsa,
                AgopBackend::KERMAC_AGOP_BACKEND_FUSED,
                AgopOutput::KERMAC_AGOP_OUTPUT_FULL,
                KERMAC_LOW_RANK_BANDWIDTH,
                d_kernel,
                d_features,
                d_solution,
                d_features,
                d_agop_full,
                stream
            );
            h_agop_full.copy_from(d_agop_full, stream);
            cuCheck(cuStreamSynchronize(stream));
            check_close("low_rank_agop_fused", h_agop_full, h_feature_matrix, kTol);

            DeviceTensor<float> d_agop_diag(dsa, num_dims);
            HostTensor<float> h_agop_diag(hsa, num_dims);

            agop.run(
                kermac, dsa,
                AgopBackend::KERMAC_AGOP_BACKEND_CUTENSOR_F32,
                AgopOutput::KERMAC_AGOP_OUTPUT_DIAG,
                KERMAC_LOW_RANK_BANDWIDTH,
                d_kernel,
                d_features,
                d_solution,
                d_features,
                d_agop_diag,
                stream
            );
            h_agop_diag.copy_from(d_agop_diag, stream);
            cuCheck(cuStreamSynchronize(stream));
            check_close_vector("low_rank_agop_diag_cutensor", h_agop_diag, h_feature_diag, kTol);

            agop.run(
                kermac, dsa,
                AgopBackend::KERMAC_AGOP_BACKEND_FUSED,
                AgopOutput::KERMAC_AGOP_OUTPUT_DIAG,
                KERMAC_LOW_RANK_BANDWIDTH,
                d_kernel,
                d_features,
                d_solution,
                d_features,
                d_agop_diag,
                stream
            );
            h_agop_diag.copy_from(d_agop_diag, stream);
            cuCheck(cuStreamSynchronize(stream));
            check_close_vector("low_rank_agop_diag_fused", h_agop_diag, h_feature_diag, kTol);
        }

        host_free(host_mem);
        device_free(kermac, device_mem);
        cuCheck(cuStreamDestroy(stream));
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << "\n";
        return 1;
    }
}
