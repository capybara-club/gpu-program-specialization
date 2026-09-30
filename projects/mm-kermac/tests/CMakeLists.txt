# SPDX-FileCopyrightText: 2026 Charles Durham
# SPDX-License-Identifier: MIT
#
# MIT License
#
# Copyright (c) 2026 Charles Durham
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

add_executable(kermac_cholesky_tests test_cholesky.cpp)
target_link_libraries(kermac_cholesky_tests PRIVATE kermac kermac_cpu)
add_test(NAME kermac.cholesky COMMAND kermac_cholesky_tests)
set_tests_properties(kermac.cholesky PROPERTIES SKIP_RETURN_CODE 77)

add_executable(kermac_linear_regression_tests test_linear_regression.cpp)
target_link_libraries(kermac_linear_regression_tests PRIVATE kermac kermac_cpu)
add_test(NAME kermac.linear_regression COMMAND kermac_linear_regression_tests)
set_tests_properties(kermac.linear_regression PROPERTIES SKIP_RETURN_CODE 77)

add_executable(kermac_syev_batched_tests test_syev_batched.cpp)
target_link_libraries(kermac_syev_batched_tests PRIVATE kermac kermac_cpu)
add_test(NAME kermac.syev_batched COMMAND kermac_syev_batched_tests)
set_tests_properties(kermac.syev_batched PROPERTIES SKIP_RETURN_CODE 77)

add_executable(kermac_cholesky_legacy_tests test_cholesky_legacy.cpp)
target_link_libraries(kermac_cholesky_legacy_tests PRIVATE kermac kermac_cpu)
add_test(NAME kermac.cholesky_legacy COMMAND kermac_cholesky_legacy_tests)
set_tests_properties(kermac.cholesky_legacy PROPERTIES SKIP_RETURN_CODE 77)

add_executable(kermac_logdet_norm_norm_h2_tests test_logdet_norm_norm_h2.cpp)
target_link_libraries(kermac_logdet_norm_norm_h2_tests PRIVATE kermac kermac_cpu)
add_test(NAME kermac.logdet_norm_norm_h2 COMMAND kermac_logdet_norm_norm_h2_tests)
set_tests_properties(kermac.logdet_norm_norm_h2 PROPERTIES SKIP_RETURN_CODE 77)

add_executable(kermac_row_stats_tests test_stdev.cpp)
target_link_libraries(kermac_row_stats_tests PRIVATE kermac kermac_cpu)
add_test(NAME kermac.row_stats COMMAND kermac_row_stats_tests)
set_tests_properties(kermac.row_stats PROPERTIES SKIP_RETURN_CODE 77)

add_executable(kermac_apply_row_stats_tests test_apply_stdev.cpp)
target_link_libraries(kermac_apply_row_stats_tests PRIVATE kermac kermac_cpu)
add_test(NAME kermac.apply_row_stats COMMAND kermac_apply_row_stats_tests)
set_tests_properties(kermac.apply_row_stats PROPERTIES SKIP_RETURN_CODE 77)

add_executable(kermac_mse_accuracy_tests test_mse_accuracy.cpp)
target_link_libraries(kermac_mse_accuracy_tests PRIVATE kermac kermac_cpu)
add_test(NAME kermac.mse_accuracy COMMAND kermac_mse_accuracy_tests)
set_tests_properties(kermac.mse_accuracy PROPERTIES SKIP_RETURN_CODE 77)

add_executable(kermac_semiring_gold_tests test_semiring_gold.cpp)
target_link_libraries(kermac_semiring_gold_tests PRIVATE kermac kermac_cpu)
add_test(NAME kermac.semiring_gold COMMAND kermac_semiring_gold_tests)
set_tests_properties(kermac.semiring_gold PROPERTIES SKIP_RETURN_CODE 77)

add_executable(kermac_semiring_copy_grad_tests test_semiring_copy_grad.cpp)
target_link_libraries(kermac_semiring_copy_grad_tests PRIVATE kermac kermac_cpu)
add_test(NAME kermac.semiring_copy_grad COMMAND kermac_semiring_copy_grad_tests)
set_tests_properties(kermac.semiring_copy_grad PROPERTIES SKIP_RETURN_CODE 77)

add_executable(kermac_semiring_gradient_l2_tests test_semiring_gradient_l2.cpp)
target_link_libraries(kermac_semiring_gradient_l2_tests PRIVATE kermac kermac_cpu)
add_test(NAME kermac.semiring_gradient_l2 COMMAND kermac_semiring_gradient_l2_tests)
set_tests_properties(kermac.semiring_gradient_l2 PROPERTIES SKIP_RETURN_CODE 77)

add_executable(kermac_p_norm_cpu_tests test_p_norm_cpu.cpp)
target_link_libraries(kermac_p_norm_cpu_tests PRIVATE kermac kermac_cpu)
add_test(NAME kermac.p_norm_cpu COMMAND kermac_p_norm_cpu_tests)
set_tests_properties(kermac.p_norm_cpu PROPERTIES SKIP_RETURN_CODE 77)

add_executable(kermac_fmnist_tests test_fmnist.cpp)
target_link_libraries(kermac_fmnist_tests PRIVATE kermac kermac_cpu)
add_test(NAME kermac.fmnist COMMAND kermac_fmnist_tests)
set_tests_properties(kermac.fmnist PROPERTIES SKIP_RETURN_CODE 77)

set(FMNIST_U8_PTX_SRC "${CMAKE_CURRENT_SOURCE_DIR}/cuda/fmnist_u8_to_f32_inject.cu")
kermac_add_ptx_obj(kermac_fmnist_u8_ptx_target "${FMNIST_U8_PTX_SRC}")

add_executable(kermac_fmnist_inject_tests test_fmnist_inject.cpp)
add_dependencies(kermac_fmnist_inject_tests kermac_fmnist_u8_ptx_target)
target_link_libraries(kermac_fmnist_inject_tests PRIVATE kermac kermac_cpu mm_ptx_headers)
target_compile_definitions(kermac_fmnist_inject_tests PRIVATE
    PTX_FMNIST_U8_INJECT="$<TARGET_OBJECTS:kermac_fmnist_u8_ptx_target>"
)
add_test(NAME kermac.fmnist_inject COMMAND kermac_fmnist_inject_tests)
set_tests_properties(kermac.fmnist_inject PROPERTIES SKIP_RETURN_CODE 77)

set(U8_PACK2X2_16_PTX_SRC "${CMAKE_CURRENT_SOURCE_DIR}/cuda/u8_pack2x2_overlap_16x16_inject.cu")
set(U8_PACK2X2_28_PTX_SRC "${CMAKE_CURRENT_SOURCE_DIR}/cuda/u8_pack2x2_overlap_28x28_inject.cu")
set(U8_PACK2X2_32_PTX_SRC "${CMAKE_CURRENT_SOURCE_DIR}/cuda/u8_pack2x2_overlap_32x32_inject.cu")

kermac_add_ptx_obj(kermac_u8_pack2x2_16_ptx_target "${U8_PACK2X2_16_PTX_SRC}")
kermac_add_ptx_obj(kermac_u8_pack2x2_28_ptx_target "${U8_PACK2X2_28_PTX_SRC}")
kermac_add_ptx_obj(kermac_u8_pack2x2_32_ptx_target "${U8_PACK2X2_32_PTX_SRC}")

set(U8_PACK2X2_16_PTX "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/kermac_u8_pack2x2_16_ptx_target.dir/cuda/u8_pack2x2_overlap_16x16_inject.ptx")
set(U8_PACK2X2_28_PTX "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/kermac_u8_pack2x2_28_ptx_target.dir/cuda/u8_pack2x2_overlap_28x28_inject.ptx")
set(U8_PACK2X2_32_PTX "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/kermac_u8_pack2x2_32_ptx_target.dir/cuda/u8_pack2x2_overlap_32x32_inject.ptx")

set_source_files_properties(
    test_u8_pack2x2_inject.cpp
    PROPERTIES
    OBJECT_DEPENDS "${U8_PACK2X2_16_PTX};${U8_PACK2X2_28_PTX};${U8_PACK2X2_32_PTX}"
)

add_executable(kermac_u8_pack2x2_inject_tests test_u8_pack2x2_inject.cpp)
add_dependencies(
    kermac_u8_pack2x2_inject_tests
    kermac_u8_pack2x2_16_ptx_target
    kermac_u8_pack2x2_28_ptx_target
    kermac_u8_pack2x2_32_ptx_target
)
target_link_libraries(kermac_u8_pack2x2_inject_tests PRIVATE kermac mm_ptx_headers)
target_compile_definitions(kermac_u8_pack2x2_inject_tests PRIVATE
    PTX_U8_PACK2X2_16_INJECT="${U8_PACK2X2_16_PTX}"
    PTX_U8_PACK2X2_28_INJECT="${U8_PACK2X2_28_PTX}"
    PTX_U8_PACK2X2_32_INJECT="${U8_PACK2X2_32_PTX}"
)
add_test(NAME kermac.u8_pack2x2_inject COMMAND kermac_u8_pack2x2_inject_tests)
set_tests_properties(kermac.u8_pack2x2_inject PROPERTIES SKIP_RETURN_CODE 77)

add_executable(kermac_fmnist_legacy_tests test_fmnist_legacy.cpp)
target_link_libraries(kermac_fmnist_legacy_tests PRIVATE kermac kermac_cpu)
add_test(NAME kermac.fmnist_legacy COMMAND kermac_fmnist_legacy_tests)
set_tests_properties(kermac.fmnist_legacy PROPERTIES SKIP_RETURN_CODE 77)

add_executable(kermac_nvrtc_compile_tests test_nvrtc_compile.cpp)
target_link_libraries(kermac_nvrtc_compile_tests PRIVATE kermac)
add_test(NAME kermac.nvrtc_compile COMMAND kermac_nvrtc_compile_tests)
set_tests_properties(kermac.nvrtc_compile PROPERTIES SKIP_RETURN_CODE 77)

set(KERMAC_CUTLASS_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/../cutlass/include")
set(KERMAC_CUTE_CUDA_ARCHITECTURES "80-real;120-real" CACHE STRING
    "CUDA architectures for the CuTe gather-SYRK bench and tests")

add_executable(kermac_cute_gather_syrk_tests test_cute_gather_syrk.cu)
target_link_libraries(kermac_cute_gather_syrk_tests PRIVATE CUDA::cudart CUDA::cublas CUDA::cublasLt)
target_include_directories(kermac_cute_gather_syrk_tests PRIVATE
    "${PROJECT_SOURCE_DIR}/bench"
    "${KERMAC_CUTLASS_INCLUDE_DIR}"
)
set_target_properties(kermac_cute_gather_syrk_tests PROPERTIES
    CUDA_ARCHITECTURES "${KERMAC_CUTE_CUDA_ARCHITECTURES}"
)
target_compile_features(kermac_cute_gather_syrk_tests PRIVATE cxx_std_17)
target_compile_options(kermac_cute_gather_syrk_tests PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>"
    "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--expt-relaxed-constexpr>"
)
add_test(NAME kermac.cute_gather_syrk COMMAND kermac_cute_gather_syrk_tests)
set_tests_properties(kermac.cute_gather_syrk PROPERTIES SKIP_RETURN_CODE 77)

if(NOT KERMAC_CUSOLVERDX_MATHDX_ROOT)
    file(GLOB _kermac_mathdx_roots
        "${PROJECT_SOURCE_DIR}/../third_party/mathdx/nvidia-mathdx-*/nvidia/mathdx/*")
    list(SORT _kermac_mathdx_roots COMPARE NATURAL ORDER DESCENDING)
    list(LENGTH _kermac_mathdx_roots _kermac_mathdx_root_count)
    if(_kermac_mathdx_root_count GREATER 0)
        list(GET _kermac_mathdx_roots 0 KERMAC_CUSOLVERDX_MATHDX_ROOT)
    endif()
endif()

if(KERMAC_CUSOLVERDX_MATHDX_ROOT)
    find_package(mathdx QUIET COMPONENTS cublasdx CONFIG
        PATHS "${KERMAC_CUSOLVERDX_MATHDX_ROOT}" NO_DEFAULT_PATH)

    if(mathdx_FOUND)
        add_executable(kermac_cublasdx_gather_syrk_tests test_cublasdx_gather_syrk.cu)
        target_link_libraries(kermac_cublasdx_gather_syrk_tests PRIVATE CUDA::cudart mathdx::cublasdx)
        target_include_directories(kermac_cublasdx_gather_syrk_tests PRIVATE
            "${PROJECT_SOURCE_DIR}/bench"
        )
        set_target_properties(kermac_cublasdx_gather_syrk_tests PROPERTIES
            CUDA_ARCHITECTURES "${KERMAC_CUSOLVERDX_CUDA_ARCHITECTURES}"
        )
        target_compile_features(kermac_cublasdx_gather_syrk_tests PRIVATE cxx_std_17)
        target_compile_options(kermac_cublasdx_gather_syrk_tests PRIVATE
            "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>"
            "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--expt-relaxed-constexpr>"
        )
        foreach(_cuda_arch IN LISTS KERMAC_CUSOLVERDX_CUDA_ARCHITECTURES)
            string(REPLACE "-" ";" _cuda_arch_parts "${_cuda_arch}")
            list(GET _cuda_arch_parts 0 _cuda_arch_num)
            string(REPLACE "a" "" _cuda_arch_num "${_cuda_arch_num}")
            string(REPLACE "f" "" _cuda_arch_num "${_cuda_arch_num}")
            target_compile_definitions(kermac_cublasdx_gather_syrk_tests PRIVATE
                KERMAC_CUBLASDX_ENABLE_SM_${_cuda_arch_num})
        endforeach()
        add_test(NAME kermac.cublasdx_gather_syrk COMMAND kermac_cublasdx_gather_syrk_tests)
        set_tests_properties(kermac.cublasdx_gather_syrk PROPERTIES SKIP_RETURN_CODE 77)
    else()
        message(STATUS "Skipping kermac.cublasdx_gather_syrk: mathdx cublasdx package not found under ${KERMAC_CUSOLVERDX_MATHDX_ROOT}")
    endif()
else()
    message(STATUS "Skipping kermac.cublasdx_gather_syrk: no MathDx SDK found")
endif()

if(Python3_Interpreter_FOUND)
    set(LOW_RANK_DATA_DIR "${CMAKE_CURRENT_BINARY_DIR}/low_rank_data")
    set(LOW_RANK_BANDWIDTH 10.0)
    set(LOW_RANK_SEED 1234)
    set(LOW_RANK_N 64)
    set(LOW_RANK_D 8)

    add_custom_command(
        OUTPUT
            "${LOW_RANK_DATA_DIR}/kernel_matrix.desc"
            "${LOW_RANK_DATA_DIR}/kernel_matrix.dat"
            "${LOW_RANK_DATA_DIR}/solution.desc"
            "${LOW_RANK_DATA_DIR}/solution.dat"
            "${LOW_RANK_DATA_DIR}/features.desc"
            "${LOW_RANK_DATA_DIR}/features.dat"
            "${LOW_RANK_DATA_DIR}/feature_matrix.desc"
            "${LOW_RANK_DATA_DIR}/feature_matrix.dat"
            "${LOW_RANK_DATA_DIR}/feature_matrix_diag.desc"
            "${LOW_RANK_DATA_DIR}/feature_matrix_diag.dat"
        COMMAND ${Python3_EXECUTABLE}
            "${CMAKE_CURRENT_SOURCE_DIR}/scripts/low_rank.py"
            --n ${LOW_RANK_N}
            --d ${LOW_RANK_D}
            --bandwidth ${LOW_RANK_BANDWIDTH}
            --seed ${LOW_RANK_SEED}
            --rng numpy
            --device cpu
            --export-dir "${LOW_RANK_DATA_DIR}"
            --export-only
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/scripts/low_rank.py"
        COMMENT "Generating low-rank AGOP test data"
        VERBATIM
    )

    add_custom_target(kermac_low_rank_data
        DEPENDS
            "${LOW_RANK_DATA_DIR}/kernel_matrix.desc"
            "${LOW_RANK_DATA_DIR}/kernel_matrix.dat"
            "${LOW_RANK_DATA_DIR}/solution.desc"
            "${LOW_RANK_DATA_DIR}/solution.dat"
            "${LOW_RANK_DATA_DIR}/features.desc"
            "${LOW_RANK_DATA_DIR}/features.dat"
            "${LOW_RANK_DATA_DIR}/feature_matrix.desc"
            "${LOW_RANK_DATA_DIR}/feature_matrix.dat"
            "${LOW_RANK_DATA_DIR}/feature_matrix_diag.desc"
            "${LOW_RANK_DATA_DIR}/feature_matrix_diag.dat"
    )

    add_executable(kermac_low_rank_agop_tests test_low_rank_agop.cpp)
    target_link_libraries(kermac_low_rank_agop_tests PRIVATE kermac kermac_cpu)
    add_dependencies(kermac_low_rank_agop_tests kermac_low_rank_data)
    target_compile_definitions(kermac_low_rank_agop_tests PRIVATE
        KERMAC_LOW_RANK_DATA_DIR="${LOW_RANK_DATA_DIR}"
        KERMAC_LOW_RANK_BANDWIDTH=${LOW_RANK_BANDWIDTH}f
    )
    add_test(NAME kermac.low_rank_agop COMMAND kermac_low_rank_agop_tests)
    set_tests_properties(kermac.low_rank_agop PROPERTIES SKIP_RETURN_CODE 77)
else()
    message(STATUS "Python3 not found; skipping low-rank AGOP test.")
endif()
