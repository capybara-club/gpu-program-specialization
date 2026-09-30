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
add_executable(kermac_kernel_matrix_bench bench_kernel_matrix.cpp)
target_link_libraries(kermac_kernel_matrix_bench PRIVATE kermac)

add_executable(kermac_solve_bench bench_solve.cpp)
target_link_libraries(kermac_solve_bench PRIVATE kermac)

add_executable(kermac_solve_only_bench bench_solve_only.cpp)
target_link_libraries(kermac_solve_only_bench PRIVATE kermac)

add_executable(kermac_philox_bench bench_philox.cu)
target_link_libraries(kermac_philox_bench PRIVATE CUDA::cudart CUDA::curand)
set_target_properties(kermac_philox_bench PROPERTIES CUDA_ARCHITECTURES native)

set(KERMAC_CUTLASS_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/cutlass/include")
set(KERMAC_CUTE_CUDA_ARCHITECTURES "80-real;120-real" CACHE STRING
    "CUDA architectures for the CuTe gather-SYRK bench and tests")

add_executable(kermac_cute_gather_syrk_bench bench_cute_gather_syrk.cu)
target_link_libraries(kermac_cute_gather_syrk_bench PRIVATE CUDA::cudart CUDA::cublas CUDA::cublasLt)
target_include_directories(kermac_cute_gather_syrk_bench PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${KERMAC_CUTLASS_INCLUDE_DIR}"
)
set_target_properties(kermac_cute_gather_syrk_bench PROPERTIES
    CUDA_ARCHITECTURES "${KERMAC_CUTE_CUDA_ARCHITECTURES}"
)
target_compile_features(kermac_cute_gather_syrk_bench PRIVATE cxx_std_17)
target_compile_options(kermac_cute_gather_syrk_bench PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>"
    "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--expt-relaxed-constexpr>"
)

set(KERMAC_CUSOLVERDX_MATHDX_ROOT "" CACHE PATH
    "Path to a MathDx SDK root containing lib/cmake/mathdx/mathdx-config.cmake")
set(KERMAC_CUSOLVERDX_CUDA_ARCHITECTURES "80-real" CACHE STRING
    "CUDA architectures for the optional cuSolverDx benchmark, for example \"80-real;89-real\"")

if(NOT KERMAC_CUSOLVERDX_MATHDX_ROOT)
    file(GLOB _kermac_mathdx_roots
        "${CMAKE_SOURCE_DIR}/third_party/mathdx/nvidia-mathdx-*/nvidia/mathdx/*")
    list(SORT _kermac_mathdx_roots COMPARE NATURAL ORDER DESCENDING)
    list(LENGTH _kermac_mathdx_roots _kermac_mathdx_root_count)
    if(_kermac_mathdx_root_count GREATER 0)
        list(GET _kermac_mathdx_roots 0 KERMAC_CUSOLVERDX_MATHDX_ROOT)
    endif()
endif()

if(KERMAC_CUSOLVERDX_MATHDX_ROOT)
    set(cusolverdx_CUTLASS_ROOT "${CMAKE_SOURCE_DIR}/mm-ptx/thirdparty/cutlass")
    find_package(mathdx QUIET COMPONENTS cusolverdx CONFIG
        PATHS "${KERMAC_CUSOLVERDX_MATHDX_ROOT}" NO_DEFAULT_PATH)

    if(mathdx_FOUND)
        add_executable(kermac_cusolverdx_posv_bench bench_cusolverdx_posv.cu)
        target_link_libraries(kermac_cusolverdx_posv_bench PRIVATE CUDA::cudart)
        if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
            target_link_libraries(kermac_cusolverdx_posv_bench PRIVATE mathdx::cusolverdx_fatbin)
        else()
            target_link_libraries(kermac_cusolverdx_posv_bench PRIVATE mathdx::cusolverdx)
        endif()
        set_target_properties(kermac_cusolverdx_posv_bench PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            INTERPROCEDURAL_OPTIMIZATION ON
            CUDA_ARCHITECTURES "${KERMAC_CUSOLVERDX_CUDA_ARCHITECTURES}"
        )
        target_compile_features(kermac_cusolverdx_posv_bench PRIVATE cxx_std_17)
        target_compile_options(kermac_cusolverdx_posv_bench PRIVATE
            "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>"
            "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--expt-relaxed-constexpr>"
            "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-rdc=true>"
        )
        target_link_options(kermac_cusolverdx_posv_bench PRIVATE
            $<DEVICE_LINK:-dlto>
        )
        foreach(_cuda_arch IN LISTS KERMAC_CUSOLVERDX_CUDA_ARCHITECTURES)
            string(REPLACE "-" ";" _cuda_arch_parts "${_cuda_arch}")
            list(GET _cuda_arch_parts 0 _cuda_arch_num)
            string(REPLACE "a" "" _cuda_arch_num "${_cuda_arch_num}")
            string(REPLACE "f" "" _cuda_arch_num "${_cuda_arch_num}")
            target_compile_definitions(kermac_cusolverdx_posv_bench PRIVATE
                KERMAC_CUSOLVERDX_ENABLE_SM_${_cuda_arch_num})
            target_compile_options(kermac_cusolverdx_posv_bench PRIVATE
                "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--generate-code arch=compute_${_cuda_arch_num},code=lto_${_cuda_arch_num}>"
            )
        endforeach()
    else()
        message(STATUS "Skipping kermac_cusolverdx_posv_bench: mathdx package not found under ${KERMAC_CUSOLVERDX_MATHDX_ROOT}")
    endif()

    find_package(mathdx QUIET COMPONENTS cublasdx CONFIG
        PATHS "${KERMAC_CUSOLVERDX_MATHDX_ROOT}" NO_DEFAULT_PATH)

    if(mathdx_FOUND)
        add_executable(kermac_cublasdx_gather_syrk_bench bench_cublasdx_gather_syrk.cu)
        target_link_libraries(kermac_cublasdx_gather_syrk_bench PRIVATE CUDA::cudart mathdx::cublasdx)
        target_include_directories(kermac_cublasdx_gather_syrk_bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
        set_target_properties(kermac_cublasdx_gather_syrk_bench PROPERTIES
            CUDA_ARCHITECTURES "${KERMAC_CUSOLVERDX_CUDA_ARCHITECTURES}"
        )
        target_compile_features(kermac_cublasdx_gather_syrk_bench PRIVATE cxx_std_17)
        target_compile_options(kermac_cublasdx_gather_syrk_bench PRIVATE
            "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>"
            "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--expt-relaxed-constexpr>"
        )
        foreach(_cuda_arch IN LISTS KERMAC_CUSOLVERDX_CUDA_ARCHITECTURES)
            string(REPLACE "-" ";" _cuda_arch_parts "${_cuda_arch}")
            list(GET _cuda_arch_parts 0 _cuda_arch_num)
            string(REPLACE "a" "" _cuda_arch_num "${_cuda_arch_num}")
            string(REPLACE "f" "" _cuda_arch_num "${_cuda_arch_num}")
            target_compile_definitions(kermac_cublasdx_gather_syrk_bench PRIVATE
                KERMAC_CUBLASDX_ENABLE_SM_${_cuda_arch_num})
        endforeach()
    else()
        message(STATUS "Skipping kermac_cublasdx_gather_syrk_bench: mathdx package not found under ${KERMAC_CUSOLVERDX_MATHDX_ROOT}")
    endif()
else()
    message(STATUS "Skipping kermac_cusolverdx_posv_bench: no MathDx SDK found")
endif()
