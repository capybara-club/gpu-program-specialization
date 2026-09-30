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
cmake_minimum_required(VERSION 3.20)

project(secant_sindy LANGUAGES C CXX)

option(SECANT_SINDY_BUILD_TESTS "Build Secant-SINDy's tests" ON)
option(SECANT_SINDY_BUILD_BENCHMARKS "Build Secant-SINDy's focused solver benchmark" OFF)
set(SECANT_SINDY_MATHDX_ROOT "" CACHE PATH "MathDx package root used by the optional generated CUDA test")
set(SECANT_SINDY_CUDA_ARCHITECTURE "" CACHE STRING "CUDA architecture used by the optional generated CUDA test")
set(SECANT_SINDY_PYTHON_EXECUTABLE "" CACHE FILEPATH "Python interpreter used by optional NumPy oracle tests")
set(SECANT_SINDY_SECANT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../secant" CACHE PATH "Secant source tree used by integration tests")

add_library(secant_sindy STATIC s_result.c s_cuda_source.c s_cpu.c)
target_include_directories(secant_sindy PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(secant_sindy PUBLIC c_std_99)
target_compile_definitions(
    secant_sindy
    PRIVATE
        SECANT_SINDY_CUDA_TARGET_STATS_SOURCE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/kernels/cuda/target_stats.cu"
        SECANT_SINDY_CUDA_SOLVER_SOURCE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/kernels/cuda/solver.cu")
set_source_files_properties(
    s_cuda_source.c
    PROPERTIES
        OBJECT_DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/cuda/target_stats.cu;${CMAKE_CURRENT_SOURCE_DIR}/kernels/cuda/solver.cu")
target_link_libraries(secant_sindy PUBLIC m)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(secant_sindy PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(secant_sindy_generate_cuda app/generate_cuda.c)
target_link_libraries(secant_sindy_generate_cuda PRIVATE secant_sindy)
target_compile_features(secant_sindy_generate_cuda PRIVATE c_std_99)

include(CTest)
if(BUILD_TESTING AND SECANT_SINDY_BUILD_TESTS)
    add_executable(secant_sindy_cpu_test tests/cpu_test.c)
    target_link_libraries(secant_sindy_cpu_test PRIVATE secant_sindy)
    add_test(NAME secant_sindy_cpu_test COMMAND secant_sindy_cpu_test)

    add_executable(secant_sindy_stlsq_edge_test tests/stlsq_edge_test.c)
    target_link_libraries(secant_sindy_stlsq_edge_test PRIVATE secant_sindy)
    add_test(NAME secant_sindy_stlsq_edge_test COMMAND secant_sindy_stlsq_edge_test)

    add_executable(secant_sindy_cpu_oracle_driver tests/cpu_oracle_driver.c)
    target_link_libraries(secant_sindy_cpu_oracle_driver PRIVATE secant_sindy)

    if(EXISTS "${SECANT_SINDY_SECANT_ROOT}/CMakeLists.txt")
        if(NOT TARGET secant_cpu)
            set(SECANT_BUILD_TESTS OFF)
            set(SECANT_BUILD_BENCHMARKS OFF)
            set(SECANT_BUILD_PYTHON_BINDINGS OFF)
            add_subdirectory("${SECANT_SINDY_SECANT_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}/secant" EXCLUDE_FROM_ALL)
        endif()
        add_executable(secant_sindy_secant_integration_test tests/secant_integration_test.c)
        target_link_libraries(secant_sindy_secant_integration_test PRIVATE secant_sindy secant_cpu)
        add_test(NAME secant_sindy_secant_integration_test COMMAND secant_sindy_secant_integration_test)
    endif()

    if(NOT SECANT_SINDY_PYTHON_EXECUTABLE AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python")
        set(SECANT_SINDY_PYTHON_EXECUTABLE "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python")
    endif()
    if(SECANT_SINDY_PYTHON_EXECUTABLE)
        execute_process(
            COMMAND "${SECANT_SINDY_PYTHON_EXECUTABLE}" -c "import numpy"
            RESULT_VARIABLE SECANT_SINDY_NUMPY_IMPORT_RESULT
            OUTPUT_QUIET
            ERROR_QUIET)
        if(SECANT_SINDY_NUMPY_IMPORT_RESULT EQUAL 0)
            add_test(
                NAME secant_sindy_numpy_oracle_test
                COMMAND "${SECANT_SINDY_PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/numpy_oracle_test.py"
                    --driver $<TARGET_FILE:secant_sindy_cpu_oracle_driver>)
        endif()
    endif()

    add_executable(secant_sindy_source_test tests/source_test.c)
    target_link_libraries(secant_sindy_source_test PRIVATE secant_sindy)
    add_test(NAME secant_sindy_source_test COMMAND secant_sindy_source_test)

    add_executable(secant_sindy_cpp_header_test tests/cpp_header_test.cpp)
    target_link_libraries(secant_sindy_cpp_header_test PRIVATE secant_sindy)
    target_compile_features(secant_sindy_cpp_header_test PRIVATE cxx_std_17)
    add_test(NAME secant_sindy_cpp_header_test COMMAND secant_sindy_cpp_header_test)

    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(secant_sindy_cpu_test PRIVATE -Wall -Wextra -Werror -pedantic-errors)
        target_compile_options(secant_sindy_stlsq_edge_test PRIVATE -Wall -Wextra -Werror -pedantic-errors)
        target_compile_options(secant_sindy_cpu_oracle_driver PRIVATE -Wall -Wextra -Werror -pedantic-errors)
        if(TARGET secant_sindy_secant_integration_test)
            target_compile_options(secant_sindy_secant_integration_test PRIVATE -Wall -Wextra -Werror -pedantic-errors)
        endif()
        target_compile_options(secant_sindy_source_test PRIVATE -Wall -Wextra -Werror -pedantic-errors)
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(secant_sindy_cpp_header_test PRIVATE -Wall -Wextra -Werror -pedantic-errors)
    endif()

    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND AND SECANT_SINDY_MATHDX_ROOT)
        if(NOT SECANT_SINDY_CUDA_ARCHITECTURE MATCHES "^[0-9]+$")
            message(FATAL_ERROR
                "Set SECANT_SINDY_CUDA_ARCHITECTURE explicitly when enabling the generated CUDA test")
        endif()
        set(SECANT_SINDY_MATHDX_INCLUDE "${SECANT_SINDY_MATHDX_ROOT}/include")
        set(SECANT_SINDY_CUTLASS_INCLUDE "${SECANT_SINDY_MATHDX_ROOT}/external/cutlass/include")
        set(SECANT_SINDY_CUSOLVERDX_FATBIN "${SECANT_SINDY_MATHDX_ROOT}/lib/libcusolverdx.fatbin")
        if(NOT EXISTS "${SECANT_SINDY_MATHDX_INCLUDE}/cusolverdx.hpp" OR
           NOT EXISTS "${SECANT_SINDY_CUTLASS_INCLUDE}/cutlass/numeric_types.h" OR
           NOT EXISTS "${SECANT_SINDY_CUSOLVERDX_FATBIN}")
            message(FATAL_ERROR "SECANT_SINDY_MATHDX_ROOT does not contain a complete cuSolverDx package")
        endif()

        set(SECANT_SINDY_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated_cuda_test")
        set(SECANT_SINDY_GENERATED_SOLVER "${SECANT_SINDY_GENERATED_DIR}/solver.cu")
        set(SECANT_SINDY_GENERATED_TARGET_STATS "${SECANT_SINDY_GENERATED_DIR}/target_stats.cu")
        set(SECANT_SINDY_GENERATED_SOLVER_OBJECT "${SECANT_SINDY_GENERATED_DIR}/solver.o")
        set(SECANT_SINDY_GENERATED_TARGET_STATS_OBJECT "${SECANT_SINDY_GENERATED_DIR}/target_stats.o")
        set(SECANT_SINDY_GENERATED_FATBIN "${SECANT_SINDY_GENERATED_DIR}/secant_sindy_test.fatbin")
        set(SECANT_SINDY_BENCHMARK_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated_cuda_bench")
        set(SECANT_SINDY_BENCHMARK_SOLVER "${SECANT_SINDY_BENCHMARK_DIR}/solver.cu")
        set(SECANT_SINDY_BENCHMARK_SOLVER_OBJECT "${SECANT_SINDY_BENCHMARK_DIR}/solver.o")
        set(SECANT_SINDY_BENCHMARK_FATBIN "${SECANT_SINDY_BENCHMARK_DIR}/secant_sindy_bench.fatbin")
        math(EXPR SECANT_SINDY_TEST_CC_MAJOR "${SECANT_SINDY_CUDA_ARCHITECTURE} / 10")
        math(EXPR SECANT_SINDY_TEST_CC_MINOR "${SECANT_SINDY_CUDA_ARCHITECTURE} % 10")

        add_custom_command(
            OUTPUT "${SECANT_SINDY_GENERATED_SOLVER}" "${SECANT_SINDY_GENERATED_TARGET_STATS}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${SECANT_SINDY_GENERATED_DIR}"
            COMMAND $<TARGET_FILE:secant_sindy_generate_cuda> solver
                "${SECANT_SINDY_TEST_CC_MAJOR}" "${SECANT_SINDY_TEST_CC_MINOR}" 4 2 4
                -o "${SECANT_SINDY_GENERATED_SOLVER}"
            COMMAND $<TARGET_FILE:secant_sindy_generate_cuda> target-stats
                -o "${SECANT_SINDY_GENERATED_TARGET_STATS}"
            DEPENDS secant_sindy_generate_cuda
            VERBATIM)
        add_custom_command(
            OUTPUT "${SECANT_SINDY_GENERATED_SOLVER_OBJECT}" "${SECANT_SINDY_GENERATED_TARGET_STATS_OBJECT}"
            COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" -std=c++17 -O3 -dlto -dc
                "-arch=sm_${SECANT_SINDY_CUDA_ARCHITECTURE}"
                "-I${SECANT_SINDY_MATHDX_INCLUDE}"
                "-I${SECANT_SINDY_CUTLASS_INCLUDE}"
                -c "${SECANT_SINDY_GENERATED_SOLVER}" -o "${SECANT_SINDY_GENERATED_SOLVER_OBJECT}"
            COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" -std=c++17 -O3 -dlto -dc
                "-arch=sm_${SECANT_SINDY_CUDA_ARCHITECTURE}"
                -c "${SECANT_SINDY_GENERATED_TARGET_STATS}" -o "${SECANT_SINDY_GENERATED_TARGET_STATS_OBJECT}"
            DEPENDS "${SECANT_SINDY_GENERATED_SOLVER}" "${SECANT_SINDY_GENERATED_TARGET_STATS}"
            VERBATIM)
        add_custom_command(
            OUTPUT "${SECANT_SINDY_GENERATED_FATBIN}"
            COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" -dlto -dlink -fatbin
                "-arch=sm_${SECANT_SINDY_CUDA_ARCHITECTURE}"
                "${SECANT_SINDY_GENERATED_SOLVER_OBJECT}"
                "${SECANT_SINDY_GENERATED_TARGET_STATS_OBJECT}"
                "${SECANT_SINDY_CUSOLVERDX_FATBIN}"
                -o "${SECANT_SINDY_GENERATED_FATBIN}"
            DEPENDS "${SECANT_SINDY_GENERATED_SOLVER_OBJECT}" "${SECANT_SINDY_GENERATED_TARGET_STATS_OBJECT}"
            VERBATIM)
        add_custom_target(secant_sindy_generated_cuda_test_fatbin DEPENDS "${SECANT_SINDY_GENERATED_FATBIN}")

        add_executable(secant_sindy_cuda_test tests/cuda_test.c)
        target_link_libraries(
            secant_sindy_cuda_test
            PRIVATE secant_sindy secant_cubin_runner CUDA::cuda_driver CUDA::nvrtc)
        add_dependencies(secant_sindy_cuda_test secant_sindy_generated_cuda_test_fatbin)
        add_test(NAME secant_sindy_cuda_test COMMAND secant_sindy_cuda_test "${SECANT_SINDY_GENERATED_FATBIN}")
        set_tests_properties(
            secant_sindy_cuda_test
            PROPERTIES
                ENVIRONMENT "CUDA_MODULE_LOADING=EAGER"
                RESOURCE_LOCK secant_sindy_nvidia_gpu
                TIMEOUT 120)

        if(SECANT_SINDY_BUILD_BENCHMARKS)
            add_custom_command(
                OUTPUT "${SECANT_SINDY_BENCHMARK_SOLVER}"
                COMMAND "${CMAKE_COMMAND}" -E make_directory "${SECANT_SINDY_BENCHMARK_DIR}"
                COMMAND $<TARGET_FILE:secant_sindy_generate_cuda> solver
                    "${SECANT_SINDY_TEST_CC_MAJOR}" "${SECANT_SINDY_TEST_CC_MINOR}" 32 4 32
                    -o "${SECANT_SINDY_BENCHMARK_SOLVER}"
                DEPENDS secant_sindy_generate_cuda
                VERBATIM)
            add_custom_command(
                OUTPUT "${SECANT_SINDY_BENCHMARK_SOLVER_OBJECT}"
                COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" -std=c++17 -O3 -dlto -dc
                    "-arch=sm_${SECANT_SINDY_CUDA_ARCHITECTURE}"
                    "-I${SECANT_SINDY_MATHDX_INCLUDE}"
                    "-I${SECANT_SINDY_CUTLASS_INCLUDE}"
                    -c "${SECANT_SINDY_BENCHMARK_SOLVER}" -o "${SECANT_SINDY_BENCHMARK_SOLVER_OBJECT}"
                DEPENDS "${SECANT_SINDY_BENCHMARK_SOLVER}"
                VERBATIM)
            add_custom_command(
                OUTPUT "${SECANT_SINDY_BENCHMARK_FATBIN}"
                COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" -dlto -dlink -fatbin
                    "-arch=sm_${SECANT_SINDY_CUDA_ARCHITECTURE}"
                    "${SECANT_SINDY_BENCHMARK_SOLVER_OBJECT}"
                    "${SECANT_SINDY_CUSOLVERDX_FATBIN}"
                    -o "${SECANT_SINDY_BENCHMARK_FATBIN}"
                DEPENDS "${SECANT_SINDY_BENCHMARK_SOLVER_OBJECT}"
                VERBATIM)
            add_custom_target(secant_sindy_solver_benchmark_fatbin DEPENDS "${SECANT_SINDY_BENCHMARK_FATBIN}")
            add_executable(secant_sindy_solver_bench bench/solver_bench.c)
            target_link_libraries(secant_sindy_solver_bench PRIVATE CUDA::cuda_driver)
            add_dependencies(secant_sindy_solver_bench secant_sindy_solver_benchmark_fatbin)
            add_executable(secant_sindy_cuda_capacity32_test tests/cuda_capacity32_test.c)
            target_link_libraries(secant_sindy_cuda_capacity32_test PRIVATE secant_sindy CUDA::cuda_driver)
            add_dependencies(secant_sindy_cuda_capacity32_test secant_sindy_solver_benchmark_fatbin)
            add_test(
                NAME secant_sindy_cuda_capacity32_test
                COMMAND secant_sindy_cuda_capacity32_test "${SECANT_SINDY_BENCHMARK_FATBIN}")
            add_test(
                NAME secant_sindy_solver_bench_smoke
                COMMAND secant_sindy_solver_bench "${SECANT_SINDY_BENCHMARK_FATBIN}" stlsq 8 2 2 2 1 4)
            set_tests_properties(
                secant_sindy_cuda_capacity32_test secant_sindy_solver_bench_smoke
                PROPERTIES RESOURCE_LOCK secant_sindy_nvidia_gpu TIMEOUT 120)
        endif()
    endif()
endif()
