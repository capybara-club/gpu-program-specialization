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

project(secant_benchmark LANGUAGES C CXX)

option(BUILD_TESTING "Build secant-benchmark tests" ON)

set(
    SECANT_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../secant"
    CACHE PATH
    "Path to the SECANT source tree")

if(NOT EXISTS "${SECANT_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "SECANT_SOURCE_DIR does not contain SECANT")
endif()

if(NOT TARGET secant)
    set(
        SECANT_BUILD_TESTS
        OFF
        CACHE BOOL
        "Build SECANT's own test targets"
        FORCE)
    add_subdirectory(
        "${SECANT_SOURCE_DIR}"
        "${CMAKE_CURRENT_BINARY_DIR}/secant"
        EXCLUDE_FROM_ALL)
endif()

find_package(CUDAToolkit QUIET)
find_package(OpenMP QUIET COMPONENTS C)
find_package(Threads QUIET)

add_library(secant_bench_ast STATIC bench/bench_ast.c)
target_include_directories(
    secant_bench_ast
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/bench")
target_link_libraries(secant_bench_ast PUBLIC secant)
target_compile_features(secant_bench_ast PUBLIC c_std_99)

if(CUDAToolkit_FOUND)
    add_library(
        secant_cuda
        STATIC
            src/cuda/compile.c
            src/cuda/generate.c)
    target_include_directories(
        secant_cuda
        PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_link_libraries(
        secant_cuda
        PUBLIC secant
        PRIVATE CUDA::nvrtc)
    target_compile_features(secant_cuda PUBLIC c_std_99)

    add_library(
        secant_benchmark_cuda_runtime
        STATIC
            tests/support/cuda_runtime.c)
    target_include_directories(
        secant_benchmark_cuda_runtime
        PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/tests/support")
    target_link_libraries(
        secant_benchmark_cuda_runtime
        PUBLIC secant_cuda
        PRIVATE CUDA::cuda_driver)
    target_compile_features(
        secant_benchmark_cuda_runtime
        PUBLIC c_std_99)

    add_executable(secant_generate_cuda app/generate_backend.c)
    target_compile_definitions(
        secant_generate_cuda
        PRIVATE SECANT_GENERATE_CUDA)
    target_link_libraries(
        secant_generate_cuda
        PRIVATE secant_cuda secant_bench_ast)

    add_executable(
        secant_compile_cubin_template
        app/compile_cubin_template.c)
    target_include_directories(
        secant_compile_cubin_template
        PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/bench")
    target_link_libraries(
        secant_compile_cubin_template
        PRIVATE secant_cubin secant_bench_ast CUDA::nvrtc)
    target_compile_features(
        secant_compile_cubin_template
        PRIVATE c_std_99)

    if(TARGET CUDA::nvptxcompiler_static)
        add_library(
            secant_ptx
            STATIC
                src/ptx/compile.c
                src/ptx/generate.c)
        target_include_directories(
            secant_ptx
            PUBLIC
                "${CMAKE_CURRENT_SOURCE_DIR}/include"
                "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty")
        target_link_libraries(
            secant_ptx
            PUBLIC secant
            PRIVATE CUDA::nvrtc CUDA::nvptxcompiler_static m)
        target_compile_features(secant_ptx PUBLIC c_std_99)

        add_executable(secant_generate_ptx app/generate_backend.c)
        target_compile_definitions(
            secant_generate_ptx
            PRIVATE SECANT_GENERATE_PTX)
        target_link_libraries(
            secant_generate_ptx
            PRIVATE secant_ptx secant_bench_ast)
    endif()
endif()

set(
    SECANT_EMBEDDED_CUBIN_SM
    ""
    CACHE STRING
    "SM number for the optional embedded SSE CUBIN benchmark template")
set(
    SECANT_EMBEDDED_CUBIN_KERNELS
    64
    CACHE STRING
    "Kernels in the optional embedded SSE CUBIN template")
set(
    SECANT_EMBEDDED_CUBIN_ASTS_PER_KERNEL
    128
    CACHE STRING
    "AST capacity per kernel in the embedded SSE CUBIN template")
set(
    SECANT_EMBEDDED_CUBIN_INPUTS
    8
    CACHE STRING
    "Inputs in the optional embedded SSE CUBIN template")
set(
    SECANT_EMBEDDED_CUBIN_TARGETS
    1
    CACHE STRING
    "Targets in the optional embedded SSE CUBIN template")
set(
    SECANT_EMBEDDED_CUBIN_TILE_ROWS
    8192
    CACHE STRING
    "Rows per CTA in the optional embedded SSE CUBIN template")
set(
    SECANT_EMBEDDED_CUBIN_THREADS
    128
    CACHE STRING
    "Threads per CTA in the optional embedded SSE CUBIN template")
set(
    SECANT_EMBEDDED_CUBIN_PATCH_INSTRUCTIONS_PER_AST
    64
    CACHE STRING
    "SASS instructions reserved per AST in the embedded CUBIN template")

if(CUDAToolkit_FOUND AND NOT SECANT_EMBEDDED_CUBIN_SM STREQUAL "")
    math(
        EXPR
        SECANT_EMBEDDED_CUBIN_PATCH_CAPACITY
        "${SECANT_EMBEDDED_CUBIN_ASTS_PER_KERNEL} * ${SECANT_EMBEDDED_CUBIN_PATCH_INSTRUCTIONS_PER_AST}")
    set(
        SECANT_EMBEDDED_CUBIN_DIR
        "${CMAKE_CURRENT_BINARY_DIR}/embedded")
    set(
        SECANT_EMBEDDED_CUBIN_SOURCE
        "${SECANT_EMBEDDED_CUBIN_DIR}/secant_sse_template.cu")
    set(
        SECANT_EMBEDDED_CUBIN_BINARY
        "${SECANT_EMBEDDED_CUBIN_DIR}/secant_sse_sm${SECANT_EMBEDDED_CUBIN_SM}_${SECANT_EMBEDDED_CUBIN_KERNELS}x${SECANT_EMBEDDED_CUBIN_ASTS_PER_KERNEL}.cubin")
    add_custom_command(
        OUTPUT "${SECANT_EMBEDDED_CUBIN_SOURCE}"
        COMMAND
            "${CMAKE_COMMAND}" -E make_directory
            "${SECANT_EMBEDDED_CUBIN_DIR}"
        COMMAND
            $<TARGET_FILE:secant_generate_cubin_template>
            static-column-sse
            --inputs "${SECANT_EMBEDDED_CUBIN_INPUTS}"
            --targets "${SECANT_EMBEDDED_CUBIN_TARGETS}"
            --tile-rows "${SECANT_EMBEDDED_CUBIN_TILE_ROWS}"
            --threads "${SECANT_EMBEDDED_CUBIN_THREADS}"
            --patch-instructions-per-ast
            "${SECANT_EMBEDDED_CUBIN_PATCH_INSTRUCTIONS_PER_AST}"
            "${SECANT_EMBEDDED_CUBIN_KERNELS}"
            "${SECANT_EMBEDDED_CUBIN_ASTS_PER_KERNEL}"
            -o "${SECANT_EMBEDDED_CUBIN_SOURCE}"
        DEPENDS secant_generate_cubin_template
        VERBATIM)
    add_custom_command(
        OUTPUT "${SECANT_EMBEDDED_CUBIN_BINARY}"
        COMMAND
            $<TARGET_FILE:secant_compile_cubin_template>
            static-column-sse
            --sm "${SECANT_EMBEDDED_CUBIN_SM}"
            --inputs "${SECANT_EMBEDDED_CUBIN_INPUTS}"
            --targets "${SECANT_EMBEDDED_CUBIN_TARGETS}"
            --tile-rows "${SECANT_EMBEDDED_CUBIN_TILE_ROWS}"
            --threads "${SECANT_EMBEDDED_CUBIN_THREADS}"
            --patch-instructions-per-ast
            "${SECANT_EMBEDDED_CUBIN_PATCH_INSTRUCTIONS_PER_AST}"
            "${SECANT_EMBEDDED_CUBIN_KERNELS}"
            "${SECANT_EMBEDDED_CUBIN_ASTS_PER_KERNEL}"
            -o "${SECANT_EMBEDDED_CUBIN_BINARY}"
        DEPENDS
            secant_compile_cubin_template
            "${SECANT_EMBEDDED_CUBIN_SOURCE}"
        VERBATIM)
    add_custom_target(
        secant_embedded_cubin_binary
        DEPENDS "${SECANT_EMBEDDED_CUBIN_BINARY}")
    add_library(
        secant_bench_cubin_embedded
        STATIC bench/bench_cubin_embedded.c)
    add_dependencies(
        secant_bench_cubin_embedded
        secant_embedded_cubin_binary)
    set_property(
        SOURCE bench/bench_cubin_embedded.c
        APPEND PROPERTY OBJECT_DEPENDS
        "${SECANT_EMBEDDED_CUBIN_BINARY}")
    target_include_directories(
        secant_bench_cubin_embedded
        PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/bench"
        PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty")
    target_compile_definitions(
        secant_bench_cubin_embedded
        PRIVATE
            SECANT_EMBEDDED_CUBIN_PATH="${SECANT_EMBEDDED_CUBIN_BINARY}"
            SECANT_EMBEDDED_CUBIN_SM=${SECANT_EMBEDDED_CUBIN_SM}
            SECANT_EMBEDDED_CUBIN_KERNELS=${SECANT_EMBEDDED_CUBIN_KERNELS}
            SECANT_EMBEDDED_CUBIN_ASTS_PER_KERNEL=${SECANT_EMBEDDED_CUBIN_ASTS_PER_KERNEL}
            SECANT_EMBEDDED_CUBIN_INPUTS=${SECANT_EMBEDDED_CUBIN_INPUTS}
            SECANT_EMBEDDED_CUBIN_TARGETS=${SECANT_EMBEDDED_CUBIN_TARGETS}
            SECANT_EMBEDDED_CUBIN_TILE_ROWS=${SECANT_EMBEDDED_CUBIN_TILE_ROWS}
            SECANT_EMBEDDED_CUBIN_THREADS=${SECANT_EMBEDDED_CUBIN_THREADS}
            SECANT_EMBEDDED_CUBIN_PATCH_CAPACITY=${SECANT_EMBEDDED_CUBIN_PATCH_CAPACITY})
    target_link_libraries(
        secant_bench_cubin_embedded
        PUBLIC secant)
    target_compile_features(
        secant_bench_cubin_embedded
        PRIVATE c_std_99)
endif()

add_executable(secant_generate_cubin app/generate_backend.c)
target_compile_definitions(
    secant_generate_cubin
    PRIVATE SECANT_GENERATE_CUBIN)
target_link_libraries(
    secant_generate_cubin
    PRIVATE secant_cubin secant_bench_ast)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    if(TARGET secant_cuda)
        target_compile_options(secant_cuda PRIVATE -Wall -Wextra)
    endif()
    if(TARGET secant_ptx)
        target_compile_options(secant_ptx PRIVATE -Wall -Wextra)
    endif()
endif()

if(CUDAToolkit_FOUND AND Threads_FOUND AND
   TARGET secant_nvidia_runner_core)
    if(TARGET secant_cuda)
        add_library(
            secant_cuda_runner
            STATIC src/runner/cuda.c)
        target_include_directories(
            secant_cuda_runner
            PRIVATE "${SECANT_SOURCE_DIR}")
        target_link_libraries(
            secant_cuda_runner
            PUBLIC secant_cuda
            PRIVATE secant_nvidia_runner_core CUDA::cuda_driver)
        target_compile_features(secant_cuda_runner PUBLIC c_std_99)
    endif()

    if(TARGET secant_ptx)
        add_library(
            secant_ptx_runner
            STATIC src/runner/ptx.c)
        target_include_directories(
            secant_ptx_runner
            PRIVATE "${SECANT_SOURCE_DIR}")
        target_link_libraries(
            secant_ptx_runner
            PUBLIC secant_ptx
            PRIVATE secant_nvidia_runner_core CUDA::cuda_driver)
        target_compile_features(secant_ptx_runner PUBLIC c_std_99)
    endif()
endif()

function(secant_benchmark_add_generated_example backend shape filename)
    set(source "${CMAKE_CURRENT_SOURCE_DIR}/examples/${backend}/${filename}.cu")
    set(temp "${CMAKE_CURRENT_BINARY_DIR}/examples/${backend}/${filename}.cu")
    set(stamp "${CMAKE_CURRENT_BINARY_DIR}/examples/${backend}/${filename}.stamp")
    add_custom_command(
        OUTPUT "${stamp}"
        COMMAND
            "${CMAKE_COMMAND}" -E make_directory
            "${CMAKE_CURRENT_BINARY_DIR}/examples/${backend}"
        COMMAND
            $<TARGET_FILE:secant_generate_${backend}>
            "${shape}" ${ARGN} -o "${temp}"
        COMMAND
            "${CMAKE_COMMAND}" -E copy_if_different
            "${temp}" "${source}"
        COMMAND "${CMAKE_COMMAND}" -E touch "${stamp}"
        DEPENDS secant_generate_${backend}
        VERBATIM)
    add_custom_target(
        secant_example_${backend}_${filename}
        ALL
        DEPENDS "${stamp}")
endfunction()

if(TARGET secant_generate_cuda)
    secant_benchmark_add_generated_example(
        cuda materialize static_column_materialize_2_2
        --kernels 2 --asts 2 --inputs 8 --ast-mode alu --seed 1)
    secant_benchmark_add_generated_example(
        cuda sse static_column_sse_2_2
        --kernels 2 --asts 2
        --inputs 8 --targets 2 --tile-rows 4096 --threads 128
        --ast-mode alu --seed 1)
    secant_benchmark_add_generated_example(
        cuda dynamic_constant_sse dynamic_constant_sse_2_2
        --kernels 2 --asts 2 --inputs 4 --constants 3
        --targets 2 --tile-rows 128 --threads 128
        --ast-mode simple --seed 1)
endif()
if(TARGET secant_generate_ptx)
    secant_benchmark_add_generated_example(
        ptx materialize static_column_materialize_2_2
        --kernels 2 --asts 2 --inputs 4)
    secant_benchmark_add_generated_example(
        ptx sse static_column_sse_2_2
        --kernels 2 --asts 2 --inputs 4
        --targets 2 --tile-rows 4096 --threads 128)
    secant_benchmark_add_generated_example(
        ptx dynamic_constant_sse dynamic_constant_sse_2_2
        --kernels 2 --asts 2 --inputs 4 --constants 3
        --targets 2 --tile-rows 128 --threads 128)
endif()
if(TARGET secant_generate_cubin)
    secant_benchmark_add_generated_example(
        cubin materialize static_column_materialize_2_2
        --kernels 2 --asts 2 --inputs 4
        --patch-instructions-per-ast 64)
    secant_benchmark_add_generated_example(
        cubin sse static_column_sse_2_2
        --kernels 2 --asts 2 --inputs 4
        --targets 2 --tile-rows 4096 --threads 128
        --patch-instructions-per-ast 64)
    secant_benchmark_add_generated_example(
        cubin dynamic_constant_sse dynamic_constant_sse_2_2
        --kernels 2 --asts 2 --inputs 4 --constants 3
        --targets 2 --tile-rows 128 --threads 128
        --patch-instructions-per-ast 64)
endif()
set(SECANT_BENCH_HAS_CUDA 0)
set(SECANT_BENCH_HAS_PTX 0)
set(SECANT_BENCH_HAS_CUBIN 0)
set(SECANT_BENCH_HAS_EMBEDDED_CUBIN 0)
set(SECANT_BENCH_HAS_OPENMP 0)
set(SECANT_BENCH_LIBRARIES secant_cpu secant_bench_ast)

if(TARGET secant_cuda)
    set(SECANT_BENCH_HAS_CUDA 1)
    list(
        APPEND SECANT_BENCH_LIBRARIES
        secant_cuda secant_benchmark_cuda_runtime CUDA::cuda_driver)
endif()
if(TARGET secant_ptx)
    set(SECANT_BENCH_HAS_PTX 1)
    list(APPEND SECANT_BENCH_LIBRARIES secant_ptx)
endif()
if(CUDAToolkit_FOUND)
    set(SECANT_BENCH_HAS_CUBIN 1)
    list(
        APPEND SECANT_BENCH_LIBRARIES
        secant_cubin secant_benchmark_cuda_runtime
        CUDA::cuda_driver CUDA::nvrtc)
endif()
if(TARGET OpenMP::OpenMP_C)
    set(SECANT_BENCH_HAS_OPENMP 1)
    list(APPEND SECANT_BENCH_LIBRARIES OpenMP::OpenMP_C)
endif()
if(TARGET secant_bench_cubin_embedded)
    set(SECANT_BENCH_HAS_EMBEDDED_CUBIN 1)
    list(
        APPEND SECANT_BENCH_LIBRARIES
        secant_bench_cubin_embedded)
endif()

function(secant_benchmark_configure executable)
    target_link_libraries(
        ${executable}
        PRIVATE ${SECANT_BENCH_LIBRARIES})
    target_compile_features(${executable} PRIVATE c_std_99)
    target_compile_definitions(
        ${executable}
        PRIVATE
            SECANT_BENCH_HAS_CUDA=${SECANT_BENCH_HAS_CUDA}
            SECANT_BENCH_HAS_PTX=${SECANT_BENCH_HAS_PTX}
            SECANT_BENCH_HAS_CUBIN=${SECANT_BENCH_HAS_CUBIN}
            SECANT_BENCH_HAS_EMBEDDED_CUBIN=${SECANT_BENCH_HAS_EMBEDDED_CUBIN}
            SECANT_BENCH_HAS_OPENMP=${SECANT_BENCH_HAS_OPENMP})
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${executable} PRIVATE -Wall -Wextra)
    endif()
endfunction()

if(SECANT_BENCH_HAS_CUDA OR SECANT_BENCH_HAS_PTX OR
   SECANT_BENCH_HAS_CUBIN)
    add_executable(
        secant_dynamic_constant_sse_bench
        bench/dynamic_constant_sse_bench.c)
    secant_benchmark_configure(secant_dynamic_constant_sse_bench)
    target_link_libraries(
        secant_dynamic_constant_sse_bench
        PRIVATE m)

    add_executable(secant_compile_bench bench/compile_bench.c)
    secant_benchmark_configure(secant_compile_bench)

    add_executable(secant_runtime_bench bench/runtime_bench.c)
    secant_benchmark_configure(secant_runtime_bench)

    add_executable(secant_end_to_end_bench bench/runtime_bench.c)
    secant_benchmark_configure(secant_end_to_end_bench)
    target_compile_definitions(
        secant_end_to_end_bench
        PRIVATE SECANT_BENCH_END_TO_END=1)

    if(TARGET OpenMP::OpenMP_C AND Threads_FOUND)
        add_executable(secant_pipeline_bench bench/compile_bench.c)
        secant_benchmark_configure(secant_pipeline_bench)
        target_link_libraries(
            secant_pipeline_bench
            PRIVATE Threads::Threads)
        target_compile_definitions(
            secant_pipeline_bench
            PRIVATE SECANT_BENCH_PIPELINE=1)
        if(TARGET secant_cuda_runner)
            target_link_libraries(
                secant_pipeline_bench PRIVATE secant_cuda_runner)
        endif()
        if(TARGET secant_ptx_runner)
            target_link_libraries(
                secant_pipeline_bench PRIVATE secant_ptx_runner)
        endif()
        if(TARGET secant_cubin_runner)
            target_link_libraries(
                secant_pipeline_bench PRIVATE secant_cubin_runner)
        endif()
    endif()
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64" AND
   CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    add_executable(
        secant_native_avx_bench
        bench/baselines/native_avx.c)
    target_compile_features(secant_native_avx_bench PRIVATE c_std_99)
    target_compile_options(
        secant_native_avx_bench
        PRIVATE -O3 -mavx2 -mfma -Wall -Wextra -Wpedantic)
    target_link_libraries(secant_native_avx_bench PRIVATE m)
    if(TARGET OpenMP::OpenMP_C)
        target_link_libraries(
            secant_native_avx_bench
            PRIVATE OpenMP::OpenMP_C)
    endif()
endif()

include(CTest)
if(BUILD_TESTING)
    find_package(Python3 QUIET COMPONENTS Interpreter)

    add_executable(
        secant_benchmark_ast_test
        tests/bench_ast_test.c)
    target_link_libraries(
        secant_benchmark_ast_test
        PRIVATE secant_bench_ast secant_cpu)
    add_test(
        NAME secant_benchmark_ast_test
        COMMAND secant_benchmark_ast_test)

    if(TARGET secant_cuda)
        add_executable(
            secant_cuda_generator_test
            tests/cuda_generator_test.c)
        target_link_libraries(
            secant_cuda_generator_test
            PRIVATE secant_cuda)
        add_test(
            NAME secant_cuda_generator_test
            COMMAND secant_cuda_generator_test)

        add_executable(
            secant_cuda_dynamic_constant_sse_test
            tests/cuda_dynamic_constant_sse_test.c)
        target_link_libraries(
            secant_cuda_dynamic_constant_sse_test
            PRIVATE
                secant_cpu
                secant_cuda
                secant_benchmark_cuda_runtime
                CUDA::cuda_driver)
        add_test(
            NAME secant_cuda_dynamic_constant_sse_test
            COMMAND secant_cuda_dynamic_constant_sse_test)
        set_tests_properties(
            secant_cuda_dynamic_constant_sse_test
            PROPERTIES
                ENVIRONMENT "CUDA_MODULE_LOADING=EAGER"
                RESOURCE_LOCK secant_nvidia_gpu)

        add_test(
            NAME secant_dynamic_constant_sse_bench_test
            COMMAND
                secant_dynamic_constant_sse_bench
                --backend cuda
                --asts-per-kernel 2
                --settings 4
                --rows 257
                --tile-rows 64
                --threads 128
                --reduction workspace
                --warmups 1
                --iterations 1)
        set_tests_properties(
            secant_dynamic_constant_sse_bench_test
            PROPERTIES
                ENVIRONMENT "CUDA_MODULE_LOADING=EAGER"
                RESOURCE_LOCK secant_nvidia_gpu)
    endif()

    if(TARGET secant_ptx)
        add_executable(
            secant_ptx_generator_test
            tests/ptx_generator_test.c)
        target_link_libraries(
            secant_ptx_generator_test
            PRIVATE secant_ptx)
        add_test(
            NAME secant_ptx_generator_test
            COMMAND secant_ptx_generator_test)

        add_executable(secant_gpu_test tests/gpu_test.c)
        target_link_libraries(
            secant_gpu_test
            PRIVATE
                secant_cpu
                secant_cuda
                secant_ptx
                secant_benchmark_cuda_runtime
                CUDA::cuda_driver)
        if(TARGET secant_cuda_runner)
            target_link_libraries(
                secant_gpu_test
                PRIVATE secant_cuda_runner)
        endif()
        if(TARGET secant_ptx_runner)
            target_link_libraries(
                secant_gpu_test
                PRIVATE secant_ptx_runner)
        endif()
        add_test(NAME secant_gpu_test COMMAND secant_gpu_test)
        set_tests_properties(
            secant_gpu_test
            PROPERTIES
                ENVIRONMENT "CUDA_MODULE_LOADING=EAGER"
                RESOURCE_LOCK secant_nvidia_gpu)

    endif()

    if(TARGET secant_pipeline_bench AND SECANT_BENCH_HAS_CUBIN)
        add_test(
            NAME secant_pipeline_cubin_materialize_test
            COMMAND secant_pipeline_bench
                --backend cubin
                --shape materialize
                --ast-mode alu
                --modules 2
                --workers 2
                --kernels 2
                --asts-per-kernel 4
                --run-rows 4096
                --check-rows 257
                --streams 2
                --tile-rows 1024
                --threads 128
                --target-sm 0
                --opt-level 1)
        add_test(
            NAME secant_pipeline_cubin_sse_test
            COMMAND secant_pipeline_bench
                --backend cubin
                --shape sse
                --ast-mode alu
                --modules 2
                --workers 2
                --kernels 2
                --asts-per-kernel 4
                --run-rows 4096
                --check-rows 257
                --streams 2
                --tile-rows 1024
                --threads 128
                --target-sm 0
                --opt-level 1)
        set_tests_properties(
            secant_pipeline_cubin_materialize_test
            secant_pipeline_cubin_sse_test
            PROPERTIES
                ENVIRONMENT "CUDA_MODULE_LOADING=EAGER"
                RESOURCE_LOCK secant_nvidia_gpu)
    endif()

    if(TARGET secant_dynamic_constant_sse_bench AND
       SECANT_BENCH_HAS_PTX)
        add_test(
            NAME secant_dynamic_constant_sse_ptx_test
            COMMAND secant_dynamic_constant_sse_bench
                --backend ptx
                --asts-per-kernel 2
                --settings 4
                --rows 257
                --tile-rows 64
                --threads 128
                --reduction workspace
                --warmups 1
                --iterations 1)
        set_tests_properties(
            secant_dynamic_constant_sse_ptx_test
            PROPERTIES
                ENVIRONMENT "CUDA_MODULE_LOADING=EAGER"
                RESOURCE_LOCK secant_nvidia_gpu)
    endif()
    if(TARGET secant_dynamic_constant_sse_bench AND
       SECANT_BENCH_HAS_CUBIN)
        add_test(
            NAME secant_dynamic_constant_sse_cubin_test
            COMMAND secant_dynamic_constant_sse_bench
                --backend cubin
                --asts-per-kernel 2
                --settings 4
                --rows 257
                --tile-rows 64
                --threads 128
                --reduction workspace
                --warmups 1
                --iterations 1)
        set_tests_properties(
            secant_dynamic_constant_sse_cubin_test
            PROPERTIES
                ENVIRONMENT "CUDA_MODULE_LOADING=EAGER"
                RESOURCE_LOCK secant_nvidia_gpu)
    endif()
    if(Python3_Interpreter_FOUND)
        add_test(
            NAME secant_compile_matrix_test
            COMMAND
                "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/compile_matrix_test.py"
                "${CMAKE_CURRENT_SOURCE_DIR}/bench/compile_matrix.py"
                "${CMAKE_CURRENT_SOURCE_DIR}/bench/compile_matrix_report.py")
        add_test(
            NAME secant_runtime_sweep_test
            COMMAND
                "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime_sweep_test.py"
                "${CMAKE_CURRENT_SOURCE_DIR}/bench/runtime_tile_sweep.py"
                "${CMAKE_CURRENT_SOURCE_DIR}/bench/runtime_packing_sweep.py"
                "${CMAKE_CURRENT_SOURCE_DIR}/bench/runtime_sse_surface_sweep.py")
    endif()
endif()
