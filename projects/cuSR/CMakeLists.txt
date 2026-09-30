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
cmake_minimum_required(VERSION 3.22)

project(cuSR C CXX CUDA)

set(CUSR_SASS_GPU_NAME "sm_120" CACHE STRING "GPU name used for generated CUDA-to-SASS cubins")
string(REGEX REPLACE "^sm_" "" CUSR_CMAKE_CUDA_ARCHITECTURE "${CUSR_SASS_GPU_NAME}")

find_package(CUDAToolkit REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
enable_testing()

option(CUSR_BUILD_PYTHON_BINDINGS "Build the nanobind Python extension" OFF)

set(CUSR_TILE_STATIC_MSE_EMBEDDED_DIR "${CMAKE_CURRENT_BINARY_DIR}/embedded")
set(CUSR_TILE_STATIC_MSE_8K_CUBIN "${CUSR_TILE_STATIC_MSE_EMBEDDED_DIR}/cusr_tile_static_mse_8k.cubin")
set(CUSR_TILE_STATIC_MSE_32K_CUBIN "${CUSR_TILE_STATIC_MSE_EMBEDDED_DIR}/cusr_tile_static_mse_32k.cubin")
set(CUSR_TILE_STATIC_MSE_128K_CUBIN "${CUSR_TILE_STATIC_MSE_EMBEDDED_DIR}/cusr_tile_static_mse_128k.cubin")

add_custom_command(
    OUTPUT "${CUSR_TILE_STATIC_MSE_8K_CUBIN}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CUSR_TILE_STATIC_MSE_EMBEDDED_DIR}"
    COMMAND
        ${CMAKE_CUDA_COMPILER}
        -O3
        --std=c++11
        --gpu-architecture=${CUSR_SASS_GPU_NAME}
        -Xptxas=--opt-level=1,-warn-spills,-Werror
        -cubin
        -I"${CMAKE_CURRENT_SOURCE_DIR}/kernels"
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/tile_static_mse_template/instantiations/cusr_tile_static_mse_8k.cu"
        -o "${CUSR_TILE_STATIC_MSE_8K_CUBIN}"
    DEPENDS
        kernels/tile_static_mse_template/cusr_tile_static_mse_template.cuh
        kernels/tile_static_mse_template/instantiations/cusr_tile_static_mse_8k.cu
    VERBATIM
)

add_custom_command(
    OUTPUT "${CUSR_TILE_STATIC_MSE_32K_CUBIN}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CUSR_TILE_STATIC_MSE_EMBEDDED_DIR}"
    COMMAND
        ${CMAKE_CUDA_COMPILER}
        -O3
        --std=c++11
        --gpu-architecture=${CUSR_SASS_GPU_NAME}
        -Xptxas=--opt-level=1,-warn-spills,-Werror
        -cubin
        -I"${CMAKE_CURRENT_SOURCE_DIR}/kernels"
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/tile_static_mse_template/instantiations/cusr_tile_static_mse_32k.cu"
        -o "${CUSR_TILE_STATIC_MSE_32K_CUBIN}"
    DEPENDS
        kernels/tile_static_mse_template/cusr_tile_static_mse_template.cuh
        kernels/tile_static_mse_template/instantiations/cusr_tile_static_mse_32k.cu
    VERBATIM
)

add_custom_command(
    OUTPUT "${CUSR_TILE_STATIC_MSE_128K_CUBIN}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CUSR_TILE_STATIC_MSE_EMBEDDED_DIR}"
    COMMAND
        ${CMAKE_CUDA_COMPILER}
        -O3
        --std=c++11
        --gpu-architecture=${CUSR_SASS_GPU_NAME}
        -Xptxas=--opt-level=1,-warn-spills,-Werror
        -cubin
        -I"${CMAKE_CURRENT_SOURCE_DIR}/kernels"
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/tile_static_mse_template/instantiations/cusr_tile_static_mse_128k.cu"
        -o "${CUSR_TILE_STATIC_MSE_128K_CUBIN}"
    DEPENDS
        kernels/tile_static_mse_template/cusr_tile_static_mse_template.cuh
        kernels/tile_static_mse_template/instantiations/cusr_tile_static_mse_128k.cu
    VERBATIM
)

add_custom_target(cusr_tile_static_mse_cubins DEPENDS
    "${CUSR_TILE_STATIC_MSE_8K_CUBIN}"
    "${CUSR_TILE_STATIC_MSE_32K_CUBIN}"
    "${CUSR_TILE_STATIC_MSE_128K_CUBIN}"
)

add_library(cusr_tile_static_mse_embedded STATIC
    src/cusr_tile_static_mse_embedded.c
)

add_dependencies(cusr_tile_static_mse_embedded cusr_tile_static_mse_cubins)
set_property(SOURCE src/cusr_tile_static_mse_embedded.c APPEND PROPERTY OBJECT_DEPENDS
    "${CUSR_TILE_STATIC_MSE_8K_CUBIN}"
    "${CUSR_TILE_STATIC_MSE_32K_CUBIN}"
    "${CUSR_TILE_STATIC_MSE_128K_CUBIN}"
)

target_include_directories(cusr_tile_static_mse_embedded
    PUBLIC src
    PRIVATE thirdparty
)

target_compile_definitions(cusr_tile_static_mse_embedded PRIVATE
    CUSR_TILE_STATIC_MSE_8K_CUBIN_PATH="${CUSR_TILE_STATIC_MSE_8K_CUBIN}"
    CUSR_TILE_STATIC_MSE_32K_CUBIN_PATH="${CUSR_TILE_STATIC_MSE_32K_CUBIN}"
    CUSR_TILE_STATIC_MSE_128K_CUBIN_PATH="${CUSR_TILE_STATIC_MSE_128K_CUBIN}"
    CUSR_TILE_STATIC_MSE_EMBEDDED_SASS_ARCH=${CUSR_CMAKE_CUDA_ARCHITECTURE}
)

target_compile_features(cusr_tile_static_mse_embedded PRIVATE c_std_99)

add_library(cusr_tile_static_mse_nvrtc STATIC
    kernels/tile_static_mse_template/cusr_tile_static_mse_nvrtc.c
)

target_include_directories(cusr_tile_static_mse_nvrtc
    PUBLIC kernels/tile_static_mse_template
    PRIVATE thirdparty
)

target_compile_definitions(cusr_tile_static_mse_nvrtc PRIVATE
    CUSR_TILE_STATIC_MSE_TEMPLATE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/kernels/tile_static_mse_template/cusr_tile_static_mse_template.cuh"
)

set_property(SOURCE kernels/tile_static_mse_template/cusr_tile_static_mse_nvrtc.c APPEND PROPERTY OBJECT_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/tile_static_mse_template/cusr_tile_static_mse_template.cuh"
)

target_compile_features(cusr_tile_static_mse_nvrtc PRIVATE c_std_99)
target_link_libraries(cusr_tile_static_mse_nvrtc PUBLIC CUDA::nvrtc)

set(CUSR_TILE_STATIC_EVAL_EMBEDDED_DIR "${CMAKE_CURRENT_BINARY_DIR}/embedded")
set(CUSR_TILE_STATIC_EVAL_8K_CUBIN "${CUSR_TILE_STATIC_EVAL_EMBEDDED_DIR}/cusr_tile_static_eval_8k.cubin")
set(CUSR_TILE_STATIC_EVAL_32K_CUBIN "${CUSR_TILE_STATIC_EVAL_EMBEDDED_DIR}/cusr_tile_static_eval_32k.cubin")

add_custom_command(
    OUTPUT "${CUSR_TILE_STATIC_EVAL_8K_CUBIN}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CUSR_TILE_STATIC_EVAL_EMBEDDED_DIR}"
    COMMAND
        ${CMAKE_CUDA_COMPILER}
        -O3
        --std=c++11
        --gpu-architecture=${CUSR_SASS_GPU_NAME}
        -Xptxas=--opt-level=1,-warn-spills,-Werror
        -cubin
        -I"${CMAKE_CURRENT_SOURCE_DIR}/kernels"
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/tile_static_eval_template/instantiations/cusr_tile_static_eval_8k.cu"
        -o "${CUSR_TILE_STATIC_EVAL_8K_CUBIN}"
    DEPENDS
        kernels/tile_static_eval_template/cusr_tile_static_eval_template.cuh
        kernels/tile_static_eval_template/instantiations/cusr_tile_static_eval_8k.cu
    VERBATIM
)

add_custom_command(
    OUTPUT "${CUSR_TILE_STATIC_EVAL_32K_CUBIN}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CUSR_TILE_STATIC_EVAL_EMBEDDED_DIR}"
    COMMAND
        ${CMAKE_CUDA_COMPILER}
        -O3
        --std=c++11
        --gpu-architecture=${CUSR_SASS_GPU_NAME}
        -Xptxas=--opt-level=1,-warn-spills,-Werror
        -cubin
        -I"${CMAKE_CURRENT_SOURCE_DIR}/kernels"
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/tile_static_eval_template/instantiations/cusr_tile_static_eval_32k.cu"
        -o "${CUSR_TILE_STATIC_EVAL_32K_CUBIN}"
    DEPENDS
        kernels/tile_static_eval_template/cusr_tile_static_eval_template.cuh
        kernels/tile_static_eval_template/instantiations/cusr_tile_static_eval_32k.cu
    VERBATIM
)

add_custom_target(cusr_tile_static_eval_cubins DEPENDS
    "${CUSR_TILE_STATIC_EVAL_8K_CUBIN}"
    "${CUSR_TILE_STATIC_EVAL_32K_CUBIN}"
)

add_library(cusr_tile_static_eval_embedded STATIC
    src/cusr_tile_static_eval_embedded.c
)

add_dependencies(cusr_tile_static_eval_embedded cusr_tile_static_eval_cubins)
set_property(SOURCE src/cusr_tile_static_eval_embedded.c APPEND PROPERTY OBJECT_DEPENDS
    "${CUSR_TILE_STATIC_EVAL_8K_CUBIN}"
    "${CUSR_TILE_STATIC_EVAL_32K_CUBIN}"
)

target_include_directories(cusr_tile_static_eval_embedded
    PUBLIC src
    PRIVATE thirdparty
)

target_compile_definitions(cusr_tile_static_eval_embedded PRIVATE
    CUSR_TILE_STATIC_EVAL_8K_CUBIN_PATH="${CUSR_TILE_STATIC_EVAL_8K_CUBIN}"
    CUSR_TILE_STATIC_EVAL_32K_CUBIN_PATH="${CUSR_TILE_STATIC_EVAL_32K_CUBIN}"
    CUSR_TILE_STATIC_EVAL_EMBEDDED_SASS_ARCH=${CUSR_CMAKE_CUDA_ARCHITECTURE}
)

target_compile_features(cusr_tile_static_eval_embedded PRIVATE c_std_99)

add_library(cusr_tile_static_eval_nvrtc STATIC
    kernels/tile_static_eval_template/cusr_tile_static_eval_nvrtc.c
)

target_include_directories(cusr_tile_static_eval_nvrtc
    PUBLIC kernels/tile_static_eval_template
    PRIVATE thirdparty
)

target_compile_definitions(cusr_tile_static_eval_nvrtc PRIVATE
    CUSR_TILE_STATIC_EVAL_TEMPLATE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/kernels/tile_static_eval_template/cusr_tile_static_eval_template.cuh"
)

set_property(SOURCE kernels/tile_static_eval_template/cusr_tile_static_eval_nvrtc.c APPEND PROPERTY OBJECT_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/tile_static_eval_template/cusr_tile_static_eval_template.cuh"
)

target_compile_features(cusr_tile_static_eval_nvrtc PRIVATE c_std_99)
target_link_libraries(cusr_tile_static_eval_nvrtc PUBLIC CUDA::nvrtc)

add_executable(cusr_generate_native_cuda_reference
    bench/native_cuda_reference/cusr_generate_native_cuda_reference.c
)

target_include_directories(cusr_generate_native_cuda_reference PRIVATE
    bench/native_cuda_reference
)

target_compile_features(cusr_generate_native_cuda_reference PRIVATE c_std_99)

add_executable(cusr_inspect_sites_cubin
    tools/cusr_inspect_sites_cubin.c
)

target_include_directories(cusr_inspect_sites_cubin PRIVATE
    src
)

target_compile_features(cusr_inspect_sites_cubin PRIVATE c_std_99)

add_executable(cusr_ast_sass_demo
    app/ast_sass_demo.c
)

target_include_directories(cusr_ast_sass_demo PRIVATE
    src
)

target_compile_features(cusr_ast_sass_demo PRIVATE c_std_99)

target_link_libraries(cusr_ast_sass_demo PRIVATE
    m
)

add_executable(cusr_ast_sass_patch_demo
    app/ast_sass_patch_demo.c
)

target_include_directories(cusr_ast_sass_patch_demo PRIVATE
    src
)

target_compile_features(cusr_ast_sass_patch_demo PRIVATE c_std_99)

add_executable(cusr_ast_sass_cpp_check
    app/ast_sass_cpp_check.cpp
)

add_executable(cusr_ast_sass_contracts
    tests/cusr_ast_sass_contracts.c
)

add_executable(cusr_tile_static_eval_nvrtc_smoke
    tests/cusr_tile_static_eval_nvrtc_smoke.c
)

target_compile_features(cusr_tile_static_eval_nvrtc_smoke PRIVATE c_std_99)
target_link_libraries(cusr_tile_static_eval_nvrtc_smoke PRIVATE
    CUDA::cuda_driver
    cusr_tile_static_eval_nvrtc
)

target_include_directories(cusr_ast_sass_contracts PRIVATE
    src
)

target_compile_features(cusr_ast_sass_contracts PRIVATE c_std_99)

target_include_directories(cusr_ast_sass_cpp_check PRIVATE
    src
)

target_compile_features(cusr_ast_sass_cpp_check PRIVATE cxx_std_20)

add_executable(cusr_nvrtc_ast_sass_gpu
    tests/cusr_nvrtc_ast_sass_gpu.c
)

add_executable(cusr_bench
    bench/cusr_bench.c
)

add_executable(cusr_ast_settings_experiment
    experiments/ast_settings/cusr_ast_settings_bench.cu
)

target_include_directories(cusr_ast_settings_experiment PRIVATE
    experiments/ast_settings
)

target_compile_features(cusr_ast_settings_experiment PRIVATE
    cxx_std_17
    cuda_std_17
)

set_target_properties(cusr_ast_settings_experiment PROPERTIES
    CUDA_ARCHITECTURES "${CUSR_CMAKE_CUDA_ARCHITECTURE}"
)

target_compile_options(cusr_ast_settings_experiment PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Werror>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xptxas=-warn-spills>
)

target_link_libraries(cusr_ast_settings_experiment PRIVATE
    CUDA::cudart
)

add_test(
    NAME cusr_ast_settings_experiment_smoke
    COMMAND
        $<TARGET_FILE:cusr_ast_settings_experiment>
        --asts 4
        --expr all
        --settings 128
        --rows 1024
        --iters 1
)

target_include_directories(cusr_nvrtc_ast_sass_gpu PRIVATE
    kernels
    src
)

target_include_directories(cusr_bench PRIVATE
    bench/native_cuda_reference
    kernels
    src
)

target_compile_definitions(cusr_bench PRIVATE
    CUSR_BENCH_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

target_compile_features(cusr_nvrtc_ast_sass_gpu PRIVATE c_std_99)
target_compile_features(cusr_bench PRIVATE c_std_99)

if (MSVC)
    target_compile_options(cusr_tile_static_eval_embedded PRIVATE /W4 /WX)
    target_compile_options(cusr_tile_static_eval_nvrtc PRIVATE /W4 /WX)
    target_compile_options(cusr_tile_static_mse_embedded PRIVATE /W4 /WX)
    target_compile_options(cusr_tile_static_mse_nvrtc PRIVATE /W4 /WX)
    target_compile_options(cusr_generate_native_cuda_reference PRIVATE /W4 /WX)
    target_compile_options(cusr_inspect_sites_cubin PRIVATE /W4 /WX)
    target_compile_options(cusr_ast_sass_demo PRIVATE /W4 /WX)
    target_compile_options(cusr_ast_sass_patch_demo PRIVATE /W4 /WX)
    target_compile_options(cusr_ast_sass_cpp_check PRIVATE /W4 /WX)
    target_compile_options(cusr_ast_sass_contracts PRIVATE /W4 /WX)
    target_compile_options(cusr_tile_static_eval_nvrtc_smoke PRIVATE /W4 /WX)
    target_compile_options(cusr_nvrtc_ast_sass_gpu PRIVATE /W4 /WX)
    target_compile_options(cusr_bench PRIVATE /W4 /WX)
else()
    target_compile_options(cusr_tile_static_eval_embedded PRIVATE -Wall -Wextra -Werror)
    target_compile_options(cusr_tile_static_eval_nvrtc PRIVATE -Wall -Wextra -Werror)
    target_compile_options(cusr_tile_static_mse_embedded PRIVATE -Wall -Wextra -Werror)
    target_compile_options(cusr_tile_static_mse_nvrtc PRIVATE -Wall -Wextra -Werror)
    target_compile_options(cusr_generate_native_cuda_reference PRIVATE -Wall -Wextra -Werror)
    target_compile_options(cusr_inspect_sites_cubin PRIVATE -Wall -Wextra -Werror)
    target_compile_options(cusr_ast_sass_demo PRIVATE -Wall -Wextra -Werror)
    target_compile_options(cusr_ast_sass_patch_demo PRIVATE -Wall -Wextra -Werror)
    target_compile_options(cusr_ast_sass_cpp_check PRIVATE -Wall -Wextra -Werror)
    target_compile_options(cusr_ast_sass_contracts PRIVATE -Wall -Wextra -Werror)
    target_compile_options(cusr_tile_static_eval_nvrtc_smoke PRIVATE -Wall -Wextra -Werror)
    target_compile_options(cusr_nvrtc_ast_sass_gpu PRIVATE -Wall -Wextra -Werror)
    target_compile_options(cusr_bench PRIVATE -Wall -Wextra -Werror)
endif()

target_link_libraries(cusr_nvrtc_ast_sass_gpu PRIVATE
    CUDA::cuda_driver
    cusr_tile_static_mse_nvrtc
    m
)

target_link_libraries(cusr_ast_sass_cpp_check PRIVATE
    CUDA::cuda_driver
    cusr_tile_static_mse_embedded
    m
)

target_link_libraries(cusr_bench PRIVATE
    CUDA::cuda_driver
    CUDA::nvrtc
    cusr_tile_static_mse_embedded
    cusr_tile_static_mse_nvrtc
    m
)

add_test(
    NAME cusr_nvrtc_ast_sass_compare
    COMMAND
        ${Python3_EXECUTABLE}
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/compare_nvrtc_ast_sass.py"
        --producer $<TARGET_FILE:cusr_nvrtc_ast_sass_gpu>
        --out-dir "${CMAKE_CURRENT_BINARY_DIR}/nvrtc_ast_sass_compare"
)

add_test(
    NAME cusr_ast_sass_cpp_check
    COMMAND $<TARGET_FILE:cusr_ast_sass_cpp_check>
)

add_test(
    NAME cusr_ast_sass_contracts
    COMMAND $<TARGET_FILE:cusr_ast_sass_contracts>
)

add_test(
    NAME cusr_tile_static_eval_nvrtc_smoke
    COMMAND $<TARGET_FILE:cusr_tile_static_eval_nvrtc_smoke>
)

add_test(
    NAME cusr_native_cuda_reference_compare
    COMMAND
        $<TARGET_FILE:cusr_bench>
        --variant native_cuda_reference
        --ast-mode depth3-alu
        --modules 1
        --kernels-per-module 1
        --asts-per-kernel 4
        --settings 128
        --columns 32
        --tile-rows 64
        --cta-threads 128
        --run-rows 257
        --check-rows 257
        --run-iters 1
)

foreach(CUSR_TEMPLATE_AST_CAPACITY IN ITEMS 8 16 32)
    add_test(
        NAME cusr_tile_static_mse_template_${CUSR_TEMPLATE_AST_CAPACITY}_compare
        COMMAND
            $<TARGET_FILE:cusr_bench>
            --variant tile_static_mse_template
            --ast-mode depth3-alu
            --modules 1
            --kernels-per-module 2
            --ast-capacity ${CUSR_TEMPLATE_AST_CAPACITY}
            --asts-per-kernel ${CUSR_TEMPLATE_AST_CAPACITY}
            --settings 64
            --columns 32
            --tile-rows 64
            --cta-threads 128
            --run-rows 257
            --check-rows 257
            --run-iters 1
    )
endforeach()

foreach(CUSR_EMBEDDED_KERNELS_PER_MODULE IN ITEMS 8 32 128)
    add_test(
        NAME cusr_tile_static_mse_embedded_${CUSR_EMBEDDED_KERNELS_PER_MODULE}k_compare
        COMMAND
            $<TARGET_FILE:cusr_bench>
            --variant tile_static_mse_template
            --ast-mode depth3-alu
            --modules 1
            --kernels-per-module ${CUSR_EMBEDDED_KERNELS_PER_MODULE}
            --ast-capacity 32
            --asts-per-kernel 1
            --settings 16
            --columns 32
            --tile-rows 64
            --cta-threads 128
            --run-rows 257
            --check-rows 257
            --run-iters 1
    )
endforeach()

add_custom_target(cusr_run_nvrtc_ast_sass_tests
    COMMAND
        ${Python3_EXECUTABLE}
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/compare_nvrtc_ast_sass.py"
        --producer $<TARGET_FILE:cusr_nvrtc_ast_sass_gpu>
        --out-dir "${CMAKE_CURRENT_BINARY_DIR}/nvrtc_ast_sass_compare"
    DEPENDS cusr_nvrtc_ast_sass_gpu
    VERBATIM
)

if (CUSR_BUILD_PYTHON_BINDINGS)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_VARIABLE CUSR_NANOBIND_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE CUSR_NANOBIND_CMAKE_RESULT
    )
    if (NOT CUSR_NANOBIND_CMAKE_RESULT EQUAL 0)
        message(FATAL_ERROR "nanobind is not installed for ${Python_EXECUTABLE}")
    endif()
    find_package(nanobind CONFIG REQUIRED PATHS "${CUSR_NANOBIND_CMAKE_DIR}" NO_DEFAULT_PATH)

    nanobind_add_module(cusr_python_native NB_STATIC
        python/bindings/cusr_native.cpp
    )
    set_target_properties(cusr_python_native PROPERTIES
        OUTPUT_NAME "_cusr_native"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/python"
    )
    target_include_directories(cusr_python_native PRIVATE src)
    target_compile_features(cusr_python_native PRIVATE cxx_std_20)

    if (MSVC)
        target_compile_options(cusr_python_native PRIVATE /W4 /WX $<$<CONFIG:Release>:/O2>)
    else()
        target_compile_options(cusr_python_native PRIVATE -Wall -Wextra -Werror $<$<CONFIG:Release>:-O3>)
    endif()

    add_test(
        NAME cusr_python_native_pipeline
        COMMAND
            ${CMAKE_COMMAND} -E env
            "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python:${CMAKE_CURRENT_SOURCE_DIR}/python"
            "${Python_EXECUTABLE}" -m unittest discover
            -s "${CMAKE_CURRENT_SOURCE_DIR}/python/tests"
            -p "test_native_pipeline.py"
            -v
    )
endif()
