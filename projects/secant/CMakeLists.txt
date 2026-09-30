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
project(secant VERSION 0.3.1 LANGUAGES C CXX)
option(SECANT_ENABLE_CUDA_INTEGRATION "Build NVIDIA runners and tests" ON)
option(SECANT_BUILD_TESTS "Build tests" ON)
option(SECANT_BUILD_BENCHMARKS "Build legacy comparison backends (not migrated)" OFF)
option(SECANT_BUILD_PYTHON_BINDINGS "Build legacy Python bindings (not migrated)" OFF)
if(SECANT_BUILD_BENCHMARKS OR SECANT_BUILD_PYTHON_BINDINGS)
    message(FATAL_ERROR "The 0.1 settings/optimizer adapters require migration to Secant 0.3 toggles. See docs/toggles.md. No legacy ABI shim is provided.")
endif()
add_library(secant STATIC s_result.c s_cuda_source.c)
target_include_directories(secant PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(secant PUBLIC c_std_99)
target_compile_definitions(secant PRIVATE SECANT_CUDA_TARGET_STATS_F32_SOURCE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/kernels/cuda/target_stats_f32.cu")
set_source_files_properties(s_cuda_source.c PROPERTIES OBJECT_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/cuda/target_stats_f32.cu")
add_library(secant_cpu STATIC s_cpu.c)
target_link_libraries(secant_cpu PUBLIC secant PRIVATE m)
add_library(secant_cubin STATIC s_cubin_api.c s_cubin_generate_materialize.c s_cubin_generate_sse.c
    s_cubin_generate_affine_stats.c s_cubin_generate_gram_stats.c s_cubin_generate_toggle_sse.c
    s_cubin_inspect.c s_cubin_specialize.c)
target_link_libraries(secant_cubin PUBLIC secant)
foreach(target secant_cpu secant_cubin)
    target_compile_features(${target} PUBLIC c_std_99)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endforeach()
add_executable(secant_generate_cubin app/generate_cubin.c)
target_link_libraries(secant_generate_cubin PRIVATE secant_cubin)
if(SECANT_ENABLE_CUDA_INTEGRATION)
    find_package(CUDAToolkit QUIET)
    find_package(Threads REQUIRED)
endif()
if(SECANT_ENABLE_CUDA_INTEGRATION AND CUDAToolkit_FOUND)
    add_library(secant_cubin_runner STATIC s_cubin_runner.c s_runner_pipeline.c)
    target_compile_features(secant_cubin_runner PUBLIC c_std_99)
    target_link_libraries(secant_cubin_runner PUBLIC secant_cubin PRIVATE CUDA::cuda_driver Threads::Threads)
    target_compile_definitions(secant_cubin_runner PRIVATE _POSIX_C_SOURCE=200809L)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(secant_cubin_runner PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endif()
include(CTest)
if(BUILD_TESTING AND SECANT_BUILD_TESTS)
    find_package(Threads REQUIRED)
    # Compile the production runner against a test-only driver model. The normal
    # library has no allocation hooks, driver dispatch table, or fault flags.
    add_library(secant_runner_fault_objects OBJECT s_cubin_runner.c s_runner_pipeline.c)
    target_include_directories(secant_runner_fault_objects PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests/support/fake_cuda")
    target_compile_features(secant_runner_fault_objects PRIVATE c_std_99)
    target_compile_definitions(secant_runner_fault_objects PRIVATE _POSIX_C_SOURCE=200809L)
    target_compile_options(secant_runner_fault_objects PRIVATE -include "${CMAKE_CURRENT_SOURCE_DIR}/tests/support/runner_fault_hooks.h")
    add_executable(secant_runner_failure_test tests/runner_failure_test.c $<TARGET_OBJECTS:secant_runner_fault_objects>)
    target_include_directories(secant_runner_failure_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests/support/fake_cuda")
    target_compile_features(secant_runner_failure_test PRIVATE c_std_11)
    target_compile_definitions(secant_runner_failure_test PRIVATE _POSIX_C_SOURCE=200809L)
    target_link_libraries(secant_runner_failure_test PRIVATE secant Threads::Threads)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(secant_runner_fault_objects PRIVATE -Wall -Wextra -Werror -pedantic-errors)
        target_compile_options(secant_runner_failure_test PRIVATE -Wall -Wextra -Werror -pedantic-errors)
    endif()
    add_test(NAME secant_runner_failure_test COMMAND secant_runner_failure_test)
    set_tests_properties(secant_runner_failure_test PROPERTIES TIMEOUT 30)
    foreach(name cpu c_header cpp_header ast_internal toggle)
        if(name STREQUAL "cpp_header")
            set(ext cpp)
        else()
            set(ext c)
        endif()
        add_executable(secant_${name}_test tests/${name}_test.${ext})
        target_link_libraries(secant_${name}_test PRIVATE secant_cpu secant_cubin)
        target_compile_features(secant_${name}_test PRIVATE c_std_99)
        if(name STREQUAL "cpp_header")
            target_compile_features(secant_${name}_test PRIVATE cxx_std_17)
        endif()
        if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(secant_${name}_test PRIVATE -Wall -Wextra -Werror -pedantic-errors)
        endif()
        add_test(NAME secant_${name}_test COMMAND secant_${name}_test)
    endforeach()
    if(TARGET secant_cubin_runner)
        add_executable(secant_cubin_stress_test tests/cubin_stress_test.c tests/support/ast_stress.c tests/support/cuda_runtime.c)
        target_link_libraries(secant_cubin_stress_test PRIVATE secant_cubin_runner secant_cpu CUDA::nvrtc CUDA::cuda_driver m)
        add_test(NAME secant_cubin_stress_test COMMAND secant_cubin_stress_test --iterations 8 --kernels 3 --asts-per-kernel 9 --rows 257 --inputs 8 --targets 3 --max-leaves 8 --max-unary-depth 3 --patch-instructions-per-ast 192 --seed 424242)
        set_tests_properties(secant_cubin_stress_test PROPERTIES ENVIRONMENT "CUDA_MODULE_LOADING=EAGER" RESOURCE_LOCK secant_nvidia_gpu TIMEOUT 180)
        add_executable(secant_toggle_cuda_test tests/toggle_cuda_test.c)
        target_link_libraries(secant_toggle_cuda_test PRIVATE secant_cubin_runner secant_cpu CUDA::nvrtc CUDA::cuda_driver m)
        add_test(NAME secant_toggle_cuda_test COMMAND secant_toggle_cuda_test)
        set_tests_properties(secant_toggle_cuda_test PROPERTIES ENVIRONMENT "CUDA_MODULE_LOADING=EAGER" RESOURCE_LOCK secant_nvidia_gpu TIMEOUT 180)
    endif()
endif()
