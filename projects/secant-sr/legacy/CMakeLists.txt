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

project(secant_sr LANGUAGES C)

option(SECANT_SR_BUILD_TESTS "Build secant-sr tests" ON)
option(SECANT_SR_ENABLE_CUDA "Build the direct-CUBIN search application" ON)

set(SECANT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../secant" CACHE PATH "Path to the Secant source tree")
if(NOT TARGET secant)
    set(SECANT_BUILD_TESTS OFF CACHE BOOL "Build Secant tests while vendored" FORCE)
    set(SECANT_BUILD_BENCHMARKS OFF CACHE BOOL "Build Secant benchmarks while vendored" FORCE)
    add_subdirectory("${SECANT_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/secant" EXCLUDE_FROM_ALL)
endif()

add_library(
    secant_sr
    STATIC
        src/result.c
        src/program.c
        src/population.c
        src/search.c
        src/dynamic_leaf.c
        src/dynamic_constant.c
        src/format.c
        src/cpu.c
        src/constant_optimization.c)
target_include_directories(secant_sr PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(secant_sr PUBLIC secant secant_cpu PRIVATE m)
target_compile_features(secant_sr PUBLIC c_std_99)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(secant_sr PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(secant_sr_search app/search.c app/dataset.c app/leaf_settings.c)
target_link_libraries(secant_sr_search PRIVATE secant_sr m)
target_compile_features(secant_sr_search PRIVATE c_std_99)

add_executable(secant_sr_bench app/bench.c)
target_link_libraries(secant_sr_bench PRIVATE secant_sr m)
target_compile_features(secant_sr_bench PRIVATE c_std_99)

if(SECANT_SR_ENABLE_CUDA)
    find_package(CUDAToolkit QUIET)
    find_package(Threads QUIET)
    if(CUDAToolkit_FOUND AND Threads_FOUND AND TARGET secant_cubin_runner AND
       TARGET secant_cubin_lm_optimizer_runner)
        find_package(SQLite3 REQUIRED)
        target_sources(secant_sr_search PRIVATE app/cubin_cache.c app/cubin_evaluator.c app/mse_reducer.c)
        target_compile_definitions(secant_sr_search PRIVATE SECANT_SR_HAS_CUBIN=1)
        target_link_libraries(
            secant_sr_search
            PRIVATE
                secant_cubin
                secant_cubin_runner
                secant_cubin_lm_optimizer_runner
                CUDA::cuda_driver
                CUDA::nvrtc
                Threads::Threads
                SQLite::SQLite3)
        set(SECANT_SR_HAS_CUBIN ON)
    endif()
endif()

include(CTest)
if(BUILD_TESTING AND SECANT_SR_BUILD_TESTS)
    add_executable(secant_sr_program_test tests/program_test.c)
    target_link_libraries(secant_sr_program_test PRIVATE secant_sr)
    add_test(NAME secant_sr_program_test COMMAND secant_sr_program_test)

    add_executable(secant_sr_search_test tests/search_test.c)
    target_link_libraries(secant_sr_search_test PRIVATE secant_sr)
    add_test(NAME secant_sr_search_test COMMAND secant_sr_search_test)

    add_executable(secant_sr_constant_optimization_test tests/constant_optimization_test.c)
    target_include_directories(secant_sr_constant_optimization_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(secant_sr_constant_optimization_test PRIVATE secant_sr)
    add_test(NAME secant_sr_constant_optimization_test COMMAND secant_sr_constant_optimization_test)

    add_executable(secant_sr_dynamic_leaf_test tests/dynamic_leaf_test.c)
    target_link_libraries(secant_sr_dynamic_leaf_test PRIVATE secant_sr)
    add_test(NAME secant_sr_dynamic_leaf_test COMMAND secant_sr_dynamic_leaf_test)

    add_executable(secant_sr_maturity_search_test tests/maturity_search_test.c)
    target_include_directories(secant_sr_maturity_search_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(secant_sr_maturity_search_test PRIVATE secant_sr)
    add_test(NAME secant_sr_maturity_search_test COMMAND secant_sr_maturity_search_test)

    add_executable(secant_sr_dynamic_constant_test tests/dynamic_constant_test.c)
    target_link_libraries(secant_sr_dynamic_constant_test PRIVATE secant_sr)
    add_test(NAME secant_sr_dynamic_constant_test COMMAND secant_sr_dynamic_constant_test)

    add_executable(secant_sr_population_stress_test tests/population_stress_test.c)
    target_link_libraries(secant_sr_population_stress_test PRIVATE secant_sr)
    add_test(NAME secant_sr_population_stress_test COMMAND secant_sr_population_stress_test)

    add_executable(secant_sr_quality_diversity_test tests/quality_diversity_test.c)
    target_include_directories(secant_sr_quality_diversity_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(secant_sr_quality_diversity_test PRIVATE secant_sr)
    add_test(NAME secant_sr_quality_diversity_test COMMAND secant_sr_quality_diversity_test)

    add_executable(secant_sr_dataset_test tests/dataset_test.c app/dataset.c)
    target_include_directories(secant_sr_dataset_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/app")
    target_link_libraries(secant_sr_dataset_test PRIVATE m)
    add_test(NAME secant_sr_dataset_test COMMAND secant_sr_dataset_test)

    add_executable(secant_sr_leaf_settings_test tests/leaf_settings_test.c app/leaf_settings.c)
    target_include_directories(secant_sr_leaf_settings_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/app")
    add_test(NAME secant_sr_leaf_settings_test COMMAND secant_sr_leaf_settings_test)

    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(NAME secant_sr_python_ast_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/python/test_ast.py")
        set_tests_properties(
            secant_sr_python_ast_test
            PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python")
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -c "import numpy, pandas, sklearn, sympy, yaml"
            RESULT_VARIABLE SECANT_SR_PYTHON_PROTOCOL_DEPENDENCIES_RESULT
            OUTPUT_QUIET
            ERROR_QUIET)
        if(SECANT_SR_PYTHON_PROTOCOL_DEPENDENCIES_RESULT EQUAL 0)
        add_test(
            NAME secant_sr_srbench_v2_test
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/python/test_srbench_v2.py")
        set_tests_properties(
            secant_sr_srbench_v2_test
            PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python")
        add_test(
            NAME secant_sr_srbench_v2_targets_test
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/python/test_srbench_v2_targets.py")
        set_tests_properties(
            secant_sr_srbench_v2_targets_test
            PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python")
        endif()
    endif()

    if(SECANT_SR_HAS_CUBIN)
        add_executable(
            secant_sr_cubin_evaluator_test
            tests/cubin_evaluator_test.c app/cubin_cache.c app/cubin_evaluator.c app/dataset.c app/mse_reducer.c)
        target_include_directories(secant_sr_cubin_evaluator_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/app")
        target_link_libraries(
            secant_sr_cubin_evaluator_test
            PRIVATE
                secant_sr
                secant_cubin
                secant_cubin_runner
                secant_cubin_lm_optimizer_runner
                CUDA::cuda_driver
                CUDA::nvrtc
                Threads::Threads
                SQLite::SQLite3)
        add_test(NAME secant_sr_cubin_evaluator_test COMMAND secant_sr_cubin_evaluator_test)
        set_tests_properties(secant_sr_cubin_evaluator_test PROPERTIES ENVIRONMENT "CUDA_MODULE_LOADING=EAGER")

        add_executable(secant_sr_cubin_cache_test tests/cubin_cache_test.c app/cubin_cache.c)
        target_include_directories(secant_sr_cubin_cache_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/app")
        target_link_libraries(secant_sr_cubin_cache_test PRIVATE secant SQLite::SQLite3)
        add_test(NAME secant_sr_cubin_cache_test COMMAND secant_sr_cubin_cache_test)

        add_test(
            NAME secant_sr_dynamic_leaf_search_gpu_test
            COMMAND secant_sr_search
                --backend cubin-dynamic-leaf
                --problem nguyen1
                --population 13
                --generations 1
                --rows 257
                --validation-rows 257
                --workers 1
                --streams 1
                --leaf-settings 8
                --dynamic-batch-asts 5
                --constant-settings 8
                --constant-optimize-budget 13
                --constant-optimize-interval 1
                --constant-optimize-random-fraction 0
                --constant-setting-credit-weight 0.01
                --dynamic-leaves 8
                --kernels 2
                --asts-per-kernel 5
                --tile-rows 64
                --final-cpu-optimize 0
                --stop-r2 2)
        set_tests_properties(
            secant_sr_dynamic_leaf_search_gpu_test
            PROPERTIES
                ENVIRONMENT "CUDA_MODULE_LOADING=EAGER"
                PASS_REGULAR_EXPRESSION "final_cpu_optimizer enabled=0")

        add_test(
            NAME secant_sr_staged_search_gpu_test
            COMMAND secant_sr_search
                --backend cubin-staged
                --problem nguyen1
                --population 13
                --generations 2
                --dynamic-generations 1
                --rows 257
                --validation-rows 257
                --workers 1
                --streams 1
                --leaf-settings 8
                --dynamic-batch-asts 5
                --constant-settings 8
                --constant-optimize-budget 13
                --constant-optimize-interval 1
                --constant-optimize-random-fraction 0
                --constant-sweep-phase both
                --constant-setting-credit-weight 0.01
                --dynamic-leaves 8
                --kernels 2
                --asts-per-kernel 5
                --tile-rows 64
                --static-tile-rows 1024
                --patch-instructions-per-ast 64
                --final-cpu-optimize 0
                --stop-r2 2)
        set_tests_properties(
            secant_sr_staged_search_gpu_test
            PROPERTIES
                ENVIRONMENT "CUDA_MODULE_LOADING=EAGER"
                PASS_REGULAR_EXPRESSION "constant_optimizer shape=ast_local policy=bounded_diverse")

        add_test(
            NAME secant_sr_staged_final_constant_search_gpu_test
            COMMAND secant_sr_search
                --backend cubin-staged
                --problem nguyen1
                --population 13
                --generations 2
                --dynamic-generations 1
                --rows 257
                --validation-rows 257
                --workers 1
                --streams 1
                --leaf-settings 8
                --dynamic-batch-asts 5
                --constant-settings 8
                --constant-optimize-budget 13
                --constant-optimize-interval 1
                --constant-optimize-random-fraction 0
                --constant-sweep-phase final
                --dynamic-leaves 8
                --kernels 2
                --asts-per-kernel 5
                --tile-rows 64
                --static-tile-rows 1024
                --patch-instructions-per-ast 64
                --final-cpu-optimize 0
                --stop-r2 2)
        set_tests_properties(
            secant_sr_staged_final_constant_search_gpu_test
            PROPERTIES
                ENVIRONMENT "CUDA_MODULE_LOADING=EAGER"
                PASS_REGULAR_EXPRESSION "constant_optimizer shape=ast_local policy=bounded_diverse")

        add_test(
            NAME secant_sr_maturity_search_gpu_test
            COMMAND secant_sr_search
                --backend cubin-maturity
                --problem nguyen1
                --population 13
                --generations 3
                --rows 257
                --validation-rows 257
                --workers 1
                --streams 1
                --leaf-settings 8
                --dynamic-batch-asts 5
                --dynamic-leaves 8
                --mixed-dynamic-leaves 4
                --mixed-refine-probability 1
                --constant-settings 8
                --constant-optimize-budget 13
                --constant-optimize-interval 1
                --constant-optimize-random-fraction 0
                --constant-optimizer-iterations 2
                --constant-sweep-phase each
                --kernels 2
                --asts-per-kernel 5
                --tile-rows 64
                --static-tile-rows 256
                --patch-instructions-per-ast 64
                --final-cpu-optimize 0
                --stop-r2 2)
        set_tests_properties(
            secant_sr_maturity_search_gpu_test
            PROPERTIES
                ENVIRONMENT "CUDA_MODULE_LOADING=EAGER"
                PASS_REGULAR_EXPRESSION "constant_optimizer shape=packed")

        add_test(
            NAME secant_sr_maturity_lm_search_gpu_test
            COMMAND secant_sr_search
                --backend cubin-maturity
                --constant-optimizer lm
                --problem nguyen5
                --population 32
                --generations 1
                --rows 257
                --validation-rows 257
                --workers 1
                --streams 2
                --leaf-settings 8
                --dynamic-batch-asts 8
                --dynamic-leaves 8
                --mixed-dynamic-leaves 4
                --mixed-refine-probability 0.25
                --constant-settings 8
                --constant-optimize-probability 1
                --constant-optimizer-iterations 2
                --constant-sweep-phase each
                --lm-batch-asts 2
                --lm-settings-per-cta 4
                --lm-tile-rows 128
                --lm-threads 128
                --lm-patch-instructions 2048
                --kernels 2
                --asts-per-kernel 8
                --tile-rows 64
                --static-tile-rows 256
                --patch-instructions-per-ast 64
                --final-cpu-optimize 0
                --stop-r2 2)
        set_tests_properties(
            secant_sr_maturity_lm_search_gpu_test
            PROPERTIES
                ENVIRONMENT "CUDA_MODULE_LOADING=EAGER"
                PASS_REGULAR_EXPRESSION "constant_optimizer shape=lm_thread_owned")
    endif()
endif()
