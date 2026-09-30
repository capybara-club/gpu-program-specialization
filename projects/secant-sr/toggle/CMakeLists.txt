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
add_library(secant_sr STATIC src/program.c src/search.c src/cpu.c src/refine.c src/variation.c src/archive.c src/lm_model.c)
target_include_directories(secant_sr PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/.." "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(secant_sr PUBLIC secant secant_cpu PRIVATE m)
target_compile_features(secant_sr PUBLIC c_std_99)
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(secant_sr PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()
if(SECANT_SR_ENABLE_CUDA AND TARGET secant_cubin_runner)
    find_package(CUDAToolkit REQUIRED)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/secant_sr_lm_source.h"
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/embed_lm.py" "${CMAKE_CURRENT_BINARY_DIR}/secant_sr_lm_source.h"
        DEPENDS embed_lm.py kernels/lm.cu src/lm_layout.h VERBATIM)
    add_library(secant_sr_cuda STATIC src/cuda.c src/lm_cuda.c "${CMAKE_CURRENT_BINARY_DIR}/secant_sr_lm_source.h")
    target_include_directories(secant_sr_cuda PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
    target_include_directories(secant_sr_cuda PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_link_libraries(secant_sr_cuda PUBLIC secant_sr secant_cubin_runner PRIVATE CUDA::cuda_driver CUDA::nvrtc m)
    target_compile_definitions(secant_sr_cuda PRIVATE _POSIX_C_SOURCE=200809L)
    target_compile_features(secant_sr_cuda PUBLIC c_std_99)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(secant_sr_cuda PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
endif()
add_executable(secant_sr_search src/main.c ../app/dataset.c)
target_include_directories(secant_sr_search PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../app")
target_link_libraries(secant_sr_search PRIVATE secant_sr m)
target_compile_definitions(secant_sr_search PRIVATE _POSIX_C_SOURCE=200809L)
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(secant_sr_search PRIVATE -Wall -Wextra -Werror)
endif()
set_target_properties(secant_sr_search PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
if(TARGET secant_sr_cuda)
    target_compile_definitions(secant_sr_search PRIVATE SR_HAS_CUDA=1)
    target_link_libraries(secant_sr_search PRIVATE secant_sr_cuda)
endif()
if(BUILD_TESTING AND SECANT_SR_BUILD_TESTS)
    add_executable(secant_sr_lm_model_test tests/lm_model_test.c)
    target_link_libraries(secant_sr_lm_model_test PRIVATE secant_sr)
    add_test(NAME secant_sr_lm_model_test COMMAND secant_sr_lm_model_test)
    add_executable(secant_sr_archive_test tests/archive_test.c)
    target_link_libraries(secant_sr_archive_test PRIVATE secant_sr)
    add_test(NAME secant_sr_archive_test COMMAND secant_sr_archive_test)
    add_executable(secant_sr_variation_test tests/variation_test.c)
    target_link_libraries(secant_sr_variation_test PRIVATE secant_sr)
    add_test(NAME secant_sr_variation_test COMMAND secant_sr_variation_test)
    set_tests_properties(secant_sr_variation_test PROPERTIES TIMEOUT 90)
    add_executable(secant_sr_refine_test tests/refine_test.c)
    target_link_libraries(secant_sr_refine_test PRIVATE secant_sr)
    add_test(NAME secant_sr_refine_test COMMAND secant_sr_refine_test)
    set_tests_properties(secant_sr_refine_test PROPERTIES TIMEOUT 90)
    add_executable(secant_sr_toggle_test tests/search_test.c)
    target_link_libraries(secant_sr_toggle_test PRIVATE secant_sr)
    add_test(NAME secant_sr_toggle_test COMMAND secant_sr_toggle_test)
    set_tests_properties(secant_sr_toggle_test PROPERTIES TIMEOUT 90)
    add_executable(secant_sr_dataset_test ../tests/dataset_test.c ../app/dataset.c)
    target_include_directories(secant_sr_dataset_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../app")
    target_link_libraries(secant_sr_dataset_test PRIVATE m)
    add_test(NAME secant_sr_dataset_test COMMAND secant_sr_dataset_test)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(NAME secant_sr_python_ast_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/../python/test_ast.py")
        add_test(NAME secant_sr_python_toggle_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/python_test.py" $<TARGET_FILE:secant_sr_search>)
        add_test(NAME secant_sr_srbench_toggle_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/../python/test_srbench_toggle.py" $<TARGET_FILE:secant_sr_search>)
        add_test(NAME secant_sr_search_comparison_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/../python/test_search_comparison.py")
        add_test(NAME secant_sr_search_overnight_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/../python/test_search_overnight.py")
        set_tests_properties(secant_sr_srbench_toggle_test PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python" TIMEOUT 90)
        set_tests_properties(secant_sr_python_ast_test secant_sr_python_toggle_test PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python" TIMEOUT 90)
    endif()
    if(TARGET secant_sr_cuda)
        add_executable(secant_sr_lm_cuda_test tests/lm_cuda_test.c)
        target_link_libraries(secant_sr_lm_cuda_test PRIVATE secant_sr_cuda)
        add_test(NAME secant_sr_lm_cuda_test COMMAND secant_sr_lm_cuda_test)
        set_tests_properties(secant_sr_lm_cuda_test PROPERTIES ENVIRONMENT "CUDA_MODULE_LOADING=EAGER" TIMEOUT 180)
        add_test(NAME secant_sr_lm_gradient_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/lm_gradient_test.py")
        set_tests_properties(secant_sr_lm_gradient_test PROPERTIES ENVIRONMENT "CUDA_MODULE_LOADING=EAGER" TIMEOUT 180)
        add_executable(secant_sr_refine_cuda_test tests/refine_test.c)
        target_compile_definitions(secant_sr_refine_cuda_test PRIVATE SR_HAS_CUDA=1)
        target_link_libraries(secant_sr_refine_cuda_test PRIVATE secant_sr_cuda)
        add_test(NAME secant_sr_refine_cuda_test COMMAND secant_sr_refine_cuda_test)
        set_tests_properties(secant_sr_refine_cuda_test PROPERTIES ENVIRONMENT "CUDA_MODULE_LOADING=EAGER" TIMEOUT 180)
        add_executable(secant_sr_toggle_cuda_test tests/cuda_test.c)
        target_link_libraries(secant_sr_toggle_cuda_test PRIVATE secant_sr_cuda)
        add_test(NAME secant_sr_toggle_cuda_test COMMAND secant_sr_toggle_cuda_test)
        set_tests_properties(secant_sr_toggle_cuda_test PROPERTIES ENVIRONMENT "CUDA_MODULE_LOADING=EAGER" TIMEOUT 180)
    endif()
endif()
