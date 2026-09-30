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
cmake_minimum_required(VERSION 3.17)

project(odezza VERSION 0.1.0 LANGUAGES C)

add_library(odezza STATIC
    core/o_elf.c
    core/o_sass_assembler.c
    core/o_generate_lm_cuda.c
    core/o_lm_source.c
    core/o_lm_ast.c
    core/o_inspect_lm_cubin.c
    core/o_specialize_lm_cubin.c
    core/o_lm_pipeline.c
    core/o_philox.c
    core/o_score_reduce.c
    core/o_ast.c
    core/o_generate_scoring_cuda.c
    core/o_inspect_scoring_cubin.c
    core/o_nvrtc_compilation.c
    core/o_scoring_pipeline.c
    core/o_scoring_template.c
    core/o_specialize_scoring_cubin.c
)
target_include_directories(odezza PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/core>
)
set_target_properties(odezza PROPERTIES
    C_STANDARD 99
    C_STANDARD_REQUIRED YES
    C_EXTENSIONS NO
    POSITION_INDEPENDENT_CODE ON
)

find_package(CUDAToolkit REQUIRED)
list(GET CUDAToolkit_INCLUDE_DIRS 0 ODEZZA_CUDA_INCLUDE_DIR)
target_compile_definitions(odezza PRIVATE ODEZZA_CUDA_INCLUDE_DIR="${ODEZZA_CUDA_INCLUDE_DIR}")
target_link_libraries(odezza PUBLIC CUDA::cuda_driver CUDA::nvrtc)

add_executable(odezza_internal_generate
    app/o_generate_scoring_cuda_main.c
)
set_target_properties(odezza_internal_generate PROPERTIES
    OUTPUT_NAME odezza-internal-generate-scoring-cuda
    C_STANDARD 99
    C_STANDARD_REQUIRED YES
    C_EXTENSIONS NO
)
target_link_libraries(odezza_internal_generate PRIVATE odezza)

add_executable(odezza_internal_pipeline_benchmark
    app/o_scoring_pipeline_benchmark_main.c
)
set_target_properties(odezza_internal_pipeline_benchmark PROPERTIES
    OUTPUT_NAME odezza-internal-pipeline-benchmark
    C_STANDARD 99
    C_STANDARD_REQUIRED YES
    C_EXTENSIONS NO
)
target_link_libraries(odezza_internal_pipeline_benchmark PRIVATE odezza)

if(NOT MSVC)
    target_compile_options(odezza PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(odezza_internal_generate PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(odezza_internal_pipeline_benchmark PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_link_libraries(odezza PUBLIC m)
endif()

find_package(Threads REQUIRED)
target_link_libraries(odezza PUBLIC Threads::Threads)

include(CTest)
add_subdirectory(frontend)
if(BUILD_TESTING)
    add_test(
        NAME c99_sources_have_no_goto
        COMMAND ${CMAKE_COMMAND}
            -DROOT=${CMAKE_CURRENT_SOURCE_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/o_verify_no_goto.cmake
    )
    if(CMAKE_NM)
        add_test(
            NAME scoring_library_has_no_context_api
            COMMAND ${CMAKE_COMMAND}
                -DNM=${CMAKE_NM}
                -DLIBRARY=$<TARGET_FILE:odezza>
                -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/o_verify_no_context_symbols.cmake
        )
    endif()
    add_executable(o_generate_scoring_cuda_test
        tests/o_generate_scoring_cuda_test.c
    )
    set_target_properties(o_generate_scoring_cuda_test PROPERTIES
        C_STANDARD 99
        C_STANDARD_REQUIRED YES
        C_EXTENSIONS NO
    )
    target_link_libraries(o_generate_scoring_cuda_test PRIVATE odezza)
    if(NOT MSVC)
        target_compile_options(o_generate_scoring_cuda_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(NAME generate_scoring_cuda_contract COMMAND o_generate_scoring_cuda_test)
    add_executable(o_inspect_scoring_cubin_test
        tests/o_inspect_scoring_cubin_test.c
    )
    set_target_properties(o_inspect_scoring_cubin_test PROPERTIES
        C_STANDARD 99
        C_STANDARD_REQUIRED YES
        C_EXTENSIONS NO
    )
    target_link_libraries(o_inspect_scoring_cubin_test PRIVATE odezza)
    if(NOT MSVC)
        target_compile_options(o_inspect_scoring_cubin_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(NAME inspect_scoring_cubin_rejections COMMAND o_inspect_scoring_cubin_test)
    add_executable(o_nvrtc_compilation_test
        tests/o_nvrtc_compilation_test.c
    )
    set_target_properties(o_nvrtc_compilation_test PROPERTIES
        C_STANDARD 99
        C_STANDARD_REQUIRED YES
        C_EXTENSIONS NO
    )
    target_link_libraries(o_nvrtc_compilation_test PRIVATE odezza)
    if(NOT MSVC)
        target_compile_options(o_nvrtc_compilation_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(NAME nvrtc_compilation_contract COMMAND o_nvrtc_compilation_test)
    add_executable(o_specialize_scoring_cubin_test
        tests/o_specialize_scoring_cubin_test.c
    )
    set_target_properties(o_specialize_scoring_cubin_test PROPERTIES
        C_STANDARD 99
        C_STANDARD_REQUIRED YES
        C_EXTENSIONS NO
    )
    target_link_libraries(o_specialize_scoring_cubin_test PRIVATE odezza)
    if(NOT MSVC)
        target_compile_options(o_specialize_scoring_cubin_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(NAME specialize_scoring_cubin_contract COMMAND o_specialize_scoring_cubin_test)
    add_executable(o_sass_test tests/o_sass_test.c)
    set_target_properties(o_sass_test PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED YES C_EXTENSIONS NO)
    target_link_libraries(o_sass_test PRIVATE odezza)
    if(NOT MSVC)
        target_compile_options(o_sass_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(NAME sass_writer_contract COMMAND o_sass_test)
    add_executable(o_sass_corpus_test tests/o_sass_corpus_test.c)
    set_target_properties(o_sass_corpus_test PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED YES C_EXTENSIONS NO)
    target_link_libraries(o_sass_corpus_test PRIVATE odezza)
    if(NOT MSVC)
        target_compile_options(o_sass_corpus_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(NAME sass_corpus_contract COMMAND o_sass_corpus_test)
    add_executable(o_scoring_pipeline_test
        tests/o_scoring_pipeline_test.c
    )
    set_target_properties(o_scoring_pipeline_test PROPERTIES
        C_STANDARD 99
        C_STANDARD_REQUIRED YES
        C_EXTENSIONS NO
    )
    target_link_libraries(o_scoring_pipeline_test PRIVATE odezza)
    if(NOT MSVC)
        target_compile_options(o_scoring_pipeline_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(NAME scoring_pipeline_handle_contract COMMAND o_scoring_pipeline_test)
    add_executable(o_scoring_cuda_test tests/o_scoring_cuda_test.c)
    set_target_properties(o_scoring_cuda_test PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED YES C_EXTENSIONS NO)
    target_link_libraries(o_scoring_cuda_test PRIVATE odezza)
    if(NOT MSVC)
        target_compile_options(o_scoring_cuda_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(NAME scoring_cuda_scalar_no_constants COMMAND o_scoring_cuda_test 1 0 1)
    add_test(NAME scoring_cuda_eight_state_one_system COMMAND o_scoring_cuda_test 8 8 1)
    add_test(NAME scoring_cuda_eight_state_packed COMMAND o_scoring_cuda_test 8 8 8)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(NAME scoring_cuda_artifact_identity
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/o_verify_c_generator.py
            --generator $<TARGET_FILE:odezza_internal_generate>)
    add_test(NAME scoring_pipeline_cuda_one_system_smoke
        COMMAND odezza_internal_pipeline_benchmark
            --submissions 4
            --workers 1
            --slots-per-worker 2
            --systems 1
            --constant-banks 1
            --toggle-bits 1
            --steps-per-observation 4
            --runs 2)
    set_tests_properties(scoring_pipeline_cuda_one_system_smoke PROPERTIES
        PASS_REGULAR_EXPRESSION "correct_system_mse")
    add_test(NAME scoring_pipeline_cuda_multi_system_smoke
        COMMAND odezza_internal_pipeline_benchmark
            --submissions 4
            --workers 1
            --slots-per-worker 2
            --systems 2
            --constant-banks 1
            --toggle-bits 1
            --steps-per-observation 4
            --runs 2)
    set_tests_properties(scoring_pipeline_cuda_multi_system_smoke PROPERTIES
        PASS_REGULAR_EXPRESSION "correct_system_mse")
    add_test(NAME scoring_pipeline_specialized_toggle_mapping
        COMMAND odezza_internal_pipeline_benchmark
            --submissions 2
            --workers 1
            --slots-per-worker 2
            --systems 1
            --constant-banks 16
            --toggle-bits 1
            --steps-per-observation 4
            --runs 1
            --validate-toggle 1)
    set_tests_properties(scoring_pipeline_specialized_toggle_mapping PROPERTIES
        PASS_REGULAR_EXPRESSION "specialized_toggle_mapping_verified.*true;constant_bank_pairing_verified.*true")
    add_test(NAME scoring_pipeline_runtime_toggle_widths
        COMMAND odezza_internal_pipeline_benchmark
            --submissions 1
            --workers 1
            --slots-per-worker 2
            --systems 1
            --constant-banks 1
            --toggle-bits 12
            --steps-per-observation 4
            --runs 1
            --validate-runtime-widths 1)
    set_tests_properties(scoring_pipeline_runtime_toggle_widths PROPERTIES
        PASS_REGULAR_EXPRESSION "same_pipeline_runtime_widths_verified.*true")
    add_test(NAME scoring_pipeline_multi_toggle_predicate_latency
        COMMAND odezza_internal_pipeline_benchmark
            --submissions 1
            --workers 1
            --slots-per-worker 2
            --systems 3
            --constant-banks 1
            --toggle-bits 2
            --steps-per-observation 4
            --runs 1
            --validate-multi-toggle 1)
    set_tests_properties(scoring_pipeline_multi_toggle_predicate_latency PROPERTIES
        PASS_REGULAR_EXPRESSION "multi_toggle_predicate_latency_verified.*true")
    add_test(NAME generate_scoring_cuda_cli_help COMMAND odezza_internal_generate --help)
    set_tests_properties(generate_scoring_cuda_cli_help PROPERTIES
        PASS_REGULAR_EXPRESSION "--state-capacity"
    )
    add_test(
        NAME generate_scoring_cuda_cli_custom_shape
        COMMAND odezza_internal_generate
            --state-capacity 6
            --constant-capacity 8
            --system-capacity 4
            --shared-patch-capacity 128
            --system-patch-capacity 256
            --output ${CMAKE_CURRENT_BINARY_DIR}/o_generate_scoring_cuda_cli_custom.cu
    )
endif()

if(BUILD_TESTING)
    add_executable(o_philox_test tests/o_philox_test.c)
    target_link_libraries(o_philox_test PRIVATE odezza)
    set_target_properties(o_philox_test PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED YES C_EXTENSIONS NO)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(o_philox_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(NAME philox_pool_cuda_exact COMMAND o_philox_test)
    add_executable(o_score_reduce_test tests/o_score_reduce_test.c)
    target_link_libraries(o_score_reduce_test PRIVATE odezza)
    set_target_properties(o_score_reduce_test PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED YES C_EXTENSIONS NO)
    if(NOT MSVC)
        target_compile_options(o_score_reduce_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(NAME score_reduction_cuda_exact COMMAND o_score_reduce_test)
endif()

if(BUILD_TESTING)
    add_executable(o_lm_core_test tests/o_lm_core_test.c)
    target_link_libraries(o_lm_core_test PRIVATE odezza)
    set_target_properties(o_lm_core_test PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED YES C_EXTENSIONS NO)
    if(NOT MSVC)
        target_compile_options(o_lm_core_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(NAME native_lm_public_api COMMAND o_lm_core_test)
    set_tests_properties(native_lm_public_api PROPERTIES ENVIRONMENT "CUDA_MODULE_LOADING=EAGER")
    foreach(lanes 1 2 4 8)
        add_test(NAME native_lm_coupled_${lanes} COMMAND o_lm_core_test ${lanes} coupled)
        set_tests_properties(native_lm_coupled_${lanes} PROPERTIES ENVIRONMENT "CUDA_MODULE_LOADING=EAGER")
    endforeach()
endif()

# Shared C99 entry point for thin language bindings; same native units and ABI.
get_target_property(ODEZZA_CORE_SOURCES odezza SOURCES)
add_library(odezza_shared SHARED ${ODEZZA_CORE_SOURCES})
set_target_properties(odezza_shared PROPERTIES OUTPUT_NAME odezza C_STANDARD 99 C_STANDARD_REQUIRED YES C_EXTENSIONS NO)
target_include_directories(odezza_shared PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/core>)
target_link_libraries(odezza_shared PUBLIC CUDA::cuda_driver CUDA::nvrtc Threads::Threads)
if(NOT MSVC)
    target_compile_options(odezza_shared PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_link_libraries(odezza_shared PUBLIC m)
endif()
include(GNUInstallDirs)
install(TARGETS odezza odezza_shared ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES core/odezza.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

if(UNIX AND NOT APPLE)
    target_link_options(odezza_shared PRIVATE "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/core/odezza.exports")
    set_property(TARGET odezza_shared APPEND PROPERTY LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/core/odezza.exports)
endif()

option(ODEZZA_BUILD_REQUEST_RUNTIME "Build the separate C99 JSON scoring service layer" OFF)
if(ODEZZA_BUILD_REQUEST_RUNTIME)
    add_subdirectory(runtime)
endif()
