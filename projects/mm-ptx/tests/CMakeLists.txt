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

set(MM_PTX_COMMON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../common")
set(MM_PTX_ROUTINE_LIBRARIES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../routine_libraries")
set(MM_PTX_TESTS_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")

add_library(mm_ptx_common INTERFACE)
target_include_directories(mm_ptx_common INTERFACE
    "${MM_PTX_COMMON_DIR}"
    "${MM_PTX_ROUTINE_LIBRARIES_DIR}"
)

set(GENERATED_INC_DIR "${MM_PTX_TESTS_GENERATED_DIR}")
set(STACK_PTX_DESCRIPTIONS_JSON "${CMAKE_CURRENT_SOURCE_DIR}/stack_ptx_descriptions.json")
set(STACK_PTX_GENERATED_C_HDR "${GENERATED_INC_DIR}/stack_ptx_example_descriptions.h")

mm_add_stack_ptx_header(
    OUT_HDR "${STACK_PTX_GENERATED_C_HDR}"
    JSON    "${STACK_PTX_DESCRIPTIONS_JSON}"
    LANG    c
)

add_custom_target(stack_ptx_test_generated_headers
  DEPENDS
    "${STACK_PTX_GENERATED_C_HDR}"
)

add_library(stack_ptx_test_generated INTERFACE)
target_include_directories(stack_ptx_test_generated INTERFACE "${GENERATED_INC_DIR}")
add_dependencies(stack_ptx_test_generated stack_ptx_test_generated_headers)

set(PTX_INJECT_00_SIMPLE_KERNEL "${CMAKE_CURRENT_SOURCE_DIR}/ptx_inject_simple_kernel.cu")
set(PTX_INJECT_00_SIMPLE_KERNEL_TARGET "mm_ptx_test_ptx_inject_00_simple_kernel_ptx")

add_library(${PTX_INJECT_00_SIMPLE_KERNEL_TARGET} OBJECT "${PTX_INJECT_00_SIMPLE_KERNEL}")
target_link_libraries(${PTX_INJECT_00_SIMPLE_KERNEL_TARGET} PRIVATE mm_ptx_headers)
set_target_properties(${PTX_INJECT_00_SIMPLE_KERNEL_TARGET} PROPERTIES
    CUDA_PTX_COMPILATION ON
    CUDA_ARCHITECTURES native
)

set(PTX_INJECT_00_SIMPLE_KERNEL_PTX "${MM_PTX_TESTS_GENERATED_DIR}/ptx_inject_simple_kernel.ptx")
add_custom_command(
    OUTPUT "${PTX_INJECT_00_SIMPLE_KERNEL_PTX}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${MM_PTX_TESTS_GENERATED_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_OBJECTS:${PTX_INJECT_00_SIMPLE_KERNEL_TARGET}>
            "${PTX_INJECT_00_SIMPLE_KERNEL_PTX}"
    DEPENDS ${PTX_INJECT_00_SIMPLE_KERNEL_TARGET}
    COMMENT "Preparing PTX for ptx_inject_00_simple_tests"
    VERBATIM
)
add_custom_target(ptx_inject_00_simple_kernel_ptx_file DEPENDS "${PTX_INJECT_00_SIMPLE_KERNEL_PTX}")

add_executable(ptx_inject_00_simple_tests test_ptx_inject_00_simple.c)
add_dependencies(ptx_inject_00_simple_tests ptx_inject_00_simple_kernel_ptx_file)
set_source_files_properties(
    test_ptx_inject_00_simple.c
    PROPERTIES
    OBJECT_DEPENDS "${PTX_INJECT_00_SIMPLE_KERNEL_PTX}"
)
target_compile_definitions(ptx_inject_00_simple_tests PRIVATE PTX_KERNEL="${PTX_INJECT_00_SIMPLE_KERNEL_PTX}")

target_link_libraries(
    ptx_inject_00_simple_tests
    PRIVATE
    CUDA::cuda_driver
    CUDA::nvrtc
    CUDA::nvptxcompiler_static
    m
    mm_ptx_headers
    mm_ptx_common
)

add_test(NAME mm_ptx.ptx_inject_00_simple COMMAND ptx_inject_00_simple_tests)
set_tests_properties(mm_ptx.ptx_inject_00_simple PROPERTIES SKIP_RETURN_CODE 77)

add_executable(ptx_inject_inconsistent_duplicate_args_tests test_ptx_inject_inconsistent_duplicate_args.c)
target_link_libraries(
    ptx_inject_inconsistent_duplicate_args_tests
    PRIVATE
    mm_ptx_headers
    mm_ptx_common
)

add_test(
    NAME mm_ptx.ptx_inject_inconsistent_duplicate_args
    COMMAND ptx_inject_inconsistent_duplicate_args_tests
)

add_executable(stack_ptx_zero_frame_depth_empty_string_tests test_stack_ptx_zero_frame_depth_empty_string.c)
target_link_libraries(
    stack_ptx_zero_frame_depth_empty_string_tests
    PRIVATE
    mm_ptx_headers
    mm_ptx_common
)

add_test(
    NAME mm_ptx.stack_ptx_zero_frame_depth_empty_string
    COMMAND stack_ptx_zero_frame_depth_empty_string_tests
)

add_executable(stack_ptx_meta_dup_empty_stack_tests test_stack_ptx_meta_dup_empty_stack.c)
target_link_libraries(
    stack_ptx_meta_dup_empty_stack_tests
    PRIVATE
    mm_ptx_headers
    mm_ptx_common
)

add_test(
    NAME mm_ptx.stack_ptx_meta_dup_empty_stack
    COMMAND stack_ptx_meta_dup_empty_stack_tests
)

add_executable(stack_ptx_meta_reverse_empty_stack_tests test_stack_ptx_meta_reverse_empty_stack.c)
target_link_libraries(
    stack_ptx_meta_reverse_empty_stack_tests
    PRIVATE
    mm_ptx_headers
    mm_ptx_common
)

add_test(
    NAME mm_ptx.stack_ptx_meta_reverse_empty_stack
    COMMAND stack_ptx_meta_reverse_empty_stack_tests
)

add_executable(ast_ptx_expected_output_tests test_ast_ptx_expected_output.c)
target_link_libraries(
    ast_ptx_expected_output_tests
    PRIVATE
    m
    mm_ptx_headers
    mm_ptx_common
)

add_test(
    NAME mm_ptx.ast_ptx_expected_output
    COMMAND ast_ptx_expected_output_tests
)

add_executable(ast_ptx_cpp_tests test_ast_ptx_cpp.cpp)
target_link_libraries(
    ast_ptx_cpp_tests
    PRIVATE
    m
    mm_ptx_headers
    mm_ptx_common
)

add_test(
    NAME mm_ptx.ast_ptx_cpp
    COMMAND ast_ptx_cpp_tests
)

add_executable(ast_ptx_inject_00_simple_tests test_ast_ptx_inject_00_simple.c)
add_dependencies(ast_ptx_inject_00_simple_tests ptx_inject_00_simple_kernel_ptx_file)
set_source_files_properties(
    test_ast_ptx_inject_00_simple.c
    PROPERTIES
    OBJECT_DEPENDS "${PTX_INJECT_00_SIMPLE_KERNEL_PTX}"
)
target_compile_definitions(ast_ptx_inject_00_simple_tests PRIVATE PTX_KERNEL="${PTX_INJECT_00_SIMPLE_KERNEL_PTX}")

target_link_libraries(
    ast_ptx_inject_00_simple_tests
    PRIVATE
    CUDA::cuda_driver
    CUDA::nvrtc
    CUDA::nvptxcompiler_static
    m
    mm_ptx_headers
    mm_ptx_common
)

add_test(NAME mm_ptx.ast_ptx_inject_00_simple COMMAND ast_ptx_inject_00_simple_tests)
set_tests_properties(mm_ptx.ast_ptx_inject_00_simple PROPERTIES SKIP_RETURN_CODE 77)

set(STACK_PTX_INJECT_00_SIMPLE_KERNEL "${PTX_INJECT_00_SIMPLE_KERNEL}")
set(STACK_PTX_INJECT_00_SIMPLE_KERNEL_TARGET "mm_ptx_test_stack_ptx_inject_00_simple_kernel_ptx")

add_library(${STACK_PTX_INJECT_00_SIMPLE_KERNEL_TARGET} OBJECT "${STACK_PTX_INJECT_00_SIMPLE_KERNEL}")
target_link_libraries(${STACK_PTX_INJECT_00_SIMPLE_KERNEL_TARGET} PRIVATE mm_ptx_headers)
set_target_properties(${STACK_PTX_INJECT_00_SIMPLE_KERNEL_TARGET} PROPERTIES
    CUDA_PTX_COMPILATION ON
    CUDA_ARCHITECTURES native
)

set(STACK_PTX_INJECT_00_SIMPLE_KERNEL_PTX "${MM_PTX_TESTS_GENERATED_DIR}/stack_ptx_inject_simple_kernel.ptx")
add_custom_command(
    OUTPUT "${STACK_PTX_INJECT_00_SIMPLE_KERNEL_PTX}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${MM_PTX_TESTS_GENERATED_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_OBJECTS:${STACK_PTX_INJECT_00_SIMPLE_KERNEL_TARGET}>
            "${STACK_PTX_INJECT_00_SIMPLE_KERNEL_PTX}"
    DEPENDS ${STACK_PTX_INJECT_00_SIMPLE_KERNEL_TARGET}
    COMMENT "Preparing PTX for stack_ptx_inject_00_simple_tests"
    VERBATIM
)
add_custom_target(stack_ptx_inject_00_simple_kernel_ptx_file DEPENDS "${STACK_PTX_INJECT_00_SIMPLE_KERNEL_PTX}")

add_executable(stack_ptx_inject_00_simple_tests test_stack_ptx_inject_00_simple.c)
add_dependencies(stack_ptx_inject_00_simple_tests stack_ptx_inject_00_simple_kernel_ptx_file)
set_source_files_properties(
    test_stack_ptx_inject_00_simple.c
    PROPERTIES
    OBJECT_DEPENDS "${STACK_PTX_INJECT_00_SIMPLE_KERNEL_PTX}"
)
target_compile_definitions(stack_ptx_inject_00_simple_tests PRIVATE PTX_KERNEL="${STACK_PTX_INJECT_00_SIMPLE_KERNEL_PTX}")

target_link_libraries(
    stack_ptx_inject_00_simple_tests
    PRIVATE
    CUDA::cuda_driver
    CUDA::nvrtc
    CUDA::nvptxcompiler_static
    m
    mm_ptx_headers
    mm_ptx_common
    stack_ptx_test_generated
)

add_test(NAME mm_ptx.stack_ptx_inject_00_simple COMMAND stack_ptx_inject_00_simple_tests)
set_tests_properties(mm_ptx.stack_ptx_inject_00_simple PROPERTIES SKIP_RETURN_CODE 77)

set(PTX_INJECT_01_GEMM_KERNEL "${CMAKE_CURRENT_SOURCE_DIR}/ptx_inject_gemm_kernel.cu")
set(PTX_INJECT_01_GEMM_KERNEL_TARGET "mm_ptx_test_ptx_inject_01_gemm_kernel_ptx")

add_library(${PTX_INJECT_01_GEMM_KERNEL_TARGET} OBJECT "${PTX_INJECT_01_GEMM_KERNEL}")
target_link_libraries(${PTX_INJECT_01_GEMM_KERNEL_TARGET} PRIVATE mm_ptx_headers)
target_include_directories(${PTX_INJECT_01_GEMM_KERNEL_TARGET} PRIVATE ${MM_PTX_CUTLASS_INCLUDE_DIR})
set_target_properties(${PTX_INJECT_01_GEMM_KERNEL_TARGET} PROPERTIES
    CUDA_PTX_COMPILATION ON
    CUDA_ARCHITECTURES native
)

set(PTX_INJECT_01_GEMM_KERNEL_PTX "${MM_PTX_TESTS_GENERATED_DIR}/ptx_inject_gemm_kernel.ptx")
add_custom_command(
    OUTPUT "${PTX_INJECT_01_GEMM_KERNEL_PTX}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${MM_PTX_TESTS_GENERATED_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_OBJECTS:${PTX_INJECT_01_GEMM_KERNEL_TARGET}>
            "${PTX_INJECT_01_GEMM_KERNEL_PTX}"
    DEPENDS ${PTX_INJECT_01_GEMM_KERNEL_TARGET}
    COMMENT "Preparing PTX for ptx_inject_01_gemm_tests"
    VERBATIM
)
add_custom_target(ptx_inject_01_gemm_kernel_ptx_file DEPENDS "${PTX_INJECT_01_GEMM_KERNEL_PTX}")

add_executable(ptx_inject_01_gemm_tests test_ptx_inject_01_gemm.c)
add_dependencies(ptx_inject_01_gemm_tests ptx_inject_01_gemm_kernel_ptx_file)
set_source_files_properties(
    test_ptx_inject_01_gemm.c
    PROPERTIES
    OBJECT_DEPENDS "${PTX_INJECT_01_GEMM_KERNEL_PTX}"
)
target_compile_definitions(ptx_inject_01_gemm_tests PRIVATE PTX_KERNEL="${PTX_INJECT_01_GEMM_KERNEL_PTX}")

target_link_libraries(
    ptx_inject_01_gemm_tests
    PRIVATE
    CUDA::cuda_driver
    CUDA::nvrtc
    CUDA::nvptxcompiler_static
    m
    mm_ptx_headers
    mm_ptx_common
)

add_test(NAME mm_ptx.ptx_inject_01_gemm COMMAND ptx_inject_01_gemm_tests)
set_tests_properties(mm_ptx.ptx_inject_01_gemm PROPERTIES SKIP_RETURN_CODE 77)

set(STACK_PTX_INJECT_01_GEMM_KERNEL "${PTX_INJECT_01_GEMM_KERNEL}")
set(STACK_PTX_INJECT_01_GEMM_KERNEL_TARGET "mm_ptx_test_stack_ptx_inject_01_gemm_kernel_ptx")

add_library(${STACK_PTX_INJECT_01_GEMM_KERNEL_TARGET} OBJECT "${STACK_PTX_INJECT_01_GEMM_KERNEL}")
target_link_libraries(${STACK_PTX_INJECT_01_GEMM_KERNEL_TARGET} PRIVATE mm_ptx_headers)
target_include_directories(${STACK_PTX_INJECT_01_GEMM_KERNEL_TARGET} PRIVATE ${MM_PTX_CUTLASS_INCLUDE_DIR})
set_target_properties(${STACK_PTX_INJECT_01_GEMM_KERNEL_TARGET} PROPERTIES
    CUDA_PTX_COMPILATION ON
    CUDA_ARCHITECTURES native
)

set(STACK_PTX_INJECT_01_GEMM_KERNEL_PTX "${MM_PTX_TESTS_GENERATED_DIR}/stack_ptx_inject_gemm_kernel.ptx")
add_custom_command(
    OUTPUT "${STACK_PTX_INJECT_01_GEMM_KERNEL_PTX}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${MM_PTX_TESTS_GENERATED_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_OBJECTS:${STACK_PTX_INJECT_01_GEMM_KERNEL_TARGET}>
            "${STACK_PTX_INJECT_01_GEMM_KERNEL_PTX}"
    DEPENDS ${STACK_PTX_INJECT_01_GEMM_KERNEL_TARGET}
    COMMENT "Preparing PTX for stack_ptx_inject_01_gemm_tests"
    VERBATIM
)
add_custom_target(stack_ptx_inject_01_gemm_kernel_ptx_file DEPENDS "${STACK_PTX_INJECT_01_GEMM_KERNEL_PTX}")

add_executable(stack_ptx_inject_01_gemm_tests test_stack_ptx_inject_01_gemm.c)
add_dependencies(stack_ptx_inject_01_gemm_tests stack_ptx_inject_01_gemm_kernel_ptx_file)
set_source_files_properties(
    test_stack_ptx_inject_01_gemm.c
    PROPERTIES
    OBJECT_DEPENDS "${STACK_PTX_INJECT_01_GEMM_KERNEL_PTX}"
)
target_compile_definitions(stack_ptx_inject_01_gemm_tests PRIVATE PTX_KERNEL="${STACK_PTX_INJECT_01_GEMM_KERNEL_PTX}")

target_link_libraries(
    stack_ptx_inject_01_gemm_tests
    PRIVATE
    CUDA::cuda_driver
    CUDA::nvrtc
    CUDA::nvptxcompiler_static
    m
    mm_ptx_headers
    mm_ptx_common
    stack_ptx_test_generated
)

add_test(NAME mm_ptx.stack_ptx_inject_01_gemm COMMAND stack_ptx_inject_01_gemm_tests)
set_tests_properties(mm_ptx.stack_ptx_inject_01_gemm PROPERTIES SKIP_RETURN_CODE 77)

set(STACK_PTX_PHILOX_KERNEL "${CMAKE_CURRENT_SOURCE_DIR}/stack_ptx_philox_kernel.cu")
set(STACK_PTX_PHILOX_KERNEL_TARGET "mm_ptx_test_stack_ptx_inject_philox_kernel_ptx")

add_library(${STACK_PTX_PHILOX_KERNEL_TARGET} OBJECT "${STACK_PTX_PHILOX_KERNEL}")
target_link_libraries(${STACK_PTX_PHILOX_KERNEL_TARGET} PRIVATE mm_ptx_headers mm_ptx_common)
set_target_properties(${STACK_PTX_PHILOX_KERNEL_TARGET} PROPERTIES
    CUDA_PTX_COMPILATION ON
    CUDA_ARCHITECTURES native
)

set(STACK_PTX_PHILOX_KERNEL_PTX "${MM_PTX_TESTS_GENERATED_DIR}/stack_ptx_philox_kernel.ptx")
add_custom_command(
    OUTPUT "${STACK_PTX_PHILOX_KERNEL_PTX}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${MM_PTX_TESTS_GENERATED_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_OBJECTS:${STACK_PTX_PHILOX_KERNEL_TARGET}>
            "${STACK_PTX_PHILOX_KERNEL_PTX}"
    DEPENDS ${STACK_PTX_PHILOX_KERNEL_TARGET}
    COMMENT "Preparing PTX for stack_ptx_inject_philox_tests"
    VERBATIM
)
add_custom_target(stack_ptx_philox_kernel_ptx_file DEPENDS "${STACK_PTX_PHILOX_KERNEL_PTX}")

add_executable(stack_ptx_inject_philox_tests test_stack_ptx_inject_philox.c)
add_dependencies(stack_ptx_inject_philox_tests stack_ptx_philox_kernel_ptx_file)
set_source_files_properties(
    test_stack_ptx_inject_philox.c
    PROPERTIES
    OBJECT_DEPENDS "${STACK_PTX_PHILOX_KERNEL_PTX}"
)
target_compile_definitions(stack_ptx_inject_philox_tests PRIVATE PTX_KERNEL="${STACK_PTX_PHILOX_KERNEL_PTX}")

target_link_libraries(
    stack_ptx_inject_philox_tests
    PRIVATE
    CUDA::cuda_driver
    CUDA::nvrtc
    CUDA::nvptxcompiler_static
    m
    mm_ptx_headers
    mm_ptx_common
    stack_ptx_test_generated
)

add_test(NAME mm_ptx.stack_ptx_inject_philox COMMAND stack_ptx_inject_philox_tests)
set_tests_properties(mm_ptx.stack_ptx_inject_philox PROPERTIES SKIP_RETURN_CODE 77)
