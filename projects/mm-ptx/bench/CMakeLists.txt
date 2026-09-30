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
set(MM_PTX_BENCH_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")

set(STACK_PTX_DESCRIPTIONS_JSON "${CMAKE_CURRENT_SOURCE_DIR}/stack_ptx_descriptions.json")
set(STACK_PTX_GENERATED_C_HDR "${MM_PTX_BENCH_GENERATED_DIR}/stack_ptx_example_descriptions.h")

mm_add_stack_ptx_header(
    OUT_HDR "${STACK_PTX_GENERATED_C_HDR}"
    JSON    "${STACK_PTX_DESCRIPTIONS_JSON}"
    LANG    c
)

add_custom_target(stack_ptx_bench_generated_headers
  DEPENDS
    "${STACK_PTX_GENERATED_C_HDR}"
)

add_library(stack_ptx_bench_generated INTERFACE)
target_include_directories(stack_ptx_bench_generated INTERFACE "${MM_PTX_BENCH_GENERATED_DIR}")
add_dependencies(stack_ptx_bench_generated stack_ptx_bench_generated_headers)

if(TARGET CUDA::nvptxcompiler_static)
  set(MM_PTX_BENCH_NVPTX_TARGET CUDA::nvptxcompiler_static)
elseif(TARGET CUDA::nvptxcompiler)
  set(MM_PTX_BENCH_NVPTX_TARGET CUDA::nvptxcompiler)
else()
  message(FATAL_ERROR "CUDAToolkit nvptxcompiler library not found.")
endif()

add_executable(mm_ptx_compile_bench
  ptx_compile_bench.c
)

target_link_libraries(mm_ptx_compile_bench PRIVATE
  CUDA::nvrtc
  ${MM_PTX_BENCH_NVPTX_TARGET}
  m
  mm_ptx_headers
  stack_ptx_bench_generated
)

find_package(OpenMP QUIET)
if(OpenMP_C_FOUND)
  target_link_libraries(mm_ptx_compile_bench PRIVATE OpenMP::OpenMP_C)
endif()

target_compile_features(mm_ptx_compile_bench PRIVATE c_std_11)
