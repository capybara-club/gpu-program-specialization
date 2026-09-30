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
cmake_minimum_required(VERSION 3.24)
project(app_stack_ptx_ptx_inject_bench C)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP)
find_package(Threads REQUIRED)

set(STACK_PTX_PTX_INJECT_BENCH_NVPTXCOMPILER_TARGET "")
set(STACK_PTX_PTX_INJECT_BENCH_NVPTXCOMPILER_IS_STATIC OFF)
if(TARGET CUDA::nvptxcompiler_static)
  set(STACK_PTX_PTX_INJECT_BENCH_NVPTXCOMPILER_TARGET CUDA::nvptxcompiler_static)
  set(STACK_PTX_PTX_INJECT_BENCH_NVPTXCOMPILER_IS_STATIC ON)
elseif(TARGET CUDA::nvptxcompiler)
  set(STACK_PTX_PTX_INJECT_BENCH_NVPTXCOMPILER_TARGET CUDA::nvptxcompiler)
else()
  set(_stack_ptx_nvptxcompiler_hints "")
  if(DEFINED CUDAToolkit_LIBRARY_DIR)
    list(APPEND _stack_ptx_nvptxcompiler_hints "${CUDAToolkit_LIBRARY_DIR}")
  endif()
  if(DEFINED CUDAToolkit_LIBRARY_DIRS)
    list(APPEND _stack_ptx_nvptxcompiler_hints ${CUDAToolkit_LIBRARY_DIRS})
  endif()
  if(DEFINED CUDAToolkit_ROOT)
    list(APPEND _stack_ptx_nvptxcompiler_hints "${CUDAToolkit_ROOT}")
  endif()
  if(DEFINED CUDA_TOOLKIT_ROOT_DIR)
    list(APPEND _stack_ptx_nvptxcompiler_hints "${CUDA_TOOLKIT_ROOT_DIR}")
  endif()
  list(REMOVE_DUPLICATES _stack_ptx_nvptxcompiler_hints)

  set(_stack_ptx_nvptxcompiler_paths "")
  foreach(path IN LISTS _stack_ptx_nvptxcompiler_hints)
    list(APPEND _stack_ptx_nvptxcompiler_paths
      "${path}"
      "${path}/lib64"
      "${path}/lib"
      "${path}/targets/${CMAKE_SYSTEM_PROCESSOR}-linux/lib"
      "${path}/targets/aarch64-linux/lib"
      "${path}/targets/x86_64-linux/lib"
    )
  endforeach()
  list(REMOVE_DUPLICATES _stack_ptx_nvptxcompiler_paths)

  find_library(STACK_PTX_NVPTXCOMPILER_STATIC
    NAMES nvptxcompiler_static
    PATHS ${_stack_ptx_nvptxcompiler_paths}
  )
  find_library(STACK_PTX_NVPTXCOMPILER_SHARED
    NAMES nvptxcompiler
    PATHS ${_stack_ptx_nvptxcompiler_paths}
  )

  if(STACK_PTX_NVPTXCOMPILER_STATIC)
    add_library(stack_ptx_nvptxcompiler_static UNKNOWN IMPORTED)
    set_target_properties(stack_ptx_nvptxcompiler_static PROPERTIES
      IMPORTED_LOCATION "${STACK_PTX_NVPTXCOMPILER_STATIC}"
    )
    set(STACK_PTX_PTX_INJECT_BENCH_NVPTXCOMPILER_TARGET stack_ptx_nvptxcompiler_static)
    set(STACK_PTX_PTX_INJECT_BENCH_NVPTXCOMPILER_IS_STATIC ON)
  elseif(STACK_PTX_NVPTXCOMPILER_SHARED)
    add_library(stack_ptx_nvptxcompiler UNKNOWN IMPORTED)
    set_target_properties(stack_ptx_nvptxcompiler PROPERTIES
      IMPORTED_LOCATION "${STACK_PTX_NVPTXCOMPILER_SHARED}"
    )
    set(STACK_PTX_PTX_INJECT_BENCH_NVPTXCOMPILER_TARGET stack_ptx_nvptxcompiler)
  else()
    message(FATAL_ERROR "CUDAToolkit nvptxcompiler library not found. Install nvptxcompiler or set CUDA_TOOLKIT_ROOT_DIR/CUDAToolkit_ROOT.")
  endif()
endif()

set(STACK_PTX_PTX_INJECT_BENCH_MAX_UNIQUE_INJECTS 16384 CACHE STRING
  "Max unique PTX inject sites per module")

add_executable(stack_ptx_ptx_inject_bench
  stack_ptx_cubin_bench_standalone.c
  stack_ptx_inject_kernel_gen.c
)

target_include_directories(stack_ptx_ptx_inject_bench PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_include_directories(stack_ptx_ptx_inject_bench SYSTEM PRIVATE
  ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(stack_ptx_ptx_inject_bench PRIVATE
  CUDA::nvrtc
  ${STACK_PTX_PTX_INJECT_BENCH_NVPTXCOMPILER_TARGET}
  Threads::Threads
  m
)

target_compile_features(stack_ptx_ptx_inject_bench PRIVATE c_std_11)

target_compile_definitions(stack_ptx_ptx_inject_bench PRIVATE
  PTX_INJECT_MAX_UNIQUE_INJECTS=${STACK_PTX_PTX_INJECT_BENCH_MAX_UNIQUE_INJECTS}
)

if(OpenMP_C_FOUND)
  target_link_libraries(stack_ptx_ptx_inject_bench PRIVATE OpenMP::OpenMP_C)
endif()

if(STACK_PTX_PTX_INJECT_BENCH_NVPTXCOMPILER_IS_STATIC)
  if(CMAKE_DL_LIBS)
    target_link_libraries(stack_ptx_ptx_inject_bench PRIVATE ${CMAKE_DL_LIBS})
  endif()
  target_link_libraries(stack_ptx_ptx_inject_bench PRIVATE stdc++)
endif()
