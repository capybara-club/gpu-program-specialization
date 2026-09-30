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
project(stack-ptx-compiler C)

find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)

if(NOT TARGET mm_ptx_headers)
  message(FATAL_ERROR "mm_ptx_headers target not found. Add mm-ptx before stack_ptx_compiler.")
endif()

if(NOT DEFINED MM_PTX_SOURCE_DIR)
  set(MM_PTX_SOURCE_DIR "${CMAKE_SOURCE_DIR}/thirdparty/mm-ptx")
endif()

set(STACK_PTX_COMPILER_NVPTXCOMPILER_TARGET "")
if(TARGET CUDA::nvptxcompiler_static)
  set(STACK_PTX_COMPILER_NVPTXCOMPILER_TARGET CUDA::nvptxcompiler_static)
elseif(TARGET CUDA::nvptxcompiler)
  set(STACK_PTX_COMPILER_NVPTXCOMPILER_TARGET CUDA::nvptxcompiler)
else()
  message(FATAL_ERROR "CUDAToolkit nvptxcompiler target not found. Install nvptxcompiler or update CMake/CUDA.")
endif()

option(STACK_PTX_COMPILER_ENABLE_OPENMP "Enable OpenMP in stack_ptx_compiler" ON)
set(STACK_PTX_COMPILER_OPENMP_MAX_NUM_CPUS 24 CACHE STRING "Max number of CPUs for OpenMP in stack_ptx_compiler")
set(STACK_PTX_COMPILER_MAX_UNIQUE_INJECTS 16384 CACHE STRING "Max unique PTX inject sites per module")

set(STACK_PTX_COMPILER_OPENMP_ENABLED 0)
set(STACK_PTX_COMPILER_OPENMP_MAX_NUM_CPUS_EFFECTIVE 1)
if(STACK_PTX_COMPILER_ENABLE_OPENMP)
  find_package(OpenMP)
  if(OpenMP_C_FOUND)
    set(STACK_PTX_COMPILER_OPENMP_ENABLED 1)
    set(STACK_PTX_COMPILER_OPENMP_MAX_NUM_CPUS_EFFECTIVE ${STACK_PTX_COMPILER_OPENMP_MAX_NUM_CPUS})
  else()
    message(WARNING "STACK_PTX_COMPILER_ENABLE_OPENMP is ON but OpenMP for C was not found; building without OpenMP.")
  endif()
endif()

if(STACK_PTX_COMPILER_OPENMP_MAX_NUM_CPUS_EFFECTIVE LESS 1)
  message(WARNING "STACK_PTX_COMPILER_OPENMP_MAX_NUM_CPUS must be >= 1; forcing to 1.")
  set(STACK_PTX_COMPILER_OPENMP_MAX_NUM_CPUS_EFFECTIVE 1)
endif()

add_library(stack_ptx_compiler STATIC
  stack_ptx_compiler.c
  stack_ptx_compiler_backend_local.c
  stack_ptx_compiler_mm_ptx.c
)

target_include_directories(stack_ptx_compiler
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_include_directories(stack_ptx_compiler SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(stack_ptx_compiler PUBLIC mm_ptx_headers Threads::Threads ${STACK_PTX_COMPILER_NVPTXCOMPILER_TARGET} m)
target_compile_features(stack_ptx_compiler PUBLIC c_std_11)

target_compile_definitions(stack_ptx_compiler PRIVATE
  STACK_PTX_COMPILER_OPENMP_ENABLED=${STACK_PTX_COMPILER_OPENMP_ENABLED}
  STACK_PTX_COMPILER_OPENMP_MAX_NUM_CPUS=${STACK_PTX_COMPILER_OPENMP_MAX_NUM_CPUS_EFFECTIVE}
  PTX_INJECT_MAX_UNIQUE_INJECTS=${STACK_PTX_COMPILER_MAX_UNIQUE_INJECTS}
)

if(STACK_PTX_COMPILER_OPENMP_ENABLED)
  target_link_libraries(stack_ptx_compiler PUBLIC OpenMP::OpenMP_C)
endif()

option(STACK_PTX_COMPILER_FETCH_NNG "Fetch nng automatically if not found (requires network)" ON)

find_package(nng CONFIG QUIET)

if(NOT nng_FOUND AND STACK_PTX_COMPILER_FETCH_NNG)
  include(FetchContent)
  FetchContent_Declare(
    nng
    GIT_REPOSITORY https://github.com/nanomsg/nng.git
    GIT_TAG v1.8.0
  )
  set(NNG_ENABLE_NNGCAT OFF CACHE BOOL "" FORCE)
  set(NNG_ENABLE_TLS OFF CACHE BOOL "" FORCE)
  set(NNG_TESTS OFF CACHE BOOL "" FORCE)
  set(NNG_TOOLS OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(nng)
endif()

if(TARGET nng::nng)
  set(NNG_TARGET nng::nng)
elseif(TARGET nng)
  set(NNG_TARGET nng)
else()
  message(FATAL_ERROR "nng not found. Install nng or set STACK_PTX_COMPILER_FETCH_NNG=ON.")
endif()

add_library(stack_ptx_compiler_nng STATIC
  stack_ptx_compiler.c
  stack_ptx_compiler_backend_local.c
  stack_ptx_compiler_backend_nng.c
  stack_ptx_compiler_mm_ptx.c
  stack_ptx_nng_wire.c
)

target_include_directories(stack_ptx_compiler_nng
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_include_directories(stack_ptx_compiler_nng SYSTEM PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(stack_ptx_compiler_nng PUBLIC mm_ptx_headers Threads::Threads ${NNG_TARGET} ${STACK_PTX_COMPILER_NVPTXCOMPILER_TARGET} m)
target_compile_features(stack_ptx_compiler_nng PUBLIC c_std_11)

target_compile_definitions(stack_ptx_compiler_nng PRIVATE
  STACK_PTX_COMPILER_ENABLE_NNG=1
  STACK_PTX_COMPILER_OPENMP_ENABLED=${STACK_PTX_COMPILER_OPENMP_ENABLED}
  STACK_PTX_COMPILER_OPENMP_MAX_NUM_CPUS=${STACK_PTX_COMPILER_OPENMP_MAX_NUM_CPUS_EFFECTIVE}
  PTX_INJECT_MAX_UNIQUE_INJECTS=${STACK_PTX_COMPILER_MAX_UNIQUE_INJECTS}
)

if(STACK_PTX_COMPILER_OPENMP_ENABLED)
  target_link_libraries(stack_ptx_compiler_nng PUBLIC OpenMP::OpenMP_C)
endif()

option(STACK_PTX_COMPILER_BUILD_APPS "Build stack_ptx_compiler apps" ON)

if(STACK_PTX_COMPILER_BUILD_APPS)
  add_subdirectory(app/stack_ptx_nng_server)
  add_subdirectory(app/stack_ptx_nng_client)
endif()
