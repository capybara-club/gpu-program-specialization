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

option(BUILD_PYTHON_BINDINGS "Build Python bindings with Nanobind" OFF)
option(IMPLICIT_SINDY_BUILD_TESTS "Build FusedSINDy tests and benchmarks" ON)
option(IMPLICIT_SINDY_BUILD_SOLVER "Build cuSolverDx ridge solve support" ON)
option(IMPLICIT_SINDY_SOLVER_USE_NVRTC "Build the cuSolverDx ridge solve module with NVRTC/nvJitLink instead of nvcc" ON)
option(IMPLICIT_SINDY_PREFER_PYTHON_CUDA "Prefer CUDA component libraries from the selected Python environment when building Python bindings" ON)
set(IMPLICIT_SINDY_PYTHON_CUDA_ROOT "" CACHE PATH
  "Optional nvidia/cuXX root from a Python CUDA package. Empty auto-detects from Python site-packages when building Python bindings.")
set(IMPLICIT_SINDY_NVRTC_CUDA_ROOT "" CACHE PATH
  "Optional CUDA root whose nvrtc/nvJitLink libraries should build the solver module. Empty uses Python CUDA root when available, otherwise CUDAToolkit_ROOT.")

find_package(Python3 REQUIRED COMPONENTS Interpreter)
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/ImplicitSindyPythonCuda.cmake")
implicit_sindy_detect_python_cuda_root()
if(IMPLICIT_SINDY_BUILD_SOLVER AND NOT IMPLICIT_SINDY_SOLVER_USE_NVRTC)
  project(implicit_sindy C CXX CUDA)
else()
  project(implicit_sindy C CXX)
endif()
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(ImplicitSindyThirdParty)
implicit_sindy_setup_thirdparty()

if(IMPLICIT_SINDY_BUILD_TESTS)
  enable_testing()
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

execute_process(
  COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
  RESULT_VARIABLE IMPLICIT_SINDY_NVIDIA_SMI_STATUS
  OUTPUT_VARIABLE IMPLICIT_SINDY_NVIDIA_SMI_COMPUTE_CAP
  OUTPUT_STRIP_TRAILING_WHITESPACE)

if(IMPLICIT_SINDY_NVIDIA_SMI_STATUS EQUAL 0 AND NOT IMPLICIT_SINDY_NVIDIA_SMI_COMPUTE_CAP STREQUAL "")
  string(REPLACE "\n" ";" IMPLICIT_SINDY_NVIDIA_SMI_COMPUTE_CAP_LIST
    "${IMPLICIT_SINDY_NVIDIA_SMI_COMPUTE_CAP}")
  list(GET IMPLICIT_SINDY_NVIDIA_SMI_COMPUTE_CAP_LIST 0 IMPLICIT_SINDY_PRIMARY_COMPUTE_CAP)
  string(REPLACE "." "" IMPLICIT_SINDY_CUDA_ARCH "${IMPLICIT_SINDY_PRIMARY_COMPUTE_CAP}")
  if(CMAKE_CUDA_ARCHITECTURES STREQUAL "native")
    set(CMAKE_CUDA_ARCHITECTURES ${IMPLICIT_SINDY_CUDA_ARCH})
  endif()
elseif(CMAKE_CUDA_ARCHITECTURES STREQUAL "native")
  set(IMPLICIT_SINDY_CUDA_ARCH 120)
  set(CMAKE_CUDA_ARCHITECTURES ${IMPLICIT_SINDY_CUDA_ARCH})
else()
  list(GET CMAKE_CUDA_ARCHITECTURES 0 IMPLICIT_SINDY_CUDA_ARCH)
endif()

set(IMPLICIT_SINDY_NVCC_EXTRA_FLAGS)
set(IMPLICIT_SINDY_NVCC_HOST_FLAGS)
set(IMPLICIT_SINDY_NVCC_COMMON_FLAGS)
if(IMPLICIT_SINDY_BUILD_SOLVER AND NOT IMPLICIT_SINDY_SOLVER_USE_NVRTC)
  if(CMAKE_CUDA_FLAGS)
    separate_arguments(IMPLICIT_SINDY_NVCC_EXTRA_FLAGS NATIVE_COMMAND "${CMAKE_CUDA_FLAGS}")
  endif()
  if(CMAKE_CUDA_HOST_COMPILER)
    list(APPEND IMPLICIT_SINDY_NVCC_HOST_FLAGS "-ccbin=${CMAKE_CUDA_HOST_COMPILER}")
  endif()
  set(IMPLICIT_SINDY_NVCC_COMMON_FLAGS
    ${IMPLICIT_SINDY_NVCC_EXTRA_FLAGS}
    ${IMPLICIT_SINDY_NVCC_HOST_FLAGS})
endif()

set(IMPLICIT_SINDY_CUDA_INCLUDE_DIR "")
if(CUDAToolkit_INCLUDE_DIRS)
  list(GET CUDAToolkit_INCLUDE_DIRS 0 IMPLICIT_SINDY_CUDA_INCLUDE_DIR)
endif()
if(NOT IMPLICIT_SINDY_CUDA_INCLUDE_DIR)
  set(IMPLICIT_SINDY_CUDA_INCLUDE_DIR "${CUDAToolkit_ROOT}/include")
endif()

set(IMPLICIT_SINDY_MAX_UNIQUE_INJECTS 16384 CACHE STRING
  "Maximum unique PTX inject sites supported by ptx_inject.h")
set(CUBIN_FUNCTION_PATCH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty" CACHE PATH
  "Path to a directory containing cubin_function_patch.h.")
if(EXISTS "${CUBIN_FUNCTION_PATCH_ROOT}/cubin_function_patch.h")
  set(CUBIN_FUNCTION_PATCH_INCLUDE_DIR "${CUBIN_FUNCTION_PATCH_ROOT}")
elseif(EXISTS "${CUBIN_FUNCTION_PATCH_ROOT}/include/cubin_function_patch.h")
  set(CUBIN_FUNCTION_PATCH_INCLUDE_DIR "${CUBIN_FUNCTION_PATCH_ROOT}/include")
else()
  message(FATAL_ERROR "CUBIN_FUNCTION_PATCH_ROOT must contain cubin_function_patch.h")
endif()

set(IMPLICIT_SINDY_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(IMPLICIT_SINDY_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(IMPLICIT_SINDY_CUDA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cuda")
set(IMPLICIT_SINDY_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/test")
set(IMPLICIT_SINDY_THIRDPARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty")
set(IMPLICIT_SINDY_INTERNAL_DIR "${IMPLICIT_SINDY_SRC_DIR}/internal")
set(IMPLICIT_SINDY_DESCRIPTIONS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/descriptions")
set(IMPLICIT_SINDY_TOOLS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tools")
set(IMPLICIT_SINDY_PUBLIC_HEADER "${IMPLICIT_SINDY_INCLUDE_DIR}/implicit_sindy.h")
set(IMPLICIT_SINDY_PUBLIC_HEADERS
  "${IMPLICIT_SINDY_INCLUDE_DIR}/implicit_sindy.h")
set(IMPLICIT_SINDY_INTERNAL_HEADERS
  "${IMPLICIT_SINDY_INTERNAL_DIR}/implicit_sindy_ast_compiler.h"
  "${IMPLICIT_SINDY_INTERNAL_DIR}/binary_ast_to_stack_ptx.h"
  "${IMPLICIT_SINDY_INTERNAL_DIR}/implicit_sindy_internal_constants.h")

set(STACK_PTX_DESCRIPTIONS_JSON
  "${IMPLICIT_SINDY_DESCRIPTIONS_DIR}/stack_ptx_descriptions.json")
set(STACK_PTX_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(STACK_PTX_GENERATED_DESCRIPTIONS
  "${STACK_PTX_GENERATED_DIR}/stack_ptx_descriptions.h")
set(IMPLICIT_FEATURE_RIDGE_SOLVE_OBJECT
  "${CMAKE_CURRENT_BINARY_DIR}/implicit_feature_ridge_solve.o")
set(IMPLICIT_FEATURE_RIDGE_SOLVE_FATBIN
  "${STACK_PTX_GENERATED_DIR}/implicit_feature_ridge_solve.fatbin")

add_custom_command(
  OUTPUT "${STACK_PTX_GENERATED_DESCRIPTIONS}"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${STACK_PTX_GENERATED_DIR}"
  COMMAND "${Python3_EXECUTABLE}"
    "${IMPLICIT_SINDY_TOOLS_DIR}/generate_stack_ptx_descriptions.py"
    --input "${STACK_PTX_DESCRIPTIONS_JSON}"
    --output "${STACK_PTX_GENERATED_DESCRIPTIONS}"
  DEPENDS
    "${IMPLICIT_SINDY_TOOLS_DIR}/generate_stack_ptx_descriptions.py"
    "${STACK_PTX_DESCRIPTIONS_JSON}"
  VERBATIM)
add_custom_target(stack_ptx_descriptions_generated
  DEPENDS "${STACK_PTX_GENERATED_DESCRIPTIONS}")

if(IMPLICIT_SINDY_BUILD_SOLVER)
  if(NOT IMPLICIT_SINDY_MATHDX_PACKAGE_ROOT)
    message(FATAL_ERROR "FusedSINDy requires MathDx/cuSolverDx for implicit_feature_ridge_solve")
  endif()

  set(IMPLICIT_SINDY_CUSOLVERDX_FATBIN
    "${IMPLICIT_SINDY_MATHDX_PACKAGE_ROOT}/lib/libcusolverdx.fatbin")
  if(NOT EXISTS "${IMPLICIT_SINDY_CUSOLVERDX_FATBIN}")
    message(FATAL_ERROR "cuSolverDx fatbin not found: ${IMPLICIT_SINDY_CUSOLVERDX_FATBIN}")
  endif()

  math(EXPR IMPLICIT_FEATURE_RIDGE_SOLVE_CUSOLVERDX_SM "${IMPLICIT_SINDY_CUDA_ARCH} * 10")

  set(IMPLICIT_FEATURE_RIDGE_SOLVE_INCLUDE_DIRS
    "${IMPLICIT_SINDY_INCLUDE_DIR}"
    "${IMPLICIT_SINDY_INTERNAL_DIR}"
    "${IMPLICIT_SINDY_THIRDPARTY_DIR}"
    "${IMPLICIT_SINDY_CUDA_INCLUDE_DIR}")
  if(EXISTS "${IMPLICIT_SINDY_CUDA_INCLUDE_DIR}/cccl")
    list(APPEND IMPLICIT_FEATURE_RIDGE_SOLVE_INCLUDE_DIRS
      "${IMPLICIT_SINDY_CUDA_INCLUDE_DIR}/cccl")
  endif()
  if(IMPLICIT_SINDY_PYTHON_CUDA_INCLUDE_DIR)
    list(APPEND IMPLICIT_FEATURE_RIDGE_SOLVE_INCLUDE_DIRS
      "${IMPLICIT_SINDY_PYTHON_CUDA_INCLUDE_DIR}")
    if(EXISTS "${IMPLICIT_SINDY_PYTHON_CUDA_INCLUDE_DIR}/cccl")
      list(APPEND IMPLICIT_FEATURE_RIDGE_SOLVE_INCLUDE_DIRS
        "${IMPLICIT_SINDY_PYTHON_CUDA_INCLUDE_DIR}/cccl")
    endif()
  endif()
  foreach(include_dir IN LISTS IMPLICIT_SINDY_THIRDPARTY_INCLUDE_DIRS)
    list(APPEND IMPLICIT_FEATURE_RIDGE_SOLVE_INCLUDE_DIRS "${include_dir}")
  endforeach()
  list(REMOVE_DUPLICATES IMPLICIT_FEATURE_RIDGE_SOLVE_INCLUDE_DIRS)

  if(IMPLICIT_SINDY_SOLVER_USE_NVRTC)
    set(IMPLICIT_FEATURE_RIDGE_SOLVE_NVRTC_INCLUDE_ARGS)
    foreach(include_dir IN LISTS IMPLICIT_FEATURE_RIDGE_SOLVE_INCLUDE_DIRS)
      list(APPEND IMPLICIT_FEATURE_RIDGE_SOLVE_NVRTC_INCLUDE_ARGS
        --include "${include_dir}")
    endforeach()

    set(IMPLICIT_FEATURE_RIDGE_SOLVE_NVRTC_CUDA_ROOT_ARGS)
    if(IMPLICIT_SINDY_NVRTC_CUDA_ROOT)
      set(IMPLICIT_FEATURE_RIDGE_SOLVE_NVRTC_CUDA_ROOT "${IMPLICIT_SINDY_NVRTC_CUDA_ROOT}")
    elseif(IMPLICIT_SINDY_PYTHON_CUDA_ROOT)
      set(IMPLICIT_FEATURE_RIDGE_SOLVE_NVRTC_CUDA_ROOT "${IMPLICIT_SINDY_PYTHON_CUDA_ROOT}")
    elseif(CUDAToolkit_ROOT)
      set(IMPLICIT_FEATURE_RIDGE_SOLVE_NVRTC_CUDA_ROOT "${CUDAToolkit_ROOT}")
    else()
      set(IMPLICIT_FEATURE_RIDGE_SOLVE_NVRTC_CUDA_ROOT "")
    endif()
    if(IMPLICIT_FEATURE_RIDGE_SOLVE_NVRTC_CUDA_ROOT)
      list(APPEND IMPLICIT_FEATURE_RIDGE_SOLVE_NVRTC_CUDA_ROOT_ARGS
        --cuda-root "${IMPLICIT_FEATURE_RIDGE_SOLVE_NVRTC_CUDA_ROOT}")
    endif()

    add_custom_command(
      OUTPUT "${IMPLICIT_FEATURE_RIDGE_SOLVE_FATBIN}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${STACK_PTX_GENERATED_DIR}"
      COMMAND "${Python3_EXECUTABLE}"
        "${IMPLICIT_SINDY_TOOLS_DIR}/build_solver_fatbin_nvrtc.py"
        --source "${IMPLICIT_SINDY_CUDA_DIR}/implicit_feature_ridge_solve.cu"
        --output "${IMPLICIT_FEATURE_RIDGE_SOLVE_FATBIN}"
        --cusolverdx-fatbin "${IMPLICIT_SINDY_CUSOLVERDX_FATBIN}"
        --sm "${IMPLICIT_SINDY_CUDA_ARCH}"
        --cusolverdx-sm "${IMPLICIT_FEATURE_RIDGE_SOLVE_CUSOLVERDX_SM}"
        ${IMPLICIT_FEATURE_RIDGE_SOLVE_NVRTC_CUDA_ROOT_ARGS}
        ${IMPLICIT_FEATURE_RIDGE_SOLVE_NVRTC_INCLUDE_ARGS}
      DEPENDS
        "${IMPLICIT_SINDY_TOOLS_DIR}/build_solver_fatbin_nvrtc.py"
        "${IMPLICIT_SINDY_CUDA_DIR}/implicit_feature_ridge_solve.cu"
        "${IMPLICIT_SINDY_INTERNAL_DIR}/implicit_sindy_internal_constants.h"
        ${IMPLICIT_SINDY_INTERNAL_HEADERS}
        "${IMPLICIT_SINDY_CUSOLVERDX_FATBIN}"
      VERBATIM)
  else()
    set(IMPLICIT_FEATURE_RIDGE_SOLVE_INCLUDE_FLAGS)
    foreach(include_dir IN LISTS IMPLICIT_FEATURE_RIDGE_SOLVE_INCLUDE_DIRS)
      list(APPEND IMPLICIT_FEATURE_RIDGE_SOLVE_INCLUDE_FLAGS "-I${include_dir}")
    endforeach()

    add_custom_command(
      OUTPUT "${IMPLICIT_FEATURE_RIDGE_SOLVE_OBJECT}"
      COMMAND "${CMAKE_CUDA_COMPILER}"
        ${IMPLICIT_SINDY_NVCC_COMMON_FLAGS}
        -O3
        "-arch=sm_${IMPLICIT_SINDY_CUDA_ARCH}"
        -std=c++17
        -dlto
        -dc
        --expt-relaxed-constexpr
        ${IMPLICIT_FEATURE_RIDGE_SOLVE_INCLUDE_FLAGS}
        "-DIMPLICIT_FEATURE_RIDGE_SOLVE_CUSOLVERDX_SM=${IMPLICIT_FEATURE_RIDGE_SOLVE_CUSOLVERDX_SM}"
        -c
        "${IMPLICIT_SINDY_CUDA_DIR}/implicit_feature_ridge_solve.cu"
        -o "${IMPLICIT_FEATURE_RIDGE_SOLVE_OBJECT}"
      DEPENDS
        "${IMPLICIT_SINDY_CUDA_DIR}/implicit_feature_ridge_solve.cu"
        "${IMPLICIT_SINDY_INTERNAL_DIR}/implicit_sindy_internal_constants.h"
        ${IMPLICIT_SINDY_INTERNAL_HEADERS}
        "${IMPLICIT_SINDY_CUSOLVERDX_FATBIN}"
      VERBATIM)

    add_custom_command(
      OUTPUT "${IMPLICIT_FEATURE_RIDGE_SOLVE_FATBIN}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${STACK_PTX_GENERATED_DIR}"
      COMMAND "${CMAKE_CUDA_COMPILER}"
        ${IMPLICIT_SINDY_NVCC_COMMON_FLAGS}
        "-arch=sm_${IMPLICIT_SINDY_CUDA_ARCH}"
        -dlto
        -dlink
        -fatbin
        "${IMPLICIT_FEATURE_RIDGE_SOLVE_OBJECT}"
        "${IMPLICIT_SINDY_CUSOLVERDX_FATBIN}"
        -o "${IMPLICIT_FEATURE_RIDGE_SOLVE_FATBIN}"
      DEPENDS
        "${IMPLICIT_FEATURE_RIDGE_SOLVE_OBJECT}"
        "${IMPLICIT_SINDY_CUSOLVERDX_FATBIN}"
      VERBATIM)
  endif()

  add_custom_target(implicit_feature_ridge_solve_fatbin_generated
    DEPENDS "${IMPLICIT_FEATURE_RIDGE_SOLVE_FATBIN}")
endif()

set(IMPLICIT_SINDY_NVPTXCOMPILER_TARGET "")
set(IMPLICIT_SINDY_NVPTXCOMPILER_IS_STATIC OFF)
set(IMPLICIT_SINDY_NVRTC_TARGET CUDA::nvrtc)
set(IMPLICIT_SINDY_NVJITLINK_TARGET CUDA::nvJitLink)
if(IMPLICIT_SINDY_PYTHON_CUDA_LIBRARY_DIR)
  find_library(IMPLICIT_SINDY_PYTHON_NVRTC
    NAMES nvrtc libnvrtc.so.13 libnvrtc.so.12
    PATHS "${IMPLICIT_SINDY_PYTHON_CUDA_LIBRARY_DIR}"
    NO_DEFAULT_PATH)
  find_library(IMPLICIT_SINDY_PYTHON_NVJITLINK
    NAMES nvJitLink libnvJitLink.so.13 libnvJitLink.so.12
    PATHS "${IMPLICIT_SINDY_PYTHON_CUDA_LIBRARY_DIR}"
    NO_DEFAULT_PATH)
  if(IMPLICIT_SINDY_PYTHON_NVRTC AND IMPLICIT_SINDY_PYTHON_NVJITLINK)
    add_library(implicit_sindy_python_nvrtc SHARED IMPORTED)
    set_target_properties(implicit_sindy_python_nvrtc PROPERTIES
      IMPORTED_LOCATION "${IMPLICIT_SINDY_PYTHON_NVRTC}")
    add_library(implicit_sindy_python_nvjitlink SHARED IMPORTED)
    set_target_properties(implicit_sindy_python_nvjitlink PROPERTIES
      IMPORTED_LOCATION "${IMPLICIT_SINDY_PYTHON_NVJITLINK}")
    if(IMPLICIT_SINDY_PYTHON_CUDA_INCLUDE_DIR)
      set_target_properties(implicit_sindy_python_nvrtc PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${IMPLICIT_SINDY_PYTHON_CUDA_INCLUDE_DIR}")
      set_target_properties(implicit_sindy_python_nvjitlink PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${IMPLICIT_SINDY_PYTHON_CUDA_INCLUDE_DIR}")
    endif()
    set(IMPLICIT_SINDY_NVRTC_TARGET implicit_sindy_python_nvrtc)
    set(IMPLICIT_SINDY_NVJITLINK_TARGET implicit_sindy_python_nvjitlink)
    message(STATUS "implicit-sindy: linking Python CUDA nvrtc ${IMPLICIT_SINDY_PYTHON_NVRTC}")
    message(STATUS "implicit-sindy: linking Python CUDA nvJitLink ${IMPLICIT_SINDY_PYTHON_NVJITLINK}")
  else()
    message(WARNING "Python CUDA root is missing nvrtc or nvJitLink libraries; falling back to CUDAToolkit targets")
  endif()
  find_library(IMPLICIT_SINDY_PYTHON_NVPTXCOMPILER_STATIC
    NAMES nvptxcompiler_static libnvptxcompiler_static.a
    PATHS "${IMPLICIT_SINDY_PYTHON_CUDA_LIBRARY_DIR}"
    NO_DEFAULT_PATH)
  find_library(IMPLICIT_SINDY_PYTHON_NVPTXCOMPILER_SHARED
    NAMES nvptxcompiler libnvptxcompiler.so libnvptxcompiler.so.13 libnvptxcompiler.so.12
    PATHS "${IMPLICIT_SINDY_PYTHON_CUDA_LIBRARY_DIR}"
    NO_DEFAULT_PATH)
  if(IMPLICIT_SINDY_PYTHON_NVPTXCOMPILER_STATIC)
    add_library(implicit_sindy_python_nvptxcompiler_static UNKNOWN IMPORTED)
    set_target_properties(implicit_sindy_python_nvptxcompiler_static PROPERTIES
      IMPORTED_LOCATION "${IMPLICIT_SINDY_PYTHON_NVPTXCOMPILER_STATIC}")
    if(IMPLICIT_SINDY_PYTHON_CUDA_INCLUDE_DIR)
      set_target_properties(implicit_sindy_python_nvptxcompiler_static PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${IMPLICIT_SINDY_PYTHON_CUDA_INCLUDE_DIR}")
    endif()
    set(IMPLICIT_SINDY_NVPTXCOMPILER_TARGET implicit_sindy_python_nvptxcompiler_static)
    set(IMPLICIT_SINDY_NVPTXCOMPILER_IS_STATIC ON)
    message(STATUS "implicit-sindy: linking Python CUDA nvPTXCompiler static ${IMPLICIT_SINDY_PYTHON_NVPTXCOMPILER_STATIC}")
  elseif(IMPLICIT_SINDY_PYTHON_NVPTXCOMPILER_SHARED)
    add_library(implicit_sindy_python_nvptxcompiler SHARED IMPORTED)
    set_target_properties(implicit_sindy_python_nvptxcompiler PROPERTIES
      IMPORTED_LOCATION "${IMPLICIT_SINDY_PYTHON_NVPTXCOMPILER_SHARED}")
    if(IMPLICIT_SINDY_PYTHON_CUDA_INCLUDE_DIR)
      set_target_properties(implicit_sindy_python_nvptxcompiler PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${IMPLICIT_SINDY_PYTHON_CUDA_INCLUDE_DIR}")
    endif()
    set(IMPLICIT_SINDY_NVPTXCOMPILER_TARGET implicit_sindy_python_nvptxcompiler)
    message(STATUS "implicit-sindy: linking Python CUDA nvPTXCompiler ${IMPLICIT_SINDY_PYTHON_NVPTXCOMPILER_SHARED}")
  else()
    message(FATAL_ERROR
      "Python CUDA root is missing nvPTXCompiler. Install the pinned "
      "nvidia-nvptxcompiler package in the build environment, or set "
      "IMPLICIT_SINDY_PYTHON_CUDA_ROOT to a Python CUDA root containing "
      "nvrtc, nvJitLink, and nvPTXCompiler. Refusing to fall back to the "
      "system CUDAToolkit because mixed CUDA JIT components can produce "
      "incompatible cubins.")
  endif()
endif()
if(NOT IMPLICIT_SINDY_NVPTXCOMPILER_TARGET AND TARGET CUDA::nvptxcompiler_static)
  set(IMPLICIT_SINDY_NVPTXCOMPILER_TARGET CUDA::nvptxcompiler_static)
  set(IMPLICIT_SINDY_NVPTXCOMPILER_IS_STATIC ON)
elseif(NOT IMPLICIT_SINDY_NVPTXCOMPILER_TARGET AND TARGET CUDA::nvptxcompiler)
  set(IMPLICIT_SINDY_NVPTXCOMPILER_TARGET CUDA::nvptxcompiler)
elseif(NOT IMPLICIT_SINDY_NVPTXCOMPILER_TARGET)
  set(_implicit_sindy_nvptx_paths "")
  if(IMPLICIT_SINDY_PYTHON_CUDA_LIBRARY_DIR)
    list(APPEND _implicit_sindy_nvptx_paths "${IMPLICIT_SINDY_PYTHON_CUDA_LIBRARY_DIR}")
  endif()
  if(DEFINED CUDAToolkit_LIBRARY_DIR)
    list(APPEND _implicit_sindy_nvptx_paths "${CUDAToolkit_LIBRARY_DIR}")
  endif()
  if(DEFINED CUDAToolkit_LIBRARY_DIRS)
    list(APPEND _implicit_sindy_nvptx_paths ${CUDAToolkit_LIBRARY_DIRS})
  endif()
  if(DEFINED CUDAToolkit_ROOT)
    list(APPEND _implicit_sindy_nvptx_paths
      "${CUDAToolkit_ROOT}/lib64"
      "${CUDAToolkit_ROOT}/lib"
      "${CUDAToolkit_ROOT}/targets/${CMAKE_SYSTEM_PROCESSOR}-linux/lib"
      "${CUDAToolkit_ROOT}/targets/x86_64-linux/lib"
      "${CUDAToolkit_ROOT}/targets/aarch64-linux/lib")
  endif()
  list(REMOVE_DUPLICATES _implicit_sindy_nvptx_paths)

  find_library(IMPLICIT_SINDY_NVPTXCOMPILER_STATIC
    NAMES nvptxcompiler_static
    PATHS ${_implicit_sindy_nvptx_paths})
  find_library(IMPLICIT_SINDY_NVPTXCOMPILER_SHARED
    NAMES nvptxcompiler
    PATHS ${_implicit_sindy_nvptx_paths})

  if(IMPLICIT_SINDY_NVPTXCOMPILER_STATIC)
    add_library(implicit_sindy_nvptxcompiler_static UNKNOWN IMPORTED)
    set_target_properties(implicit_sindy_nvptxcompiler_static PROPERTIES
      IMPORTED_LOCATION "${IMPLICIT_SINDY_NVPTXCOMPILER_STATIC}")
    set(IMPLICIT_SINDY_NVPTXCOMPILER_TARGET implicit_sindy_nvptxcompiler_static)
    set(IMPLICIT_SINDY_NVPTXCOMPILER_IS_STATIC ON)
  elseif(IMPLICIT_SINDY_NVPTXCOMPILER_SHARED)
    add_library(implicit_sindy_nvptxcompiler UNKNOWN IMPORTED)
    set_target_properties(implicit_sindy_nvptxcompiler PROPERTIES
      IMPORTED_LOCATION "${IMPLICIT_SINDY_NVPTXCOMPILER_SHARED}")
    set(IMPLICIT_SINDY_NVPTXCOMPILER_TARGET implicit_sindy_nvptxcompiler)
  else()
    message(FATAL_ERROR "nvPTXCompiler library not found")
  endif()
endif()

set(IMPLICIT_SINDY_CUDA_DRIVER_TARGET "")
if(TARGET CUDA::cuda_driver)
  set(IMPLICIT_SINDY_CUDA_DRIVER_TARGET CUDA::cuda_driver)
else()
  set(_implicit_sindy_cuda_paths "")
  if(DEFINED CUDAToolkit_LIBRARY_DIR)
    list(APPEND _implicit_sindy_cuda_paths "${CUDAToolkit_LIBRARY_DIR}" "${CUDAToolkit_LIBRARY_DIR}/stubs")
  endif()
  if(DEFINED CUDAToolkit_LIBRARY_DIRS)
    foreach(path IN LISTS CUDAToolkit_LIBRARY_DIRS)
      list(APPEND _implicit_sindy_cuda_paths "${path}" "${path}/stubs")
    endforeach()
  endif()
  find_library(IMPLICIT_SINDY_CUDA_DRIVER_LIBRARY
    NAMES cuda
    PATHS ${_implicit_sindy_cuda_paths})
  if(NOT IMPLICIT_SINDY_CUDA_DRIVER_LIBRARY)
    message(FATAL_ERROR "CUDA Driver library not found")
  endif()
  add_library(implicit_sindy_cuda_driver UNKNOWN IMPORTED)
  set_target_properties(implicit_sindy_cuda_driver PROPERTIES
    IMPORTED_LOCATION "${IMPLICIT_SINDY_CUDA_DRIVER_LIBRARY}")
  set(IMPLICIT_SINDY_CUDA_DRIVER_TARGET implicit_sindy_cuda_driver)
endif()

function(add_implicit_sindy_executable target source)
  add_executable(${target} ${source})
  add_dependencies(${target} stack_ptx_descriptions_generated)
  target_include_directories(${target} PRIVATE
    "${STACK_PTX_GENERATED_DIR}"
    "${IMPLICIT_SINDY_INCLUDE_DIR}"
    "${IMPLICIT_SINDY_INTERNAL_DIR}"
    "${IMPLICIT_SINDY_TEST_DIR}/support"
    "${IMPLICIT_SINDY_THIRDPARTY_DIR}")
  target_include_directories(${target} SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  target_compile_features(${target} PRIVATE c_std_99)
  target_compile_options(${target} PRIVATE -Wall -Wextra -Wno-unused-function)
  target_compile_definitions(${target} PRIVATE
    PTX_INJECT_MAX_UNIQUE_INJECTS=${IMPLICIT_SINDY_MAX_UNIQUE_INJECTS})
  target_link_libraries(${target} PRIVATE
    ${IMPLICIT_SINDY_NVPTXCOMPILER_TARGET}
    ${IMPLICIT_SINDY_CUDA_DRIVER_TARGET}
    Threads::Threads
    m)
  if(IMPLICIT_SINDY_NVPTXCOMPILER_IS_STATIC)
    if(CMAKE_DL_LIBS)
      target_link_libraries(${target} PRIVATE ${CMAKE_DL_LIBS})
    endif()
    target_link_libraries(${target} PRIVATE stdc++)
  endif()
endfunction()

set(IMPLICIT_SINDY_SOURCES
  "${IMPLICIT_SINDY_SRC_DIR}/implicit_sindy_ast_compiler.c"
  "${IMPLICIT_SINDY_SRC_DIR}/implicit_sindy_ast_column.c"
  "${IMPLICIT_SINDY_SRC_DIR}/implicit_feature_columns_launch.c"
  "${IMPLICIT_SINDY_SRC_DIR}/implicit_feature_gram_launch.c")
if(IMPLICIT_SINDY_BUILD_SOLVER)
  list(APPEND IMPLICIT_SINDY_SOURCES
    "${IMPLICIT_SINDY_SRC_DIR}/implicit_feature_ridge_solve.c")
endif()

add_library(implicit_sindy STATIC ${IMPLICIT_SINDY_SOURCES})
add_dependencies(implicit_sindy
  stack_ptx_descriptions_generated)
if(IMPLICIT_SINDY_BUILD_SOLVER)
  add_dependencies(implicit_sindy
    implicit_feature_ridge_solve_fatbin_generated)
endif()
if(IMPLICIT_SINDY_BUILD_SOLVER)
  set_source_files_properties("${IMPLICIT_SINDY_SRC_DIR}/implicit_feature_ridge_solve.c" PROPERTIES
    OBJECT_DEPENDS "${IMPLICIT_FEATURE_RIDGE_SOLVE_FATBIN}")
endif()
target_include_directories(implicit_sindy PUBLIC
  "${IMPLICIT_SINDY_INCLUDE_DIR}")
target_include_directories(implicit_sindy PRIVATE
  "${IMPLICIT_SINDY_INTERNAL_DIR}"
  "${IMPLICIT_SINDY_THIRDPARTY_DIR}"
  "${CUBIN_FUNCTION_PATCH_INCLUDE_DIR}"
  "${STACK_PTX_GENERATED_DIR}")
if(IMPLICIT_SINDY_PYTHON_CUDA_INCLUDE_DIR)
  target_include_directories(implicit_sindy SYSTEM PRIVATE
    "${IMPLICIT_SINDY_PYTHON_CUDA_INCLUDE_DIR}")
endif()
target_include_directories(implicit_sindy SYSTEM PRIVATE
  ${CUDAToolkit_INCLUDE_DIRS})
target_compile_definitions(implicit_sindy PRIVATE
  PTX_INJECT_MAX_UNIQUE_INJECTS=${IMPLICIT_SINDY_MAX_UNIQUE_INJECTS}
  "IMPLICIT_SINDY_NVRTC_INCLUDE_DIR_INCLUDE=\"${IMPLICIT_SINDY_INCLUDE_DIR}\""
  "IMPLICIT_SINDY_NVRTC_INCLUDE_DIR_INTERNAL=\"${IMPLICIT_SINDY_INTERNAL_DIR}\""
  "IMPLICIT_SINDY_NVRTC_INCLUDE_DIR_CUDA=\"${IMPLICIT_SINDY_CUDA_DIR}\"")
if(IMPLICIT_SINDY_BUILD_SOLVER)
  target_compile_definitions(implicit_sindy PRIVATE
    "IMPLICIT_FEATURE_RIDGE_SOLVE_FATBIN_PATH=\"${IMPLICIT_FEATURE_RIDGE_SOLVE_FATBIN}\"")
endif()
target_compile_features(implicit_sindy PUBLIC c_std_99)
target_link_libraries(implicit_sindy PUBLIC
  ${IMPLICIT_SINDY_CUDA_DRIVER_TARGET})
target_link_libraries(implicit_sindy PRIVATE
  ${IMPLICIT_SINDY_NVPTXCOMPILER_TARGET}
  ${IMPLICIT_SINDY_NVRTC_TARGET}
  ${IMPLICIT_SINDY_NVJITLINK_TARGET}
  Threads::Threads
  m)
if(IMPLICIT_SINDY_NVPTXCOMPILER_IS_STATIC)
  if(CMAKE_DL_LIBS)
    target_link_libraries(implicit_sindy PRIVATE ${CMAKE_DL_LIBS})
  endif()
  target_link_libraries(implicit_sindy PRIVATE stdc++)
endif()
set_target_properties(implicit_sindy PROPERTIES
  POSITION_INDEPENDENT_CODE ON)

if(IMPLICIT_SINDY_BUILD_TESTS)
  add_implicit_sindy_executable(binary_ast_to_stack_ptx_test
    "${IMPLICIT_SINDY_TEST_DIR}/binary_ast_to_stack_ptx_test.c")
  add_implicit_sindy_executable(implicit_sindy_ast_compile_api_test
    "${IMPLICIT_SINDY_TEST_DIR}/implicit_sindy_ast_compile_api_test.c")
  add_implicit_sindy_executable(implicit_sindy_gram_cpu_reference_test
    "${IMPLICIT_SINDY_TEST_DIR}/implicit_sindy_gram_cpu_reference_test.c")
  add_implicit_sindy_executable(implicit_sindy_ast_columns_cpu_reference_test
    "${IMPLICIT_SINDY_TEST_DIR}/implicit_sindy_ast_columns_cpu_reference_test.c")
  if(IMPLICIT_SINDY_BUILD_SOLVER)
    add_implicit_sindy_executable(implicit_feature_ridge_solve_cpu_reference_test
      "${IMPLICIT_SINDY_TEST_DIR}/implicit_feature_ridge_solve_cpu_reference_test.c")
  endif()
  add_implicit_sindy_executable(implicit_sindy_ast_compile_api_bench
    "${IMPLICIT_SINDY_TEST_DIR}/bench/implicit_sindy_ast_compile_api_bench.c")
  add_implicit_sindy_executable(implicit_sindy_runtime_pipeline_bench
    "${IMPLICIT_SINDY_TEST_DIR}/bench/implicit_sindy_runtime_pipeline_bench.c")
  add_implicit_sindy_executable(implicit_sindy_reserved_cubin_runtime_bench
    "${IMPLICIT_SINDY_TEST_DIR}/bench/implicit_sindy_reserved_cubin_runtime_bench.c")

  add_test(NAME binary_ast_to_stack_ptx_test COMMAND binary_ast_to_stack_ptx_test)
  add_test(NAME implicit_sindy_ast_compile_api_test COMMAND implicit_sindy_ast_compile_api_test)
  add_test(NAME implicit_sindy_gram_cpu_reference_test COMMAND implicit_sindy_gram_cpu_reference_test)
  add_test(NAME implicit_sindy_ast_columns_cpu_reference_test COMMAND implicit_sindy_ast_columns_cpu_reference_test)
  if(IMPLICIT_SINDY_BUILD_SOLVER)
    add_test(NAME implicit_feature_ridge_solve_cpu_reference_test
      COMMAND "$<TARGET_FILE:implicit_feature_ridge_solve_cpu_reference_test>")
  endif()

  target_link_libraries(binary_ast_to_stack_ptx_test PRIVATE
    implicit_sindy)
  target_link_libraries(implicit_sindy_ast_compile_api_test PRIVATE
    implicit_sindy)
  target_link_libraries(implicit_sindy_gram_cpu_reference_test PRIVATE
    implicit_sindy)
  target_link_libraries(implicit_sindy_ast_columns_cpu_reference_test PRIVATE
    implicit_sindy)
  if(IMPLICIT_SINDY_BUILD_SOLVER)
    target_link_libraries(implicit_feature_ridge_solve_cpu_reference_test PRIVATE
      implicit_sindy)
  endif()
  target_link_libraries(implicit_sindy_ast_compile_api_bench PRIVATE
    implicit_sindy)
  target_link_libraries(implicit_sindy_runtime_pipeline_bench PRIVATE
    implicit_sindy)
  target_sources(implicit_sindy_reserved_cubin_runtime_bench PRIVATE
    "${IMPLICIT_SINDY_SRC_DIR}/implicit_feature_gram_launch.c")
  target_compile_definitions(implicit_sindy_reserved_cubin_runtime_bench PRIVATE
    "IMPLICIT_SINDY_NVRTC_INCLUDE_DIR_INCLUDE=\"${IMPLICIT_SINDY_INCLUDE_DIR}\""
    "IMPLICIT_SINDY_NVRTC_INCLUDE_DIR_INTERNAL=\"${IMPLICIT_SINDY_INTERNAL_DIR}\""
    "IMPLICIT_SINDY_NVRTC_INCLUDE_DIR_CUDA=\"${IMPLICIT_SINDY_CUDA_DIR}\"")
  target_link_libraries(implicit_sindy_reserved_cubin_runtime_bench PRIVATE
    ${IMPLICIT_SINDY_NVRTC_TARGET}
    ${IMPLICIT_SINDY_NVJITLINK_TARGET})
endif()

if(BUILD_PYTHON_BINDINGS)
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(nanobind CONFIG REQUIRED)

  nanobind_add_module(_implicit_sindy
    "${CMAKE_CURRENT_SOURCE_DIR}/python/bindings/implicit_sindy_bindings.cpp")
  target_include_directories(_implicit_sindy PRIVATE
    "${IMPLICIT_SINDY_INCLUDE_DIR}"
    "${IMPLICIT_SINDY_CUDA_INCLUDE_DIR}")
  target_compile_features(_implicit_sindy PRIVATE cxx_std_17)
  target_compile_options(_implicit_sindy PRIVATE -Wall -Wextra)
  target_link_libraries(_implicit_sindy PRIVATE
    implicit_sindy)
  if(IMPLICIT_SINDY_PYTHON_CUDA_LIBRARY_DIR)
    set_target_properties(_implicit_sindy PROPERTIES
      BUILD_RPATH "${IMPLICIT_SINDY_PYTHON_CUDA_LIBRARY_DIR}"
      INSTALL_RPATH "$ORIGIN/../../nvidia/cu13/lib;$ORIGIN/../../nvidia/cu12/lib")
  endif()
  install(TARGETS _implicit_sindy LIBRARY DESTINATION implicit_sindy/_impl)
  install(FILES "${IMPLICIT_SINDY_CUDA_DIR}/implicit_feature_gram.cuh"
    DESTINATION implicit_sindy/cuda)
endif()
