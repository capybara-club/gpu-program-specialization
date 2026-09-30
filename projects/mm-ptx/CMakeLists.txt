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

project(mm-ptx)

find_package(CUDAToolkit REQUIRED)

if(NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "CUDA toolkit not found. Please ensure CUDA is installed.")
endif()

enable_language(CUDA)

string(REGEX MATCH "^[0-9]+" CUDA_MAJOR_VERSION "${CMAKE_CUDA_COMPILER_VERSION}")

# cuContextCreate changed in cuda 13
if(CUDA_MAJOR_VERSION EQUAL 13) 
	add_definitions(-DCUDA_VERSION_13) 
endif()

add_library(mm_ptx_headers INTERFACE)
add_subdirectory(thirdparty/boost_subset)
target_link_libraries(mm_ptx_headers INTERFACE vendor_boost_preprocessor)

target_include_directories(mm_ptx_headers INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/tools
  ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty
)

if(CUDA_MAJOR_VERSION EQUAL 13)
    target_compile_definitions(mm_ptx_headers INTERFACE CUDA_VERSION_13)
endif()

find_package(Python3 COMPONENTS Interpreter REQUIRED)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(mm_ptx_codegen)

add_subdirectory(thirdparty)
add_subdirectory(tools)

include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

set(MMPTX_BUILD_BENCH_DEFAULT ON)
option(MMPTX_BUILD_BENCH "Build mm-ptx benchmarks" ${MMPTX_BUILD_BENCH_DEFAULT})
message(STATUS "MMPTX_BUILD_BENCH = ${MMPTX_BUILD_BENCH}")

if(MMPTX_BUILD_BENCH)
    add_subdirectory(bench)
endif()

set(MMPTX_BUILD_EXAMPLES_DEFAULT ON)
option(MMPTX_BUILD_EXAMPLES "Build mm-ptx examples" ${MMPTX_BUILD_EXAMPLES_DEFAULT})
message(STATUS "MMPTX_BUILD_EXAMPLES = ${MMPTX_BUILD_EXAMPLES}")

if(MMPTX_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
