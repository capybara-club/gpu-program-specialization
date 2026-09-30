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
# SPDX-FileCopyrightText: Copyright (c) 2026 Charles Durham
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.18)

project(cubin_function_patch
  VERSION 1.0.0
  LANGUAGES C)

include(CTest)

option(CUBIN_FUNCTION_PATCH_BUILD_BENCHMARKS
  "Build cubin-function-patch benchmark programs"
  ${BUILD_TESTING})

add_library(cubin_function_patch INTERFACE)
target_include_directories(cubin_function_patch INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>)

install(FILES cubin_function_patch.h DESTINATION include)
install(FILES LICENSE DESTINATION share/cubin-function-patch)

if(BUILD_TESTING)
  add_subdirectory(test)
endif()

if(CUBIN_FUNCTION_PATCH_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()
