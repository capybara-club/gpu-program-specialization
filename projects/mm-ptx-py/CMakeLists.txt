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
cmake_minimum_required(VERSION 3.15...3.30)
project(mm_ptx)

# Option to enable/disable Python bindings
option(BUILD_PYTHON_BINDINGS "Build Python bindings with Nanobind" OFF)

# Check if Python bindings are enabled
if(BUILD_PYTHON_BINDINGS)
    # Find Python (required for Nanobind)
    find_package(Python QUIET COMPONENTS Interpreter Development.Module)
    if(Python_FOUND)
        # Find Nanobind
        find_package(nanobind CONFIG QUIET)
        if(nanobind_FOUND)
            message(STATUS "Python and Nanobind found, building Python bindings")
            nanobind_add_module(
                _ptx_inject
                src/bindings/ptx_inject_bindings.cpp
            )
            target_include_directories(_ptx_inject PRIVATE src/mm_ptx/include)
            install(TARGETS _ptx_inject LIBRARY DESTINATION mm_ptx/ptx_inject/_impl)
            nanobind_add_module(
                _stack_ptx
                src/bindings/stack_ptx_bindings.cpp
            )
            target_include_directories(_stack_ptx PRIVATE src/mm_ptx/include)
            install(TARGETS _stack_ptx LIBRARY DESTINATION mm_ptx/stack_ptx/_impl)
            install(DIRECTORY
                src/mm_ptx/include/
                DESTINATION mm_ptx/include
                FILES_MATCHING
                PATTERN "*.h"
                PATTERN "*.hpp"
                PATTERN "LICENSE_1_0.txt"
            )
        else()
            message(WARNING "Nanobind not found. Disabling Python bindings.")
            set(BUILD_PYTHON_BINDINGS OFF)
        endif()
    else()
        message(WARNING "Python not found. Disabling Python bindings.")
        set(BUILD_PYTHON_BINDINGS OFF)
    endif()
endif()
