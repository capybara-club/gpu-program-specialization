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
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(odezza_request LANGUAGES C)
    include(CTest)
endif()

add_library(odezza_request STATIC
    odr_json.c odr_allocations.c odr_expr.c odr_trajectories.c odr_static.c
    odr_grammar.c odr_producer.c odr_sampling.c)
target_include_directories(odezza_request PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(odezza_request PROPERTIES
    C_STANDARD 99 C_STANDARD_REQUIRED YES C_EXTENSIONS NO POSITION_INDEPENDENT_CODE ON)
if(NOT MSVC)
    target_compile_options(odezza_request PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_link_libraries(odezza_request PUBLIC m)
endif()

if(TARGET odezza)
    find_package(Threads REQUIRED)
    if(NOT TARGET odezza_sqlite)
        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../third_party/sqlite ${CMAKE_BINARY_DIR}/sqlite)
    endif()
    add_library(odezza_scoring_request STATIC odr_scoring.c odr_scoring_payload.c odr_cache.c)
    target_link_libraries(odezza_scoring_request PUBLIC odezza_request odezza odezza_sqlite CUDA::nvrtc)
    set_target_properties(odezza_scoring_request PROPERTIES
        C_STANDARD 99 C_STANDARD_REQUIRED YES C_EXTENSIONS NO POSITION_INDEPENDENT_CODE ON)
endif()

if(BUILD_TESTING)
    foreach(case parse grammar contracts mutation)
        add_executable(odr_${case}_test ../tests/frontend/${case}_test.c)
        target_link_libraries(odr_${case}_test PRIVATE odezza_request)
        set_target_properties(odr_${case}_test PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED YES C_EXTENSIONS NO)
        if(NOT MSVC)
            target_compile_options(odr_${case}_test PRIVATE -UNDEBUG)
        endif()
        add_test(NAME frontend_${case} COMMAND odr_${case}_test)
    endforeach()
endif()
