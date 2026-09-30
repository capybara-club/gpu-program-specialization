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
# Optional service layer; the core target never depends on request/search policy.
find_package(Threads REQUIRED)
    if(NOT TARGET odezza_sqlite)
        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../third_party/sqlite ${CMAKE_BINARY_DIR}/sqlite)
    endif()
add_library(odezza_runtime SHARED
    odz_job.c odz_workers.c odz_buffers.c odz_memory.c odz_status.c odz_gpu.c odz_execute.c odz_sizing.c odz_retention.c odz_report.c odz_report_bound.c
    ../frontend/odr_scoring.c ../frontend/odr_scoring_payload.c ../frontend/odr_cache.c)
target_include_directories(odezza_runtime PRIVATE . ../frontend ../core)
target_link_libraries(odezza_runtime PRIVATE odezza_request odezza_shared Threads::Threads odezza_sqlite)
set_target_properties(odezza_runtime PROPERTIES
    C_STANDARD 99 C_STANDARD_REQUIRED YES C_EXTENSIONS NO)
if(NOT MSVC)
    target_compile_options(odezza_runtime PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

if(UNIX AND NOT APPLE)
    target_link_options(odezza_runtime PRIVATE "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/odezza_runtime.exports")
    set_property(TARGET odezza_runtime APPEND PROPERTY LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/odezza_runtime.exports)
endif()
