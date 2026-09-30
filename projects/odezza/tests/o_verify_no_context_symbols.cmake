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
if(NOT DEFINED NM OR NM STREQUAL "")
    message(FATAL_ERROR "NM was not supplied")
endif()
if(NOT DEFINED LIBRARY OR LIBRARY STREQUAL "")
    message(FATAL_ERROR "LIBRARY was not supplied")
endif()

execute_process(
    COMMAND "${NM}" -u "${LIBRARY}"
    RESULT_VARIABLE nm_result
    OUTPUT_VARIABLE undefined_symbols
    ERROR_VARIABLE nm_error
)
if(NOT nm_result EQUAL 0)
    message(FATAL_ERROR "nm failed (${nm_result}): ${nm_error}")
endif()

if(undefined_symbols MATCHES "cuCtx[A-Za-z0-9_]*" OR
   undefined_symbols MATCHES "cuDevicePrimaryCtx[A-Za-z0-9_]*")
    message(FATAL_ERROR
        "the Odezza library imports a forbidden CUDA context-management API:\n${undefined_symbols}")
endif()

message(STATUS "Odezza imports no CUDA context-management symbols")
