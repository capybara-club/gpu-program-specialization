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
if(NOT DEFINED ROOT)
    message(FATAL_ERROR "ROOT is required")
endif()

file(GLOB_RECURSE C_SOURCES
    "${ROOT}/core/*.c"
    "${ROOT}/core/*.h"
    "${ROOT}/src/*.c"
    "${ROOT}/src/*.h"
    "${ROOT}/app/*.c"
    "${ROOT}/app/*.h"
    "${ROOT}/tests/*.c"
    "${ROOT}/tests/*.h"
    "${ROOT}/scratch/*.c"
    "${ROOT}/scratch/*.h"
)

foreach(SOURCE IN LISTS C_SOURCES)
    file(READ "${SOURCE}" CONTENTS)
    string(REGEX MATCH "(^|[^A-Za-z0-9_])goto([^A-Za-z0-9_]|$)" FOUND "${CONTENTS}")
    if(FOUND)
        message(FATAL_ERROR "goto is not allowed in C99 source: ${SOURCE}")
    endif()
endforeach()
