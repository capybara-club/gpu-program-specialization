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

project(cutensor)

# Option to control cuTENSOR linking
option(USE_CUTENSOR_STANDALONE "Enable cuTENSOR downloading and linking" ON)

string(REGEX MATCH "^[0-9]+" CUDA_MAJOR_VERSION "${CMAKE_CUDA_COMPILER_VERSION}")

if (CUDA_MAJOR_VERSION EQUAL 12)
    set(CUDA_VERSION "12")        # Match your CUDA version
elseif (CUDA_MAJOR_VERSION EQUAL 13)
    set(CUDA_VERSION "13")
else()
    message(FATAL_ERROR "Unsupported CUDA version: ${CMAKE_CUDA_COMPILER_VERSION}, requires 12 or 13")
endif()

# Specify cuTENSOR version and platform
set(CUTENSOR_VERSION_MAJOR "2")
set(CUTENSOR_VERSION_SHORT "2.5.0")
set(CUTENSOR_VERSION "2.5.0.2")  # Update to desired version

set(OS "linux")           # Adjust for your OS (e.g., ubuntu20.04, rhel8)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64")
    set(ARCH "sbsa")            # Adjust if needed (e.g., aarch64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
    set(ARCH "x86_64")
else()
    message(FATAL_ERROR "Unsupported architecture detected: ${CMAKE_SYSTEM_PROCESSOR}. This project only supports ARM (arm/aarch64) or x86_64 (x86_64/amd64).")
endif()

if(NOT DEFINED DOWNLOADS_DIR OR DOWNLOADS_DIR STREQUAL "")
    set(DOWNLOADS_DIR "${CMAKE_CURRENT_LIST_DIR}/../downloads")
endif()
set(DOWNLOADS_DIR "${DOWNLOADS_DIR}" CACHE PATH "Directory for cuTENSOR downloads")
file(MAKE_DIRECTORY "${DOWNLOADS_DIR}")
# Construct the URL (based on NVIDIA's typical pattern)
set(TAR_URL "https://developer.download.nvidia.com/compute/cutensor/redist/libcutensor/${OS}-${ARCH}/libcutensor-${OS}-${ARCH}-${CUTENSOR_VERSION}_cuda${CUDA_MAJOR_VERSION}-archive.tar.xz")
set(TAR_FILE "${DOWNLOADS_DIR}/libcutensor-${OS}-${ARCH}-${CUTENSOR_VERSION}_cuda${CUDA_MAJOR_VERSION}-archive.tar.xz")
set(EXTRACT_DIR "${DOWNLOADS_DIR}")
set(CUTENSOR_ROOT "${EXTRACT_DIR}/libcutensor-${OS}-${ARCH}-${CUTENSOR_VERSION}_cuda${CUDA_MAJOR_VERSION}-archive")

if(NOT EXISTS "${TAR_FILE}")
    message(STATUS "Downloading cutensor ${CUTENSOR_VERSION}")
    message(STATUS "${TAR_URL}")
    # Download the tar file
    file(DOWNLOAD 
        "${TAR_URL}" 
        "${TAR_FILE}"
        SHOW_PROGRESS
        STATUS download_status
    )
    list(GET download_status 0 status_code)
    if(NOT status_code EQUAL 0)
        message(FATAL_ERROR "Failed to download cuTENSOR: ${download_status}")
    endif()
else()
    message(STATUS "Using existing cuTENSOR tar file: ${TAR_FILE}")
endif()

if(NOT EXISTS "${CUTENSOR_ROOT}")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xJf "${TAR_FILE}"
        WORKING_DIRECTORY "${EXTRACT_DIR}"
        RESULT_VARIABLE extract_result
    )

    if(NOT extract_result EQUAL 0)
        message(FATAL_ERROR "Failed to extract cuTENSOR")
    endif()
else()
    message(STATUS "Using existing extracted cuTENSOR: ${EXTRACT_DIR}")
endif()

set(CUTENSOR_INCLUDE "${CUTENSOR_ROOT}/include")
set(CUTENSOR_LIB "${CUTENSOR_ROOT}/lib/")

add_library(${PROJECT_NAME} INTERFACE)

# Add to your project
target_include_directories(
    ${PROJECT_NAME}
    INTERFACE
    ${CUTENSOR_INCLUDE}
)

target_link_libraries(
    ${PROJECT_NAME}
    INTERFACE
    ${CUTENSOR_LIB}/libcutensor.so
)
