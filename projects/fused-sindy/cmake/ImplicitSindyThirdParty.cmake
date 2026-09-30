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
include_guard(GLOBAL)

option(IMPLICIT_SINDY_ENABLE_CUSOLVERDX "Enable cuSolverDx-backed device solve kernels." ON)
option(IMPLICIT_SINDY_MATHDX_REQUIRE_CUDA13 "Reject CUDA 12 MathDx bootstrap unless a CUDA 12 package path/version is provided explicitly." OFF)

set(
    IMPLICIT_SINDY_THIRDPARTY_DOWNLOAD_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/thirdparty"
    CACHE PATH
    "Directory used for downloaded and extracted implicit-sindy third-party dependencies."
)

set(
    IMPLICIT_SINDY_MATHDX_VERSION_CUDA12
    "25.06.1"
    CACHE STRING
    "Pinned MathDx package version used when CUDA 12 is selected."
)
set(
    IMPLICIT_SINDY_MATHDX_VERSION_CUDA13
    "26.03.0"
    CACHE STRING
    "Pinned MathDx package version used when CUDA 13 is selected."
)
set(
    IMPLICIT_SINDY_MATHDX_URL
    ""
    CACHE STRING
    "Override URL for the MathDx source archive. Empty uses the official NVIDIA redistributable URL."
)
set(
    IMPLICIT_SINDY_MATHDX_SOURCE_DIR
    ""
    CACHE PATH
    "Override path to an already-extracted MathDx source tree or package root."
)

function(implicit_sindy_assert_directory_exists description path_value)
    if(NOT EXISTS "${path_value}" OR NOT IS_DIRECTORY "${path_value}")
        message(FATAL_ERROR "${description} does not exist or is not a directory: ${path_value}")
    endif()
endfunction()

function(implicit_sindy_download_and_extract_archive staged_name archive_url out_root)
    set(downloads_dir "${IMPLICIT_SINDY_THIRDPARTY_DOWNLOAD_DIR}/downloads")
    set(extracts_dir "${IMPLICIT_SINDY_THIRDPARTY_DOWNLOAD_DIR}/src")
    set(tmp_dir "${IMPLICIT_SINDY_THIRDPARTY_DOWNLOAD_DIR}/tmp/${staged_name}")
    set(root_dir "${extracts_dir}/${staged_name}")
    set(stamp_path "${root_dir}/.implicit-sindy-origin-url.txt")

    file(MAKE_DIRECTORY "${downloads_dir}")
    file(MAKE_DIRECTORY "${extracts_dir}")

    string(REGEX REPLACE ".*/" "" archive_name "${archive_url}")
    set(archive_path "${downloads_dir}/${archive_name}")

    set(need_refresh TRUE)
    if(EXISTS "${stamp_path}")
        file(READ "${stamp_path}" existing_url)
        if(existing_url STREQUAL "${archive_url}\n")
            set(need_refresh FALSE)
        endif()
    endif()

    if(need_refresh)
        file(REMOVE_RECURSE "${root_dir}" "${tmp_dir}")
        file(MAKE_DIRECTORY "${tmp_dir}")

        file(
            DOWNLOAD
            "${archive_url}"
            "${archive_path}"
            STATUS download_status
            SHOW_PROGRESS
            TLS_VERIFY ON
        )
        list(GET download_status 0 download_code)
        list(GET download_status 1 download_message)
        if(NOT download_code EQUAL 0)
            message(FATAL_ERROR "Failed to download ${archive_url}: ${download_message}")
        endif()

        file(ARCHIVE_EXTRACT INPUT "${archive_path}" DESTINATION "${tmp_dir}")
        file(GLOB extracted_entries RELATIVE "${tmp_dir}" "${tmp_dir}/*")
        list(LENGTH extracted_entries extracted_count)

        if(extracted_count EQUAL 1)
            list(GET extracted_entries 0 extracted_entry)
            if(IS_DIRECTORY "${tmp_dir}/${extracted_entry}")
                file(RENAME "${tmp_dir}/${extracted_entry}" "${root_dir}")
            else()
                file(MAKE_DIRECTORY "${root_dir}")
                file(RENAME "${tmp_dir}/${extracted_entry}" "${root_dir}/${extracted_entry}")
            endif()
        else()
            file(MAKE_DIRECTORY "${root_dir}")
            foreach(extracted_entry IN LISTS extracted_entries)
                file(RENAME "${tmp_dir}/${extracted_entry}" "${root_dir}/${extracted_entry}")
            endforeach()
        endif()

        file(WRITE "${stamp_path}" "${archive_url}\n")
        file(REMOVE_RECURSE "${tmp_dir}")
    endif()

    set(${out_root} "${root_dir}" PARENT_SCOPE)
endfunction()

function(implicit_sindy_resolve_mathdx_roots out_archive_root out_package_root out_version)
    if(NOT IMPLICIT_SINDY_ENABLE_CUSOLVERDX)
        set(${out_archive_root} "" PARENT_SCOPE)
        set(${out_package_root} "" PARENT_SCOPE)
        set(${out_version} "" PARENT_SCOPE)
        return()
    endif()

    if(CUDAToolkit_VERSION_MAJOR EQUAL 13)
        set(mathdx_cuda_major 13)
        set(mathdx_version "${IMPLICIT_SINDY_MATHDX_VERSION_CUDA13}")
    elseif(CUDAToolkit_VERSION_MAJOR EQUAL 12)
        if(IMPLICIT_SINDY_MATHDX_REQUIRE_CUDA13)
            message(FATAL_ERROR "MathDx CUDA 12 bootstrap is disabled. Detected CUDA ${CUDAToolkit_VERSION}. Set IMPLICIT_SINDY_MATHDX_REQUIRE_CUDA13=OFF, or provide IMPLICIT_SINDY_MATHDX_SOURCE_DIR/IMPLICIT_SINDY_MATHDX_VERSION_CUDA12 explicitly.")
        endif()
        set(mathdx_cuda_major 12)
        set(mathdx_version "${IMPLICIT_SINDY_MATHDX_VERSION_CUDA12}")
    else()
        message(FATAL_ERROR "MathDx bootstrap only supports CUDA 12 or 13. Detected CUDA ${CUDAToolkit_VERSION}.")
    endif()

    if(IMPLICIT_SINDY_MATHDX_SOURCE_DIR)
        set(mathdx_root "${IMPLICIT_SINDY_MATHDX_SOURCE_DIR}")
    else()
        foreach(candidate
                "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/mathdx-cuda${mathdx_cuda_major}-${mathdx_version}")
            if(EXISTS "${candidate}")
                set(mathdx_root "${candidate}")
                break()
            endif()
        endforeach()

        if(NOT mathdx_root)
            if(IMPLICIT_SINDY_MATHDX_URL)
                set(mathdx_url "${IMPLICIT_SINDY_MATHDX_URL}")
            else()
                set(mathdx_url "https://developer.nvidia.com/downloads/compute/cuSOLVERDx/redist/cuSOLVERDx/cuda${mathdx_cuda_major}/nvidia-mathdx-${mathdx_version}-cuda${mathdx_cuda_major}.tar.gz")
            endif()
            implicit_sindy_download_and_extract_archive("mathdx-cuda${mathdx_cuda_major}-${mathdx_version}" "${mathdx_url}" mathdx_root)
        endif()
    endif()

    implicit_sindy_assert_directory_exists("MathDx root" "${mathdx_root}")

    string(REGEX REPLACE "^([0-9]+)\\.([0-9]+).*$" "\\1.\\2" mathdx_series "${mathdx_version}")
    set(candidate_package_root "${mathdx_root}/nvidia/mathdx/${mathdx_series}")
    if(EXISTS "${candidate_package_root}/include" AND EXISTS "${candidate_package_root}/lib/cmake/cusolverdx")
        set(mathdx_package_root "${candidate_package_root}")
    elseif(EXISTS "${mathdx_root}/include" AND EXISTS "${mathdx_root}/lib/cmake/cusolverdx")
        set(mathdx_package_root "${mathdx_root}")
    else()
        message(FATAL_ERROR "Unable to locate MathDx package root under ${mathdx_root}.")
    endif()

    implicit_sindy_assert_directory_exists("MathDx include directory" "${mathdx_package_root}/include")
    implicit_sindy_assert_directory_exists("MathDx CMake directory" "${mathdx_package_root}/lib/cmake/cusolverdx")

    set(${out_archive_root} "${mathdx_root}" PARENT_SCOPE)
    set(${out_package_root} "${mathdx_package_root}" PARENT_SCOPE)
    set(${out_version} "${mathdx_version}" PARENT_SCOPE)
endfunction()

function(implicit_sindy_setup_thirdparty)
    if(NOT IMPLICIT_SINDY_ENABLE_CUSOLVERDX)
        set(IMPLICIT_SINDY_THIRDPARTY_INCLUDE_DIRS "" PARENT_SCOPE)
        set(IMPLICIT_SINDY_MATHDX_PACKAGE_ROOT "" PARENT_SCOPE)
        return()
    endif()

    implicit_sindy_resolve_mathdx_roots(mathdx_archive_root mathdx_package_root mathdx_version)

    set(mathdx_cutlass_root "${mathdx_package_root}/external/cutlass")
    if(NOT EXISTS "${mathdx_cutlass_root}/include/cute/tensor.hpp")
        message(FATAL_ERROR "MathDx bundled CUTLASS not found under ${mathdx_cutlass_root}")
    endif()

    set(thirdparty_include_dirs
        "${mathdx_package_root}/include"
        "${mathdx_cutlass_root}/include")

    set(cusolverdx_CUTLASS_ROOT "${mathdx_cutlass_root}")
    set(commondx_DIR "${mathdx_package_root}/lib/cmake/commondx" CACHE PATH "commondx package directory" FORCE)
    set(cusolverdx_DIR "${mathdx_package_root}/lib/cmake/cusolverdx" CACHE PATH "cusolverdx package directory" FORCE)
    find_package(commondx CONFIG REQUIRED PATHS "${mathdx_package_root}/lib/cmake/commondx" NO_DEFAULT_PATH)
    find_package(cusolverdx CONFIG REQUIRED PATHS "${mathdx_package_root}/lib/cmake/cusolverdx" NO_DEFAULT_PATH)

    if(NOT TARGET implicit_sindy_thirdparty_cusolverdx)
        add_library(implicit_sindy_thirdparty_cusolverdx INTERFACE)
    endif()
    target_link_libraries(implicit_sindy_thirdparty_cusolverdx INTERFACE cusolverdx::cusolverdx)

    list(REMOVE_DUPLICATES thirdparty_include_dirs)

    set(IMPLICIT_SINDY_MATHDX_ROOT "${mathdx_archive_root}" PARENT_SCOPE)
    set(IMPLICIT_SINDY_MATHDX_PACKAGE_ROOT "${mathdx_package_root}" PARENT_SCOPE)
    set(IMPLICIT_SINDY_MATHDX_VERSION "${mathdx_version}" PARENT_SCOPE)
    set(IMPLICIT_SINDY_THIRDPARTY_INCLUDE_DIRS "${thirdparty_include_dirs}" PARENT_SCOPE)

    message(STATUS "implicit-sindy: using MathDx ${mathdx_version} from ${mathdx_package_root}")
endfunction()
