/*
 * SPDX-FileCopyrightText: 2026 Charles Durham
 * SPDX-License-Identifier: MIT
 *
 * MIT License
 *
 * Copyright (c) 2026 Charles Durham
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
/*
 * SPDX-FileCopyrightText: 2026 Charles Durham
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#define ASSERT(X)                                                           \
    do {                                                            		\
        if (!(X)) {                                                 		\
            assert(0);                                          		    \
            printf("ASSERT:  %s %d\n", __FILE__, __LINE__);	                \
            exit(EXIT_FAILURE);                                     		\
        }                                                           		\
    } while(0)

#define kermacCheck(ans)                                                                        \
    do {                                                                                        \
        KermacResult _result = (ans);                                                           \
        if (_result != KERMAC_SUCCESS) {                                                        \
            const char *error_name = kermac_result_to_string(_result);                          \
            fprintf(stderr, "kermacCheck: %s \n  %s %d\n", error_name, __FILE__, __LINE__);     \
            assert(0);                                                                          \
            exit(1);                                                                            \
        }                                                                                       \
    } while(0)

#define ptxInjectCheck(ans)                                                                     \
    do {                                                                                        \
        PtxInjectResult _result = (ans);                                                        \
        if (_result != PTX_INJECT_SUCCESS) {                                                    \
            const char *error_name = ptx_inject_result_to_string(_result);                      \
            fprintf(stderr, "ptxInjectCheck: %s \n  %s %d\n", error_name, __FILE__, __LINE__);  \
            assert(0);                                                                          \
            exit(1);                                                                            \
        }                                                                                       \
    } while(0)

#define stackPtxCheck(ans)                                                                      \
    do {                                                                                        \
        StackPtxResult _result = (ans);                                                         \
        if (_result != STACK_PTX_SUCCESS) {                                                     \
            const char *error_name = stack_ptx_result_to_string(_result);                       \
            fprintf(stderr, "stackPtxCheck: %s \n  %s %d\n", error_name, __FILE__, __LINE__);   \
            assert(0);                                                                          \
            exit(1);                                                                            \
        }                                                                                       \
    } while(0)

#define stackPtxInjectSerializeCheck(ans)                                                       \
    do {                                                                                        \
        StackPtxResult _result = (ans);                                                         \
        if (_result != STACK_PTX_INJECT_SERIALIZE_SUCCESS) {                                    \
            const char *error_name = stack_ptx_inject_serialize_result_to_string(_result);      \
            fprintf(stderr, "stackPtxCheck: %s \n  %s %d\n", error_name, __FILE__, __LINE__);   \
            assert(0);                                                                          \
            exit(1);                                                                            \
        }                                                                                       \
    } while(0)

#define cuCheck(ans)                                                                            \
    do {                                                                                        \
        CUresult _result = (ans);                                                               \
        if (_result != CUDA_SUCCESS) {                                                          \
            const char* _error_name;                                                            \
            CUresult _name_result = cuGetErrorName(_result, &_error_name);                      \
            if (_name_result != CUDA_SUCCESS) {                                                 \
                fprintf(stderr, "cuCheck: failed to get error name: %d %d %s %d",               \
                    _result, _name_result, __FILE__, __LINE__                                   \
                );                                                                              \
                assert( 0 );                                                                    \
                exit(1);                                                                        \
            }                                                                                   \
            const char* _error_string;                                                          \
            CUresult _string_result = cuGetErrorString(_result, &_error_string);                \
            if (_string_result != CUDA_SUCCESS) {                                               \
                fprintf(stderr, "cuCheck: failed to get error string: %d %d %s %d",             \
                    _result, _string_result, __FILE__, __LINE__                                 \
                );                                                                              \
                assert( 0 );                                                                    \
                exit(1);                                                                        \
            }                                                                                   \
            fprintf(stderr, "cuCheck: %s \n  %s \n  %s %d\n",                                   \
                _error_name, _error_string, __FILE__, __LINE__                                  \
            );                                                                                  \
            assert( 0 );                                                                        \
            exit(1);                                                                            \
        }                                                                                       \
    } while(0)

#define nvptxCheck(x)                                                                                                   \
    do {                                                                                                                \
        static const char* nvPtxCompileResult_strings[] = {                                                             \
            "NVPTXCOMPILE_SUCCESS",                             \
            "NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE",       \
            "NVPTXCOMPILE_ERROR_INVALID_INPUT",                 \
            "NVPTXCOMPILE_ERROR_COMPILATION_FAILURE",           \
            "NVPTXCOMPILE_ERROR_INTERNAL",                      \
            "NVPTXCOMPILE_ERROR_OUT_OF_MEMORY",                 \
            "NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE",\
            "NVPTXCOMPILE_ERROR_UNSUPPORTED_PTX_VERSION",       \
            "NVPTXCOMPILE_ERROR_UNSUPPORTED_DEVSIDE_SYNC",      \
            "NVPTXCOMPILE_ERROR_CANCELLED"                      \
        };                                                                                                              \
        nvPTXCompileResult _result = x;                                                                                 \
        if (_result != NVPTXCOMPILE_SUCCESS) {                                                                          \
            fprintf(stderr, "nvptxCheck: %s \n  %s %d\n", nvPtxCompileResult_strings[_result], __FILE__, __LINE__);     \
            assert( 0 );                                                                                                \
            exit(1);                                                                                                    \
        }                                                                                                               \
    } while(0)

#define nvJitLinkCheck(x)                                                                                       \
    do {                                                                                                        \
        static const char* nvJitLinkResult_strings[] = {                                                        \
            [NVJITLINK_SUCCESS] =                       "NVJITLINK_SUCCESS",                                    \
            [NVJITLINK_ERROR_UNRECOGNIZED_OPTION] =     "NVJITLINK_ERROR_UNRECOGNIZED_OPTION",                  \
            [NVJITLINK_ERROR_MISSING_ARCH] =            "NVJITLINK_ERROR_MISSING_ARCH",                         \
            [NVJITLINK_ERROR_INVALID_INPUT] =           "NVJITLINK_ERROR_INVALID_INPUT",                        \
            [NVJITLINK_ERROR_PTX_COMPILE] =             "NVJITLINK_ERROR_PTX_COMPILE",                          \
            [NVJITLINK_ERROR_NVVM_COMPILE] =            "NVJITLINK_ERROR_NVVM_COMPILE",                         \
            [NVJITLINK_ERROR_INTERNAL] =                "NVJITLINK_ERROR_INTERNAL",                             \
            [NVJITLINK_ERROR_THREADPOOL] =              "NVJITLINK_ERROR_THREADPOOL",                           \
            [NVJITLINK_ERROR_UNRECOGNIZED_INPUT] =      "NVJITLINK_ERROR_UNRECOGNIZED_INPUT",                   \
            [NVJITLINK_ERROR_FINALIZE] =                "NVJITLINK_ERROR_FINALIZE",                             \
            [NVJITLINK_ERROR_NULL_INPUT] =              "NVJITLINK_ERROR_NULL_INPUT",                           \
            [NVJITLINK_ERROR_INCOMPATIBLE_OPTIONS] =    "NVJITLINK_ERROR_INCOMPATIBLE_OPTIONS",                 \
            [NVJITLINK_ERROR_INCORRECT_INPUT_TYPE] =    "NVJITLINK_ERROR_INCORRECT_INPUT_TYPE",                 \
            [NVJITLINK_ERROR_ARCH_MISMATCH] =           "NVJITLINK_ERROR_ARCH_MISMATCH",                        \
            [NVJITLINK_ERROR_OUTDATED_LIBRARY] =        "NVJITLINK_ERROR_OUTDATED_LIBRARY",                     \
            [NVJITLINK_ERROR_MISSING_FATBIN] =          "NVJITLINK_ERROR_MISSING_FATBIN",                       \
            /* [NVJITLINK_ERROR_UNRECOGNIZED_ARCH] =       "NVJITLINK_ERROR_UNRECOGNIZED_ARCH", */              \
            /* [NVJITLINK_ERROR_UNSUPPORTED_ARCH] =        "NVJITLINK_ERROR_UNSUPPORTED_ARCH",  */              \
            /* [NVJITLINK_ERROR_LTO_NOT_ENABLED] =         "NVJITLINK_ERROR_LTO_NOT_ENABLED"    */              \
        };                                                                                                      \
        nvJitLinkResult _result = x;                                                                            \
        if (_result != NVJITLINK_SUCCESS) {                                                                     \
            fprintf(stderr, "nvJitLinkCheck: %s \n  %s %d\n", nvJitLinkResult_strings[x], __FILE__, __LINE__);  \
            assert( 0 );                                                                                        \
            exit(1);                                                                                            \
        }                                                                                                       \
    } while(0)  

#define nvrtcCheck(ans)                                                                             \
    do {                                                                                            \
        nvrtcResult result = (ans);                                                                 \
        if (result != NVRTC_SUCCESS) {                                                              \
            const char* error_string = nvrtcGetErrorString(result);                                 \
            fprintf(stderr, "nvrtcCheck: %s \n  %s %d\n", error_string, __FILE__, __LINE__);        \
            assert(0);                                                                              \
            exit(1);                                                                                \
        }                                                                                           \
    } while(0)
