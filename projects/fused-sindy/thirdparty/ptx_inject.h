/*
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
 *
 * SPDX-License-Identifier: MIT
 */

 /**
 * @file
 * @brief This file contains all headers and source for PTX Inject.
 */

#ifndef PTX_INJECT_H_INCLUDE
#define PTX_INJECT_H_INCLUDE

#define PTX_INJECT_VERSION_MAJOR 1 //!< PTX Inject major version.
#define PTX_INJECT_VERSION_MINOR 1 //!< PTX Inject minor version.
#define PTX_INJECT_VERSION_PATCH 1 //!< PTX Inject patch version.

/**
 * \brief String representation of the PTX Inject library version (e.g., "1.1.1").
 */
#define PTX_INJECT_VERSION_STRING "1.1.1"

#define PTX_INJECT_VERSION (PTX_INJECT_VERSION_MAJOR * 10000 + PTX_INJECT_VERSION_MINOR * 100 + PTX_INJECT_VERSION_PATCH)

#if !defined(__CUDACC_RTC__)

#ifdef __cplusplus
#define PTX_INJECT_PUBLIC_DEC extern "C"
#define PTX_INJECT_PUBLIC_DEF extern "C"
#else
#define PTX_INJECT_PUBLIC_DEC extern
#define PTX_INJECT_PUBLIC_DEF
#endif

#include <stddef.h>

/**
 * \mainpage PTX Inject: A library for injecting PTX into compiled CUDA code.
 *
 * \section usage Usage
 *
 * PTX Inject is a single-header, C99 library. Define PTX_INJECT_IMPLEMENTATION in
 * exactly one translation unit to compile the implementation:
 *
 * \code
 * #define PTX_INJECT_IMPLEMENTATION
 * #include <ptx_inject.h>
 * \endcode
 *
 * Define PTX_INJECT_DEBUG to turn errors into asserts at the call site:
 *
 * \code
 * #define PTX_INJECT_DEBUG
 * #define PTX_INJECT_IMPLEMENTATION
 * #include <ptx_inject.h>
 * \endcode
 *
 * Typical workflow:
 * 1) In CUDA code compiled with nvcc, mark an inject site with PTX_INJECT and
 *    PTX_IN/PTX_OUT/PTX_MOD operand descriptors.
 * 2) Compile the CUDA source to PTX (nvcc or nvrtc). The PTX contains
 *    PTX_INJECT_START/END markers and per-operand metadata.
 * 3) Call ptx_inject_create on that PTX string to parse inject sites.
 * 4) Query variable register names and build PTX stubs.
 * 5) Call ptx_inject_render_ptx to insert stubs and produce injected PTX.
 *
 * You can override PTX_INJECT_MAX_UNIQUE_INJECTS to change the maximum number
 * of unique inject sites. For custom operand types, define PTX_TYPE_INFO_<TOKEN>
 * entries (or disable defaults with PTX_INJECT_NO_DEFAULT_TYPES).
 */

/**
 * \brief PTX Inject status return codes.
 *
 * \details All PTX Inject API functions return a PtxInjectResult value.
 */
typedef enum {
    /** PTX Inject Operation was successful */
    PTX_INJECT_SUCCESS                              = 0,
    /** PTX Inject formatting is wrong.*/
    PTX_INJECT_ERROR_FORMATTING                     = 1,
    /** The buffer passed in is not large enough.*/
    PTX_INJECT_ERROR_INSUFFICIENT_BUFFER            = 2,
    /** An internal error occurred.*/
    PTX_INJECT_ERROR_INTERNAL                       = 3,
    /** An value passed to the function is wrong.*/
    PTX_INJECT_ERROR_INVALID_INPUT                  = 4,
    /** The amount of injects found in the file exceeds the maximum.*/
    PTX_INJECT_ERROR_MAX_UNIQUE_INJECTS_EXCEEDED    = 5,
    /** The amount of stubs passed in does not match the amount of injects found in the file.*/
    PTX_INJECT_ERROR_WRONG_NUM_STUBS                = 6,
    /** The index passed in is out of bounds of the range of values being indexed.*/
    PTX_INJECT_ERROR_OUT_OF_BOUNDS_IDX              = 7,
    /** An inject site found in the file has a different signature than another inject site found with the same name.*/
    PTX_INJECT_ERROR_INCONSISTENT_INJECTION         = 8,
    /** Inject name not found.*/
    PTX_INJECT_ERROR_INJECTION_NAME_NOT_FOUND       = 9,
    /** Inject arg name not found.*/
    PTX_INJECT_ERROR_INJECTION_ARG_NAME_NOT_FOUND   = 10,
    /** PTX Inject is out of memory, malloc failed. */
    PTX_INJECT_ERROR_OUT_OF_MEMORY                  = 11,
    /** The number of result enums.*/
    PTX_INJECT_RESULT_NUM_ENUMS
} PtxInjectResult;

/**
 * \brief PTX Inject mutation types.
 * 
 * \details Specifies how the inline PTX treats a variable: output-only, read-write, or input-only.
 */
typedef enum {
    PTX_INJECT_MUT_TYPE_OUT,
    PTX_INJECT_MUT_TYPE_MOD,
    PTX_INJECT_MUT_TYPE_IN,
    PTX_INJECT_MUT_TYPE_NUM_ENUMS
} PtxInjectMutType;

struct PtxInjectHandleImpl;
/**
 * \brief Opaque structure representing a PTX Inject handle.
 */
typedef struct PtxInjectHandleImpl* PtxInjectHandle;

/**
 * \brief Converts a PtxInjectResult enum value to a human-readable string.
 *
 * \param[in] result The PtxInjectResult enum value to convert.
 * \return A null-terminated string describing the result or
 * "PTX_INJECT_ERROR_INVALID_RESULT_ENUM" if `result` is out of bounds.
 * \remarks Thread-safe.
 */
PTX_INJECT_PUBLIC_DEF const char* ptx_inject_result_to_string(PtxInjectResult result);

/**
 * \brief Creates a PTX injection context from PTX source code.
 *
 * \details The PTX must contain PTX_INJECT_START/END markers emitted by PTX_INJECT
 * in CUDA code. The resulting handle stores inject metadata and a stub buffer that
 * will be used by ptx_inject_render_ptx.
 *
 * \param[out] handle Pointer to a PtxInjectHandle to initialize.
 * \param[in] processed_ptx_src Null-terminated string containing PTX source code
 * with PTX Inject markers.
 * \return PtxInjectResult indicating success or an error code.
 * \remarks Blocking, thread-safe.
 */
PTX_INJECT_PUBLIC_DEC PtxInjectResult ptx_inject_create(PtxInjectHandle* handle, const char* processed_ptx_src);

/**
 * \brief Destroys a PTX injection context and frees associated resources.
 *
 * \param[in] handle The PtxInjectHandle to destroy.
 * \return PtxInjectResult indicating success or an error code.
 * \remarks Blocking, thread-safe.
 */
PTX_INJECT_PUBLIC_DEC PtxInjectResult ptx_inject_destroy(PtxInjectHandle handle);

/**
 * \brief Gets the number of unique injects found in the PTX.
 *
 * \param[in] handle The PtxInjectHandle.
 * \param[out] num_injects_out The number of unique inject names found.
 * \return PtxInjectResult indicating success or an error code.
 * \remarks Blocking, thread-safe.
 */
PTX_INJECT_PUBLIC_DEC PtxInjectResult ptx_inject_num_injects(const PtxInjectHandle handle, size_t* num_injects_out);

/**
 * \brief Gets information about an inject by name.
 *
 * \param[in] handle The PtxInjectHandle.
 * \param[in] inject_name The name of the inject as specified in PTX_INJECT.
 * \param[out] inject_idx_out The index of the inject. Use this index to place a stub
 * in the array passed to ptx_inject_render_ptx. Can be NULL to ignore.
 * \param[out] inject_num_args_out The number of variables declared in the inject. Can be NULL to ignore.
 * \param[out] inject_num_sites_out The number of sites where this inject appears (e.g., inlined or unrolled).
 * Can be NULL to ignore.
 * \return PtxInjectResult indicating success or an error code.
 * \remarks Blocking, thread-safe.
 */
PTX_INJECT_PUBLIC_DEC PtxInjectResult ptx_inject_inject_info_by_name(
    const PtxInjectHandle handle,
    const char* inject_name,
    size_t* inject_idx_out, 
    size_t* inject_num_args_out,
    size_t* inject_num_sites_out 
);

/**
 * \brief Gets information about an inject by index.
 *
 * \param[in] handle The PtxInjectHandle.
 * \param[in] inject_idx Index of the inject, in the range [0, num_injects).
 * \param[out] inject_name_out The inject name. Can be NULL to ignore.
 * \param[out] inject_num_args_out The number of variables declared in the inject. Can be NULL to ignore.
 * \param[out] inject_num_sites_out The number of sites where this inject appears. Can be NULL to ignore.
 * \return PtxInjectResult indicating success or an error code.
 * \remarks Blocking, thread-safe.
 */
PTX_INJECT_PUBLIC_DEC PtxInjectResult ptx_inject_inject_info_by_index(
    const PtxInjectHandle handle,
    size_t inject_idx,
    const char** inject_name_out,
    size_t* inject_num_args_out,
    size_t* inject_num_sites_out
);

/**
 * \brief Gets information about a variable by inject index and variable name.
 *
 * \param[in] handle The PtxInjectHandle.
 * \param[in] inject_idx Index of the inject, in the range [0, num_injects).
 * \param[in] inject_variable_name Variable name as specified in PTX_INJECT.
 * \param[out] inject_variable_arg_idx_out Variable index within the inject. Can be NULL to ignore.
 * \param[out] inject_variable_register_name_out Stable PTX register name (e.g., "_x0"). Can be NULL to ignore.
 * \param[out] inject_variable_mut_type_out Mutation type (out/mod/in). Can be NULL to ignore.
 * \param[out] inject_variable_register_type_name_out PTX register type string (e.g., "f32"). Can be NULL to ignore.
 * \param[out] inject_variable_data_type_name_out Data type token string (e.g., "F32"). Can be NULL to ignore.
 * \return PtxInjectResult indicating success or an error code.
 * \remarks Blocking, thread-safe. Stable register names are consistent across duplicated sites.
 */
PTX_INJECT_PUBLIC_DEC PtxInjectResult ptx_inject_variable_info_by_name(
    const PtxInjectHandle handle,
    size_t inject_idx,
    const char* inject_variable_name,
    size_t* inject_variable_arg_idx_out,
    const char** inject_variable_register_name_out,
    PtxInjectMutType* inject_variable_mut_type_out,
    const char** inject_variable_register_type_name_out,
    const char** inject_variable_data_type_name_out
);

/**
 * \brief Gets information about a variable by inject index and variable index.
 *
 * \param[in] handle The PtxInjectHandle.
 * \param[in] inject_idx Index of the inject, in the range [0, num_injects).
 * \param[in] inject_variable_arg_idx Variable index within the inject.
 * \param[out] inject_variable_name_out Variable name as specified in PTX_INJECT. Can be NULL to ignore.
 * \param[out] inject_variable_register_name_out Stable PTX register name (e.g., "_x0"). Can be NULL to ignore.
 * \param[out] inject_variable_mut_type_out Mutation type (out/mod/in). Can be NULL to ignore.
 * \param[out] inject_variable_register_type_name_out PTX register type string (e.g., "f32"). Can be NULL to ignore.
 * \param[out] inject_variable_data_type_name_out Data type token string (e.g., "F32"). Can be NULL to ignore.
 * \return PtxInjectResult indicating success or an error code.
 * \remarks Blocking, thread-safe.
 */
PTX_INJECT_PUBLIC_DEC PtxInjectResult ptx_inject_variable_info_by_index(
    const PtxInjectHandle handle,
    size_t inject_idx,
    size_t inject_variable_arg_idx,
    const char** inject_variable_name_out,
    const char** inject_variable_register_name_out,
    PtxInjectMutType* inject_variable_mut_type_out,
    const char** inject_variable_register_type_name_out,
    const char** inject_variable_data_type_name_out
);

/**
 * \brief Renders PTX code with injected stubs, supporting measure-and-allocate usage.
 *
 * \param[in] handle The PtxInjectHandle.
 * \param[in] ptx_stubs Array of null-terminated PTX snippets to inject. The array is ordered by inject index.
 * \param[in] num_ptx_stubs Number of entries in ptx_stubs.
 * \param[out] rendered_ptx_buffer Buffer to store the rendered PTX. Can be NULL to measure required size.
 * \param[in] rendered_ptx_buffer_size Size of rendered_ptx_buffer in bytes. Ignored if rendered_ptx_buffer is NULL.
 * \param[out] rendered_ptx_bytes_written_out Bytes written, or required size if rendered_ptx_buffer is NULL.
 * \return PtxInjectResult indicating success or an error code.
 * \remarks Blocking, thread-safe. To measure required size, pass NULL for rendered_ptx_buffer and
 * 0 for rendered_ptx_buffer_size, then allocate and call again.
 */
PTX_INJECT_PUBLIC_DEC PtxInjectResult ptx_inject_render_ptx(
    const PtxInjectHandle handle,
    const char* const* ptx_stubs,
    size_t num_ptx_stubs,
    char* rendered_ptx_buffer,
    size_t rendered_ptx_buffer_size,
    size_t* rendered_ptx_bytes_written_out
);

#endif /* PTX_INJECT_H_INCLUDE */

#ifdef PTX_INJECT_IMPLEMENTATION
#ifndef PTX_INJECT_IMPLEMENTATION_ONCE
#define PTX_INJECT_IMPLEMENTATION_ONCE

#define _PTX_INJECT_ALIGNMENT 16 // Standard malloc alignment
#define _PTX_INJECT_ALIGNMENT_UP(size, align) (((size) + (align) - 1) & ~((align) - 1))

#include <stdarg.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifdef PTX_INJECT_DEBUG
#include <assert.h>
#define _PTX_INJECT_ERROR(ans)                                                                      \
    do {                                                                                            \
        PtxInjectResult _result = (ans);                                                            \
        const char* error_name = ptx_inject_result_to_string(_result);                              \
        fprintf(stderr, "PTX_INJECT_ERROR: %s \n  %s %d\n", error_name, __FILE__, __LINE__);        \
        assert(0);                                                                                  \
        exit(1);                                                                                    \
    } while(0);

#define _PTX_INJECT_CHECK_RET(ans)                                                                  \
    do {                                                                                            \
        PtxInjectResult _result = (ans);                                                            \
        if (_result != PTX_INJECT_SUCCESS) {                                                        \
            const char* error_name = ptx_inject_result_to_string(_result);                          \
            fprintf(stderr, "PTX_INJECT_CHECK: %s \n  %s %d\n", error_name, __FILE__, __LINE__);    \
            assert(0);                                                                              \
            exit(1);                                                                                \
            return _result;                                                                         \
        }                                                                                           \
    } while(0);
#else
#define _PTX_INJECT_ERROR(ans)                              \
    do {                                                    \
        PtxInjectResult _result = (ans);                    \
        return _result;                                     \
    } while(0);

#define _PTX_INJECT_CHECK_RET(ans)                          \
    do {                                                    \
        PtxInjectResult _result = (ans);                    \
        if (_result != PTX_INJECT_SUCCESS) return _result;  \
    } while(0);
#endif // PTX_INJECT_DEBUG

static const char* const _ptx_inject_ptx_header_str_start =             "// PTX_INJECT_START";
static const char* const _ptx_inject_ptx_header_str_end =               "// PTX_INJECT_END";

#ifndef PTX_INJECT_MAX_UNIQUE_INJECTS
#define PTX_INJECT_MAX_UNIQUE_INJECTS 1024
#endif // PTX_INJECT_MAX_UNIQUE_INJECTS

typedef struct {
    PtxInjectMutType mut_type;
    const char* name;
    const char* register_type_name;
    const char* data_type_name;
    const char* register_name;
} PtxInjectInjectionArg;

typedef struct {
    const char* name;
    size_t name_length;
    PtxInjectInjectionArg* args;
    size_t num_args;
    size_t num_sites;
    size_t unique_idx;
} PtxInjectInjection;

struct PtxInjectHandleImpl {
    // All unique injects found in ptx
    PtxInjectInjection* injects;
    size_t num_injects;

    // Sites where a unique inject is found in one or more places
    const char** inject_sites;
    size_t* inject_site_to_inject_idx;
    size_t num_inject_sites;

    // All unique injection args stored in one array
    PtxInjectInjectionArg* inject_args;
    size_t num_inject_args;

    // All buffers that will be copied in to rendered ptx
    // Injected ptx will be copied between these stubs
    char* stub_buffer;
    size_t stub_buffer_size;

    // All names from injects and inject_args in one blob
    char* names_blob;
    size_t names_blob_size;
};

PTX_INJECT_PUBLIC_DEF
const char* 
ptx_inject_result_to_string(
    PtxInjectResult result
) {
    switch(result) {
        case PTX_INJECT_SUCCESS:                            return "PTX_INJECT_SUCCESS";
        case PTX_INJECT_ERROR_FORMATTING:                   return "PTX_INJECT_ERROR_FORMATTING";
        case PTX_INJECT_ERROR_INSUFFICIENT_BUFFER:          return "PTX_INJECT_ERROR_INSUFFICIENT_BUFFER";
        case PTX_INJECT_ERROR_INTERNAL:                     return "PTX_INJECT_ERROR_INTERNAL";
        case PTX_INJECT_ERROR_INVALID_INPUT:                return "PTX_INJECT_ERROR_INVALID_INPUT";
        case PTX_INJECT_ERROR_MAX_UNIQUE_INJECTS_EXCEEDED:  return "PTX_INJECT_ERROR_MAX_UNIQUE_INJECTS_EXCEEDED";
        case PTX_INJECT_ERROR_WRONG_NUM_STUBS:              return "PTX_INJECT_ERROR_WRONG_NUM_STUBS";
        case PTX_INJECT_ERROR_OUT_OF_BOUNDS_IDX:            return "PTX_INJECT_ERROR_OUT_OF_BOUNDS_IDX";
        case PTX_INJECT_ERROR_INCONSISTENT_INJECTION:       return "PTX_INJECT_ERROR_INCONSISTENT_INJECTION";
        case PTX_INJECT_ERROR_INJECTION_NAME_NOT_FOUND:     return "PTX_INJECT_ERROR_INJECTION_NAME_NOT_FOUND";
        case PTX_INJECT_ERROR_INJECTION_ARG_NAME_NOT_FOUND: return "PTX_INJECT_ERROR_INJECTION_ARG_NAME_NOT_FOUND";
        case PTX_INJECT_ERROR_OUT_OF_MEMORY:                return "PTX_INJECT_ERROR_OUT_OF_MEMORY";
        case PTX_INJECT_RESULT_NUM_ENUMS: break;
    }
    return "PTX_INJECT_ERROR_INVALID_RESULT_ENUM";
}

static
inline
PtxInjectResult
_ptx_inject_snprintf_append(
    char* buffer, 
    size_t buffer_size, 
    size_t* total_bytes_ref, 
    const char* fmt,
    ...
) {
    va_list args;
    va_start(args, fmt);
    if (buffer && buffer_size < *total_bytes_ref) {
        buffer = NULL;
    }
    int bytes = 
		vsnprintf(
			buffer ? buffer + *total_bytes_ref : NULL, 
			buffer ? buffer_size - *total_bytes_ref : 0, 
			fmt, 
			args
		);
    va_end(args);
    if (bytes < 0) {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_INTERNAL );
    }
    *total_bytes_ref += (size_t)bytes;
    return PTX_INJECT_SUCCESS;
}

static
inline
PtxInjectResult
_ptx_inject_get_name_to_newline_trim_whitespace(
    const char* input,
    size_t* start,
    size_t* length
) {
    size_t i = 0;

    if (input[i] != ' ' && input[i] != '\t') {
        _PTX_INJECT_ERROR(  PTX_INJECT_ERROR_FORMATTING );
    }

    i++;

    while (input[i] == ' ' || input[i] == '\t') {
        i++;
    }

    if (input[i] == '\n' || input[i] == '\0') {
        _PTX_INJECT_ERROR(  PTX_INJECT_ERROR_FORMATTING );
    }

    *start = i;
    size_t len = 0;

    while (true) {
        while (input[i] != ' ' && input[i] != '\t' && input[i] != '\n' && input[i] != '\0') {
            i++;
        }

        len = i - *start;

        if (input[i] == '\n' || input[i] == '\0') {
            break;
        }

        while (input[i] == ' ' || input[i] == '\t') {
            i++;
        }

        if (input[i] == '\n' || input[i] == '\0') {
            break;
        }
    }

    if (input[i] != '\n') {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_FORMATTING );
    }

    *length = len;
    return PTX_INJECT_SUCCESS;
}

static
inline
bool
_ptx_inject_is_whitespace(
    char c
) {
    return (c == ' ' || c == '\t');
}


static
inline
const char* 
_ptx_inject_str_whitespace(
    const char* str
) {
    char c = *str;
	const char* str_ptr = str;

    while (_ptx_inject_is_whitespace(c)) {
        str_ptr++;
        c = *str_ptr;
    }

	return str_ptr;
}

static
inline
const char* 
_ptx_inject_str_whitespace_to_newline(
    const char* str
) {
    char c = *str;
	const char* str_ptr = str;

    while (c != '\n' && c != '\0') {
        str_ptr++;
        c = *str_ptr;
    }

	return str_ptr;
}

static
bool
_ptx_inject_strcmp_advance(
    const char** ptr_ref, 
    const char* needle
) {
	const char* ptr = *ptr_ref;
	if (strncmp(ptr, needle, strlen(needle)) == 0) {
		ptr += strlen(needle);
		*ptr_ref = ptr;
		return true;
	}
	return false;
}

static
inline
PtxInjectResult
_ptx_inject_get_name_trim_whitespace(
    const char* input,
    const char** name,
    size_t* length_out,
    const char** end
) {
    const char* ptr = input;

    if (*ptr != ' ' && *ptr != '\t') {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_FORMATTING );
    }
    ptr++;

    while (*ptr == ' ' || *ptr == '\t') {
        ptr++;
    }

    if (*ptr == '\n' || *ptr == '\0') {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_FORMATTING );
    }

    *name = ptr;
    size_t length = 0;
    while (*ptr != ' ' && *ptr != '\t' && *ptr != '\n' && *ptr != '\0') {
        ptr++;
        length++;
    }
    *length_out = length;
    *end = ptr;

    return PTX_INJECT_SUCCESS;
}

static
inline
PtxInjectResult
_ptx_inject_ptx_parse_argument(
    const char* argument_start,
    const char** register_name_ref,
    size_t* register_name_length_ref,
    PtxInjectMutType* mut_type_ref,
    const char** register_type_name_ref,
    size_t* register_type_name_length_ref,
    const char** data_type_field_name_ref,
    size_t* data_type_field_name_length_ref,
    const char** argument_name_ref,
    size_t* argument_name_length_ref,
    const char** argument_end_ref,
    bool* found_argument
) {
    const char* argument_ptr = argument_start;
    if(_ptx_inject_strcmp_advance(&argument_ptr, _ptx_inject_ptx_header_str_end)) {
        *found_argument = false;
        *argument_end_ref = argument_ptr;
        return PTX_INJECT_SUCCESS;
    }

    *found_argument = true;
    if(!_ptx_inject_strcmp_advance(&argument_ptr, "//")) {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_FORMATTING );
    }

    _PTX_INJECT_CHECK_RET(
        _ptx_inject_get_name_trim_whitespace(
            argument_ptr,
            register_name_ref, 
            register_name_length_ref,
            &argument_ptr
        )
    );

    if (*argument_ptr++ != ' ') {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_FORMATTING );
    }

    char mut_type_char = *argument_ptr++;
    switch(mut_type_char) {
        case 'm': *mut_type_ref = PTX_INJECT_MUT_TYPE_MOD; break;
        case 'o': *mut_type_ref = PTX_INJECT_MUT_TYPE_OUT; break;
        case 'i': *mut_type_ref = PTX_INJECT_MUT_TYPE_IN; break;
        default:
            _PTX_INJECT_ERROR( PTX_INJECT_ERROR_FORMATTING );
    }

    // if (*argument_ptr++ != ' ') {
    //     _PTX_INJECT_ERROR( PTX_INJECT_ERROR_FORMATTING );
    // }

    _PTX_INJECT_CHECK_RET(
        _ptx_inject_get_name_trim_whitespace(
            argument_ptr,
            register_type_name_ref, 
            register_type_name_length_ref,
            &argument_ptr
        )
    );

    _PTX_INJECT_CHECK_RET(
        _ptx_inject_get_name_trim_whitespace(
            argument_ptr,
            data_type_field_name_ref, 
            data_type_field_name_length_ref,
            &argument_ptr
        )
    );

    size_t var_name_start;
    size_t var_name_length;
    _PTX_INJECT_CHECK_RET(
        _ptx_inject_get_name_to_newline_trim_whitespace(
            argument_ptr, 
            &var_name_start, 
            &var_name_length
        )
    );

    *argument_name_ref = argument_ptr + var_name_start;
    *argument_name_length_ref = var_name_length;

    argument_ptr += var_name_start + var_name_length;
    argument_ptr = _ptx_inject_str_whitespace_to_newline(argument_ptr);
    if(*argument_ptr != '\n') {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_FORMATTING );
    }
    argument_ptr++;

    *argument_end_ref = argument_ptr;

    return PTX_INJECT_SUCCESS;
}

static
inline
PtxInjectResult
_ptx_inject_create(
    struct PtxInjectHandleImpl* ptx_inject,
    const char* processed_ptx_src
) {
    const char* src_ptr = processed_ptx_src;

    size_t stubs_bytes_written = 0;
    size_t names_blob_bytes_written = 0;

    size_t num_unique_injects = 0;
    size_t num_unique_inject_args = 0;
    size_t num_inject_sites = 0;

    PtxInjectInjection* unique_injects = ptx_inject->injects;

    while(true) {
        const char* start_of_inject = strstr(src_ptr, _ptx_inject_ptx_header_str_start);
        
        if (start_of_inject == NULL) break;

        // If character before this is a tab, we should not copy it to keep injected PTX nice looking
        size_t maybe_clobber_tab = 0;
        if (start_of_inject != processed_ptx_src && *(start_of_inject-1) == '\t') {
            maybe_clobber_tab = 1;
        }
        ptx_inject->num_inject_sites++;

        _PTX_INJECT_CHECK_RET(
            _ptx_inject_snprintf_append(
                ptx_inject->stub_buffer,
                ptx_inject->stub_buffer_size,
                &stubs_bytes_written,
                "%.*s",
                start_of_inject - src_ptr - maybe_clobber_tab,
                src_ptr
            )
        );

        src_ptr = start_of_inject + strlen(_ptx_inject_ptx_header_str_start);

        size_t inject_name_start, inject_name_length;
        _PTX_INJECT_CHECK_RET( 
            _ptx_inject_get_name_to_newline_trim_whitespace(
                src_ptr, 
                &inject_name_start, 
                &inject_name_length
            )
        );

        const char* const inject_name = src_ptr + inject_name_start;
        src_ptr += inject_name_start + inject_name_length;
        src_ptr = _ptx_inject_str_whitespace_to_newline(src_ptr);
        if(*src_ptr != '\n') {
            _PTX_INJECT_ERROR( PTX_INJECT_ERROR_FORMATTING );
        }
        src_ptr++;

        PtxInjectInjection* unique_inject_site;
        bool is_unique = true;
        for (size_t i = 0; i < num_unique_injects; i++) {
            PtxInjectInjection* this_unique_inject = &unique_injects[i];
            if (this_unique_inject->name_length == inject_name_length &&
                strncmp(this_unique_inject->name, inject_name, inject_name_length) == 0
            ) {
                is_unique = false;
                unique_inject_site = this_unique_inject;
            }
        }
        if (is_unique) {
            if (num_unique_injects >= PTX_INJECT_MAX_UNIQUE_INJECTS) {
                _PTX_INJECT_ERROR( PTX_INJECT_ERROR_MAX_UNIQUE_INJECTS_EXCEEDED );
            }
            size_t unique_inject_idx = num_unique_injects++;
            const char* local_names_blob =
                ptx_inject->names_blob == NULL ? NULL : ptx_inject->names_blob + names_blob_bytes_written;
            _PTX_INJECT_CHECK_RET(
                _ptx_inject_snprintf_append(
                    ptx_inject->names_blob,
                    ptx_inject->names_blob_size,
                    &names_blob_bytes_written,
                    "%.*s%c",
                    inject_name_length,
                    inject_name,
                    '\0'
                )
            );
            // If we're in measure mode, use the passed in ptx to calculate the unique names.
            // If we're in the second pass, use the locally allocated memory for the name
            const char* this_inject_name = ptx_inject->names_blob == NULL ? inject_name : local_names_blob;
            PtxInjectInjectionArg* inject_args;
            if (ptx_inject->inject_args == NULL) {
                inject_args = NULL;
            } else {
                inject_args = &ptx_inject->inject_args[num_unique_inject_args];
            }
            PtxInjectInjection inject = {0};

            inject.name =  this_inject_name;
            inject.name_length = inject_name_length;
            inject.args = inject_args;
            inject.num_args = 0;
            inject.num_sites = 0;
            inject.unique_idx = unique_inject_idx;
            
            unique_injects[unique_inject_idx] = inject;
            unique_inject_site = &unique_injects[unique_inject_idx];
        }

        unique_inject_site->num_sites++;
        src_ptr = _ptx_inject_str_whitespace(src_ptr);

        if(ptx_inject->inject_site_to_inject_idx != NULL) {
            ptx_inject->inject_site_to_inject_idx[num_inject_sites] = unique_inject_site->unique_idx;
        }
        if(ptx_inject->inject_sites != NULL) {
            const char* stub_location = ptx_inject->stub_buffer + stubs_bytes_written;
            ptx_inject->inject_sites[num_inject_sites] = stub_location;
        }
        _PTX_INJECT_CHECK_RET(
            _ptx_inject_snprintf_append(
                ptx_inject->stub_buffer,
                ptx_inject->stub_buffer_size,
                &stubs_bytes_written,
                "\n"
            )
        );
        num_inject_sites++;
        
        size_t num_args = 0;
        while(true) {
            size_t arg_num = num_args++;
            const char* argument_register_name;
            size_t argument_register_name_length;
            PtxInjectMutType argument_mut_type;
            const char* argument_register_type_name;
            size_t argument_register_type_name_length;
            const char* argument_data_type_name;
            size_t argument_data_type_name_length;
            const char* argument_name;
            size_t argument_name_length;
            bool found_argument;
            _PTX_INJECT_CHECK_RET(
                _ptx_inject_ptx_parse_argument(
                    src_ptr,
                    &argument_register_name,
                    &argument_register_name_length,
                    &argument_mut_type,
                    &argument_register_type_name,
                    &argument_register_type_name_length,
                    &argument_data_type_name,
                    &argument_data_type_name_length,
                    &argument_name,
                    &argument_name_length,
                    &src_ptr,
                    &found_argument
                )
            );

            if (!found_argument) {
                if (!is_unique && unique_inject_site != NULL) {
                    if (num_args-1 != unique_inject_site->num_args) {
                        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_INCONSISTENT_INJECTION );
                    }
                }
                if(*src_ptr != '\n') {
                    _PTX_INJECT_ERROR( PTX_INJECT_ERROR_FORMATTING );
                }
                src_ptr++;
                break;
	            }

	            if (!is_unique) {
	                if (arg_num >= unique_inject_site->num_args) {
	                    _PTX_INJECT_ERROR( PTX_INJECT_ERROR_INCONSISTENT_INJECTION );
	                }
	                if (unique_inject_site->args != NULL) {
	                    PtxInjectInjectionArg* args = &unique_inject_site->args[arg_num];

                    if (argument_name_length != strlen(args->name)) 
                        _PTX_INJECT_CHECK_RET( PTX_INJECT_ERROR_INCONSISTENT_INJECTION );
                    if (strncmp(argument_name, args->name, argument_name_length) != 0)
                        _PTX_INJECT_CHECK_RET( PTX_INJECT_ERROR_INCONSISTENT_INJECTION );

                    if (argument_register_name_length != strlen(args->register_name))
                        _PTX_INJECT_CHECK_RET( PTX_INJECT_ERROR_INCONSISTENT_INJECTION );
                    if (strncmp(argument_register_name, args->register_name, argument_register_name_length))
                        _PTX_INJECT_CHECK_RET( PTX_INJECT_ERROR_INCONSISTENT_INJECTION );

                    if (argument_mut_type != args->mut_type) 
                        _PTX_INJECT_CHECK_RET( PTX_INJECT_ERROR_INCONSISTENT_INJECTION );

                    if (argument_register_type_name_length != strlen(args->register_type_name))
                        _PTX_INJECT_CHECK_RET( PTX_INJECT_ERROR_INCONSISTENT_INJECTION );
                    if (strncmp(argument_register_type_name, args->register_type_name, argument_register_type_name_length))
                        _PTX_INJECT_CHECK_RET( PTX_INJECT_ERROR_INCONSISTENT_INJECTION );

                    if (argument_data_type_name_length != strlen(args->data_type_name))
                        _PTX_INJECT_CHECK_RET( PTX_INJECT_ERROR_INCONSISTENT_INJECTION );
                    if (strncmp(argument_data_type_name, args->data_type_name, argument_data_type_name_length))
                        _PTX_INJECT_CHECK_RET( PTX_INJECT_ERROR_INCONSISTENT_INJECTION );
                }
            } else {
                num_unique_inject_args++;
                unique_inject_site->num_args++;
                const char* name =
                    ptx_inject->names_blob == NULL ? NULL : ptx_inject->names_blob + names_blob_bytes_written;
                _PTX_INJECT_CHECK_RET(
                    _ptx_inject_snprintf_append(
                        ptx_inject->names_blob,
                        ptx_inject->names_blob_size,
                        &names_blob_bytes_written,
                        "%.*s%c",
                        argument_name_length,
                        argument_name,
                        '\0'
                    )
                );
                const char* register_name =
                    ptx_inject->names_blob == NULL ? NULL : ptx_inject->names_blob + names_blob_bytes_written;
                _PTX_INJECT_CHECK_RET(
                    _ptx_inject_snprintf_append(
                        ptx_inject->names_blob,
                        ptx_inject->names_blob_size,
                        &names_blob_bytes_written,
                        "%.*s%c",
                        argument_register_name_length,
                        argument_register_name,
                        '\0'
                    )
                );
                const char* register_type_name =
                    ptx_inject->names_blob == NULL ? NULL : ptx_inject->names_blob + names_blob_bytes_written;
                _PTX_INJECT_CHECK_RET(
                    _ptx_inject_snprintf_append(
                        ptx_inject->names_blob,
                        ptx_inject->names_blob_size,
                        &names_blob_bytes_written,
                        "%.*s%c",
                        argument_register_type_name_length,
                        argument_register_type_name,
                        '\0'
                    )
                );
                const char* data_type_name =
                    ptx_inject->names_blob == NULL ? NULL : ptx_inject->names_blob + names_blob_bytes_written;
                _PTX_INJECT_CHECK_RET(
                    _ptx_inject_snprintf_append(
                        ptx_inject->names_blob,
                        ptx_inject->names_blob_size,
                        &names_blob_bytes_written,
                        "%.*s%c",
                        argument_data_type_name_length,
                        argument_data_type_name,
                        '\0'
                    )
                );
                if (unique_inject_site->args != NULL) {
                    PtxInjectInjectionArg* args = &unique_inject_site->args[arg_num];
                    args->mut_type = argument_mut_type;
                    args->data_type_name = data_type_name;
                    args->register_type_name = register_type_name;
                    args->name = name;
                    args->register_name = register_name;
                }
            }
            
            src_ptr = _ptx_inject_str_whitespace(src_ptr);
        }
    }
    _PTX_INJECT_CHECK_RET(
        _ptx_inject_snprintf_append(
            ptx_inject->stub_buffer,
            ptx_inject->stub_buffer_size,
            &stubs_bytes_written,
            "%s",
            src_ptr
        )
    );

    if (ptx_inject->stub_buffer && stubs_bytes_written >= ptx_inject->stub_buffer_size) {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_INSUFFICIENT_BUFFER );
    }
    
    ptx_inject->num_inject_sites = num_inject_sites;
    ptx_inject->num_injects = num_unique_injects;
    ptx_inject->num_inject_args = num_unique_inject_args;
    ptx_inject->names_blob_size = names_blob_bytes_written;
    ptx_inject->stub_buffer_size = stubs_bytes_written+1;

    return PTX_INJECT_SUCCESS;
}

PTX_INJECT_PUBLIC_DEF
PtxInjectResult
ptx_inject_create(
    PtxInjectHandle* handle,
    const char* processed_ptx_src
) {
    if (handle == NULL || processed_ptx_src == NULL) {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_INVALID_INPUT );
    }

    PtxInjectResult result;

    struct PtxInjectHandleImpl ptx_inject = {0};

    void* memory_block_injects = malloc(PTX_INJECT_MAX_UNIQUE_INJECTS * sizeof(PtxInjectInjection));
    if (memory_block_injects == NULL) {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_OUT_OF_MEMORY );
    }

    ptx_inject.injects = (PtxInjectInjection*)memory_block_injects;

    // This call populates a bunch of size data for the handle to be used to allocate the rest of the handle.
    result = _ptx_inject_create(&ptx_inject, processed_ptx_src);
    free(ptx_inject.injects);
    ptx_inject.injects = NULL;
    if (result != PTX_INJECT_SUCCESS) {
        return result;
    }

    size_t handle_num_bytes = sizeof(struct PtxInjectHandleImpl);
    size_t injects_num_bytes = ptx_inject.num_injects * sizeof(PtxInjectInjection);
    size_t inject_sites_num_bytes = ptx_inject.num_inject_sites * sizeof(const char *);
    size_t inject_site_to_inject_idx_num_bytes = ptx_inject.num_inject_sites * sizeof(size_t);
    size_t inject_args_num_bytes = ptx_inject.num_inject_args * sizeof(PtxInjectInjectionArg);
    size_t stub_buffer_num_bytes = ptx_inject.stub_buffer_size * sizeof(char);
    size_t names_blob_num_bytes = ptx_inject.names_blob_size * sizeof(char);

    size_t handle_offset = 0;
    size_t injects_offset =                     handle_offset +                     _PTX_INJECT_ALIGNMENT_UP(handle_num_bytes,                      _PTX_INJECT_ALIGNMENT);
    size_t inject_sites_offset =                injects_offset +                    _PTX_INJECT_ALIGNMENT_UP(injects_num_bytes,                     _PTX_INJECT_ALIGNMENT);
    size_t inject_site_to_inject_idx_offset =   inject_sites_offset +               _PTX_INJECT_ALIGNMENT_UP(inject_sites_num_bytes,                _PTX_INJECT_ALIGNMENT);
    size_t inject_args_offset =                 inject_site_to_inject_idx_offset +  _PTX_INJECT_ALIGNMENT_UP(inject_site_to_inject_idx_num_bytes,   _PTX_INJECT_ALIGNMENT);
    size_t stub_buffer_offset =                 inject_args_offset +                _PTX_INJECT_ALIGNMENT_UP(inject_args_num_bytes,                 _PTX_INJECT_ALIGNMENT);
    size_t names_blob_offset =                  stub_buffer_offset +                _PTX_INJECT_ALIGNMENT_UP(stub_buffer_num_bytes,                 _PTX_INJECT_ALIGNMENT);
    size_t total_size = names_blob_offset + names_blob_num_bytes;

    void* memory_block = malloc(total_size);
	if (memory_block == NULL) {
		_PTX_INJECT_ERROR( PTX_INJECT_ERROR_OUT_OF_MEMORY );
	}
	memset(memory_block, 0, total_size);

    *handle = (PtxInjectHandle)((char*)memory_block + handle_offset);

    (*handle)->injects = (PtxInjectInjection*)((char*)memory_block + injects_offset);
    (*handle)->num_injects = ptx_inject.num_injects;

    (*handle)->inject_sites = (const char**)((char*)memory_block + inject_sites_offset);
    (*handle)->inject_site_to_inject_idx = (size_t *)((char*)memory_block + inject_site_to_inject_idx_offset);
    (*handle)->num_inject_sites = ptx_inject.num_inject_sites;

    (*handle)->inject_args = (PtxInjectInjectionArg*)((char*)memory_block + inject_args_offset);
    (*handle)->num_inject_args = ptx_inject.num_inject_args;

    (*handle)->stub_buffer = (char*)((char*)memory_block + stub_buffer_offset);
    (*handle)->stub_buffer_size = ptx_inject.stub_buffer_size;

    (*handle)->names_blob = (char*)((char*)memory_block + names_blob_offset);
    (*handle)->names_blob_size = ptx_inject.names_blob_size;

    result = _ptx_inject_create(*handle, processed_ptx_src);
    if (result != PTX_INJECT_SUCCESS) {
        ptx_inject_destroy(*handle);
        return result;
    }

    return PTX_INJECT_SUCCESS;
}

PTX_INJECT_PUBLIC_DEF
PtxInjectResult
ptx_inject_destroy(
    PtxInjectHandle handle
) {
    if (handle != NULL) {
        free(handle);
    }
    return PTX_INJECT_SUCCESS;
}

PTX_INJECT_PUBLIC_DEF
PtxInjectResult
ptx_inject_num_injects(
    const PtxInjectHandle handle,
    size_t* num_injects_out
) {
    if (handle == NULL) {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_INVALID_INPUT );
    }

    *num_injects_out = handle->num_injects;
    return PTX_INJECT_SUCCESS;
}

PTX_INJECT_PUBLIC_DEF
PtxInjectResult 
ptx_inject_inject_info_by_name(
    const PtxInjectHandle handle,
    const char* inject_name,
    size_t* inject_idx_out, 
    size_t* inject_num_args_out,
    size_t* inject_num_sites_out 
) {
    if (handle == NULL) {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_INVALID_INPUT );
    }

    if (inject_name == NULL) {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_INVALID_INPUT );
    }

    for (size_t i = 0; i < handle->num_injects; i++) {
        PtxInjectInjection* inject = &handle->injects[i];
        if (strcmp(inject_name, inject->name) == 0) {
            if (inject_idx_out != NULL) {
                *inject_idx_out = i;
            }
            if (inject_num_args_out != NULL) {
                *inject_num_args_out = inject->num_args;
            }
            if (inject_num_sites_out != NULL) {
                *inject_num_sites_out = inject->num_sites;
            }
            return PTX_INJECT_SUCCESS;
        }
    }

    _PTX_INJECT_ERROR( PTX_INJECT_ERROR_INJECTION_NAME_NOT_FOUND );
}

PTX_INJECT_PUBLIC_DEF
PtxInjectResult 
ptx_inject_inject_info_by_index(
    const PtxInjectHandle handle,
    size_t inject_idx,
    const char** inject_name_out,
    size_t* inject_num_args_out,
    size_t* inject_num_sites_out
) {
    if (handle == NULL) {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_INVALID_INPUT );
    }

    if (inject_idx >= handle->num_injects) {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_OUT_OF_BOUNDS_IDX );
    }

    PtxInjectInjection* inject = &handle->injects[inject_idx];

    if (inject_name_out != NULL) {
        *inject_name_out = inject->name;
    }
    if (inject_num_args_out != NULL) {
        *inject_num_args_out = inject->num_args;
    }
    if (inject_num_sites_out != NULL) {
        *inject_num_sites_out = inject->num_sites;
    }
    return PTX_INJECT_SUCCESS;
}

PTX_INJECT_PUBLIC_DEF
PtxInjectResult 
ptx_inject_variable_info_by_name(
    const PtxInjectHandle handle,
    size_t inject_idx,
    const char* inject_variable_name,
    size_t* inject_variable_arg_idx_out,
    const char** inject_variable_register_name_out,
    PtxInjectMutType* inject_variable_mut_type_out,
    const char** inject_variable_register_type_name_out,
    const char** inject_variable_data_type_name_out
) {
    if (handle == NULL) {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_INVALID_INPUT );
    }

    if (inject_idx >= handle->num_injects) {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_INVALID_INPUT );
    }

    if (inject_variable_name == NULL) {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_INVALID_INPUT );
    }

    PtxInjectInjection* inject = &handle->injects[inject_idx];

    for (size_t i = 0; i < inject->num_args; i++) {
        PtxInjectInjectionArg* arg = &inject->args[i];
        if (strcmp(inject_variable_name, arg->name) == 0) {
            if (inject_variable_arg_idx_out != NULL) {
                *inject_variable_arg_idx_out = i;
            }
            if (inject_variable_register_name_out != NULL) {
                *inject_variable_register_name_out = arg->register_name;
            }
            if (inject_variable_mut_type_out != NULL) {
                *inject_variable_mut_type_out = arg->mut_type;
            }
            if (inject_variable_register_type_name_out != NULL) {
                *inject_variable_register_type_name_out = arg->register_type_name;
            }
            if (inject_variable_data_type_name_out != NULL) {
                *inject_variable_data_type_name_out = arg->data_type_name;
            }
            return PTX_INJECT_SUCCESS;
        }
    }

    _PTX_INJECT_ERROR( PTX_INJECT_ERROR_INJECTION_ARG_NAME_NOT_FOUND );
}

PTX_INJECT_PUBLIC_DEF
PtxInjectResult 
ptx_inject_variable_info_by_index(
    const PtxInjectHandle handle,
    size_t inject_idx,
    size_t inject_variable_arg_idx,
    const char** inject_variable_name_out,
    const char** inject_variable_register_name_out,
    PtxInjectMutType* inject_variable_mut_type_out,
    const char** inject_variable_register_type_name_out,
    const char** inject_variable_data_type_name_out
) {
    if (handle == NULL) {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_INVALID_INPUT );
    }

    if (inject_idx >= handle->num_injects) {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_OUT_OF_BOUNDS_IDX );
    }

    PtxInjectInjection* inject = &handle->injects[inject_idx];

    if (inject_variable_arg_idx >= inject->num_args) {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_INVALID_INPUT );
    }

    PtxInjectInjectionArg* arg = &inject->args[inject_variable_arg_idx];

    if(inject_variable_name_out != NULL) {
        *inject_variable_name_out = arg->name;
    }
    if(inject_variable_register_name_out != NULL) {
        *inject_variable_register_name_out = arg->register_name;
    }
    if (inject_variable_mut_type_out != NULL) {
        *inject_variable_mut_type_out = arg->mut_type;
    }
    if(inject_variable_register_type_name_out != NULL) {
        *inject_variable_register_type_name_out = arg->register_type_name;
    }
    if(inject_variable_data_type_name_out != NULL) {
        *inject_variable_data_type_name_out = arg->data_type_name;
    }

    return PTX_INJECT_SUCCESS;
}

PTX_INJECT_PUBLIC_DEF
PtxInjectResult
ptx_inject_render_ptx(
    const PtxInjectHandle handle,
    const char* const* ptx_stubs,
    size_t num_ptx_stubs,
    char* rendered_ptx_buffer,
    size_t rendered_ptx_buffer_size,
    size_t* rendered_ptx_bytes_written_out
) {
    if (handle == NULL) {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_INVALID_INPUT );
    }

    if (ptx_stubs == NULL) {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_INVALID_INPUT );
    }

    for (size_t i = 0; i < num_ptx_stubs; i++) {
        if (ptx_stubs[i] == NULL) {
            _PTX_INJECT_ERROR( PTX_INJECT_ERROR_INVALID_INPUT );
        }
    }

    if (num_ptx_stubs != handle->num_injects) {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_WRONG_NUM_STUBS );
    }

    size_t rendered_ptx_bytes_written = 0;

    if (rendered_ptx_bytes_written_out != NULL) {
        *rendered_ptx_bytes_written_out = 0;
    } else {
        rendered_ptx_bytes_written_out = &rendered_ptx_bytes_written;
    }

    if (rendered_ptx_buffer == NULL) {
        for (size_t i = 0; i < num_ptx_stubs; i++) {
            size_t num_sites = handle->injects[i].num_sites;
            size_t stub_length = strlen(ptx_stubs[i]);
            *rendered_ptx_bytes_written_out += num_sites * stub_length;
        }

        *rendered_ptx_bytes_written_out += handle->stub_buffer_size;

        return PTX_INJECT_SUCCESS;
    }

    const char* current_location = handle->stub_buffer;
    for (size_t site_idx = 0; site_idx < handle->num_inject_sites; site_idx++) {
        size_t unique_idx = handle->inject_site_to_inject_idx[site_idx];
        const char* stub_location = handle->inject_sites[site_idx];
        const char* ptx_stub = ptx_stubs[unique_idx];
        _PTX_INJECT_CHECK_RET(
            _ptx_inject_snprintf_append(
                rendered_ptx_buffer, 
                rendered_ptx_buffer_size, 
                rendered_ptx_bytes_written_out,
                "%.*s",
                stub_location - current_location,
                current_location
            )
        );
        _PTX_INJECT_CHECK_RET(
            _ptx_inject_snprintf_append(
                rendered_ptx_buffer, 
                rendered_ptx_buffer_size,
                rendered_ptx_bytes_written_out,
                "%.*s",
                strlen(ptx_stub),
                ptx_stub
            )
        );
        current_location = stub_location;
    }

    _PTX_INJECT_CHECK_RET(
        _ptx_inject_snprintf_append(
            rendered_ptx_buffer, 
            rendered_ptx_buffer_size, 
            rendered_ptx_bytes_written_out,
            "%.*s",
            handle->stub_buffer_size - (current_location - handle->stub_buffer),
            current_location
        )
    );

    if (rendered_ptx_buffer && *rendered_ptx_bytes_written_out >= rendered_ptx_buffer_size) {
        _PTX_INJECT_ERROR( PTX_INJECT_ERROR_INSUFFICIENT_BUFFER );
    }

    return PTX_INJECT_SUCCESS;
}

#endif // PTX_INJECT_IMPLEMENTATION_ONCE
#endif // PTX_INJECT_IMPLEMENTATION

#endif // !__CUDACC_RTC__

/**
 * \brief CUDA-side helper macros for defining inject sites.
 *
 * \details These macros are available only when compiling with nvcc (__CUDACC__).
 * Use PTX_INJECT or PTX_INJECT_TOK to mark a site, and PTX_IN/PTX_OUT/PTX_MOD
 * to describe operands. Each operand can be declared as (type, name) or
 * (type, name, expr), where type is a token with a PTX_TYPE_INFO_<TOKEN> entry.
 */
#if defined(__CUDACC__)

#define PTX_TYPES_CAT2(a,b) a##b
#define PTX_TYPES_CAT(a,b)  PTX_TYPES_CAT2(a,b)

#define PTX_TYPES_STR2(x) #x
#define PTX_TYPES_STR(x)  PTX_TYPES_STR2(x)

// Descriptor tuple: (reg_suffix, mov_postfix, constraint, bind_kind)
#define PTX_TYPES_DESC(reg_suffix, mov_postfix, constraint, bind_kind) \
  (reg_suffix, mov_postfix, constraint, bind_kind)

// Expand-then-paste: PTX_TYPE_INFO(<expanded tok>)
#define PTX_TYPES_INFO(tok)   PTX_TYPES_INFO_I(tok)
#define PTX_TYPES_INFO_I(tok) PTX_TYPES_CAT(PTX_TYPE_INFO_, tok)

// Tuple extractors
#define PTX_TYPES_T0(t) PTX_TYPES_T0_I t
#define PTX_TYPES_T1(t) PTX_TYPES_T1_I t
#define PTX_TYPES_T2(t) PTX_TYPES_T2_I t
#define PTX_TYPES_T3(t) PTX_TYPES_T3_I t
#define PTX_TYPES_T0_I(a,b,c,d) a
#define PTX_TYPES_T1_I(a,b,c,d) b
#define PTX_TYPES_T2_I(a,b,c,d) c
#define PTX_TYPES_T3_I(a,b,c,d) d

// Bind implementations
#define PTX_TYPES_BIND_ID(x)  (x)
#define PTX_TYPES_BIND_U16(x) (*reinterpret_cast<unsigned short*>(& (x) ))
#define PTX_TYPES_BIND_U32(x) (*reinterpret_cast<unsigned int  *>(& (x) ))

// Bind kind dispatch (expand-then-paste)
#define PTX_TYPES_BIND_KIND(k)   PTX_TYPES_BIND_KIND_I(k)
#define PTX_TYPES_BIND_KIND_I(k) PTX_TYPES_CAT(PTX_TYPES_BIND_, k)

// ---- include types registry (configurable) ----
#ifndef PTX_INJECT_NO_DEFAULT_TYPES
    #define PTX_TYPE_INFO_F16         PTX_TYPES_DESC(b16, b16, h, U16)
    #define PTX_TYPE_INFO_F16X2       PTX_TYPES_DESC(b32, b32, r, U32)
    #define PTX_TYPE_INFO_S32         PTX_TYPES_DESC(s32, s32, r, ID)
    #define PTX_TYPE_INFO_U32         PTX_TYPES_DESC(u32, u32, r, ID)
    #define PTX_TYPE_INFO_F32         PTX_TYPES_DESC(f32, f32, f, ID)
    #define PTX_TYPE_INFO_B32         PTX_TYPES_DESC(b32, b32, r, ID)
#endif

#define PTX_TYPE_INFO_PHILOX PTX_TYPES_DESC(u32, u32, r, ID)

#define PTX_PHILOX(state)                           \
    PTX_IN (PHILOX, philox_key_x, state.key.x),     \
    PTX_IN (PHILOX, philox_key_y, state.key.y),     \
    PTX_MOD (PHILOX, philox_ctr_x, state.ctr.x),    \
    PTX_IN (PHILOX, philox_ctr_y, state.ctr.y),     \
    PTX_IN (PHILOX, philox_ctr_z, state.ctr.z),     \
    PTX_IN (PHILOX, philox_ctr_w, state.ctr.w)

// ---------- API consumed by the rest of this header ----------
#define PTX_REGTYPE_STR(tok)    PTX_TYPES_STR(PTX_TYPES_T0(PTX_TYPES_INFO(tok)))
#define PTX_MOV_STR(tok)        PTX_TYPES_STR(PTX_TYPES_T1(PTX_TYPES_INFO(tok)))
#define PTX_CONSTRAINT_STR(tok) PTX_TYPES_STR(PTX_TYPES_T2(PTX_TYPES_INFO(tok)))
#define PTX_BIND(tok, x)        PTX_TYPES_BIND_KIND(PTX_TYPES_T3(PTX_TYPES_INFO(tok)))(x)

#include <boost/preprocessor/variadic/to_seq.hpp>
#include <boost/preprocessor/variadic/size.hpp>
#include <boost/preprocessor/seq/filter.hpp>
#include <boost/preprocessor/seq/for_each_i.hpp>
#include <boost/preprocessor/seq/size.hpp>
#include <boost/preprocessor/arithmetic/add.hpp>
#include <boost/preprocessor/punctuation/comma_if.hpp>
#include <boost/preprocessor/stringize.hpp>

// ============================================================================
// Helpers
// ============================================================================
#define PTX_CAT_(a,b) a##b
#define PTX_CAT(a,b)  PTX_CAT_(a,b)

#define PTX_STR_I(x) BOOST_PP_STRINGIZE(x)
#define PTX_STR(x)   PTX_STR_I(x)

// Stable PTX temp register name for operand index N.
// NOTE: inside inline-asm text, "%%" becomes a literal '%' in final PTX.
#define PTX_TMP_REG_STR(idx) "%%_x" PTX_STR(idx)
#define PTX_TMP_REG_NAME_STR(idx) "_x" PTX_STR(idx)

// Operand placeholder "%N" (single '%' is correct here — it's an asm placeholder)
#define PTX_OP_STR(idx) "%" PTX_STR(idx)

// ============================================================================
// Operand tuple
// ============================================================================
// Tuple layout: (kind, type_token, name_token, expr)
#define PTX_KIND(e) BOOST_PP_TUPLE_ELEM(4, 0, e)
#define PTX_TYPE(e) BOOST_PP_TUPLE_ELEM(4, 1, e)
#define PTX_NAME(e) BOOST_PP_TUPLE_ELEM(4, 2, e)
#define PTX_EXPR(e) BOOST_PP_TUPLE_ELEM(4, 3, e)

// Arity-dispatch: allow (type, name) shorthand => expr=name
#define PTX_IN(...)  PTX_CAT(PTX_IN_,  BOOST_PP_VARIADIC_SIZE(__VA_ARGS__))(__VA_ARGS__)
#define PTX_MOD(...) PTX_CAT(PTX_MOD_, BOOST_PP_VARIADIC_SIZE(__VA_ARGS__))(__VA_ARGS__)
#define PTX_OUT(...) PTX_CAT(PTX_OUT_, BOOST_PP_VARIADIC_SIZE(__VA_ARGS__))(__VA_ARGS__)

#define PTX_IN_2(type_tok, name_tok)            (in,  type_tok, name_tok, name_tok)
#define PTX_IN_3(type_tok, name_tok, expr)      (in,  type_tok, name_tok, expr)

#define PTX_MOD_2(type_tok, name_tok)           (mod, type_tok, name_tok, name_tok)
#define PTX_MOD_3(type_tok, name_tok, expr)     (mod, type_tok, name_tok, expr)

#define PTX_OUT_2(type_tok, name_tok)           (out, type_tok, name_tok, name_tok)
#define PTX_OUT_3(type_tok, name_tok, expr)     (out, type_tok, name_tok, expr)

// ============================================================================
// Kind classification
// ============================================================================
#define PTX_IS_MOD_in  0
#define PTX_IS_MOD_mod 1
#define PTX_IS_MOD_out 0

#define PTX_IS_OUT_in  0
#define PTX_IS_OUT_mod 0
#define PTX_IS_OUT_out 1

#define PTX_IS_IN_in   1
#define PTX_IS_IN_mod  0
#define PTX_IS_IN_out  0

#define PTX_PRED_MOD(s, data, e) PTX_CAT(PTX_IS_MOD_, PTX_KIND(e))
#define PTX_PRED_OUT(s, data, e) PTX_CAT(PTX_IS_OUT_, PTX_KIND(e))
#define PTX_PRED_IN(s,  data, e) PTX_CAT(PTX_IS_IN_,  PTX_KIND(e))

// Marker mode chars
#define PTX_KINDCHAR_in  "i"
#define PTX_KINDCHAR_mod "m"
#define PTX_KINDCHAR_out "o"
#define PTX_KINDCHAR(kind) PTX_CAT(PTX_KINDCHAR_, kind)

// ============================================================================
// Sequences and counts
// ============================================================================
#define PTX_ARGS_SEQ(...) BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__)

#define PTX_MOD_SEQ(seq) BOOST_PP_SEQ_FILTER(PTX_PRED_MOD, _, seq)
#define PTX_OUT_SEQ(seq) BOOST_PP_SEQ_FILTER(PTX_PRED_OUT, _, seq)
#define PTX_IN_SEQ(seq)  BOOST_PP_SEQ_FILTER(PTX_PRED_IN,  _, seq)

#define PTX_NMOD(seq) BOOST_PP_SEQ_SIZE(PTX_MOD_SEQ(seq))
#define PTX_NOUT(seq) BOOST_PP_SEQ_SIZE(PTX_OUT_SEQ(seq))
#define PTX_NIN(seq)  BOOST_PP_SEQ_SIZE(PTX_IN_SEQ(seq))

// Operand numbering (asm placeholders) is: [mods][outs][ins]
#define PTX_OFF_OUT(seq) PTX_NMOD(seq)
#define PTX_OFF_IN(seq)  BOOST_PP_ADD(PTX_NMOD(seq), PTX_NOUT(seq))

#define PTX_HAS_OUTS(seq) BOOST_PP_BOOL(BOOST_PP_ADD(PTX_NMOD(seq), PTX_NOUT(seq)))
#define PTX_HAS_INS(seq)  BOOST_PP_BOOL(PTX_NIN(seq))

// ============================================================================
// Emit PTX text pieces
// ============================================================================
#define PTX_EMIT_DECL_AT(idx, e) \
  ".reg ." PTX_REGTYPE_STR(PTX_TYPE(e)) " " PTX_TMP_REG_STR(idx) ";\n\t"

#define PTX_EMIT_LOAD_AT(idx, e) \
  "mov." PTX_MOV_STR(PTX_TYPE(e)) " " PTX_TMP_REG_STR(idx) ", " PTX_OP_STR(idx) ";\n\t"

#define PTX_EMIT_STORE_AT(idx, e) \
  "mov." PTX_MOV_STR(PTX_TYPE(e)) " " PTX_OP_STR(idx) ", " PTX_TMP_REG_STR(idx) ";\n\t"

// Marker line: explicit stable reg name + metadata
// (idx must be the stable operand index: [mods][outs][ins])
#define PTX_EMIT_MARK_AT(idx, e) \
  "// " PTX_TMP_REG_NAME_STR(idx) " " PTX_KINDCHAR(PTX_KIND(e)) " " \
  PTX_REGTYPE_STR(PTX_TYPE(e)) " " PTX_STR(PTX_TYPE(e)) " " PTX_STR(PTX_NAME(e)) "\n\t"

// ============================================================================
// Per-kind loops with correct indices
// ============================================================================
// Decls
#define PTX_DECL_MOD_I(r, data, i, e) PTX_EMIT_DECL_AT(i, e)
#define PTX_DECL_OUT_I(r, off,  i, e) PTX_EMIT_DECL_AT(BOOST_PP_ADD(i, off), e)
#define PTX_DECL_IN_I(r, off,   i, e) PTX_EMIT_DECL_AT(BOOST_PP_ADD(i, off), e)

// Loads: MOD + IN only
#define PTX_LOAD_MOD_I(r, data, i, e) PTX_EMIT_LOAD_AT(i, e)
#define PTX_LOAD_IN_I(r, off,   i, e) PTX_EMIT_LOAD_AT(BOOST_PP_ADD(i, off), e)

// Stores: MOD + OUT only
#define PTX_STORE_MOD_I(r, data, i, e) PTX_EMIT_STORE_AT(i, e)
#define PTX_STORE_OUT_I(r, off,  i, e) PTX_EMIT_STORE_AT(BOOST_PP_ADD(i, off), e)

// Marks (canonical order: MOD, OUT, IN) with correct stable-reg indices
#define PTX_MARK_MOD_I(r, data, i, e) PTX_EMIT_MARK_AT(i, e)
#define PTX_MARK_OUT_I(r, off,  i, e) PTX_EMIT_MARK_AT(BOOST_PP_ADD(i, off), e)
#define PTX_MARK_IN_I(r, off,   i, e) PTX_EMIT_MARK_AT(BOOST_PP_ADD(i, off), e)

// ============================================================================
// C++ asm operand emitters
// ============================================================================
// Output/mod operands (go in output clause)
#define PTX_OUT_OPERAND_mod(e) "+" PTX_CONSTRAINT_STR(PTX_TYPE(e)) ( PTX_BIND(PTX_TYPE(e), PTX_EXPR(e)) )
#define PTX_OUT_OPERAND_out(e) "=" PTX_CONSTRAINT_STR(PTX_TYPE(e)) ( PTX_BIND(PTX_TYPE(e), PTX_EXPR(e)) )
#define PTX_OUT_OPERAND(e)     PTX_CAT(PTX_OUT_OPERAND_, PTX_KIND(e))(e)

// Input operands (go in input clause)
#define PTX_IN_OPERAND_in(e)   PTX_CONSTRAINT_STR(PTX_TYPE(e)) ( PTX_BIND(PTX_TYPE(e), PTX_EXPR(e)) )
#define PTX_IN_OPERAND(e)      PTX_CAT(PTX_IN_OPERAND_, PTX_KIND(e))(e)

// Comma-safe enumeration
#define PTX_ENUM_OUT_MOD_I(r, data, i, e) \
  BOOST_PP_COMMA_IF(i) PTX_OUT_OPERAND(e)

#define PTX_ENUM_OUT_OUT_I(r, nmods, i, e) \
  BOOST_PP_COMMA_IF(BOOST_PP_ADD(i, nmods)) PTX_OUT_OPERAND(e)

#define PTX_ENUM_IN_I(r, data, i, e) \
  BOOST_PP_COMMA_IF(i) PTX_IN_OPERAND(e)

// Output list = [mods][outs]
#define PTX_OUT_OPERANDS(seq) \
  BOOST_PP_SEQ_FOR_EACH_I(PTX_ENUM_OUT_MOD_I, _, PTX_MOD_SEQ(seq)) \
  BOOST_PP_SEQ_FOR_EACH_I(PTX_ENUM_OUT_OUT_I, PTX_NMOD(seq), PTX_OUT_SEQ(seq))

// Input list = [ins]
#define PTX_IN_OPERANDS(seq) \
  BOOST_PP_SEQ_FOR_EACH_I(PTX_ENUM_IN_I, _, PTX_IN_SEQ(seq))

// asm operand clause selection (no BOOST_PP_IF around comma-heavy lists)
#define PTX_ASM_OPERANDS_00(seq) \
  static_assert(false, "PTX_INJECT requires at least one operand.");

#define PTX_ASM_OPERANDS_10(seq)  : PTX_OUT_OPERANDS(seq)
#define PTX_ASM_OPERANDS_01(seq)  : : PTX_IN_OPERANDS(seq)
#define PTX_ASM_OPERANDS_11(seq)  : PTX_OUT_OPERANDS(seq) : PTX_IN_OPERANDS(seq)

#define PTX_ASM_OPERANDS_SELECT(ho, hi, seq) PTX_CAT(PTX_ASM_OPERANDS_, PTX_CAT(ho, hi))(seq)
#define PTX_ASM_OPERANDS(seq) PTX_ASM_OPERANDS_SELECT(PTX_HAS_OUTS(seq), PTX_HAS_INS(seq), seq)

// ============================================================================
// PTX_INJECT entrypoints
// ============================================================================
// site_str must be a string literal, e.g. "func"
#define PTX_INJECT(site_str, ...) \
  PTX_INJECT_IMPL(site_str, PTX_ARGS_SEQ(__VA_ARGS__))

// optional: site as identifier token, e.g. PTX_INJECT_TOK(func, ...)
#define PTX_INJECT_TOK(site_tok, ...) \
  PTX_INJECT_IMPL(PTX_STR(site_tok), PTX_ARGS_SEQ(__VA_ARGS__))

#define PTX_INJECT_IMPL(site_str, seq) do { \
  asm ( \
    "{\n\t" \
    /* Declare stable regs for all operands (indices match asm operand numbering) */ \
    BOOST_PP_SEQ_FOR_EACH_I(PTX_DECL_MOD_I, _,                PTX_MOD_SEQ(seq)) \
    BOOST_PP_SEQ_FOR_EACH_I(PTX_DECL_OUT_I, PTX_OFF_OUT(seq), PTX_OUT_SEQ(seq)) \
    BOOST_PP_SEQ_FOR_EACH_I(PTX_DECL_IN_I,  PTX_OFF_IN(seq),  PTX_IN_SEQ(seq)) \
    \
    /* Marshal C operands -> stable regs (mods + ins) */ \
    BOOST_PP_SEQ_FOR_EACH_I(PTX_LOAD_MOD_I, _,                PTX_MOD_SEQ(seq)) \
    BOOST_PP_SEQ_FOR_EACH_I(PTX_LOAD_IN_I,  PTX_OFF_IN(seq),  PTX_IN_SEQ(seq)) \
    \
    "// PTX_INJECT_START " site_str "\n\t" \
    /* Marker lines (canonical order: MOD, OUT, IN) */ \
    BOOST_PP_SEQ_FOR_EACH_I(PTX_MARK_MOD_I, _,                PTX_MOD_SEQ(seq)) \
    BOOST_PP_SEQ_FOR_EACH_I(PTX_MARK_OUT_I, PTX_OFF_OUT(seq), PTX_OUT_SEQ(seq)) \
    BOOST_PP_SEQ_FOR_EACH_I(PTX_MARK_IN_I,  PTX_OFF_IN(seq),  PTX_IN_SEQ(seq)) \
    "// PTX_INJECT_END\n\t" \
    \
    /* Marshal stable regs -> C outputs (mods + outs) */ \
    BOOST_PP_SEQ_FOR_EACH_I(PTX_STORE_MOD_I, _,                PTX_MOD_SEQ(seq)) \
    BOOST_PP_SEQ_FOR_EACH_I(PTX_STORE_OUT_I, PTX_OFF_OUT(seq), PTX_OUT_SEQ(seq)) \
    "}" \
    PTX_ASM_OPERANDS(seq) \
  ); \
} while(0)

#endif // __CUDACC__
