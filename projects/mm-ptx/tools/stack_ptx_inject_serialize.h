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

#ifndef STACK_PTX_INJECT_SERIALIZE_H_INCLUDE
#define STACK_PTX_INJECT_SERIALIZE_H_INCLUDE

#define STACK_PTX_INJECT_SERIALIZE_VERSION_MAJOR 1 //!< PTX Inject major version.
#define STACK_PTX_INJECT_SERIALIZE_VERSION_MINOR 1 //!< PTX Inject minor version.
#define STACK_PTX_INJECT_SERIALIZE_VERSION_PATCH 1 //!< PTX Inject patch version.

/**
 * \brief String representation of the PTX Inject library version (e.g., "1.1.1").
 */
#define STACK_PTX_INJECT_SERIALIZE_VERSION_STRING "1.1.1"

#define STACK_PTX_INJECT_SERIALIZE_VERSION (STACK_PTX_INJECT_SERIALIZE_VERSION_MAJOR * 10000 + STACK_PTX_INJECT_SERIALIZE_VERSION_MINOR * 100 + STACK_PTX_INJECT_SERIALIZE_VERSION_PATCH)

#ifdef __cplusplus
#define STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEC extern "C"
#define STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEF extern "C"
#else
#define STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEC extern
#define STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEF
#endif

#include <stack_ptx.h>
#include <ptx_inject.h>

#include <stdint.h>

typedef enum {
    STACK_PTX_INJECT_SERIALIZE_SUCCESS,
    STACK_PTX_INJECT_SERIALIZE_ERROR_INSUFFICIENT_BUFFER,
    STACK_PTX_INJECT_SERIALIZE_ERROR_INVALID_INPUT,
    STACK_PTX_INJECT_SERIALIZE_ERROR_INTERNAL,
    STACK_PTX_INJECT_SERIALIZE_RESULT_NUM_ENUM
} StackPtxInjectSerializeResult;

typedef struct {
    unsigned int device_capability_major;
    unsigned int device_capability_minor;
    size_t execution_limit;
} StackPtxExtraInfo;

typedef struct {
    const char* annotated_ptx;

    const StackPtxCompilerInfo* compiler_info;
    const StackPtxStackInfo* stack_info;
    StackPtxExtraInfo* extra;

    const StackPtxRegister* registers;
    size_t num_registers;

    const size_t* const* request_stubs;
    const size_t*        request_stub_sizes;
    size_t               num_request_stubs;
} StackPtxInjectCompilerStateSerialize;

typedef struct {
    char* annotated_ptx;

    StackPtxCompilerInfo* compiler_info;
    StackPtxStackInfo* stack_info;
    StackPtxExtraInfo* extra;

    StackPtxRegister* registers;
    size_t num_registers;

    size_t** request_stubs;
    size_t*  request_stub_sizes;
    size_t   num_request_stubs;
} StackPtxInjectCompilerStateDeserialize;

typedef struct {
    const StackPtxInstruction* const* instruction_stubs;
    const size_t*                     instruction_stub_sizes;
    size_t                            num_instruction_stubs;
} StackPtxInjectCompilerInputSerialize;

typedef struct {
    StackPtxInstruction** instruction_stubs;
    size_t*               instruction_stub_sizes;
    size_t                num_instruction_stubs;
} StackPtxInjectCompilerInputDeserialize;

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEC 
const char* 
stack_ptx_inject_serialize_result_to_string(
    StackPtxInjectSerializeResult result
);

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEC
StackPtxInjectSerializeResult
stack_ptx_compiler_info_serialize(
    const StackPtxCompilerInfo* compiler_info_ref,
    uint8_t* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_out
);

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEC
StackPtxInjectSerializeResult
stack_ptx_compiler_info_deserialize(
    const uint8_t* wire,
    size_t wire_size,
    size_t* wire_used_out,
    uint8_t* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_out,
    StackPtxCompilerInfo** compiler_info_out
);

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEC
bool
stack_ptx_compiler_info_equal(
    const StackPtxCompilerInfo* compiler_info_x,
    const StackPtxCompilerInfo* compiler_info_y
);

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEC
StackPtxInjectSerializeResult
stack_ptx_compiler_info_print(
    const StackPtxCompilerInfo* compiler_info
);

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEC
StackPtxInjectSerializeResult
stack_ptx_stack_info_serialize(
    const StackPtxStackInfo* stack_info_ref,
    uint8_t* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_out
);

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEC
StackPtxInjectSerializeResult
stack_ptx_stack_info_deserialize(
    const uint8_t* wire,
    size_t wire_size,
    size_t* wire_used_out,
    uint8_t* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_out,
    StackPtxStackInfo** stack_info_out
);

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEC
bool
stack_ptx_stack_info_equal(
    const StackPtxStackInfo* stack_info_x,
    const StackPtxStackInfo* stack_info_y
);

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEC
StackPtxInjectSerializeResult
stack_ptx_stack_info_print(
    const StackPtxStackInfo* stack_info
);

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEC
StackPtxInjectSerializeResult
stack_ptx_registers_serialize(
    const StackPtxRegister* registers,
    size_t num_registers,
    uint8_t* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_out
);

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEC
StackPtxInjectSerializeResult
stack_ptx_registers_deserialize(
    const uint8_t* wire,
    size_t wire_size,
    size_t* wire_used_out,
    uint8_t* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_out,
    StackPtxRegister** registers_out,
    size_t* num_registers_out
);

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEC
bool
stack_ptx_registers_equal(
    const StackPtxRegister* registers_x,
    size_t num_registers_x,
    const StackPtxRegister* registers_y,
    size_t num_registers_y
);

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEC
StackPtxInjectSerializeResult
stack_ptx_requests_serialize(
    const size_t* const* request_stubs,
    const size_t* request_stubs_sizes,
    size_t num_request_stubs,
    uint8_t* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_out
);

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEC
StackPtxInjectSerializeResult
stack_ptx_requests_deserialize(
    const uint8_t* wire,
    size_t wire_size,
    size_t* wire_used_out,
    uint8_t* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_out,
    size_t*** requests_stubs_out,
    size_t** request_stubs_sizes_out,
    size_t* num_requests_stubs_out
);

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEC
bool
stack_ptx_requests_equal(
    const size_t* const* requests_stubs_x,
    const size_t* request_stubs_sizes_x,
    size_t num_request_stubs_x,
    const size_t* const* requests_stubs_y,
    const size_t* request_stubs_sizes_y,
    size_t num_request_stubs_y
);

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEC
StackPtxInjectSerializeResult
stack_ptx_instructions_serialize(
    const StackPtxInstruction* const* instruction_stubs,
    size_t num_instruction_stubs,
    uint8_t* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_out
);

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEC
StackPtxInjectSerializeResult
stack_ptx_instructions_deserialize(
    const uint8_t* wire,
    size_t wire_size,
    size_t* wire_used_out,
    uint8_t* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_out,
    StackPtxInstruction*** instruction_stubs_out,
    size_t* num_instruction_stubs_out
);

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEC
bool
stack_ptx_instructions_equal(
    const StackPtxInstruction* const* instruction_stubs_x,
    size_t num_instruction_stubs_x,
    const StackPtxInstruction* const* instruction_stubs_y,
    size_t num_instruction_stubs_y
);

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEC
StackPtxInjectSerializeResult
ptx_inject_ptx_serialize(
    const char* ptx,
    uint8_t* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_out
);

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEC
StackPtxInjectSerializeResult
ptx_inject_ptx_deserialize(
    const uint8_t* wire,
    size_t wire_size,
    size_t* wire_used_out,
    uint8_t* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_out,
    char** ptx
);

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEC
bool
ptx_inject_ptx_equal(
    const char* ptx_x,
    const char* ptx_y
);

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEC
StackPtxInjectSerializeResult
stack_ptx_extra_serialize(
    const StackPtxExtraInfo* extra_ref,
    uint8_t* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_out
);

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEC
StackPtxInjectSerializeResult
stack_ptx_extra_deserialize(
    const uint8_t* wire,
    size_t wire_size,
    size_t* wire_used_out,
    uint8_t* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_out,
    StackPtxExtraInfo** extra_out
);

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEC
bool
stack_ptx_extra_equal(
    const StackPtxExtraInfo* extra_x,
    const StackPtxExtraInfo* extra_y
);

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEC
StackPtxInjectSerializeResult
stack_ptx_inject_compiler_state_serialize(
    const StackPtxInjectCompilerStateSerialize* compiler_state,
    uint8_t* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_out
);

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEC
StackPtxInjectSerializeResult
stack_ptx_inject_compiler_state_deserialize(
    const uint8_t* wire,
    size_t wire_size,
    size_t* wire_used_out,
    uint8_t* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_out,
    StackPtxInjectCompilerStateDeserialize** compiler_state_out
);

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEC
bool
stack_ptx_inject_compiler_state_equal(
    const StackPtxInjectCompilerStateSerialize* compiler_state_x,
    StackPtxInjectCompilerStateDeserialize* compiler_state_y
);

#endif /* STACK_PTX_INJECT_SERIALIZE_H_INCLUDE */

#ifdef STACK_PTX_INJECT_SERIALIZE_IMPLEMENTATION
#ifndef STACK_PTX_INJECT_SERIALIZE_IMPLEMENTATION_ONCE
#define STACK_PTX_INJECT_SERIALIZE_IMPLEMENTATION_ONCE

#define _STACK_PTX_INJECT_SERIALIZE_ALIGNMENT 16 // Standard malloc alignment
#define _STACK_PTX_INJECT_SERIALIZE_ALIGNMENT_UP(size, align) (((size) + (align) - 1) & ~((align) - 1))

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifdef STACK_PTX_INJECT_SERIALIZE_DEBUG
#include <assert.h>
#define _STACK_PTX_INJECT_SERIALIZE_ERROR(ans)                                                                      \
    do {                                                                                            \
        StackPtxInjectSerializeResult _result = (ans);                                                            \
        const char* error_name = stack_ptx_inject_serialize_result_to_string(_result);                              \
        fprintf(stderr, "STACK_PTX_INJECT_SERIALIZE_ERROR: %s \n  %s %d\n", error_name, __FILE__, __LINE__);        \
        assert(0);                                                                                  \
        exit(1);                                                                                    \
    } while(0);

#define _STACK_PTX_INJECT_SERIALIZE_CHECK_RET(ans)                                                                  \
    do {                                                                                            \
        StackPtxInjectSerializeResult _result = (ans);                                                            \
        if (_result != STACK_PTX_INJECT_SERIALIZE_SUCCESS) {                                                        \
            const char* error_name = stack_ptx_inject_serialize_result_to_string(_result);                          \
            fprintf(stderr, "STACK_PTX_INJECT_SERIALIZE_CHECK: %s \n  %s %d\n", error_name, __FILE__, __LINE__);    \
            assert(0);                                                                              \
            exit(1);                                                                                \
            return _result;                                                                         \
        }                                                                                           \
    } while(0);
#else
#define _STACK_PTX_INJECT_SERIALIZE_ERROR(ans)                              \
    do {                                                    \
        StackPtxInjectSerializeResult _result = (ans);                    \
        return _result;                                     \
    } while(0);

#define _STACK_PTX_INJECT_SERIALIZE_CHECK_RET(ans)                          \
    do {                                                    \
        StackPtxInjectSerializeResult _result = (ans);                    \
        if (_result != STACK_PTX_INJECT_SERIALIZE_SUCCESS) return _result;  \
    } while(0);
#endif // STACK_PTX_INJECT_SERIALIZE_DEBUG

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEF
const char* 
stack_ptx_inject_serialize_result_to_string(
    StackPtxInjectSerializeResult result
) {
    switch(result) {
        case STACK_PTX_INJECT_SERIALIZE_SUCCESS:            return "STACK_PTX_INJECT_SERIALIZE_SUCCESS";
        case STACK_PTX_INJECT_SERIALIZE_ERROR_INSUFFICIENT_BUFFER:  return "STACK_PTX_INJECT_SERIALIZE_ERROR_INSUFFICIENT_BUFFER";
        case STACK_PTX_INJECT_SERIALIZE_ERROR_INVALID_INPUT:        return "STACK_PTX_INJECT_SERIALIZE_ERROR_INVALID_INPUT";
        case STACK_PTX_INJECT_SERIALIZE_ERROR_INTERNAL:         return "STACK_PTX_INJECT_SERIALIZE_ERROR_INTERNAL";
        case STACK_PTX_INJECT_SERIALIZE_RESULT_NUM_ENUM: break;
    }
    return "STACK_PTX_INJECT_SERIALIZE_ERROR_INVALID_RESULT_ENUM";
}

static 
inline 
size_t 
_stack_ptx_inject_serialize_string_size(
    const char *s
) {
    return s ? strlen(s) + 1 : 1;
}

static 
inline 
bool 
_stack_ptx_inject_string_equal(
    const char *s_x,
    const char *s_y
) {
    size_t string_size_x = strlen(s_x);
    size_t string_size_y = strlen(s_y);

    if (string_size_x != string_size_y) {
        return false;
    }

    if (strncmp(s_x, s_y, string_size_x) == 0) {
        return true;
    }

    return false;
}

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEF
StackPtxInjectSerializeResult
stack_ptx_compiler_info_serialize(
    const StackPtxCompilerInfo* compiler_info_ref,
    uint8_t* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_out
) {
    if (!compiler_info_ref || !buffer_bytes_written_out) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INVALID_INPUT );
    }

    *buffer_bytes_written_out = sizeof(StackPtxCompilerInfo);

    if(!buffer) {
        return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
    }

    if (buffer_size < *buffer_bytes_written_out) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INSUFFICIENT_BUFFER );
    }

    memcpy(buffer, compiler_info_ref, sizeof(StackPtxCompilerInfo));

    return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
}

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEF
StackPtxInjectSerializeResult
stack_ptx_compiler_info_deserialize(
    const uint8_t* wire,
    size_t wire_size,
    size_t* wire_used_out,
    uint8_t* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_out,
    StackPtxCompilerInfo** compiler_info_out
) {
    if (!wire || !wire_used_out || !buffer_bytes_written_out) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INVALID_INPUT );
    }

    *wire_used_out = sizeof(StackPtxCompilerInfo);
    *buffer_bytes_written_out = sizeof(StackPtxCompilerInfo) + _STACK_PTX_INJECT_SERIALIZE_ALIGNMENT;

    if (!buffer) {
        return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
    }

    if (buffer && buffer_size < *buffer_bytes_written_out) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INSUFFICIENT_BUFFER );
    }

    buffer = (uint8_t*)_STACK_PTX_INJECT_SERIALIZE_ALIGNMENT_UP((uintptr_t)buffer, _STACK_PTX_INJECT_SERIALIZE_ALIGNMENT);

    *compiler_info_out = (StackPtxCompilerInfo*)buffer;
    memcpy(buffer, wire, sizeof(StackPtxCompilerInfo));

    return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
}

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEF
bool
stack_ptx_compiler_info_equal(
    const StackPtxCompilerInfo* compiler_info_x,
    const StackPtxCompilerInfo* compiler_info_y
) {
    if (
        compiler_info_x->max_ast_size != compiler_info_y->max_ast_size ||
        compiler_info_x->max_ast_to_visit_stack_depth != compiler_info_y->max_ast_to_visit_stack_depth ||
        compiler_info_x->max_frame_depth != compiler_info_y->max_frame_depth ||
        compiler_info_x->stack_size != compiler_info_y->stack_size ||
        compiler_info_x->max_frame_depth != compiler_info_y->max_frame_depth
    ) {
        return false;
    }
    return true;
}

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEF
StackPtxInjectSerializeResult
stack_ptx_compiler_info_print(
    const StackPtxCompilerInfo* compiler_info
) {
    if (!compiler_info) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INVALID_INPUT );
    }

    printf("Stack PTX Compiler Info:\n");
    printf(
        "\tMax AST Size: %zu\n"
        "\tMax AST to Visit Stack Depth: %zu\n"
        "\tStack Size: %zu\n"
        "\tMax Frame Depth: %zu\n"
        "\tStore Size: %zu\n",
        compiler_info->max_ast_size,
        compiler_info->max_ast_to_visit_stack_depth,
        compiler_info->stack_size,
        compiler_info->max_frame_depth,
        compiler_info->store_size
    );
    return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
}

static
inline
size_t
_stack_ptx_measure_string_array_size(
    const char* const* strings,
    size_t num_strings
) {
    size_t total = 0;
    total += sizeof(size_t);
    for (size_t i = 0; i < num_strings; i++) {
        const char* string = strings[i];
        total += strlen(string) + 1;
    }
    return total;
}

static
inline
StackPtxInjectSerializeResult
_stack_ptx_serialize_string_array(
    const char* const* strings,
    size_t num_strings,
    uint8_t* buffer,
    size_t* buffer_bytes_written_out
) {
    uint8_t* p = (uint8_t*)buffer;

    memcpy(p, &num_strings, sizeof(size_t));
    p += sizeof(size_t);

    size_t total = 0;
    for (size_t i = 0; i < num_strings; i++) {
        const char* string = strings[i];
        size_t string_size = strlen(string) + 1;
        memcpy(p, string, string_size);
        p += string_size;
    }

    *buffer_bytes_written_out = p - buffer;

    return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
}

StackPtxInjectSerializeResult
_stack_ptx_stack_info_serialize_size(
    const StackPtxStackInfo* stack_info_ref,
    size_t* buffer_size_out
) {
    size_t total = 0;

    total += 
        _stack_ptx_measure_string_array_size(
            stack_info_ref->ptx_instruction_strings, 
            stack_info_ref->num_ptx_instructions
        );

    total += 
        _stack_ptx_measure_string_array_size(
            stack_info_ref->special_register_strings, 
            stack_info_ref->num_special_registers
        );

    total += stack_info_ref->num_ptx_instructions * sizeof(StackPtxPtxInstructionDescriptor);
    total += stack_info_ref->num_special_registers * sizeof(StackPtxSpecialRegisterDescriptor);
    
    total += 
        _stack_ptx_measure_string_array_size(
            stack_info_ref->stack_literal_prefixes, 
            stack_info_ref->num_stacks
        );

    total += sizeof(size_t);
    total += stack_info_ref->num_arg_types * sizeof(StackPtxArgTypeInfo);

    *buffer_size_out = total;

    return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
}

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEF
StackPtxInjectSerializeResult
stack_ptx_stack_info_serialize(
    const StackPtxStackInfo* stack_info_ref,
    uint8_t* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_out
) {
    if (!stack_info_ref || !buffer_bytes_written_out) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INVALID_INPUT );
    }

    _STACK_PTX_INJECT_SERIALIZE_CHECK_RET(
        _stack_ptx_stack_info_serialize_size(
            stack_info_ref, 
            buffer_bytes_written_out
        )
    );

    if (!buffer) {
        return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
    }

    if (buffer_size < *buffer_bytes_written_out) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INSUFFICIENT_BUFFER );
    }

    uint8_t* p = (uint8_t*)buffer;
    size_t buffer_offset = 0;

    _STACK_PTX_INJECT_SERIALIZE_CHECK_RET(
        _stack_ptx_serialize_string_array(
            stack_info_ref->ptx_instruction_strings, 
            stack_info_ref->num_ptx_instructions,
            p,
            &buffer_offset
        )
    );
    p += buffer_offset;

    memcpy(
        p,
        stack_info_ref->ptx_instruction_descriptors,
        stack_info_ref->num_ptx_instructions * sizeof(StackPtxPtxInstructionDescriptor)
    );
    p += stack_info_ref->num_ptx_instructions * sizeof(StackPtxPtxInstructionDescriptor);

    _STACK_PTX_INJECT_SERIALIZE_CHECK_RET(
        _stack_ptx_serialize_string_array(
            stack_info_ref->special_register_strings, 
            stack_info_ref->num_special_registers,
            p,
            &buffer_offset
        )
    );
    p += buffer_offset;

    memcpy(
        p,
        stack_info_ref->special_register_descriptors,
        stack_info_ref->num_special_registers * sizeof(StackPtxSpecialRegisterDescriptor)
    );
    p += stack_info_ref->num_special_registers * sizeof(StackPtxSpecialRegisterDescriptor);

    _STACK_PTX_INJECT_SERIALIZE_CHECK_RET(
        _stack_ptx_serialize_string_array(
            stack_info_ref->stack_literal_prefixes, 
            stack_info_ref->num_stacks,
            p,
            &buffer_offset
        )
    );
    p += buffer_offset;

    memcpy(p, &stack_info_ref->num_arg_types, sizeof(size_t));
    p += sizeof(size_t);

    for (size_t i = 0; i < stack_info_ref->num_arg_types; i++) {
        const StackPtxArgTypeInfo* arg_type_info = &stack_info_ref->arg_type_info[i];
        memcpy(p, arg_type_info, sizeof(StackPtxArgTypeInfo));
        p += sizeof(StackPtxArgTypeInfo);
    }

    size_t buffer_bytes_written = p - buffer;
    
    if (buffer_bytes_written != *buffer_bytes_written_out) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INTERNAL );
    }

    return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
}

static
inline
StackPtxInjectSerializeResult
_stack_ptx_stack_info_deserialize_size(
    const uint8_t* wire,
    size_t wire_size,
    size_t* wire_used_out,
    size_t* buffer_bytes_written_out
) {
    size_t total_bytes = 0;
    total_bytes += _STACK_PTX_INJECT_SERIALIZE_ALIGNMENT;
    total_bytes += sizeof(StackPtxStackInfo);

    const uint8_t* p = wire;

    size_t num_ptx_instructions;
    memcpy(&num_ptx_instructions, p, sizeof(size_t));
    p += sizeof(size_t);

    total_bytes += num_ptx_instructions * sizeof(const char* const *);
    for (size_t i = 0; i < num_ptx_instructions; i++) {
        size_t string_size = strlen(p) + 1;
        p += string_size;
        total_bytes += string_size;
    }

    total_bytes += num_ptx_instructions * sizeof(StackPtxPtxInstructionDescriptor);
    p += num_ptx_instructions * sizeof(StackPtxPtxInstructionDescriptor);

    size_t num_special_registers;
    memcpy(&num_special_registers, p, sizeof(size_t));
    p += sizeof(size_t);

    total_bytes += num_special_registers * sizeof(const char* const *);
    for (size_t i = 0; i < num_special_registers; i++) {
        size_t string_size = strlen(p) + 1;
        p += string_size;
        total_bytes += string_size;
    }

    total_bytes += num_special_registers * sizeof(StackPtxSpecialRegisterDescriptor);
    p += num_special_registers * sizeof(StackPtxSpecialRegisterDescriptor);

    size_t num_stacks;
    memcpy(&num_stacks, p, sizeof(size_t));
    p += sizeof(size_t);

    total_bytes += num_stacks * sizeof(const char* const *);
    for (size_t i = 0; i < num_stacks; i++) {
        size_t string_size = strlen(p) + 1;
        p += string_size;
        total_bytes += string_size;
    }

    size_t num_arg_types;
    memcpy(&num_arg_types, p, sizeof(size_t));
    p += sizeof(size_t);

    total_bytes += num_arg_types * sizeof(StackPtxArgTypeInfo);
    p += num_arg_types * sizeof(StackPtxArgTypeInfo);

    *wire_used_out = p - wire;
    *buffer_bytes_written_out = total_bytes;

    return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
}

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEF
StackPtxInjectSerializeResult
stack_ptx_stack_info_deserialize(
    const uint8_t* wire,
    size_t wire_size,
    size_t* wire_used_out,
    uint8_t* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_out,
    StackPtxStackInfo** stack_info_out
) {
    if (!wire || !wire_used_out || !buffer_bytes_written_out) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INVALID_INPUT );
    }

    _STACK_PTX_INJECT_SERIALIZE_CHECK_RET(
        _stack_ptx_stack_info_deserialize_size(
            wire,
            wire_size,
            wire_used_out,
            buffer_bytes_written_out
        )
    );

    if (!buffer) {
        return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
    }

    if (buffer && buffer_size < *buffer_bytes_written_out) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INSUFFICIENT_BUFFER );
    }

    uint8_t* aligned_buffer = (uint8_t*)_STACK_PTX_INJECT_SERIALIZE_ALIGNMENT_UP((uintptr_t)buffer, _STACK_PTX_INJECT_SERIALIZE_ALIGNMENT);

    const uint8_t* p = wire;

    uint8_t* deserialize_offset = aligned_buffer + sizeof(StackPtxStackInfo);

    *stack_info_out = (StackPtxStackInfo*)aligned_buffer;

    {
        size_t num_ptx_instructions;
        memcpy(&num_ptx_instructions, p, sizeof(size_t));
        p += sizeof(size_t);
        (*stack_info_out)->ptx_instruction_strings = (const char *const *)deserialize_offset;
        (*stack_info_out)->num_ptx_instructions = num_ptx_instructions;

        char** deserialize_ptrs_offset = (char**)deserialize_offset;
        char* deserialize_string_offset = (uint8_t*)deserialize_offset + num_ptx_instructions * sizeof(const char* const *);
        for (size_t i = 0; i < num_ptx_instructions; i++) {
            deserialize_ptrs_offset[i] = (uint8_t*)deserialize_string_offset;
            size_t string_size = strlen(p) + 1;
            memcpy(deserialize_string_offset, p, string_size);
            deserialize_string_offset += string_size;
            p += string_size;
        }
        deserialize_offset = deserialize_string_offset;
        (*stack_info_out)->ptx_instruction_descriptors = (StackPtxPtxInstructionDescriptor*)deserialize_offset;
        memcpy(
            deserialize_offset,
            p,
            num_ptx_instructions * sizeof(StackPtxPtxInstructionDescriptor)
        );
        p += num_ptx_instructions * sizeof(StackPtxPtxInstructionDescriptor);
        deserialize_offset += num_ptx_instructions * sizeof(StackPtxPtxInstructionDescriptor);
    }

    {
        size_t num_special_registers;
        memcpy(&num_special_registers, p, sizeof(size_t));
        p += sizeof(size_t);
        (*stack_info_out)->special_register_strings = (const char *const *)deserialize_offset;
        (*stack_info_out)->num_special_registers = num_special_registers;

        char** deserialize_ptrs_offset = (char**)deserialize_offset;
        char* deserialize_string_offset = (uint8_t*)deserialize_offset + num_special_registers * sizeof(const char* const *);
        for (size_t i = 0; i < num_special_registers; i++) {
            deserialize_ptrs_offset[i] = (uint8_t*)deserialize_string_offset;
            size_t string_size = strlen(p) + 1;
            memcpy(deserialize_string_offset, p, string_size);
            deserialize_string_offset += string_size;
            p += string_size;
        }
        deserialize_offset = deserialize_string_offset;
        (*stack_info_out)->special_register_descriptors = (StackPtxSpecialRegisterDescriptor*)deserialize_offset;
        memcpy(
            deserialize_offset,
            p,
            num_special_registers * sizeof(StackPtxSpecialRegisterDescriptor)
        );
        p += num_special_registers * sizeof(StackPtxSpecialRegisterDescriptor);
        deserialize_offset += num_special_registers * sizeof(StackPtxSpecialRegisterDescriptor);
    }

    {
        size_t num_stacks;
        memcpy(&num_stacks, p, sizeof(size_t));
        p += sizeof(size_t);
        (*stack_info_out)->stack_literal_prefixes = (const char *const *)deserialize_offset;
        (*stack_info_out)->num_stacks = num_stacks;

        char** deserialize_ptrs_offset = (char**)deserialize_offset;
        char* deserialize_string_offset = (uint8_t*)deserialize_offset + num_stacks * sizeof(const char* const *);
        for (size_t i = 0; i < num_stacks; i++) {
            deserialize_ptrs_offset[i] = (uint8_t*)deserialize_string_offset;
            size_t string_size = strlen(p) + 1;
            memcpy(deserialize_string_offset, p, string_size);
            deserialize_string_offset += string_size;
            p += string_size;
        }
        deserialize_offset = deserialize_string_offset;
    }

    {
        size_t num_arg_types;
        memcpy(&num_arg_types, p, sizeof(size_t));
        p += sizeof(size_t);

        (*stack_info_out)->arg_type_info = (StackPtxArgTypeInfo*)deserialize_offset;
        (*stack_info_out)->num_arg_types = num_arg_types;

        memcpy(deserialize_offset, p, num_arg_types * sizeof(StackPtxArgTypeInfo));
        p += num_arg_types * sizeof(StackPtxArgTypeInfo);
        deserialize_offset += num_arg_types * sizeof(StackPtxArgTypeInfo);
    }

    size_t wire_used = p - wire;
    size_t buffer_bytes_written = deserialize_offset - aligned_buffer;

    if (wire_used != *wire_used_out) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INTERNAL );
    }

    if (buffer_bytes_written != *buffer_bytes_written_out - _STACK_PTX_INJECT_SERIALIZE_ALIGNMENT) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INTERNAL );
    }

    return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
}

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEF
bool
stack_ptx_stack_info_equal(
    const StackPtxStackInfo* stack_info_x,
    const StackPtxStackInfo* stack_info_y
) {
    if (
        stack_info_x->num_ptx_instructions != stack_info_y->num_ptx_instructions ||
        stack_info_x->num_special_registers != stack_info_y->num_special_registers ||
        stack_info_x->num_stacks != stack_info_y->num_stacks ||
        stack_info_x->num_arg_types != stack_info_y->num_arg_types
    ) {
        return false;
    }

    for (size_t i = 0; i < stack_info_x->num_ptx_instructions; i++) {
        if (
            strcmp(
                stack_info_x->ptx_instruction_strings[i], 
                stack_info_y->ptx_instruction_strings[i]
            ) != 0
        ) {
            return false;
        }
    }

    for (size_t i = 0; i < stack_info_x->num_special_registers; i++) {
        if (
            strcmp(
                stack_info_x->special_register_strings[i], 
                stack_info_y->special_register_strings[i]
            ) != 0
        ) {
            return false;
        }
    }

    for (size_t i = 0; i < stack_info_x->num_stacks; i++) {
        if (
            strcmp(
                stack_info_x->stack_literal_prefixes[i], 
                stack_info_y->stack_literal_prefixes[i]
            ) != 0
        ) {
            return false;
        }
    }

    if (
        memcmp(
            stack_info_x->arg_type_info, 
            stack_info_y->arg_type_info,
            stack_info_x->num_arg_types * sizeof(StackPtxArgTypeInfo)
        ) != 0
    ) {
        return false;
    }

    if (
        memcmp(
            stack_info_x->ptx_instruction_descriptors,
            stack_info_y->ptx_instruction_descriptors,
            stack_info_x->num_ptx_instructions * sizeof(StackPtxPtxInstructionDescriptor)
        ) != 0
    ) {
        return false;
    }

    if (
        memcmp(
            stack_info_x->special_register_descriptors,
            stack_info_y->special_register_descriptors,
            stack_info_x->num_special_registers * sizeof(StackPtxSpecialRegisterDescriptor)
        ) != 0
    ) {
        return false;
    }

    return true;

}

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEF
StackPtxInjectSerializeResult
stack_ptx_stack_info_print(
    const StackPtxStackInfo* stack_info
) {
    if (!stack_info) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INVALID_INPUT );
    }
    printf("Stack PTX Stack Info:\n");

    printf("\tPTX Instructions:\n");
    for (size_t i = 0; i < stack_info->num_ptx_instructions; i++) {
        printf("\t\t%s\n", stack_info->ptx_instruction_strings[i]);
    }

    printf("\tSpecial Registers:\n");
    for (size_t i = 0; i < stack_info->num_special_registers; i++) {
        printf("\t\t%s\n", stack_info->special_register_strings[i]);
    }

    printf("\tPTX Instruction Descriptors:\n");
    for (size_t i = 0; i < stack_info->num_ptx_instructions; i++) {
        const StackPtxPtxInstructionDescriptor* descriptor = &stack_info->ptx_instruction_descriptors[i];
        printf(
            "\t\t%zu : aligned=%u unique_stacks=%u\n",
            i,
            (unsigned)descriptor->is_aligned,
            (unsigned)descriptor->num_unique_stacks
        );
    }

    printf("\tSpecial Register Descriptors:\n");
    for (size_t i = 0; i < stack_info->num_special_registers; i++) {
        const StackPtxSpecialRegisterDescriptor* descriptor = &stack_info->special_register_descriptors[i];
        printf("\t\t%zu : %u\n", i, (unsigned)descriptor->arg_type_idx);
    }

    printf("\tLiteral Prefixes:\n");
    for (size_t i = 0; i < stack_info->num_stacks; i++) {
        printf("\t\t%s\n", stack_info->stack_literal_prefixes[i]);
    }

    printf("\tArg Type Info:\n");
    for (size_t i = 0; i < stack_info->num_arg_types; i++) {
        const StackPtxArgTypeInfo* arg_type_info = &stack_info->arg_type_info[i];
        printf(
            "\t\t%u : %u\n",
            (unsigned)arg_type_info->stack_idx,
            (unsigned)arg_type_info->num_vec_elems
        );
    }

    return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
}

static
inline
StackPtxInjectSerializeResult
_stack_ptx_registers_serialize_size(
    const StackPtxRegister* registers,
    size_t num_registers,
    size_t* buffer_bytes_written_out
) {
    size_t total = 0;
    total += sizeof(size_t);
    total += num_registers * sizeof(StackPtxStackIdx);
    for (size_t i = 0; i < num_registers; i++) {
        const char* name = registers[i].name;
        size_t name_size = strlen(name) + 1;
        total += name_size;
    }
    *buffer_bytes_written_out = total;

    return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
}

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEF
StackPtxInjectSerializeResult
stack_ptx_registers_serialize(
    const StackPtxRegister* registers,
    size_t num_registers,
    uint8_t* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_out
) {
    if (!registers || !buffer_bytes_written_out) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INVALID_INPUT );
    }

    _STACK_PTX_INJECT_SERIALIZE_CHECK_RET(
        _stack_ptx_registers_serialize_size(
            registers,
            num_registers,
            buffer_bytes_written_out
        )
    );

    if (!buffer) {
        return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
    }

    if (buffer_size < *buffer_bytes_written_out) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INSUFFICIENT_BUFFER );
    }

    uint8_t* p = (uint8_t*)buffer;

    memcpy(p, &num_registers, sizeof(size_t));
    p += sizeof(size_t);

    for (size_t i = 0; i < num_registers; i++) {
        StackPtxStackIdx stack_idx = registers[i].stack_idx;
        memcpy(p, &stack_idx, sizeof(StackPtxStackIdx));
        p += sizeof(StackPtxStackIdx);
    }

    for (size_t i = 0; i < num_registers; i++) {
        const char* name = registers[i].name;
        size_t name_size = strlen(name) + 1;
        memcpy(p, name, name_size);
        p += name_size;
    }

    size_t buffer_bytes_written = p - buffer;
    if (buffer_bytes_written > buffer_size || buffer_bytes_written != *buffer_bytes_written_out) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INSUFFICIENT_BUFFER );
    }

    return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
}

static
inline
StackPtxInjectSerializeResult
_stack_ptx_registers_deserialize_size(
    const uint8_t* wire,
    size_t wire_size,
    size_t* wire_used_out,
    size_t* buffer_bytes_written_out
) {
    size_t total_bytes = 0;
    total_bytes += _STACK_PTX_INJECT_SERIALIZE_ALIGNMENT;

    const uint8_t *p = wire;
    const uint8_t *end = wire + wire_size;

    size_t num_registers;

    memcpy(&num_registers, p, sizeof(size_t));
    p += sizeof(size_t);

    total_bytes += num_registers * sizeof(StackPtxRegister);
    p += num_registers * sizeof(StackPtxStackIdx);

    for (size_t i = 0; i < num_registers; i++) {
        size_t string_size = strlen(p) + 1;
        p += string_size;
        total_bytes += string_size;
    }

    *wire_used_out = p - wire;
    *buffer_bytes_written_out = total_bytes;

    return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
}

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEF
StackPtxInjectSerializeResult
stack_ptx_registers_deserialize(
    const uint8_t* wire,
    size_t wire_size,
    size_t* wire_used_out,
    uint8_t* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_out,
    StackPtxRegister** registers_out,
    size_t* num_registers_out
) {
    if (!wire || !wire_used_out || !buffer_bytes_written_out) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INVALID_INPUT );
    }

    _STACK_PTX_INJECT_SERIALIZE_CHECK_RET(
        _stack_ptx_registers_deserialize_size(
            wire,
            wire_size,
            wire_used_out,
            buffer_bytes_written_out
        )
    );

    if (!buffer) {
        return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
    }

    if (buffer && buffer_size < *buffer_bytes_written_out) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( PTX_INJECT_ERROR_INSUFFICIENT_BUFFER );
    }

    uint8_t* aligned_buffer = (uint8_t*)_STACK_PTX_INJECT_SERIALIZE_ALIGNMENT_UP((uintptr_t)buffer, _STACK_PTX_INJECT_SERIALIZE_ALIGNMENT);
    
    const uint8_t* p = wire;
    
    size_t num_registers;
    memcpy(&num_registers, p, sizeof(size_t));
    p += sizeof(size_t);

    *registers_out = (StackPtxRegister*)aligned_buffer;
    for (size_t i = 0; i < num_registers; i++) {
        StackPtxRegister* reg = &(*registers_out)[i];
        memcpy(&reg->stack_idx, p, sizeof(StackPtxStackIdx));
        p += sizeof(StackPtxStackIdx);
    }
    
    uint8_t* deserialize_offset = aligned_buffer + num_registers * sizeof(StackPtxRegister);
    for (size_t i = 0; i < num_registers; i++) {
        StackPtxRegister* reg = &(*registers_out)[i];
        size_t name_size = strlen(p) + 1;
        reg->name = deserialize_offset;
        memcpy(deserialize_offset, p, name_size);
        p += name_size;
        deserialize_offset += name_size;
    }

    size_t wire_used = p - wire;
    size_t buffer_bytes_written = deserialize_offset - aligned_buffer;

    if (wire_used != *wire_used_out) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INTERNAL );
    }

    if (buffer_bytes_written != *buffer_bytes_written_out - _STACK_PTX_INJECT_SERIALIZE_ALIGNMENT) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INTERNAL );
    }

    *num_registers_out = num_registers;

    return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
}

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEF
bool
stack_ptx_registers_equal(
    const StackPtxRegister* registers_x,
    size_t num_registers_x,
    const StackPtxRegister* registers_y,
    size_t num_registers_y
) {
    if (num_registers_x != num_registers_y) {
        return false;
    }

    for (size_t i = 0; i < num_registers_x; i++) {
        const StackPtxRegister* reg_x = &registers_x[i];
        const StackPtxRegister* reg_y = &registers_y[i];
        if (reg_x->stack_idx != reg_y->stack_idx) {
            return false;
        }

        size_t name_len_x = strlen(reg_x->name);
        size_t name_len_y = strlen(reg_y->name);

        if (name_len_x != name_len_y) {
            return false;
        }

        if (
            strcmp(
                reg_x->name,
                reg_y->name
            ) != 0
        ) {
            return false;
        }
    }
    
    return true;
}

static
inline
StackPtxInjectSerializeResult
_stack_ptx_requests_serialize_size(
    const size_t* const* request_stubs,
    const size_t* request_stubs_sizes,
    size_t num_request_stubs,
    size_t* buffer_bytes_written_out
) {
    size_t total = 0;
    total += sizeof(size_t);
    total += num_request_stubs * sizeof(size_t);

    for (size_t i = 0; i < num_request_stubs; i++) {
        size_t request_stub_size = request_stubs_sizes[i];
        total += request_stub_size * sizeof(size_t);
    }

    *buffer_bytes_written_out = total;
    return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
}

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEF
StackPtxInjectSerializeResult
stack_ptx_requests_serialize(
    const size_t* const* request_stubs,
    const size_t* request_stubs_sizes,
    size_t num_request_stubs,
    uint8_t* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_out
) {
    if (!request_stubs || !request_stubs_sizes || !buffer_bytes_written_out) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INVALID_INPUT );
    }

    _STACK_PTX_INJECT_SERIALIZE_CHECK_RET(
        _stack_ptx_requests_serialize_size(
            request_stubs,
            request_stubs_sizes,
            num_request_stubs,
            buffer_bytes_written_out
        )
    );

    if (!buffer) {
        return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
    }

    if (buffer_size < *buffer_bytes_written_out) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INSUFFICIENT_BUFFER );
    }

    uint8_t* p = buffer;

    memcpy(p, &num_request_stubs, sizeof(size_t));
    p += sizeof(size_t);

    for (size_t i = 0; i < num_request_stubs; i++) {
        const size_t request_stub_size = request_stubs_sizes[i];
        memcpy(p, &request_stub_size, sizeof(size_t));
        p += sizeof(size_t);
    }

    for (size_t i = 0; i < num_request_stubs; i++) {
        const size_t request_stub_size = request_stubs_sizes[i];
        const size_t* request_stub = request_stubs[i];
        for (size_t j = 0; j < request_stub_size; j++) {
            size_t request = request_stub[j];
            memcpy(p, &request, sizeof(size_t));
            p += sizeof(size_t);
        }
    }

    size_t buffer_bytes_written = p - buffer;
    if (buffer_bytes_written > buffer_size || buffer_bytes_written != *buffer_bytes_written_out) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INSUFFICIENT_BUFFER );
    }

    return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
}

static
inline
StackPtxInjectSerializeResult
_stack_ptx_requests_deserialize_size(
    const uint8_t* wire,
    size_t wire_size,
    size_t* wire_used_out,
    size_t* buffer_bytes_written_out
) {
    size_t total_bytes = 0;
    total_bytes += _STACK_PTX_INJECT_SERIALIZE_ALIGNMENT;
    
    const uint8_t* p = wire;

    size_t num_requests_stubs;
    memcpy(&num_requests_stubs, p, sizeof(size_t));
    p += sizeof(size_t);

    total_bytes += num_requests_stubs * sizeof(size_t);
    total_bytes += num_requests_stubs * sizeof(size_t*);
    size_t total_requests = 0;
    for (size_t i = 0; i < num_requests_stubs; i++) {
        size_t request_stubs_size;
        memcpy(&request_stubs_size, p, sizeof(size_t));
        p += sizeof(size_t);
        total_requests += request_stubs_size;
    }

    p += total_requests * sizeof(size_t);
    total_bytes += total_requests * sizeof(size_t);

    *wire_used_out = p - wire;
    *buffer_bytes_written_out = total_bytes;

    return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
}

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEF
StackPtxInjectSerializeResult
stack_ptx_requests_deserialize(
    const uint8_t* wire,
    size_t wire_size,
    size_t* wire_used_out,
    uint8_t* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_out,
    size_t*** requests_stubs_out,
    size_t** request_stubs_sizes_out,
    size_t* num_requests_stubs_out
) {
    if (!wire || !wire_used_out || !buffer_bytes_written_out) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INVALID_INPUT );
    }

    _STACK_PTX_INJECT_SERIALIZE_CHECK_RET(
        _stack_ptx_requests_deserialize_size(
            wire,
            wire_size,
            wire_used_out,
            buffer_bytes_written_out
        )
    );

    if (!buffer) {
        return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
    }

    if (buffer && buffer_size < *buffer_bytes_written_out) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INSUFFICIENT_BUFFER );
    }

    uint8_t* aligned_buffer = (uint8_t*)_STACK_PTX_INJECT_SERIALIZE_ALIGNMENT_UP((uintptr_t)buffer, _STACK_PTX_INJECT_SERIALIZE_ALIGNMENT);

    const uint8_t* p = wire;
    size_t num_requests_stubs;
    memcpy(&num_requests_stubs, p, sizeof(size_t));
    p += sizeof(size_t);

    *num_requests_stubs_out = num_requests_stubs;

    uint8_t* deserialize_offset = aligned_buffer;
    *request_stubs_sizes_out = (size_t*)deserialize_offset;

    memcpy(*request_stubs_sizes_out, p, num_requests_stubs * sizeof(size_t));
    p += num_requests_stubs * sizeof(size_t);
    deserialize_offset += num_requests_stubs * sizeof(size_t);

    *requests_stubs_out = (size_t**)deserialize_offset;
    deserialize_offset += num_requests_stubs * sizeof(size_t*);

    size_t total_requests = 0;
    for (size_t i = 0; i < num_requests_stubs; i++) {
        (*requests_stubs_out)[i] = (size_t*)(deserialize_offset + total_requests * sizeof(size_t));
        size_t* request_stub = (*requests_stubs_out)[i];

        size_t request_stubs_size = (*request_stubs_sizes_out)[i];
        total_requests += request_stubs_size;
    }
    memcpy(deserialize_offset, p, total_requests * sizeof(size_t));
    deserialize_offset += total_requests * sizeof(size_t);
    p += total_requests * sizeof(size_t);

    size_t wire_used = p - wire;
    size_t buffer_bytes_written = deserialize_offset - aligned_buffer;

    if (wire_used != *wire_used_out) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INTERNAL );
    }

    if (buffer_bytes_written != *buffer_bytes_written_out - _STACK_PTX_INJECT_SERIALIZE_ALIGNMENT) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INTERNAL );
    }

    return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
}

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEF
bool
stack_ptx_requests_equal(
    const size_t* const* requests_stubs_x,
    const size_t* request_stubs_sizes_x,
    size_t num_request_stubs_x,
    const size_t* const* requests_stubs_y,
    const size_t* request_stubs_sizes_y,
    size_t num_request_stubs_y
) {
    if (num_request_stubs_x != num_request_stubs_y) {
        return false;
    }

    for (size_t i = 0; i < num_request_stubs_x; i++) {
        size_t request_stubs_size_x = request_stubs_sizes_x[i];
        size_t request_stubs_size_y = request_stubs_sizes_y[i];
        if (request_stubs_size_x != request_stubs_size_y) {
            return false;
        }

        const size_t* requests_stub_x = requests_stubs_x[i];
        const size_t* requests_stub_y = requests_stubs_y[i];

        for (size_t j = 0; j < request_stubs_size_x; j++) {
            if (requests_stub_x[j] != requests_stub_y[j]) {
                return false;
            }
        }
    }

    return true;
}

static
inline
StackPtxInjectSerializeResult
_stack_ptx_instructions_serialize_size(
    const StackPtxInstruction* const* instruction_stubs,
    size_t num_instruction_stubs,
    size_t* buffer_bytes_written_out
) {
    size_t total = 0;
    total += sizeof(size_t);

    size_t total_instructions = 0;
    for (size_t i = 0; i < num_instruction_stubs; i++) {
        const StackPtxInstruction* instruction_stub = instruction_stubs[i];
        size_t instruction_idx = 0;
        while(true) {
            StackPtxInstruction instruction = instruction_stub[instruction_idx++];
            total_instructions++;
            if (instruction.instruction_type == STACK_PTX_INSTRUCTION_TYPE_RETURN) {
                break;
            }
        }
    }
    total += total_instructions * sizeof(StackPtxInstruction);

    *buffer_bytes_written_out = total;
    return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
}

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEF
StackPtxInjectSerializeResult
stack_ptx_instructions_serialize(
    const StackPtxInstruction* const* instruction_stubs,
    size_t num_instruction_stubs,
    uint8_t* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_out
) {
    if (!instruction_stubs || !buffer_bytes_written_out) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INVALID_INPUT );
    }

    _STACK_PTX_INJECT_SERIALIZE_CHECK_RET(
        _stack_ptx_instructions_serialize_size(
            instruction_stubs,
            num_instruction_stubs,
            buffer_bytes_written_out
        )
    );

    if (!buffer) {
        return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
    }

    if (buffer_size < *buffer_bytes_written_out) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INSUFFICIENT_BUFFER );
    }

    uint8_t* p = buffer;

    memcpy(p, &num_instruction_stubs, sizeof(size_t));
    p += sizeof(size_t);

    for (size_t i = 0; i < num_instruction_stubs; i++) {
        const StackPtxInstruction* instruction_stub = instruction_stubs[i];
        size_t instruction_idx = 0;
        while(true) {
            StackPtxInstruction instruction = instruction_stub[instruction_idx++];
            memcpy(p, &instruction, sizeof(StackPtxInstruction));
            p += sizeof(StackPtxInstruction);
            if (instruction.instruction_type == STACK_PTX_INSTRUCTION_TYPE_RETURN) {
                break;
            }
        }
    }

    size_t buffer_bytes_written = p - buffer;
    if (buffer_bytes_written > buffer_size || buffer_bytes_written != *buffer_bytes_written_out) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INSUFFICIENT_BUFFER );
    }

    return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
}

static
inline
StackPtxInjectSerializeResult
_stack_ptx_instructions_deserialize_size(
    const uint8_t* wire,
    size_t wire_size,
    size_t* wire_used_out,
    size_t* buffer_bytes_written_out
) {
    size_t total_bytes = 0;
    total_bytes += _STACK_PTX_INJECT_SERIALIZE_ALIGNMENT;
    
    const uint8_t* p = wire;

    size_t num_instruction_stubs;
    memcpy(&num_instruction_stubs, p, sizeof(size_t));
    p += sizeof(size_t);

    total_bytes += num_instruction_stubs * sizeof(StackPtxInstruction*);
    for (size_t i = 0; i < num_instruction_stubs; i++) {
        while (true) {
            StackPtxInstruction instruction;
            memcpy(&instruction, p, sizeof(StackPtxInstruction));
            p += sizeof(StackPtxInstruction);
            total_bytes += sizeof(StackPtxInstruction);
            if (instruction.instruction_type == STACK_PTX_INSTRUCTION_TYPE_RETURN) {
                break;
            }
        }
    }

    *wire_used_out = p - wire;
    *buffer_bytes_written_out = total_bytes;

    return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
}

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEF
StackPtxInjectSerializeResult
stack_ptx_instructions_deserialize(
    const uint8_t* wire,
    size_t wire_size,
    size_t* wire_used_out,
    uint8_t* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_out,
    StackPtxInstruction*** instruction_stubs_out,
    size_t* num_instruction_stubs_out
) {
    if (!wire || !wire_used_out || !buffer_bytes_written_out) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INVALID_INPUT );
    }

    _STACK_PTX_INJECT_SERIALIZE_CHECK_RET(
        _stack_ptx_instructions_deserialize_size(
            wire,
            wire_size,
            wire_used_out,
            buffer_bytes_written_out
        )
    );

    if (!buffer) {
        return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
    }

    if (buffer && buffer_size < *buffer_bytes_written_out) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INSUFFICIENT_BUFFER );
    }

    uint8_t* aligned_buffer = (uint8_t*)_STACK_PTX_INJECT_SERIALIZE_ALIGNMENT_UP((uintptr_t)buffer, _STACK_PTX_INJECT_SERIALIZE_ALIGNMENT);

    const uint8_t* p = wire;
    size_t num_instructs_stubs;
    memcpy(&num_instructs_stubs, p, sizeof(size_t));
    p += sizeof(size_t);

    *num_instruction_stubs_out = num_instructs_stubs;

    uint8_t* deserialize_offset = aligned_buffer;

    *instruction_stubs_out = (StackPtxInstruction**)deserialize_offset;
    deserialize_offset += num_instructs_stubs * sizeof(StackPtxInstruction*);

    size_t total_instruction = 0;
    for (size_t i = 0; i < num_instructs_stubs; i++) {
        (*instruction_stubs_out)[i] = (StackPtxInstruction*)deserialize_offset;
        StackPtxInstruction* request_stub = (*instruction_stubs_out)[i];
        while(true) {
            StackPtxInstruction instruction;
            memcpy(&instruction, p, sizeof(StackPtxInstruction));
            memcpy(deserialize_offset, &instruction, sizeof(StackPtxInstruction));
            p += sizeof(StackPtxInstruction);
            deserialize_offset += sizeof(StackPtxInstruction);
            if (instruction.instruction_type == STACK_PTX_INSTRUCTION_TYPE_RETURN) {
                break;
            }
        }
    }

    size_t wire_used = p - wire;
    size_t buffer_bytes_written = deserialize_offset - aligned_buffer;

    if (wire_used != *wire_used_out) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INTERNAL );
    }

    if (buffer_bytes_written != *buffer_bytes_written_out - _STACK_PTX_INJECT_SERIALIZE_ALIGNMENT) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INTERNAL );
    }

    return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
}

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEF
bool
stack_ptx_instructions_equal(
    const StackPtxInstruction* const* instruction_stubs_x,
    size_t num_instruction_stubs_x,
    const StackPtxInstruction* const* instruction_stubs_y,
    size_t num_instruction_stubs_y
) {
    if (num_instruction_stubs_x != num_instruction_stubs_y) {
        return false;
    }

    for (size_t i = 0; i < num_instruction_stubs_x; i++) {
        const StackPtxInstruction* instruction_stub_x = instruction_stubs_x[i];
        const StackPtxInstruction* instruction_stub_y = instruction_stubs_y[i];

        size_t instruction_idx = 0;
        while(true) {
            StackPtxInstruction instruction_x = instruction_stub_x[instruction_idx];
            StackPtxInstruction instruction_y = instruction_stub_y[instruction_idx];
            if (
                memcmp(
                    &instruction_x,
                    &instruction_y,
                    sizeof(StackPtxInstruction)
                ) != 0
            ) {
                return false;
            }

            if (instruction_x.instruction_type == STACK_PTX_INSTRUCTION_TYPE_RETURN) {
                break;
            }

            instruction_idx++;
        }
    }

    return true;
}

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEF
StackPtxInjectSerializeResult
ptx_inject_ptx_serialize(
    const char* ptx,
    uint8_t* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_out
) {
    if (!ptx || !buffer_bytes_written_out) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INVALID_INPUT );
    }

    size_t ptx_size = strlen(ptx) + 1;

    *buffer_bytes_written_out = ptx_size;

    if (!buffer) {
        return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
    }

    if (buffer_size < *buffer_bytes_written_out) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INSUFFICIENT_BUFFER );
    }

    uint8_t* p = buffer;
    
    memcpy(p, ptx, ptx_size);
    p += ptx_size;

    size_t buffer_bytes_written = p - buffer;
    if (buffer_bytes_written > buffer_size || buffer_bytes_written != *buffer_bytes_written_out) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INSUFFICIENT_BUFFER );
    }

    return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
}

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEF
StackPtxInjectSerializeResult
ptx_inject_ptx_deserialize(
    const uint8_t* wire,
    size_t wire_size,
    size_t* wire_used_out,
    uint8_t* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_out,
    char** ptx
) {
    if (!wire || !wire_used_out || !buffer_bytes_written_out) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INVALID_INPUT );
    }

    const uint8_t* p = wire;

    size_t ptx_size = strlen(p) + 1;

    *wire_used_out = ptx_size;
    *buffer_bytes_written_out = ptx_size + _STACK_PTX_INJECT_SERIALIZE_ALIGNMENT;

    if (!buffer) {
        return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
    }

    if (buffer && buffer_size < *buffer_bytes_written_out) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INSUFFICIENT_BUFFER );
    }

    uint8_t* aligned_buffer = (uint8_t*)_STACK_PTX_INJECT_SERIALIZE_ALIGNMENT_UP((uintptr_t)buffer, _STACK_PTX_INJECT_SERIALIZE_ALIGNMENT);

    uint8_t* deserialize_offset = aligned_buffer;

    *ptx = deserialize_offset;
    memcpy(deserialize_offset, p, ptx_size);
    p += ptx_size;
    deserialize_offset += ptx_size;

    size_t wire_used = p - wire;
    size_t buffer_bytes_written = deserialize_offset - aligned_buffer;

    if (wire_used != *wire_used_out) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INTERNAL );
    }

    if (buffer_bytes_written != *buffer_bytes_written_out - _STACK_PTX_INJECT_SERIALIZE_ALIGNMENT) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INTERNAL );
    }

    return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
}

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEF
bool
ptx_inject_ptx_equal(
    const char* ptx_x,
    const char* ptx_y
) {
    size_t ptx_size_x = strlen(ptx_x) + 1;
    size_t ptx_size_y = strlen(ptx_y) + 1;

    if (ptx_size_x != ptx_size_y) {
        return false;
    }

    if (strcmp(ptx_x, ptx_y) != 0) {
        return false;
    }

    return true;
}

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEF
StackPtxInjectSerializeResult
stack_ptx_extra_serialize(
    const StackPtxExtraInfo* extra_ref,
    uint8_t* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_out
) {
    if (!extra_ref || !buffer_bytes_written_out) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INVALID_INPUT );
    }

    *buffer_bytes_written_out = sizeof(StackPtxExtraInfo);

    if(!buffer) {
        return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
    }

    if (buffer_size < *buffer_bytes_written_out) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INSUFFICIENT_BUFFER );
    }

    memcpy(buffer, extra_ref, sizeof(StackPtxExtraInfo));

    return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
}

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEF
StackPtxInjectSerializeResult
stack_ptx_extra_deserialize(
    const uint8_t* wire,
    size_t wire_size,
    size_t* wire_used_out,
    uint8_t* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_out,
    StackPtxExtraInfo** extra_out
) {
    if (!wire || !wire_used_out || !buffer_bytes_written_out) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INVALID_INPUT );
    }

    *wire_used_out = sizeof(StackPtxExtraInfo);
    *buffer_bytes_written_out = sizeof(StackPtxExtraInfo) + _STACK_PTX_INJECT_SERIALIZE_ALIGNMENT;

    if (!buffer) {
        return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
    }

    if (buffer && buffer_size < *buffer_bytes_written_out) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INSUFFICIENT_BUFFER );
    }

    buffer = (uint8_t*)_STACK_PTX_INJECT_SERIALIZE_ALIGNMENT_UP((uintptr_t)buffer, _STACK_PTX_INJECT_SERIALIZE_ALIGNMENT);

    *extra_out = (StackPtxExtraInfo*)buffer;
    memcpy(buffer, wire, sizeof(StackPtxExtraInfo));

    return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
}

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEF
bool
stack_ptx_extra_equal(
    const StackPtxExtraInfo* extra_x,
    const StackPtxExtraInfo* extra_y
) {
    if (
        extra_x->device_capability_major != extra_y->device_capability_major ||
        extra_x->device_capability_minor != extra_y->device_capability_minor ||
        extra_x->execution_limit != extra_y->execution_limit
    ) {
        return false;
    }
    return true;
}

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEF
StackPtxInjectSerializeResult
stack_ptx_inject_compiler_state_serialize(
    const StackPtxInjectCompilerStateSerialize* compiler_state,
    uint8_t* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_out
) {
    *buffer_bytes_written_out = 0;

    size_t total_bytes = 0;
    size_t buffer_offset = 0;

    _STACK_PTX_INJECT_SERIALIZE_CHECK_RET(
        ptx_inject_ptx_serialize(
            compiler_state->annotated_ptx,
            buffer ? buffer + total_bytes : NULL,
            buffer ? buffer_size - total_bytes : 0,
            &buffer_offset
        )
    );
    total_bytes += buffer_offset;

    _STACK_PTX_INJECT_SERIALIZE_CHECK_RET(
        stack_ptx_compiler_info_serialize(
            compiler_state->compiler_info,
            buffer ? buffer + total_bytes : NULL,
            buffer ? buffer_size - total_bytes : 0,
            &buffer_offset
        )
    );
    total_bytes += buffer_offset;

    _STACK_PTX_INJECT_SERIALIZE_CHECK_RET(
        stack_ptx_stack_info_serialize(
            compiler_state->stack_info,
            buffer ? buffer + total_bytes : NULL,
            buffer ? buffer_size - total_bytes : 0,
            &buffer_offset
        )
    );
    total_bytes += buffer_offset;

    _STACK_PTX_INJECT_SERIALIZE_CHECK_RET(
        stack_ptx_extra_serialize(
            compiler_state->extra,
            buffer ? buffer + total_bytes : NULL,
            buffer ? buffer_size - total_bytes : 0,
            &buffer_offset
        )
    );
    total_bytes += buffer_offset;

    _STACK_PTX_INJECT_SERIALIZE_CHECK_RET(
        stack_ptx_registers_serialize(
            compiler_state->registers,
            compiler_state->num_registers,
            buffer ? buffer + total_bytes : NULL,
            buffer ? buffer_size - total_bytes : 0,
            &buffer_offset
        )
    );
    total_bytes += buffer_offset;

    _STACK_PTX_INJECT_SERIALIZE_CHECK_RET(
        stack_ptx_requests_serialize(
            compiler_state->request_stubs,
            compiler_state->request_stub_sizes,
            compiler_state->num_request_stubs,
            buffer ? buffer + total_bytes : NULL,
            buffer ? buffer_size - total_bytes : 0,
            &buffer_offset
        )
    );
    total_bytes += buffer_offset;

    *buffer_bytes_written_out = total_bytes;

    return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
}

static
inline
StackPtxInjectSerializeResult
_stack_ptx_inject_compiler_state_deserialize_size(
    const uint8_t* wire,
    size_t wire_size,
    size_t* wire_used_out,
    size_t* buffer_bytes_written_out
) {
    size_t total_bytes = 0;
    total_bytes += _STACK_PTX_INJECT_SERIALIZE_ALIGNMENT;
    total_bytes += sizeof(StackPtxInjectCompilerStateDeserialize);

    const uint8_t* p = wire;

    size_t wire_used;
    size_t total_wire_used = 0;
    size_t buffer_offset;

    _STACK_PTX_INJECT_SERIALIZE_CHECK_RET(
        ptx_inject_ptx_deserialize(
            wire + total_wire_used,
            wire_size - total_wire_used,
            &wire_used,
            NULL,
            0,
            &buffer_offset,
            NULL
        )
    );
    total_wire_used += wire_used;
    total_bytes += buffer_offset;

    _STACK_PTX_INJECT_SERIALIZE_CHECK_RET(
        stack_ptx_compiler_info_deserialize(
            wire + total_wire_used,
            wire_size - total_wire_used,
            &wire_used,
            NULL,
            0,
            &buffer_offset,
            NULL
        )
    );
    total_wire_used += wire_used;
    total_bytes += buffer_offset;

    _STACK_PTX_INJECT_SERIALIZE_CHECK_RET(
        stack_ptx_stack_info_deserialize(
            wire + total_wire_used,
            wire_size - total_wire_used,
            &wire_used,
            NULL,
            0,
            &buffer_offset,
            NULL
        )
    );
    total_wire_used += wire_used;
    total_bytes += buffer_offset;

    _STACK_PTX_INJECT_SERIALIZE_CHECK_RET(
        stack_ptx_extra_deserialize(
            wire + total_wire_used,
            wire_size - total_wire_used,
            &wire_used,
            NULL,
            0,
            &buffer_offset,
            NULL
        )
    );
    total_wire_used += wire_used;
    total_bytes += buffer_offset;

    _STACK_PTX_INJECT_SERIALIZE_CHECK_RET(
        stack_ptx_registers_deserialize(
            wire + total_wire_used,
            wire_size - total_wire_used,
            &wire_used,
            NULL,
            0,
            &buffer_offset,
            NULL,
            NULL
        )
    );
    total_wire_used += wire_used;
    total_bytes += buffer_offset;

    _STACK_PTX_INJECT_SERIALIZE_CHECK_RET(
        stack_ptx_requests_deserialize(
            wire + total_wire_used,
            wire_size - total_wire_used,
            &wire_used,
            NULL,
            0,
            &buffer_offset,
            NULL,
            NULL,
            NULL
        )
    );
    total_wire_used += wire_used;
    total_bytes += buffer_offset;

    *wire_used_out = total_wire_used;
    *buffer_bytes_written_out = total_bytes;

    return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
}

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEF
StackPtxInjectSerializeResult
stack_ptx_inject_compiler_state_deserialize(
    const uint8_t* wire,
    size_t wire_size,
    size_t* wire_used_out,
    uint8_t* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_out,
    StackPtxInjectCompilerStateDeserialize** compiler_state_out
) {
    if (!wire || !wire_used_out || !buffer_bytes_written_out) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INVALID_INPUT );
    }

    _STACK_PTX_INJECT_SERIALIZE_CHECK_RET(
        _stack_ptx_inject_compiler_state_deserialize_size(
            wire,
            wire_size,
            wire_used_out,
            buffer_bytes_written_out
        )
    );

    if (!buffer) {
        return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
    }

    if (buffer && buffer_size < *buffer_bytes_written_out) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INSUFFICIENT_BUFFER );
    }

    size_t wire_used;
    size_t total_wire_used = 0;
    size_t buffer_offset;
    size_t total_bytes = 0;

    uint8_t* aligned_buffer = (uint8_t*)_STACK_PTX_INJECT_SERIALIZE_ALIGNMENT_UP((uintptr_t)buffer, _STACK_PTX_INJECT_SERIALIZE_ALIGNMENT);

    *compiler_state_out = (StackPtxInjectCompilerStateDeserialize*)(aligned_buffer + total_bytes);
    total_bytes += sizeof(StackPtxInjectCompilerStateDeserialize);

    _STACK_PTX_INJECT_SERIALIZE_CHECK_RET(
        ptx_inject_ptx_deserialize(
            wire + total_wire_used,
            wire_size - total_wire_used,
            &wire_used,
            buffer ? aligned_buffer + total_bytes : NULL,
            buffer ? buffer_size - total_bytes : 0,
            &buffer_offset,
            &(*compiler_state_out)->annotated_ptx
        )
    );
    total_bytes += buffer_offset;
    total_wire_used += wire_used;

    _STACK_PTX_INJECT_SERIALIZE_CHECK_RET(
        stack_ptx_compiler_info_deserialize(
            wire + total_wire_used,
            wire_size - total_wire_used,
            &wire_used,
            buffer ? aligned_buffer + total_bytes : NULL,
            buffer ? buffer_size - total_bytes : 0,
            &buffer_offset,
            &(*compiler_state_out)->compiler_info
        )
    );
    total_bytes += buffer_offset;
    total_wire_used += wire_used;

    _STACK_PTX_INJECT_SERIALIZE_CHECK_RET(
        stack_ptx_stack_info_deserialize(
            wire + total_wire_used,
            wire_size - total_wire_used,
            &wire_used,
            buffer ? aligned_buffer + total_bytes : NULL,
            buffer ? buffer_size - total_bytes : 0,
            &buffer_offset,
            &(*compiler_state_out)->stack_info
        )
    );
    total_bytes += buffer_offset;
    total_wire_used += wire_used;

    _STACK_PTX_INJECT_SERIALIZE_CHECK_RET(
        stack_ptx_extra_deserialize(
            wire + total_wire_used,
            wire_size - total_wire_used,
            &wire_used,
            buffer ? aligned_buffer + total_bytes : NULL,
            buffer ? buffer_size - total_bytes : 0,
            &buffer_offset,
            &(*compiler_state_out)->extra
        )
    );
    total_bytes += buffer_offset;
    total_wire_used += wire_used;

    _STACK_PTX_INJECT_SERIALIZE_CHECK_RET(
        stack_ptx_registers_deserialize(
            wire + total_wire_used,
            wire_size - total_wire_used,
            &wire_used,
            buffer ? aligned_buffer + total_bytes : NULL,
            buffer ? buffer_size - total_bytes : 0,
            &buffer_offset,
            &(*compiler_state_out)->registers,
            &(*compiler_state_out)->num_registers
        )
    );
    total_bytes += buffer_offset;
    total_wire_used += wire_used;

    _STACK_PTX_INJECT_SERIALIZE_CHECK_RET(
        stack_ptx_requests_deserialize(
            wire + total_wire_used,
            wire_size - total_wire_used,
            &wire_used,
            buffer ? aligned_buffer + total_bytes : NULL,
            buffer ? buffer_size - total_bytes : 0,
            &buffer_offset,
            &(*compiler_state_out)->request_stubs,
            &(*compiler_state_out)->request_stub_sizes,
            &(*compiler_state_out)->num_request_stubs
        )
    );
    total_bytes += buffer_offset;
    total_wire_used += wire_used;
    
    if (buffer && buffer_size < total_bytes) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INSUFFICIENT_BUFFER );
    }

    if (total_wire_used != *wire_used_out) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INTERNAL );
    }

    if (total_bytes != *buffer_bytes_written_out - _STACK_PTX_INJECT_SERIALIZE_ALIGNMENT) {
        _STACK_PTX_INJECT_SERIALIZE_ERROR( STACK_PTX_INJECT_SERIALIZE_ERROR_INTERNAL );
    }

    *wire_used_out = total_wire_used;
    *buffer_bytes_written_out = total_bytes;

    return STACK_PTX_INJECT_SERIALIZE_SUCCESS;
}

STACK_PTX_INJECT_SERIALIZE_PUBLIC_DEF
bool
stack_ptx_inject_compiler_state_equal(
    const StackPtxInjectCompilerStateSerialize* compiler_state_x,
    StackPtxInjectCompilerStateDeserialize* compiler_state_y
) {
    if (
        !ptx_inject_ptx_equal(
            compiler_state_x->annotated_ptx,
            compiler_state_y->annotated_ptx
        )
    ) {
        return false;
    }

    if (
        !stack_ptx_compiler_info_equal(
            compiler_state_x->compiler_info,
            compiler_state_y->compiler_info
        )
    ) {
        return false;
    }

    if ( 
        !stack_ptx_stack_info_equal(
            compiler_state_x->stack_info,
            compiler_state_y->stack_info
        )
    ) {
        return false;
    }

    if (
        !stack_ptx_extra_equal(
            compiler_state_x->extra,
            compiler_state_y->extra
        )
    ) {
        return false;
    }

    if (
        !stack_ptx_registers_equal(
            compiler_state_x->registers,
            compiler_state_x->num_registers,
            compiler_state_y->registers,
            compiler_state_y->num_registers
        )
    ) {
        return false;
    }

    if (
        !stack_ptx_requests_equal(
            compiler_state_x->request_stubs,
            compiler_state_x->request_stub_sizes,
            compiler_state_x->num_request_stubs,
            (const size_t* const*)compiler_state_y->request_stubs,
            compiler_state_y->request_stub_sizes,
            compiler_state_y->num_request_stubs
        )
    ) {
        return false;
    }

    return true;
}

#endif // STACK_PTX_INJECT_SERIALIZE_IMPLEMENTATION_ONCE
#endif // STACK_PTX_INJECT_SERIALIZE_IMPLEMENTATION
