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
// SPDX-FileCopyrightText: 2026 Charles Durham
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <vector>

#define STACK_PTX_IMPLEMENTATION
#include <stack_ptx.h>

namespace nb = nanobind;

NB_MODULE(_stack_ptx, m) {
    nb::enum_<StackPtxResult>(m, "StackPtxResult", "Stack PTX status type returns")
        .value("STACK_PTX_SUCCESS", STACK_PTX_SUCCESS, "Success (0)")
        .value("STACK_PTX_ERROR_INTERNAL", STACK_PTX_ERROR_INTERNAL, "Stack PTX internal error occurred.")
        .value("STACK_PTX_ERROR_WRONG_TYPE_DISPATCH", STACK_PTX_ERROR_WRONG_TYPE_DISPATCH, "An unexpected instruction type was processed internally.")
        .value("STACK_PTX_ERROR_INSUFFICIENT_BUFFER", STACK_PTX_ERROR_INSUFFICIENT_BUFFER, "The buffer passed in is not large enough.")
        .value("STACK_PTX_ERROR_INVALID_VALUE", STACK_PTX_ERROR_INVALID_VALUE, "A value passed to the function is wrong.")
        .value("STACK_PTX_ERROR_INSUFFICIENT_AST_SIZE", STACK_PTX_ERROR_INSUFFICIENT_AST_SIZE, "AST is not large enough to contain program.")
        .value("STACK_PTX_ERROR_INSUFFICIENT_AST_VISIT_SIZE", STACK_PTX_ERROR_INSUFFICIENT_AST_VISIT_SIZE, "AST visit size is not large enough to traverse AST.")
        .value("STACK_PTX_ERROR_BAD_INSTRUCTION_MAYBE_FORGOT_RETURN", STACK_PTX_ERROR_BAD_INSTRUCTION_MAYBE_FORGOT_RETURN, "A StackPtxInstruction has an invalid instruction type, did you forget to terminate the instruction array with a return?")
        .value("STACK_PTX_ERROR_INSUFFICIENT_WORKSPACE", STACK_PTX_ERROR_INSUFFICIENT_WORKSPACE, "The buffer passed in is not large enough.")
        .value("STACK_PTX_ERROR_PTX_INSTRUCTION_IDX_OUT_OF_BOUNDS", STACK_PTX_ERROR_PTX_INSTRUCTION_IDX_OUT_OF_BOUNDS, "The PTX Instruction index is out of bounds")
        .value("STACK_PTX_ERROR_META_INSTRUCTION_IDX_OUT_OF_BOUNDS", STACK_PTX_ERROR_META_INSTRUCTION_IDX_OUT_OF_BOUNDS, "The meta instruction index is out of bounds")
        .value("STACK_PTX_ERROR_INPUT_IDX_OUT_OF_BOUNDS", STACK_PTX_ERROR_INPUT_IDX_OUT_OF_BOUNDS, "The input index is out of bounds")
        .value("STACK_PTX_ERROR_REGISTER_IDX_OUT_OF_BOUNDS", STACK_PTX_ERROR_REGISTER_IDX_OUT_OF_BOUNDS, "The register index is out of bounds")
        .value("STACK_PTX_ERROR_SPECIAL_REGISTER_IDX_OUT_OF_BOUNDS", STACK_PTX_ERROR_SPECIAL_REGISTER_IDX_OUT_OF_BOUNDS, "The special register index is out of bounds")
        .value("STACK_PTX_ERROR_STACK_IDX_OUT_OF_BOUNDS", STACK_PTX_ERROR_STACK_IDX_OUT_OF_BOUNDS, "The stack index is out of bounds")
        .value("STACK_PTX_ERROR_ARG_IDX_OUT_OF_BOUNDS", STACK_PTX_ERROR_ARG_IDX_OUT_OF_BOUNDS, "The argument index is out of bounds")
        .value("STACK_PTX_ERROR_ROUTINES_IDX_OUT_OF_BOUNDS", STACK_PTX_ERROR_ROUTINES_IDX_OUT_OF_BOUNDS, "The routines index is out of bounds")
        .value("STACK_PTX_ERROR_STORE_IDX_OUT_OF_BOUNDS", STACK_PTX_ERROR_STORE_IDX_OUT_OF_BOUNDS, "The store index is out of bounds")
        .value("STACK_PTX_ERROR_LOAD_IDX_OUT_OF_BOUNDS", STACK_PTX_ERROR_LOAD_IDX_OUT_OF_BOUNDS, "The load index is out of bounds")
        .value("STACK_PTX_RESULT_NUM_ELEMS", STACK_PTX_RESULT_NUM_ELEMS, "The number of result enums.")
        .export_values();

    nb::enum_<StackPtxInstructionType>(m, "StackPtxInstructionType", "Stack PTX instruction type")
        .value("STACK_PTX_INSTRUCTION_TYPE_NONE", STACK_PTX_INSTRUCTION_TYPE_NONE, "No instruction type")
        .value("STACK_PTX_INSTRUCTION_TYPE_PTX", STACK_PTX_INSTRUCTION_TYPE_PTX, "PTX instruction type")
        .value("STACK_PTX_INSTRUCTION_TYPE_CONSTANT", STACK_PTX_INSTRUCTION_TYPE_CONSTANT, "Constant instruction type")
        .value("STACK_PTX_INSTRUCTION_TYPE_INPUT", STACK_PTX_INSTRUCTION_TYPE_INPUT, "Input instruction type")
        .value("STACK_PTX_INSTRUCTION_TYPE_SPECIAL", STACK_PTX_INSTRUCTION_TYPE_SPECIAL, "Special instruction type")
        .value("STACK_PTX_INSTRUCTION_TYPE_META", STACK_PTX_INSTRUCTION_TYPE_META, "Meta instruction type")
        .value("STACK_PTX_INSTRUCTION_TYPE_ROUTINE", STACK_PTX_INSTRUCTION_TYPE_ROUTINE, "Routine instruction type")
        .value("STACK_PTX_INSTRUCTION_TYPE_REGISTER", STACK_PTX_INSTRUCTION_TYPE_REGISTER, "Register instruction type")
        .value("STACK_PTX_INSTRUCTION_TYPE_AST_IDX", STACK_PTX_INSTRUCTION_TYPE_AST_IDX, "AST index instruction type")
        .value("STACK_PTX_INSTRUCTION_TYPE_RETURN", STACK_PTX_INSTRUCTION_TYPE_RETURN, "Return instruction type")
        .value("STACK_PTX_INSTRUCTION_TYPE_STORE", STACK_PTX_INSTRUCTION_TYPE_STORE, "Store instruction type")
        .value("STACK_PTX_INSTRUCTION_TYPE_LOAD", STACK_PTX_INSTRUCTION_TYPE_LOAD, "Load index instruction type")
        .export_values();

    nb::enum_<StackPtxMetaInstruction>(m, "StackPtxMetaInstruction", "Stack PTX meta instruction type")
        .value("STACK_PTX_META_INSTRUCTION_CONSTANT", STACK_PTX_META_INSTRUCTION_CONSTANT, "Pushes an s32 constant onto the special push_stack. Should be encoded with stack_ptx_encode_meta_constant.")
        .value("STACK_PTX_META_INSTRUCTION_DUP", STACK_PTX_META_INSTRUCTION_DUP, "Duplicates a value at the top of the respective stack. Should be encoded with stack_ptx_encode_meta.")
        .value("STACK_PTX_META_INSTRUCTION_YANK_DUP", STACK_PTX_META_INSTRUCTION_YANK_DUP, "Duplicates a value at a depth indicated by the value popped from the push_stack. No-op if push_stack is empty. Should be encoded with stack_ptx_encode_meta.")
        .value("STACK_PTX_META_INSTRUCTION_SWAP", STACK_PTX_META_INSTRUCTION_SWAP, "Swaps the two values at the top of the indicated stack. Should be encoded with stack_ptx_encode_meta.")
        .value("STACK_PTX_META_INSTRUCTION_SWAP_WITH", STACK_PTX_META_INSTRUCTION_SWAP_WITH, "Swaps the value at the top of the indicated stack with the value at a depth obtained from the push_stack. Should be encoded with stack_ptx_encode_meta.")
        .value("STACK_PTX_META_INSTRUCTION_REPLACE", STACK_PTX_META_INSTRUCTION_REPLACE, "Replaces the value at the top of the indicated stack with the value at a depth obtained from the push_stack. Should be encoded with stack_ptx_encode_meta.")
        .value("STACK_PTX_META_INSTRUCTION_DROP", STACK_PTX_META_INSTRUCTION_DROP, "Drops the top x values of the indicated stack where x is obtained from the push_stack. Should be encoded with stack_ptx_encode_meta.")
        .value("STACK_PTX_META_INSTRUCTION_ROTATE", STACK_PTX_META_INSTRUCTION_ROTATE, "Takes the value at the top of the indicated stack and pushes it 2 deep. Should be encoded with stack_ptx_encode_meta.")
        .value("STACK_PTX_META_INSTRUCTION_REVERSE", STACK_PTX_META_INSTRUCTION_REVERSE, "Reverses the indicated stack. Should be encoded with stack_ptx_encode_meta.")
        .value("STACK_PTX_META_INSTRUCTION_NUM_ENUMS", STACK_PTX_META_INSTRUCTION_NUM_ENUMS, "The number of meta instruction enums.")
        .export_values();

    nb::class_<StackPtxPTXArgs>(m, "StackPtxPTXArgs", "Stack PTX arguments structure")
        .def(nb::init<>(), "Default constructor")
        .def(nb::init<size_t, size_t, size_t, size_t, size_t, size_t, bool>(),
             nb::arg("arg_0"), nb::arg("arg_1"), nb::arg("arg_2"), nb::arg("arg_3"),
             nb::arg("ret_0"), nb::arg("ret_1"), nb::arg("flag_is_aligned"),
             "Constructs a StackPtxPTXArgs instance with the specified arguments, return values, and alignment flag")
        .def_prop_rw("arg_0",
            [](const StackPtxPTXArgs& self) { return self.arg_0; },
            [](StackPtxPTXArgs& self, size_t value) { self.arg_0 = value; },
            "First argument type")
        .def_prop_rw("arg_1",
            [](const StackPtxPTXArgs& self) { return self.arg_1; },
            [](StackPtxPTXArgs& self, size_t value) { self.arg_1 = value; },
            "Second argument type")
        .def_prop_rw("arg_2",
            [](const StackPtxPTXArgs& self) { return self.arg_2; },
            [](StackPtxPTXArgs& self, size_t value) { self.arg_2 = value; },
            "Third argument type")
        .def_prop_rw("arg_3",
            [](const StackPtxPTXArgs& self) { return self.arg_3; },
            [](StackPtxPTXArgs& self, size_t value) { self.arg_3 = value; },
            "Fourth argument type")
        .def_prop_rw("ret_0",
            [](const StackPtxPTXArgs& self) { return self.ret_0; },
            [](StackPtxPTXArgs& self, size_t value) { self.ret_0 = value; },
            "First return value type")
        .def_prop_rw("ret_1",
            [](const StackPtxPTXArgs& self) { return self.ret_1; },
            [](StackPtxPTXArgs& self, size_t value) { self.ret_1 = value; },
            "Second return value type")
        .def_prop_rw("flag_is_aligned",
            [](const StackPtxPTXArgs& self) { return self.flag_is_aligned; },
            [](StackPtxPTXArgs& self, bool value) { self.flag_is_aligned = value; },
            "Flag indicating if the arguments are aligned")
        .def_prop_rw("unused",
            [](const StackPtxPTXArgs& self) { return self.unused; },
            [](StackPtxPTXArgs& self, uint32_t value) { self.unused = value & 0x1; },
            "Unused bit field");
    
    // m.doc() = "Stack PTX module\n\n"
    //           "Type aliases (mapped to Python int):\n"
    //           "- StackPtxRegister: 32-bit unsigned integer for Stack PTX register\n"
    //           "- StackPtxPushConstant: 32-bit signed integer for Stack PTX push constant\n"
    //           "- StackPtxAstIdx: 32-bit signed integer for Stack PTX AST index\n"
    //           "- StackPtxIdx: 16-bit unsigned integer for Stack PTX index\n"
    //           "- StackPtxReturnIdx: 32-bit unsigned integer for Stack PTX return index";

    // Bind StackPtxPayload struct
    nb::class_<StackPtxPayload>(m, "StackPtxPayload", "Stack PTX payload union (4 bytes)")
        .def(nb::init<>(), "Default constructor (initializes to u = 0)")
        .def_static("from_u", [](uint32_t v) { StackPtxPayload p; p.u = v; return p; }, nb::arg("u"), "Create payload from unsigned 32-bit integer")
        .def_static("from_s", [](int32_t v) { StackPtxPayload p; p.s = v; return p; }, nb::arg("s"), "Create payload from signed 32-bit integer")
        .def_static("from_f", [](float v) { StackPtxPayload p; p.f = v; return p; }, nb::arg("f"), "Create payload from 32-bit float")
        .def_static("from_meta_constant", [](StackPtxMetaConstant v) { StackPtxPayload p; p.meta_constant = v; return p; }, nb::arg("meta_constant"), "Create payload from meta constant")
        .def_static("from_ptx_args", [](StackPtxPTXArgs v) { StackPtxPayload p; p.ptx_args = v; return p; }, nb::arg("ptx_args"), "Create payload from PTX args struct")
        .def_static("from_special_arg", [](StackPtxArgIdx v) { StackPtxPayload p; p.special_arg = v; return p; }, nb::arg("special_arg"), "Create payload from special arg (5-bit)")
        .def_static("from_reg", [](StackPtxRegisterCounter v) { StackPtxPayload p; p.reg = v; return p; }, nb::arg("reg"), "Create payload from register")
        .def_static("from_ast_idx", [](StackPtxAstIdx v) { StackPtxPayload p; p.ast_idx = v; return p; }, nb::arg("ast_idx"), "Create payload from AST index")
        .def_prop_rw("u",
            [](const StackPtxPayload& self) { return self.u; },
            [](StackPtxPayload& self, uint32_t value) { self.u = value; },
            "Unsigned 32-bit integer value of the payload")
        .def_prop_rw("s",
            [](const StackPtxPayload& self) { return self.s; },
            [](StackPtxPayload& self, int32_t value) { self.s = value; },
            "Signed 32-bit integer value of the payload")
        .def_prop_rw("f",
            [](const StackPtxPayload& self) { return self.f; },
            [](StackPtxPayload& self, float value) { self.f = value; },
            "32-bit floating-point value of the payload")
        .def_prop_rw("meta_constant",
            [](const StackPtxPayload& self) { return self.meta_constant; },
            [](StackPtxPayload& self, StackPtxMetaConstant value) { self.meta_constant = value; },
            "Meta constant value of the payload")
        .def_prop_rw("ptx_args",
            [](const StackPtxPayload& self) { return self.ptx_args; },
            [](StackPtxPayload& self, StackPtxPTXArgs value) { self.ptx_args = value; },
            "PTX arguments structure of the payload")
        .def_prop_rw("special_arg",
            [](const StackPtxPayload& self) { return self.special_arg; },
            [](StackPtxPayload& self, StackPtxArgIdx value) { self.special_arg = value; },
            "Special argument type (5-bit field)")
        .def_prop_rw("reg",
            [](const StackPtxPayload& self) { return self.reg; },
            [](StackPtxPayload& self, StackPtxRegisterCounter value) { self.reg = value; },
            "Register value of the payload")
        .def_prop_rw("ast_idx",
            [](const StackPtxPayload& self) { return self.ast_idx; },
            [](StackPtxPayload& self, StackPtxAstIdx value) { self.ast_idx = value; },
            "AST index value of the payload");

    nb::class_<StackPtxInstruction>(m, "StackPtxInstruction", "Stack PTX instruction structure (8 bytes)")
        .def(nb::init<>(), "Default constructor")
        .def(nb::init<StackPtxInstructionType, size_t, uint32_t, uint16_t, StackPtxPayload>(),
             nb::arg("instruction_type"), nb::arg("stack_idx"), nb::arg("ret_idx"), nb::arg("idx"), nb::arg("payload"),
             "Constructs a StackPtxInstruction with the specified fields")
        .def(nb::init<StackPtxInstructionType, size_t, uint32_t, StackPtxMetaInstruction, StackPtxPayload>(),
             nb::arg("instruction_type"), nb::arg("stack_idx"), nb::arg("ret_idx"), nb::arg("idx"), nb::arg("payload"),
             "Constructs a StackPtxInstruction with the specified fields")
        .def_prop_rw("instruction_type",
            [](const StackPtxInstruction& self) { return self.instruction_type; },
            [](StackPtxInstruction& self, StackPtxInstructionType value) { self.instruction_type = value; },
            "Instruction type (4-bit field)")
        .def_prop_rw("stack_idx",
            [](const StackPtxInstruction& self) { return self.stack_idx; },
            [](StackPtxInstruction& self, size_t value) { self.stack_idx = value; },
            "Stack type (5-bit field)")
        .def_prop_rw("ret_idx",
            [](const StackPtxInstruction& self) { return self.ret_idx; },
            [](StackPtxInstruction& self, uint32_t value) {
                if (value > 127) throw nb::value_error("ret_idx must be in range [0, 127]");
                self.ret_idx = value;
            },
            "Return index (7-bit field, 0-127)")
        .def_prop_rw("idx",
            [](const StackPtxInstruction& self) { return self.idx; },
            [](StackPtxInstruction& self, uint16_t value) { self.idx = value; },
            "Index (16-bit unsigned integer)")
        .def_prop_rw("payload",
            [](const StackPtxInstruction& self) { return self.payload; },
            [](StackPtxInstruction& self, StackPtxPayload value) { self.payload = value; },
            "Payload union (4 bytes)");

    
    m.def(
        "stack_ptx_result_to_string", 
        &stack_ptx_result_to_string, 
        nb::arg("StackPtxResult"),
        "Converts a StackPtxResult enum value to a human-readable string."
    );

    m.def("stack_ptx_compile_workspace_size", 
        [](
            size_t max_ast_size, 
            size_t max_ast_to_visit_stack_depth,
            size_t stack_size,
            size_t max_frame_depth,
            size_t store_size,
            size_t num_stacks,
            size_t num_arg_types
        ) {
            StackPtxCompilerInfo compiler_info = {
                max_ast_size, 
                max_ast_to_visit_stack_depth,
                stack_size,
                max_frame_depth,
                store_size,
            };
            StackPtxStackInfo stack_info = {
                NULL, 0,
                NULL, 0,
                NULL, num_stacks,
                NULL, num_arg_types
            };
            size_t workspace_in_bytes;
            StackPtxResult result = 
                stack_ptx_compile_workspace_size(
                    &compiler_info, 
                    &stack_info, 
                    &workspace_in_bytes
                );
            return nb::make_tuple(result, workspace_in_bytes);
        },
        nb::arg("max_ast_size"), 
        nb::arg("max_ast_to_visit_stack_depth"),
        nb::arg("stack_size"), 
        nb::arg("max_frame_depth"),
        nb::arg("store_size"),
        nb::arg("num_stacks"), 
        nb::arg("num_arg_types"),
        "Calculates the required workspace size in bytes for Stack PTX compilation.\n"
        "\n"
        "Args:\n"
        "    max_ast_size (int): Maximum size of the abstract syntax tree.\n"
        "    max_ast_to_visit_stack_depth (int): Maximum depth of the AST visit stack.\n"
        "\n"
        "Returns:\n"
        "    tuple: A tuple containing (StackPtxResult, workspace_in_bytes). The first element is the result status, "
        "and the second element is the required workspace size in bytes."
    );

    m.def(
        "stack_ptx_compile",
        []( 
            size_t max_ast_size, 
            size_t max_ast_to_visit_stack_depth,
            size_t stack_size,
            size_t max_frame_depth,
            size_t store_size,

            nb::object ptx_instruction_strings_obj,
            nb::object special_register_strings_obj,
            nb::list stack_literal_prefixes,
            nb::list arg_stack_indices,
            nb::list arg_stack_num_vector_elements,
            nb::list instructions_list,
            nb::list register_names_list,
            nb::list register_stack_types_list,
            nb::object routines_obj,
            nb::list requests_list,
            size_t    execution_limit,
            nb::object workspace_obj,     
            nb::object buffer_obj
        ) {
            StackPtxCompilerInfo compiler_info = {
                max_ast_size, 
                max_ast_to_visit_stack_depth,
                stack_size,
                max_frame_depth,
                store_size
            };

            // --- ptx_instruction_strings (nullable) ---
            const char* const* ptx_instruction_strings = nullptr;\
            size_t num_ptx_instruction_strings = 0;
            std::vector<const char*> ptx_instruction_strings_array;
            if (!ptx_instruction_strings_obj.is_none()) {
                nb::list lst = nb::cast<nb::list>(ptx_instruction_strings_obj);
                num_ptx_instruction_strings = lst.size();
                ptx_instruction_strings_array.resize(num_ptx_instruction_strings);
                for (size_t i = 0; i < num_ptx_instruction_strings; ++i) {
                    ptx_instruction_strings_array[i] = nb::cast<const char*>(lst[i]);
                }
                ptx_instruction_strings = num_ptx_instruction_strings > 0 ? ptx_instruction_strings_array.data() : nullptr;
            }

            // --- special_register_strings (nullable) ---
            const char* const* special_register_strings = nullptr;
            size_t num_special_register_strings = 0;
            std::vector<const char*> special_register_strings_array;
            if (!special_register_strings_obj.is_none()) {
                nb::list lst = nb::cast<nb::list>(special_register_strings_obj);
                num_special_register_strings = lst.size();
                special_register_strings_array.resize(num_special_register_strings);
                for (size_t i = 0; i < num_special_register_strings; ++i) {
                    special_register_strings_array[i] = nb::cast<const char*>(lst[i]);
                }
                special_register_strings = num_special_register_strings > 0 ? special_register_strings_array.data() : nullptr;
            }

            size_t num_stacks = stack_literal_prefixes.size();
            std::vector<const char*> stack_literal_prefixes_array(num_stacks);
            for (size_t i = 0; i < num_stacks; ++i) {
                stack_literal_prefixes_array[i] = nb::cast<const char*>(stack_literal_prefixes[i]);
            }
            const char* const* stack_literal_prefixes_ptr = num_stacks ? stack_literal_prefixes_array.data() : nullptr;

            size_t num_arg_types = arg_stack_indices.size();
            if (arg_stack_indices.size() != arg_stack_num_vector_elements.size()) {
                nb::raise_type_error("arg_stack_indices and arg_stack_num_elements must have the same length");
            }
            std::vector<StackPtxArgTypeInfo> arg_type_info_array(num_arg_types);
            for (size_t i = 0; i < num_arg_types; ++i) {
                size_t stack_idx = nb::cast<size_t>(arg_stack_indices[i]);
                size_t num_stack_elems = nb::cast<size_t>(arg_stack_num_vector_elements[i]);
                StackPtxArgTypeInfo arg_type_info = {
                    stack_idx,
                    num_stack_elems
                };
                arg_type_info_array[i] = arg_type_info;
            }
            const StackPtxArgTypeInfo* arg_type_info_ptr = num_arg_types ? arg_type_info_array.data() : nullptr;

            StackPtxStackInfo stack_info = {
                ptx_instruction_strings,
                num_ptx_instruction_strings,
                special_register_strings,
                num_special_register_strings,
                stack_literal_prefixes_ptr,
                num_stacks,
                arg_type_info_ptr,
                num_arg_types
            };

            size_t num_instructions = instructions_list.size();
            std::vector<StackPtxInstruction> instructions_array(num_instructions);
            for (size_t i = 0; i < num_instructions; ++i) {
                instructions_array[i] = nb::cast<StackPtxInstruction>(instructions_list[i]);
            }
            const StackPtxInstruction* instructions = num_instructions > 0 ? instructions_array.data() : nullptr;

            
            if (register_names_list.size() != register_stack_types_list.size()) {
                nb::raise_type_error("request_names and request_stack_types must have the same length");
            }

            size_t num_registers = register_names_list.size();
            std::vector<StackPtxRegister> registers_array(num_registers);
            for (size_t i = 0; i < num_registers; i++) {
                const char* register_name = nb::cast<const char*>(register_names_list[i]);
                size_t register_stack_type = nb::cast<size_t>(register_stack_types_list[i]);
                StackPtxRegister reg = { register_name, register_stack_type};
                registers_array[i] = reg;
            }
            const StackPtxRegister* registers = num_registers > 0 ? registers_array.data() : nullptr;

            // --- routines (nullable) ---
            const StackPtxInstruction** routines = nullptr;
            size_t num_routines = 0;
            std::vector<std::vector<StackPtxInstruction>> routines_storage;
            // array of pointers to those inner arrays
            std::vector<const StackPtxInstruction*> routines_ptrs;

            if (!routines_obj.is_none()) {
                nb::list outer = nb::cast<nb::list>(routines_obj);
                num_routines = outer.size();
                routines_storage.resize(num_routines);
                routines_ptrs.resize(num_routines);

                for (size_t i = 0; i < num_routines; ++i) {
                    nb::list inner = nb::cast<nb::list>(outer[i]);
                    size_t ni = inner.size();
                    auto &vec = routines_storage[i];
                    vec.resize(ni);
                    for (size_t j = 0; j < ni; ++j) {
                        vec[j] = nb::cast<StackPtxInstruction>(inner[j]); // per-element cast
                    }
                    routines_ptrs[i] = ni ? vec.data() : nullptr; // pointer to contiguous block
                }

                routines = num_routines ? routines_ptrs.data() : nullptr;
            }

            // --- request_names / request_stack_types (required) ---
            size_t num_requests = requests_list.size();
             std::vector<size_t> requests_array(num_requests);
            for (size_t i = 0; i < num_requests; ++i) {
                requests_array[i] = nb::cast<size_t>(requests_list[i]);;
            }
            const size_t* requests_ptr = num_requests ? requests_array.data() : nullptr;

            // --- workspace: measure/allocate style (bytearray or None) ---
            void*  workspace = nullptr;
            size_t workspace_in_bytes = 0;
            if (nb::isinstance<nb::bytearray>(workspace_obj)) {
                nb::bytearray ba = nb::cast<nb::bytearray>(workspace_obj);
                workspace = ba.data();
                workspace_in_bytes = ba.size();
            } else {
                nb::raise_type_error("workspace must be bytearray");
            }

            // --- output buffer: measure/allocate style (bytearray or None) ---
            char*  buffer = nullptr;
            size_t buffer_size = 0;
            if (nb::isinstance<nb::bytearray>(buffer_obj)) {
                nb::bytearray ba = nb::cast<nb::bytearray>(buffer_obj);
                buffer = reinterpret_cast<char*>(ba.data());
                buffer_size = ba.size();
            } else if (!buffer_obj.is_none()) {
                nb::raise_type_error("buffer must be bytearray or None");
            }

            size_t buffer_bytes_written = 0;

            StackPtxResult result = stack_ptx_compile(
                &compiler_info,
                &stack_info,
                instructions,
                registers,
                num_registers,
                routines,
                num_routines,
                requests_ptr,
                num_requests,
                execution_limit,
                workspace,
                workspace_in_bytes,
                buffer,
                buffer_size,
                &buffer_bytes_written
            );

            // StackPtxResult result = STACK_PTX_SUCCESS;

            return nb::make_tuple(
                nb::cast(result),
                nb::cast(buffer_bytes_written)
            );

        },
        nb::arg("max_ast_size"),
        nb::arg("max_ast_to_visit_stack_depth"),
        nb::arg("stack_size"),
        nb::arg("max_frame_depth"),
        nb::arg("store_size"),

        nb::arg("ptx_instruction_strings")    = nb::none(),   // list[str] or None
        nb::arg("special_register_strings")   = nb::none(),   // list[str] or None
        nb::arg("stack_literal_prefixes"),
        nb::arg("arg_stack_indices"),
        nb::arg("arg_stack_num_vector_elements"),
        nb::arg("instructions_list"),
        nb::arg("register_names_list"),
        nb::arg("register_stack_types_list"),
        nb::arg("routines")                   = nb::none(),   // list[StackPtxInstruction*] or None
        nb::arg("requests_list"),                              // list[str]
        nb::arg("execution_limit"),
        nb::arg("workspace"),                                   // bytearray or None
        nb::arg("buffer")                     = nb::none(),      // bytearray or None
        "Invokes stack_ptx_compile. Returns (result, buffer_bytes_written).\n"
        "Pass lists for pointer-to-pointer args; use bytearray/None for workspace and output buffer."
    );
}