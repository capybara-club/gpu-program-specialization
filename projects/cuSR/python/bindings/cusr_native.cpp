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
#define CUSR_AST_SASS_IMPLEMENTATION
#define CUSR_AST_SASS_PATCH_IMPLEMENTATION

#include "cusr_ast_sass_patch.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <deque>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;

using CusrPackedPrograms = nb::ndarray<nb::numpy, const uint64_t, nb::c_contig, nb::device::cpu>;

static_assert(sizeof(CusrAstInstruction) == sizeof(uint64_t), "CusrAstInstruction must remain 8 bytes");
static_assert(alignof(uint64_t) >= alignof(CusrAstInstruction), "packed AST alignment is insufficient");

static uint64_t
cusr_python_pack_instruction(CusrAstInstruction instruction)
{
    uint64_t word = 0u;
    std::memcpy(&word, &instruction, sizeof(word));
    return word;
}

static nb::tuple
cusr_python_decode_instruction(uint64_t word)
{
    CusrAstInstruction instruction = {};
    std::memcpy(&instruction, &word, sizeof(instruction));
    return nb::make_tuple(instruction.instruction_type, instruction.aux, instruction.payload.bits);
}

static size_t
cusr_python_size_attr(nb::handle object, const char* name)
{
    return nb::cast<size_t>(object.attr(name));
}

static uint32_t
cusr_python_u32_attr(nb::handle object, const char* name)
{
    return nb::cast<uint32_t>(object.attr(name));
}

static uint64_t
cusr_python_u64_attr(nb::handle object, const char* name)
{
    return nb::cast<uint64_t>(object.attr(name));
}

static nb::sequence
cusr_python_sequence_attr(nb::handle object, const char* name)
{
    nb::object value = nb::borrow<nb::object>(object.attr(name));
    if (!nb::isinstance<nb::sequence>(value)) {
        throw nb::type_error("inspection fields containing records must be sequences");
    }
    return nb::borrow<nb::sequence>(value);
}

static void
cusr_python_copy_name(char* destination, size_t capacity, const std::string& source)
{
    if (source.empty() || source.size() >= capacity) {
        throw nb::value_error("inspection contains an invalid kernel name");
    }
    std::memset(destination, 0, capacity);
    std::memcpy(destination, source.data(), source.size());
}

template <size_t Capacity>
static size_t
cusr_python_copy_registers(uint8_t (&destination)[Capacity], nb::handle object, const char* name)
{
    nb::sequence values = cusr_python_sequence_attr(object, name);
    const size_t count = nb::len(values);
    if (count > Capacity) {
        throw nb::value_error("inspection register list exceeds its native capacity");
    }

    std::fill(destination, destination + Capacity, uint8_t{0u});
    for (size_t i = 0u; i < count; ++i) {
        destination[i] = nb::cast<uint8_t>(values[i]);
    }
    return count;
}

class CusrPythonSassPatchLayout {
public:
    explicit CusrPythonSassPatchLayout(nb::handle inspection)
    {
        build(inspection);
    }

    CusrPythonSassPatchLayout(const CusrPythonSassPatchLayout&) = delete;
    CusrPythonSassPatchLayout& operator=(const CusrPythonSassPatchLayout&) = delete;

    size_t cubin_size() const { return handle_.cubin_size; }
    uint32_t sass_arch() const { return handle_.sass_arch; }
    size_t num_kernels() const { return handle_.num_kernels; }
    size_t ast_capacity() const { return handle_.ast_capacity; }
    size_t num_sites() const { return handle_.num_sites; }

    CusrAstSassPatchStats patch(
        const CusrPackedPrograms& programs,
        nb::bytearray cubin,
        CusrAstSassPatchEpilogue epilogue,
        nb::object routines)
    {
        CusrAstSassPatchStats stats = {};
        prepare_program_pointers(programs, program_ptrs_, "programs");

        if (routines.is_none()) {
            routine_ptrs_.clear();
        } else {
            CusrPackedPrograms packed_routines = nb::cast<CusrPackedPrograms>(routines);
            prepare_program_pointers(packed_routines, routine_ptrs_, "routines");
        }

        if (cubin.size() != handle_.cubin_size) {
            throw nb::value_error("cubin size does not match the inspection layout");
        }

        const uint32_t capability_major = handle_.sass_arch / 10u;
        const uint32_t capability_minor = handle_.sass_arch % 10u;
        CusrAstSassPatchResult result = cusr_ast_sass_patch_cubin(
            &handle_,
            capability_major,
            capability_minor,
            epilogue,
            routine_ptrs_.empty() ? nullptr : routine_ptrs_.data(),
            routine_ptrs_.size(),
            program_ptrs_.data(),
            program_ptrs_.size(),
            static_cast<unsigned char*>(cubin.data()),
            cubin.size(),
            &stats
        );
        if (result != CUSR_AST_SASS_PATCH_SUCCESS) {
            throw std::runtime_error(cusr_ast_sass_patch_result_to_string(result));
        }
        return stats;
    }

    uint32_t max_register_count(const nb::bytearray& cubin) const
    {
        uint32_t value = 0u;
        if (cubin.size() != handle_.cubin_size) {
            throw nb::value_error("cubin size does not match the inspection layout");
        }

        CusrAstSassPatchResult result = cusr_ast_sass_patch_cubin_max_register_count(
            &handle_,
            static_cast<const unsigned char*>(cubin.data()),
            cubin.size(),
            &value
        );
        if (result != CUSR_AST_SASS_PATCH_SUCCESS) {
            throw std::runtime_error(cusr_ast_sass_patch_result_to_string(result));
        }
        return value;
    }

private:
    static void prepare_program_pointers(
        const CusrPackedPrograms& programs,
        std::vector<const CusrAstInstruction*>& pointers,
        const char* label)
    {
        if (programs.ndim() != 2u ||
            programs.shape(0) == 0u ||
            programs.shape(1) == 0u ||
            programs.shape(1) > CUSR_AST_MAX_PROGRAM_INSTRUCTIONS) {
            throw nb::value_error((std::string(label) + " must have shape (count, stride), with stride in [1, 1024]").c_str());
        }

        const CusrAstInstruction* base = reinterpret_cast<const CusrAstInstruction*>(programs.data());
        const size_t count = programs.shape(0);
        const size_t stride = programs.shape(1);
        pointers.resize(count);

        for (size_t program_idx = 0u; program_idx < count; ++program_idx) {
            const CusrAstInstruction* program = base + program_idx * stride;
            bool found_return = false;
            pointers[program_idx] = program;

            for (size_t instruction_idx = 0u; instruction_idx < stride; ++instruction_idx) {
                if (program[instruction_idx].instruction_type == CUSR_AST_INSTRUCTION_TYPE_RETURN) {
                    found_return = true;
                    break;
                }
            }
            if (!found_return) {
                throw nb::value_error((std::string(label) + " contains a row without a return instruction").c_str());
            }
        }
    }

    void build(nb::handle inspection)
    {
        nb::sequence python_kernels = cusr_python_sequence_attr(inspection, "kernels");
        const size_t num_kernels = nb::len(python_kernels);
        const size_t declared_kernels = cusr_python_size_attr(inspection, "num_kernels");

        if (num_kernels == 0u || num_kernels != declared_kernels) {
            throw nb::value_error("inspection kernel count is inconsistent");
        }

        kernels_.reserve(num_kernels);
        for (size_t kernel_idx = 0u; kernel_idx < num_kernels; ++kernel_idx) {
            nb::object python_kernel = nb::borrow<nb::object>(python_kernels[kernel_idx]);
            CusrSassInspectKernel kernel = {};
            nb::sequence python_regcounts = cusr_python_sequence_attr(python_kernel, "regcount_records");
            nb::sequence python_sites = cusr_python_sequence_attr(python_kernel, "sites");

            kernel.kernel_index = cusr_python_size_attr(python_kernel, "kernel_index");
            kernel.symbol_index = cusr_python_size_attr(python_kernel, "symbol_index");
            kernel.start_address = cusr_python_u64_attr(python_kernel, "start_address");
            kernel.end_address = cusr_python_u64_attr(python_kernel, "end_address");
            kernel.start_file_offset = cusr_python_size_attr(python_kernel, "start_file_offset");
            kernel.end_file_offset = cusr_python_size_attr(python_kernel, "end_file_offset");
            kernel.first_instruction = cusr_python_size_attr(python_kernel, "first_instruction");
            kernel.num_instructions = cusr_python_size_attr(python_kernel, "num_instructions");
            kernel.first_regcount_record = regcount_records_.size();
            kernel.num_regcount_records = nb::len(python_regcounts);
            kernel.first_site = sites_.size();
            kernel.num_sites = nb::len(python_sites);
            text_section_names_.push_back(nb::cast<std::string>(python_kernel.attr("text_section_name")));
            kernel.text_section_name = text_section_names_.back().c_str();
            cusr_python_copy_name(
                kernel.name,
                sizeof(kernel.name),
                nb::cast<std::string>(python_kernel.attr("name"))
            );

            if (kernel.kernel_index != kernel_idx ||
                kernel.first_regcount_record != cusr_python_size_attr(python_kernel, "first_regcount_record") ||
                kernel.num_regcount_records != cusr_python_size_attr(python_kernel, "num_regcount_records") ||
                kernel.first_site != cusr_python_size_attr(python_kernel, "first_site") ||
                kernel.num_sites != cusr_python_size_attr(python_kernel, "num_sites")) {
                throw nb::value_error("inspection kernel indexing is inconsistent");
            }

            for (size_t record_idx = 0u; record_idx < kernel.num_regcount_records; ++record_idx) {
                nb::object python_record = nb::borrow<nb::object>(python_regcounts[record_idx]);
                CusrSassInspectRegcountRecord record = {};
                record.kernel_index = cusr_python_size_attr(python_record, "kernel_index");
                record.symbol_index = cusr_python_size_attr(python_record, "symbol_index");
                record.tag_file_offset = cusr_python_size_attr(python_record, "tag_file_offset");
                record.value_file_offset = cusr_python_size_attr(python_record, "value_file_offset");
                record.value = cusr_python_u32_attr(python_record, "value");
                regcount_section_names_.push_back(nb::cast<std::string>(python_record.attr("section_name")));
                record.section_name = regcount_section_names_.back().c_str();
                if (record.kernel_index != kernel_idx || record.symbol_index != kernel.symbol_index) {
                    throw nb::value_error("inspection register-count indexing is inconsistent");
                }
                regcount_records_.push_back(record);
            }

            for (size_t site_idx = 0u; site_idx < kernel.num_sites; ++site_idx) {
                nb::object python_site = nb::borrow<nb::object>(python_sites[site_idx]);
                CusrSassInspectSite site = {};
                site.kernel_index = cusr_python_size_attr(python_site, "kernel_index");
                site.occurrence_index = cusr_python_size_attr(python_site, "occurrence_index");
                site.marker = cusr_python_u32_attr(python_site, "marker");
                site.start_instruction = cusr_python_size_attr(python_site, "start_instruction");
                site.end_instruction = cusr_python_size_attr(python_site, "end_instruction");
                site.start_file_offset = cusr_python_size_attr(python_site, "start_file_offset");
                site.end_file_offset = cusr_python_size_attr(python_site, "end_file_offset");
                site.start_address = cusr_python_u64_attr(python_site, "start_address");
                site.end_address = cusr_python_u64_attr(python_site, "end_address");
                site.target_reg = nb::cast<uint8_t>(python_site.attr("target_reg"));
                site.incoming_wait_mask = cusr_python_u32_attr(python_site, "incoming_wait_mask");
                const size_t num_inputs = cusr_python_copy_registers(site.input_regs, python_site, "input_regs");
                site.num_output_regs = cusr_python_copy_registers(site.output_regs, python_site, "output_regs");
                site.num_available_regs = cusr_python_copy_registers(site.available_regs, python_site, "available_regs");

                if (site.kernel_index != kernel_idx || num_inputs != CUSR_SASS_INSPECT_INPUTS) {
                    throw nb::value_error("inspection site indexing or input count is inconsistent");
                }
                sites_.push_back(site);
            }

            kernels_.push_back(kernel);
        }

        handle_ = {};
        handle_.cubin_size = cusr_python_size_attr(inspection, "cubin_size");
        handle_.sass_arch = cusr_python_u32_attr(inspection, "sass_arch");
        handle_.num_kernels = num_kernels;
        handle_.ast_capacity = cusr_python_size_attr(inspection, "ast_capacity");
        handle_.first_marker_bits = cusr_python_u32_attr(inspection, "first_marker_bits");
        handle_.expected_occurrences = cusr_python_size_attr(inspection, "expected_occurrences");
        handle_.num_sites = sites_.size();
        handle_.num_regcount_records = regcount_records_.size();

        if (handle_.num_sites != cusr_python_size_attr(inspection, "num_sites") ||
            handle_.num_regcount_records != cusr_python_size_attr(inspection, "num_regcount_records") ||
            handle_.ast_capacity == 0u ||
            handle_.ast_capacity > CUSR_SASS_INSPECT_MAX_OUTPUTS) {
            throw nb::value_error("inspection aggregate counts are inconsistent");
        }

        handle_.kernels = kernels_.data();
        handle_.sites = sites_.data();
        handle_.regcount_records = regcount_records_.data();
    }

    CusrSassInspectHandle handle_ = {};
    std::vector<CusrSassInspectKernel> kernels_;
    std::vector<CusrSassInspectSite> sites_;
    std::vector<CusrSassInspectRegcountRecord> regcount_records_;
    std::deque<std::string> text_section_names_;
    std::deque<std::string> regcount_section_names_;
    std::vector<const CusrAstInstruction*> program_ptrs_;
    std::vector<const CusrAstInstruction*> routine_ptrs_;
};

NB_MODULE(_cusr_native, module)
{
    module.doc() = "Native cuSR AST encoding and in-place NVIDIA cubin patching";
    module.attr("INSTRUCTION_SIZE") = sizeof(CusrAstInstruction);
    module.attr("MAX_PROGRAM_INSTRUCTIONS") = CUSR_AST_MAX_PROGRAM_INSTRUCTIONS;

    nb::enum_<CusrAstInstructionType>(module, "InstructionType")
        .value("NONE", CUSR_AST_INSTRUCTION_TYPE_NONE)
        .value("INPUT", CUSR_AST_INSTRUCTION_TYPE_INPUT)
        .value("CONSTANT_BITS", CUSR_AST_INSTRUCTION_TYPE_CONSTANT_BITS)
        .value("OP", CUSR_AST_INSTRUCTION_TYPE_OP)
        .value("ROUTINE", CUSR_AST_INSTRUCTION_TYPE_ROUTINE)
        .value("RETURN", CUSR_AST_INSTRUCTION_TYPE_RETURN)
        .value("ROUTINE_ARG", CUSR_AST_INSTRUCTION_TYPE_ROUTINE_ARG);

    nb::enum_<CusrAstOp>(module, "Op")
        .value("NONE", CUSR_AST_OP_NONE)
        .value("ADD", CUSR_AST_OP_ADD)
        .value("SUB", CUSR_AST_OP_SUB)
        .value("MUL", CUSR_AST_OP_MUL)
        .value("DIV", CUSR_AST_OP_DIV)
        .value("NEG", CUSR_AST_OP_NEG)
        .value("SQRT", CUSR_AST_OP_SQRT)
        .value("RCP", CUSR_AST_OP_RCP)
        .value("ABS", CUSR_AST_OP_ABS)
        .value("MIN", CUSR_AST_OP_MIN)
        .value("MAX", CUSR_AST_OP_MAX)
        .value("FMA", CUSR_AST_OP_FMA)
        .value("SIN", CUSR_AST_OP_SIN)
        .value("COS", CUSR_AST_OP_COS)
        .value("EX2", CUSR_AST_OP_EX2)
        .value("LG2", CUSR_AST_OP_LG2)
        .value("RSQRT", CUSR_AST_OP_RSQRT)
        .value("TANH", CUSR_AST_OP_TANH);

    module.def("encode_input", [](uint32_t input_idx) { return cusr_python_pack_instruction(cusr_ast_encode_input(input_idx)); });
    module.def("encode_constant_bits", [](uint32_t bits) { return cusr_python_pack_instruction(cusr_ast_encode_constant_bits(bits)); });
    module.def("encode_constant", [](float value) { return cusr_python_pack_instruction(cusr_ast_encode_constant(value)); });
    module.def("encode_op", [](CusrAstOp op, uint16_t num_args) { return cusr_python_pack_instruction(cusr_ast_encode_op(op, num_args)); });
    module.def("encode_routine", [](uint32_t routine_idx, uint16_t num_args) { return cusr_python_pack_instruction(cusr_ast_encode_routine(routine_idx, num_args)); });
    module.def("encode_routine_arg", [](uint32_t arg_idx) { return cusr_python_pack_instruction(cusr_ast_encode_routine_arg(arg_idx)); });
    module.def("encode_return", []() { return cusr_python_pack_instruction(cusr_ast_encode_return); });
    module.def("decode_instruction", &cusr_python_decode_instruction);

    nb::enum_<CusrAstSassPatchEpilogue>(module, "PatchEpilogue")
        .value("SSE", CUSR_AST_SASS_PATCH_EPILOGUE_SSE)
        .value("VALUE", CUSR_AST_SASS_PATCH_EPILOGUE_VALUE);

    nb::class_<CusrAstSassPatchStats>(module, "PatchStats")
        .def_ro("sites_patched", &CusrAstSassPatchStats::sites_patched)
        .def_ro("asts_patched", &CusrAstSassPatchStats::asts_patched)
        .def_ro("sass_instructions_written", &CusrAstSassPatchStats::sass_instructions_written)
        .def_ro("sass_bytes_written", &CusrAstSassPatchStats::sass_bytes_written)
        .def_ro("max_original_register_count", &CusrAstSassPatchStats::max_original_register_count)
        .def_ro("max_expanded_register_count", &CusrAstSassPatchStats::max_expanded_register_count)
        .def_ro("max_patched_register_count", &CusrAstSassPatchStats::max_patched_register_count);

    nb::class_<CusrPythonSassPatchLayout>(module, "SassPatchLayout")
        .def(nb::init<nb::handle>(), nb::arg("inspection"))
        .def_prop_ro("cubin_size", &CusrPythonSassPatchLayout::cubin_size)
        .def_prop_ro("sass_arch", &CusrPythonSassPatchLayout::sass_arch)
        .def_prop_ro("num_kernels", &CusrPythonSassPatchLayout::num_kernels)
        .def_prop_ro("ast_capacity", &CusrPythonSassPatchLayout::ast_capacity)
        .def_prop_ro("num_sites", &CusrPythonSassPatchLayout::num_sites)
        .def(
            "patch",
            &CusrPythonSassPatchLayout::patch,
            nb::arg("programs"),
            nb::arg("cubin"),
            nb::arg("epilogue") = CUSR_AST_SASS_PATCH_EPILOGUE_SSE,
            nb::arg("routines") = nb::none(),
            "Patch packed postfix ASTs into a writable cubin bytearray in place."
        )
        .def("max_register_count", &CusrPythonSassPatchLayout::max_register_count, nb::arg("cubin"));
}
