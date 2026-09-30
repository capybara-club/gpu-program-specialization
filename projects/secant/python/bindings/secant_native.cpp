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
#include "secant.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;

enum class SecantPythonCubinShape {
    Materialize,
    SSE,
    AffineStats,
    GramStats,
    DynamicConstantSSE,
    DynamicLeafSSE
};

static void
secant_python_check(SecantResult result, const char* operation) {
    if (result != SECANT_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed: " + secant_result_to_string(result));
    }
}

static std::string
secant_python_source_generate(const SecantCubinRecipeHeader* recipe) {
    size_t required_size = 0u;

    secant_python_check(
        secant_cubin_source_size(recipe, &required_size),
        "secant_cubin_source_size");
    std::string source(required_size, '\0');
    secant_python_check(
        secant_cubin_source_write(recipe, source.data(), source.size()),
        "secant_cubin_source_write");
    source.resize(required_size - 1u);
    return source;
}

static std::string
secant_python_generate_materialize_source(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    size_t patch_capacity_instructions
) {
    SecantCubinMaterializeRecipe recipe = secant_cubin_materialize_recipe_init();

    recipe.num_kernels = num_kernels;
    recipe.asts_per_kernel = asts_per_kernel;
    recipe.num_inputs = num_inputs;
    recipe.patch_capacity_instructions = patch_capacity_instructions;
    return secant_python_source_generate(&recipe.header);
}

static std::string
secant_python_generate_sse_source(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    size_t patch_capacity_instructions
) {
    SecantCubinSSERecipe recipe = secant_cubin_sse_recipe_init();

    recipe.num_kernels = num_kernels;
    recipe.asts_per_kernel = asts_per_kernel;
    recipe.num_inputs = num_inputs;
    recipe.num_targets = num_targets;
    recipe.tile_rows = tile_rows;
    recipe.threads_per_block = threads_per_block;
    recipe.patch_capacity_instructions = patch_capacity_instructions;
    return secant_python_source_generate(&recipe.header);
}

static std::string
secant_python_generate_affine_stats_source(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    size_t patch_capacity_instructions
) {
    SecantCubinAffineStatsRecipe recipe = secant_cubin_affine_stats_recipe_init();

    recipe.num_kernels = num_kernels;
    recipe.asts_per_kernel = asts_per_kernel;
    recipe.num_inputs = num_inputs;
    recipe.num_targets = num_targets;
    recipe.tile_rows = tile_rows;
    recipe.threads_per_block = threads_per_block;
    recipe.patch_capacity_instructions = patch_capacity_instructions;
    return secant_python_source_generate(&recipe.header);
}

static std::string
secant_python_generate_gram_stats_source(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    size_t patch_capacity_instructions
) {
    SecantCubinGramStatsRecipe recipe = secant_cubin_gram_stats_recipe_init();

    recipe.num_kernels = num_kernels;
    recipe.asts_per_kernel = asts_per_kernel;
    recipe.num_inputs = num_inputs;
    recipe.num_targets = num_targets;
    recipe.tile_rows = tile_rows;
    recipe.threads_per_block = threads_per_block;
    recipe.patch_capacity_instructions = patch_capacity_instructions;
    return secant_python_source_generate(&recipe.header);
}

static std::string
secant_python_generate_dynamic_constant_sse_source(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_input_columns,
    size_t num_input_constants,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    size_t patch_capacity_instructions
) {
    SecantCubinDynamicConstantSSERecipe recipe = secant_cubin_dynamic_constant_sse_recipe_init();

    recipe.num_kernels = num_kernels;
    recipe.asts_per_kernel = asts_per_kernel;
    recipe.num_input_columns = num_input_columns;
    recipe.num_input_constants = num_input_constants;
    recipe.num_targets = num_targets;
    recipe.tile_rows = tile_rows;
    recipe.threads_per_block = threads_per_block;
    recipe.patch_capacity_instructions = patch_capacity_instructions;
    return secant_python_source_generate(&recipe.header);
}

static std::string
secant_python_generate_dynamic_leaf_sse_source(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_input_columns,
    size_t num_static_input_columns,
    size_t num_dynamic_leaves,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    size_t patch_capacity_instructions
) {
    SecantCubinDynamicLeafSSERecipe recipe = secant_cubin_dynamic_leaf_sse_recipe_init();

    recipe.num_kernels = num_kernels;
    recipe.asts_per_kernel = asts_per_kernel;
    recipe.num_input_columns = num_input_columns;
    recipe.num_static_input_columns = num_static_input_columns;
    recipe.num_dynamic_leaves = num_dynamic_leaves;
    recipe.num_targets = num_targets;
    recipe.tile_rows = tile_rows;
    recipe.threads_per_block = threads_per_block;
    recipe.patch_capacity_instructions = patch_capacity_instructions;
    return secant_python_source_generate(&recipe.header);
}

static void
secant_python_validate_program_data(
    const SecantAstInstruction* data,
    size_t size,
    const char* label
) {
    size_t offset = 0u;
    size_t instruction_count;

    if (data == nullptr || size == 0u) {
        throw nb::value_error((std::string(label) + " contains an empty program").c_str());
    }
    for (instruction_count = 0u; instruction_count < SECANT_AST_MAX_PROGRAM_INSTRUCTIONS; ++instruction_count) {
        size_t instruction_size;
        SecantAstInstructionType instruction_type;

        if (offset >= size) {
            throw nb::value_error((std::string(label) + " is not return terminated").c_str());
        }
        instruction_type = secant_ast_instruction_type_get(data + offset);
        instruction_size = secant_ast_instruction_size_get(data + offset);
        if (instruction_size == 0u || instruction_size > size - offset) {
            throw nb::value_error((std::string(label) + " contains a truncated or invalid instruction").c_str());
        }
        if (instruction_type == SECANT_AST_INSTRUCTION_TYPE_RETURN_F32) {
            if (offset + instruction_size != size) {
                throw nb::value_error((std::string(label) + " contains bytes after return").c_str());
            }
            return;
        }
        offset += instruction_size;
    }
    throw nb::value_error((std::string(label) + " exceeds the instruction limit").c_str());
}

static void
secant_python_validate_program(nb::handle object, const char* label) {
    if (!nb::isinstance<nb::bytes>(object)) {
        throw nb::type_error((std::string(label) + " must contain bytes objects").c_str());
    }
    const nb::bytes program = nb::borrow<nb::bytes>(object);

    secant_python_validate_program_data(
        reinterpret_cast<const SecantAstInstruction*>(program.data()),
        program.size(),
        label);
}

static std::vector<const SecantAstInstruction*>
secant_python_program_pointers(nb::sequence programs, const char* label) {
    std::vector<const SecantAstInstruction*> pointers;
    const size_t count = nb::len(programs);

    pointers.reserve(count);
    for (size_t idx = 0u; idx < count; ++idx) {
        nb::handle program = programs[idx];

        secant_python_validate_program(program, label);
        pointers.push_back(reinterpret_cast<const SecantAstInstruction*>(nb::cast<nb::bytes>(program).data()));
    }
    return pointers;
}

using SecantPythonByteArray = nb::ndarray<nb::numpy, const uint8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using SecantPythonOffsetArray = nb::ndarray<nb::numpy, const uint64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

static std::vector<const SecantAstInstruction*>
secant_python_packed_program_pointers(
    const SecantPythonByteArray& data,
    const SecantPythonOffsetArray& offsets
) {
    std::vector<const SecantAstInstruction*> pointers;
    const size_t data_size = data.shape(0);
    const size_t num_offsets = offsets.shape(0);
    const auto* bytes = data.data();
    const auto* starts = offsets.data();
    size_t program_idx;

    if (num_offsets < 2u || starts[0] != 0u || starts[num_offsets - 1u] != data_size) {
        throw nb::value_error("AST offsets must begin at zero and end at the packed byte size");
    }
    pointers.reserve(num_offsets - 1u);
    for (program_idx = 0u; program_idx + 1u < num_offsets; ++program_idx) {
        const uint64_t begin_u64 = starts[program_idx];
        const uint64_t end_u64 = starts[program_idx + 1u];
        size_t begin;
        size_t end;

        if (begin_u64 > SIZE_MAX || end_u64 > SIZE_MAX) {
            throw nb::value_error("AST offset exceeds the native address range");
        }
        begin = static_cast<size_t>(begin_u64);
        end = static_cast<size_t>(end_u64);
        if (begin > end || end > data_size) {
            throw nb::value_error("AST offsets must be monotonic and within the packed byte array");
        }
        secant_python_validate_program_data(
            reinterpret_cast<const SecantAstInstruction*>(bytes + begin),
            end - begin,
            "packed ASTs");
        pointers.push_back(reinterpret_cast<const SecantAstInstruction*>(bytes + begin));
    }
    return pointers;
}

class SecantPythonCubinPlan {
public:
    SecantPythonCubinPlan(const SecantPythonCubinPlan&) = delete;
    SecantPythonCubinPlan& operator=(const SecantPythonCubinPlan&) = delete;

    static std::unique_ptr<SecantPythonCubinPlan>
    inspect_materialize(
        const nb::bytes& cubin,
        size_t num_kernels,
        size_t asts_per_kernel,
        size_t num_inputs,
        size_t patch_capacity_instructions
    ) {
        SecantCubinMaterializeRecipe recipe = secant_cubin_materialize_recipe_init();
        auto result = std::unique_ptr<SecantPythonCubinPlan>(
            new SecantPythonCubinPlan(SecantPythonCubinShape::Materialize, cubin));

        recipe.num_kernels = num_kernels;
        recipe.asts_per_kernel = asts_per_kernel;
        recipe.num_inputs = num_inputs;
        recipe.patch_capacity_instructions = patch_capacity_instructions;
        result->inspect(&recipe.header);
        result->read_recipe();
        return result;
    }

    static std::unique_ptr<SecantPythonCubinPlan>
    inspect_sse(
        const nb::bytes& cubin,
        size_t num_kernels,
        size_t asts_per_kernel,
        size_t num_inputs,
        size_t num_targets,
        size_t tile_rows,
        size_t threads_per_block,
        size_t patch_capacity_instructions
    ) {
        SecantCubinSSERecipe recipe = secant_cubin_sse_recipe_init();
        auto result = std::unique_ptr<SecantPythonCubinPlan>(
            new SecantPythonCubinPlan(SecantPythonCubinShape::SSE, cubin));

        recipe.num_kernels = num_kernels;
        recipe.asts_per_kernel = asts_per_kernel;
        recipe.num_inputs = num_inputs;
        recipe.num_targets = num_targets;
        recipe.tile_rows = tile_rows;
        recipe.threads_per_block = threads_per_block;
        recipe.patch_capacity_instructions = patch_capacity_instructions;
        result->inspect(&recipe.header);
        result->read_recipe();
        return result;
    }

    static std::unique_ptr<SecantPythonCubinPlan>
    inspect_affine_stats(
        const nb::bytes& cubin,
        size_t num_kernels,
        size_t asts_per_kernel,
        size_t num_inputs,
        size_t num_targets,
        size_t tile_rows,
        size_t threads_per_block,
        size_t patch_capacity_instructions
    ) {
        SecantCubinAffineStatsRecipe recipe = secant_cubin_affine_stats_recipe_init();
        auto result = std::unique_ptr<SecantPythonCubinPlan>(
            new SecantPythonCubinPlan(SecantPythonCubinShape::AffineStats, cubin));

        recipe.num_kernels = num_kernels;
        recipe.asts_per_kernel = asts_per_kernel;
        recipe.num_inputs = num_inputs;
        recipe.num_targets = num_targets;
        recipe.tile_rows = tile_rows;
        recipe.threads_per_block = threads_per_block;
        recipe.patch_capacity_instructions = patch_capacity_instructions;
        result->inspect(&recipe.header);
        result->read_recipe();
        return result;
    }

    static std::unique_ptr<SecantPythonCubinPlan>
    inspect_gram_stats(
        const nb::bytes& cubin,
        size_t num_kernels,
        size_t asts_per_kernel,
        size_t num_inputs,
        size_t num_targets,
        size_t tile_rows,
        size_t threads_per_block,
        size_t patch_capacity_instructions
    ) {
        SecantCubinGramStatsRecipe recipe = secant_cubin_gram_stats_recipe_init();
        auto result = std::unique_ptr<SecantPythonCubinPlan>(
            new SecantPythonCubinPlan(SecantPythonCubinShape::GramStats, cubin));

        recipe.num_kernels = num_kernels;
        recipe.asts_per_kernel = asts_per_kernel;
        recipe.num_inputs = num_inputs;
        recipe.num_targets = num_targets;
        recipe.tile_rows = tile_rows;
        recipe.threads_per_block = threads_per_block;
        recipe.patch_capacity_instructions = patch_capacity_instructions;
        result->inspect(&recipe.header);
        result->read_recipe();
        return result;
    }

    static std::unique_ptr<SecantPythonCubinPlan>
    inspect_dynamic_constant_sse(
        const nb::bytes& cubin,
        size_t num_kernels,
        size_t asts_per_kernel,
        size_t num_input_columns,
        size_t num_input_constants,
        size_t num_targets,
        size_t tile_rows,
        size_t threads_per_block,
        size_t patch_capacity_instructions
    ) {
        SecantCubinDynamicConstantSSERecipe recipe = secant_cubin_dynamic_constant_sse_recipe_init();
        auto result = std::unique_ptr<SecantPythonCubinPlan>(
            new SecantPythonCubinPlan(SecantPythonCubinShape::DynamicConstantSSE, cubin));

        recipe.num_kernels = num_kernels;
        recipe.asts_per_kernel = asts_per_kernel;
        recipe.num_input_columns = num_input_columns;
        recipe.num_input_constants = num_input_constants;
        recipe.num_targets = num_targets;
        recipe.tile_rows = tile_rows;
        recipe.threads_per_block = threads_per_block;
        recipe.patch_capacity_instructions = patch_capacity_instructions;
        result->inspect(&recipe.header);
        result->read_recipe();
        return result;
    }

    static std::unique_ptr<SecantPythonCubinPlan>
    inspect_dynamic_leaf_sse(
        const nb::bytes& cubin,
        size_t num_kernels,
        size_t asts_per_kernel,
        size_t num_input_columns,
        size_t num_static_input_columns,
        size_t num_dynamic_leaves,
        size_t num_targets,
        size_t tile_rows,
        size_t threads_per_block,
        size_t patch_capacity_instructions
    ) {
        SecantCubinDynamicLeafSSERecipe recipe = secant_cubin_dynamic_leaf_sse_recipe_init();
        auto result = std::unique_ptr<SecantPythonCubinPlan>(
            new SecantPythonCubinPlan(SecantPythonCubinShape::DynamicLeafSSE, cubin));

        recipe.num_kernels = num_kernels;
        recipe.asts_per_kernel = asts_per_kernel;
        recipe.num_input_columns = num_input_columns;
        recipe.num_static_input_columns = num_static_input_columns;
        recipe.num_dynamic_leaves = num_dynamic_leaves;
        recipe.num_targets = num_targets;
        recipe.tile_rows = tile_rows;
        recipe.threads_per_block = threads_per_block;
        recipe.patch_capacity_instructions = patch_capacity_instructions;
        result->inspect(&recipe.header);
        result->read_recipe();
        return result;
    }

    void specialize_into(nb::sequence routines, nb::sequence asts, nb::bytearray cubin) const {
        if (cubin.size() != cubin_size_) {
            throw nb::value_error("cubin size does not match the inspected template");
        }
        if ((reinterpret_cast<uintptr_t>(cubin.data()) & (alignof(uint64_t) - 1u)) != 0u) {
            throw nb::value_error("cubin bytearray storage is not eight-byte aligned");
        }
        const auto routine_pointers = secant_python_program_pointers(routines, "routines");
        const auto ast_pointers = secant_python_program_pointers(asts, "asts");
        const SecantAstProgramSet programs = {
            {routine_pointers.empty() ? nullptr : routine_pointers.data(), routine_pointers.size()},
            {ast_pointers.empty() ? nullptr : ast_pointers.data(), ast_pointers.size()},
            {},
            {}
        };
        SecantResult result;
        {
            nb::gil_scoped_release release;
            result = secant_cubin_specialize_into(
                plan_,
                &programs,
                cubin.data(),
                cubin.size());
        }
        secant_python_check(result, "secant_cubin_specialize_into");
    }

    size_t cubin_size() const { return cubin_size_; }
    size_t num_kernels() const { return num_kernels_; }
    size_t asts_per_kernel() const { return asts_per_kernel_; }
    size_t num_inputs() const { return num_inputs_; }
    size_t num_input_columns() const { return num_input_columns_; }
    size_t num_input_constants() const { return num_input_constants_; }
    size_t num_targets() const { return num_targets_; }
    size_t tile_rows() const { return tile_rows_; }
    size_t threads_per_block() const { return threads_per_block_; }
    SecantPythonCubinShape shape() const { return shape_; }
    const SecantCubinPlan* native_plan() const { return plan_; }

private:
    SecantPythonCubinPlan(SecantPythonCubinShape shape, const nb::bytes& cubin)
        : shape_(shape),
          cubin_size_(cubin.size()),
          aligned_cubin_((cubin.size() + sizeof(uint64_t) - 1u) / sizeof(uint64_t), 0u) {
        if (cubin_size_ == 0u) {
            throw nb::value_error("cubin must not be empty");
        }
        std::memcpy(aligned_cubin_.data(), cubin.data(), cubin_size_);
    }

    void inspect(const SecantCubinRecipeHeader* recipe) {
        size_t required_size = 0u;

        secant_python_check(
            secant_cubin_plan_storage_size(
                recipe,
                aligned_cubin_.data(),
                cubin_size_,
                &required_size),
            "secant_cubin_plan_storage_size");
        if (required_size == 0u) {
            throw std::runtime_error("CUBIN inspect returned invalid measure data");
        }
        plan_storage_.resize((required_size + sizeof(std::max_align_t) - 1u) / sizeof(std::max_align_t));
        secant_python_check(
            secant_cubin_plan_init(
                recipe,
                aligned_cubin_.data(),
                cubin_size_,
                plan_storage_.data(),
                plan_storage_.size() * sizeof(std::max_align_t),
                &plan_),
            "secant_cubin_plan_init");
        if (plan_ == nullptr) {
            throw std::runtime_error("CUBIN inspect returned a null plan");
        }
    }

    void read_recipe() {
        SecantCubinPlanInfo info = secant_cubin_plan_info_init();

        secant_python_check(secant_cubin_plan_info_get(plan_, &info), "secant_cubin_plan_info_get");
        cubin_size_ = info.cubin_size;
        switch (shape_) {
            case SecantPythonCubinShape::Materialize: {
                SecantCubinMaterializeRecipe recipe = secant_cubin_materialize_recipe_init();
                secant_python_check(secant_cubin_plan_recipe_get(plan_, &recipe.header),
                    "secant_cubin_plan_recipe_get");
                num_kernels_ = recipe.num_kernels;
                asts_per_kernel_ = recipe.asts_per_kernel;
                num_inputs_ = recipe.num_inputs;
                num_input_columns_ = recipe.num_inputs;
                break;
            }
            case SecantPythonCubinShape::SSE:
            case SecantPythonCubinShape::AffineStats:
            case SecantPythonCubinShape::GramStats: {
                if (shape_ == SecantPythonCubinShape::SSE) {
                    SecantCubinSSERecipe recipe = secant_cubin_sse_recipe_init();
                    secant_python_check(secant_cubin_plan_recipe_get(plan_, &recipe.header),
                        "secant_cubin_plan_recipe_get");
                    values_set(recipe);
                } else if (shape_ == SecantPythonCubinShape::AffineStats) {
                    SecantCubinAffineStatsRecipe recipe = secant_cubin_affine_stats_recipe_init();
                    secant_python_check(secant_cubin_plan_recipe_get(plan_, &recipe.header),
                        "secant_cubin_plan_recipe_get");
                    values_set(recipe);
                } else {
                    SecantCubinGramStatsRecipe recipe = secant_cubin_gram_stats_recipe_init();
                    secant_python_check(secant_cubin_plan_recipe_get(plan_, &recipe.header),
                        "secant_cubin_plan_recipe_get");
                    values_set(recipe);
                }
                break;
            }
            case SecantPythonCubinShape::DynamicConstantSSE: {
                SecantCubinDynamicConstantSSERecipe recipe = secant_cubin_dynamic_constant_sse_recipe_init();
                secant_python_check(secant_cubin_plan_recipe_get(plan_, &recipe.header),
                    "secant_cubin_plan_recipe_get");
                num_kernels_ = recipe.num_kernels;
                asts_per_kernel_ = recipe.asts_per_kernel;
                num_input_columns_ = recipe.num_input_columns;
                num_input_constants_ = recipe.num_input_constants;
                num_inputs_ = recipe.num_input_columns + recipe.num_input_constants;
                num_targets_ = recipe.num_targets;
                tile_rows_ = recipe.tile_rows;
                threads_per_block_ = recipe.threads_per_block;
                break;
            }
            case SecantPythonCubinShape::DynamicLeafSSE: {
                SecantCubinDynamicLeafSSERecipe recipe = secant_cubin_dynamic_leaf_sse_recipe_init();
                secant_python_check(secant_cubin_plan_recipe_get(plan_, &recipe.header),
                    "secant_cubin_plan_recipe_get");
                num_kernels_ = recipe.num_kernels;
                asts_per_kernel_ = recipe.asts_per_kernel;
                num_inputs_ = recipe.num_static_input_columns + recipe.num_dynamic_leaves;
                num_input_columns_ = recipe.num_input_columns;
                num_input_constants_ = recipe.num_dynamic_leaves;
                num_targets_ = recipe.num_targets;
                tile_rows_ = recipe.tile_rows;
                threads_per_block_ = recipe.threads_per_block;
                break;
            }
        }
    }

    template <typename Recipe>
    void values_set(const Recipe& recipe) {
        num_kernels_ = recipe.num_kernels;
        asts_per_kernel_ = recipe.asts_per_kernel;
        num_inputs_ = recipe.num_inputs;
        num_input_columns_ = recipe.num_inputs;
        num_targets_ = recipe.num_targets;
        tile_rows_ = recipe.tile_rows;
        threads_per_block_ = recipe.threads_per_block;
    }

    SecantPythonCubinShape shape_;
    SecantCubinPlan* plan_ = nullptr;
    size_t cubin_size_ = 0u;
    size_t num_kernels_ = 0u;
    size_t asts_per_kernel_ = 0u;
    size_t num_inputs_ = 0u;
    size_t num_input_columns_ = 0u;
    size_t num_input_constants_ = 0u;
    size_t num_targets_ = 0u;
    size_t tile_rows_ = 0u;
    size_t threads_per_block_ = 0u;
    std::vector<uint64_t> aligned_cubin_;
    std::vector<std::max_align_t> plan_storage_;
};

static nb::dict
secant_python_runner_stats(const SecantRunnerStats& stats) {
    nb::dict result;

    result["num_modules"] = stats.num_modules;
    result["num_asts"] = stats.num_asts;
    result["modules_loaded"] = stats.modules_loaded;
    result["compile_window_seconds"] = stats.compile_window_seconds;
    result["compile_critical_seconds"] = stats.compile_critical_seconds;
    result["compile_work_seconds"] = stats.compile_work_seconds;
    result["module_load_seconds"] = stats.module_load_seconds;
    result["completion_wait_seconds"] = stats.completion_wait_seconds;
    result["module_unload_seconds"] = stats.module_unload_seconds;
    result["runtime_seconds"] = stats.runtime_seconds;
    result["total_seconds"] = stats.total_seconds;
    return result;
}

class SecantPythonCubinSSERunner {
public:
    SecantPythonCubinSSERunner(
        const SecantPythonCubinPlan& plan,
        const nb::bytes& cubin,
        size_t num_workers,
        size_t num_streams
    ) {
        SecantCubinRunnerOptions options = secant_cubin_runner_options_init();

        if (plan.shape() != SecantPythonCubinShape::SSE) {
            throw nb::value_error("CUBIN plan is not an SSE plan");
        }
        if (cubin.size() != plan.cubin_size()) {
            throw nb::value_error("CUBIN size does not match the inspected SSE plan");
        }
        options.num_workers = num_workers;
        options.num_streams = num_streams;
        secant_python_check(
            secant_cubin_runner_create(
                plan.native_plan(),
                cubin.data(),
                cubin.size(),
                &options,
                &runner_),
            "secant_cubin_runner_create");
    }

    SecantPythonCubinSSERunner(const SecantPythonCubinSSERunner&) = delete;
    SecantPythonCubinSSERunner& operator=(const SecantPythonCubinSSERunner&) = delete;

    ~SecantPythonCubinSSERunner() {
        if (runner_ != nullptr) {
            (void)secant_cubin_runner_destroy(runner_);
        }
    }

    void close() {
        SecantResult result;

        if (runner_ == nullptr) {
            return;
        }
        result = secant_cubin_runner_destroy(runner_);
        if (result == SECANT_SUCCESS) {
            runner_ = nullptr;
        }
        secant_python_check(result, "secant_cubin_runner_destroy");
    }

    nb::dict run_all(
        nb::sequence routines,
        const SecantPythonByteArray& ast_data,
        const SecantPythonOffsetArray& ast_offsets,
        uintptr_t input_device_address,
        size_t input_num_elements,
        size_t input_leading_dimension,
        size_t num_targets,
        uintptr_t targets_device_address,
        size_t targets_num_elements,
        size_t targets_leading_dimension,
        size_t num_rows,
        uintptr_t output_device_address,
        size_t output_num_elements,
        size_t output_leading_dimension
    ) {
        const auto routine_pointers = secant_python_program_pointers(routines, "routines");
        const auto ast_pointers = secant_python_packed_program_pointers(ast_data, ast_offsets);
        SecantCubinSSERun run = secant_cubin_sse_run_init();
        SecantRunnerStats stats = secant_runner_stats_init();
        SecantResult result;

        if (runner_ == nullptr) {
            throw std::runtime_error("CUBIN SSE runner is closed");
        }
        run.programs.routines.items = routine_pointers.empty() ? nullptr : routine_pointers.data();
        run.programs.routines.count = routine_pointers.size();
        run.programs.asts.items = ast_pointers.data();
        run.programs.asts.count = ast_pointers.size();
        run.input.address = input_device_address;
        run.input.num_elements = input_num_elements;
        run.input.leading_dimension = input_leading_dimension;
        run.targets.address = targets_device_address;
        run.targets.num_elements = targets_num_elements;
        run.targets.leading_dimension = targets_leading_dimension;
        run.num_rows = num_rows;
        run.num_targets = num_targets;
        run.output.address = output_device_address;
        run.output.num_elements = output_num_elements;
        run.output.leading_dimension = output_leading_dimension;
        {
            nb::gil_scoped_release release;
            result = secant_cubin_runner_run_sse(runner_, &run, &stats);
        }
        secant_python_check(result, "secant_cubin_runner_run_sse");
        return secant_python_runner_stats(stats);
    }

private:
    SecantCubinRunner runner_ = nullptr;
};

NB_MODULE(_secant_native, module) {
    module.doc() = "Native CUBIN generation, inspection, and SASS specialization for Secant";

    module.def(
        "materialize_source_generate",
        &secant_python_generate_materialize_source,
        nb::arg("num_kernels"),
        nb::arg("asts_per_kernel"),
        nb::arg("num_inputs"),
        nb::arg("patch_capacity_instructions"));
    module.def(
        "sse_source_generate",
        &secant_python_generate_sse_source,
        nb::arg("num_kernels"),
        nb::arg("asts_per_kernel"),
        nb::arg("num_inputs"),
        nb::arg("num_targets"),
        nb::arg("tile_rows"),
        nb::arg("threads_per_block"),
        nb::arg("patch_capacity_instructions"));
    module.def(
        "affine_stats_source_generate",
        &secant_python_generate_affine_stats_source,
        nb::arg("num_kernels"),
        nb::arg("asts_per_kernel"),
        nb::arg("num_inputs"),
        nb::arg("num_targets"),
        nb::arg("tile_rows"),
        nb::arg("threads_per_block"),
        nb::arg("patch_capacity_instructions"));
    module.def(
        "gram_stats_source_generate",
        &secant_python_generate_gram_stats_source,
        nb::arg("num_kernels"),
        nb::arg("asts_per_kernel"),
        nb::arg("num_inputs"),
        nb::arg("num_targets"),
        nb::arg("tile_rows"),
        nb::arg("threads_per_block"),
        nb::arg("patch_capacity_instructions"));
    module.def(
        "dynamic_constant_sse_source_generate",
        &secant_python_generate_dynamic_constant_sse_source,
        nb::arg("num_kernels"),
        nb::arg("asts_per_kernel"),
        nb::arg("num_input_columns"),
        nb::arg("num_input_constants"),
        nb::arg("num_targets"),
        nb::arg("tile_rows"),
        nb::arg("threads_per_block"),
        nb::arg("patch_capacity_instructions"));
    module.def(
        "dynamic_leaf_sse_source_generate",
        &secant_python_generate_dynamic_leaf_sse_source,
        nb::arg("num_kernels"),
        nb::arg("asts_per_kernel"),
        nb::arg("num_input_columns"),
        nb::arg("num_static_input_columns"),
        nb::arg("num_dynamic_leaves"),
        nb::arg("num_targets"),
        nb::arg("tile_rows"),
        nb::arg("threads_per_block"),
        nb::arg("patch_capacity_instructions"));

    nb::class_<SecantPythonCubinPlan>(module, "CubinPlan")
        .def_static(
            "inspect_materialize",
            &SecantPythonCubinPlan::inspect_materialize,
            nb::arg("cubin"),
            nb::arg("num_kernels"),
            nb::arg("asts_per_kernel"),
            nb::arg("num_inputs"),
            nb::arg("patch_capacity_instructions"))
        .def_static(
            "inspect_sse",
            &SecantPythonCubinPlan::inspect_sse,
            nb::arg("cubin"),
            nb::arg("num_kernels"),
            nb::arg("asts_per_kernel"),
            nb::arg("num_inputs"),
            nb::arg("num_targets"),
            nb::arg("tile_rows"),
            nb::arg("threads_per_block"),
            nb::arg("patch_capacity_instructions"))
        .def_static(
            "inspect_affine_stats",
            &SecantPythonCubinPlan::inspect_affine_stats,
            nb::arg("cubin"),
            nb::arg("num_kernels"),
            nb::arg("asts_per_kernel"),
            nb::arg("num_inputs"),
            nb::arg("num_targets"),
            nb::arg("tile_rows"),
            nb::arg("threads_per_block"),
            nb::arg("patch_capacity_instructions"))
        .def_static(
            "inspect_gram_stats",
            &SecantPythonCubinPlan::inspect_gram_stats,
            nb::arg("cubin"),
            nb::arg("num_kernels"),
            nb::arg("asts_per_kernel"),
            nb::arg("num_inputs"),
            nb::arg("num_targets"),
            nb::arg("tile_rows"),
            nb::arg("threads_per_block"),
            nb::arg("patch_capacity_instructions"))
        .def_static(
            "inspect_dynamic_constant_sse",
            &SecantPythonCubinPlan::inspect_dynamic_constant_sse,
            nb::arg("cubin"),
            nb::arg("num_kernels"),
            nb::arg("asts_per_kernel"),
            nb::arg("num_input_columns"),
            nb::arg("num_input_constants"),
            nb::arg("num_targets"),
            nb::arg("tile_rows"),
            nb::arg("threads_per_block"),
            nb::arg("patch_capacity_instructions"))
        .def_static(
            "inspect_dynamic_leaf_sse",
            &SecantPythonCubinPlan::inspect_dynamic_leaf_sse,
            nb::arg("cubin"),
            nb::arg("num_kernels"),
            nb::arg("asts_per_kernel"),
            nb::arg("num_input_columns"),
            nb::arg("num_static_input_columns"),
            nb::arg("num_dynamic_leaves"),
            nb::arg("num_targets"),
            nb::arg("tile_rows"),
            nb::arg("threads_per_block"),
            nb::arg("patch_capacity_instructions"))
        .def(
            "specialize_into",
            &SecantPythonCubinPlan::specialize_into,
            nb::arg("routines"),
            nb::arg("asts"),
            nb::arg("cubin"))
        .def_prop_ro("cubin_size", &SecantPythonCubinPlan::cubin_size)
        .def_prop_ro("num_kernels", &SecantPythonCubinPlan::num_kernels)
        .def_prop_ro("asts_per_kernel", &SecantPythonCubinPlan::asts_per_kernel)
        .def_prop_ro("num_inputs", &SecantPythonCubinPlan::num_inputs)
        .def_prop_ro("num_input_columns", &SecantPythonCubinPlan::num_input_columns)
        .def_prop_ro("num_input_constants", &SecantPythonCubinPlan::num_input_constants)
        .def_prop_ro("num_targets", &SecantPythonCubinPlan::num_targets)
        .def_prop_ro("tile_rows", &SecantPythonCubinPlan::tile_rows)
        .def_prop_ro("threads_per_block", &SecantPythonCubinPlan::threads_per_block);

    nb::class_<SecantPythonCubinSSERunner>(module, "CubinSSERunner")
        .def(
            nb::init<const SecantPythonCubinPlan&, const nb::bytes&, size_t, size_t>(),
            nb::arg("plan"),
            nb::arg("cubin"),
            nb::arg("num_workers"),
            nb::arg("num_streams"),
            nb::keep_alive<1, 2>())
        .def(
            "run_all",
            &SecantPythonCubinSSERunner::run_all,
            nb::arg("routines"),
            nb::arg("ast_data"),
            nb::arg("ast_offsets"),
            nb::arg("input_device_address"),
            nb::arg("input_num_elements"),
            nb::arg("input_leading_dimension"),
            nb::arg("num_targets"),
            nb::arg("targets_device_address"),
            nb::arg("targets_num_elements"),
            nb::arg("targets_leading_dimension"),
            nb::arg("num_rows"),
            nb::arg("output_device_address"),
            nb::arg("output_num_elements"),
            nb::arg("output_leading_dimension"))
        .def("close", &SecantPythonCubinSSERunner::close);
}
