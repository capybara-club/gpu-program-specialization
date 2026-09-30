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
#include "o_odezza_internal.h"

#include "o_sha256.h"
#include "o_writer.h"
#include "o_sampled_constants_source.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#define O_FIRST_MARKER_BITS 0x7fc0a11eu
#define O_SHARED_BEGIN_MARKER_BITS 0x7fc1a11eu
#define O_SHARED_END_MARKER_BITS 0x7fc2a11eu
#define O_OUTPUT_LIVE_MARKER_BITS 0x7fc3a11eu
#define O_CONFIGURATION_ORDER "grid-y selects system; specialized SASS bit masks read low configuration bits and the runtime uniform active-toggle count shifts to the constant bank"

#define O_TRY(expression) \
    do { \
        OdezzaResult o_result_ = (expression); \
        if (o_result_ != ODEZZA_SUCCESS) return o_result_; \
    } while (0)

typedef struct OScoringCudaInfo {
    size_t input_count;
    size_t output_count;
    size_t arena_instruction_count;
} OScoringCudaInfo;

static OdezzaResult o_scoring_size_multiply(size_t lhs, size_t rhs, size_t *value_ret) {
    if (value_ret == NULL) {
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    }
    if (rhs != 0u && lhs > SIZE_MAX / rhs) {
        return ODEZZA_ERROR_OVERFLOW;
    }
    *value_ret = lhs * rhs;
    return ODEZZA_SUCCESS;
}

static OdezzaResult o_shape_validate(
    size_t state_capacity,
    uint32_t constant_capacity,
    uint32_t system_capacity,
    uint32_t shared_patch_capacity,
    uint32_t system_patch_capacity,
    OScoringCudaInfo *info
) {
    if (info == NULL || state_capacity == 0u) {
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    }
    if (system_capacity == 0u || shared_patch_capacity == 0u || system_patch_capacity == 0u || system_capacity > 65535u
    ) {
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    }
    if (state_capacity > SIZE_MAX - constant_capacity) {
        return ODEZZA_ERROR_OVERFLOW;
    }
    info->input_count = state_capacity + constant_capacity;
    info->output_count = state_capacity;
    if (info->input_count > SIZE_MAX - info->output_count - 1u || info->input_count + info->output_count + 1u >= 255u) {
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    }
    O_TRY(o_scoring_size_multiply(system_capacity, system_patch_capacity, &info->arena_instruction_count));
    if (info->arena_instruction_count < (size_t)system_capacity * 2u) {
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    }
    return ODEZZA_SUCCESS;
}

static OdezzaResult o_write_manifest_canonical(
    OWriter *writer,
    size_t state_capacity,
    uint32_t constant_capacity,
    uint32_t system_capacity,
    uint32_t shared_patch_capacity,
    uint32_t system_patch_capacity,
    const OScoringCudaInfo *info
) {
    O_TRY(o_writer_cstr(writer, "{\"abi\":{\"input_register_order\":[\"states\",\"system_constants\",\"permutation_u32\"],\"layout\":\"shared-prelude-then-"
                                "compact-system-dispatch-v1\",\"marker_version\":2,\"output_register_order\":\"rhs_by_state\",\"postorder\":\"odezza-postorder-"
                                "f32-v2\",\"sass\":\"odezza-sm89-sm90-sm120-v2\",\"template_id_symbol\":\"odezza_template_id\"},"));
    O_TRY(o_writer_cstr(writer,
                        "\"configuration\":{\"constant_banks\":\"runtime system-major\",\"ordering\":\"" O_CONFIGURATION_ORDER "\",\"permutation_word_bits\":"
                        "32,\"toggle_bit_positions\":\"specialized AST immediates\",\"toggle_permutations\":\"runtime power of two\"},"));
    O_TRY(o_writer_cstr(writer, "\"data\":{\"constant_layout\":\"constant[system,bank,index]\",\"offset_layout\":\"trajectory_offsets[trajectory+1]\","
                                "\"output_layout\":\"mse[system,configuration]\",\"reference_layout\":\"reference[state,concatenated_point]\","
                                "\"time_layout\":\"time[concatenated_point]\",\"trajectory_layout\":\"runtime_ragged_dense_state\"},"));
    O_TRY(o_writer_cstr(writer, "\"generator_abi_version\":5,"));
    O_TRY(o_writer_cstr(writer, "\"kernel\":{\"integrator\":\"rk4\",\"name\":\"odezza_scoring\",\"ownership\":\"one candidate system per grid-y coordinate and "
                                "one configuration per thread\",\"score\":\"ragged dense-state mean squared trajectory error\",\"shape\":\"scoring\"},"));
    O_TRY(o_writer_cstr(writer, "\"schema\":\"odezza.cuda-template\",\"schema_version\":2,\"shape\":{"));
    O_TRY(o_writer_format(writer,
                          "\"arena_instruction_count\":%zu,\"constant_capacity\":%u,\"input_count\":%zu,\"output_count\":%zu,"
                          "\"shared_patch_capacity\":%u,\"state_capacity\":%zu,\"system_capacity\":%u,\"system_patch_capacity\":%u},",
                          info->arena_instruction_count, constant_capacity, info->input_count, info->output_count, shared_patch_capacity, state_capacity,
                          system_capacity, system_patch_capacity));
    O_TRY(o_writer_format(writer,
                          "\"specialization\":{\"branches\":\"one complete complementary derivative program set per candidate system\",\"shared\":\"common "
                          "derivative programs execute before uniform system "
                          "dispatch\",\"shared_patch_capacity\":%u,\"system_capacity\":%u,\"system_patch_capacity\":%u},\"template_id_algorithm\":\"sha256\"}",
                          shared_patch_capacity, system_capacity, system_patch_capacity));
    return ODEZZA_SUCCESS;
}

static OdezzaResult o_digest_hex(const unsigned char digest[32], char output[65]) {
    static const char digits[] = "0123456789abcdef";
    size_t index;
    if (digest == NULL || output == NULL) {
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    }
    for (index = 0u; index < 32u; ++index) {
        output[index * 2u] = digits[digest[index] >> 4u];
        output[index * 2u + 1u] = digits[digest[index] & 15u];
    }
    output[64] = '\0';
    return ODEZZA_SUCCESS;
}

static OdezzaResult o_write_manifest_pretty(
    OWriter *writer,
    size_t state_capacity,
    uint32_t constant_capacity,
    uint32_t system_capacity,
    uint32_t shared_patch_capacity,
    uint32_t system_patch_capacity,
    const OScoringCudaInfo *info,
    const unsigned char template_id[32],
    const unsigned char source_sha256[32]
) {
    char template_hex[65];
    char source_hex[65];
    O_TRY(o_digest_hex(template_id, template_hex));
    O_TRY(o_digest_hex(source_sha256, source_hex));
    O_TRY(o_writer_cstr(writer, "/* ODEZZA_MANIFEST_BEGIN\n"
                                "{\n"
                                "  \"abi\": {\n"
                                "    \"input_register_order\": [\n"
                                "      \"states\",\n"
                                "      \"system_constants\",\n"
                                "      \"permutation_u32\"\n"
                                "    ],\n"
                                "    \"layout\": \"shared-prelude-then-compact-system-dispatch-v1\",\n"
                                "    \"marker_version\": 2,\n"
                                "    \"output_register_order\": \"rhs_by_state\",\n"
                                "    \"postorder\": \"odezza-postorder-f32-v2\",\n"
                                "    \"sass\": \"odezza-sm89-sm90-sm120-v2\",\n"
                                "    \"template_id_symbol\": \"odezza_template_id\"\n"
                                "  },\n"
                                "  \"configuration\": {\n"
                                "    \"constant_banks\": \"runtime system-major\",\n"
                                "    \"ordering\": \"" O_CONFIGURATION_ORDER "\",\n"
                                "    \"permutation_word_bits\": 32,\n"));
    O_TRY(o_writer_cstr(writer,
                        "    \"toggle_bit_positions\": \"specialized AST immediates\",\n"
                        "    \"toggle_permutations\": \"runtime power of two\"\n"));
    O_TRY(o_writer_cstr(writer, "  },\n"
                                "  \"data\": {\n"
                                "    \"constant_layout\": \"constant[system,bank,index]\",\n"
                                "    \"offset_layout\": \"trajectory_offsets[trajectory+1]\",\n"
                                "    \"output_layout\": \"mse[system,configuration]\",\n"
                                "    \"reference_layout\": \"reference[state,concatenated_point]\",\n"
                                "    \"time_layout\": \"time[concatenated_point]\",\n"
                                "    \"trajectory_layout\": \"runtime_ragged_dense_state\"\n"
                                "  },\n"
                                "  \"generator_abi_version\": 5,\n"
                                "  \"kernel\": {\n"
                                "    \"integrator\": \"rk4\",\n"
                                "    \"name\": \"odezza_scoring\",\n"
                                "    \"ownership\": \"one candidate system per grid-y coordinate and one configuration per thread\",\n"
                                "    \"score\": \"ragged dense-state mean squared trajectory error\",\n"
                                "    \"shape\": \"scoring\"\n"
                                "  },\n"
                                "  \"schema\": \"odezza.cuda-template\",\n"
                                "  \"schema_version\": 2,\n"
                                "  \"shape\": {\n"));
    O_TRY(o_writer_format(writer,
                          "    \"arena_instruction_count\": %zu,\n"
                          "    \"constant_capacity\": %u,\n"
                          "    \"input_count\": %zu,\n"
                          "    \"output_count\": %zu,\n"
                          "    \"shared_patch_capacity\": %u,\n"
                          "    \"state_capacity\": %zu,\n",
                          info->arena_instruction_count, constant_capacity, info->input_count, info->output_count, shared_patch_capacity, state_capacity));
    O_TRY(o_writer_format(writer,
                          "    \"system_capacity\": %u,\n"
                          "    \"system_patch_capacity\": %u\n"
                          "  },\n"
                          "  \"source_sha256\": \"%s\",\n"
                          "  \"specialization\": {\n"
                          "    \"branches\": \"one complete complementary derivative program set per candidate system\",\n"
                          "    \"shared\": \"common derivative programs execute before uniform system dispatch\",\n"
                          "    \"shared_patch_capacity\": %u,\n"
                          "    \"system_capacity\": %u,\n"
                          "    \"system_patch_capacity\": %u\n"
                          "  },\n"
                          "  \"template_id\": \"%s\",\n"
                          "  \"template_id_algorithm\": \"sha256\"\n"
                          "}\n"
                          "ODEZZA_MANIFEST_END */\n",
                          system_capacity, system_patch_capacity, source_hex, shared_patch_capacity, system_capacity,
                          system_patch_capacity, template_hex));
    return ODEZZA_SUCCESS;
}

static OdezzaResult o_load_u32_le(const unsigned char *data, uint32_t *value_ret) {
    if (data == NULL || value_ret == NULL) {
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    }
    *value_ret = (uint32_t)data[0] | ((uint32_t)data[1] << 8u) | ((uint32_t)data[2] << 16u) | ((uint32_t)data[3] << 24u);
    return ODEZZA_SUCCESS;
}

static OdezzaResult o_write_template_declaration(
    OWriter *writer,
    const unsigned char template_id[32],
    size_t state_capacity,
    uint32_t constant_capacity,
    uint32_t system_capacity
) {
    size_t index;
    uint32_t word;
    O_TRY(o_writer_cstr(writer, "extern \"C\" __device__ __constant__ unsigned int odezza_template_id[8] = {"));
    for (index = 0u; index < 8u; ++index) {
        O_TRY(o_load_u32_le(template_id + index * 4u, &word));
        O_TRY(o_writer_format(writer, "%s0x%08xu", index == 0u ? "" : ", ", word));
    }
    O_TRY(o_writer_cstr(writer, "};\n"));
    O_TRY(o_writer_format(
        writer,
        "extern \"C\" __device__ __constant__ unsigned int odezza_state_capacity = %zuu;\n",
        state_capacity
    ));
    O_TRY(o_writer_format(
        writer,
        "extern \"C\" __device__ __constant__ unsigned int odezza_constant_capacity = %uu;\n",
        constant_capacity
    ));
    return o_writer_format(
        writer,
        "extern \"C\" __device__ __constant__ unsigned int odezza_system_capacity = %uu;\n\n",
        system_capacity
    );
}

static OdezzaResult o_write_ptx_format(OWriter *writer, const char *format, ...) {
    char line[768];
    va_list arguments;
    int count;
    if (writer == NULL || format == NULL) {
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    }
    va_start(arguments, format);
    count = vsnprintf(line, sizeof(line), format, arguments);
    va_end(arguments);
    if (count < 0) {
        return ODEZZA_ERROR_FORMAT;
    }
    if ((size_t)count >= sizeof(line)) {
        return ODEZZA_ERROR_OVERFLOW;
    }
    O_TRY(o_writer_cstr(writer, "                        \""));
    O_TRY(o_writer_write(writer, line, (size_t)count));
    return o_writer_cstr(writer, "\\n\\t\"\n");
}

/* Reserved instructions dominate large templates; no formatting is needed. */
static OdezzaResult o_write_breakpoints(OWriter *writer, size_t count) {
    static const char line[] = "                        \"brkpt;\\n\\t\"\n";
    size_t index;
    for (index = 0u; index < count; ++index) {
        O_TRY(o_writer_write(writer, line, sizeof(line) - 1u));
    }
    return ODEZZA_SUCCESS;
}

static OdezzaResult o_write_marker(
    OWriter *writer,
    uint32_t system_capacity,
    uint32_t shared_patch_capacity,
    const OScoringCudaInfo *info
) {
    size_t index;
    size_t keepalive_operand = info->input_count + info->output_count + 2u;
    size_t input_operand_start = info->input_count + info->output_count + 3u;
    size_t permutation_operand = input_operand_start + info->input_count;
    size_t dispatch_operand = permutation_operand + 1u;
    size_t placeholder_instructions = (size_t)system_capacity * 2u;
    size_t next_input;

    O_TRY(o_write_breakpoints(writer, 1u));
    for (index = 0u; index < info->input_count; ++index) {
        O_TRY(o_write_ptx_format(writer, "add.rn.ftz.f32 %%%zu, %%%zu, 0f%08x;", index, input_operand_start + index, O_FIRST_MARKER_BITS + (uint32_t)index));
    }
    O_TRY(o_write_breakpoints(writer, 1u));
    for (index = 1u; index < info->input_count; ++index) {
        O_TRY(o_write_ptx_format(writer, "add.rn.ftz.f32 %%0, %%0, %%%zu;", index));
    }
    O_TRY(o_write_ptx_format(writer, ".reg .pred odezza_toggle_predicate;"));
    O_TRY(o_write_ptx_format(writer, ".reg .b32 odezza_toggle_masked;"));
    O_TRY(o_write_ptx_format(writer, "and.b32 odezza_toggle_masked, %%%zu, 1;", permutation_operand));
    O_TRY(o_write_ptx_format(writer, "setp.ne.u32 odezza_toggle_predicate, odezza_toggle_masked, 0;"));
    /* With one input, operand 1 is an RHS output and is not initialized yet. */
    O_TRY(o_write_ptx_format(writer, "selp.f32 %%0, %%0, %%%zu, odezza_toggle_predicate;",
                             info->input_count > 1u ? 1u : input_operand_start));
    for (index = 0u; index < info->output_count; ++index) {
        O_TRY(o_write_ptx_format(writer, "add.rn.ftz.f32 %%%zu, %%0, 0f%08x;", info->input_count + index, O_OUTPUT_LIVE_MARKER_BITS + (uint32_t)index));
    }
    O_TRY(o_write_ptx_format(writer, "add.rn.ftz.f32 %%%zu, %%%zu, 0f%08x;", info->input_count + info->output_count, input_operand_start,
                             O_SHARED_BEGIN_MARKER_BITS));
    O_TRY(o_write_breakpoints(writer, shared_patch_capacity));
    O_TRY(o_write_ptx_format(writer, "add.rn.ftz.f32 %%%zu, %%%zu, 0f%08x;", info->input_count + info->output_count + 1u, input_operand_start,
                             O_SHARED_END_MARKER_BITS));

    if (system_capacity > 1u) {
        O_TRY(o_writer_cstr(writer, "                        \"odezza_system_targets: .branchtargets "));
        for (index = 0u; index < system_capacity; ++index) {
            O_TRY(o_writer_format(writer, "%sodezza_system_%zu", index == 0u ? "" : ", ", index));
        }
        O_TRY(o_writer_cstr(writer, ";\\n\\t\"\n"));
        O_TRY(o_write_ptx_format(writer, "brx.idx.uni %%%zu, odezza_system_targets;", dispatch_operand));
    }
    for (index = 0u; index < system_capacity; ++index) {
        size_t reserve;
        O_TRY(o_write_ptx_format(writer, "odezza_system_%zu:", index));
        O_TRY(o_write_breakpoints(writer, 1u));
        reserve = index + 1u == system_capacity ? info->arena_instruction_count - placeholder_instructions : 0u;
        O_TRY(o_write_breakpoints(writer, reserve));
        O_TRY(o_write_ptx_format(writer, "bra.uni odezza_system_continue;"));
    }
    O_TRY(o_write_ptx_format(writer, "odezza_system_continue:"));
    for (index = 0u; index < info->output_count; ++index) {
        O_TRY(o_write_ptx_format(writer, "add.rn.ftz.f32 %%%zu, %%%zu, 0f%08x;", info->input_count + index, info->input_count + index,
                                 O_FIRST_MARKER_BITS + (uint32_t)(info->input_count + index)));
    }
    O_TRY(o_write_breakpoints(writer, 1u));
    if (info->output_count == 1u) {
        O_TRY(o_write_ptx_format(writer, "add.rn.ftz.f32 %%%zu, %%%zu, %%%zu;", keepalive_operand, info->input_count, input_operand_start));
        next_input = 1u;
    } else {
        O_TRY(o_write_ptx_format(writer, "add.rn.ftz.f32 %%%zu, %%%zu, %%%zu;", keepalive_operand, info->input_count, info->input_count + 1u));
        for (index = 2u; index < info->output_count; ++index) {
            O_TRY(o_write_ptx_format(writer, "add.rn.ftz.f32 %%%zu, %%%zu, %%%zu;", keepalive_operand, keepalive_operand, info->input_count + index));
        }
        next_input = 0u;
    }
    for (index = next_input; index < info->input_count; ++index) {
        O_TRY(o_write_ptx_format(writer, "add.rn.ftz.f32 %%%zu, %%%zu, %%%zu;", keepalive_operand, keepalive_operand, input_operand_start + index));
    }
    /* Specialized toggles read permutation throughout the arena. Keep the
     * input live past output materialization so ptxas cannot reuse its register. */
    O_TRY(o_write_ptx_format(writer, "add.rn.ftz.f32 %%%zu, %%%zu, %%%zu;", keepalive_operand, keepalive_operand, permutation_operand));
    O_TRY(o_write_ptx_format(writer, "add.rn.ftz.f32 %%%zu, %%%zu, %%%zu;", keepalive_operand, keepalive_operand, info->input_count + info->output_count));
    O_TRY(o_write_ptx_format(writer, "add.rn.ftz.f32 %%%zu, %%%zu, %%%zu;", keepalive_operand, keepalive_operand, info->input_count + info->output_count + 1u));
    O_TRY(o_write_ptx_format(writer, "mov.u32 keepalive_address, 0;"));
    O_TRY(o_write_ptx_format(writer, "st.volatile.shared.f32 [keepalive_address], %%%zu;", keepalive_operand));
    return o_write_breakpoints(writer, 1u);
}

static OdezzaResult o_write_constraints(OWriter *writer, const OScoringCudaInfo *info) {
    size_t index;
    int first = 1;
    O_TRY(o_writer_cstr(writer, "                        : "));
    for (index = 0u; index < info->input_count; ++index) {
        O_TRY(o_writer_format(writer, "%s\"=&f\"(marked%zu)", first ? "" : ", ", index));
        first = 0;
    }
    for (index = 0u; index < info->output_count; ++index) {
        O_TRY(o_writer_format(writer, ", \"=&f\"(rhs%zu)", index));
    }
    O_TRY(o_writer_cstr(writer, ", \"=&f\"(shared_boundary_begin), \"=&f\"(shared_boundary_end), \"=&f\"(keepalive)\n"));
    O_TRY(o_writer_cstr(writer, "                        : "));
    for (index = 0u; index < info->input_count; ++index) {
        O_TRY(o_writer_format(writer, "%s\"f\"(site_input%zu)", index == 0u ? "" : ", ", index));
    }
    return o_writer_cstr(writer, ", \"r\"(permutation), \"r\"(system_index)\n                        : \"memory\");\n");
}

static OdezzaResult o_write_cuda_body(
    OWriter *writer,
    size_t state_capacity,
    uint32_t constant_capacity,
    uint32_t system_capacity,
    uint32_t shared_patch_capacity,
    const OScoringCudaInfo *info
) {
    size_t index;
    O_TRY(o_writer_cstr(writer, "/* Generated CUDA template. Do not edit by hand.\n"
                                " * shape: scoring\n"
                                " * ownership: grid-y selects one candidate system; each thread owns one configuration\n"
                                " * integration: fused RK4 over runtime ragged trajectories\n"
                                " * specialization: common derivative prelude followed by uniform candidate-system dispatch\n"
                                " * equations: every shared and candidate expression is inserted by the postorder specializer\n"
                                " */\n"));
    O_TRY(o_writer_format(writer,
                          "#define ODEZZA_STATE_CAPACITY %zu\n"
                          "#define ODEZZA_CONSTANT_CAPACITY %u\n"
                          "#define ODEZZA_SYSTEM_CAPACITY %uu\n\n",
                          state_capacity, constant_capacity, system_capacity));
    for (index=0u;index<sizeof(o_sampled_source)/sizeof(*o_sampled_source);++index)
        O_TRY(o_writer_cstr(writer,o_sampled_source[index]));
    O_TRY(o_writer_cstr(writer,
                        "extern \"C\" __global__ void odezza_scoring(\n"
                        "    const float *__restrict__ constant_banks,\n"
                        "    unsigned int constant_bank_count,\n"
                        "    const unsigned int *__restrict__ trajectory_offsets,\n"
                        "    const float *__restrict__ trajectory_times,\n"
                        "    const float *__restrict__ reference_data,\n"
                        "    unsigned int trajectory_count,\n"
                        "    unsigned int trajectory_point_count,\n"
                        "    unsigned int active_state_count,\n"
                        "    unsigned int active_constant_count,\n"
                        "    unsigned int active_toggle_count,\n"
                        "    unsigned int active_system_count,\n"
                        "    unsigned int steps_per_observation,\n"
                        "    float *__restrict__ mse_out,\n"
                        "    const OParameter *__restrict__ sampled_parameters,\n"
                        "    const float *__restrict__ uniform_pool,\n"
                        "    const float *__restrict__ normal_pool,\n"
                        "    unsigned long long rng_pool_size,\n"
                        "    unsigned int allow_missing_observations\n"
                        ") {\n"
                        "    if (active_state_count == 0u || active_state_count > ODEZZA_STATE_CAPACITY ||\n"
                        "        active_constant_count > ODEZZA_CONSTANT_CAPACITY || active_toggle_count > 32u ||\n"
                        "        trajectory_count == 0u || trajectory_point_count < trajectory_count ||\n"
                        "        constant_bank_count == 0u || steps_per_observation == 0u) return;\n"
                        "    const unsigned int reference_float_count = active_state_count * trajectory_point_count;\n"
                        "    extern __shared__ unsigned int shared_words[];\n"
                        "    float *reference = reinterpret_cast<float *>(shared_words);\n"
                        "    float *times = reference + reference_float_count;\n"
                        "    unsigned int *offsets = reinterpret_cast<unsigned int *>(times + trajectory_point_count);\n"
                        "    int invalid_data = 0;\n"
                        "    for (unsigned int index = threadIdx.x; index < reference_float_count; index += blockDim.x) {\n"
                        "        reference[index] = reference_data[index];\n"
                        "        invalid_data |= !isfinite(reference[index]) && !(allow_missing_observations && isnan(reference[index]));\n"
                        "    }\n"
                        "    for (unsigned int index = threadIdx.x; index < trajectory_point_count; index += blockDim.x) {\n"
                        "        times[index] = trajectory_times[index];\n"
                        "        invalid_data |= !isfinite(times[index]);\n"
                        "    }\n"
                        "    for (unsigned int index = threadIdx.x; index <= trajectory_count; index += blockDim.x) {\n"
                        "        offsets[index] = trajectory_offsets[index];\n"
                        "    }\n"
                        "    const int invalid_observations = __syncthreads_or(invalid_data);\n\n"
                        "    const unsigned int system_index = blockIdx.y;\n"
                        "    const unsigned long long configuration = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n"
                        "    const unsigned long long configuration_count = (unsigned long long)constant_bank_count << active_toggle_count;\n"
                        "    if (system_index >= active_system_count || system_index >= ODEZZA_SYSTEM_CAPACITY || configuration >= configuration_count) return;\n"
                        "    const unsigned long long output_index = (unsigned long long)system_index * configuration_count + configuration;\n"
                        "    if (invalid_observations) { mse_out[output_index] = 3.402823466e+38F; return; }\n"
                        "    const unsigned int permutation = (unsigned int)configuration;\n"
                        "    const unsigned int constant_bank = (unsigned int)(configuration >> active_toggle_count);\n"));
    for (index = 0u; index < constant_capacity; ++index) {
        O_TRY(o_writer_format(writer,
                              "    const float constant%zu = %zuu < active_constant_count\n"
                              "        ? (sampled_parameters ? o_sample_constant(sampled_parameters,uniform_pool,normal_pool,rng_pool_size, (unsigned long long)system_index*active_constant_count+%zuull,constant_bank) : constant_banks[((unsigned long long)system_index * constant_bank_count + constant_bank) * active_constant_count + "
                              "%zuull])\n"
                              "        : 0.0f;\n",
                              index, index, index, index));
    }
    O_TRY(o_writer_cstr(writer,
                        "\n"
                        "    float squared_error = 0.0f;\n"
                        "    unsigned int scored_point_count = 0u;\n"
                        "    unsigned int scored_residual_count = 0u;\n"
                        "    unsigned int expected_start = 0u;\n"
                        "    int valid = 1;\n\n"
                        "#pragma unroll 1\n"
                        "    for (unsigned int trajectory = 0u; trajectory < trajectory_count; ++trajectory) {\n"
                        "        const unsigned int point_begin = offsets[trajectory];\n"
                        "        const unsigned int point_end = offsets[trajectory + 1u];\n"
                        "        if (point_begin != expected_start || point_end <= point_begin || point_end > trajectory_point_count) {\n"
                        "            valid = 0;\n"
                        "            break;\n"
                        "        }\n"
                        "        expected_start = point_end;\n"
                        "        float state[ODEZZA_STATE_CAPACITY];\n"));
    for (index = 0u; index < state_capacity; ++index) {
        O_TRY(o_writer_format(writer,
                              "        state[%zu] = %zuu < active_state_count ? reference[%zu * trajectory_point_count + point_begin] : 0.0f;\n"
                              "        valid = valid && isfinite(state[%zu]);\n",
                              index, index, index, index));
    }
    O_TRY(o_writer_cstr(writer, "\n"
                                "#pragma unroll 1\n"
                                "        for (unsigned int point = point_begin + 1u; point < point_end; ++point) {\n"
                                "            const float interval = times[point] - times[point - 1u];\n"
                                "            const float h = interval / (float)steps_per_observation;\n"
                                "            valid = valid && isfinite(interval) && interval > 0.0f && h > 0.0f;\n"
                                "            const float half_h = 0.5f * h;\n"
                                "            const float sixth_h = h / 6.0f;\n"
                                "#pragma unroll 1\n"
                                "            for (unsigned int step = 0; step < steps_per_observation; ++step) {\n"
                                "                float base_state[ODEZZA_STATE_CAPACITY];\n"
                                "                float stage_state[ODEZZA_STATE_CAPACITY];\n"
                                "                float sum_state[ODEZZA_STATE_CAPACITY];\n"
                                "#pragma unroll\n"
                                "                for (int component = 0; component < ODEZZA_STATE_CAPACITY; ++component) {\n"
                                "                    base_state[component] = state[component];\n"
                                "                    stage_state[component] = state[component];\n"
                                "                    sum_state[component] = 0.0f;\n"
                                "                }\n\n"
                                "#pragma unroll 1\n"
                                "                for (int rk_stage = 0; rk_stage < 4; ++rk_stage) {\n"));
    for (index = 0u; index < state_capacity; ++index) {
        O_TRY(o_writer_format(writer, "                    const float site_input%zu = stage_state[%zu];\n", index, index));
    }
    for (index = 0u; index < constant_capacity; ++index) {
        O_TRY(o_writer_format(writer, "                    const float site_input%zu = constant%zu;\n", state_capacity + index, index));
    }
    for (index = 0u; index < info->input_count; ++index) {
        O_TRY(o_writer_format(writer, "                    float marked%zu __attribute__((unused));\n", index));
    }
    for (index = 0u; index < info->output_count; ++index) {
        O_TRY(o_writer_format(writer, "                    float rhs%zu;\n", index));
    }
    O_TRY(o_writer_cstr(writer, "                    float shared_boundary_begin __attribute__((unused));\n"
                                "                    float shared_boundary_end __attribute__((unused));\n"
                                "                    float keepalive __attribute__((unused));\n"
                                "                    asm volatile(\n"
                                "                        \"{\\n\\t\"\n"
                                "                        \".reg .u32 keepalive_address;\\n\\t\"\n"));
    O_TRY(o_write_marker(writer, system_capacity, shared_patch_capacity, info));
    O_TRY(o_writer_cstr(writer, "                        \"}\\n\\t\"\n"));
    O_TRY(o_write_constraints(writer, info));
    O_TRY(o_writer_cstr(writer, "\n                    const float weight = (rk_stage == 0 || rk_stage == 3) ? 1.0f : 2.0f;\n"));
    for (index = 0u; index < state_capacity; ++index) {
        O_TRY(o_writer_format(writer, "                    sum_state[%zu] = fmaf(weight, rhs%zu, sum_state[%zu]);\n", index, index, index));
    }
    O_TRY(o_writer_cstr(writer, "                    if (rk_stage != 3) {\n                        const float stage_h = rk_stage == 2 ? h : half_h;\n"));
    for (index = 0u; index < state_capacity; ++index) {
        O_TRY(o_writer_format(writer,
                              "                        if (%zuu < active_state_count) stage_state[%zu] = fmaf(stage_h, rhs%zu, base_state[%zu]);\n",
                              index, index, index, index));
    }
    O_TRY(o_writer_cstr(writer, "                    }\n                }\n"));
    for (index = 0u; index < state_capacity; ++index) {
        O_TRY(o_writer_format(writer,
                              "                if (%zuu < active_state_count) state[%zu] = fmaf(sixth_h, sum_state[%zu], base_state[%zu]);\n",
                              index, index, index, index));
    }
    O_TRY(o_writer_cstr(writer, "            }\n\n            valid = valid"));
    for (index = 0u; index < state_capacity; ++index) {
        O_TRY(o_writer_format(writer, " && (%zuu >= active_state_count || isfinite(state[%zu]))", index, index));
    }
    O_TRY(o_writer_cstr(writer, ";\n"
                                "            ++scored_point_count;\n"));
    for (index = 0u; index < state_capacity; ++index) {
        O_TRY(o_writer_format(writer,
                              "            if (%zuu < active_state_count && !(allow_missing_observations && isnan(reference[%zu * trajectory_point_count + point]))) {\n"
                              "                ++scored_residual_count;\n"
                              "                const float error%zu = state[%zu] - reference[%zu * trajectory_point_count + point];\n"
                              "                squared_error = fmaf(error%zu, error%zu, squared_error);\n"
                              "            }\n",
                              index, index, index, index, index, index, index));
    }
    return o_writer_cstr(writer, "        }\n"
                                 "    }\n"
                                 "    valid = valid && expected_start == trajectory_point_count && scored_point_count != 0u && scored_residual_count != 0u;\n\n"
                                 "    mse_out[output_index] = valid && isfinite(squared_error)\n"
                                 "        ? squared_error / (float)scored_residual_count\n"
                                 "        : 3.402823466e+38F;\n"
                                 "}\n");
}

static OdezzaResult o_write_payload(
    OWriter *writer,
    size_t state_capacity,
    uint32_t constant_capacity,
    uint32_t system_capacity,
    uint32_t shared_patch_capacity,
    const OScoringCudaInfo *info,
    const unsigned char template_id[32]
) {
    O_TRY(o_write_template_declaration(writer, template_id, state_capacity, constant_capacity, system_capacity));
    return o_write_cuda_body(
        writer,
        state_capacity,
        constant_capacity,
        system_capacity,
        shared_patch_capacity,
        info
    );
}

static OdezzaResult o_hash_template(
    size_t state_capacity,
    uint32_t constant_capacity,
    uint32_t system_capacity,
    uint32_t shared_patch_capacity,
    uint32_t system_patch_capacity,
    const OScoringCudaInfo *info,
    unsigned char template_id[32]
) {
    static const unsigned char separator = 0u;
    static const unsigned char zero_id[32] = {0};
    OSha256 sha;
    OWriter writer;
    O_TRY(o_sha256_init(&sha));
    O_TRY(o_writer_init(&writer, NULL, 0u, &sha));
    O_TRY(o_write_manifest_canonical(
        &writer,
        state_capacity,
        constant_capacity,
        system_capacity,
        shared_patch_capacity,
        system_patch_capacity,
        info
    ));
    O_TRY(o_writer_write(&writer, &separator, 1u));
    O_TRY(o_write_payload(
        &writer,
        state_capacity,
        constant_capacity,
        system_capacity,
        shared_patch_capacity,
        info,
        zero_id
    ));
    return o_sha256_final(&sha, template_id);
}

static OdezzaResult o_hash_payload(
    size_t state_capacity,
    uint32_t constant_capacity,
    uint32_t system_capacity,
    uint32_t shared_patch_capacity,
    const OScoringCudaInfo *info,
    const unsigned char template_id[32],
    unsigned char source_sha256[32]
) {
    OSha256 sha;
    OWriter writer;
    O_TRY(o_sha256_init(&sha));
    O_TRY(o_writer_init(&writer, NULL, 0u, &sha));
    O_TRY(o_write_payload(
        &writer,
        state_capacity,
        constant_capacity,
        system_capacity,
        shared_patch_capacity,
        info,
        template_id
    ));
    return o_sha256_final(&sha, source_sha256);
}

static OdezzaResult o_write_complete(
    OWriter *writer,
    size_t state_capacity,
    uint32_t constant_capacity,
    uint32_t system_capacity,
    uint32_t shared_patch_capacity,
    uint32_t system_patch_capacity,
    const OScoringCudaInfo *info,
    const unsigned char template_id[32],
    const unsigned char source_sha256[32]
) {
    O_TRY(o_write_manifest_pretty(
        writer,
        state_capacity,
        constant_capacity,
        system_capacity,
        shared_patch_capacity,
        system_patch_capacity,
        info,
        template_id,
        source_sha256
    ));
    return o_write_payload(
        writer,
        state_capacity,
        constant_capacity,
        system_capacity,
        shared_patch_capacity,
        info,
        template_id
    );
}

OdezzaResult odezza_generate_scoring_cuda(
    size_t state_capacity,
    uint32_t constant_capacity,
    uint32_t system_capacity,
    uint32_t shared_patch_capacity,
    uint32_t system_patch_capacity,
    char *buffer,
    size_t buffer_size,
    size_t *buffer_bytes_written_ret
) {
    OScoringCudaInfo info;
    unsigned char template_id[32];
    unsigned char source_sha256[32];
    OWriter measure;
    OWriter output;
    size_t content_bytes;

    if (buffer_bytes_written_ret == NULL) {
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    }
    *buffer_bytes_written_ret = 0u;
    O_TRY(o_shape_validate(
        state_capacity,
        constant_capacity,
        system_capacity,
        shared_patch_capacity,
        system_patch_capacity,
        &info
    ));
    /* Digests have fixed-width encodings; sizing needs no payload hashing. */
    memset(template_id, 0, sizeof(template_id));
    memset(source_sha256, 0, sizeof(source_sha256));
    O_TRY(o_writer_init(&measure, NULL, 0u, NULL));
    O_TRY(o_write_complete(
        &measure,
        state_capacity,
        constant_capacity,
        system_capacity,
        shared_patch_capacity,
        system_patch_capacity,
        &info,
        template_id,
        source_sha256
    ));
    O_TRY(o_writer_bytes(&measure, &content_bytes));
    if (content_bytes == SIZE_MAX) {
        return ODEZZA_ERROR_OVERFLOW;
    }
    *buffer_bytes_written_ret = content_bytes;
    if (buffer == NULL) {
        return ODEZZA_SUCCESS;
    }
    if (buffer_size <= content_bytes) {
        return ODEZZA_ERROR_INSUFFICIENT_BUFFER;
    }
    O_TRY(o_hash_template(
        state_capacity,
        constant_capacity,
        system_capacity,
        shared_patch_capacity,
        system_patch_capacity,
        &info,
        template_id
    ));
    O_TRY(o_hash_payload(
        state_capacity,
        constant_capacity,
        system_capacity,
        shared_patch_capacity,
        &info,
        template_id,
        source_sha256
    ));
    O_TRY(o_writer_init(&output, buffer, buffer_size - 1u, NULL));
    O_TRY(o_write_complete(
        &output,
        state_capacity,
        constant_capacity,
        system_capacity,
        shared_patch_capacity,
        system_patch_capacity,
        &info,
        template_id,
        source_sha256
    ));
    buffer[content_bytes] = '\0';
    return ODEZZA_SUCCESS;
}
