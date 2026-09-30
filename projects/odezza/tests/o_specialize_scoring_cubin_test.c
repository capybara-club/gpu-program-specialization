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
#include "o_specialize_scoring_cubin.h"
#include "o_scoring_fixture.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int expect(OdezzaResult actual, OdezzaResult expected, const char *name) {
    if (actual == expected) return 1;
    fprintf(stderr, "%s: got result %d, expected %d\n", name, (int)actual, (int)expected);
    return 0;
}

static int test_patch_preflight(void) {
    unsigned char cubin[CUBIN_BYTES];
    unsigned char snapshot[CUBIN_BYTES];
    OdezzaScoringCubinInspection inspection;
    OdezzaScoringPrespecialization result;
    AlignedBytes workspace;
    size_t output_offsets[2] = {3072u, CUBIN_BYTES - 8u};
    size_t target_offsets[4] = {3500u, 3504u, 3508u, CUBIN_BYTES - 2u};
    size_t bad_offset = SIZE_MAX;
    uint8_t duplicate_registers[2] = {32u, 32u};
    unsigned int test;
    for (test = 0u; test < 9u; ++test) {
        initialize_inspection(cubin, &inspection);
        memcpy(snapshot, cubin, sizeof(snapshot));
        switch (test) {
        case 0u: inspection.output_materialization_file_offsets = output_offsets; break;
        case 1u: inspection.cleanup_file_offsets = &bad_offset; break;
        case 2u: inspection.target_table_file_offsets = target_offsets; break;
        case 3u: inspection.shared_instruction_count = SIZE_MAX; break;
        case 4u: inspection.arena_start_file_offset = SIZE_MAX; break;
        case 5u: inspection.register_count_file_offset_count = 1u; inspection.register_count_file_offsets = &bad_offset; break;
        case 6u: inspection.register_count_header_file_offset_count = 1u; inspection.register_count_header_file_offsets = &bad_offset; break;
        case 7u: inspection.dispatch_file_offsets = &bad_offset; break;
        default: inspection.available_register_count = 2u; inspection.available_registers = duplicate_registers; break;
        }
        if (!expect(odezza_prespecialize_scoring_cubin(cubin, sizeof(cubin), &inspection, 2u, 4u, NULL, 0u,
                                                       workspace.bytes, sizeof(workspace.bytes), &result),
                    ODEZZA_ERROR_FORMAT, "invalid patch metadata")) return 0;
        if (memcmp(snapshot, cubin, sizeof(cubin)) != 0) {
            fprintf(stderr, "invalid metadata case %u changed the CUBIN\n", test);
            return 0;
        }
    }
    return 1;
}

static int test_ast_contract(void) {
    static const uint8_t program_bytes[] = {ODEZZA_AST_STATE_F32,   0u, ODEZZA_AST_LITERAL_F32, 0u, 0u, 0x80u, 0xbfu,
                                            ODEZZA_AST_TOGGLE2_F32, 1u, ODEZZA_AST_RETURN_F32};
    static const uint8_t four_way_bytes[] = {ODEZZA_AST_STATE_F32,
                                             0u,
                                             ODEZZA_AST_STATE_F32,
                                             1u,
                                             ODEZZA_AST_CONSTANT_F32,
                                             0u,
                                             ODEZZA_AST_LITERAL_F32,
                                             0u,
                                             0u,
                                             0u,
                                             0u,
                                             ODEZZA_AST_TOGGLE4_F32,
                                             0u,
                                             1u,
                                             ODEZZA_AST_RETURN_F32};
    static const uint8_t computed_choice_bytes[] = {ODEZZA_AST_STATE_F32, 0u, ODEZZA_AST_STATE_F32,   1u, ODEZZA_AST_ADD_F32,
                                                    ODEZZA_AST_STATE_F32, 0u, ODEZZA_AST_TOGGLE2_F32, 0u, ODEZZA_AST_RETURN_F32};
    OdezzaAstAnalysis analysis;
    OdezzaAstProgram program = {program_bytes, sizeof(program_bytes)};
    OdezzaAstProgram four_way = {four_way_bytes, sizeof(four_way_bytes)};
    OdezzaAstProgram computed = {computed_choice_bytes, sizeof(computed_choice_bytes)};

    if (!expect(odezza_validate_ast_program(program, 2u, 2u, 2u, 1, &analysis), ODEZZA_SUCCESS, "two-way literal toggle")) return 0;
    if (analysis.required_toggle_bit_count != 2u || !analysis.contains_toggle || analysis.maximum_stack_depth != 2u) return 0;
    if (!expect(odezza_validate_ast_program(four_way, 2u, 2u, 2u, 1, &analysis), ODEZZA_SUCCESS, "four-way toggle")) return 0;
    if (analysis.maximum_stack_depth != 4u) return 0;
    if (!expect(odezza_validate_ast_program(program, 2u, 2u, 2u, 0, NULL), ODEZZA_ERROR_AST, "fixed toggle rejection")) return 0;
    if (!expect(odezza_validate_ast_program(computed, 2u, 2u, 2u, 1, NULL), ODEZZA_ERROR_AST, "computed toggle choice rejection")) return 0;
    return 1;
}

static int test_two_stage_specialization(void) {
    static const uint8_t fixed_bytes[] = {ODEZZA_AST_CONSTANT_F32, 0u, ODEZZA_AST_STATE_F32, 0u, ODEZZA_AST_MUL_F32, ODEZZA_AST_RETURN_F32};
    static const uint8_t candidate0_bytes[] =
        {ODEZZA_AST_STATE_F32, 0u, ODEZZA_AST_LITERAL_F32, 0u, 0u, 0x80u, 0xbfu, ODEZZA_AST_TOGGLE2_F32, 0u, ODEZZA_AST_STATE_F32, 1u, ODEZZA_AST_MUL_F32,
         ODEZZA_AST_RETURN_F32};
    static const uint8_t candidate1_bytes[] = {ODEZZA_AST_STATE_F32, 0u, ODEZZA_AST_CONSTANT_F32, 1u, ODEZZA_AST_ADD_F32, ODEZZA_AST_RETURN_F32};
    static const uint8_t candidate2_bytes[] = {
        ODEZZA_AST_STATE_F32, 0u, ODEZZA_AST_STATE_F32, 1u, ODEZZA_AST_CONSTANT_F32, 0u, ODEZZA_AST_CONSTANT_F32, 1u,
        ODEZZA_AST_TOGGLE4_F32, 0u, 11u, ODEZZA_AST_RETURN_F32
    };
    OdezzaScoringRhs fixed_rhs;
    OdezzaScoringRhs branch_rhs[3];
    OdezzaScoringSystem systems[3];
    OdezzaScoringCubinInspection inspection;
    OdezzaScoringPrespecialization prespecialization;
    OdezzaScoringSpecializationReport report;
    OdezzaScoringSpecializationReport fresh_report;
    AlignedBytes workspace;
    unsigned char cubin[CUBIN_BYTES];
    unsigned char corrupted[CUBIN_BYTES];
    unsigned char fresh[CUBIN_BYTES];
    unsigned char snapshot[CUBIN_BYTES];
    size_t workspace_size;
    uint32_t target0;
    uint32_t target1;

    initialize_inspection(cubin, &inspection);
    fixed_rhs.state_index = 0u;
    fixed_rhs.program.bytes = fixed_bytes;
    fixed_rhs.program.byte_count = sizeof(fixed_bytes);
    branch_rhs[0].state_index = 1u;
    branch_rhs[0].program.bytes = candidate0_bytes;
    branch_rhs[0].program.byte_count = sizeof(candidate0_bytes);
    branch_rhs[1].state_index = 1u;
    branch_rhs[1].program.bytes = candidate1_bytes;
    branch_rhs[1].program.byte_count = sizeof(candidate1_bytes);
    branch_rhs[2].state_index = 1u;
    branch_rhs[2].program.bytes = candidate2_bytes;
    branch_rhs[2].program.byte_count = sizeof(candidate2_bytes);
    systems[0].rhs = &branch_rhs[0];
    systems[0].rhs_count = 1u;
    systems[1].rhs = &branch_rhs[1];
    systems[1].rhs_count = 1u;
    systems[2].rhs = &branch_rhs[2];
    systems[2].rhs_count = 1u;

    if (!expect(odezza_scoring_specialization_workspace_size(&inspection, &workspace_size), ODEZZA_SUCCESS, "workspace size")) return 0;
    if (workspace_size > sizeof(workspace.bytes)) return 0;
    memcpy(snapshot, cubin, sizeof(cubin));
    {
        OdezzaScoringRhs invalid_fixed = branch_rhs[0];
        invalid_fixed.state_index = 0u;
        if (!expect(odezza_prespecialize_scoring_cubin(cubin, sizeof(cubin), &inspection, 2u, 2u, &invalid_fixed, 1u, workspace.bytes,
                                                       sizeof(workspace.bytes), &prespecialization),
                    ODEZZA_ERROR_AST, "fixed toggle transactional rejection"))
            return 0;
        if (memcmp(snapshot, cubin, sizeof(cubin)) != 0) return 0;
    }
    if (!expect(odezza_prespecialize_scoring_cubin(cubin, sizeof(cubin), &inspection, 2u, 2u, &fixed_rhs, 1u, workspace.bytes,
                                                   sizeof(workspace.bytes), &prespecialization),
                ODEZZA_SUCCESS, "fixed prespecialization"))
        return 0;
    if (prespecialization.fixed_state_mask[0] != 1u || prespecialization.fixed_rhs_count != 1u || prespecialization.shared_instruction_count == 0u ||
        prespecialization.high_water_register != inspection.register_count)
        return 0;

    memcpy(snapshot, cubin, sizeof(cubin));
    {
        OdezzaScoringRhs invalid_rhs = branch_rhs[0];
        OdezzaScoringSystem invalid_system;
        invalid_rhs.state_index = 0u;
        invalid_system.rhs = &invalid_rhs;
        invalid_system.rhs_count = 1u;
        if (!expect(odezza_specialize_scoring_cubin_systems(cubin, sizeof(cubin), &inspection, &prespecialization, 12u, &invalid_system, 1u, workspace.bytes,
                                                            sizeof(workspace.bytes), NULL),
                    ODEZZA_ERROR_STATE_LAYOUT, "candidate coverage transactional rejection"))
            return 0;
        if (memcmp(snapshot, cubin, sizeof(cubin)) != 0) return 0;
    }

    memcpy(corrupted, cubin, sizeof(cubin));
    corrupted[17] ^= 1u;
    if (!expect(odezza_specialize_scoring_cubin_systems(corrupted, sizeof(corrupted), &inspection, &prespecialization, 12u, systems, 3u, workspace.bytes,
                                                        sizeof(workspace.bytes), NULL),
                ODEZZA_ERROR_FORMAT, "prespecialized CUBIN identity"))
        return 0;

    memcpy(fresh, cubin, sizeof(fresh));
    if (!expect(o_specialize_scoring_cubin_systems_fresh(fresh, sizeof(fresh), &inspection, &prespecialization, 12u, systems, 3u, workspace.bytes,
                                                         sizeof(workspace.bytes), &fresh_report),
                ODEZZA_SUCCESS, "trusted fresh-copy specialization"))
        return 0;

    if (!expect(odezza_specialize_scoring_cubin_systems(cubin, sizeof(cubin), &inspection, &prespecialization, 12u, systems, 3u, workspace.bytes,
                                                        sizeof(workspace.bytes), &report),
                ODEZZA_SUCCESS, "candidate specialization"))
        return 0;
    if (report.system_count != 3u || report.specialized_rhs_count != 3u || report.body_instruction_count == 0u || report.maximum_body_instruction_count == 0u)
        return 0;
    if (report.high_water_register < inspection.register_count) return 0;
    if (memcmp(&report, &fresh_report, sizeof(report)) != 0 || memcmp(cubin, fresh, sizeof(cubin)) != 0) return 0;
    target0 = (uint32_t)cubin[3500] | ((uint32_t)cubin[3501] << 8u) | ((uint32_t)cubin[3502] << 16u) | ((uint32_t)cubin[3503] << 24u);
    target1 = (uint32_t)cubin[3504] | ((uint32_t)cubin[3505] << 8u) | ((uint32_t)cubin[3506] << 16u) | ((uint32_t)cubin[3507] << 24u);
    if (target0 == target1 || target0 != 1024u) return 0;
    {
        uint32_t found_masks = 0u;
        size_t offset;
        for (offset = inspection.arena_start_file_offset; offset < inspection.arena_end_file_offset; offset += sizeof(OdezzaScoringInstruction)) {
            uint64_t word0;
            uint64_t word1;
            memcpy(&word0, cubin + offset, sizeof(word0));
            memcpy(&word1, cubin + offset + sizeof(word0), sizeof(word1));
            if ((word0 & 0xffffu) == UINT64_C(0x7812)) {
                uint32_t mask = (uint32_t)(word0 >> 32u);
                if (((word1 >> 40u) & 0xfu) != 12u) return 0;
                if (mask == 1u) found_masks |= 1u;
                if (mask == 2048u) found_masks |= 2u;
            }
        }
        if (found_masks != 3u) return 0;
    }
    return 1;
}

static int specialize_real_fixture_work(
    const char *input_path,
    const char *output_path,
    FILE **file_ret,
    unsigned char **cubin_ret,
    void **inspection_arena_ret,
    void **workspace_ret
) {
    static const uint8_t fixed_bytes[] = {ODEZZA_AST_CONSTANT_F32,
                                          0u,
                                          ODEZZA_AST_STATE_F32,
                                          0u,
                                          ODEZZA_AST_MUL_F32,
                                          ODEZZA_AST_CONSTANT_F32,
                                          1u,
                                          ODEZZA_AST_STATE_F32,
                                          0u,
                                          ODEZZA_AST_MUL_F32,
                                          ODEZZA_AST_STATE_F32,
                                          1u,
                                          ODEZZA_AST_MUL_F32,
                                          ODEZZA_AST_SUB_F32,
                                          ODEZZA_AST_RETURN_F32};
    static const uint8_t candidate0_bytes[] = {ODEZZA_AST_CONSTANT_F32,
                                               2u,
                                               ODEZZA_AST_STATE_F32,
                                               0u,
                                               ODEZZA_AST_MUL_F32,
                                               ODEZZA_AST_STATE_F32,
                                               1u,
                                               ODEZZA_AST_MUL_F32,
                                               ODEZZA_AST_CONSTANT_F32,
                                               3u,
                                               ODEZZA_AST_STATE_F32,
                                               1u,
                                               ODEZZA_AST_MUL_F32,
                                               ODEZZA_AST_SUB_F32,
                                               ODEZZA_AST_RETURN_F32};
    static const uint8_t candidate1_bytes[] = {ODEZZA_AST_CONSTANT_F32,
                                               2u,
                                               ODEZZA_AST_STATE_F32,
                                               0u,
                                               ODEZZA_AST_LITERAL_F32,
                                               0u,
                                               0u,
                                               0x80u,
                                               0x3fu,
                                               ODEZZA_AST_TOGGLE2_F32,
                                               0u,
                                               ODEZZA_AST_DIV_F32,
                                               ODEZZA_AST_CONSTANT_F32,
                                               3u,
                                               ODEZZA_AST_STATE_F32,
                                               1u,
                                               ODEZZA_AST_MUL_F32,
                                               ODEZZA_AST_SUB_F32,
                                               ODEZZA_AST_RETURN_F32};
    long measured;
    size_t cubin_size;
    size_t inspection_arena_size;
    size_t workspace_size;
    const OdezzaScoringCubinInspection *inspection = NULL;
    OdezzaScoringPrespecialization prespecialization;
    OdezzaScoringSpecializationReport report;
    OdezzaScoringRhs fixed_rhs;
    OdezzaScoringRhs candidate_rhs[2];
    OdezzaScoringSystem systems[2];
    *file_ret = fopen(input_path, "rb");
    if (*file_ret == NULL || fseek(*file_ret, 0, SEEK_END) != 0) return 0;
    measured = ftell(*file_ret);
    if (measured <= 0 || fseek(*file_ret, 0, SEEK_SET) != 0) return 0;
    cubin_size = (size_t)measured;
    *cubin_ret = (unsigned char *)malloc(cubin_size);
    if (*cubin_ret == NULL || fread(*cubin_ret, 1u, cubin_size, *file_ret) != cubin_size) return 0;
    if (fclose(*file_ret) != 0) {
        *file_ret = NULL;
        return 0;
    }
    *file_ret = NULL;

    if (odezza_inspect_scoring_cubin(*cubin_ret, cubin_size, NULL, 0u, &inspection_arena_size, NULL) != ODEZZA_SUCCESS) return 0;
    *inspection_arena_ret = malloc(inspection_arena_size);
    if (*inspection_arena_ret == NULL ||
        odezza_inspect_scoring_cubin(*cubin_ret, cubin_size, *inspection_arena_ret, inspection_arena_size, &inspection_arena_size, &inspection) !=
            ODEZZA_SUCCESS ||
        odezza_scoring_specialization_workspace_size(inspection, &workspace_size) != ODEZZA_SUCCESS)
        return 0;
    *workspace_ret = malloc(workspace_size);
    if (*workspace_ret == NULL) return 0;

    fixed_rhs.state_index = 0u;
    fixed_rhs.program.bytes = fixed_bytes;
    fixed_rhs.program.byte_count = sizeof(fixed_bytes);
    candidate_rhs[0].state_index = 1u;
    candidate_rhs[0].program.bytes = candidate0_bytes;
    candidate_rhs[0].program.byte_count = sizeof(candidate0_bytes);
    candidate_rhs[1].state_index = 1u;
    candidate_rhs[1].program.bytes = candidate1_bytes;
    candidate_rhs[1].program.byte_count = sizeof(candidate1_bytes);
    systems[0].rhs = &candidate_rhs[0];
    systems[0].rhs_count = 1u;
    systems[1].rhs = &candidate_rhs[1];
    systems[1].rhs_count = 1u;
    if (odezza_prespecialize_scoring_cubin(*cubin_ret, cubin_size, inspection, 2u, 4u, &fixed_rhs, 1u, *workspace_ret, workspace_size,
                                           &prespecialization) != ODEZZA_SUCCESS)
        return 0;
    if (odezza_specialize_scoring_cubin_systems(*cubin_ret, cubin_size, inspection, &prespecialization, 5u, systems, 2u, *workspace_ret, workspace_size,
                                                &report) != ODEZZA_SUCCESS)
        return 0;

    *file_ret = fopen(output_path, "wb");
    if (*file_ret == NULL || fwrite(*cubin_ret, 1u, cubin_size, *file_ret) != cubin_size) return 0;
    if (fclose(*file_ret) != 0) {
        *file_ret = NULL;
        return 0;
    }
    *file_ret = NULL;
    printf(
        "real CUBIN specialization: %zu shared instructions, %zu systems, %zu body instructions, %u registers\n",
        prespecialization.shared_instruction_count,
        report.system_count,
        report.body_instruction_count,
        report.register_count
    );
    return 1;
}

static int specialize_real_fixture(const char *input_path, const char *output_path) {
    FILE *file = NULL;
    unsigned char *cubin = NULL;
    void *inspection_arena = NULL;
    void *workspace = NULL;
    int success = specialize_real_fixture_work(input_path, output_path, &file, &cubin, &inspection_arena, &workspace);
    if (file != NULL) fclose(file);
    if (workspace != NULL) free(workspace);
    if (inspection_arena != NULL) free(inspection_arena);
    if (cubin != NULL) free(cubin);
    return success;
}

static int compile_real_fixture_work(
    const char *source_path,
    const char *output_path,
    FILE **file_ret,
    char **source_ret,
    unsigned char **cubin_ret,
    OdezzaNvrtcCompilation **compilation_ret
) {
    static const char *const options[] = {"--std=c++17", "--gpu-architecture=sm_120", "--use_fast_math", "--ptxas-options=-O3"};
    long measured;
    OdezzaResult compilation_result;
    size_t cubin_size;

    *file_ret = fopen(source_path, "rb");
    if (*file_ret == NULL || fseek(*file_ret, 0, SEEK_END) != 0) return 0;
    measured = ftell(*file_ret);
    if (measured <= 0 || fseek(*file_ret, 0, SEEK_SET) != 0) return 0;
    *source_ret = (char *)malloc((size_t)measured + 1u);
    if (*source_ret == NULL || fread(*source_ret, 1u, (size_t)measured, *file_ret) != (size_t)measured) return 0;
    (*source_ret)[(size_t)measured] = '\0';
    if (fclose(*file_ret) != 0) {
        *file_ret = NULL;
        return 0;
    }
    *file_ret = NULL;
    if (odezza_nvrtc_compilation_create(*source_ret, source_path, options, sizeof(options) / sizeof(options[0]), compilation_ret) != ODEZZA_SUCCESS ||
        odezza_nvrtc_compilation_result(*compilation_ret, &compilation_result) != ODEZZA_SUCCESS || compilation_result != ODEZZA_SUCCESS ||
        odezza_nvrtc_compilation_cubin_size(*compilation_ret, &cubin_size) != ODEZZA_SUCCESS)
        return 0;
    *cubin_ret = (unsigned char *)malloc(cubin_size);
    if (*cubin_ret == NULL || odezza_nvrtc_compilation_write_cubin(*compilation_ret, *cubin_ret, cubin_size) != ODEZZA_SUCCESS) return 0;
    *file_ret = fopen(output_path, "wb");
    if (*file_ret == NULL || fwrite(*cubin_ret, 1u, cubin_size, *file_ret) != cubin_size) return 0;
    if (fclose(*file_ret) != 0) {
        *file_ret = NULL;
        return 0;
    }
    *file_ret = NULL;
    return 1;
}

static int compile_real_fixture(const char *source_path, const char *output_path) {
    FILE *file = NULL;
    char *source = NULL;
    unsigned char *cubin = NULL;
    OdezzaNvrtcCompilation *compilation = NULL;
    int success = compile_real_fixture_work(source_path, output_path, &file, &source, &cubin, &compilation);
    if (file != NULL) fclose(file);
    if (compilation != NULL) (void)odezza_nvrtc_compilation_destroy(compilation);
    if (cubin != NULL) free(cubin);
    if (source != NULL) free(source);
    return success;
}

int main(int argc, char **argv) {
    if (argc == 3) return specialize_real_fixture(argv[1], argv[2]) ? 0 : 1;
    if (argc == 5 && strcmp(argv[1], "--nvrtc") == 0) {
        if (!compile_real_fixture(argv[2], argv[3])) return 1;
        return specialize_real_fixture(argv[3], argv[4]) ? 0 : 1;
    }
    if (argc != 1) {
        fprintf(
            stderr,
            "usage: %s [input.cubin output.cubin | --nvrtc input.cu template.cubin output.cubin]\n",
            argv[0]
        );
        return 2;
    }
    if (!test_ast_contract()) return 1;
    if (!test_patch_preflight()) return 1;
    if (!test_two_stage_specialization()) return 1;
    puts("C AST and two-stage scoring specialization contracts: verified");
    return 0;
}
