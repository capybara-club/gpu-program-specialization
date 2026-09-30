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
#include "src/internal.h"

#include <stdio.h>
#include <string.h>

int
main(void) {
    static const SecantAstInstruction program[] = {
        secant_ast_encode_static_column_input_f32(0),
        secant_ast_encode_static_column_input_f32(1),
        secant_ast_encode_mul_f32,
        secant_ast_encode_static_column_input_f32(2),
        secant_ast_encode_add_f32,
        secant_ast_encode_return_f32
    };
    SecantSRNodeInfo nodes[8];
    size_t program_bytes = 0u;
    size_t num_nodes = 0u;
    size_t num_leaves = 0u;
    size_t num_constants = 0u;
    uint32_t complexity = 0u;
    uint16_t depth = 0u;
    uint64_t fingerprint = 0u;
    SecantSRProgramFeatures features;
    SecantSRResult result;
    SecantSRIndividual individual;
    char expression[64];
    size_t expression_size = 0u;

    result = secant_sr_program_annotate(
        program,
        sizeof(program),
        NULL,
        0u,
        8u,
        32u,
        nodes,
        8u,
        &program_bytes,
        &num_nodes,
        &num_leaves,
        &num_constants,
        &complexity,
        &depth,
        &fingerprint);
    if (result != SECANT_SR_SUCCESS) {
        fprintf(stderr, "annotate failed: %s\n", secant_sr_result_to_string(result));
        return 1;
    }
    if (program_bytes != sizeof(program) || num_nodes != 5u || num_leaves != 3u || num_constants != 0u ||
        complexity != 5u || depth != 3u || fingerprint == 0u) {
        fprintf(stderr, "unexpected summary bytes=%zu nodes=%zu leaves=%zu constants=%zu complexity=%u depth=%u\n",
            program_bytes, num_nodes, num_leaves, num_constants, complexity, depth);
        return 1;
    }
    {
        static const SecantAstInstruction trailing_program[] = {
            secant_ast_encode_static_column_input_f32(0),
            secant_ast_encode_return_f32,
            secant_ast_encode_static_column_input_f32(1)
        };

        if (secant_sr_program_features_get(program, sizeof(program), NULL, 0u, &features) != SECANT_SR_SUCCESS ||
            features.static_column_mask[0] != UINT64_C(7) || features.static_column_mask[1] != 0u ||
            features.num_unique_static_columns != 3u || features.num_static_column_occurrences != 3u ||
            features.num_constants != 0u || features.num_unary_operations != 0u ||
            features.num_binary_operations != 2u || features.num_ternary_operations != 0u ||
            features.num_transcendental_operations != 0u) {
            fprintf(stderr, "unexpected on-demand features\n");
            return 1;
        }
        if (secant_sr_program_features_get(trailing_program, sizeof(trailing_program), NULL, 0u, &features) !=
            SECANT_SR_ERROR_BAD_PROGRAM) {
            fprintf(stderr, "on-demand features accepted trailing instructions\n");
            return 1;
        }
    }
    if (nodes[2].subtree_offset != 0u || nodes[2].subtree_bytes != 5u ||
        nodes[2].subtree_first_node != 0u || nodes[2].subtree_nodes != 3u || nodes[2].depth != 2u) {
        fprintf(stderr, "unexpected multiply sidecar\n");
        return 1;
    }
    if (nodes[4].subtree_offset != 0u || nodes[4].subtree_bytes != 8u ||
        nodes[4].subtree_first_node != 0u || nodes[4].subtree_nodes != 5u || nodes[4].depth != 3u) {
        fprintf(stderr, "unexpected root sidecar\n");
        return 1;
    }
    memset(&individual, 0, sizeof(individual));
    individual.program = program;
    individual.nodes = nodes;
    individual.program_bytes = program_bytes;
    individual.num_nodes = num_nodes;
    if (secant_sr_individual_format(&individual, expression, sizeof(expression), &expression_size) !=
            SECANT_SR_SUCCESS ||
        strcmp(expression, "((x0 * x1) + x2)") != 0 || expression_size != strlen(expression) + 1u) {
        fprintf(stderr, "unexpected expression: %s\n", expression);
        return 1;
    }
    {
        static const SecantAstInstruction square_f32[] = {
            secant_ast_encode_routine_arg_f32(0u),
            secant_ast_encode_routine_arg_f32(0u),
            secant_ast_encode_mul_f32,
            secant_ast_encode_return_f32
        };
        static const SecantAstInstruction cube_f32[] = {
            secant_ast_encode_routine_arg_f32(0u),
            secant_ast_encode_routine_arg_f32(0u),
            secant_ast_encode_mul_f32,
            secant_ast_encode_routine_arg_f32(0u),
            secant_ast_encode_mul_f32,
            secant_ast_encode_return_f32
        };
        static const SecantSRRoutine routines[] = {
            {square_f32, "square", 1u, 1u, 0u},
            {cube_f32, "cube", 1u, 1u, 0u}
        };
        static const SecantAstInstruction routine_program[] = {
            secant_ast_encode_static_column_input_f32(0u),
            secant_ast_encode_routine_f32(0u),
            secant_ast_encode_routine_f32(1u),
            secant_ast_encode_return_f32
        };
        SecantSRNodeInfo routine_nodes[3];
        SecantSRIndividual routine_individual;

        result = secant_sr_program_annotate(
            routine_program,
            sizeof(routine_program),
            routines,
            sizeof(routines) / sizeof(routines[0]),
            8u,
            32u,
            routine_nodes,
            sizeof(routine_nodes) / sizeof(routine_nodes[0]),
            &program_bytes,
            &num_nodes,
            &num_leaves,
            &num_constants,
            &complexity,
            &depth,
            &fingerprint);
        if (result != SECANT_SR_SUCCESS || program_bytes != sizeof(routine_program) || num_nodes != 3u ||
            num_leaves != 1u || num_constants != 0u || complexity != 3u || depth != 3u ||
            secant_sr_program_features_get(
                routine_program,
                sizeof(routine_program),
                routines,
                sizeof(routines) / sizeof(routines[0]),
                &features) != SECANT_SR_SUCCESS ||
            features.num_unary_operations != 2u || features.num_binary_operations != 0u) {
            fprintf(stderr, "unexpected routine annotations\n");
            return 1;
        }
        memset(&routine_individual, 0, sizeof(routine_individual));
        routine_individual.program = routine_program;
        routine_individual.nodes = routine_nodes;
        routine_individual.routines = routines;
        routine_individual.program_bytes = program_bytes;
        routine_individual.num_nodes = num_nodes;
        routine_individual.num_routines = sizeof(routines) / sizeof(routines[0]);
        if (secant_sr_individual_format(
                &routine_individual, expression, sizeof(expression), &expression_size) != SECANT_SR_SUCCESS ||
            strcmp(expression, "cube(square(x0))") != 0) {
            fprintf(stderr, "unexpected routine expression: %s\n", expression);
            return 1;
        }
        {
            static const SecantAstInstruction feynman_test_9[] = {
                secant_ast_encode_constant_f32_bits(0xc0cccccd),
                secant_ast_encode_static_column_input_f32(0u),
                secant_ast_encode_routine_f32(0u),
                secant_ast_encode_routine_f32(0u),
                secant_ast_encode_mul_f32,
                secant_ast_encode_static_column_input_f32(1u),
                secant_ast_encode_routine_f32(1u),
                secant_ast_encode_static_column_input_f32(1u),
                secant_ast_encode_routine_f32(0u),
                secant_ast_encode_mul_f32,
                secant_ast_encode_div_f32,
                secant_ast_encode_static_column_input_f32(2u),
                secant_ast_encode_static_column_input_f32(3u),
                secant_ast_encode_mul_f32,
                secant_ast_encode_routine_f32(0u),
                secant_ast_encode_mul_f32,
                secant_ast_encode_static_column_input_f32(2u),
                secant_ast_encode_static_column_input_f32(3u),
                secant_ast_encode_add_f32,
                secant_ast_encode_mul_f32,
                secant_ast_encode_static_column_input_f32(4u),
                secant_ast_encode_routine_f32(1u),
                secant_ast_encode_static_column_input_f32(4u),
                secant_ast_encode_routine_f32(0u),
                secant_ast_encode_mul_f32,
                secant_ast_encode_div_f32,
                secant_ast_encode_return_f32
            };
            SecantSRNodeInfo feynman_nodes[26];

            result = secant_sr_program_annotate(
                feynman_test_9,
                sizeof(feynman_test_9),
                routines,
                sizeof(routines) / sizeof(routines[0]),
                16u,
                40u,
                feynman_nodes,
                sizeof(feynman_nodes) / sizeof(feynman_nodes[0]),
                &program_bytes,
                &num_nodes,
                &num_leaves,
                &num_constants,
                &complexity,
                &depth,
                &fingerprint);
            if (result != SECANT_SR_SUCCESS || program_bytes != sizeof(feynman_test_9) || num_nodes != 26u ||
                num_leaves != 10u || num_constants != 1u || complexity != 30u || depth != 8u) {
                fprintf(stderr,
                    "feynman_test_9 annotation result=%s bytes=%zu/%zu nodes=%zu leaves=%zu constants=%zu "
                    "complexity=%u depth=%u\n",
                    secant_sr_result_to_string(result),
                    program_bytes,
                    sizeof(feynman_test_9),
                    num_nodes,
                    num_leaves,
                    num_constants,
                    complexity,
                    depth);
                return 1;
            }
        }
    }
    {
        static const SecantAstInstruction fixed_constant_program[] = {
            secant_ast_encode_static_column_input_f32(0),
            secant_ast_encode_constant_f32_bits(SECANT_F32_BITS_ONE),
            secant_ast_encode_add_f32,
            secant_ast_encode_return_f32
        };
        static const SecantAstInstruction expected_dynamic_program[] = {
            secant_ast_encode_static_column_input_f32(0),
            secant_ast_encode_dynamic_constant_input_f32(3),
            secant_ast_encode_add_f32,
            secant_ast_encode_return_f32
        };
        SecantAstInstruction rewritten[sizeof(fixed_constant_program)];
        size_t rewritten_size = 0u;

        result = secant_sr_program_leaf_replace_copy(
            fixed_constant_program,
            sizeof(fixed_constant_program),
            2u,
            SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_CONSTANT_INPUT_F32,
            3u,
            NULL,
            0u,
            &rewritten_size);
        if (result != SECANT_SR_SUCCESS || rewritten_size != sizeof(expected_dynamic_program)) {
            fprintf(stderr, "rewrite measure failed\n");
            return 1;
        }
        result = secant_sr_program_leaf_replace_copy(
            fixed_constant_program,
            sizeof(fixed_constant_program),
            2u,
            SECANT_AST_INSTRUCTION_TYPE_DYNAMIC_CONSTANT_INPUT_F32,
            3u,
            rewritten,
            sizeof(rewritten),
            &rewritten_size);
        if (result != SECANT_SR_SUCCESS ||
            memcmp(rewritten, expected_dynamic_program, sizeof(expected_dynamic_program)) != 0 ||
            fixed_constant_program[2] != SECANT_AST_INSTRUCTION_TYPE_CONSTANT_BITS_F32) {
            fprintf(stderr, "rewrite output failed\n");
            return 1;
        }
    }
    return 0;
}
