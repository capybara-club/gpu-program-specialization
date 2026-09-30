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
#include "ssid_internal.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SSID_GP_CONSTANT_BITS_F32 0x81u
#define SSID_GP_RETURN_F32 0x83u
#define SSID_GP_ADD_F32 0x85u
#define SSID_GP_SUB_F32 0x86u
#define SSID_GP_MUL_F32 0x87u
#define SSID_GP_DIV_F32 0x88u
#define SSID_GP_NEG_F32 0x89u
#define SSID_GP_SQRT_F32 0x8au
#define SSID_GP_RCP_F32 0x8bu
#define SSID_GP_ABS_F32 0x8cu
#define SSID_GP_MIN_F32 0x8du
#define SSID_GP_MAX_F32 0x8eu
#define SSID_GP_SIN_F32 0x90u
#define SSID_GP_COS_F32 0x91u
#define SSID_GP_EX2_F32 0x92u
#define SSID_GP_LG2_F32 0x93u
#define SSID_GP_RSQRT_F32 0x94u
#define SSID_GP_TANH_F32 0x95u
#define SSID_GP_STATIC_INPUT_F32 0xb6u
#define SSID_GP_EXP_F32 0xbau
#define SSID_GP_LOG_F32 0xbbu

#define SSID_GP_ORIGIN_INITIAL 0u
#define SSID_GP_ORIGIN_ELITE 1u
#define SSID_GP_ORIGIN_SUBTREE_CROSSOVER 2u
#define SSID_GP_ORIGIN_WHOLE_SITE_CROSSOVER 3u
#define SSID_GP_ORIGIN_SUBTREE_MUTATION 4u
#define SSID_GP_ORIGIN_POINT_MUTATION 5u
#define SSID_GP_ORIGIN_RANDOM 6u

typedef struct ssid_gp_node {
    uint32_t instruction_offset;
    uint32_t instruction_size;
    uint32_t subtree_offset;
    uint32_t subtree_size;
    uint32_t subtree_nodes;
    uint16_t depth;
    uint8_t arity;
    uint8_t opcode;
} ssid_gp_node;

typedef struct ssid_gp_individual {
    float mse;
    float objective;
    uint32_t complexity;
    uint32_t winner_setting;
    uint32_t parent_a;
    uint32_t parent_b;
    uint8_t origin;
} ssid_gp_individual;

typedef struct ssid_gp_population {
    ssid_genome_desc *genomes;
    ssid_ast_desc *asts;
    uint8_t *programs;
    ssid_gp_node *nodes;
    uint32_t *node_counts;
    ssid_gp_individual *individuals;
    float *constants;
    uint32_t *bindings;
} ssid_gp_population;

struct ssid_gp {
    const ssid_template *template_value;
    ssid_gp_config config;
    uint8_t *unary_operations;
    uint8_t *binary_operations;
    uint32_t unary_operation_count;
    uint32_t binary_operation_count;
    uint32_t site_count;
    uint32_t input_count;
    uint32_t bank_slot_count;
    uint32_t module_capacity;
    uint32_t batch_count;
    uint32_t current_population;
    uint32_t generation;
    int population_scored;
    uint64_t rng_state;
    ssid_gp_population populations[2];
    float *winner_mse;
    uint32_t *winner_settings;
    uint32_t *elite_indices;
    uint8_t *best_programs;
    ssid_ast_desc *best_asts;
    ssid_genome_desc best_genome;
    float *best_constants;
    uint32_t *best_bindings;
    float best_mse;
    float best_objective;
    uint32_t best_generation;
    uint32_t best_complexity;
    int has_best;
};

static int ssid_gp_mul_size(size_t lhs, size_t rhs, size_t *result) {
    if (lhs != 0u && rhs > SIZE_MAX / lhs) return 0;
    *result = lhs * rhs;
    return 1;
}

static uint64_t ssid_gp_mix64(uint64_t value) {
    value += UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30u)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27u)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31u);
}

static uint64_t ssid_gp_random_u64(ssid_gp *gp) {
    gp->rng_state = ssid_gp_mix64(gp->rng_state);
    return gp->rng_state;
}

static uint32_t ssid_gp_random_bounded(ssid_gp *gp, uint32_t bound) {
    return bound == 0u ? 0u : (uint32_t)(ssid_gp_random_u64(gp) % bound);
}

static float ssid_gp_random_unit(ssid_gp *gp) {
    return (float)((ssid_gp_random_u64(gp) >> 40u) * (1.0 / 16777216.0));
}

static float ssid_gp_hash_unit(uint64_t value) {
    return (float)((ssid_gp_mix64(value) >> 40u) * (1.0 / 16777216.0));
}

static int ssid_gp_opcode_arity(uint8_t opcode) {
    switch (opcode) {
        case SSID_GP_NEG_F32:
        case SSID_GP_SQRT_F32:
        case SSID_GP_RCP_F32:
        case SSID_GP_ABS_F32:
        case SSID_GP_SIN_F32:
        case SSID_GP_COS_F32:
        case SSID_GP_EX2_F32:
        case SSID_GP_LG2_F32:
        case SSID_GP_RSQRT_F32:
        case SSID_GP_TANH_F32:
        case SSID_GP_EXP_F32:
        case SSID_GP_LOG_F32:
            return 1;
        case SSID_GP_ADD_F32:
        case SSID_GP_SUB_F32:
        case SSID_GP_MUL_F32:
        case SSID_GP_DIV_F32:
        case SSID_GP_MIN_F32:
        case SSID_GP_MAX_F32:
            return 2;
        default:
            return 0;
    }
}

static int ssid_gp_grammar_valid(const ssid_gp_grammar *grammar) {
    uint32_t index;
    if (grammar == NULL || grammar->binary_operation_count == 0u || grammar->binary_operations == NULL ||
        (grammar->unary_operation_count != 0u && grammar->unary_operations == NULL)) return 0;
    for (index = 0u; index < grammar->unary_operation_count; ++index) {
        if (ssid_gp_opcode_arity(grammar->unary_operations[index]) != 1) return 0;
    }
    for (index = 0u; index < grammar->binary_operation_count; ++index) {
        if (ssid_gp_opcode_arity(grammar->binary_operations[index]) != 2) return 0;
    }
    return 1;
}

static int ssid_gp_config_valid(const ssid_gp_config *config) {
    float variation;
    if (config == NULL || config->population_size == 0u || config->settings_per_genome == 0u ||
        config->state_count == 0u || config->constant_count == 0u || config->max_nodes_per_ast == 0u ||
        config->max_program_bytes_per_ast < 4u || config->initial_max_nodes == 0u ||
        config->initial_max_nodes > config->max_nodes_per_ast || config->max_depth == 0u || config->max_depth > 128u ||
        config->tournament_size == 0u || config->elite_count > config->population_size ||
        !isfinite(config->subtree_crossover_probability) ||
        !isfinite(config->whole_site_crossover_probability) ||
        !isfinite(config->subtree_mutation_probability) ||
        !isfinite(config->point_mutation_probability) ||
        !isfinite(config->binding_keep_probability) ||
        !isfinite(config->constant_mutation_scale) ||
        !isfinite(config->parsimony_coefficient) ||
        !isfinite(config->minimum_valid_mse) ||
        config->subtree_crossover_probability < 0.0f ||
        config->whole_site_crossover_probability < 0.0f ||
        config->subtree_mutation_probability < 0.0f ||
        config->point_mutation_probability < 0.0f ||
        config->binding_keep_probability < 0.0f || config->binding_keep_probability > 1.0f ||
        config->constant_mutation_scale < 0.0f || config->parsimony_coefficient < 0.0f ||
        config->minimum_valid_mse < 0.0f) return 0;
    variation = config->subtree_crossover_probability + config->whole_site_crossover_probability +
        config->subtree_mutation_probability + config->point_mutation_probability;
    return isfinite(variation) && variation <= 1.0f;
}

static uint8_t *ssid_gp_program(ssid_gp *gp, ssid_gp_population *population, uint32_t genome, uint32_t site) {
    size_t ast_index = (size_t)genome * gp->site_count + site;
    return population->programs + ast_index * gp->config.max_program_bytes_per_ast;
}

static const uint8_t *ssid_gp_program_const(const ssid_gp *gp, const ssid_gp_population *population, uint32_t genome, uint32_t site) {
    size_t ast_index = (size_t)genome * gp->site_count + site;
    return population->programs + ast_index * gp->config.max_program_bytes_per_ast;
}

static ssid_gp_node *ssid_gp_nodes(ssid_gp *gp, ssid_gp_population *population, uint32_t genome, uint32_t site) {
    size_t ast_index = (size_t)genome * gp->site_count + site;
    return population->nodes + ast_index * gp->config.max_nodes_per_ast;
}

static const ssid_gp_node *ssid_gp_nodes_const(const ssid_gp *gp, const ssid_gp_population *population, uint32_t genome, uint32_t site) {
    size_t ast_index = (size_t)genome * gp->site_count + site;
    return population->nodes + ast_index * gp->config.max_nodes_per_ast;
}

static int ssid_gp_annotate(ssid_gp *gp, ssid_gp_population *population, uint32_t genome, uint32_t site) {
    size_t ast_index = (size_t)genome * gp->site_count + site;
    ssid_ast_desc *ast = population->asts + ast_index;
    const uint8_t *program = ssid_gp_program_const(gp, population, genome, site);
    ssid_gp_node *nodes = ssid_gp_nodes(gp, population, genome, site);
    uint32_t stack[128];
    uint32_t stack_count = 0u;
    uint32_t offset = 0u;
    uint32_t node_count = 0u;
    int returned = 0;
    while (offset < ast->byte_count) {
        uint32_t instruction_offset = offset;
        uint8_t opcode = program[offset++];
        uint32_t instruction_size = 1u;
        int arity = ssid_gp_opcode_arity(opcode);
        if (opcode == SSID_GP_RETURN_F32) {
            if (stack_count != 1u || offset != ast->byte_count) return SSID_INVALID_ARGUMENT;
            returned = 1;
            break;
        }
        if (node_count >= gp->config.max_nodes_per_ast) return SSID_OUT_OF_RANGE;
        memset(nodes + node_count, 0, sizeof(*nodes));
        nodes[node_count].instruction_offset = instruction_offset;
        nodes[node_count].opcode = opcode;
        if (opcode == SSID_GP_STATIC_INPUT_F32) {
            uint32_t input_begin = gp->template_value->site_input_offsets[site];
            uint32_t input_count = gp->template_value->site_input_counts[site];
            uint32_t input;
            if (offset >= ast->byte_count) return SSID_INVALID_ARGUMENT;
            input = program[offset++];
            instruction_size = 2u;
            if (input < input_begin || input >= input_begin + input_count) return SSID_INVALID_ARGUMENT;
            nodes[node_count].subtree_offset = instruction_offset;
            nodes[node_count].subtree_size = instruction_size;
            nodes[node_count].subtree_nodes = 1u;
            nodes[node_count].depth = 1u;
            nodes[node_count].arity = 0u;
        } else if (opcode == SSID_GP_CONSTANT_BITS_F32) {
            if (ast->byte_count - offset < 4u) return SSID_INVALID_ARGUMENT;
            offset += 4u;
            instruction_size = 5u;
            nodes[node_count].subtree_offset = instruction_offset;
            nodes[node_count].subtree_size = instruction_size;
            nodes[node_count].subtree_nodes = 1u;
            nodes[node_count].depth = 1u;
            nodes[node_count].arity = 0u;
        } else if (arity == 1) {
            const ssid_gp_node *child;
            if (stack_count < 1u) return SSID_INVALID_ARGUMENT;
            child = nodes + stack[--stack_count];
            nodes[node_count].subtree_offset = child->subtree_offset;
            nodes[node_count].subtree_size = offset - child->subtree_offset;
            nodes[node_count].subtree_nodes = child->subtree_nodes + 1u;
            nodes[node_count].depth = (uint16_t)(child->depth + 1u);
            nodes[node_count].arity = 1u;
        } else if (arity == 2) {
            const ssid_gp_node *lhs;
            const ssid_gp_node *rhs;
            if (stack_count < 2u) return SSID_INVALID_ARGUMENT;
            rhs = nodes + stack[--stack_count];
            lhs = nodes + stack[--stack_count];
            nodes[node_count].subtree_offset = lhs->subtree_offset;
            nodes[node_count].subtree_size = offset - lhs->subtree_offset;
            nodes[node_count].subtree_nodes = lhs->subtree_nodes + rhs->subtree_nodes + 1u;
            nodes[node_count].depth = (uint16_t)((lhs->depth > rhs->depth ? lhs->depth : rhs->depth) + 1u);
            nodes[node_count].arity = 2u;
        } else {
            return SSID_UNSUPPORTED;
        }
        nodes[node_count].instruction_size = instruction_size;
        if (nodes[node_count].depth > gp->config.max_depth || stack_count >= 128u) return SSID_OUT_OF_RANGE;
        stack[stack_count++] = node_count++;
    }
    if (!returned || node_count == 0u) return SSID_INVALID_ARGUMENT;
    population->node_counts[ast_index] = node_count;
    return SSID_OK;
}

static int ssid_gp_emit_leaf(ssid_gp *gp, uint32_t site, uint8_t *output, uint32_t capacity, uint32_t *offset) {
    uint32_t begin = gp->template_value->site_input_offsets[site];
    uint32_t count = gp->template_value->site_input_counts[site];
    if (count == 0u || capacity - *offset < 2u || begin + count > 256u) return SSID_OUT_OF_RANGE;
    output[(*offset)++] = SSID_GP_STATIC_INPUT_F32;
    output[(*offset)++] = (uint8_t)(begin + ssid_gp_random_bounded(gp, count));
    return SSID_OK;
}

static int ssid_gp_emit_random_subtree(ssid_gp *gp, uint32_t site, uint32_t target_nodes, uint32_t depth_remaining, uint8_t *output, uint32_t capacity, uint32_t *offset) {
    int use_binary;
    int use_unary;
    int status;
    if (target_nodes <= 1u || depth_remaining <= 1u) return ssid_gp_emit_leaf(gp, site, output, capacity, offset);
    use_binary = target_nodes >= 3u && gp->config.max_depth > 1u && gp->config.max_nodes_per_ast >= 3u &&
        (gp->config.max_depth == depth_remaining || ssid_gp_random_unit(gp) < 0.70f);
    use_unary = !use_binary && gp->config.max_nodes_per_ast >= 2u && gp->config.max_depth > 1u &&
        gp->config.max_nodes_per_ast >= target_nodes && gp->unary_operations != NULL;
    if (use_binary) {
        uint32_t remaining = target_nodes - 1u;
        uint32_t lhs_nodes = 1u + ssid_gp_random_bounded(gp, remaining - 1u);
        uint32_t rhs_nodes = remaining - lhs_nodes;
        if ((status = ssid_gp_emit_random_subtree(gp, site, lhs_nodes, depth_remaining - 1u, output, capacity, offset)) != SSID_OK ||
            (status = ssid_gp_emit_random_subtree(gp, site, rhs_nodes, depth_remaining - 1u, output, capacity, offset)) != SSID_OK) return status;
        if (*offset >= capacity) return SSID_OUT_OF_RANGE;
        output[(*offset)++] = gp->binary_operations[ssid_gp_random_bounded(gp, gp->binary_operation_count)];
        return SSID_OK;
    }
    if (use_unary && gp->unary_operation_count != 0u) {
        if ((status = ssid_gp_emit_random_subtree(gp, site, target_nodes - 1u, depth_remaining - 1u, output, capacity, offset)) != SSID_OK) return status;
        if (*offset >= capacity) return SSID_OUT_OF_RANGE;
        output[(*offset)++] = gp->unary_operations[ssid_gp_random_bounded(gp, gp->unary_operation_count)];
        return SSID_OK;
    }
    return ssid_gp_emit_leaf(gp, site, output, capacity, offset);
}

static void ssid_gp_population_release(ssid_gp_population *population) {
    if (population == NULL) return;
    free(population->genomes);
    free(population->asts);
    free(population->programs);
    free(population->nodes);
    free(population->node_counts);
    free(population->individuals);
    free(population->constants);
    free(population->bindings);
    memset(population, 0, sizeof(*population));
}

static int ssid_gp_population_allocate(ssid_gp *gp, ssid_gp_population *population) {
    size_t ast_count;
    size_t program_bytes;
    size_t node_count;
    size_t constant_count;
    size_t binding_count;
    uint32_t genome;
    uint32_t site;
    if (!ssid_gp_mul_size(gp->config.population_size, gp->site_count, &ast_count) ||
        !ssid_gp_mul_size(ast_count, gp->config.max_program_bytes_per_ast, &program_bytes) ||
        !ssid_gp_mul_size(ast_count, gp->config.max_nodes_per_ast, &node_count) ||
        !ssid_gp_mul_size(gp->config.population_size, gp->config.constant_count, &constant_count) ||
        !ssid_gp_mul_size(gp->config.population_size, gp->input_count, &binding_count)) return SSID_OUT_OF_RANGE;
    memset(population, 0, sizeof(*population));
    population->genomes = (ssid_genome_desc *)calloc(gp->config.population_size, sizeof(*population->genomes));
    population->asts = (ssid_ast_desc *)calloc(ast_count, sizeof(*population->asts));
    population->programs = (uint8_t *)calloc(program_bytes, 1u);
    population->nodes = (ssid_gp_node *)calloc(node_count, sizeof(*population->nodes));
    population->node_counts = (uint32_t *)calloc(ast_count, sizeof(*population->node_counts));
    population->individuals = (ssid_gp_individual *)calloc(gp->config.population_size, sizeof(*population->individuals));
    population->constants = (float *)calloc(constant_count, sizeof(*population->constants));
    population->bindings = (uint32_t *)calloc(binding_count, sizeof(*population->bindings));
    if (population->genomes == NULL || population->asts == NULL || population->programs == NULL ||
        population->nodes == NULL || population->node_counts == NULL || population->individuals == NULL ||
        population->constants == NULL || population->bindings == NULL) {
        ssid_gp_population_release(population);
        return SSID_OUT_OF_MEMORY;
    }
    for (genome = 0u; genome < gp->config.population_size; ++genome) {
        population->genomes[genome].first_ast = genome * gp->site_count;
        population->genomes[genome].ast_count = (uint16_t)gp->site_count;
        for (site = 0u; site < gp->site_count; ++site) {
            size_t ast_index = (size_t)genome * gp->site_count + site;
            population->asts[ast_index].byte_offset = (uint32_t)(ast_index * gp->config.max_program_bytes_per_ast);
            population->asts[ast_index].site_index = (uint16_t)site;
        }
    }
    return SSID_OK;
}

static uint32_t ssid_gp_genome_complexity(const ssid_gp *gp, const ssid_gp_population *population, uint32_t genome) {
    uint32_t site;
    uint32_t result = 0u;
    for (site = 0u; site < gp->site_count; ++site) result += population->node_counts[(size_t)genome * gp->site_count + site];
    return result;
}

static int ssid_gp_random_ast(ssid_gp *gp, ssid_gp_population *population, uint32_t genome, uint32_t site, uint32_t target_nodes) {
    size_t ast_index = (size_t)genome * gp->site_count + site;
    ssid_ast_desc *ast = population->asts + ast_index;
    uint8_t *program = ssid_gp_program(gp, population, genome, site);
    uint32_t offset = 0u;
    int status = ssid_gp_emit_random_subtree(
        gp, site, target_nodes, gp->config.max_depth, program,
        gp->config.max_program_bytes_per_ast - 1u, &offset);
    if (status != SSID_OK) return status;
    if (offset >= gp->config.max_program_bytes_per_ast) return SSID_OUT_OF_RANGE;
    program[offset++] = SSID_GP_RETURN_F32;
    ast->byte_count = offset;
    return ssid_gp_annotate(gp, population, genome, site);
}

static float ssid_gp_initial_constant(ssid_gp *gp) {
    int exponent = (int)ssid_gp_random_bounded(gp, 15u) - 7;
    float mantissa = 0.5f + ssid_gp_random_unit(gp);
    float value = ldexpf(mantissa, exponent);
    return (ssid_gp_random_u64(gp) & 1u) != 0u ? value : -value;
}

static int ssid_gp_population_seed(ssid_gp *gp, ssid_gp_population *population) {
    uint32_t genome;
    uint32_t site;
    for (genome = 0u; genome < gp->config.population_size; ++genome) {
        for (site = 0u; site < gp->site_count; ++site) {
            uint32_t target = 1u + (genome * gp->site_count + site) % gp->config.initial_max_nodes;
            uint32_t attempt;
            int status = SSID_INTERNAL_ERROR;
            for (attempt = 0u; attempt < 16u; ++attempt) {
                status = ssid_gp_random_ast(gp, population, genome, site, target);
                if (status == SSID_OK) break;
                target = target > 1u ? (target + 1u) / 2u : 1u;
            }
            if (status != SSID_OK) return status;
        }
        for (site = 0u; site < gp->config.constant_count; ++site) {
            population->constants[(size_t)genome * gp->config.constant_count + site] = ssid_gp_initial_constant(gp);
        }
        for (site = 0u; site < gp->input_count; ++site) {
            population->bindings[(size_t)genome * gp->input_count + site] = ssid_gp_random_bounded(gp, gp->bank_slot_count);
        }
        population->individuals[genome].mse = FLT_MAX;
        population->individuals[genome].objective = FLT_MAX;
        population->individuals[genome].complexity = ssid_gp_genome_complexity(gp, population, genome);
        population->individuals[genome].winner_setting = UINT32_MAX;
        population->individuals[genome].parent_a = UINT32_MAX;
        population->individuals[genome].parent_b = UINT32_MAX;
        population->individuals[genome].origin = SSID_GP_ORIGIN_INITIAL;
    }
    return SSID_OK;
}

static void ssid_gp_release(ssid_gp *gp) {
    if (gp == NULL) return;
    ssid_gp_population_release(&gp->populations[0]);
    ssid_gp_population_release(&gp->populations[1]);
    free(gp->unary_operations);
    free(gp->binary_operations);
    free(gp->winner_mse);
    free(gp->winner_settings);
    free(gp->elite_indices);
    free(gp->best_programs);
    free(gp->best_asts);
    free(gp->best_constants);
    free(gp->best_bindings);
    free(gp);
}

int ssid_gp_create(const ssid_template *template_value, const ssid_gp_config *config, const ssid_gp_grammar *grammar, ssid_gp **result) {
    ssid_gp *gp;
    size_t best_program_bytes;
    size_t total_ast_count;
    size_t total_program_bytes;
    uint32_t site;
    int status;
    if (result != NULL) *result = NULL;
    if (template_value == NULL || result == NULL || !ssid_gp_config_valid(config) || !ssid_gp_grammar_valid(grammar) ||
        template_value->site_count == 0u || template_value->site_count > 255u ||
        template_value->kernels[0].input_count == 0u ||
        config->state_count > UINT32_MAX - config->constant_count) {
        ssid_set_error("invalid C99 GP configuration or grammar");
        return SSID_INVALID_ARGUMENT;
    }
    for (site = 0u; site < template_value->site_count; ++site) {
        uint32_t begin = template_value->site_input_offsets[site];
        uint32_t count = template_value->site_input_counts[site];
        if (count == 0u || begin > template_value->kernels[0].input_count ||
            count > template_value->kernels[0].input_count - begin || begin > 256u || count > 256u - begin) {
            ssid_set_error("GP site input slices must be nonempty and fit the bytecode operand");
            return SSID_INVALID_ARGUMENT;
        }
    }
    gp = (ssid_gp *)calloc(1u, sizeof(*gp));
    if (gp == NULL) return SSID_OUT_OF_MEMORY;
    gp->template_value = template_value;
    gp->config = *config;
    gp->site_count = template_value->site_count;
    gp->input_count = template_value->kernels[0].input_count;
    gp->bank_slot_count = config->state_count + config->constant_count;
    gp->module_capacity = template_value->kernels[template_value->kernel_count - 1u].genome_base +
        template_value->kernels[template_value->kernel_count - 1u].genome_capacity;
    gp->batch_count = (config->population_size + gp->module_capacity - 1u) / gp->module_capacity;
    gp->rng_state = config->seed != 0u ? config->seed : UINT64_C(0x6a09e667f3bcc909);
    gp->unary_operation_count = grammar->unary_operation_count;
    gp->binary_operation_count = grammar->binary_operation_count;
    if (grammar->unary_operation_count != 0u) {
        gp->unary_operations = (uint8_t *)malloc(grammar->unary_operation_count);
        if (gp->unary_operations != NULL) memcpy(gp->unary_operations, grammar->unary_operations, grammar->unary_operation_count);
    }
    gp->binary_operations = (uint8_t *)malloc(grammar->binary_operation_count);
    if (gp->binary_operations != NULL) memcpy(gp->binary_operations, grammar->binary_operations, grammar->binary_operation_count);
    if ((grammar->unary_operation_count != 0u && gp->unary_operations == NULL) || gp->binary_operations == NULL) {
        ssid_gp_release(gp);
        return SSID_OUT_OF_MEMORY;
    }
    if (!ssid_gp_mul_size(config->population_size, gp->site_count, &total_ast_count) ||
        total_ast_count > UINT32_MAX ||
        !ssid_gp_mul_size(total_ast_count, config->max_program_bytes_per_ast, &total_program_bytes) ||
        total_program_bytes > UINT32_MAX ||
        !ssid_gp_mul_size(gp->site_count, config->max_program_bytes_per_ast, &best_program_bytes)) {
        ssid_gp_release(gp);
        return SSID_OUT_OF_RANGE;
    }
    gp->winner_mse = (float *)malloc(config->population_size * sizeof(*gp->winner_mse));
    gp->winner_settings = (uint32_t *)malloc(config->population_size * sizeof(*gp->winner_settings));
    gp->elite_indices = config->elite_count == 0u ? NULL :
        (uint32_t *)malloc(config->elite_count * sizeof(*gp->elite_indices));
    gp->best_programs = (uint8_t *)calloc(best_program_bytes, 1u);
    gp->best_asts = (ssid_ast_desc *)calloc(gp->site_count, sizeof(*gp->best_asts));
    gp->best_constants = (float *)malloc(config->constant_count * sizeof(*gp->best_constants));
    gp->best_bindings = (uint32_t *)malloc(gp->input_count * sizeof(*gp->best_bindings));
    if (gp->winner_mse == NULL || gp->winner_settings == NULL ||
        (config->elite_count != 0u && gp->elite_indices == NULL) ||
        gp->best_programs == NULL || gp->best_asts == NULL ||
        gp->best_constants == NULL || gp->best_bindings == NULL) {
        ssid_gp_release(gp);
        return SSID_OUT_OF_MEMORY;
    }
    for (site = 0u; site < gp->site_count; ++site) {
        gp->best_asts[site].byte_offset = site * config->max_program_bytes_per_ast;
        gp->best_asts[site].site_index = (uint16_t)site;
    }
    gp->best_genome.first_ast = 0u;
    gp->best_genome.ast_count = (uint16_t)gp->site_count;
    gp->best_mse = FLT_MAX;
    gp->best_objective = FLT_MAX;
    if ((status = ssid_gp_population_allocate(gp, &gp->populations[0])) != SSID_OK ||
        (status = ssid_gp_population_allocate(gp, &gp->populations[1])) != SSID_OK ||
        (status = ssid_gp_population_seed(gp, &gp->populations[0])) != SSID_OK) {
        ssid_set_error("could not initialize the C99 GP population");
        ssid_gp_release(gp);
        return status;
    }
    *result = gp;
    return SSID_OK;
}

void ssid_gp_destroy(ssid_gp *gp) {
    ssid_gp_release(gp);
}

uint32_t ssid_gp_generation_get(const ssid_gp *gp) {
    return gp == NULL ? 0u : gp->generation;
}

uint32_t ssid_gp_batch_count(const ssid_gp *gp) {
    return gp == NULL ? 0u : gp->batch_count;
}

int ssid_gp_batch_get(const ssid_gp *gp, uint32_t batch_index, ssid_genome_batch *batch, uint32_t *population_offset) {
    const ssid_gp_population *population;
    uint32_t begin;
    uint32_t count;
    size_t ast_count;
    size_t program_bytes;
    if (gp == NULL || batch == NULL || batch_index >= gp->batch_count) return SSID_INVALID_ARGUMENT;
    population = &gp->populations[gp->current_population];
    begin = batch_index * gp->module_capacity;
    count = gp->config.population_size - begin;
    if (count > gp->module_capacity) count = gp->module_capacity;
    if (!ssid_gp_mul_size(gp->config.population_size, gp->site_count, &ast_count) ||
        !ssid_gp_mul_size(ast_count, gp->config.max_program_bytes_per_ast, &program_bytes)) return SSID_OUT_OF_RANGE;
    batch->genomes = population->genomes + begin;
    batch->genome_count = count;
    batch->asts = population->asts;
    batch->ast_count = (uint32_t)ast_count;
    batch->program_bytes = population->programs;
    batch->program_byte_count = program_bytes;
    if (population_offset != NULL) *population_offset = begin;
    return SSID_OK;
}

int ssid_gp_settings_generate(ssid_gp *gp, float *settings, size_t settings_count, uint32_t *bindings, size_t bindings_count) {
    const ssid_gp_population *population;
    size_t required_settings;
    size_t required_bindings;
    uint32_t genome;
    uint32_t slot;
    uint32_t setting;
    if (gp == NULL || settings == NULL || bindings == NULL ||
        !ssid_gp_mul_size(gp->config.population_size, gp->config.constant_count, &required_settings) ||
        !ssid_gp_mul_size(required_settings, gp->config.settings_per_genome, &required_settings) ||
        !ssid_gp_mul_size(gp->config.population_size, gp->input_count, &required_bindings) ||
        !ssid_gp_mul_size(required_bindings, gp->config.settings_per_genome, &required_bindings) ||
        settings_count < required_settings || bindings_count < required_bindings) {
        ssid_set_error("GP settings output buffers are too small");
        return SSID_INVALID_ARGUMENT;
    }
    population = &gp->populations[gp->current_population];
    for (genome = 0u; genome < gp->config.population_size; ++genome) {
        for (slot = 0u; slot < gp->config.constant_count; ++slot) {
            const float incumbent = population->constants[(size_t)genome * gp->config.constant_count + slot];
            const size_t base = ((size_t)genome * gp->config.constant_count + slot) * gp->config.settings_per_genome;
            settings[base] = incumbent;
            for (setting = 1u; setting < gp->config.settings_per_genome; ++setting) {
                uint64_t key = gp->config.seed ^ ((uint64_t)gp->generation << 48u) ^
                    ((uint64_t)genome << 28u) ^ ((uint64_t)slot << 20u) ^ setting;
                float unit = ssid_gp_hash_unit(key);
                float radius = (fabsf(incumbent) + 1.0f) * gp->config.constant_mutation_scale;
                settings[base + setting] = incumbent + (2.0f * unit - 1.0f) * radius;
            }
        }
        for (slot = 0u; slot < gp->input_count; ++slot) {
            uint32_t incumbent = population->bindings[(size_t)genome * gp->input_count + slot];
            size_t base = ((size_t)genome * gp->input_count + slot) * gp->config.settings_per_genome;
            bindings[base] = incumbent;
            for (setting = 1u; setting < gp->config.settings_per_genome; ++setting) {
                uint64_t key = gp->config.seed ^ UINT64_C(0xa0761d6478bd642f) ^
                    ((uint64_t)gp->generation << 48u) ^ ((uint64_t)genome << 28u) ^
                    ((uint64_t)slot << 20u) ^ setting;
                bindings[base + setting] = ssid_gp_hash_unit(key) < gp->config.binding_keep_probability
                    ? incumbent
                    : (uint32_t)(ssid_gp_mix64(key ^ UINT64_C(0xe7037ed1a0b428db)) % gp->bank_slot_count);
            }
        }
    }
    return SSID_OK;
}

static void ssid_gp_best_consider(ssid_gp *gp, const ssid_gp_population *population, uint32_t genome) {
    const ssid_gp_individual *individual = population->individuals + genome;
    uint32_t site;
    if (!isfinite(individual->objective) || individual->objective >= gp->best_objective) return;
    for (site = 0u; site < gp->site_count; ++site) {
        size_t ast_index = (size_t)genome * gp->site_count + site;
        const ssid_ast_desc *source_ast = population->asts + ast_index;
        uint8_t *destination = gp->best_programs + (size_t)site * gp->config.max_program_bytes_per_ast;
        memcpy(destination, ssid_gp_program_const(gp, population, genome, site), source_ast->byte_count);
        gp->best_asts[site].byte_count = source_ast->byte_count;
    }
    memcpy(gp->best_constants, population->constants + (size_t)genome * gp->config.constant_count,
        gp->config.constant_count * sizeof(*gp->best_constants));
    memcpy(gp->best_bindings, population->bindings + (size_t)genome * gp->input_count,
        gp->input_count * sizeof(*gp->best_bindings));
    gp->best_mse = individual->mse;
    gp->best_objective = individual->objective;
    gp->best_generation = gp->generation;
    gp->best_complexity = individual->complexity;
    gp->has_best = 1;
}

int ssid_gp_winners_apply(ssid_gp *gp, const float *winner_mse, const uint32_t *winner_settings, const float *settings, size_t settings_count, const uint32_t *bindings, size_t bindings_count) {
    ssid_gp_population *population;
    size_t required_settings;
    size_t required_bindings;
    uint32_t genome;
    if (gp == NULL || winner_mse == NULL || winner_settings == NULL || settings == NULL || bindings == NULL ||
        !ssid_gp_mul_size(gp->config.population_size, gp->config.constant_count, &required_settings) ||
        !ssid_gp_mul_size(required_settings, gp->config.settings_per_genome, &required_settings) ||
        !ssid_gp_mul_size(gp->config.population_size, gp->input_count, &required_bindings) ||
        !ssid_gp_mul_size(required_bindings, gp->config.settings_per_genome, &required_bindings) ||
        settings_count < required_settings || bindings_count < required_bindings) return SSID_INVALID_ARGUMENT;
    population = &gp->populations[gp->current_population];
    for (genome = 0u; genome < gp->config.population_size; ++genome) {
        ssid_gp_individual *individual = population->individuals + genome;
        uint32_t selected = winner_settings[genome];
        uint32_t slot;
        if (selected >= gp->config.settings_per_genome) {
            ssid_set_error("GPU returned setting %u outside the GP setting population", selected);
            return SSID_OUT_OF_RANGE;
        }
        for (slot = 0u; slot < gp->config.constant_count; ++slot) {
            size_t source = ((size_t)genome * gp->config.constant_count + slot) * gp->config.settings_per_genome + selected;
            population->constants[(size_t)genome * gp->config.constant_count + slot] = settings[source];
        }
        for (slot = 0u; slot < gp->input_count; ++slot) {
            size_t source = ((size_t)genome * gp->input_count + slot) * gp->config.settings_per_genome + selected;
            uint32_t value = bindings[source];
            if (value >= gp->bank_slot_count) {
                ssid_set_error("materialized binding %u for genome %u slot %u exceeds bank size %u",
                    value, genome, slot, gp->bank_slot_count);
                return SSID_OUT_OF_RANGE;
            }
            population->bindings[(size_t)genome * gp->input_count + slot] = value;
        }
        individual->mse = isfinite(winner_mse[genome]) && winner_mse[genome] >= 0.0f ? winner_mse[genome] : FLT_MAX;
        individual->complexity = ssid_gp_genome_complexity(gp, population, genome);
        individual->objective = individual->mse < FLT_MAX
            ? individual->mse + gp->config.parsimony_coefficient * individual->complexity
            : FLT_MAX;
        individual->winner_setting = selected;
        ssid_gp_best_consider(gp, population, genome);
    }
    gp->population_scored = 1;
    return SSID_OK;
}

static int ssid_gp_hashed_winners_apply(ssid_gp *gp, const float *winner_mse, const uint32_t *winner_settings) {
    ssid_gp_population *population = &gp->populations[gp->current_population];
    uint32_t genome;
    for (genome = 0u; genome < gp->config.population_size; ++genome) {
        ssid_gp_individual *individual = population->individuals + genome;
        uint32_t selected = winner_settings[genome];
        float selected_mse = winner_mse[genome];
        uint32_t slot;
        if (!isfinite(selected_mse) || selected_mse < gp->config.minimum_valid_mse) {
            selected = 0u;
            selected_mse = FLT_MAX;
        }
        if (selected >= gp->config.settings_per_genome) {
            ssid_set_error(
                "fused winner setting %u for generation %u genome %u exceeds setting count %u (score %.9g)",
                selected, gp->generation, genome, gp->config.settings_per_genome, selected_mse);
            return SSID_OUT_OF_RANGE;
        }
        if (selected != 0u) {
            for (slot = 0u; slot < gp->config.constant_count; ++slot) {
                size_t index = (size_t)genome * gp->config.constant_count + slot;
                float incumbent = population->constants[index];
                uint64_t key = gp->config.seed ^ ((uint64_t)gp->generation << 48u) ^
                    ((uint64_t)genome << 28u) ^ ((uint64_t)slot << 20u) ^ selected;
                float radius = (fabsf(incumbent) + 1.0f) * gp->config.constant_mutation_scale;
                population->constants[index] = incumbent + (2.0f * ssid_gp_hash_unit(key) - 1.0f) * radius;
            }
            for (slot = 0u; slot < gp->input_count; ++slot) {
                size_t index = (size_t)genome * gp->input_count + slot;
                uint32_t incumbent = population->bindings[index];
                uint64_t key = gp->config.seed ^ UINT64_C(0xa0761d6478bd642f) ^
                    ((uint64_t)gp->generation << 48u) ^ ((uint64_t)genome << 28u) ^
                    ((uint64_t)slot << 20u) ^ selected;
                population->bindings[index] = ssid_gp_hash_unit(key) < gp->config.binding_keep_probability
                    ? incumbent
                    : (uint32_t)(ssid_gp_mix64(key ^ UINT64_C(0xe7037ed1a0b428db)) % gp->bank_slot_count);
            }
        }
        individual->mse = selected_mse >= 0.0f ? selected_mse : FLT_MAX;
        individual->complexity = ssid_gp_genome_complexity(gp, population, genome);
        individual->objective = individual->mse < FLT_MAX
            ? individual->mse + gp->config.parsimony_coefficient * individual->complexity
            : FLT_MAX;
        individual->winner_setting = selected;
        ssid_gp_best_consider(gp, population, genome);
    }
    gp->population_scored = 1;
    return SSID_OK;
}

int ssid_gp_best_get(const ssid_gp *gp, ssid_gp_best *best) {
    if (gp == NULL || best == NULL || !gp->has_best) return SSID_INVALID_ARGUMENT;
    memset(best, 0, sizeof(*best));
    best->mse = gp->best_mse;
    best->objective = gp->best_objective;
    best->generation = gp->best_generation;
    best->complexity = gp->best_complexity;
    best->genome.genomes = &gp->best_genome;
    best->genome.genome_count = 1u;
    best->genome.asts = gp->best_asts;
    best->genome.ast_count = gp->site_count;
    best->genome.program_bytes = gp->best_programs;
    best->genome.program_byte_count = (size_t)gp->site_count * gp->config.max_program_bytes_per_ast;
    best->constants = gp->best_constants;
    best->bindings = gp->best_bindings;
    return SSID_OK;
}

int ssid_gp_candidate_get(const ssid_gp *gp, uint32_t genome_index, ssid_gp_best *candidate) {
    const ssid_gp_population *population;
    const ssid_gp_individual *individual;
    if (gp == NULL || candidate == NULL || genome_index >= gp->config.population_size ||
        !gp->population_scored) return SSID_INVALID_ARGUMENT;
    population = &gp->populations[gp->current_population];
    individual = population->individuals + genome_index;
    memset(candidate, 0, sizeof(*candidate));
    candidate->mse = individual->mse;
    candidate->objective = individual->objective;
    candidate->generation = gp->generation;
    candidate->complexity = individual->complexity;
    candidate->genome.genomes = population->genomes + genome_index;
    candidate->genome.genome_count = 1u;
    candidate->genome.asts = population->asts;
    candidate->genome.ast_count = gp->config.population_size * gp->site_count;
    candidate->genome.program_bytes = population->programs;
    candidate->genome.program_byte_count = (size_t)gp->config.population_size * gp->site_count *
        gp->config.max_program_bytes_per_ast;
    candidate->constants = population->constants + (size_t)genome_index * gp->config.constant_count;
    candidate->bindings = population->bindings + (size_t)genome_index * gp->input_count;
    return SSID_OK;
}

int ssid_gp_candidate_improve(ssid_gp *gp, uint32_t genome_index, float mse,
    const float *constants, size_t constant_count, const uint32_t *bindings,
    size_t binding_count, int *accepted) {
    ssid_gp_population *population;
    ssid_gp_individual *individual;
    uint32_t slot;
    if (accepted != NULL) *accepted = 0;
    if (gp == NULL || constants == NULL || bindings == NULL || accepted == NULL ||
        genome_index >= gp->config.population_size || !gp->population_scored ||
        constant_count < gp->config.constant_count || binding_count < gp->input_count ||
        !isfinite(mse) || mse < gp->config.minimum_valid_mse) return SSID_INVALID_ARGUMENT;
    population = &gp->populations[gp->current_population];
    individual = population->individuals + genome_index;
    if (!(mse < individual->mse)) return SSID_OK;
    for (slot = 0u; slot < gp->config.constant_count; ++slot) {
        if (!isfinite(constants[slot])) {
            ssid_set_error("promoted constant %u for genome %u is not finite",
                slot, genome_index);
            return SSID_INVALID_ARGUMENT;
        }
    }
    for (slot = 0u; slot < gp->input_count; ++slot) {
        if (bindings[slot] >= gp->bank_slot_count) {
            ssid_set_error("promoted binding %u for genome %u slot %u exceeds bank size %u",
                bindings[slot], genome_index, slot, gp->bank_slot_count);
            return SSID_OUT_OF_RANGE;
        }
    }
    memcpy(population->constants + (size_t)genome_index * gp->config.constant_count,
        constants, gp->config.constant_count * sizeof(*constants));
    memcpy(population->bindings + (size_t)genome_index * gp->input_count,
        bindings, gp->input_count * sizeof(*bindings));
    individual->mse = mse;
    individual->objective = mse + gp->config.parsimony_coefficient * individual->complexity;
    individual->winner_setting = UINT32_MAX;
    ssid_gp_best_consider(gp, population, genome_index);
    *accepted = 1;
    return SSID_OK;
}

static void ssid_gp_copy_genome(ssid_gp *gp, ssid_gp_population *destination, uint32_t destination_index, const ssid_gp_population *source, uint32_t source_index, uint8_t origin, uint32_t parent_a, uint32_t parent_b) {
    uint32_t site;
    for (site = 0u; site < gp->site_count; ++site) {
        size_t source_ast_index = (size_t)source_index * gp->site_count + site;
        size_t destination_ast_index = (size_t)destination_index * gp->site_count + site;
        uint32_t byte_count = source->asts[source_ast_index].byte_count;
        uint32_t node_count = source->node_counts[source_ast_index];
        memcpy(ssid_gp_program(gp, destination, destination_index, site),
            ssid_gp_program_const(gp, source, source_index, site), byte_count);
        memcpy(ssid_gp_nodes(gp, destination, destination_index, site),
            ssid_gp_nodes_const(gp, source, source_index, site), node_count * sizeof(ssid_gp_node));
        destination->asts[destination_ast_index].byte_count = byte_count;
        destination->node_counts[destination_ast_index] = node_count;
    }
    memcpy(destination->constants + (size_t)destination_index * gp->config.constant_count,
        source->constants + (size_t)source_index * gp->config.constant_count,
        gp->config.constant_count * sizeof(*destination->constants));
    memcpy(destination->bindings + (size_t)destination_index * gp->input_count,
        source->bindings + (size_t)source_index * gp->input_count,
        gp->input_count * sizeof(*destination->bindings));
    memset(destination->individuals + destination_index, 0, sizeof(*destination->individuals));
    destination->individuals[destination_index].mse = FLT_MAX;
    destination->individuals[destination_index].objective = FLT_MAX;
    destination->individuals[destination_index].complexity = ssid_gp_genome_complexity(gp, destination, destination_index);
    destination->individuals[destination_index].winner_setting = UINT32_MAX;
    destination->individuals[destination_index].origin = origin;
    destination->individuals[destination_index].parent_a = parent_a;
    destination->individuals[destination_index].parent_b = parent_b;
}

static uint32_t ssid_gp_tournament_select(ssid_gp *gp, const ssid_gp_population *population) {
    uint32_t best = ssid_gp_random_bounded(gp, gp->config.population_size);
    uint32_t index;
    for (index = 1u; index < gp->config.tournament_size; ++index) {
        uint32_t candidate = ssid_gp_random_bounded(gp, gp->config.population_size);
        float candidate_score = population->individuals[candidate].objective;
        float best_score = population->individuals[best].objective;
        if (candidate_score < best_score || (candidate_score == best_score && candidate < best)) best = candidate;
    }
    return best;
}

static int ssid_gp_subtree_crossover(ssid_gp *gp, ssid_gp_population *next, uint32_t child, const ssid_gp_population *current, uint32_t parent_a, uint32_t parent_b) {
    uint32_t site = ssid_gp_random_bounded(gp, gp->site_count);
    size_t ast_a = (size_t)parent_a * gp->site_count + site;
    size_t ast_b = (size_t)parent_b * gp->site_count + site;
    const ssid_gp_node *nodes_a = ssid_gp_nodes_const(gp, current, parent_a, site);
    const ssid_gp_node *nodes_b = ssid_gp_nodes_const(gp, current, parent_b, site);
    const ssid_gp_node *target = nodes_a + ssid_gp_random_bounded(gp, current->node_counts[ast_a]);
    const ssid_gp_node *donor = nodes_b + ssid_gp_random_bounded(gp, current->node_counts[ast_b]);
    uint32_t new_nodes = current->node_counts[ast_a] - target->subtree_nodes + donor->subtree_nodes;
    uint32_t new_bytes = current->asts[ast_a].byte_count - target->subtree_size + donor->subtree_size;
    const uint8_t *source_a;
    const uint8_t *source_b;
    uint8_t *output;
    uint32_t suffix;
    if (new_nodes > gp->config.max_nodes_per_ast || new_bytes > gp->config.max_program_bytes_per_ast) return SSID_OUT_OF_RANGE;
    ssid_gp_copy_genome(gp, next, child, current, parent_a, SSID_GP_ORIGIN_SUBTREE_CROSSOVER, parent_a, parent_b);
    source_a = ssid_gp_program_const(gp, current, parent_a, site);
    source_b = ssid_gp_program_const(gp, current, parent_b, site);
    output = ssid_gp_program(gp, next, child, site);
    memcpy(output, source_a, target->subtree_offset);
    memcpy(output + target->subtree_offset, source_b + donor->subtree_offset, donor->subtree_size);
    suffix = target->subtree_offset + target->subtree_size;
    memcpy(output + target->subtree_offset + donor->subtree_size, source_a + suffix,
        current->asts[ast_a].byte_count - suffix);
    next->asts[(size_t)child * gp->site_count + site].byte_count = new_bytes;
    return ssid_gp_annotate(gp, next, child, site);
}

static int ssid_gp_whole_site_crossover(ssid_gp *gp, ssid_gp_population *next, uint32_t child, const ssid_gp_population *current, uint32_t parent_a, uint32_t parent_b) {
    uint32_t site = ssid_gp_random_bounded(gp, gp->site_count);
    size_t source_ast = (size_t)parent_b * gp->site_count + site;
    size_t destination_ast = (size_t)child * gp->site_count + site;
    uint32_t byte_count;
    uint32_t node_count;
    uint32_t input_begin;
    uint32_t input_count;
    uint32_t constant;
    ssid_gp_copy_genome(gp, next, child, current, parent_a, SSID_GP_ORIGIN_WHOLE_SITE_CROSSOVER, parent_a, parent_b);
    byte_count = current->asts[source_ast].byte_count;
    node_count = current->node_counts[source_ast];
    memcpy(ssid_gp_program(gp, next, child, site), ssid_gp_program_const(gp, current, parent_b, site), byte_count);
    memcpy(ssid_gp_nodes(gp, next, child, site), ssid_gp_nodes_const(gp, current, parent_b, site),
        node_count * sizeof(ssid_gp_node));
    next->asts[destination_ast].byte_count = byte_count;
    next->node_counts[destination_ast] = node_count;
    input_begin = gp->template_value->site_input_offsets[site];
    input_count = gp->template_value->site_input_counts[site];
    memcpy(next->bindings + (size_t)child * gp->input_count + input_begin,
        current->bindings + (size_t)parent_b * gp->input_count + input_begin,
        input_count * sizeof(*next->bindings));
    for (constant = 0u; constant < gp->config.constant_count; ++constant) {
        if ((ssid_gp_random_u64(gp) & 1u) != 0u) {
            next->constants[(size_t)child * gp->config.constant_count + constant] =
                current->constants[(size_t)parent_b * gp->config.constant_count + constant];
        }
    }
    next->individuals[child].complexity = ssid_gp_genome_complexity(gp, next, child);
    return SSID_OK;
}

static int ssid_gp_subtree_mutation(ssid_gp *gp, ssid_gp_population *next, uint32_t child, const ssid_gp_population *current, uint32_t parent) {
    uint32_t site = ssid_gp_random_bounded(gp, gp->site_count);
    size_t source_ast_index = (size_t)parent * gp->site_count + site;
    const ssid_gp_node *source_nodes = ssid_gp_nodes_const(gp, current, parent, site);
    const ssid_gp_node *target = source_nodes + ssid_gp_random_bounded(gp, current->node_counts[source_ast_index]);
    const uint8_t *source = ssid_gp_program_const(gp, current, parent, site);
    uint8_t *output;
    uint32_t retained_nodes = current->node_counts[source_ast_index] - target->subtree_nodes;
    uint32_t maximum_replacement = gp->config.max_nodes_per_ast - retained_nodes;
    uint32_t target_nodes = 1u + ssid_gp_random_bounded(gp, maximum_replacement);
    uint32_t suffix = target->subtree_offset + target->subtree_size;
    uint32_t suffix_bytes = current->asts[source_ast_index].byte_count - suffix;
    uint32_t offset = target->subtree_offset;
    int status;
    ssid_gp_copy_genome(gp, next, child, current, parent, SSID_GP_ORIGIN_SUBTREE_MUTATION, parent, UINT32_MAX);
    output = ssid_gp_program(gp, next, child, site);
    memcpy(output, source, target->subtree_offset);
    status = ssid_gp_emit_random_subtree(gp, site, target_nodes, gp->config.max_depth, output,
        gp->config.max_program_bytes_per_ast - suffix_bytes, &offset);
    if (status != SSID_OK || suffix_bytes > gp->config.max_program_bytes_per_ast - offset) return SSID_OUT_OF_RANGE;
    memcpy(output + offset, source + suffix, suffix_bytes);
    offset += suffix_bytes;
    next->asts[(size_t)child * gp->site_count + site].byte_count = offset;
    return ssid_gp_annotate(gp, next, child, site);
}

static int ssid_gp_point_mutation(ssid_gp *gp, ssid_gp_population *next, uint32_t child, const ssid_gp_population *current, uint32_t parent) {
    uint32_t site = ssid_gp_random_bounded(gp, gp->site_count);
    size_t ast_index = (size_t)parent * gp->site_count + site;
    const ssid_gp_node *source_nodes = ssid_gp_nodes_const(gp, current, parent, site);
    const ssid_gp_node *target = source_nodes + ssid_gp_random_bounded(gp, current->node_counts[ast_index]);
    uint8_t *program;
    ssid_gp_copy_genome(gp, next, child, current, parent, SSID_GP_ORIGIN_POINT_MUTATION, parent, UINT32_MAX);
    program = ssid_gp_program(gp, next, child, site);
    if (target->opcode == SSID_GP_STATIC_INPUT_F32) {
        uint32_t begin = gp->template_value->site_input_offsets[site];
        uint32_t count = gp->template_value->site_input_counts[site];
        program[target->instruction_offset + 1u] = (uint8_t)(begin + ssid_gp_random_bounded(gp, count));
    } else if (target->arity == 1u && gp->unary_operation_count != 0u) {
        program[target->instruction_offset] = gp->unary_operations[ssid_gp_random_bounded(gp, gp->unary_operation_count)];
    } else if (target->arity == 2u) {
        program[target->instruction_offset] = gp->binary_operations[ssid_gp_random_bounded(gp, gp->binary_operation_count)];
    } else {
        return SSID_OUT_OF_RANGE;
    }
    return ssid_gp_annotate(gp, next, child, site);
}

static int ssid_gp_random_genome(ssid_gp *gp, ssid_gp_population *next, uint32_t child, uint32_t parent) {
    uint32_t site;
    uint32_t slot;
    int status;
    for (site = 0u; site < gp->site_count; ++site) {
        uint32_t target = 1u + ssid_gp_random_bounded(gp, gp->config.initial_max_nodes);
        if ((status = ssid_gp_random_ast(gp, next, child, site, target)) != SSID_OK) return status;
    }
    for (slot = 0u; slot < gp->config.constant_count; ++slot) {
        next->constants[(size_t)child * gp->config.constant_count + slot] = ssid_gp_initial_constant(gp);
    }
    for (slot = 0u; slot < gp->input_count; ++slot) {
        next->bindings[(size_t)child * gp->input_count + slot] = ssid_gp_random_bounded(gp, gp->bank_slot_count);
    }
    memset(next->individuals + child, 0, sizeof(*next->individuals));
    next->individuals[child].mse = FLT_MAX;
    next->individuals[child].objective = FLT_MAX;
    next->individuals[child].complexity = ssid_gp_genome_complexity(gp, next, child);
    next->individuals[child].winner_setting = UINT32_MAX;
    next->individuals[child].origin = SSID_GP_ORIGIN_RANDOM;
    next->individuals[child].parent_a = parent;
    next->individuals[child].parent_b = UINT32_MAX;
    return SSID_OK;
}

static int ssid_gp_individual_better(const ssid_gp_population *population, uint32_t lhs, uint32_t rhs) {
    float lhs_score = population->individuals[lhs].objective;
    float rhs_score = population->individuals[rhs].objective;
    return lhs_score < rhs_score || (lhs_score == rhs_score && lhs < rhs);
}

static void ssid_gp_elites_select(ssid_gp *gp, const ssid_gp_population *population) {
    uint32_t index;
    uint32_t slot;
    for (slot = 0u; slot < gp->config.elite_count; ++slot) gp->elite_indices[slot] = UINT32_MAX;
    for (index = 0u; index < gp->config.population_size; ++index) {
        for (slot = 0u; slot < gp->config.elite_count; ++slot) {
            if (gp->elite_indices[slot] == UINT32_MAX || ssid_gp_individual_better(population, index, gp->elite_indices[slot])) {
                uint32_t move;
                for (move = gp->config.elite_count - 1u; move > slot; --move) gp->elite_indices[move] = gp->elite_indices[move - 1u];
                gp->elite_indices[slot] = index;
                break;
            }
        }
    }
}

int ssid_gp_generation_advance(ssid_gp *gp) {
    ssid_gp_population *current;
    ssid_gp_population *next;
    uint32_t child;
    if (gp == NULL || !gp->population_scored) {
        ssid_set_error("the current GP population must be scored before advancing");
        return SSID_INVALID_ARGUMENT;
    }
    current = &gp->populations[gp->current_population];
    next = &gp->populations[1u - gp->current_population];
    ssid_gp_elites_select(gp, current);
    for (child = 0u; child < gp->config.elite_count; ++child) {
        uint32_t parent = gp->elite_indices[child];
        ssid_gp_copy_genome(gp, next, child, current, parent, SSID_GP_ORIGIN_ELITE, parent, UINT32_MAX);
    }
    while (child < gp->config.population_size) {
        float choice = ssid_gp_random_unit(gp);
        uint32_t attempt;
        int status = SSID_OUT_OF_RANGE;
        for (attempt = 0u; attempt < 16u && status != SSID_OK; ++attempt) {
            uint32_t parent_a = ssid_gp_tournament_select(gp, current);
            if (choice < gp->config.subtree_crossover_probability) {
                uint32_t parent_b = ssid_gp_tournament_select(gp, current);
                status = ssid_gp_subtree_crossover(gp, next, child, current, parent_a, parent_b);
            } else if (choice < gp->config.subtree_crossover_probability + gp->config.whole_site_crossover_probability) {
                uint32_t parent_b = ssid_gp_tournament_select(gp, current);
                status = ssid_gp_whole_site_crossover(gp, next, child, current, parent_a, parent_b);
            } else if (choice < gp->config.subtree_crossover_probability + gp->config.whole_site_crossover_probability + gp->config.subtree_mutation_probability) {
                status = ssid_gp_subtree_mutation(gp, next, child, current, parent_a);
            } else if (choice < gp->config.subtree_crossover_probability + gp->config.whole_site_crossover_probability + gp->config.subtree_mutation_probability + gp->config.point_mutation_probability) {
                status = ssid_gp_point_mutation(gp, next, child, current, parent_a);
            } else {
                status = ssid_gp_random_genome(gp, next, child, parent_a);
            }
        }
        if (status != SSID_OK) {
            uint32_t parent = ssid_gp_tournament_select(gp, current);
            status = ssid_gp_random_genome(gp, next, child, parent);
            if (status != SSID_OK) {
                ssid_set_error("GP generation %u could not construct child %u after variation retries",
                    gp->generation + 1u, child);
                return status;
            }
        }
        next->individuals[child].complexity = ssid_gp_genome_complexity(gp, next, child);
        child += 1u;
    }
    gp->current_population = 1u - gp->current_population;
    gp->generation += 1u;
    gp->population_scored = 0;
    return SSID_OK;
}

static int ssid_gp_device_alloc(ssid_pipeline *pipeline, size_t count, size_t item_size, uint64_t *pointer) {
    size_t bytes;
    if (!ssid_gp_mul_size(count, item_size, &bytes)) return SSID_OUT_OF_RANGE;
    return ssid_device_alloc(pipeline, bytes, pointer);
}

static int ssid_gp_trace_validate(const ssid_gp *gp, uint32_t generations, ssid_gp_trace *trace) {
    size_t program_count;
    size_t ast_count;
    size_t constant_count;
    size_t binding_count;
    uint32_t required_checkpoints;
    if (trace == NULL) return SSID_OK;
    trace->checkpoint_count = 0u;
    if (trace->checkpoint_stride == 0u) {
        ssid_set_error("C99 GP checkpoint stride must be positive");
        return SSID_INVALID_ARGUMENT;
    }
    required_checkpoints = 1u + (generations - 1u) / trace->checkpoint_stride;
    if ((generations - 1u) % trace->checkpoint_stride != 0u) required_checkpoints += 1u;
    if (trace->checkpoints == NULL || trace->program_bytes == NULL || trace->ast_byte_counts == NULL ||
        trace->constants == NULL || trace->bindings == NULL || trace->checkpoint_capacity < required_checkpoints ||
        !ssid_gp_mul_size(trace->checkpoint_capacity, gp->site_count, &ast_count) ||
        !ssid_gp_mul_size(ast_count, gp->config.max_program_bytes_per_ast, &program_count) ||
        !ssid_gp_mul_size(trace->checkpoint_capacity, gp->config.constant_count, &constant_count) ||
        !ssid_gp_mul_size(trace->checkpoint_capacity, gp->input_count, &binding_count) ||
        trace->program_byte_count < program_count || trace->ast_byte_count_count < ast_count ||
        trace->constant_count < constant_count || trace->binding_count < binding_count) {
        ssid_set_error("C99 GP trace buffers are missing or too small");
        return SSID_INVALID_ARGUMENT;
    }
    return SSID_OK;
}

static void ssid_gp_trace_capture(ssid_gp *gp, ssid_gp_trace *trace, uint32_t checkpoint_index, double elapsed_seconds) {
    const ssid_gp_population *population;
    uint32_t generation_best = 0u;
    uint32_t genome;
    uint32_t site;
    ssid_gp_checkpoint *checkpoint;
    size_t program_stride;
    if (trace == NULL) return;
    population = &gp->populations[gp->current_population];
    for (genome = 1u; genome < gp->config.population_size; ++genome) {
        if (ssid_gp_individual_better(population, genome, generation_best)) generation_best = genome;
    }
    checkpoint = trace->checkpoints + checkpoint_index;
    memset(checkpoint, 0, sizeof(*checkpoint));
    checkpoint->elapsed_seconds = elapsed_seconds;
    checkpoint->generation_best_mse = population->individuals[generation_best].mse;
    checkpoint->generation_best_objective = population->individuals[generation_best].objective;
    checkpoint->best_mse = gp->best_mse;
    checkpoint->best_objective = gp->best_objective;
    checkpoint->generation = gp->generation;
    checkpoint->best_generation = gp->best_generation;
    checkpoint->best_complexity = gp->best_complexity;
    program_stride = (size_t)gp->site_count * gp->config.max_program_bytes_per_ast;
    memcpy(trace->program_bytes + (size_t)checkpoint_index * program_stride, gp->best_programs, program_stride);
    for (site = 0u; site < gp->site_count; ++site) {
        trace->ast_byte_counts[(size_t)checkpoint_index * gp->site_count + site] = gp->best_asts[site].byte_count;
    }
    memcpy(trace->constants + (size_t)checkpoint_index * gp->config.constant_count, gp->best_constants,
        gp->config.constant_count * sizeof(*trace->constants));
    memcpy(trace->bindings + (size_t)checkpoint_index * gp->input_count, gp->best_bindings,
        gp->input_count * sizeof(*trace->bindings));
    trace->checkpoint_count = checkpoint_index + 1u;
}

static int ssid_gp_run_internal(ssid_gp *gp, ssid_pipeline *pipeline, const ssid_gp_run_config *config, ssid_gp_run_stats *stats, ssid_gp_trace *trace) {
    ssid_genome_batch *batches = NULL;
    ssid_fedbatch_launch *launches = NULL;
    ssid_ticket **tickets = NULL;
    uint64_t settings_device = 0u;
    uint64_t bindings_device = 0u;
    uint64_t cta_score_device = 0u;
    uint64_t cta_setting_device = 0u;
    uint64_t winner_score_device = 0u;
    uint64_t winner_setting_device = 0u;
    uint64_t mse_device = 0u;
    float *full_mse = NULL;
    float *materialized_settings = NULL;
    uint32_t *materialized_bindings = NULL;
    size_t settings_count;
    size_t bindings_count;
    size_t device_settings_count;
    size_t device_bindings_count;
    size_t materialized_settings_bytes;
    size_t materialized_bindings_bytes;
    size_t cta_count;
    size_t full_mse_count = 0u;
    size_t full_mse_bytes;
    uint32_t setting_tiles;
    uint32_t batch;
    uint32_t generation;
    uint32_t checkpoint_index = 0u;
    double wall_started;
    int status = SSID_OK;
    if (stats != NULL) memset(stats, 0, sizeof(*stats));
    if (gp == NULL || pipeline == NULL || config == NULL || stats == NULL ||
        (gp->template_value->settings_mode != SSID_SETTINGS_HASHED_INCUMBENT &&
            gp->template_value->settings_mode != SSID_SETTINGS_MATERIALIZED) || config->generations == 0u ||
        config->steps_per_observation == 0u || config->threads_per_block == 0u ||
        config->threads_per_block > 1024u || (config->threads_per_block & (config->threads_per_block - 1u)) != 0u ||
        config->reduction_threads == 0u || config->reduction_threads > 1024u ||
        (config->reduction_threads & (config->reduction_threads - 1u)) != 0u ||
        (config->output_mode != SSID_OUTPUT_FULL_MSE && config->output_mode != SSID_OUTPUT_GENOME_WINNERS) ||
        config->reference_device == 0u) {
        ssid_set_error("invalid C99 GP runner configuration");
        return SSID_INVALID_ARGUMENT;
    }
    if ((status = ssid_gp_trace_validate(gp, config->generations, trace)) != SSID_OK) return status;
    if (!ssid_gp_mul_size(gp->config.population_size, gp->config.constant_count, &settings_count) ||
        !ssid_gp_mul_size(gp->config.population_size, gp->input_count, &bindings_count)) return SSID_OUT_OF_RANGE;
    device_settings_count = settings_count;
    device_bindings_count = bindings_count;
    if (gp->template_value->settings_mode == SSID_SETTINGS_MATERIALIZED) {
        if (!ssid_gp_mul_size(settings_count, gp->config.settings_per_genome, &device_settings_count) ||
            !ssid_gp_mul_size(bindings_count, gp->config.settings_per_genome, &device_bindings_count) ||
            !ssid_gp_mul_size(device_settings_count, sizeof(*materialized_settings), &materialized_settings_bytes) ||
            !ssid_gp_mul_size(device_bindings_count, sizeof(*materialized_bindings), &materialized_bindings_bytes)) {
            status = SSID_OUT_OF_RANGE;
            goto cleanup;
        }
        materialized_settings = (float *)malloc(materialized_settings_bytes);
        materialized_bindings = (uint32_t *)malloc(materialized_bindings_bytes);
        if (materialized_settings == NULL || materialized_bindings == NULL) {
            status = SSID_OUT_OF_MEMORY;
            goto cleanup;
        }
    }
    setting_tiles = (gp->config.settings_per_genome + config->threads_per_block - 1u) / config->threads_per_block;
    if (!ssid_gp_mul_size(gp->config.population_size, setting_tiles, &cta_count)) {
        status = SSID_OUT_OF_RANGE;
        goto cleanup;
    }
    batches = (ssid_genome_batch *)calloc(gp->batch_count, sizeof(*batches));
    launches = (ssid_fedbatch_launch *)calloc(gp->batch_count, sizeof(*launches));
    tickets = (ssid_ticket **)calloc(gp->batch_count, sizeof(*tickets));
    if (batches == NULL || launches == NULL || tickets == NULL) {
        status = SSID_OUT_OF_MEMORY;
        goto cleanup;
    }
    if ((status = ssid_gp_device_alloc(pipeline, device_settings_count, sizeof(float), &settings_device)) != SSID_OK ||
        (status = ssid_gp_device_alloc(pipeline, device_bindings_count, sizeof(uint32_t), &bindings_device)) != SSID_OK) goto cleanup;
    if (config->output_mode == SSID_OUTPUT_GENOME_WINNERS) {
        if ((status = ssid_gp_device_alloc(pipeline, cta_count, sizeof(float), &cta_score_device)) != SSID_OK ||
            (status = ssid_gp_device_alloc(pipeline, cta_count, sizeof(uint32_t), &cta_setting_device)) != SSID_OK ||
            (status = ssid_gp_device_alloc(pipeline, gp->config.population_size, sizeof(float), &winner_score_device)) != SSID_OK ||
            (status = ssid_gp_device_alloc(pipeline, gp->config.population_size, sizeof(uint32_t), &winner_setting_device)) != SSID_OK) goto cleanup;
    } else {
        if (!ssid_gp_mul_size(gp->config.population_size, gp->config.settings_per_genome, &full_mse_count) ||
            !ssid_gp_mul_size(full_mse_count, sizeof(*full_mse), &full_mse_bytes)) {
            status = SSID_OUT_OF_RANGE;
            goto cleanup;
        }
        full_mse = (float *)malloc(full_mse_bytes);
        if (full_mse == NULL) {
            status = SSID_OUT_OF_MEMORY;
            goto cleanup;
        }
        if ((status = ssid_gp_device_alloc(pipeline, full_mse_count, sizeof(float), &mse_device)) != SSID_OK) goto cleanup;
    }
    wall_started = ssid_monotonic_seconds();
    for (generation = 0u; generation < config->generations; ++generation) {
        ssid_gp_population *population = &gp->populations[gp->current_population];
        double started = ssid_monotonic_seconds();
        if (gp->template_value->settings_mode == SSID_SETTINGS_MATERIALIZED) {
            if ((status = ssid_gp_settings_generate(gp, materialized_settings, device_settings_count,
                    materialized_bindings, device_bindings_count)) != SSID_OK ||
                (status = ssid_device_upload(pipeline, settings_device, materialized_settings,
                    device_settings_count * sizeof(float))) != SSID_OK ||
                (status = ssid_device_upload(pipeline, bindings_device, materialized_bindings,
                    device_bindings_count * sizeof(uint32_t))) != SSID_OK) goto cleanup;
        } else if ((status = ssid_device_upload(pipeline, settings_device, population->constants,
                settings_count * sizeof(float))) != SSID_OK ||
            (status = ssid_device_upload(pipeline, bindings_device, population->bindings,
                bindings_count * sizeof(uint32_t))) != SSID_OK) goto cleanup;
        stats->upload_seconds += ssid_monotonic_seconds() - started;
        for (batch = 0u; batch < gp->batch_count; ++batch) {
            uint32_t begin;
            uint64_t settings_offset;
            uint64_t bindings_offset;
            uint64_t cta_offset;
            uint64_t winner_offset;
            uint64_t mse_offset;
            if ((status = ssid_gp_batch_get(gp, batch, &batches[batch], &begin)) != SSID_OK) goto cleanup;
            settings_offset = (uint64_t)begin * gp->config.constant_count *
                (gp->template_value->settings_mode == SSID_SETTINGS_MATERIALIZED ? gp->config.settings_per_genome : 1u) * sizeof(float);
            bindings_offset = (uint64_t)begin * gp->input_count *
                (gp->template_value->settings_mode == SSID_SETTINGS_MATERIALIZED ? gp->config.settings_per_genome : 1u) * sizeof(uint32_t);
            cta_offset = (uint64_t)begin * setting_tiles * sizeof(uint32_t);
            winner_offset = (uint64_t)begin * sizeof(uint32_t);
            mse_offset = (uint64_t)begin * gp->config.settings_per_genome * sizeof(float);
            memset(&launches[batch], 0, sizeof(launches[batch]));
            launches[batch].settings_device = settings_device + settings_offset;
            launches[batch].settings_leading_dimension = gp->template_value->settings_mode == SSID_SETTINGS_MATERIALIZED
                ? gp->config.settings_per_genome : gp->config.seed;
            launches[batch].bindings_device = bindings_device + bindings_offset;
            launches[batch].bindings_leading_dimension = gp->template_value->settings_mode == SSID_SETTINGS_MATERIALIZED
                ? gp->config.settings_per_genome : ((uint64_t)gp->generation << 32u) | begin;
            launches[batch].num_settings = gp->config.settings_per_genome;
            launches[batch].num_genomes = batches[batch].genome_count;
            launches[batch].reference_device = config->reference_device;
            launches[batch].steps_per_observation = config->steps_per_observation;
            launches[batch].threads_per_block = config->threads_per_block;
            launches[batch].shared_memory_bytes = config->shared_memory_bytes;
            launches[batch].output_mode = config->output_mode;
            launches[batch].reduction_threads = config->reduction_threads;
            launches[batch].mse_device = mse_device + mse_offset;
            launches[batch].cta_score_device = cta_score_device + cta_offset;
            launches[batch].cta_setting_device = cta_setting_device + cta_offset;
            launches[batch].winner_score_device = winner_score_device + winner_offset;
            launches[batch].winner_setting_device = winner_setting_device + winner_offset;
        }
        started = ssid_monotonic_seconds();
        for (batch = 0u; batch < gp->batch_count; ++batch) {
            status = ssid_pipeline_submit(pipeline, &batches[batch], &launches[batch], &tickets[batch]);
            if (status != SSID_OK) {
                char detail[512];
                snprintf(detail, sizeof(detail), "%s", ssid_last_error());
                ssid_set_error("GP generation %u batch %u submission failed: %s",
                    gp->generation, batch, detail);
                break;
            }
        }
        {
            uint32_t submitted = batch;
            uint32_t wait_index;
            for (wait_index = 0u; wait_index < submitted; ++wait_index) {
                int wait_status = ssid_ticket_wait(tickets[wait_index], NULL);
                if (status == SSID_OK && wait_status != SSID_OK) {
                    ssid_set_error("GP generation %u batch %u pipeline ticket failed: %s",
                        gp->generation, wait_index, ssid_ticket_error(tickets[wait_index]));
                    status = wait_status;
                }
                ssid_ticket_destroy(tickets[wait_index]);
                tickets[wait_index] = NULL;
            }
        }
        stats->pipeline_seconds += ssid_monotonic_seconds() - started;
        if (status != SSID_OK) goto cleanup;
        started = ssid_monotonic_seconds();
        if (config->output_mode == SSID_OUTPUT_GENOME_WINNERS) {
            if ((status = ssid_device_download(pipeline, gp->winner_mse, winner_score_device,
                    gp->config.population_size * sizeof(*gp->winner_mse))) != SSID_OK ||
                (status = ssid_device_download(pipeline, gp->winner_settings, winner_setting_device,
                    gp->config.population_size * sizeof(*gp->winner_settings))) != SSID_OK) goto cleanup;
        } else {
            uint32_t genome;
            if ((status = ssid_device_download(pipeline, full_mse, mse_device,
                    full_mse_count * sizeof(*full_mse))) != SSID_OK) goto cleanup;
            for (genome = 0u; genome < gp->config.population_size; ++genome) {
                uint32_t setting;
                float best_score = FLT_MAX;
                uint32_t best_setting = 0u;
                for (setting = 0u; setting < gp->config.settings_per_genome; ++setting) {
                    float score = full_mse[(size_t)genome * gp->config.settings_per_genome + setting];
                    if (isfinite(score) && score >= gp->config.minimum_valid_mse &&
                        (score < best_score || (score == best_score && setting < best_setting))) {
                        best_score = score;
                        best_setting = setting;
                    }
                }
                gp->winner_mse[genome] = best_score;
                gp->winner_settings[genome] = best_setting;
            }
        }
        stats->download_seconds += ssid_monotonic_seconds() - started;
        started = ssid_monotonic_seconds();
        if (gp->template_value->settings_mode == SSID_SETTINGS_MATERIALIZED) {
            status = ssid_gp_winners_apply(gp, gp->winner_mse, gp->winner_settings,
                materialized_settings, device_settings_count, materialized_bindings, device_bindings_count);
        } else {
            status = ssid_gp_hashed_winners_apply(gp, gp->winner_mse, gp->winner_settings);
        }
        if (status != SSID_OK) goto cleanup;
        stats->winner_materialization_seconds += ssid_monotonic_seconds() - started;
        if (trace != NULL &&
            (generation % trace->checkpoint_stride == 0u || generation + 1u == config->generations)) {
            ssid_gp_trace_capture(gp, trace, checkpoint_index++, ssid_monotonic_seconds() - wall_started);
        }
        if (generation + 1u < config->generations) {
            started = ssid_monotonic_seconds();
            status = ssid_gp_generation_advance(gp);
            stats->evolution_seconds += ssid_monotonic_seconds() - started;
            if (status != SSID_OK) {
                char detail[512];
                snprintf(detail, sizeof(detail), "%s", ssid_last_error());
                ssid_set_error("GP generation %u advance failed: %s", gp->generation, detail);
                goto cleanup;
            }
        }
    }
    stats->wall_seconds = ssid_monotonic_seconds() - wall_started;
    stats->generations = config->generations;
    stats->batches_per_generation = gp->batch_count;
    stats->evaluated_genomes = (uint64_t)config->generations * gp->config.population_size;
    stats->evaluated_configurations = stats->evaluated_genomes * gp->config.settings_per_genome;
    stats->genomes_per_second = stats->evaluated_genomes / stats->wall_seconds;
    stats->configurations_per_second = stats->evaluated_configurations / stats->wall_seconds;
    stats->best_mse = gp->best_mse;
    stats->best_objective = gp->best_objective;
    stats->best_generation = gp->best_generation;
    stats->best_complexity = gp->best_complexity;

cleanup:
    if (tickets != NULL) {
        for (batch = 0u; batch < gp->batch_count; ++batch) {
            if (tickets[batch] != NULL) {
                (void)ssid_ticket_wait(tickets[batch], NULL);
                ssid_ticket_destroy(tickets[batch]);
            }
        }
    }
    if (winner_setting_device != 0u) (void)ssid_device_free(pipeline, winner_setting_device);
    if (winner_score_device != 0u) (void)ssid_device_free(pipeline, winner_score_device);
    if (cta_setting_device != 0u) (void)ssid_device_free(pipeline, cta_setting_device);
    if (cta_score_device != 0u) (void)ssid_device_free(pipeline, cta_score_device);
    if (mse_device != 0u) (void)ssid_device_free(pipeline, mse_device);
    if (bindings_device != 0u) (void)ssid_device_free(pipeline, bindings_device);
    if (settings_device != 0u) (void)ssid_device_free(pipeline, settings_device);
    free(tickets);
    free(launches);
    free(batches);
    free(full_mse);
    free(materialized_bindings);
    free(materialized_settings);
    return status;
}

int ssid_gp_run(ssid_gp *gp, ssid_pipeline *pipeline, const ssid_gp_run_config *config, ssid_gp_run_stats *stats) {
    return ssid_gp_run_internal(gp, pipeline, config, stats, NULL);
}

int ssid_gp_run_traced(ssid_gp *gp, ssid_pipeline *pipeline, const ssid_gp_run_config *config, ssid_gp_run_stats *stats, ssid_gp_trace *trace) {
    if (trace == NULL) {
        ssid_set_error("C99 GP traced run requires trace storage");
        return SSID_INVALID_ARGUMENT;
    }
    return ssid_gp_run_internal(gp, pipeline, config, stats, trace);
}
