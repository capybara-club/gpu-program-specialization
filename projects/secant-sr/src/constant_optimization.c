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
#include "internal.h"

#define SECANT_SR_LBFGS_HISTORY 4u
#define SECANT_SR_LINE_SEARCH_STEPS 16u

typedef enum SecantSRObjectiveStatus {
    SECANT_SR_OBJECTIVE_OK = 0,
    SECANT_SR_OBJECTIVE_LIMIT = 1,
    SECANT_SR_OBJECTIVE_ERROR = 2
} SecantSRObjectiveStatus;

static uint64_t
secant_sr_optimizer_rng_next(SecantSRSearch search) {
    uint64_t value = search->optimizer_rng_state;

    value ^= value >> 12u;
    value ^= value << 25u;
    value ^= value >> 27u;
    search->optimizer_rng_state = value;
    return value * UINT64_C(2685821657736338717);
}

double
secant_sr_optimizer_rng_unit(SecantSRSearch search) {
    return (double)(secant_sr_optimizer_rng_next(search) >> 11u) * (1.0 / 9007199254740992.0);
}

static void
secant_sr_constant_write(SecantAstInstruction* program, size_t instruction_offset, double value) {
    const float encoded_value = (float)value;
    uint32_t bits;

    memcpy(&bits, &encoded_value, sizeof(bits));
    program[instruction_offset + 1u] = (uint8_t)(bits >> 0u);
    program[instruction_offset + 2u] = (uint8_t)(bits >> 8u);
    program[instruction_offset + 3u] = (uint8_t)(bits >> 16u);
    program[instruction_offset + 4u] = (uint8_t)(bits >> 24u);
}

static void
secant_sr_constants_write(
    SecantAstInstruction* program,
    const size_t* constant_offsets,
    const double* constants,
    size_t num_constants
) {
    size_t constant_idx;

    for (constant_idx = 0u; constant_idx < num_constants; ++constant_idx) {
        secant_sr_constant_write(program, constant_offsets[constant_idx], constants[constant_idx]);
    }
}

static size_t
secant_sr_constants_read(
    const SecantSRIndividual* individual,
    size_t* constant_offsets,
    double* constants
) {
    size_t num_constants = 0u;
    size_t node_idx;

    for (node_idx = 0u; node_idx < individual->num_nodes; ++node_idx) {
        const size_t instruction_offset = individual->nodes[node_idx].instruction_offset;
        const SecantAstInstruction* instruction = individual->program + instruction_offset;

        if (secant_ast_instruction_type_get(instruction) == SECANT_AST_INSTRUCTION_TYPE_CONSTANT_BITS_F32) {
            constant_offsets[num_constants] = instruction_offset;
            constants[num_constants] = (double)secant_ast_constant_f32_get(instruction);
            ++num_constants;
        }
    }
    return num_constants;
}

static double
secant_sr_vector_dot(const double* left, const double* right, size_t count) {
    double value = 0.0;
    size_t idx;

    for (idx = 0u; idx < count; ++idx) {
        value += left[idx] * right[idx];
    }
    return value;
}

static SecantSRObjectiveStatus
secant_sr_constant_objective(
    SecantSRSearch search,
    SecantAstInstruction* program,
    const size_t* constant_offsets,
    const double* constants,
    size_t num_constants,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const float* input,
    size_t input_num_elements,
    size_t input_leading_dimension,
    const float* target,
    size_t target_num_elements,
    size_t num_rows,
    size_t f_calls_limit,
    size_t* f_calls,
    double* objective_ret
) {
    const SecantAstInstruction* asts[] = {program};
    SecantCpuSSERun run = secant_cpu_sse_run_init();
    float sse = 0.0f;

    if (*f_calls >= f_calls_limit) {
        return SECANT_SR_OBJECTIVE_LIMIT;
    }
    secant_sr_constants_write(program, constant_offsets, constants, num_constants);
    ++*f_calls;
    run.programs.routines.items = routines;
    run.programs.routines.count = num_routines;
    run.programs.asts.items = asts;
    run.programs.asts.count = 1u;
    run.num_inputs = search->config.num_inputs;
    run.num_targets = 1u;
    run.input = (SecantConstHostMatrixF32){input, input_num_elements, input_leading_dimension};
    run.targets = (SecantConstHostMatrixF32){target, target_num_elements, num_rows};
    run.num_rows = num_rows;
    run.output = (SecantHostMatrixF32){&sse, 1u, 1u};
    if (secant_cpu_run_sse(&run) != SECANT_SUCCESS) {
        return SECANT_SR_OBJECTIVE_ERROR;
    }
    *objective_ret = isfinite(sse) && sse >= 0.0f ? (double)sse / (double)num_rows : DBL_MAX;
    return SECANT_SR_OBJECTIVE_OK;
}

static SecantSRObjectiveStatus
secant_sr_constant_gradient(
    SecantSRSearch search,
    SecantAstInstruction* program,
    const size_t* constant_offsets,
    double* constants,
    size_t num_constants,
    double objective,
    double finite_difference_relative_step,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const float* input,
    size_t input_num_elements,
    size_t input_leading_dimension,
    const float* target,
    size_t target_num_elements,
    size_t num_rows,
    size_t f_calls_limit,
    size_t* f_calls,
    double* gradient
) {
    size_t constant_idx;

    for (constant_idx = 0u; constant_idx < num_constants; ++constant_idx) {
        const double original = constants[constant_idx];
        const double scale = fmax(1.0, fabs(original));
        const double step = fmax(finite_difference_relative_step * scale, 4.0 * FLT_EPSILON * scale);
        double plus_objective;
        double minus_objective;
        SecantSRObjectiveStatus status;

        constants[constant_idx] = original + step;
        status = secant_sr_constant_objective(
            search,
            program,
            constant_offsets,
            constants,
            num_constants,
            routines,
            num_routines,
            input,
            input_num_elements,
            input_leading_dimension,
            target,
            target_num_elements,
            num_rows,
            f_calls_limit,
            f_calls,
            &plus_objective);
        if (status != SECANT_SR_OBJECTIVE_OK) {
            constants[constant_idx] = original;
            return status;
        }
        constants[constant_idx] = original - step;
        status = secant_sr_constant_objective(
            search,
            program,
            constant_offsets,
            constants,
            num_constants,
            routines,
            num_routines,
            input,
            input_num_elements,
            input_leading_dimension,
            target,
            target_num_elements,
            num_rows,
            f_calls_limit,
            f_calls,
            &minus_objective);
        constants[constant_idx] = original;
        if (status != SECANT_SR_OBJECTIVE_OK) {
            return status;
        }

        if (plus_objective < DBL_MAX && minus_objective < DBL_MAX) {
            gradient[constant_idx] = (plus_objective - minus_objective) / (2.0 * step);
        } else if (plus_objective < DBL_MAX && objective < DBL_MAX) {
            gradient[constant_idx] = (plus_objective - objective) / step;
        } else if (minus_objective < DBL_MAX && objective < DBL_MAX) {
            gradient[constant_idx] = (objective - minus_objective) / step;
        } else {
            gradient[constant_idx] = 0.0;
        }
    }
    return SECANT_SR_OBJECTIVE_OK;
}

static double
secant_sr_normal_random(SecantSRSearch search) {
    double first = secant_sr_optimizer_rng_unit(search);
    const double second = secant_sr_optimizer_rng_unit(search);

    if (first < DBL_MIN) {
        first = DBL_MIN;
    }
    return sqrt(-2.0 * log(first)) * cos(6.28318530717958647692 * second);
}

static void
secant_sr_lbfgs_direction(
    const double* gradient,
    size_t num_constants,
    const double* history_s,
    const double* history_y,
    const double* history_rho,
    size_t history_count,
    double* history_alpha,
    double* direction
) {
    size_t history_idx;
    size_t constant_idx;
    double gamma = 1.0;

    memcpy(direction, gradient, num_constants * sizeof(*direction));
    for (history_idx = history_count; history_idx-- > 0u;) {
        const double* s = history_s + history_idx * SECANT_AST_MAX_PROGRAM_INSTRUCTIONS;
        const double* y = history_y + history_idx * SECANT_AST_MAX_PROGRAM_INSTRUCTIONS;

        history_alpha[history_idx] = history_rho[history_idx] * secant_sr_vector_dot(s, direction, num_constants);
        for (constant_idx = 0u; constant_idx < num_constants; ++constant_idx) {
            direction[constant_idx] -= history_alpha[history_idx] * y[constant_idx];
        }
    }
    if (history_count != 0u) {
        const double* newest_s = history_s + (history_count - 1u) * SECANT_AST_MAX_PROGRAM_INSTRUCTIONS;
        const double* newest_y = history_y + (history_count - 1u) * SECANT_AST_MAX_PROGRAM_INSTRUCTIONS;
        const double yy = secant_sr_vector_dot(newest_y, newest_y, num_constants);

        if (yy > DBL_MIN) {
            gamma = secant_sr_vector_dot(newest_s, newest_y, num_constants) / yy;
        }
    }
    for (constant_idx = 0u; constant_idx < num_constants; ++constant_idx) {
        direction[constant_idx] *= gamma;
    }
    for (history_idx = 0u; history_idx < history_count; ++history_idx) {
        const double* s = history_s + history_idx * SECANT_AST_MAX_PROGRAM_INSTRUCTIONS;
        const double* y = history_y + history_idx * SECANT_AST_MAX_PROGRAM_INSTRUCTIONS;
        const double beta = history_rho[history_idx] * secant_sr_vector_dot(y, direction, num_constants);

        for (constant_idx = 0u; constant_idx < num_constants; ++constant_idx) {
            direction[constant_idx] += s[constant_idx] * (history_alpha[history_idx] - beta);
        }
    }
    for (constant_idx = 0u; constant_idx < num_constants; ++constant_idx) {
        direction[constant_idx] = -direction[constant_idx];
    }
}

static void
secant_sr_lbfgs_history_append(
    const double* constants,
    const double* next_constants,
    const double* gradient,
    const double* next_gradient,
    size_t num_constants,
    double* history_s,
    double* history_y,
    double* history_rho,
    size_t* history_count
) {
    size_t slot = *history_count;
    double* s;
    double* y;
    double ys;
    size_t constant_idx;

    if (slot == SECANT_SR_LBFGS_HISTORY) {
        memmove(
            history_s,
            history_s + SECANT_AST_MAX_PROGRAM_INSTRUCTIONS,
            (SECANT_SR_LBFGS_HISTORY - 1u) * SECANT_AST_MAX_PROGRAM_INSTRUCTIONS * sizeof(*history_s));
        memmove(
            history_y,
            history_y + SECANT_AST_MAX_PROGRAM_INSTRUCTIONS,
            (SECANT_SR_LBFGS_HISTORY - 1u) * SECANT_AST_MAX_PROGRAM_INSTRUCTIONS * sizeof(*history_y));
        memmove(history_rho, history_rho + 1u, (SECANT_SR_LBFGS_HISTORY - 1u) * sizeof(*history_rho));
        slot = SECANT_SR_LBFGS_HISTORY - 1u;
    }
    s = history_s + slot * SECANT_AST_MAX_PROGRAM_INSTRUCTIONS;
    y = history_y + slot * SECANT_AST_MAX_PROGRAM_INSTRUCTIONS;
    for (constant_idx = 0u; constant_idx < num_constants; ++constant_idx) {
        s[constant_idx] = next_constants[constant_idx] - constants[constant_idx];
        y[constant_idx] = next_gradient[constant_idx] - gradient[constant_idx];
    }
    ys = secant_sr_vector_dot(y, s, num_constants);
    if (isfinite(ys) && ys > 1.0e-12) {
        history_rho[slot] = 1.0 / ys;
        if (*history_count < SECANT_SR_LBFGS_HISTORY) {
            ++*history_count;
        }
    }
}

static SecantSRResult
secant_sr_individual_constants_optimize(
    SecantSRSearch search,
    SecantSRIndividual* individual,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const float* input,
    size_t input_num_elements,
    size_t input_leading_dimension,
    const float* target,
    size_t target_num_elements,
    size_t optimizer_num_rows,
    size_t num_rows,
    size_t optimizer_iterations,
    size_t optimizer_restarts,
    size_t optimizer_f_calls_limit,
    double finite_difference_relative_step,
    size_t* constant_offsets,
    double* original_constants,
    double* constants,
    double* best_constants,
    double* gradient,
    double* next_gradient,
    double* direction,
    double* next_constants,
    double* history_s,
    double* history_y,
    size_t* num_f_calls_ret,
    int* improved_ret,
    double* best_objective_ret
) {
    double history_rho[SECANT_SR_LBFGS_HISTORY];
    double history_alpha[SECANT_SR_LBFGS_HISTORY];
    SecantAstInstruction* program = (SecantAstInstruction*)individual->program;
    const size_t num_constants = secant_sr_constants_read(individual, constant_offsets, original_constants);
    double baseline_objective;
    double best_objective;
    size_t f_calls = 0u;
    size_t start_idx;
    SecantSRObjectiveStatus objective_status;

    *num_f_calls_ret = 0u;
    *improved_ret = 0;
    if (num_constants == 0u) {
        *best_objective_ret = individual->fitness.sse;
        return SECANT_SR_SUCCESS;
    }
    objective_status = secant_sr_constant_objective(
        search,
        program,
        constant_offsets,
        original_constants,
        num_constants,
        routines,
        num_routines,
        input,
        input_num_elements,
        input_leading_dimension,
        target,
        target_num_elements,
        optimizer_num_rows,
        optimizer_f_calls_limit,
        &f_calls,
        &baseline_objective);
    if (objective_status != SECANT_SR_OBJECTIVE_OK) {
        secant_sr_constants_write(program, constant_offsets, original_constants, num_constants);
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_SECANT);
    }
    best_objective = baseline_objective;
    memcpy(best_constants, original_constants, num_constants * sizeof(*best_constants));

    for (start_idx = 0u; start_idx <= optimizer_restarts && f_calls < optimizer_f_calls_limit; ++start_idx) {
        double objective = baseline_objective;
        size_t history_count = 0u;
        size_t iteration;
        size_t constant_idx;

        if (start_idx == 0u) {
            memcpy(constants, original_constants, num_constants * sizeof(*constants));
        } else {
            for (constant_idx = 0u; constant_idx < num_constants; ++constant_idx) {
                constants[constant_idx] = original_constants[constant_idx] *
                    (1.0 + 0.5 * secant_sr_normal_random(search));
            }
            objective_status = secant_sr_constant_objective(
                search,
                program,
                constant_offsets,
                constants,
                num_constants,
                routines,
                num_routines,
                input,
                input_num_elements,
                input_leading_dimension,
                target,
                target_num_elements,
                optimizer_num_rows,
                optimizer_f_calls_limit,
                &f_calls,
                &objective);
            if (objective_status == SECANT_SR_OBJECTIVE_LIMIT) {
                break;
            }
            if (objective_status == SECANT_SR_OBJECTIVE_ERROR) {
                secant_sr_constants_write(program, constant_offsets, original_constants, num_constants);
                _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_SECANT);
            }
        }
        if (objective >= DBL_MAX) {
            continue;
        }
        if (objective < best_objective) {
            best_objective = objective;
            memcpy(best_constants, constants, num_constants * sizeof(*best_constants));
        }
        objective_status = secant_sr_constant_gradient(
            search,
            program,
            constant_offsets,
            constants,
            num_constants,
            objective,
            finite_difference_relative_step,
            routines,
            num_routines,
            input,
            input_num_elements,
            input_leading_dimension,
            target,
            target_num_elements,
            optimizer_num_rows,
            optimizer_f_calls_limit,
            &f_calls,
            gradient);
        if (objective_status == SECANT_SR_OBJECTIVE_LIMIT) {
            break;
        }
        if (objective_status == SECANT_SR_OBJECTIVE_ERROR) {
            secant_sr_constants_write(program, constant_offsets, original_constants, num_constants);
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_SECANT);
        }

        for (iteration = 0u; iteration < optimizer_iterations && f_calls < optimizer_f_calls_limit; ++iteration) {
            double directional_derivative;
            double next_objective = DBL_MAX;
            double line_step = 1.0;
            int accepted = 0;
            size_t line_idx;

            if (secant_sr_vector_dot(gradient, gradient, num_constants) <= 1.0e-18) {
                break;
            }
            secant_sr_lbfgs_direction(
                gradient,
                num_constants,
                history_s,
                history_y,
                history_rho,
                history_count,
                history_alpha,
                direction);
            directional_derivative = secant_sr_vector_dot(gradient, direction, num_constants);
            if (!isfinite(directional_derivative) || directional_derivative >= 0.0) {
                history_count = 0u;
                for (constant_idx = 0u; constant_idx < num_constants; ++constant_idx) {
                    direction[constant_idx] = -gradient[constant_idx];
                }
                directional_derivative = -secant_sr_vector_dot(gradient, gradient, num_constants);
            }

            for (line_idx = 0u; line_idx < SECANT_SR_LINE_SEARCH_STEPS; ++line_idx) {
                for (constant_idx = 0u; constant_idx < num_constants; ++constant_idx) {
                    next_constants[constant_idx] = constants[constant_idx] + line_step * direction[constant_idx];
                }
                objective_status = secant_sr_constant_objective(
                    search,
                    program,
                    constant_offsets,
                    next_constants,
                    num_constants,
                    routines,
                    num_routines,
                    input,
                    input_num_elements,
                    input_leading_dimension,
                    target,
                    target_num_elements,
                    optimizer_num_rows,
                    optimizer_f_calls_limit,
                    &f_calls,
                    &next_objective);
                if (objective_status == SECANT_SR_OBJECTIVE_LIMIT) {
                    break;
                }
                if (objective_status == SECANT_SR_OBJECTIVE_ERROR) {
                    secant_sr_constants_write(program, constant_offsets, original_constants, num_constants);
                    _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_SECANT);
                }
                if (next_objective <= objective + 1.0e-4 * line_step * directional_derivative) {
                    accepted = 1;
                    break;
                }
                line_step *= 0.5;
            }
            if (!accepted) {
                break;
            }
            objective_status = secant_sr_constant_gradient(
                search,
                program,
                constant_offsets,
                next_constants,
                num_constants,
                next_objective,
                finite_difference_relative_step,
                routines,
                num_routines,
                input,
                input_num_elements,
                input_leading_dimension,
                target,
                target_num_elements,
                optimizer_num_rows,
                optimizer_f_calls_limit,
                &f_calls,
                next_gradient);
            if (objective_status != SECANT_SR_OBJECTIVE_OK) {
                if (objective_status == SECANT_SR_OBJECTIVE_ERROR) {
                    secant_sr_constants_write(program, constant_offsets, original_constants, num_constants);
                    _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_SECANT);
                }
                break;
            }
            secant_sr_lbfgs_history_append(
                constants,
                next_constants,
                gradient,
                next_gradient,
                num_constants,
                history_s,
                history_y,
                history_rho,
                &history_count);
            memcpy(constants, next_constants, num_constants * sizeof(*constants));
            memcpy(gradient, next_gradient, num_constants * sizeof(*gradient));
            objective = next_objective;
            if (objective < best_objective) {
                best_objective = objective;
                memcpy(best_constants, constants, num_constants * sizeof(*best_constants));
            }
        }
    }

    secant_sr_constants_write(program, constant_offsets, best_constants, num_constants);
    if (optimizer_num_rows != num_rows) {
        double full_objective;

        objective_status = secant_sr_constant_objective(
            search,
            program,
            constant_offsets,
            best_constants,
            num_constants,
            routines,
            num_routines,
            input,
            input_num_elements,
            input_leading_dimension,
            target,
            target_num_elements,
            num_rows,
            SIZE_MAX,
            &f_calls,
            &full_objective);
        if (objective_status != SECANT_SR_OBJECTIVE_OK) {
            secant_sr_constants_write(program, constant_offsets, original_constants, num_constants);
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_SECANT);
        }
        *best_objective_ret = full_objective < DBL_MAX ? full_objective * (double)num_rows : DBL_MAX;
    } else {
        *best_objective_ret = best_objective < DBL_MAX ? best_objective * (double)num_rows : DBL_MAX;
    }
    *num_f_calls_ret = f_calls;
    *improved_ret = best_objective < baseline_objective;
    return SECANT_SR_SUCCESS;
}

SecantSRResult
secant_sr_search_best_constants_optimize_cpu(
    SecantSRSearch search,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const float* input,
    size_t input_num_elements,
    size_t input_leading_dimension,
    const float* target,
    size_t target_num_elements,
    size_t optimizer_num_rows,
    size_t num_rows,
    double target_sum_squared_deviation,
    size_t optimizer_iterations,
    size_t optimizer_restarts,
    size_t optimizer_f_calls_limit,
    double finite_difference_relative_step,
    size_t* num_constants_ret,
    int* improved_ret,
    size_t* num_f_calls_ret
) {
    size_t constant_offsets[SECANT_AST_MAX_PROGRAM_INSTRUCTIONS];
    double original_constants[SECANT_AST_MAX_PROGRAM_INSTRUCTIONS];
    double constants[SECANT_AST_MAX_PROGRAM_INSTRUCTIONS];
    double best_constants[SECANT_AST_MAX_PROGRAM_INSTRUCTIONS];
    double gradient[SECANT_AST_MAX_PROGRAM_INSTRUCTIONS];
    double next_gradient[SECANT_AST_MAX_PROGRAM_INSTRUCTIONS];
    double direction[SECANT_AST_MAX_PROGRAM_INSTRUCTIONS];
    double next_constants[SECANT_AST_MAX_PROGRAM_INSTRUCTIONS];
    double history_s[SECANT_SR_LBFGS_HISTORY * SECANT_AST_MAX_PROGRAM_INSTRUCTIONS];
    double history_y[SECANT_SR_LBFGS_HISTORY * SECANT_AST_MAX_PROGRAM_INSTRUCTIONS];
    SecantSRIndividual* individual = NULL;
    double original_objective;
    double original_sse;
    double optimized_sse;
    size_t baseline_f_calls = 0u;
    size_t num_constants;
    size_t slot_idx;
    size_t num_f_calls = 0u;
    int optimizer_improved;

    if (num_constants_ret != NULL) {
        *num_constants_ret = 0u;
    }
    if (improved_ret != NULL) {
        *improved_ret = 0;
    }
    if (num_f_calls_ret != NULL) {
        *num_f_calls_ret = 0u;
    }
    if (search == NULL || input == NULL || target == NULL || optimizer_num_rows == 0u || num_rows == 0u ||
        optimizer_num_rows > num_rows || target_num_elements < num_rows ||
        target_sum_squared_deviation <= 0.0 || optimizer_iterations == 0u || optimizer_f_calls_limit == 0u ||
        finite_difference_relative_step <= 0.0 ||
        !isfinite(finite_difference_relative_step) || (num_routines != 0u && routines == NULL)) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
    }

    for (slot_idx = 0u; slot_idx < search->num_elite_slots; ++slot_idx) {
        SecantSREliteSlot* slot = search->elites + slot_idx;

        if (slot->valid && (individual == NULL || slot->individual.fitness.score > individual->fitness.score)) {
            individual = &slot->individual;
        }
    }
    if (individual == NULL) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_NOT_SCORED);
    }
    num_constants = secant_sr_constants_read(individual, constant_offsets, original_constants);
    if (num_constants_ret != NULL) {
        *num_constants_ret = num_constants;
    }
    if (num_constants == 0u) {
        return SECANT_SR_SUCCESS;
    }
    if (secant_sr_constant_objective(
            search,
            (SecantAstInstruction*)individual->program,
            constant_offsets,
            original_constants,
            num_constants,
            routines,
            num_routines,
            input,
            input_num_elements,
            input_leading_dimension,
            target,
            target_num_elements,
            num_rows,
            SIZE_MAX,
            &baseline_f_calls,
            &original_objective) != SECANT_SR_OBJECTIVE_OK) {
        secant_sr_constants_write(
            (SecantAstInstruction*)individual->program, constant_offsets, original_constants, num_constants);
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_SECANT);
    }
    original_sse = original_objective < DBL_MAX ? original_objective * (double)num_rows : DBL_MAX;
    _SECANT_SR_CHECK_RET(secant_sr_individual_constants_optimize(
        search,
        individual,
        routines,
        num_routines,
        input,
        input_num_elements,
        input_leading_dimension,
        target,
        target_num_elements,
        optimizer_num_rows,
        num_rows,
        optimizer_iterations,
        optimizer_restarts,
        optimizer_f_calls_limit,
        finite_difference_relative_step,
        constant_offsets,
        original_constants,
        constants,
        best_constants,
        gradient,
        next_gradient,
        direction,
        next_constants,
        history_s,
        history_y,
        &num_f_calls,
        &optimizer_improved,
        &optimized_sse));
    num_f_calls += baseline_f_calls;
    (void)optimizer_improved;
    if (isfinite(optimized_sse) && optimized_sse < original_sse) {
        size_t annotated_program_bytes;
        size_t annotated_num_nodes;
        size_t annotated_num_leaves;
        size_t annotated_num_constants;
        uint32_t annotated_complexity;
        uint16_t annotated_depth;
        uint64_t annotated_fingerprint;

        const SecantSRResult annotate_result = secant_sr_program_annotate(
            individual->program,
            individual->program_bytes,
            search->routines,
            search->num_routines,
            search->config.max_depth,
            search->config.max_complexity,
            (SecantSRNodeInfo*)individual->nodes,
            search->config.max_nodes,
            &annotated_program_bytes,
            &annotated_num_nodes,
            &annotated_num_leaves,
            &annotated_num_constants,
            &annotated_complexity,
            &annotated_depth,
            &annotated_fingerprint);

        if (annotate_result != SECANT_SR_SUCCESS) {
            secant_sr_constants_write(
                (SecantAstInstruction*)individual->program, constant_offsets, original_constants, num_constants);
            _SECANT_SR_ERROR_RET(annotate_result);
        }
        if (annotated_program_bytes != individual->program_bytes || annotated_num_nodes != individual->num_nodes ||
            annotated_num_leaves != individual->num_leaves || annotated_num_constants != individual->num_constants ||
            annotated_complexity != individual->complexity || annotated_depth != individual->depth) {
            secant_sr_constants_write(
                (SecantAstInstruction*)individual->program, constant_offsets, original_constants, num_constants);
            _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_BAD_PROGRAM);
        }
        individual->fingerprint = annotated_fingerprint;
        secant_sr_individual_score_set(search, individual, optimized_sse, num_rows, target_sum_squared_deviation);
        if (improved_ret != NULL) {
            *improved_ret = 1;
        }
    } else {
        secant_sr_constants_write(
            (SecantAstInstruction*)individual->program, constant_offsets, original_constants, num_constants);
    }
    if (num_f_calls_ret != NULL) {
        *num_f_calls_ret = num_f_calls;
    }
    return SECANT_SR_SUCCESS;
}
