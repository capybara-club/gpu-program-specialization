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
#ifndef ODEZZA_REQUEST_H
#define ODEZZA_REQUEST_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* CPU-only, C99. Every parser/generator buffer belongs to the caller.
 * No heap allocation, CUDA dependency, database, or filesystem access.
 * NULL arena measures; a short arena returns ODR_BUFFER with required_bytes.
 * Failed calls may write inside the supplied extent; outputs are valid only
 * on success. Arenas must be aligned to odr_arena_alignment(). */
typedef enum OdrResult {
    ODR_OK=0, ODR_BUFFER, ODR_JSON, ODR_SCHEMA, ODR_UNSUPPORTED,
    ODR_OVERFLOW, ODR_AST, ODR_INDEX, ODR_CAPACITY, ODR_DONE
} OdrResult;
typedef struct OdrError { OdrResult code; size_t offset; char message[192]; } OdrError;
typedef struct OdrJson { const char *data; size_t size; } OdrJson;
size_t odr_arena_alignment(void);
const char *odr_result_string(OdrResult result);
/* Borrowed slice into validated input. Missing field returns ODR_SCHEMA. */
OdrResult odr_section(OdrJson json, const char *key, OdrJson *out, OdrError *error);

typedef struct OdrTrajectoryInfo {
    uint32_t first_point, point_count, observed_count, uniform_spacing;
    float first_time, last_time, min_dt, max_dt, dt;
} OdrTrajectoryInfo;
typedef struct OdrTrajectories {
    uint32_t state_count, trajectory_count, point_count, allow_missing_observations;
    uint64_t observed_count;
    const char *const *states;
    const uint32_t *offsets; /* trajectory_count+1 */
    const float *times;     /* point_count */
    const float *values;    /* state-major: state*point_count+point */
    const OdrTrajectoryInfo *info;
} OdrTrajectories;
/* Accepts {states,trajectories[,known_rhs]} or a full request's problem object.
 * FP32 RK4 format only; complete initial vectors; null/mask at later points
 * become NaNs. Validates after FP32 conversion, including distinct times. */
OdrResult odr_trajectories_parse(OdrJson json, void *arena, size_t capacity,
    size_t *required_bytes, const OdrTrajectories **out, OdrError *error);
typedef struct OdrRk4Layout {
    uint32_t steps_per_observation;
    uint64_t steps_per_configuration;
} OdrRk4Layout;
OdrResult odr_trajectories_rk4(const OdrTrajectories *trajectories,
    float max_dt, uint64_t max_steps, OdrRk4Layout *out, OdrError *error);
/* Full request preflight, without allocations/AST expansion. Validates the
 * trajectory section and integration settings using the same FP32 layout rules
 * as execution. ODR_CAPACITY leaves required work counts available in out. */
typedef struct OdrRk4Work {
    uint32_t steps_per_observation;
    uint64_t steps_per_configuration;
    uint32_t trajectory_count, point_count;
} OdrRk4Work;
OdrResult odr_request_rk4_work(OdrJson request, OdrRk4Work *out, OdrError *error);

typedef struct OdrProgram { const unsigned char *bytes; size_t byte_count; } OdrProgram;
typedef struct OdrRhs { uint32_t state_index; OdrProgram program; } OdrRhs;
typedef struct OdrStatic {
    uint32_t state_count, rhs_count;
    const char *const *states;
    const OdrRhs *rhs;
} OdrStatic;
/* Accepts {states,known_rhs}; missing equations remain variable. */
OdrResult odr_static_parse(OdrJson json, void *arena, size_t capacity,
    size_t *required_bytes, const OdrStatic **out, OdrError *error);

typedef struct OdrGrammar OdrGrammar;
typedef struct OdrProducer OdrProducer;
typedef struct OdrFamilyInfo {
    const char *name;
    uint64_t max_skeletons, max_variants, max_configurations, max_derivations;
    uint64_t max_expansion_steps;
    uint64_t max_configurations_per_skeleton; /* zero: no additional cap */
} OdrFamilyInfo;
/* Declared reserved work, not cardinality of the eventual deduplicated output.
 * Uses the same allocation rules as grammar preparation. No ASTs or arenas. */
typedef struct OdrAllocationInfo {
    uint64_t families, skeletons, variants, configurations, derivations, expansion_steps;
} OdrAllocationInfo;
OdrResult odr_request_allocations(OdrJson request,OdrAllocationInfo *out,OdrError *error);
typedef struct OdrGrammarInfo {
    float rk4_max_dt; /* zero: caller must supply an integration step */
    uint64_t rk4_max_steps; /* per configuration; zero: unspecified */
    float max_seconds; /* scheduler deadline; producer does not read a clock */
    OdrJson retention; /* arena-owned policy JSON; consumed by retention phase */
} OdrGrammarInfo;
typedef enum OdrStopReason {
    ODR_STOP_NONE=0, ODR_STOP_EXHAUSTED, ODR_STOP_VARIANTS,
    ODR_STOP_SKELETONS, ODR_STOP_DERIVATIONS, ODR_STOP_EXPANSION_STEPS,
    ODR_STOP_CONFIGURATIONS
} OdrStopReason;
/* Duplicate executions are omitted, but later tag memberships can be retained.
 * Synchronous callback; tags are borrowed until return; do not reenter producer. */
typedef void (*OdrDuplicateProvenance)(void *user, uint64_t candidate_index,
    uint64_t derivation_index, uint64_t variant_index,
    const char *const *tags, size_t tag_count);
typedef struct OdrProducerOptions {
    uint64_t max_asts, max_attempts;
    size_t dedup_capacity; /* bytes of exact canonical-program storage */
    uint32_t family_index;
    uint64_t max_attempts_per_batch; /* default 4096; bounds derivations AND variant visits */
    OdrDuplicateProvenance on_duplicate;
    void *provenance_user;
} OdrProducerOptions;
typedef struct OdrAxis {
    uint32_t kind; /* 0=fixed, 1=constant grid, 2=uniform, 3=normal */
    const char *name, *axis, *stream;
    const float *values;
    uint64_t count, seed;
    uint32_t transform; /* identity, affine, uniform, normal, log_uniform */
    float scale, shift; /* uniform/log_uniform: low, high; normal: std, mean */
    const char *bank;
    uint32_t skeleton_scope;
    int32_t scale_slot, shift_slot; /* -1: literal; otherwise coefficient slot */
    uint64_t numeric_stride; /* axis index = (bank_index/stride)%count; fixed: 0 */
    uint32_t instance; /* local binding occurrence; zero for global bindings */
} OdrAxis;
typedef struct OdrCandidate {
    uint64_t index, derivation_index, variant_index;
    uint32_t family_index, rhs_count, slot_count, toggle_bits;
    uint64_t numeric_count;
    uint64_t bank_start, bank_count; /* allocated prefix; full toggle set per row */
    const OdrRhs *rhs;
    const OdrAxis *slots;
    const char *family;
    const char *const *tags;
    size_t tag_count;
} OdrCandidate;
typedef struct OdrBatch {
    uint64_t start, next_index, attempts, duplicates, pruned;
    size_t count;
    const OdrCandidate *candidates;
    int exhausted;
    int yielded;
    uint64_t expansion_steps, configurations_reserved; /* planned work, not GPU completions */
    uint64_t variant_visits;
    uint64_t unused_configurations;
    OdrStopReason stop_reason;
    uint64_t configuration_limited_derivations;
} OdrBatch;
OdrResult odr_grammar_parse(OdrJson json, const OdrStatic *fixed,
    void *arena, size_t capacity, size_t *required_bytes,
    const OdrGrammar **out, OdrError *error);
size_t odr_grammar_family_count(const OdrGrammar *grammar);
const char *odr_grammar_family_name(const OdrGrammar *grammar, size_t index);
OdrResult odr_grammar_info(const OdrGrammar *grammar, OdrGrammarInfo *out);
OdrResult odr_grammar_family_info(const OdrGrammar *grammar, size_t index,
    OdrFamilyInfo *out);
/* Enumeration matches v1 ordering; native sampling has its own stable profile. */
const char *odr_grammar_sampling_profile(void);
OdrResult odr_producer_create(const OdrGrammar *grammar,
    const OdrProducerOptions *options, void *arena, size_t capacity,
    size_t *required_bytes, OdrProducer **out, OdrError *error);
/* Measures a conservative batch capacity WITHOUT advancing the producer. */
OdrResult odr_batch_requirements(const OdrProducer *producer, size_t count,
    size_t *required_bytes);
/* start must equal the cursor's next accepted index. No replay of prefixes.
 * Separate producers may be used for independent streams. NULL output arena
 * measures capacity; no generation or cursor mutation. After generation begins,
 * even an error returns any completed prefix in out. Consume that prefix and
 * resume at out->next_index. ODR_BUFFER never advances the producer. */
OdrResult odr_producer_next(OdrProducer *producer, uint64_t start, size_t count,
    void *arena, size_t capacity, size_t *required_bytes, OdrBatch *out, OdrError *error);
#ifdef __cplusplus
}
#endif
#endif
