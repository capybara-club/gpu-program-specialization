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
#ifndef SECANT_SR_APP_LEAF_SETTINGS_H_INCLUDED
#define SECANT_SR_APP_LEAF_SETTINGS_H_INCLUDED

#include <stddef.h>
#include <stdint.h>

typedef enum SecantSRAppLeafSettingsPolicy {
    SECANT_SR_APP_LEAF_SETTINGS_POLICY_LEGACY = 0,
    SECANT_SR_APP_LEAF_SETTINGS_POLICY_VIRTUAL_BANK = 1,
    SECANT_SR_APP_LEAF_SETTINGS_POLICY_LEGACY_ROTATING = 2,
    SECANT_SR_APP_LEAF_SETTINGS_POLICY_VIRTUAL_BANK_FIXED = 3
} SecantSRAppLeafSettingsPolicy;

int secant_sr_app_leaf_settings_policy_parse(
    const char* name,
    SecantSRAppLeafSettingsPolicy* policy_ret
);

const char* secant_sr_app_leaf_settings_policy_name(SecantSRAppLeafSettingsPolicy policy);

int secant_sr_app_leaf_settings_policy_is_structured(SecantSRAppLeafSettingsPolicy policy);

int secant_sr_app_leaf_settings_policy_is_rotating(SecantSRAppLeafSettingsPolicy policy);

void secant_sr_app_leaf_settings_fill(
    size_t num_settings,
    size_t num_dynamic_leaves,
    size_t num_inputs,
    const float* constants,
    size_t num_constants,
    uint64_t seed,
    uint32_t* leaf_masks,
    uint32_t* leaf_words
);

int secant_sr_app_leaf_settings_virtual_bank_fill(
    size_t num_settings,
    size_t num_dynamic_leaves,
    size_t leaf_words_leading_dimension,
    size_t num_inputs,
    const float* constants,
    size_t num_constants,
    uint64_t seed,
    uint64_t generation,
    uint32_t cohort,
    uint32_t* leaf_masks,
    uint32_t* leaf_words
);

/**
 * Writes `[binding * start][parameter]` mixed LM settings.
 *
 * Binding zero keeps every parameter constant. Remaining bindings cover
 * single-column substitutions first, then all-column and deterministic mixed
 * masks. Every binding is repeated `starts_per_binding` times so adjacent LM
 * lanes share the binding and differ only in their constant start vector.
 */
int secant_sr_app_lm_binding_settings_fill(
    size_t num_bindings,
    size_t starts_per_binding,
    size_t num_parameters,
    size_t parameter_leading_dimension,
    size_t num_inputs,
    uint64_t seed,
    uint64_t generation,
    uint64_t ast_key,
    uint32_t* leaf_masks,
    uint32_t* leaf_words
);

/** Writes deterministic `[constant][setting]` f32 values for constant search. */
void secant_sr_app_constant_settings_fill(
    size_t num_settings,
    size_t num_input_constants,
    const float* constants,
    size_t num_constants,
    uint64_t seed,
    float* constant_settings
);

#endif /* SECANT_SR_APP_LEAF_SETTINGS_H_INCLUDED */
