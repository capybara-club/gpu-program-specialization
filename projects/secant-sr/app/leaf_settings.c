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
#include "leaf_settings.h"

#include <limits.h>
#include <string.h>

#define SECANT_SR_APP_VIRTUAL_BANK_MAX_LEAVES 32u

typedef struct SecantSRAppPhilox4x32 {
    uint32_t x;
    uint32_t y;
    uint32_t z;
    uint32_t w;
} SecantSRAppPhilox4x32;

static uint32_t secant_sr_app_f32_bits(float value);

static SecantSRAppPhilox4x32
secant_sr_app_philox_round(SecantSRAppPhilox4x32 counter, uint32_t key0, uint32_t key1) {
    const uint64_t product0 = UINT64_C(0xd2511f53) * counter.x;
    const uint64_t product1 = UINT64_C(0xcd9e8d57) * counter.z;
    SecantSRAppPhilox4x32 result;

    result.x = (uint32_t)(product1 >> 32u) ^ counter.y ^ key0;
    result.y = (uint32_t)product1;
    result.z = (uint32_t)(product0 >> 32u) ^ counter.w ^ key1;
    result.w = (uint32_t)product0;
    return result;
}

static SecantSRAppPhilox4x32
secant_sr_app_philox(
    uint64_t seed,
    uint64_t generation,
    uint32_t cohort,
    uint64_t setting,
    uint32_t stream
) {
    SecantSRAppPhilox4x32 counter;
    uint32_t key0 = (uint32_t)seed ^ (uint32_t)generation * UINT32_C(0x9e3779b9);
    uint32_t key1 = (uint32_t)(seed >> 32u) ^ (uint32_t)(generation >> 32u) * UINT32_C(0xbb67ae85);
    size_t round;

    counter.x = (uint32_t)setting;
    counter.y = (uint32_t)(setting >> 32u);
    counter.z = stream;
    counter.w = cohort;
    for (round = 0u; round < 10u; ++round) {
        counter = secant_sr_app_philox_round(counter, key0, key1);
        key0 += UINT32_C(0x9e3779b9);
        key1 += UINT32_C(0xbb67ae85);
    }
    return counter;
}

static uint32_t
secant_sr_app_bounded_u32(uint32_t word, uint32_t bound) {
    return (uint32_t)(((uint64_t)word * bound) >> 32u);
}

static uint32_t
secant_sr_app_leaf_mask_exact_popcount(
    size_t num_dynamic_leaves,
    size_t num_columns,
    uint64_t seed,
    uint64_t generation,
    uint32_t cohort,
    uint64_t setting,
    uint32_t stream
) {
    uint32_t positions[SECANT_SR_APP_VIRTUAL_BANK_MAX_LEAVES];
    uint32_t mask = 0u;
    size_t leaf;

    for (leaf = 0u; leaf < num_dynamic_leaves; ++leaf) {
        positions[leaf] = (uint32_t)leaf;
    }
    for (leaf = 0u; leaf < num_columns; ++leaf) {
        const SecantSRAppPhilox4x32 random = secant_sr_app_philox(
            seed, generation, cohort, setting, stream + (uint32_t)leaf);
        const uint32_t remaining = (uint32_t)(num_dynamic_leaves - leaf);
        const size_t selected = leaf + secant_sr_app_bounded_u32(random.x, remaining);
        const uint32_t position = positions[selected];

        positions[selected] = positions[leaf];
        positions[leaf] = position;
        mask |= UINT32_C(1) << position;
    }
    return mask;
}

static uint32_t
secant_sr_app_exploratory_constant_bits(
    const float* constants,
    size_t num_constants,
    uint32_t selector,
    uint32_t value_word
) {
    static const float scales[] = {0.25f, 1.0f, 4.0f, 16.0f, 32.0f};

    if ((selector & 3u) != 0u) {
        return secant_sr_app_f32_bits(constants[secant_sr_app_bounded_u32(value_word, (uint32_t)num_constants)]);
    }
    {
        const float unit = (float)(value_word >> 8u) * 0x1.0p-24f;
        const float scale = scales[secant_sr_app_bounded_u32(
            selector >> 2u, (uint32_t)(sizeof(scales) / sizeof(scales[0])))];

        return secant_sr_app_f32_bits(scale * (2.0f * unit - 1.0f));
    }
}

static void
secant_sr_app_setting_uniform_fill(
    size_t num_dynamic_leaves,
    size_t leaf_words_leading_dimension,
    size_t num_inputs,
    const float* constants,
    size_t num_constants,
    uint64_t seed,
    uint64_t generation,
    uint32_t cohort,
    uint64_t setting,
    int exploratory_constants,
    uint32_t* mask_ret,
    uint32_t* words
) {
    uint32_t mask = 0u;
    size_t leaf;

    memset(words, 0, leaf_words_leading_dimension * sizeof(*words));
    for (leaf = 0u; leaf < num_dynamic_leaves; ++leaf) {
        const SecantSRAppPhilox4x32 random = secant_sr_app_philox(
            seed, generation, cohort, setting, (uint32_t)leaf);

        if ((random.x & 1u) != 0u) {
            mask |= UINT32_C(1) << leaf;
            words[leaf] = secant_sr_app_bounded_u32(random.y, (uint32_t)num_inputs);
        } else if (exploratory_constants) {
            words[leaf] = secant_sr_app_exploratory_constant_bits(
                constants, num_constants, random.z, random.w);
        } else {
            words[leaf] = secant_sr_app_f32_bits(
                constants[secant_sr_app_bounded_u32(random.y, (uint32_t)num_constants)]);
        }
    }
    *mask_ret = mask;
}

int
secant_sr_app_leaf_settings_policy_parse(
    const char* name,
    SecantSRAppLeafSettingsPolicy* policy_ret
) {
    if (name == NULL || policy_ret == NULL) {
        return 0;
    }
    if (strcmp(name, "legacy") == 0) {
        *policy_ret = SECANT_SR_APP_LEAF_SETTINGS_POLICY_LEGACY;
        return 1;
    }
    if (strcmp(name, "virtual-bank") == 0) {
        *policy_ret = SECANT_SR_APP_LEAF_SETTINGS_POLICY_VIRTUAL_BANK;
        return 1;
    }
    if (strcmp(name, "legacy-rotating") == 0) {
        *policy_ret = SECANT_SR_APP_LEAF_SETTINGS_POLICY_LEGACY_ROTATING;
        return 1;
    }
    if (strcmp(name, "virtual-bank-fixed") == 0) {
        *policy_ret = SECANT_SR_APP_LEAF_SETTINGS_POLICY_VIRTUAL_BANK_FIXED;
        return 1;
    }
    return 0;
}

const char*
secant_sr_app_leaf_settings_policy_name(SecantSRAppLeafSettingsPolicy policy) {
    switch (policy) {
        case SECANT_SR_APP_LEAF_SETTINGS_POLICY_LEGACY:
            return "legacy";
        case SECANT_SR_APP_LEAF_SETTINGS_POLICY_VIRTUAL_BANK:
            return "virtual-bank";
        case SECANT_SR_APP_LEAF_SETTINGS_POLICY_LEGACY_ROTATING:
            return "legacy-rotating";
        case SECANT_SR_APP_LEAF_SETTINGS_POLICY_VIRTUAL_BANK_FIXED:
            return "virtual-bank-fixed";
        default:
            return "unknown";
    }
}

int
secant_sr_app_leaf_settings_policy_is_structured(SecantSRAppLeafSettingsPolicy policy) {
    return policy == SECANT_SR_APP_LEAF_SETTINGS_POLICY_VIRTUAL_BANK ||
        policy == SECANT_SR_APP_LEAF_SETTINGS_POLICY_VIRTUAL_BANK_FIXED;
}

int
secant_sr_app_leaf_settings_policy_is_rotating(SecantSRAppLeafSettingsPolicy policy) {
    return policy == SECANT_SR_APP_LEAF_SETTINGS_POLICY_VIRTUAL_BANK ||
        policy == SECANT_SR_APP_LEAF_SETTINGS_POLICY_LEGACY_ROTATING;
}

static uint32_t
secant_sr_app_hash32(uint32_t value) {
    value ^= value >> 16u;
    value *= UINT32_C(0x7feb352d);
    value ^= value >> 15u;
    value *= UINT32_C(0x846ca68b);
    return value ^ (value >> 16u);
}

static uint32_t
secant_sr_app_f32_bits(float value) {
    uint32_t bits;

    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

void
secant_sr_app_leaf_settings_fill(
    size_t num_settings,
    size_t num_dynamic_leaves,
    size_t num_inputs,
    const float* constants,
    size_t num_constants,
    uint64_t seed,
    uint32_t* leaf_masks,
    uint32_t* leaf_words
) {
    size_t setting;

    for (setting = 0u; setting < num_settings; ++setting) {
        uint32_t mask = 0u;
        size_t leaf;

        for (leaf = 0u; leaf < num_dynamic_leaves; ++leaf) {
            const uint32_t hash = secant_sr_app_hash32(
                (uint32_t)setting * UINT32_C(0x9e3779b9) ^
                (uint32_t)leaf * UINT32_C(0x85ebca6b) ^
                (uint32_t)seed ^ (uint32_t)(seed >> 32u));
            const int column = setting == 0u || (setting > 1u && (hash & 1u) != 0u);
            const uint32_t value_hash = hash >> 1u;

            if (column) {
                mask |= UINT32_C(1) << leaf;
                leaf_words[setting * num_dynamic_leaves + leaf] = value_hash % num_inputs;
            } else {
                leaf_words[setting * num_dynamic_leaves + leaf] =
                    secant_sr_app_f32_bits(constants[value_hash % num_constants]);
            }
        }
        leaf_masks[setting] = mask;
    }
}

int
secant_sr_app_leaf_settings_virtual_bank_fill(
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
) {
    size_t setting;

    if (num_settings == 0u || num_dynamic_leaves == 0u ||
        num_dynamic_leaves > SECANT_SR_APP_VIRTUAL_BANK_MAX_LEAVES ||
        leaf_words_leading_dimension < num_dynamic_leaves || num_inputs == 0u || num_inputs > UINT32_MAX ||
        constants == NULL || num_constants == 0u || num_constants > UINT32_MAX ||
        leaf_words_leading_dimension > SIZE_MAX / sizeof(*leaf_words) ||
        num_settings > SIZE_MAX / leaf_words_leading_dimension ||
        leaf_masks == NULL || leaf_words == NULL) {
        return 0;
    }
    for (setting = 0u; setting < num_settings; ++setting) {
        uint32_t* words = leaf_words + setting * leaf_words_leading_dimension;

        if (setting < 256u) {
            const size_t num_masks = num_dynamic_leaves < 8u ? (size_t)1u << num_dynamic_leaves : 256u;
            const size_t mask_index = setting % num_masks;
            const size_t replicate = setting / num_masks;
            uint32_t mask;
            size_t leaf;

            if (num_dynamic_leaves <= 8u) {
                mask = (uint32_t)mask_index;
            } else if (setting == 0u) {
                mask = 0u;
            } else if (setting == 255u) {
                mask = UINT32_MAX;
            } else {
                mask = secant_sr_app_philox(seed, 0u, 0u, setting, 0u).x;
                if (num_dynamic_leaves < 32u) {
                    mask &= (UINT32_C(1) << num_dynamic_leaves) - 1u;
                }
            }
            memset(words, 0, leaf_words_leading_dimension * sizeof(*words));
            for (leaf = 0u; leaf < num_dynamic_leaves; ++leaf) {
                if ((mask & (UINT32_C(1) << leaf)) != 0u) {
                    const size_t column = (setting & 32u) != 0u
                        ? (setting + replicate) % num_inputs
                        : (setting + replicate + leaf) % num_inputs;

                    words[leaf] = (uint32_t)column;
                } else {
                    words[leaf] = secant_sr_app_f32_bits(
                        constants[(setting + replicate * num_dynamic_leaves + leaf) % num_constants]);
                }
            }
            leaf_masks[setting] = mask;
        } else if (setting < 1024u) {
            const size_t local_setting = setting - 256u;
            const size_t num_columns = local_setting % (num_dynamic_leaves + 1u);
            const uint32_t mask = secant_sr_app_leaf_mask_exact_popcount(
                num_dynamic_leaves, num_columns, seed, 0u, 0u, setting, 0u);
            size_t leaf;

            memset(words, 0, leaf_words_leading_dimension * sizeof(*words));
            for (leaf = 0u; leaf < num_dynamic_leaves; ++leaf) {
                const SecantSRAppPhilox4x32 random = secant_sr_app_philox(seed, 0u, 0u, setting, (uint32_t)leaf + 64u);

                words[leaf] = (mask & (UINT32_C(1) << leaf)) != 0u
                    ? secant_sr_app_bounded_u32(random.y, (uint32_t)num_inputs)
                    : secant_sr_app_f32_bits(
                          constants[secant_sr_app_bounded_u32(random.z, (uint32_t)num_constants)]);
            }
            leaf_masks[setting] = mask;
        } else if (setting < 2048u) {
            secant_sr_app_setting_uniform_fill(
                num_dynamic_leaves,
                leaf_words_leading_dimension,
                num_inputs,
                constants,
                num_constants,
                seed,
                0u,
                0u,
                setting,
                0,
                leaf_masks + setting,
                words);
        } else if (setting < 3072u && num_dynamic_leaves >= 2u) {
            const size_t local_setting = setting - 2048u;
            const size_t num_ordered_hole_pairs = num_dynamic_leaves * (num_dynamic_leaves - 1u);
            const size_t hole_pair = ((size_t)generation + cohort * 17u) % num_ordered_hole_pairs;
            const size_t first_hole = hole_pair / (num_dynamic_leaves - 1u);
            const size_t second_slot = hole_pair % (num_dynamic_leaves - 1u);
            const size_t second_hole = second_slot >= first_hole ? second_slot + 1u : second_slot;
            const size_t num_column_pairs = num_inputs <= 32u ? num_inputs * num_inputs : 0u;
            const SecantSRAppPhilox4x32 pair_random = secant_sr_app_philox(
                seed, generation, cohort, setting, 63u);
            uint32_t mask;
            uint32_t first_column;
            uint32_t second_column;

            secant_sr_app_setting_uniform_fill(
                num_dynamic_leaves,
                leaf_words_leading_dimension,
                num_inputs,
                constants,
                num_constants,
                seed,
                generation,
                cohort,
                setting,
                0,
                &mask,
                words);
            if (num_column_pairs != 0u) {
                const size_t pair = local_setting % num_column_pairs;

                first_column = (uint32_t)(pair / num_inputs);
                second_column = (uint32_t)(pair % num_inputs);
            } else {
                first_column = secant_sr_app_bounded_u32(pair_random.x, (uint32_t)num_inputs);
                second_column = secant_sr_app_bounded_u32(pair_random.y, (uint32_t)num_inputs);
            }
            mask |= (UINT32_C(1) << first_hole) | (UINT32_C(1) << second_hole);
            words[first_hole] = first_column;
            words[second_hole] = second_column;
            leaf_masks[setting] = mask;
        } else if (setting < 6144u) {
            const SecantSRAppPhilox4x32 density_random = secant_sr_app_philox(
                seed, generation, cohort, setting, 61u);
            const size_t num_columns = secant_sr_app_bounded_u32(
                density_random.x, (uint32_t)(num_dynamic_leaves + 1u));
            const uint32_t mask = secant_sr_app_leaf_mask_exact_popcount(
                num_dynamic_leaves, num_columns, seed, generation, cohort, setting, 80u);
            size_t leaf;

            memset(words, 0, leaf_words_leading_dimension * sizeof(*words));
            for (leaf = 0u; leaf < num_dynamic_leaves; ++leaf) {
                const SecantSRAppPhilox4x32 random = secant_sr_app_philox(
                    seed, generation, cohort, setting, (uint32_t)leaf + 112u);

                words[leaf] = (mask & (UINT32_C(1) << leaf)) != 0u
                    ? secant_sr_app_bounded_u32(random.y, (uint32_t)num_inputs)
                    : secant_sr_app_exploratory_constant_bits(constants, num_constants, random.z, random.w);
            }
            leaf_masks[setting] = mask;
        } else {
            const SecantSRAppPhilox4x32 density_random = secant_sr_app_philox(
                seed, generation, cohort, setting, 62u);
            const size_t num_columns = secant_sr_app_bounded_u32(
                density_random.x, (uint32_t)(num_dynamic_leaves + 1u));
            const uint32_t mask = secant_sr_app_leaf_mask_exact_popcount(
                num_dynamic_leaves, num_columns, seed, generation, cohort, setting, 144u);
            const int repeated_columns = (setting & 7u) == 0u;
            const uint32_t repeated_column = secant_sr_app_bounded_u32(
                density_random.y, (uint32_t)num_inputs);
            size_t leaf;

            memset(words, 0, leaf_words_leading_dimension * sizeof(*words));
            for (leaf = 0u; leaf < num_dynamic_leaves; ++leaf) {
                const SecantSRAppPhilox4x32 random = secant_sr_app_philox(
                    seed, generation, cohort, setting, (uint32_t)leaf + 176u);

                words[leaf] = (mask & (UINT32_C(1) << leaf)) != 0u
                    ? (repeated_columns
                        ? repeated_column
                        : secant_sr_app_bounded_u32(random.y, (uint32_t)num_inputs))
                    : secant_sr_app_exploratory_constant_bits(constants, num_constants, random.z, random.w);
            }
            leaf_masks[setting] = mask;
        }
    }
    return 1;
}

int
secant_sr_app_lm_binding_settings_fill(
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
) {
    const uint64_t binding_seed = seed ^
        secant_sr_app_philox(ast_key, generation, 0u, ast_key, 255u).x ^
        (ast_key << 32u);
    size_t num_settings;
    size_t binding;

    if (num_bindings == 0u || starts_per_binding == 0u || num_parameters == 0u ||
        num_parameters > SECANT_SR_APP_VIRTUAL_BANK_MAX_LEAVES ||
        parameter_leading_dimension < num_parameters || num_inputs == 0u || num_inputs > UINT32_MAX ||
        num_bindings > SIZE_MAX / starts_per_binding ||
        (num_settings = num_bindings * starts_per_binding) > SIZE_MAX / parameter_leading_dimension ||
        leaf_masks == NULL || leaf_words == NULL) {
        return 0;
    }
    for (binding = 0u; binding < num_bindings; ++binding) {
        uint32_t words[SECANT_SR_APP_VIRTUAL_BANK_MAX_LEAVES] = {0u};
        uint32_t mask = 0u;
        size_t start;
        size_t parameter;

        if (binding != 0u) {
            if (binding <= num_parameters) {
                mask = UINT32_C(1) << (binding - 1u);
            } else if (binding == num_parameters + 1u) {
                mask = num_parameters == 32u
                    ? UINT32_MAX
                    : (UINT32_C(1) << num_parameters) - 1u;
            } else {
                const size_t num_columns = 1u +
                    (binding - num_parameters - 2u) % num_parameters;

                mask = secant_sr_app_leaf_mask_exact_popcount(
                    num_parameters,
                    num_columns,
                    binding_seed,
                    generation,
                    (uint32_t)ast_key,
                    binding,
                    224u);
            }
        }
        for (parameter = 0u; parameter < num_parameters; ++parameter) {
            if ((mask & (UINT32_C(1) << parameter)) != 0u) {
                const SecantSRAppPhilox4x32 random = secant_sr_app_philox(
                    binding_seed,
                    generation,
                    (uint32_t)ast_key,
                    binding,
                    (uint32_t)parameter + 240u);

                words[parameter] = secant_sr_app_bounded_u32(random.y, (uint32_t)num_inputs);
            }
        }
        for (start = 0u; start < starts_per_binding; ++start) {
            const size_t setting = binding * starts_per_binding + start;
            uint32_t* destination = leaf_words + setting * parameter_leading_dimension;

            leaf_masks[setting] = mask;
            memset(destination, 0, parameter_leading_dimension * sizeof(*destination));
            memcpy(destination, words, num_parameters * sizeof(*destination));
        }
    }
    return 1;
}

void
secant_sr_app_constant_settings_fill(
    size_t num_settings,
    size_t num_input_constants,
    const float* constants,
    size_t num_constants,
    uint64_t seed,
    float* constant_settings
) {
    size_t constant_idx;

    for (constant_idx = 0u; constant_idx < num_input_constants; ++constant_idx) {
        size_t setting;

        for (setting = 0u; setting < num_settings; ++setting) {
            const uint32_t hash = secant_sr_app_hash32(
                (uint32_t)setting * UINT32_C(0x9e3779b9) ^
                (uint32_t)constant_idx * UINT32_C(0x85ebca6b) ^
                (uint32_t)seed ^ (uint32_t)(seed >> 32u));
            float value;

            if (setting == 0u) {
                value = constants[constant_idx % num_constants];
            } else if (setting == 1u) {
                value = 0.0f;
            } else if (setting == 2u) {
                value = 1.0f;
            } else if ((hash & 3u) == 0u) {
                value = constants[(hash >> 2u) % num_constants];
            } else {
                const float unit = (float)(hash >> 8u) * (1.0f / 16777215.0f);
                const float radius = (hash & 3u) == 1u ? 2.0f : ((hash & 3u) == 2u ? 8.0f : 32.0f);

                value = radius * (2.0f * unit - 1.0f);
            }
            constant_settings[constant_idx * num_settings + setting] = value;
        }
    }
}
