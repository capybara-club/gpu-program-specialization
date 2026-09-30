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

#include <math.h>
#include <stdint.h>
#include <string.h>

#define TEST_SETTINGS 4096u
#define TEST_VIRTUAL_SETTINGS 8192u
#define TEST_LEAVES 8u
#define TEST_MAX_INPUTS 7u

static uint32_t
test_f32_bits(float value) {
    uint32_t bits;

    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static int
test_constant_bits_contains(
    const float* constants,
    size_t num_constants,
    uint32_t bits
) {
    size_t constant_idx;

    for (constant_idx = 0u; constant_idx < num_constants; ++constant_idx) {
        if (test_f32_bits(constants[constant_idx]) == bits) {
            return 1;
        }
    }
    return 0;
}

static int
test_distribution(
    size_t num_inputs,
    const float* constants,
    size_t num_constants,
    uint64_t seed
) {
    static uint32_t leaf_masks[TEST_SETTINGS];
    static uint32_t leaf_words[TEST_SETTINGS * TEST_LEAVES];
    unsigned char saw_constant[TEST_LEAVES] = { 0u };
    unsigned char saw_column[TEST_LEAVES][TEST_MAX_INPUTS] = { { 0u } };
    size_t setting;
    size_t leaf;
    size_t input_idx;

    if (num_inputs == 0u || num_inputs > TEST_MAX_INPUTS) {
        return 0;
    }
    secant_sr_app_leaf_settings_fill(
        TEST_SETTINGS,
        TEST_LEAVES,
        num_inputs,
        constants,
        num_constants,
        seed,
        leaf_masks,
        leaf_words);
    if ((leaf_masks[0] & UINT32_C(0xff)) != UINT32_C(0xff) ||
        (leaf_masks[1] & UINT32_C(0xff)) != 0u) {
        return 0;
    }
    for (setting = 0u; setting < TEST_SETTINGS; ++setting) {
        for (leaf = 0u; leaf < TEST_LEAVES; ++leaf) {
            const uint32_t word = leaf_words[setting * TEST_LEAVES + leaf];

            if ((leaf_masks[setting] & (UINT32_C(1) << leaf)) != 0u) {
                if (word >= num_inputs) {
                    return 0;
                }
                saw_column[leaf][word] = 1u;
            } else {
                if (!test_constant_bits_contains(constants, num_constants, word)) {
                    return 0;
                }
                saw_constant[leaf] = 1u;
            }
        }
    }
    for (leaf = 0u; leaf < TEST_LEAVES; ++leaf) {
        if (!saw_constant[leaf]) {
            return 0;
        }
        for (input_idx = 0u; input_idx < num_inputs; ++input_idx) {
            if (!saw_column[leaf][input_idx]) {
                return 0;
            }
        }
    }
    return 1;
}

static int
test_constant_settings(void) {
    static const float constants[] = {-2.0f, -1.0f, 0.5f, 2.0f};
    float first[TEST_SETTINGS * TEST_LEAVES];
    float second[TEST_SETTINGS * TEST_LEAVES];
    size_t constant_idx;
    size_t setting;

    secant_sr_app_constant_settings_fill(
        TEST_SETTINGS,
        TEST_LEAVES,
        constants,
        sizeof(constants) / sizeof(constants[0]),
        UINT64_C(0x636f6e7374616e74),
        first);
    secant_sr_app_constant_settings_fill(
        TEST_SETTINGS,
        TEST_LEAVES,
        constants,
        sizeof(constants) / sizeof(constants[0]),
        UINT64_C(0x636f6e7374616e74),
        second);
    if (memcmp(first, second, sizeof(first)) != 0) {
        return 0;
    }
    for (constant_idx = 0u; constant_idx < TEST_LEAVES; ++constant_idx) {
        int varied = 0;

        if (first[constant_idx * TEST_SETTINGS] !=
                constants[constant_idx % (sizeof(constants) / sizeof(constants[0]))] ||
            first[constant_idx * TEST_SETTINGS + 1u] != 0.0f ||
            first[constant_idx * TEST_SETTINGS + 2u] != 1.0f) {
            return 0;
        }
        for (setting = 0u; setting < TEST_SETTINGS; ++setting) {
            const float value = first[constant_idx * TEST_SETTINGS + setting];

            if (!isfinite(value) || value < -32.0f || value > 32.0f) {
                return 0;
            }
            varied |= value != first[constant_idx * TEST_SETTINGS];
        }
        if (!varied) {
            return 0;
        }
    }
    return 1;
}

static size_t
test_popcount(uint32_t value) {
    size_t count = 0u;

    while (value != 0u) {
        count += value & 1u;
        value >>= 1u;
    }
    return count;
}

static int
test_virtual_bank(const float* constants, size_t num_constants) {
    static uint32_t generation0_masks[TEST_VIRTUAL_SETTINGS];
    static uint32_t generation0_words[TEST_VIRTUAL_SETTINGS * TEST_LEAVES];
    static uint32_t generation1_masks[TEST_VIRTUAL_SETTINGS];
    static uint32_t generation1_words[TEST_VIRTUAL_SETTINGS * TEST_LEAVES];
    uint32_t four_leaf_masks[256u];
    uint32_t four_leaf_words[256u * TEST_LEAVES];
    unsigned char pair_seen[32u * 32u] = {0u};
    uint32_t pair_mask_intersection = UINT32_MAX;
    size_t setting;
    size_t leaf;
    size_t first_pair_hole = TEST_LEAVES;
    size_t second_pair_hole = TEST_LEAVES;
    size_t pair_count = 0u;
    SecantSRAppLeafSettingsPolicy policy;

    if (!secant_sr_app_leaf_settings_policy_parse("legacy", &policy) ||
        policy != SECANT_SR_APP_LEAF_SETTINGS_POLICY_LEGACY ||
        secant_sr_app_leaf_settings_policy_is_structured(policy) ||
        secant_sr_app_leaf_settings_policy_is_rotating(policy) ||
        !secant_sr_app_leaf_settings_policy_parse("legacy-rotating", &policy) ||
        policy != SECANT_SR_APP_LEAF_SETTINGS_POLICY_LEGACY_ROTATING ||
        secant_sr_app_leaf_settings_policy_is_structured(policy) ||
        !secant_sr_app_leaf_settings_policy_is_rotating(policy) ||
        !secant_sr_app_leaf_settings_policy_parse("virtual-bank-fixed", &policy) ||
        policy != SECANT_SR_APP_LEAF_SETTINGS_POLICY_VIRTUAL_BANK_FIXED ||
        !secant_sr_app_leaf_settings_policy_is_structured(policy) ||
        secant_sr_app_leaf_settings_policy_is_rotating(policy) ||
        !secant_sr_app_leaf_settings_policy_parse("virtual-bank", &policy) ||
        policy != SECANT_SR_APP_LEAF_SETTINGS_POLICY_VIRTUAL_BANK ||
        !secant_sr_app_leaf_settings_policy_is_structured(policy) ||
        !secant_sr_app_leaf_settings_policy_is_rotating(policy) ||
        strcmp(secant_sr_app_leaf_settings_policy_name(policy), "virtual-bank") != 0 ||
        secant_sr_app_leaf_settings_policy_parse("invalid", &policy)) {
        return 0;
    }
    if (!secant_sr_app_leaf_settings_virtual_bank_fill(
            TEST_VIRTUAL_SETTINGS,
            TEST_LEAVES,
            TEST_LEAVES,
            32u,
            constants,
            num_constants,
            UINT64_C(0x73657474696e6773),
            0u,
            0u,
            generation0_masks,
            generation0_words) ||
        !secant_sr_app_leaf_settings_virtual_bank_fill(
            TEST_VIRTUAL_SETTINGS,
            TEST_LEAVES,
            TEST_LEAVES,
            32u,
            constants,
            num_constants,
            UINT64_C(0x73657474696e6773),
            1u,
            0u,
            generation1_masks,
            generation1_words)) {
        return 0;
    }
    for (setting = 0u; setting < 256u; ++setting) {
        if (generation0_masks[setting] != setting) {
            return 0;
        }
    }
    for (setting = 256u; setting < 1024u; ++setting) {
        if (test_popcount(generation0_masks[setting]) != (setting - 256u) % (TEST_LEAVES + 1u)) {
            return 0;
        }
    }
    if (memcmp(generation0_masks, generation1_masks, 2048u * sizeof(*generation0_masks)) != 0 ||
        memcmp(generation0_words, generation1_words, 2048u * TEST_LEAVES * sizeof(*generation0_words)) != 0 ||
        (memcmp(generation0_masks + 3072u, generation1_masks + 3072u,
             (TEST_VIRTUAL_SETTINGS - 3072u) * sizeof(*generation0_masks)) == 0 &&
         memcmp(generation0_words + 3072u * TEST_LEAVES, generation1_words + 3072u * TEST_LEAVES,
             (TEST_VIRTUAL_SETTINGS - 3072u) * TEST_LEAVES * sizeof(*generation0_words)) == 0)) {
        return 0;
    }
    for (setting = 2048u; setting < 3072u; ++setting) {
        pair_mask_intersection &= generation0_masks[setting];
    }
    if (test_popcount(pair_mask_intersection) != 2u) {
        return 0;
    }
    for (leaf = 0u; leaf < TEST_LEAVES; ++leaf) {
        if ((pair_mask_intersection & (UINT32_C(1) << leaf)) != 0u) {
            if (first_pair_hole == TEST_LEAVES) {
                first_pair_hole = leaf;
            } else {
                second_pair_hole = leaf;
            }
        }
    }
    for (setting = 2048u; setting < 3072u; ++setting) {
        const uint32_t* words = generation0_words + setting * TEST_LEAVES;
        const size_t pair = words[first_pair_hole] * 32u + words[second_pair_hole];

        if (pair >= sizeof(pair_seen) || pair_seen[pair]) {
            return 0;
        }
        pair_seen[pair] = 1u;
        ++pair_count;
    }
    if (pair_count != 1024u) {
        return 0;
    }
    if (!secant_sr_app_leaf_settings_virtual_bank_fill(
            256u,
            4u,
            TEST_LEAVES,
            3u,
            constants,
            num_constants,
            UINT64_C(0x73657474696e6773),
            0u,
            1u,
            four_leaf_masks,
            four_leaf_words)) {
        return 0;
    }
    for (setting = 0u; setting < 256u; ++setting) {
        if (four_leaf_masks[setting] != setting % 16u) {
            return 0;
        }
        for (leaf = 4u; leaf < TEST_LEAVES; ++leaf) {
            if (four_leaf_words[setting * TEST_LEAVES + leaf] != 0u) {
                return 0;
            }
        }
    }
    return 1;
}

static int
test_lm_binding_settings(void) {
    enum {
        TEST_BINDINGS = 32,
        TEST_STARTS = 4,
        TEST_LM_SETTINGS = TEST_BINDINGS * TEST_STARTS
    };
    static uint32_t masks[TEST_LM_SETTINGS];
    static uint32_t words[TEST_LM_SETTINGS * TEST_LEAVES];
    uint32_t covered = 0u;
    size_t binding;

    if (!secant_sr_app_lm_binding_settings_fill(
            TEST_BINDINGS,
            TEST_STARTS,
            TEST_LEAVES,
            TEST_LEAVES,
            7u,
            UINT64_C(0x6c6d62696e64696e),
            13u,
            UINT64_C(0x123456789abcdef0),
            masks,
            words) ||
        masks[0] != 0u ||
        secant_sr_app_lm_binding_settings_fill(
            0u,
            TEST_STARTS,
            TEST_LEAVES,
            TEST_LEAVES,
            7u,
            0u,
            0u,
            0u,
            masks,
            words)) {
        return 0;
    }
    for (binding = 0u; binding < TEST_BINDINGS; ++binding) {
        const size_t first_setting = binding * TEST_STARTS;
        size_t start;
        size_t leaf;

        if ((binding <= TEST_LEAVES && binding != 0u &&
             masks[first_setting] != (UINT32_C(1) << (binding - 1u))) ||
            (binding == TEST_LEAVES + 1u && masks[first_setting] != UINT32_C(0xff))) {
            return 0;
        }
        covered |= masks[first_setting];
        for (start = 1u; start < TEST_STARTS; ++start) {
            const size_t setting = first_setting + start;

            if (masks[setting] != masks[first_setting] ||
                memcmp(
                    words + setting * TEST_LEAVES,
                    words + first_setting * TEST_LEAVES,
                    TEST_LEAVES * sizeof(*words)) != 0) {
                return 0;
            }
        }
        for (leaf = 0u; leaf < TEST_LEAVES; ++leaf) {
            if ((masks[first_setting] & (UINT32_C(1) << leaf)) != 0u &&
                words[first_setting * TEST_LEAVES + leaf] >= 7u) {
                return 0;
            }
        }
    }
    return covered == UINT32_C(0xff);
}

int
main(void) {
    static const float constants[] = {
        -3.0f, -2.0f, -1.5f, -1.0f, -0.5f, -0.25f,
        0.25f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f
    };
    static uint32_t leaf_masks[TEST_SETTINGS];
    static uint32_t leaf_words[TEST_SETTINGS * TEST_LEAVES];
    const uint64_t seed = UINT64_C(1) ^ UINT64_C(0x6c656166736574);
    const uint32_t one_bits = test_f32_bits(1.0f);
    static const size_t input_counts[] = { 1u, 2u, 3u, 7u };
    int distance_found = 0;
    int rational_found = 0;
    size_t input_count_idx;
    size_t setting;

    for (input_count_idx = 0u;
         input_count_idx < sizeof(input_counts) / sizeof(input_counts[0]);
         ++input_count_idx) {
        if (!test_distribution(
                input_counts[input_count_idx],
                constants,
                sizeof(constants) / sizeof(constants[0]),
                seed ^ (uint64_t)input_counts[input_count_idx])) {
            return 1;
        }
    }

    secant_sr_app_leaf_settings_fill(
        TEST_SETTINGS,
        TEST_LEAVES,
        2u,
        constants,
        sizeof(constants) / sizeof(constants[0]),
        seed,
        leaf_masks,
        leaf_words);
    for (setting = 0u; setting < TEST_SETTINGS; ++setting) {
        const uint32_t* words = leaf_words + setting * TEST_LEAVES;
        const uint32_t mask = leaf_masks[setting];

        if ((mask & UINT32_C(0x0f)) == UINT32_C(0x0f) &&
            words[0] == 0u && words[1] == 0u && words[2] == 1u && words[3] == 1u) {
            distance_found = 1;
        }
        if ((mask & UINT32_C(0x0f)) == UINT32_C(0x0d) &&
            words[0] == 0u && words[1] == one_bits && words[2] == 1u && words[3] == 1u) {
            rational_found = 1;
        }
    }
    return distance_found && rational_found && test_constant_settings() && test_lm_binding_settings() &&
            test_virtual_bank(constants, sizeof(constants) / sizeof(constants[0]))
        ? 0
        : 1;
}
