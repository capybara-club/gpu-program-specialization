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
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Charles Durham
 * SPDX-License-Identifier: MIT
 *
 * Sparse replacement multi-site panel evaluators.
 *
 * The functions intentionally keep the dense switch/constant-table shape, then
 * zero the computed value before writing. This acts like a padded evaluator
 * slot while keeping CUDA side-section sizes compatible with the reserved
 * template across SM90 and SM120.
 */

#define CFP_MULTI_PANEL_SPARSE_CASE(SITE, I) \
    case (I): { \
        asm volatile( \
            "{\n\t" \
            ".reg .f32 %%a, %%b;\n\t" \
            "mul.ftz.f32 %%a, %1, %2;\n\t" \
            "add.ftz.f32 %%b, %%a, %3;\n\t" \
            "add.ftz.f32 %0, %%b, %4;\n\t" \
            "}" \
            : "+f"(value) \
            : "f"(leaf0), "f"(leaf1), "f"(leaf2), "f"(leaf3)); \
        value = value + (float)((SITE) * 19 + (I)) * 0.00390625f; \
        value = value - value; \
        break; \
    }

#define CFP_MULTI_PANEL_SPARSE_SITE(SITE) \
    extern "C" __device__ __noinline__ \
    void \
    patch_multi_panel_site_##SITE( \
        float* panel, \
        const float* const* leaf_ptrs, \
        const int* leaf_strides, \
        int active_rows \
    ) { \
        const int tid = (int)threadIdx.x; \
        for (int linear = tid; linear < 32 * 128; linear += 128) { \
            const int feature = linear >> 7; \
            const int row = linear & 127; \
            float value = 0.0f; \
            if (row < active_rows) { \
                const int base = feature * 8; \
                const float leaf0 = leaf_ptrs[base + 0][row * leaf_strides[base + 0]]; \
                const float leaf1 = leaf_ptrs[base + 1][row * leaf_strides[base + 1]]; \
                const float leaf2 = leaf_ptrs[base + 2][row * leaf_strides[base + 2]]; \
                const float leaf3 = leaf_ptrs[base + 3][row * leaf_strides[base + 3]]; \
                switch (feature) { \
                    CFP_MULTI_PANEL_SPARSE_CASE(SITE, 0) \
                    CFP_MULTI_PANEL_SPARSE_CASE(SITE, 1) \
                    CFP_MULTI_PANEL_SPARSE_CASE(SITE, 2) \
                    CFP_MULTI_PANEL_SPARSE_CASE(SITE, 3) \
                    CFP_MULTI_PANEL_SPARSE_CASE(SITE, 4) \
                    CFP_MULTI_PANEL_SPARSE_CASE(SITE, 5) \
                    CFP_MULTI_PANEL_SPARSE_CASE(SITE, 6) \
                    CFP_MULTI_PANEL_SPARSE_CASE(SITE, 7) \
                    CFP_MULTI_PANEL_SPARSE_CASE(SITE, 8) \
                    CFP_MULTI_PANEL_SPARSE_CASE(SITE, 9) \
                    CFP_MULTI_PANEL_SPARSE_CASE(SITE, 10) \
                    CFP_MULTI_PANEL_SPARSE_CASE(SITE, 11) \
                    CFP_MULTI_PANEL_SPARSE_CASE(SITE, 12) \
                    CFP_MULTI_PANEL_SPARSE_CASE(SITE, 13) \
                    CFP_MULTI_PANEL_SPARSE_CASE(SITE, 14) \
                    CFP_MULTI_PANEL_SPARSE_CASE(SITE, 15) \
                    CFP_MULTI_PANEL_SPARSE_CASE(SITE, 16) \
                    CFP_MULTI_PANEL_SPARSE_CASE(SITE, 17) \
                    CFP_MULTI_PANEL_SPARSE_CASE(SITE, 18) \
                    CFP_MULTI_PANEL_SPARSE_CASE(SITE, 19) \
                    CFP_MULTI_PANEL_SPARSE_CASE(SITE, 20) \
                    CFP_MULTI_PANEL_SPARSE_CASE(SITE, 21) \
                    CFP_MULTI_PANEL_SPARSE_CASE(SITE, 22) \
                    CFP_MULTI_PANEL_SPARSE_CASE(SITE, 23) \
                    CFP_MULTI_PANEL_SPARSE_CASE(SITE, 24) \
                    CFP_MULTI_PANEL_SPARSE_CASE(SITE, 25) \
                    CFP_MULTI_PANEL_SPARSE_CASE(SITE, 26) \
                    CFP_MULTI_PANEL_SPARSE_CASE(SITE, 27) \
                    CFP_MULTI_PANEL_SPARSE_CASE(SITE, 28) \
                    CFP_MULTI_PANEL_SPARSE_CASE(SITE, 29) \
                    CFP_MULTI_PANEL_SPARSE_CASE(SITE, 30) \
                    CFP_MULTI_PANEL_SPARSE_CASE(SITE, 31) \
                    default: value = 0.0f; break; \
                } \
            } \
            panel[linear] = value; \
        } \
    }

CFP_MULTI_PANEL_SPARSE_SITE(0)
CFP_MULTI_PANEL_SPARSE_SITE(1)
CFP_MULTI_PANEL_SPARSE_SITE(2)
CFP_MULTI_PANEL_SPARSE_SITE(3)
