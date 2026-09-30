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
 * Replacement panel evaluator for CUDA side-section patch tests.
 *
 * This keeps the same ABI as patch_panel_reserved.cu and uses a dense feature
 * switch so sm_90/CUDA 12.8 emits function-owned constant-table relocations.
 */

#define CFP_PANEL_REPLACEMENT_CASE(I) \
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
        value = value + (float)(I) * 0.00390625f; \
        break; \
    }

extern "C" __device__ __noinline__
void
patch_panel_site(
    float* panel,
    const float* const* leaf_ptrs,
    const int* leaf_strides,
    int active_rows
) {
    const int tid = (int)threadIdx.x;
    for (int linear = tid; linear < 32 * 128; linear += 128) {
        const int feature = linear >> 7;
        const int row = linear & 127;
        float value = 0.0f;
        if (row < active_rows) {
            const int base = feature * 8;
            const float leaf0 = leaf_ptrs[base + 0][row * leaf_strides[base + 0]];
            const float leaf1 = leaf_ptrs[base + 1][row * leaf_strides[base + 1]];
            const float leaf2 = leaf_ptrs[base + 2][row * leaf_strides[base + 2]];
            const float leaf3 = leaf_ptrs[base + 3][row * leaf_strides[base + 3]];
            switch (feature) {
                CFP_PANEL_REPLACEMENT_CASE(0)
                CFP_PANEL_REPLACEMENT_CASE(1)
                CFP_PANEL_REPLACEMENT_CASE(2)
                CFP_PANEL_REPLACEMENT_CASE(3)
                CFP_PANEL_REPLACEMENT_CASE(4)
                CFP_PANEL_REPLACEMENT_CASE(5)
                CFP_PANEL_REPLACEMENT_CASE(6)
                CFP_PANEL_REPLACEMENT_CASE(7)
                CFP_PANEL_REPLACEMENT_CASE(8)
                CFP_PANEL_REPLACEMENT_CASE(9)
                CFP_PANEL_REPLACEMENT_CASE(10)
                CFP_PANEL_REPLACEMENT_CASE(11)
                CFP_PANEL_REPLACEMENT_CASE(12)
                CFP_PANEL_REPLACEMENT_CASE(13)
                CFP_PANEL_REPLACEMENT_CASE(14)
                CFP_PANEL_REPLACEMENT_CASE(15)
                CFP_PANEL_REPLACEMENT_CASE(16)
                CFP_PANEL_REPLACEMENT_CASE(17)
                CFP_PANEL_REPLACEMENT_CASE(18)
                CFP_PANEL_REPLACEMENT_CASE(19)
                CFP_PANEL_REPLACEMENT_CASE(20)
                CFP_PANEL_REPLACEMENT_CASE(21)
                CFP_PANEL_REPLACEMENT_CASE(22)
                CFP_PANEL_REPLACEMENT_CASE(23)
                CFP_PANEL_REPLACEMENT_CASE(24)
                CFP_PANEL_REPLACEMENT_CASE(25)
                CFP_PANEL_REPLACEMENT_CASE(26)
                CFP_PANEL_REPLACEMENT_CASE(27)
                CFP_PANEL_REPLACEMENT_CASE(28)
                CFP_PANEL_REPLACEMENT_CASE(29)
                CFP_PANEL_REPLACEMENT_CASE(30)
                CFP_PANEL_REPLACEMENT_CASE(31)
                default:
                    value = 0.0f;
                    break;
            }
        }
        panel[linear] = value;
    }
}
