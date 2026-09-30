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
 * Reserved switch-template function for CUDA side-section patch tests.
 *
 * Dense switches can produce CUDA constant-table sections plus text relocations
 * in nvPTXCompiler RDC output. This reserved body gives the patcher a larger
 * linked slot and matching side sections for replacement switch bodies.
 */

#define CFP_SWITCH_RESERVED_CASE(I) \
    case (I): { \
        volatile float y = x + (float)(I); \
        y = y * 1.0001f + 0.03125f * x; \
        y = y * 0.9999f - 0.015625f * x; \
        y = y + (float)((I) & 7) * 0.125f; \
        return (float)y; \
    }

extern "C" __device__ __noinline__
float
patch_site(
    float x
) {
    int selector = ((int)(x * 17.0f)) & 31;
    switch (selector) {
        CFP_SWITCH_RESERVED_CASE(0)
        CFP_SWITCH_RESERVED_CASE(1)
        CFP_SWITCH_RESERVED_CASE(2)
        CFP_SWITCH_RESERVED_CASE(3)
        CFP_SWITCH_RESERVED_CASE(4)
        CFP_SWITCH_RESERVED_CASE(5)
        CFP_SWITCH_RESERVED_CASE(6)
        CFP_SWITCH_RESERVED_CASE(7)
        CFP_SWITCH_RESERVED_CASE(8)
        CFP_SWITCH_RESERVED_CASE(9)
        CFP_SWITCH_RESERVED_CASE(10)
        CFP_SWITCH_RESERVED_CASE(11)
        CFP_SWITCH_RESERVED_CASE(12)
        CFP_SWITCH_RESERVED_CASE(13)
        CFP_SWITCH_RESERVED_CASE(14)
        CFP_SWITCH_RESERVED_CASE(15)
        CFP_SWITCH_RESERVED_CASE(16)
        CFP_SWITCH_RESERVED_CASE(17)
        CFP_SWITCH_RESERVED_CASE(18)
        CFP_SWITCH_RESERVED_CASE(19)
        CFP_SWITCH_RESERVED_CASE(20)
        CFP_SWITCH_RESERVED_CASE(21)
        CFP_SWITCH_RESERVED_CASE(22)
        CFP_SWITCH_RESERVED_CASE(23)
        CFP_SWITCH_RESERVED_CASE(24)
        CFP_SWITCH_RESERVED_CASE(25)
        CFP_SWITCH_RESERVED_CASE(26)
        CFP_SWITCH_RESERVED_CASE(27)
        CFP_SWITCH_RESERVED_CASE(28)
        CFP_SWITCH_RESERVED_CASE(29)
        CFP_SWITCH_RESERVED_CASE(30)
        CFP_SWITCH_RESERVED_CASE(31)
        default:
            return x;
    }
}
