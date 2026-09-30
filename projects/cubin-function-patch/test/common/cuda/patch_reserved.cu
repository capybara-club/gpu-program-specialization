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
 * Reserved template function for patch tests.
 *
 * This deliberately large `patch_site` body creates the function slot in the
 * fully linked template cubin. Replacement RDC functions must fit inside this
 * slot; metadata from this worst-case template is preserved by the patcher.
 */

#define CFP_RESERVED_STEP(I) \
    y = y + (float)(I); \
    y = y * 1.0001f

#define CFP_RESERVED_STEP_16(BASE) \
    CFP_RESERVED_STEP((BASE) + 0); \
    CFP_RESERVED_STEP((BASE) + 1); \
    CFP_RESERVED_STEP((BASE) + 2); \
    CFP_RESERVED_STEP((BASE) + 3); \
    CFP_RESERVED_STEP((BASE) + 4); \
    CFP_RESERVED_STEP((BASE) + 5); \
    CFP_RESERVED_STEP((BASE) + 6); \
    CFP_RESERVED_STEP((BASE) + 7); \
    CFP_RESERVED_STEP((BASE) + 8); \
    CFP_RESERVED_STEP((BASE) + 9); \
    CFP_RESERVED_STEP((BASE) + 10); \
    CFP_RESERVED_STEP((BASE) + 11); \
    CFP_RESERVED_STEP((BASE) + 12); \
    CFP_RESERVED_STEP((BASE) + 13); \
    CFP_RESERVED_STEP((BASE) + 14); \
    CFP_RESERVED_STEP((BASE) + 15)

extern "C" __device__ __noinline__
float
patch_site(
    float x
) {
    volatile float y = x;
    CFP_RESERVED_STEP_16(0);
    CFP_RESERVED_STEP_16(16);
    CFP_RESERVED_STEP_16(32);
    CFP_RESERVED_STEP_16(48);
    CFP_RESERVED_STEP_16(64);
    CFP_RESERVED_STEP_16(80);
    CFP_RESERVED_STEP_16(96);
    CFP_RESERVED_STEP_16(112);
    CFP_RESERVED_STEP_16(128);
    CFP_RESERVED_STEP_16(144);
    CFP_RESERVED_STEP_16(160);
    CFP_RESERVED_STEP_16(176);
    return (float)y;
}
