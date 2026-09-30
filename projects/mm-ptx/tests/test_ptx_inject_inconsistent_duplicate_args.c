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
 * SPDX-FileCopyrightText: 2026 Charles Durham
 *
 * SPDX-License-Identifier: MIT
 */

#define PTX_INJECT_IMPLEMENTATION
#include <ptx_inject.h>

#include <check_result_helper.h>

int
main(void) {
    static const char malformed_ptx[] =
        "// PTX_INJECT_START func\n"
        "// _x0 i f32 F32 v_x\n"
        "// PTX_INJECT_END\n"
        "// PTX_INJECT_START func\n"
        "// _x0 i f32 F32 v_x\n"
        "// _x1 i f32 F32 v_y\n"
        "// PTX_INJECT_END\n";

    PtxInjectHandle handle = NULL;
    PtxInjectResult result = ptx_inject_create(&handle, malformed_ptx);

    ASSERT(result == PTX_INJECT_ERROR_INCONSISTENT_INJECTION);

    if (handle != NULL) {
        ptxInjectCheck( ptx_inject_destroy(handle) );
    }

    return 0;
}
