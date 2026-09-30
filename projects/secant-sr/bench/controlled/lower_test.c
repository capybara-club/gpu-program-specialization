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
#include "lower.h"
#include "secant_instructions.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"line %d: %s\n",__LINE__,#x);exit(1);}}while(0)
int
main(void)
{
  uint8_t code[128] = {0}, old[128], resolved[128];
  BTree t;
  BChoice choices[32];
  size_t at = 0, slots = 0;
  float banks[4] = {2, -3, 4, 5};
  uint32_t masks[8], words[8 * 32];
  unsigned q;
  code[at++] = 0xb6;
  code[at++] = 0;
  code[at++] = 0xbc;
  code[at++] = 1;
  secant_ast_affine_bank_write(code + at, 0, .25f, 1.f);
  at += 10;
  code[at++] = 0x81; {
    float v = -2;
    memcpy(code + at, &v, 4);
    at += 4;
  }
  code[at++] = 0xbe;
  code[at++] = 0;
  code[at++] = 1;
  code[at++] = 0xb6;
  code[at++] = 1;
  code[at++] = 0x85;
  code[at++] = 0x83;
  CHECK(b_parse(code, at, 2, 2, 2, &t));
  CHECK(t.bytes == at);
  CHECK(b_lower(&t, old, sizeof(old), choices, &slots, 32));
  CHECK(slots == 2);
  CHECK(old[0] == 0xb9 && old[1] == 0 && old[2] == 0xb9 && old[3] == 1 && old[4] == 0x85);
  CHECK(b_tables(choices, slots, 32, banks, 2, 2, 2, masks, words));
  for (q = 0; q < 8; ++q) {
    float value;
    CHECK(masks[q] & 2);
    CHECK(words[q * 32 + 1] == 1);
    if (q % 4 == 0)
      CHECK((masks[q] & 1) && words[q * 32] == 0);
    else {
      CHECK(!(masks[q] & 1));
      memcpy(&value, words + q * 32, 4);
      CHECK(value == (q % 4 == 1 ? banks[(q / 4) * 2 + 1] : q % 4 == 2 ? .25f * banks[(q / 4) * 2] + 1 : -2));
    }
    CHECK(b_resolve(&t, q, banks + (q / 4) * 2, resolved, sizeof(resolved)));
  }
  CHECK(!b_parse(code, at - 1, 2, 2, 2, &t));
  CHECK(!b_parse(code, at, 2, 2, 1, &t));
  CHECK(b_parse(code, at, 2, 2, 2, &t));
  slots = 31;
  CHECK(!b_lower(&t, old, sizeof(old), choices, &slots, 32));
  code[at - 2] = 0x8d;
  CHECK(b_parse(code, at, 2, 2, 2, &t));
  code[at - 2] = 0xff;
  CHECK(!b_parse(code, at, 2, 2, 2, &t));
  puts("settings lowering, affine proposals, bit masks and rejection checks passed");
  return 0;
}
