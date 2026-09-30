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
#include "internal.h"

unsigned sr_selector_choice(const SRNode *n, uint32_t permutation) {
  unsigned choice = n->choices > 1 ? (permutation >> n->low_bit) & 1u : 0;
  if (n->choices == 4) choice |= ((permutation >> n->high_bit) & 1u) << 1;
  return choice;
}

void sr_selector_rebind(SRNode *n, unsigned low, unsigned high,
                        uint32_t old_permutation, uint32_t new_permutation) {
  SRLeaf old[4];
  unsigned i, before = sr_selector_choice(n, old_permutation), after;
  if (n->choices < 2) return;
  memcpy(old, n->leaf, sizeof(old));
  n->low_bit = (uint8_t)low;
  n->high_bit = n->choices == 4 ? (uint8_t)high : 0;
  after = sr_selector_choice(n, new_permutation);
  /* XOR is a bijection over the 2/4-way alternatives. It also preserves
   * correlations between nodes using the same mapped bit and polarity. */
  for (i = 0; i < n->choices; ++i) n->leaf[i ^ before ^ after] = old[i];
}

unsigned sr_align_subtree(SRNode *nodes, size_t count, size_t start, size_t length,
                          unsigned bits, uint32_t donor, uint32_t recipient) {
  unsigned map[SECANT_SR_MAX_BITS], used = 0, inside = 0, assigned = 0, reused = 0, b;
  size_t i;
  for (i = 0; i < count; ++i) {
    const SRNode *n = nodes+i;
    unsigned mask = 0;
    if (n->choices > 1) mask |= 1u << n->low_bit;
    if (n->choices == 4) mask |= 1u << n->high_bit;
    if (i >= start && i-start < length) inside |= mask;
    else used |= mask;
  }
  for (b = 0; b < bits; ++b) if (inside & (1u << b)) {
    unsigned target;
    if (!((used | assigned) & (1u << b))) target = b;
    else {
      for (target = 0; target < bits; ++target)
        if (!((used | assigned) & (1u << target))) break;
      if (target == bits) {
        for (target = 0; target < bits; ++target)
          if (!(assigned & (1u << target))) break;
      }
    }
    map[b] = target;
    assigned |= 1u << target;
    if (used & (1u << target)) ++reused;
  }
  for (i = start; i < start+length; ++i) {
    SRNode *n = nodes+i;
    if (n->choices > 1)
      sr_selector_rebind(n, map[n->low_bit], n->choices == 4 ? map[n->high_bit] : 0,
                          donor, recipient);
  }
  return reused;
}
