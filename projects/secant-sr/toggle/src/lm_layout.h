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
#ifndef SECANT_SR_LM_LAYOUT_H
#define SECANT_SR_LM_LAYOUT_H
#define SR_LM_NODES 63
#define SR_LM_PARAMETERS 8
#define SR_LM_STATISTICS 45
/* Private host/device wire layout. No pointers or native-size fields. */
typedef struct SRLMInstruction {
  unsigned op, a, b;
  float value;
} SRLMInstruction;
typedef struct SRLMProgram {
  unsigned count, parameters, model, permutation;
  unsigned parameter_ids[SR_LM_PARAMETERS];
  float center[SR_LM_PARAMETERS];
  SRLMInstruction code[SR_LM_NODES];
} SRLMProgram;
typedef struct SRLMState {
  float best[SR_LM_PARAMETERS], trial[SR_LM_PARAMETERS];
  double loss, damping;
  unsigned accepted, invalid;
} SRLMState;
#endif
