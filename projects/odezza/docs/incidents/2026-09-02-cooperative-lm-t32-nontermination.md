<!--
SPDX-FileCopyrightText: 2026 Charles Durham
SPDX-License-Identifier: MIT

MIT License

Copyright (c) 2026 Charles Durham

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
-->

# Cooperative LM 32-thread validation non-termination

Date: 2026-09-02  
Status: open; process terminated manually with user approval  
Affected path: specialized eight-state/eight-constant LM with 32 threads per fit and a 12-iteration limit

## Observation

A post-study validation requested eight fits, one timed repetition, and at most 12 LM iterations from `odezza-lm-8x8-t32-replicated-sm120-specialized.cubin`. It held the RTX 5090 at full compute utilization for more than 56 minutes and produced no result. The process was terminated before running the eight-state scoring benchmark.

This execution is not the completed 32-thread result in the cooperative topology table, which used the validated study configuration and reported 49.6 thousand fits per second on the RTX 5090. The non-terminating run is excluded from every performance result.

## Impact

The 32-thread cooperative shape must not be treated as production-safe at the longer iteration setting until this is reproduced under a bounded watchdog and diagnosed. The one- and four-thread decisions and the scoring results do not depend on this run.

## Required follow-up

Reproduce with progressively larger iteration limits, add device-side progress counters or a bounded single-fit diagnostic, and determine whether the failure is an optimizer-loop control defect, subgroup synchronization defect, or corrupted specialized branch path.
