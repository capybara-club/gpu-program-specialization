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
#ifndef ODZ_SIZING_H
#define ODZ_SIZING_H
#include <stdint.h>
#define ODZ_SCORING_BLOCK_THREADS 128u
#define ODZ_TILE_SIZING_POLICY "parallel-work-v1"
/* Pure host policy, independent of CUDA handles and grammar traversal. A wave
 * is the hardware thread-capacity upper bound, not achieved SM occupancy. */
typedef struct OdzSizing {
    uint32_t sms,threads_per_sm;
    uint64_t wave_configurations,target_configurations,work_configurations;
    uint64_t minimum_configurations,maximum_systems,minimum_blocks,maximum_blocks,total_blocks;
    uint64_t underfilled_tiles,underfilled_configurations;
    uint64_t memory_limited_tiles,explicit_limited_tiles,available_limited_tiles;
    uint64_t balanced_tiles,over_target_seconds,call_histogram[32];
    double maximum_call_seconds,underfilled_call_seconds;
} OdzSizing;
int odz_sizing_init(OdzSizing *s,uint32_t sms,uint32_t threads_per_sm,uint64_t work_units);
uint64_t odz_sizing_chunk(const OdzSizing *s,uint64_t ceiling,int automatic,double rate,double seconds);
/* Evenly partition a bank without changing its indices or evaluating padding. */
uint64_t odz_sizing_banks(uint64_t remaining,uint64_t ceiling);
void odz_sizing_record(OdzSizing *s,uint64_t systems,uint64_t banks,uint64_t permutations,
    int memory_limited,int explicit_limited,int available_limited,int balanced,
    double call_seconds,double target_seconds);
double odz_sizing_percentile(const OdzSizing *s,unsigned percent);
#endif
