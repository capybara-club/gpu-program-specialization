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
#include "astaxanthin_ground_truth.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s MAXIMUM_RK4_STEP_HOURS CANDIDATE_COUNT\n", argv[0]);
        return 2;
    }

    char *step_end = NULL;
    char *count_end = NULL;
    errno = 0;
    const double max_step = strtod(argv[1], &step_end);
    const long candidate_count = strtol(argv[2], &count_end, 10);
    if (errno != 0 || step_end == argv[1] || *step_end != '\0' || !(max_step > 0.0)
        || count_end == argv[2] || *count_end != '\0' || candidate_count <= 0) {
        fprintf(stderr, "step and candidate count must be positive numbers\n");
        return 2;
    }

    const int observation_count = 13;
    const double observation_times[13] = {
        0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0,
        56.0, 64.0, 72.0, 80.0, 88.0, 96.0
    };
    const double initial_biomass[3] = {0.1, 0.1, 0.2};
    const double initial_glucose[3] = {10.0, 20.0, 5.0};
    const double initial_sucrose[3] = {5.0, 5.0, 2.5};

    double biomass[13];
    double glucose[13];
    double sucrose[13];
    double astaxanthin[13];
    volatile double checksum = 0.0;

    const clock_t start = clock();
    for (long candidate = 0; candidate < candidate_count; ++candidate) {
        for (int experiment = 0; experiment < 3; ++experiment) {
            simulate_astaxanthin_ground_truth(
                observation_times, observation_count, max_step,
                initial_biomass[experiment], initial_glucose[experiment],
                initial_sucrose[experiment], 0.0, biomass, glucose, sucrose,
                astaxanthin);
            checksum += biomass[12] + glucose[12] + sucrose[12] + astaxanthin[12];
        }
    }
    const clock_t finish = clock();

    const double elapsed_seconds = (double)(finish - start) / CLOCKS_PER_SEC;
    const double rk4_steps_per_candidate = 3.0 * 96.0 / max_step;
    const double rhs_evaluations_per_candidate = 4.0 * rk4_steps_per_candidate;

    printf("maximum RK4 step: %.6g h\n", max_step);
    printf("candidates: %ld\n", candidate_count);
    printf("RK4 steps per candidate: %.0f\n", rk4_steps_per_candidate);
    printf("RHS evaluations per candidate: %.0f\n", rhs_evaluations_per_candidate);
    printf("elapsed CPU time: %.6f s\n", elapsed_seconds);
    printf("time per candidate: %.3f us\n", 1.0e6 * elapsed_seconds / candidate_count);
    printf("candidate throughput: %.3f candidates/s\n", candidate_count / elapsed_seconds);
    printf("RHS throughput: %.3f million/s\n",
           candidate_count * rhs_evaluations_per_candidate / elapsed_seconds / 1.0e6);
    printf("checksum: %.17g\n", checksum);
    return 0;
}
