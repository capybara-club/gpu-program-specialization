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

int main(int argc, char **argv)
{
    double max_step = 0.01;

    if (argc > 2) {
        fprintf(stderr, "usage: %s [maximum_rk4_step_hours]\n", argv[0]);
        return 2;
    }

    if (argc == 2) {
        char *end = NULL;
        errno = 0;
        max_step = strtod(argv[1], &end);
        if (errno != 0 || end == argv[1] || *end != '\0' || !(max_step > 0.0)) {
            fprintf(stderr, "maximum RK4 step must be a positive number\n");
            return 2;
        }
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

    printf("experiment,time_h,biomass_g_L,glucose_g_L,sucrose_g_L,astaxanthin_mg_L\n");

    for (int experiment = 0; experiment < 3; ++experiment) {
        simulate_astaxanthin_ground_truth(
            observation_times, observation_count, max_step,
            initial_biomass[experiment], initial_glucose[experiment],
            initial_sucrose[experiment], 0.0, biomass, glucose, sucrose,
            astaxanthin);

        for (int observation = 0; observation < observation_count; ++observation) {
            printf("%d,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                   experiment + 1, observation_times[observation],
                   biomass[observation], glucose[observation], sucrose[observation],
                   astaxanthin[observation]);
        }
    }

    return 0;
}
