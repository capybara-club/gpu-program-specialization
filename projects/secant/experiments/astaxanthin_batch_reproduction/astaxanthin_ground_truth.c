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

/*
 * Ground-truth dual-substrate batch model from Riezzo et al. (2026),
 * Equations 6a-6d and Table 4. S1 is glucose and S2 is sucrose.
 *
 * The four RK4 stages share one inline RHS site. This is intentionally close
 * to the shape that can later be placed inside a specialized kernel, but this
 * file is only a CPU reproduction reference and contains no search machinery.
 */
void simulate_astaxanthin_ground_truth(
    const double *observation_times, int observation_count, double max_step,
    double initial_biomass, double initial_glucose, double initial_sucrose,
    double initial_astaxanthin, double *biomass_out, double *glucose_out,
    double *sucrose_out, double *astaxanthin_out)
{
    const double mu_m1 = 0.43;
    const double mu_m2 = 0.132;
    const double K_c1 = 63.7;
    const double K_c2 = 3.68;
    const double k_1 = 5.8;
    const double k_2 = 0.0;
    const double mu_d = 0.0055;
    const double Y_S1 = 2.58;
    const double Y_S2 = 1.71;
    const double alpha_1 = 0.0;
    const double alpha_2 = 0.0;
    const double beta = 0.21;
    const double k_d = 0.0466;

    double biomass = initial_biomass;
    double glucose = initial_glucose;
    double sucrose = initial_sucrose;
    double astaxanthin = initial_astaxanthin;
    double current_time = 0.0;

    for (int observation = 0; observation < observation_count; ++observation) {
        const double observation_time = observation_times[observation];

        while (current_time < observation_time) {
            double h = observation_time - current_time;
            if (h > max_step) h = max_step;

            const double half_h = h * 0.5;
            const double sixth_h = h / 6.0;

            const double base_biomass = biomass;
            const double base_glucose = glucose;
            const double base_sucrose = sucrose;
            const double base_astaxanthin = astaxanthin;

            double stage_biomass = biomass;
            double stage_glucose = glucose;
            double stage_sucrose = sucrose;
            double stage_astaxanthin = astaxanthin;

            double sum_biomass = 0.0;
            double sum_glucose = 0.0;
            double sum_sucrose = 0.0;
            double sum_astaxanthin = 0.0;

            double glucose_growth;
            double sucrose_growth;
            double d_biomass;
            double d_glucose;
            double d_sucrose;
            double d_astaxanthin;
            int stage = 0;

evaluate_rhs:
            /* P is a state but does not feed back into this published RHS. */
            (void)stage_astaxanthin;
            glucose_growth = mu_m1 * stage_glucose * stage_biomass
                           / ((stage_glucose + K_c1 * stage_biomass)
                           * (1.0 + k_1 * stage_sucrose));
            sucrose_growth = mu_m2 * stage_sucrose * stage_biomass
                           / ((stage_sucrose + K_c2 * stage_biomass)
                           * (1.0 + k_2 * stage_glucose));

            d_biomass = glucose_growth + sucrose_growth - mu_d * stage_biomass;
            d_glucose = -Y_S1 * glucose_growth;
            d_sucrose = -Y_S2 * sucrose_growth;
            d_astaxanthin = alpha_1 * glucose_growth + alpha_2 * sucrose_growth
                          + beta * stage_biomass - k_d * stage_biomass * stage_biomass;

            if (stage == 0) goto finish_stage_0;
            if (stage == 1) goto finish_stage_1;
            if (stage == 2) goto finish_stage_2;
            goto finish_stage_3;

finish_stage_0:
            sum_biomass = d_biomass;
            sum_glucose = d_glucose;
            sum_sucrose = d_sucrose;
            sum_astaxanthin = d_astaxanthin;

            stage_biomass = base_biomass + half_h * d_biomass;
            stage_glucose = base_glucose + half_h * d_glucose;
            stage_sucrose = base_sucrose + half_h * d_sucrose;
            stage_astaxanthin = base_astaxanthin + half_h * d_astaxanthin;
            stage = 1;
            goto evaluate_rhs;

finish_stage_1:
            sum_biomass += 2.0 * d_biomass;
            sum_glucose += 2.0 * d_glucose;
            sum_sucrose += 2.0 * d_sucrose;
            sum_astaxanthin += 2.0 * d_astaxanthin;

            stage_biomass = base_biomass + half_h * d_biomass;
            stage_glucose = base_glucose + half_h * d_glucose;
            stage_sucrose = base_sucrose + half_h * d_sucrose;
            stage_astaxanthin = base_astaxanthin + half_h * d_astaxanthin;
            stage = 2;
            goto evaluate_rhs;

finish_stage_2:
            sum_biomass += 2.0 * d_biomass;
            sum_glucose += 2.0 * d_glucose;
            sum_sucrose += 2.0 * d_sucrose;
            sum_astaxanthin += 2.0 * d_astaxanthin;

            stage_biomass = base_biomass + h * d_biomass;
            stage_glucose = base_glucose + h * d_glucose;
            stage_sucrose = base_sucrose + h * d_sucrose;
            stage_astaxanthin = base_astaxanthin + h * d_astaxanthin;
            stage = 3;
            goto evaluate_rhs;

finish_stage_3:
            sum_biomass += d_biomass;
            sum_glucose += d_glucose;
            sum_sucrose += d_sucrose;
            sum_astaxanthin += d_astaxanthin;

            biomass = base_biomass + sixth_h * sum_biomass;
            glucose = base_glucose + sixth_h * sum_glucose;
            sucrose = base_sucrose + sixth_h * sum_sucrose;
            astaxanthin = base_astaxanthin + sixth_h * sum_astaxanthin;
            current_time += h;
        }

        biomass_out[observation] = biomass;
        glucose_out[observation] = glucose;
        sucrose_out[observation] = sucrose;
        astaxanthin_out[observation] = astaxanthin;
    }
}
