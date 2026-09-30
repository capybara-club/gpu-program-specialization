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
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#ifndef PERELSON_MAX_STEP
#define PERELSON_MAX_STEP 0.04f
#endif

/*
 * Complete one-setting trajectory score for the PEtab Perelson_Science1996
 * HIV model. The held-out infected-cell RHS is marked below as the Secant
 * specialization site. The kernel returns both the search-friendly log10 SSE
 * and the formal PEtab log10-normal negative log likelihood.
 *
 * Benchmark problem:
 * https://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab/tree/master/Benchmark-Models/Perelson_Science1996
 *
 * The repository's simulatedData table is not a nominal-parameter reference.
 * The maintainers list Perelson among the invalid simulation tables in issue
 * #278. This program therefore validates the nominal trajectory against the
 * closed-form solution of the declared SBML model. It also reports a separate,
 * explicitly labeled compatibility check for the historical table.
 */

extern "C" __global__ void score_perelson_candidate(
    const float *observation_times, const float *measured_total_virus, int observation_count,
    float max_step, float initial_infected_cells, float initial_total_virus,
    float initial_infectious_virus, float initial_noninfectious_virus,
    float infection_rate, float target_cell_count, float clearance,
    float infected_cell_death, float virions_per_infected_cell,
    float noise_standard_deviation, float *predicted_total_virus,
    float *log10_sse_out, float *negative_log_likelihood_out)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    float infected_cells = initial_infected_cells;
    float total_virus = initial_total_virus;
    float infectious_virus = initial_infectious_virus;
    float noninfectious_virus = initial_noninfectious_virus;
    float current_time = 0.0f;
    float log10_sse = 0.0f;
    float negative_log_likelihood = 0.0f;

    for (int observation = 0; observation < observation_count; ++observation) {
        float observation_time = observation_times[observation];

        while (current_time < observation_time) {
            float h = observation_time - current_time;
            if (h > max_step) h = max_step;

            float half_h = h * 0.5f;
            float sixth_h = h * 0.1666666716f;

            float base_infected_cells = infected_cells;
            float base_total_virus = total_virus;
            float base_infectious_virus = infectious_virus;
            float base_noninfectious_virus = noninfectious_virus;

            float stage_infected_cells = infected_cells;
            float stage_infectious_virus = infectious_virus;
            float stage_noninfectious_virus = noninfectious_virus;

            float sum_infected_cells = 0.0f;
            float sum_total_virus = 0.0f;
            float sum_infectious_virus = 0.0f;
            float sum_noninfectious_virus = 0.0f;

            float d_infected_cells;
            float d_total_virus;
            float d_infectious_virus;
            float d_noninfectious_virus;
            int stage = 0;

evaluate_rhs:
            /* SECANT SPECIALIZATION SITE: the held-out infected-cell RHS. */
            d_infected_cells = infection_rate * target_cell_count * stage_infectious_virus
                             - infected_cell_death * stage_infected_cells;

            float virus_production = infected_cell_death * virions_per_infected_cell * stage_infected_cells;
            d_total_virus = virus_production - clearance * stage_infectious_virus
                          - clearance * stage_noninfectious_virus;
            d_infectious_virus = -clearance * stage_infectious_virus;
            d_noninfectious_virus = virus_production - clearance * stage_noninfectious_virus;

            if (stage == 0) goto finish_stage_0;
            if (stage == 1) goto finish_stage_1;
            if (stage == 2) goto finish_stage_2;
            goto finish_stage_3;

finish_stage_0:
            sum_infected_cells = d_infected_cells;
            sum_total_virus = d_total_virus;
            sum_infectious_virus = d_infectious_virus;
            sum_noninfectious_virus = d_noninfectious_virus;

            stage_infected_cells = base_infected_cells + half_h * d_infected_cells;
            stage_infectious_virus = base_infectious_virus + half_h * d_infectious_virus;
            stage_noninfectious_virus = base_noninfectious_virus + half_h * d_noninfectious_virus;

            stage = 1;
            goto evaluate_rhs;

finish_stage_1:
            sum_infected_cells += d_infected_cells + d_infected_cells;
            sum_total_virus += d_total_virus + d_total_virus;
            sum_infectious_virus += d_infectious_virus + d_infectious_virus;
            sum_noninfectious_virus += d_noninfectious_virus + d_noninfectious_virus;

            stage_infected_cells = base_infected_cells + half_h * d_infected_cells;
            stage_infectious_virus = base_infectious_virus + half_h * d_infectious_virus;
            stage_noninfectious_virus = base_noninfectious_virus + half_h * d_noninfectious_virus;

            stage = 2;
            goto evaluate_rhs;

finish_stage_2:
            sum_infected_cells += d_infected_cells + d_infected_cells;
            sum_total_virus += d_total_virus + d_total_virus;
            sum_infectious_virus += d_infectious_virus + d_infectious_virus;
            sum_noninfectious_virus += d_noninfectious_virus + d_noninfectious_virus;

            stage_infected_cells = base_infected_cells + h * d_infected_cells;
            stage_infectious_virus = base_infectious_virus + h * d_infectious_virus;
            stage_noninfectious_virus = base_noninfectious_virus + h * d_noninfectious_virus;

            stage = 3;
            goto evaluate_rhs;

finish_stage_3:
            sum_infected_cells += d_infected_cells;
            sum_total_virus += d_total_virus;
            sum_infectious_virus += d_infectious_virus;
            sum_noninfectious_virus += d_noninfectious_virus;

            infected_cells = base_infected_cells + sixth_h * sum_infected_cells;
            total_virus = base_total_virus + sixth_h * sum_total_virus;
            infectious_virus = base_infectious_virus + sixth_h * sum_infectious_virus;
            noninfectious_virus = base_noninfectious_virus + sixth_h * sum_noninfectious_virus;
            current_time += h;
        }

        float measurement = measured_total_virus[observation];
        if (!(total_virus > 0.0f) || !(measurement > 0.0f) || !isfinite(total_virus)) {
            *log10_sse_out = INFINITY;
            *negative_log_likelihood_out = INFINITY;
            return;
        }

        predicted_total_virus[observation] = total_virus;
        float residual = __log10f(measurement) - __log10f(total_virus);
        log10_sse += residual * residual;

        float standardized_residual = residual / noise_standard_deviation;
        negative_log_likelihood += 0.9189385332f + __logf(noise_standard_deviation)
                                 + __logf(measurement) + 0.8340324452f
                                 + 0.5f * standardized_residual * standardized_residual;
    }

    *log10_sse_out = log10_sse;
    *negative_log_likelihood_out = negative_log_likelihood;
}

int main(void)
{
    const int observation_count = 16;
    const float max_step = PERELSON_MAX_STEP;

    const float observation_times[observation_count] = {
        0.0f, 0.105f, 0.169f, 0.282f, 0.492f, 0.757f, 1.029f, 1.253f,
        1.533f, 1.750f, 2.038f, 3.013f, 3.987f, 4.968f, 5.972f, 6.973f
    };

    const float measured_total_virus[observation_count] = {
        1029000.0f, 1087000.0f, 2158000.0f, 1860000.0f,
        2195000.0f, 1567000.0f, 3208000.0f, 2293000.0f,
        1342000.0f, 1197000.0f, 987100.0f, 697300.0f,
        503800.0f, 282200.0f, 172700.0f, 91618.242f
    };

    const float repository_simulated_total_virus[observation_count] = {
        1860000.0f, 1979067.19116451f, 2036200.82349015f, 2110605.53393048f,
        2170386.58687064f, 2133014.95269381f, 2007892.04532337f, 1866529.16992371f,
        1667316.74545722f, 1508578.7515073f, 1304224.95565882f, 745147.675932322f,
        404793.605719818f, 214718.67877913f, 111378.497695453f, 57731.3816378903f
    };

    float *device_times;
    float *device_targets;
    float *device_predictions;
    float *device_log10_sse;
    float *device_negative_log_likelihood;
    cudaError_t error;

    error = cudaMalloc((void **)&device_times, sizeof(observation_times));
    if (error != cudaSuccess) return 1;
    error = cudaMalloc((void **)&device_targets, sizeof(measured_total_virus));
    if (error != cudaSuccess) return 1;
    error = cudaMalloc((void **)&device_predictions, sizeof(measured_total_virus));
    if (error != cudaSuccess) return 1;
    error = cudaMalloc((void **)&device_log10_sse, sizeof(float));
    if (error != cudaSuccess) return 1;
    error = cudaMalloc((void **)&device_negative_log_likelihood, sizeof(float));
    if (error != cudaSuccess) return 1;

    cudaMemcpy(device_times, observation_times, sizeof(observation_times), cudaMemcpyHostToDevice);
    cudaMemcpy(device_targets, measured_total_virus, sizeof(measured_total_virus), cudaMemcpyHostToDevice);

    float nominal_predictions[observation_count];
    float nominal_log10_sse;
    float nominal_negative_log_likelihood;
    const float nominal_noise_standard_deviation = 100000.0f;

    score_perelson_candidate<<<1, 1>>>(
        device_times, device_targets, observation_count,
        max_step, 15061.32075f, 1860000.0f, 1860000.0f, 0.0f,
        3.9e-7f, 11000.0f, 2.06f, 0.53f, 480.0f,
        nominal_noise_standard_deviation, device_predictions,
        device_log10_sse, device_negative_log_likelihood);

    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        fprintf(stderr, "nominal kernel failed: %s\n", cudaGetErrorString(error));
        return 1;
    }

    cudaMemcpy(nominal_predictions, device_predictions, sizeof(nominal_predictions), cudaMemcpyDeviceToHost);
    cudaMemcpy(&nominal_log10_sse, device_log10_sse, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&nominal_negative_log_likelihood, device_negative_log_likelihood, sizeof(float), cudaMemcpyDeviceToHost);

    double cpu_log10_sse = 0.0;
    double cpu_negative_log_likelihood = 0.0;
    double maximum_nominal_relative_error = 0.0;

    for (int observation = 0; observation < observation_count; ++observation) {
        double time = observation_times[observation];
        double clearance = 2.06;
        double infected_cell_death = 0.53;
        double rate_difference = clearance - infected_cell_death;
        double infection_source = 3.9e-7 * 11000.0 * 1860000.0;
        double source_ratio = infection_source / rate_difference;
        double infected_coefficient = 15061.32075 + source_ratio;
        double expected_total_virus = 1860000.0 * exp(-clearance * time)
            + infected_cell_death * 480.0
            * (infected_coefficient * (exp(-infected_cell_death * time) - exp(-clearance * time))
               / rate_difference - source_ratio * time * exp(-clearance * time));

        double relative_error = fabs((double)nominal_predictions[observation] / expected_total_virus - 1.0);
        if (relative_error > maximum_nominal_relative_error) maximum_nominal_relative_error = relative_error;

        double measurement = measured_total_virus[observation];
        double residual = log10(measurement) - log10(expected_total_virus);
        cpu_log10_sse += residual * residual;
        cpu_negative_log_likelihood += 0.5 * log(2.0 * 3.14159265358979323846)
            + log(nominal_noise_standard_deviation) + log(measurement) + log(log(10.0))
            + 0.5 * residual * residual
            / ((double)nominal_noise_standard_deviation * nominal_noise_standard_deviation);
    }

    float wrong_log10_sse;
    score_perelson_candidate<<<1, 1>>>(
        device_times, device_targets, observation_count,
        max_step, 15061.32075f, 1860000.0f, 1860000.0f, 0.0f,
        1.5e-7f, 11000.0f, 2.06f, 0.53f, 480.0f,
        nominal_noise_standard_deviation, device_predictions,
        device_log10_sse, device_negative_log_likelihood);

    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        fprintf(stderr, "wrong-candidate kernel failed: %s\n", cudaGetErrorString(error));
        return 1;
    }
    cudaMemcpy(&wrong_log10_sse, device_log10_sse, sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemcpy(
        device_targets, repository_simulated_total_virus,
        sizeof(repository_simulated_total_virus), cudaMemcpyHostToDevice);

    float historical_predictions[observation_count];
    float historical_table_log10_sse;
    score_perelson_candidate<<<1, 1>>>(
        device_times, device_targets, observation_count,
        max_step, 15061.32075f, 1860000.0f, 1860000.0f, 0.0f,
        3.9e-7f, 11000.0f, 1.8639249479939883f, 0.6578941256622852f, 480.0f,
        1.0f, device_predictions, device_log10_sse, device_negative_log_likelihood);

    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        fprintf(stderr, "historical-table kernel failed: %s\n", cudaGetErrorString(error));
        return 1;
    }

    cudaMemcpy(historical_predictions, device_predictions, sizeof(historical_predictions), cudaMemcpyDeviceToHost);
    cudaMemcpy(&historical_table_log10_sse, device_log10_sse, sizeof(float), cudaMemcpyDeviceToHost);

    double maximum_historical_relative_error = 0.0;
    for (int observation = 0; observation < observation_count; ++observation) {
        double relative_error = fabs(
            (double)historical_predictions[observation]
            / repository_simulated_total_virus[observation] - 1.0);
        if (relative_error > maximum_historical_relative_error) maximum_historical_relative_error = relative_error;
    }

    printf("nominal CUDA log10 SSE:                    %.9g\n", nominal_log10_sse);
    printf("nominal analytic log10 SSE:                %.9g\n", cpu_log10_sse);
    printf("nominal CUDA PEtab negative log likelihood: %.9g\n", nominal_negative_log_likelihood);
    printf("nominal analytic negative log likelihood:   %.9g\n", cpu_negative_log_likelihood);
    printf("nominal maximum trajectory relative error:  %.9g\n", maximum_nominal_relative_error);
    printf("wrong-candidate log10 SSE:                  %.9g\n", wrong_log10_sse);
    printf("historical table inferred-parameter SSE:    %.9g\n", historical_table_log10_sse);
    printf("historical table maximum relative error:    %.9g\n", maximum_historical_relative_error);

    int validation_failed = maximum_nominal_relative_error > 2.0e-5
        || fabs((double)nominal_log10_sse - cpu_log10_sse) > 1.0e-5
        || fabs((double)nominal_negative_log_likelihood - cpu_negative_log_likelihood) > 1.0e-3
        || historical_table_log10_sse > 1.0e-8;

    cudaFree(device_negative_log_likelihood);
    cudaFree(device_log10_sse);
    cudaFree(device_predictions);
    cudaFree(device_targets);
    cudaFree(device_times);

    if (validation_failed) {
        fprintf(stderr, "validation failed\n");
        return 1;
    }

    printf("validation passed\n");
    return 0;
}
