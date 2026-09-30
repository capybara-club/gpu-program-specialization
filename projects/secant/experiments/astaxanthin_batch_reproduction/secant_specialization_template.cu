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
 * Secant specialization template for the concentration-space search in
 * Riezzo et al. (2026).
 *
 * This is an integration design example, not yet a benchmark target. Compile
 * it once to a CUBIN, inspect the marker island as a materialize-shaped Secant
 * site with 12 inputs, 2 AST outputs, and 128 patch instructions, specialize
 * the two ASTs, and only then load and launch the resulting CUBIN. The
 * unspecialized kernel contains BRKPT instructions and must never be launched.
 *
 * One CUDA thread owns one parameter setting. The specialized AST pair is:
 *
 *     output 0: mu_1(X, S1, S2, P, c0, ..., c7)
 *     output 1: mu_2(X, S1, S2, P, c0, ..., c7)
 *
 * The complete RK4 trajectory is designed to stay in registers. The compiled
 * CUBIN must still be checked for local-memory spills. Only experimental
 * targets are read during integration and one final MSE is written per setting.
 */

#define SECANT_BPT_16 \
    "brkpt;\n\t" "brkpt;\n\t" "brkpt;\n\t" "brkpt;\n\t" \
    "brkpt;\n\t" "brkpt;\n\t" "brkpt;\n\t" "brkpt;\n\t" \
    "brkpt;\n\t" "brkpt;\n\t" "brkpt;\n\t" "brkpt;\n\t" \
    "brkpt;\n\t" "brkpt;\n\t" "brkpt;\n\t" "brkpt;\n\t"

#define SECANT_BPT_128 \
    SECANT_BPT_16 SECANT_BPT_16 SECANT_BPT_16 SECANT_BPT_16 \
    SECANT_BPT_16 SECANT_BPT_16 SECANT_BPT_16 SECANT_BPT_16

extern "C" __global__
void secant_cubin_materialize_000(
    const float *__restrict__ settings,
    unsigned long long settings_leading_dimension,
    unsigned long long num_settings,
    const float *__restrict__ initial_biomass,
    const float *__restrict__ initial_glucose,
    const float *__restrict__ initial_sucrose,
    const float *__restrict__ initial_astaxanthin,
    const float *__restrict__ target_biomass,
    const float *__restrict__ target_glucose,
    const float *__restrict__ target_sucrose,
    const float *__restrict__ target_astaxanthin,
    unsigned int steps_per_observation,
    float *__restrict__ mse_out)
{
    const unsigned long long setting =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (setting >= num_settings || steps_per_observation == 0u) return;

    /* Structure-of-arrays settings keep each warp's parameter loads coalesced. */
    const float c0 = settings[0ull * settings_leading_dimension + setting];
    const float c1 = settings[1ull * settings_leading_dimension + setting];
    const float c2 = settings[2ull * settings_leading_dimension + setting];
    const float c3 = settings[3ull * settings_leading_dimension + setting];
    const float c4 = settings[4ull * settings_leading_dimension + setting];
    const float c5 = settings[5ull * settings_leading_dimension + setting];
    const float c6 = settings[6ull * settings_leading_dimension + setting];
    const float c7 = settings[7ull * settings_leading_dimension + setting];

    const float h = 8.0f / (float)steps_per_observation;
    const float half_h = 0.5f * h;
    const float sixth_h = h / 6.0f;
    float squared_error = 0.0f;
    int valid = 1;

    /* These terms are the known model backbone and are never specialized. */
    const float mu_d = 0.0055f;
    const float Y_S1 = 2.58f;
    const float Y_S2 = 1.71f;
    const float alpha_1 = 0.0f;
    const float alpha_2 = 0.0f;
    const float beta = 0.21f;
    const float k_d = 0.0466f;

#pragma unroll 1
    for (int experiment = 0; experiment < 3; ++experiment) {
        float biomass = initial_biomass[experiment];
        float glucose = initial_glucose[experiment];
        float sucrose = initial_sucrose[experiment];
        float astaxanthin = initial_astaxanthin[experiment];

#pragma unroll 1
        for (int observation = 0; observation < 12; ++observation) {
#pragma unroll 1
            for (unsigned int step = 0; step < steps_per_observation; ++step) {
                const float base_biomass = biomass;
                const float base_glucose = glucose;
                const float base_sucrose = sucrose;
                const float base_astaxanthin = astaxanthin;

                float stage_biomass = biomass;
                float stage_glucose = glucose;
                float stage_sucrose = sucrose;
                float stage_astaxanthin = astaxanthin;

                float sum_biomass = 0.0f;
                float sum_glucose = 0.0f;
                float sum_sucrose = 0.0f;
                float sum_astaxanthin = 0.0f;

#pragma unroll 1
                for (int stage = 0; stage < 4; ++stage) {
                    float marked0 __attribute__((unused));
                    float marked1 __attribute__((unused));
                    float marked2 __attribute__((unused));
                    float marked3 __attribute__((unused));
                    float marked4 __attribute__((unused));
                    float marked5 __attribute__((unused));
                    float marked6 __attribute__((unused));
                    float marked7 __attribute__((unused));
                    float marked8 __attribute__((unused));
                    float marked9 __attribute__((unused));
                    float marked10 __attribute__((unused));
                    float marked11 __attribute__((unused));
                    float specific_growth_1;
                    float specific_growth_2;
                    float keepalive __attribute__((unused));

                    /*
                     * Secant marker ABI:
                     *
                     *   inputs 0..3  = X, S1, S2, P
                     *   inputs 4..11 = c0..c7
                     *   outputs 0..1 = mu_1, mu_2
                     *
                     * The sentinel FADDs let the CUBIN inspector recover the
                     * physical source/output registers. The dummy reduction,
                     * shared keepalive store, and BRKPT reserve are replaced by
                     * specialized SASS and a branch over unused patch space.
                     */
                    asm volatile(
                        "{\n\t"
                        ".reg .u32 keepalive_address;\n\t"
                        "brkpt;\n\t"
                        "add.rn.ftz.f32 %0, %15, 0f7fc0ffee;\n\t"
                        "add.rn.ftz.f32 %1, %16, 0f7fc0ffef;\n\t"
                        "add.rn.ftz.f32 %2, %17, 0f7fc0fff0;\n\t"
                        "add.rn.ftz.f32 %3, %18, 0f7fc0fff1;\n\t"
                        "add.rn.ftz.f32 %4, %19, 0f7fc0fff2;\n\t"
                        "add.rn.ftz.f32 %5, %20, 0f7fc0fff3;\n\t"
                        "add.rn.ftz.f32 %6, %21, 0f7fc0fff4;\n\t"
                        "add.rn.ftz.f32 %7, %22, 0f7fc0fff5;\n\t"
                        "add.rn.ftz.f32 %8, %23, 0f7fc0fff6;\n\t"
                        "add.rn.ftz.f32 %9, %24, 0f7fc0fff7;\n\t"
                        "add.rn.ftz.f32 %10, %25, 0f7fc0fff8;\n\t"
                        "add.rn.ftz.f32 %11, %26, 0f7fc0fff9;\n\t"
                        "brkpt;\n\t"
                        "add.rn.ftz.f32 %0, %0, %1;\n\t"
                        "add.rn.ftz.f32 %0, %0, %2;\n\t"
                        "add.rn.ftz.f32 %0, %0, %3;\n\t"
                        "add.rn.ftz.f32 %0, %0, %4;\n\t"
                        "add.rn.ftz.f32 %0, %0, %5;\n\t"
                        "add.rn.ftz.f32 %0, %0, %6;\n\t"
                        "add.rn.ftz.f32 %0, %0, %7;\n\t"
                        "add.rn.ftz.f32 %0, %0, %8;\n\t"
                        "add.rn.ftz.f32 %0, %0, %9;\n\t"
                        "add.rn.ftz.f32 %0, %0, %10;\n\t"
                        "add.rn.ftz.f32 %0, %0, %11;\n\t"
                        "add.rn.ftz.f32 %12, %0, 0f7fc0fffa;\n\t"
                        "add.rn.ftz.f32 %13, %0, 0f7fc0fffb;\n\t"
                        "brkpt;\n\t"
                        "add.rn.ftz.f32 %14, %12, %13;\n\t"
                        "add.rn.ftz.f32 %14, %14, %15;\n\t"
                        "add.rn.ftz.f32 %14, %14, %16;\n\t"
                        "add.rn.ftz.f32 %14, %14, %17;\n\t"
                        "add.rn.ftz.f32 %14, %14, %18;\n\t"
                        "add.rn.ftz.f32 %14, %14, %19;\n\t"
                        "add.rn.ftz.f32 %14, %14, %20;\n\t"
                        "add.rn.ftz.f32 %14, %14, %21;\n\t"
                        "add.rn.ftz.f32 %14, %14, %22;\n\t"
                        "add.rn.ftz.f32 %14, %14, %23;\n\t"
                        "add.rn.ftz.f32 %14, %14, %24;\n\t"
                        "add.rn.ftz.f32 %14, %14, %25;\n\t"
                        "add.rn.ftz.f32 %14, %14, %26;\n\t"
                        "mov.u32 keepalive_address, 0;\n\t"
                        "st.volatile.shared.f32 [keepalive_address], %14;\n\t"
                        SECANT_BPT_128
                        "}\n\t"
                        : "=&f"(marked0), "=&f"(marked1), "=&f"(marked2),
                          "=&f"(marked3), "=&f"(marked4), "=&f"(marked5),
                          "=&f"(marked6), "=&f"(marked7), "=&f"(marked8),
                          "=&f"(marked9), "=&f"(marked10), "=&f"(marked11),
                          "=&f"(specific_growth_1), "=&f"(specific_growth_2),
                          "=&f"(keepalive)
                        : "f"(stage_biomass), "f"(stage_glucose),
                          "f"(stage_sucrose), "f"(stage_astaxanthin),
                          "f"(c0), "f"(c1), "f"(c2), "f"(c3),
                          "f"(c4), "f"(c5), "f"(c6), "f"(c7)
                        : "memory");

                    const float glucose_growth = specific_growth_1 * stage_biomass;
                    const float sucrose_growth = specific_growth_2 * stage_biomass;
                    const float d_biomass = glucose_growth + sucrose_growth
                                            - mu_d * stage_biomass;
                    const float d_glucose = -Y_S1 * glucose_growth;
                    const float d_sucrose = -Y_S2 * sucrose_growth;
                    const float d_astaxanthin = alpha_1 * glucose_growth
                                              + alpha_2 * sucrose_growth
                                              + beta * stage_biomass
                                              - k_d * stage_biomass * stage_biomass;
                    const float weight = (stage == 0 || stage == 3) ? 1.0f : 2.0f;

                    sum_biomass += weight * d_biomass;
                    sum_glucose += weight * d_glucose;
                    sum_sucrose += weight * d_sucrose;
                    sum_astaxanthin += weight * d_astaxanthin;

                    if (stage != 3) {
                        const float stage_h = stage == 2 ? h : half_h;
                        stage_biomass = base_biomass + stage_h * d_biomass;
                        stage_glucose = base_glucose + stage_h * d_glucose;
                        stage_sucrose = base_sucrose + stage_h * d_sucrose;
                        stage_astaxanthin = base_astaxanthin + stage_h * d_astaxanthin;
                    }
                }

                biomass = base_biomass + sixth_h * sum_biomass;
                glucose = base_glucose + sixth_h * sum_glucose;
                sucrose = base_sucrose + sixth_h * sum_sucrose;
                astaxanthin = base_astaxanthin + sixth_h * sum_astaxanthin;
            }

            if (((__float_as_uint(biomass) & 0x7fffffffu) >= 0x7f800000u) ||
                ((__float_as_uint(glucose) & 0x7fffffffu) >= 0x7f800000u) ||
                ((__float_as_uint(sucrose) & 0x7fffffffu) >= 0x7f800000u) ||
                ((__float_as_uint(astaxanthin) & 0x7fffffffu) >= 0x7f800000u)) {
                valid = 0;
                break;
            }

            const int target = experiment * 12 + observation;
            const float error_biomass = biomass - target_biomass[target];
            const float error_glucose = glucose - target_glucose[target];
            const float error_sucrose = sucrose - target_sucrose[target];
            const float error_astaxanthin = astaxanthin - target_astaxanthin[target];
            squared_error += error_biomass * error_biomass
                           + error_glucose * error_glucose
                           + error_sucrose * error_sucrose
                           + error_astaxanthin * error_astaxanthin;
        }

        if (!valid) break;
    }

    mse_out[setting] = valid ? squared_error / 144.0f : 3.402823466e+38F;
}

#undef SECANT_BPT_128
#undef SECANT_BPT_16
