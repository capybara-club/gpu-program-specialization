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
#include "o_odezza_internal.h"
#include "o_writer.h"
#include <string.h>
#define TRY O_RETURN_IF_ERROR
OdezzaResult o_lm_write_body_0(OWriter *);
OdezzaResult o_lm_write_body_1(OWriter *);

OdezzaResult o_lm_validate_shape(const OdezzaLmShape *s) {
    if (!s || !s->state_count || !s->parameter_count || !s->site_patch_capacity)
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    if (s->state_count > O_LM_MAX_STATES || s->parameter_count > 16u ||
        s->state_count + s->parameter_count > O_LM_MAX_INPUTS ||
        (s->lanes_per_fit != 1u && s->lanes_per_fit != 2u &&
         s->lanes_per_fit != 4u && s->lanes_per_fit != 8u) ||
        s->site_patch_capacity > 65536u)
        return ODEZZA_ERROR_UNSUPPORTED;
    return ODEZZA_SUCCESS;
}
OdezzaResult o_lm_site_layout(const OdezzaLmShape *s, OLmInspection *p) {
    uint32_t rhs, i, j, k = 0, cap, n;
    TRY(o_lm_validate_shape(s));
    if (!p)
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    p->shape = *s;
    cap = 28u - s->state_count - s->parameter_count;
    for (i = 0; i < s->state_count; i += cap) {
        OLmSite *site = &p->sites[k++];
        n = s->state_count - i;
        if (n > cap)
            n = cap;
        site->input_count = s->state_count + s->parameter_count;
        site->output_count = n;
        for (j = 0; j < n; ++j)
            site->output_indices[j] = i + j;
    }
    for (rhs = 0; rhs < s->state_count; ++rhs)
        for (i = 0; i < s->state_count + s->parameter_count; i += cap) {
            OLmSite *site;
            if (k >= O_LM_MAX_SITES)
                return ODEZZA_ERROR_UNSUPPORTED;
            site = &p->sites[k++];
            n = s->state_count + s->parameter_count - i;
            if (n > cap)
                n = cap;
            site->input_count = s->state_count + s->parameter_count;
            site->output_count = n;
            for (j = 0; j < n; ++j)
                site->output_indices[j] = s->state_count + (i + j) * s->state_count + rhs;
        }
    p->site_count = k;
    return ODEZZA_SUCCESS;
}
static OdezzaResult input_name(OWriter *w, uint32_t i, uint32_t states) {
    if (i < states)
        return o_writer_format(w, "stage_state[%u]", i);
    return o_writer_format(w, "evaluation_parameters[%u]", i - states);
}
static OdezzaResult site_source(OWriter *w, const OLmSite *s, uint32_t k, const OdezzaLmShape *shape) {
    uint32_t i, in = s->input_count, out = s->output_count, keep = in + out, perm = keep + 1u;
    uint32_t marker = 0x4fc40000u + k * 256u;
    for (i = 0; i < out; ++i)
        TRY(o_writer_format(w, "float lm_r_%u_%u = 0.0f;\n", k, i));
    TRY(o_writer_format(w, "float lm_keepalive_%u __attribute__((unused));\nasm volatile(\n", k));
    TRY(o_writer_cstr(
        w,
        "\"{\\n\\t.reg .pred pred; .reg .b32 perm, salted, masked; .reg .u32 addr;\\n\\tbrkpt;\\n\\t\"\n"));
    for (i = 0; i < in; ++i)
        TRY(o_writer_format(w, "\"add.rn.ftz.f32 %%%u, %%%u, 0f%08x;\\n\\t\"\n", i, i, marker + i));
    TRY(o_writer_format(w,
                        "\"brkpt;\\n\\tmov.b32 perm, %%%u;\\n\\tadd.u32 salted, perm, %u;\\n\\tand.b32 "
                        "masked, salted, 1;\\n\\tsetp.ne.u32 pred, masked, 0;\\n\\t\"\n",
                        perm, k + 1));
    TRY(o_writer_format(w, "\"selp.f32 %%%u, %%0, %%1, pred;\\n\\tmov.b32 %%%u, %%%u;\\n\\t\"\n", in, keep,
                        in));
    for (i = 0; i < out; ++i)
        TRY(o_writer_format(w, "\"add.rn.ftz.f32 %%%u, %%%u, 0f%08x;\\n\\t\"\n", in + i, i == 0 ? in : 0,
                            marker + 128 + i));
    for (i = 0; i < out; ++i)
        TRY(o_writer_format(w, "\"add.rn.ftz.f32 %%%u, %%%u, %%%u;\\n\\t\"\n", keep, keep, in + i));
    for (i = 0; i < in; ++i)
        TRY(o_writer_format(w, "\"add.rn.ftz.f32 %%%u, %%%u, %%%u;\\n\\t\"\n", keep, keep, i));
    TRY(o_writer_format(w,
                        "\"add.rn.ftz.f32 %%%u, %%%u, %%%u;\\n\\tmov.u32 addr, "
                        "0;\\n\\tst.volatile.shared.f32 [addr], %%%u;\\n\\t\"\n",
                        keep, keep, perm, keep));
    for (i = 0; i < shape->site_patch_capacity; ++i)
        TRY(o_writer_cstr(w, "\"brkpt;\\n\\t\"\n"));
    TRY(o_writer_cstr(w, "\"}\\n\\t\"\n: "));
    for (i = 0; i < in; ++i) {
        if (i)
            TRY(o_writer_cstr(w, ", "));
        TRY(o_writer_cstr(w, "\"+f\"("));
        TRY(input_name(w, i, shape->state_count));
        TRY(o_writer_cstr(w, ")"));
    }
    for (i = 0; i < out; ++i)
        TRY(o_writer_format(w, ", \"=&f\"(lm_r_%u_%u)", k, i));
    return o_writer_format(w, ", \"=&f\"(lm_keepalive_%u) : \"f\"(lm_permutation_input) : \"memory\");\n", k);
}
static OdezzaResult stage_source(OWriter *w, const OLmInspection *p) {
    uint32_t k = 0, i, r, n = p->shape.state_count;
    TRY(o_writer_cstr(w, "float rhs[ODEZZA_LM_STATE_COUNT];\n"));
    while (k < p->site_count && p->sites[k].output_indices[0] < n) {
        TRY(site_source(w, &p->sites[k], k, &p->shape));
        for (i = 0; i < p->sites[k].output_count; ++i)
            TRY(o_writer_format(w, "rhs[%u] = lm_r_%u_%u;\n", p->sites[k].output_indices[i], k, i));
        ++k;
    }
    if (p->shape.lanes_per_fit > 1u)
        TRY(o_writer_cstr(w,
            "#pragma unroll\nfor(int i=0;i<ODEZZA_LM_STATE_COUNT;++i) "
            "rhs[i]=__shfl_sync(lm_fit_mask,rhs[i],0,ODEZZA_LM_FIT_THREAD_COUNT);\n"));
    TRY(o_writer_cstr(
        w, "const float weight = (rk_stage == 0 || rk_stage == 3) ? 1.0f : 2.0f;\n#pragma unroll\nfor (int "
           "component=0; component<ODEZZA_LM_STATE_COUNT; ++component) sum_state[component] = "
           "fmaf(weight,rhs[component],sum_state[component]);\nif (statistics_phase) {\n"));
    for (r = 0; r < n; ++r) {
        TRY(o_writer_format(w, "float deriv_%u[ODEZZA_LM_LOCAL_PARAMETER_COUNT] = {};\n", r));
        while (k < p->site_count && (p->sites[k].output_indices[0] - n) % n == r) {
            TRY(site_source(w, &p->sites[k], k, &p->shape));
            TRY(o_writer_cstr(
                w, "#pragma unroll\nfor(int lp=0;lp<ODEZZA_LM_LOCAL_PARAMETER_COUNT;++lp) {\nconst int "
                   "parameter=(int)lm_fit_lane+lp*ODEZZA_LM_FIT_THREAD_COUNT;\nif(parameter>=ODEZZA_LM_"
                   "PARAMETER_COUNT)continue;\n"));
            for (i = 0; i < p->sites[k].output_count; ++i) {
                uint32_t input = (p->sites[k].output_indices[i] - n) / n;
                if (input < n)
                    TRY(o_writer_format(w,
                                        "deriv_%u[lp]=fmaf(lm_r_%u_%u,stage_sensitivity[%u*ODEZZA_LM_LOCAL_"
                                        "PARAMETER_COUNT+lp],deriv_%u[lp]);\n",
                                        r, k, i, input, r));
                else
                    TRY(o_writer_format(w, "if(parameter==%u)deriv_%u[lp]+=lm_r_%u_%u;\n", input - n, r, k,
                                        i));
            }
            TRY(o_writer_cstr(w, "}\n"));
            ++k;
        }
    }
    /* All RHS derivatives must observe the same RK stage. Updating a row
     * while later derivatives still read it changes the coupled Jacobian. */
    for (r = 0; r < n; ++r) {
        TRY(o_writer_format(
            w,
            "#pragma unroll\nfor(int lp=0;lp<ODEZZA_LM_LOCAL_PARAMETER_COUNT;++lp) {\nconst int "
            "si=%u*ODEZZA_LM_LOCAL_PARAMETER_COUNT+lp;\nsum_sensitivity[si]=fmaf(weight,deriv_%u[lp],sum_"
            "sensitivity[si]);\nif(rk_stage!=3){const float "
            "stage_h=rk_stage==2?h:half_h;stage_sensitivity[si]=fmaf(stage_h,deriv_%u[lp],sensitivities[si]);"
            "}\n}\n",
            r, r, r));
    }
    return o_writer_cstr(w, "}\n");
}
OdezzaResult o_lm_generate_cuda(const OdezzaLmShape *s, char *buffer, size_t capacity, size_t *size) {
    OLmInspection p;
    OWriter w;
    uint32_t g;
    if (!size)
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    memset(&p, 0, sizeof(p));
    TRY(o_lm_site_layout(s, &p));
    g = s->parameter_count * (s->parameter_count + 1u) / 2u;
    TRY(o_writer_init(&w, buffer, capacity, NULL));
    TRY(o_writer_format(
        &w,
        "#define ODEZZA_LM_STATE_COUNT %u\n#define ODEZZA_LM_PARAMETER_COUNT %u\n#define "
        "ODEZZA_LM_INPUT_COUNT %u\n#define ODEZZA_LM_GRAM_COUNT %u\n#define ODEZZA_LM_SITE_COUNT %u\n#define "
        "ODEZZA_LM_FIT_THREAD_COUNT %u\n#define ODEZZA_LM_LOCAL_PARAMETER_COUNT %u\n#define "
        "ODEZZA_LM_LOCAL_OPTIMIZER_COUNT %u\n#define ODEZZA_LM_FITS_PER_WARP %u\n"
        "#define ODEZZA_LM_LOCAL_FACTOR_COUNT %u\n#define ODEZZA_LM_LOCAL_SOLVE_COUNT %u\n",
        s->state_count, s->parameter_count, s->state_count + s->parameter_count, g, p.site_count,
        s->lanes_per_fit, (s->parameter_count + s->lanes_per_fit - 1u) / s->lanes_per_fit,
        (s->parameter_count + g + s->lanes_per_fit - 1u) / s->lanes_per_fit,
        32u / s->lanes_per_fit, (g + s->lanes_per_fit - 1u) / s->lanes_per_fit,
        (s->parameter_count + s->lanes_per_fit - 1u) / s->lanes_per_fit));
    TRY(o_lm_write_body_0(&w));
    TRY(stage_source(&w, &p));
    TRY(o_lm_write_body_1(&w));
    TRY(o_writer_write(&w, "", 1));
    *size = w.bytes_written;
    return ODEZZA_SUCCESS;
}
