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
#include <kermac_cpu.h>
#include <k_internal.h>

#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

enum {
    REL_LABEL_DX = 0,
    REL_LABEL_DY,
    REL_LABEL_DIST,
    REL_LABEL_LR,
    REL_LABEL_UD,
    REL_LABEL_QUAD,
    REL_LABEL_ANGLE8,
    REL_LABEL_DIST_BIN,
    REL_LABEL_RA,
    REL_LABEL_RB,
    REL_LABEL_XA,
    REL_LABEL_YA,
    REL_LABEL_XB,
    REL_LABEL_YB,
    REL_LABEL_COUNT
};

static uint32_t g_rng = 1u;

static uint32_t xorshift32(void) {
    uint32_t x = g_rng;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    g_rng = x;
    return x;
}

static int irand(int lo, int hi) {
    if (hi < lo) {
        int t = lo;
        lo = hi;
        hi = t;
    }
    uint32_t span = (uint32_t)(hi - lo + 1);
    uint32_t r = xorshift32();
    return lo + (int)(r % span);
}

static float frand01(void) {
    uint32_t r = xorshift32();
    return (float)r / 4294967296.0f;
}

static int clampi(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static float clampf(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static uint8_t clampu8(int v) {
    if (v < 0) return 0;
    if (v > 255) return 255;
    return (uint8_t)v;
}

static uint8_t f32_to_u8(float v) {
    float clamped = clampf(v, 0.0f, 1.0f);
    int iv = (int)floorf(clamped * 255.0f + 0.5f);
    return clampu8(iv);
}

static uint32_t pack_rgba(uint8_t r, uint8_t g, uint8_t b) {
    return ((uint32_t)r) | ((uint32_t)g << 8) | ((uint32_t)b << 16) | (0xFFu << 24);
}

static void set_px(
    float* sample,
    int64_t ld,
    int w,
    int h,
    int c,
    int x,
    int y,
    const float col[3]
) {
    if ((unsigned)x >= (unsigned)w || (unsigned)y >= (unsigned)h) {
        return;
    }
    int64_t plane = (int64_t)w * (int64_t)h;
    int64_t idx = (int64_t)y * (int64_t)w + (int64_t)x;
    if (c == 1) {
        sample[idx * ld] = col[0];
    } else {
        sample[(0 * plane + idx) * ld] = col[0];
        sample[(1 * plane + idx) * ld] = col[1];
        sample[(2 * plane + idx) * ld] = col[2];
    }
}

static void set_px_u32(
    uint32_t* sample,
    int64_t ld,
    int w,
    int h,
    int x,
    int y,
    const uint8_t col[3]
) {
    if ((unsigned)x >= (unsigned)w || (unsigned)y >= (unsigned)h) {
        return;
    }
    int64_t idx = (int64_t)y * (int64_t)w + (int64_t)x;
    sample[idx * ld] = pack_rgba(col[0], col[1], col[2]);
}

static void set_px_u8(
    uint8_t* sample,
    int64_t ld,
    int w,
    int h,
    int x,
    int y,
    uint8_t v
) {
    if ((unsigned)x >= (unsigned)w || (unsigned)y >= (unsigned)h) {
        return;
    }
    int64_t idx = (int64_t)y * (int64_t)w + (int64_t)x;
    sample[idx * ld] = v;
}

static void draw_circle(
    float* sample,
    int64_t ld,
    int w,
    int h,
    int c,
    int cx,
    int cy,
    int r,
    const float col[3]
) {
    int r2 = r * r;
    int y0 = cy - r;
    int y1 = cy + r;
    for (int y = y0; y <= y1; y++) {
        int dy = y - cy;
        int dy2 = dy * dy;
        int x0 = cx - r;
        int x1 = cx + r;
        for (int x = x0; x <= x1; x++) {
            int dx = x - cx;
            if (dx * dx + dy2 <= r2) {
                set_px(sample, ld, w, h, c, x, y, col);
            }
        }
    }
}

static void draw_circle_u32(
    uint32_t* sample,
    int64_t ld,
    int w,
    int h,
    int cx,
    int cy,
    int r,
    const uint8_t col[3]
) {
    int r2 = r * r;
    int y0 = cy - r;
    int y1 = cy + r;
    for (int y = y0; y <= y1; y++) {
        int dy = y - cy;
        int dy2 = dy * dy;
        int x0 = cx - r;
        int x1 = cx + r;
        for (int x = x0; x <= x1; x++) {
            int dx = x - cx;
            if (dx * dx + dy2 <= r2) {
                set_px_u32(sample, ld, w, h, x, y, col);
            }
        }
    }
}

static void draw_circle_u8(
    uint8_t* sample,
    int64_t ld,
    int w,
    int h,
    int cx,
    int cy,
    int r,
    uint8_t v
) {
    int r2 = r * r;
    int y0 = cy - r;
    int y1 = cy + r;
    for (int y = y0; y <= y1; y++) {
        int dy = y - cy;
        int dy2 = dy * dy;
        int x0 = cx - r;
        int x1 = cx + r;
        for (int x = x0; x <= x1; x++) {
            int dx = x - cx;
            if (dx * dx + dy2 <= r2) {
                set_px_u8(sample, ld, w, h, x, y, v);
            }
        }
    }
}

static void draw_square(
    float* sample,
    int64_t ld,
    int w,
    int h,
    int c,
    int cx,
    int cy,
    int half,
    const float col[3]
) {
    int x0 = cx - half;
    int x1 = cx + half;
    int y0 = cy - half;
    int y1 = cy + half;
    for (int y = y0; y <= y1; y++) {
        for (int x = x0; x <= x1; x++) {
            set_px(sample, ld, w, h, c, x, y, col);
        }
    }
}

static void draw_square_u32(
    uint32_t* sample,
    int64_t ld,
    int w,
    int h,
    int cx,
    int cy,
    int half,
    const uint8_t col[3]
) {
    int x0 = cx - half;
    int x1 = cx + half;
    int y0 = cy - half;
    int y1 = cy + half;
    for (int y = y0; y <= y1; y++) {
        for (int x = x0; x <= x1; x++) {
            set_px_u32(sample, ld, w, h, x, y, col);
        }
    }
}

static void draw_square_u8(
    uint8_t* sample,
    int64_t ld,
    int w,
    int h,
    int cx,
    int cy,
    int half,
    uint8_t v
) {
    int x0 = cx - half;
    int x1 = cx + half;
    int y0 = cy - half;
    int y1 = cy + half;
    for (int y = y0; y <= y1; y++) {
        for (int x = x0; x <= x1; x++) {
            set_px_u8(sample, ld, w, h, x, y, v);
        }
    }
}

static void draw_line(
    float* sample,
    int64_t ld,
    int w,
    int h,
    int c,
    int x0,
    int y0,
    int x1,
    int y1,
    const float col[3]
) {
    int dx = abs(x1 - x0);
    int sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0);
    int sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    while (1) {
        set_px(sample, ld, w, h, c, x0, y0, col);
        if (x0 == x1 && y0 == y1) break;
        int e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

static void draw_line_u32(
    uint32_t* sample,
    int64_t ld,
    int w,
    int h,
    int x0,
    int y0,
    int x1,
    int y1,
    const uint8_t col[3]
) {
    int dx = abs(x1 - x0);
    int sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0);
    int sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    while (1) {
        set_px_u32(sample, ld, w, h, x0, y0, col);
        if (x0 == x1 && y0 == y1) break;
        int e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

static void draw_line_u8(
    uint8_t* sample,
    int64_t ld,
    int w,
    int h,
    int x0,
    int y0,
    int x1,
    int y1,
    uint8_t v
) {
    int dx = abs(x1 - x0);
    int sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0);
    int sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    while (1) {
        set_px_u8(sample, ld, w, h, x0, y0, v);
        if (x0 == x1 && y0 == y1) break;
        int e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

static void add_noise(
    float* sample,
    int64_t ld,
    int w,
    int h,
    int c,
    float noise_amp01
) {
    if (noise_amp01 <= 0.0f) {
        return;
    }
    int64_t plane = (int64_t)w * (int64_t)h;
    int64_t dims = plane * (int64_t)c;
    for (int64_t d = 0; d < dims; d++) {
        float v = sample[d * ld];
        float n0 = (frand01() * 2.0f - 1.0f) * noise_amp01;
        sample[d * ld] = clampf(v + n0, 0.0f, 1.0f);
    }
}

static void add_noise_u32(
    uint32_t* sample,
    int64_t ld,
    int w,
    int h,
    int channels,
    float noise_amp01
) {
    if (noise_amp01 <= 0.0f) {
        return;
    }
    int64_t plane = (int64_t)w * (int64_t)h;
    for (int64_t d = 0; d < plane; d++) {
        uint32_t pixel = sample[d * ld];
        int r = (int)(pixel & 0xFFu);
        int g = (int)((pixel >> 8) & 0xFFu);
        int b = (int)((pixel >> 16) & 0xFFu);
        if (channels == 1) {
            float v = (float)r / 255.0f;
            float n0 = (frand01() * 2.0f - 1.0f) * noise_amp01;
            uint8_t out = f32_to_u8(v + n0);
            r = out;
            g = out;
            b = out;
        } else {
            float v = (float)r / 255.0f;
            float n0 = (frand01() * 2.0f - 1.0f) * noise_amp01;
            r = (int)f32_to_u8(v + n0);

            v = (float)g / 255.0f;
            n0 = (frand01() * 2.0f - 1.0f) * noise_amp01;
            g = (int)f32_to_u8(v + n0);

            v = (float)b / 255.0f;
            n0 = (frand01() * 2.0f - 1.0f) * noise_amp01;
            b = (int)f32_to_u8(v + n0);
        }
        sample[d * ld] = pack_rgba((uint8_t)r, (uint8_t)g, (uint8_t)b);
    }
}

static void add_noise_u8(
    uint8_t* sample,
    int64_t ld,
    int w,
    int h,
    float noise_amp01
) {
    if (noise_amp01 <= 0.0f) {
        return;
    }
    int64_t plane = (int64_t)w * (int64_t)h;
    for (int64_t d = 0; d < plane; d++) {
        float v = (float)sample[d * ld] / 255.0f;
        float n0 = (frand01() * 2.0f - 1.0f) * noise_amp01;
        sample[d * ld] = f32_to_u8(v + n0);
    }
}

static void threshold_binary(
    float* sample,
    int64_t ld,
    int w,
    int h,
    int c,
    float thr
) {
    int64_t plane = (int64_t)w * (int64_t)h;
    int64_t dims = plane * (int64_t)c;
    for (int64_t d = 0; d < dims; d++) {
        sample[d * ld] = (sample[d * ld] > thr) ? 1.0f : 0.0f;
    }
}

static void threshold_binary_u32(
    uint32_t* sample,
    int64_t ld,
    int w,
    int h,
    int channels,
    uint8_t thr
) {
    int64_t plane = (int64_t)w * (int64_t)h;
    for (int64_t d = 0; d < plane; d++) {
        uint32_t pixel = sample[d * ld];
        int r = (int)(pixel & 0xFFu);
        int g = (int)((pixel >> 8) & 0xFFu);
        int b = (int)((pixel >> 16) & 0xFFu);
        if (channels == 1) {
            uint8_t out = (r > thr) ? 255u : 0u;
            r = out;
            g = out;
            b = out;
        } else {
            r = (r > thr) ? 255 : 0;
            g = (g > thr) ? 255 : 0;
            b = (b > thr) ? 255 : 0;
        }
        sample[d * ld] = pack_rgba((uint8_t)r, (uint8_t)g, (uint8_t)b);
    }
}

static void threshold_binary_u8(
    uint8_t* sample,
    int64_t ld,
    int w,
    int h,
    uint8_t thr
) {
    int64_t plane = (int64_t)w * (int64_t)h;
    for (int64_t d = 0; d < plane; d++) {
        sample[d * ld] = (sample[d * ld] > thr) ? 255u : 0u;
    }
}

static void pick_key_colors(KermacRelSpritesMode mode, float col_circle[3], float col_square[3]) {
    if (mode == KERMAC_RELSPRITES_MODE_COLOR) {
        col_circle[0] = 1.0f; col_circle[1] = 0.0f; col_circle[2] = 0.0f;
        col_square[0] = 0.0f; col_square[1] = 1.0f; col_square[2] = 0.0f;
    } else if (mode == KERMAC_RELSPRITES_MODE_GRAY) {
        col_circle[0] = 220.0f / 255.0f;
        col_circle[1] = 220.0f / 255.0f;
        col_circle[2] = 220.0f / 255.0f;
        col_square[0] = 60.0f / 255.0f;
        col_square[1] = 60.0f / 255.0f;
        col_square[2] = 60.0f / 255.0f;
    } else {
        col_circle[0] = 1.0f; col_circle[1] = 1.0f; col_circle[2] = 1.0f;
        col_square[0] = 1.0f; col_square[1] = 1.0f; col_square[2] = 1.0f;
    }
}

static void pick_key_colors_u8(KermacRelSpritesMode mode, uint8_t col_circle[3], uint8_t col_square[3]) {
    if (mode == KERMAC_RELSPRITES_MODE_COLOR) {
        col_circle[0] = 255; col_circle[1] = 0;   col_circle[2] = 0;
        col_square[0] = 0;   col_square[1] = 255; col_square[2] = 0;
    } else if (mode == KERMAC_RELSPRITES_MODE_GRAY) {
        col_circle[0] = 220; col_circle[1] = 220; col_circle[2] = 220;
        col_square[0] = 60;  col_square[1] = 60;  col_square[2] = 60;
    } else {
        col_circle[0] = 255; col_circle[1] = 255; col_circle[2] = 255;
        col_square[0] = 255; col_square[1] = 255; col_square[2] = 255;
    }
}

static void random_distractor_color(KermacRelSpritesMode mode, float col[3]) {
    if (mode == KERMAC_RELSPRITES_MODE_COLOR) {
        col[0] = frand01();
        col[1] = frand01();
        col[2] = frand01();
        if (col[0] > 0.94f && col[1] < 0.08f && col[2] < 0.08f) col[2] = 0.31f;
        if (col[1] > 0.94f && col[0] < 0.08f && col[2] < 0.08f) col[2] = 0.31f;
    } else if (mode == KERMAC_RELSPRITES_MODE_GRAY) {
        float v = (float)irand(30, 225) / 255.0f;
        col[0] = v; col[1] = v; col[2] = v;
    } else {
        col[0] = 1.0f; col[1] = 1.0f; col[2] = 1.0f;
    }
}

static void random_distractor_color_u8(KermacRelSpritesMode mode, uint8_t col[3]) {
    if (mode == KERMAC_RELSPRITES_MODE_COLOR) {
        float r = frand01();
        float g = frand01();
        float b = frand01();
        if (r > 0.94f && g < 0.08f && b < 0.08f) b = 0.31f;
        if (g > 0.94f && r < 0.08f && b < 0.08f) b = 0.31f;
        col[0] = f32_to_u8(r);
        col[1] = f32_to_u8(g);
        col[2] = f32_to_u8(b);
    } else if (mode == KERMAC_RELSPRITES_MODE_GRAY) {
        float v = (float)irand(30, 225) / 255.0f;
        uint8_t vv = f32_to_u8(v);
        col[0] = vv; col[1] = vv; col[2] = vv;
    } else {
        col[0] = 255; col[1] = 255; col[2] = 255;
    }
}

static int far_enough(int x, int y, int xA, int yA, int xB, int yB, int min_d2_A, int min_d2_B) {
    int dxA = x - xA; int dyA = y - yA;
    int dxB = x - xB; int dyB = y - yB;
    int d2A = dxA * dxA + dyA * dyA;
    int d2B = dxB * dxB + dyB * dyB;
    return (d2A >= min_d2_A) && (d2B >= min_d2_B);
}

static void compute_labels(
    int w, int h, int xA, int yA, int xB, int yB,
    int* out_dx, int* out_dy, float* out_dist,
    int* out_lr, int* out_ud, int* out_quad,
    int* out_angle8, int* out_dist_bin
) {
    int dx = xB - xA;
    int dy = yB - yA;
    float dist = sqrtf((float)(dx * dx + dy * dy));

    int lr = (dx > 0) ? 1 : 0;
    int ud = (dy > 0) ? 1 : 0;
    int quad = lr + 2 * ud;

    float ang = atan2f(-(float)dy, (float)dx);
    if (ang < 0.0f) ang += 2.0f * (float)M_PI;
    float binw = (2.0f * (float)M_PI) / 8.0f;
    int angle8 = (int)floorf((ang + 0.5f * binw) / binw) & 7;

    int mindim = (w < h) ? w : h;
    float t0 = 0.25f * (float)mindim;
    float t1 = 0.50f * (float)mindim;
    int dist_bin = (dist < t0) ? 0 : ((dist < t1) ? 1 : 2);

    *out_dx = dx;
    *out_dy = dy;
    *out_dist = dist;
    *out_lr = lr;
    *out_ud = ud;
    *out_quad = quad;
    *out_angle8 = angle8;
    *out_dist_bin = dist_bin;
}

KERMAC_PUBLIC_DEF
KermacResult
kermac_relsprites_dims(
    const KermacRelSpritesConfig* cfg,
    int64_t* num_samples,
    int64_t* num_rows,
    int64_t* num_cols,
    int64_t* num_channels,
    int64_t* num_labels
) {
    if (!cfg) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    int64_t channels = (cfg->mode == KERMAC_RELSPRITES_MODE_COLOR) ? 3 : 1;
    if (num_samples) *num_samples = cfg->n;
    if (num_rows) *num_rows = cfg->h;
    if (num_cols) *num_cols = cfg->w;
    if (num_channels) *num_channels = channels;
    if (num_labels) *num_labels = REL_LABEL_COUNT;
    return KERMAC_SUCCESS;
}

KERMAC_PUBLIC_DEF
KermacResult
kermac_relsprites_generate(
    const KermacRelSpritesConfig* cfg,
    KermacTensor images,
    KermacTensor labels
) {
    if (!cfg) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (cfg->w < 8 || cfg->h < 8 || cfg->n <= 0) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (cfg->distractors < 0) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (cfg->noise < 0.0f || cfg->noise > 1.0f) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (images.data_type != KERMAC_DATA_TYPE_FLOAT || labels.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (images.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_HOST ||
        labels.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_HOST) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    int channels = (cfg->mode == KERMAC_RELSPRITES_MODE_COLOR) ? 3 : 1;
    int64_t plane = (int64_t)cfg->w * (int64_t)cfg->h;
    int64_t num_dims = plane * (int64_t)channels;

    if (images.extent[0] < cfg->n || images.extent[1] < num_dims) {
        _KERMAC_ERROR( KERMAC_ERROR_SIZE_MISMATCH );
    }
    if (labels.extent[0] < cfg->n || labels.extent[1] < REL_LABEL_COUNT) {
        _KERMAC_ERROR( KERMAC_ERROR_SIZE_MISMATCH );
    }

    float* img_ptr = NULL;
    float* label_ptr = NULL;
    _KERMAC_CHECK_RET( kermac_memory_pointer(images.memory, (void**)&img_ptr) );
    _KERMAC_CHECK_RET( kermac_memory_pointer(labels.memory, (void**)&label_ptr) );

    int64_t ld_img = images.stride[1];
    int64_t ld_lbl = labels.stride[1];

    g_rng = (cfg->seed == 0u) ? 1u : cfg->seed;

    int mindim = (cfg->w < cfg->h) ? cfg->w : cfg->h;
    int base = mindim / 8;
    if (base < 2) base = 2;

    for (int i = 0; i < cfg->n; i++) {
        float* sample = img_ptr + i;
        for (int64_t d = 0; d < num_dims; d++) {
            sample[d * ld_img] = 0.0f;
        }

        int rA = base + irand(-1, 1);
        int rB = base + irand(-1, 1);
        if (rA < 1) rA = 1;
        if (rB < 1) rB = 1;

        int margin = (rA > rB ? rA : rB) + 1;
        int xA = irand(margin, cfg->w - 1 - margin);
        int yA = irand(margin, cfg->h - 1 - margin);

        int xB = 0, yB = 0;
        int tries = 0;
        int min_d = rA + rB + 2;
        int min_d2 = min_d * min_d;
        do {
            xB = irand(margin, cfg->w - 1 - margin);
            yB = irand(margin, cfg->h - 1 - margin);
            int dx = xB - xA;
            int dy = yB - yA;
            if (dx * dx + dy * dy >= min_d2) break;
            tries++;
        } while (tries < 2000);

        float col_circle[3], col_square[3];
        pick_key_colors(cfg->mode, col_circle, col_square);
        draw_circle(sample, ld_img, cfg->w, cfg->h, channels, xA, yA, rA, col_circle);
        draw_square(sample, ld_img, cfg->w, cfg->h, channels, xB, yB, rB, col_square);

        for (int d = 0; d < cfg->distractors; d++) {
            float col[3];
            random_distractor_color(cfg->mode, col);

            int t = irand(0, 2);
            if (t == 0 || t == 1) {
                int rr = clampi(base - 1 + irand(-1, 1), 1, base + 1);
                int m2 = rr + 1;
                int x = 0, y = 0;
                int min_dA = (rA + rr + 1);
                int min_dB = (rB + rr + 1);
                int min_d2_A = min_dA * min_dA;
                int min_d2_B = min_dB * min_dB;
                int ok = 0;
                for (int k = 0; k < 200; k++) {
                    x = irand(m2, cfg->w - 1 - m2);
                    y = irand(m2, cfg->h - 1 - m2);
                    if (far_enough(x, y, xA, yA, xB, yB, min_d2_A, min_d2_B)) { ok = 1; break; }
                }
                if (!ok) {
                    x = irand(m2, cfg->w - 1 - m2);
                    y = irand(m2, cfg->h - 1 - m2);
                }
                if (t == 0) {
                    draw_circle(sample, ld_img, cfg->w, cfg->h, channels, x, y, rr, col);
                } else {
                    draw_square(sample, ld_img, cfg->w, cfg->h, channels, x, y, rr, col);
                }
            } else {
                int x0 = irand(0, cfg->w - 1);
                int y0 = irand(0, cfg->h - 1);
                int x1 = irand(0, cfg->w - 1);
                int y1 = irand(0, cfg->h - 1);
                draw_line(sample, ld_img, cfg->w, cfg->h, channels, x0, y0, x1, y1, col);
            }
        }

        add_noise(sample, ld_img, cfg->w, cfg->h, channels, cfg->noise);

        if (cfg->mode == KERMAC_RELSPRITES_MODE_BINARY) {
            threshold_binary(sample, ld_img, cfg->w, cfg->h, channels, 0.5f);
        }

        int dx, dy, lr, ud, quad, angle8, dist_bin;
        float dist;
        compute_labels(cfg->w, cfg->h, xA, yA, xB, yB,
                       &dx, &dy, &dist,
                       &lr, &ud, &quad, &angle8, &dist_bin);

        float* lbl = label_ptr + i;
        lbl[REL_LABEL_DX * ld_lbl] = (float)dx;
        lbl[REL_LABEL_DY * ld_lbl] = (float)dy;
        lbl[REL_LABEL_DIST * ld_lbl] = dist;
        lbl[REL_LABEL_LR * ld_lbl] = (float)lr;
        lbl[REL_LABEL_UD * ld_lbl] = (float)ud;
        lbl[REL_LABEL_QUAD * ld_lbl] = (float)quad;
        lbl[REL_LABEL_ANGLE8 * ld_lbl] = (float)angle8;
        lbl[REL_LABEL_DIST_BIN * ld_lbl] = (float)dist_bin;
        lbl[REL_LABEL_RA * ld_lbl] = (float)rA;
        lbl[REL_LABEL_RB * ld_lbl] = (float)rB;
        lbl[REL_LABEL_XA * ld_lbl] = (float)xA;
        lbl[REL_LABEL_YA * ld_lbl] = (float)yA;
        lbl[REL_LABEL_XB * ld_lbl] = (float)xB;
        lbl[REL_LABEL_YB * ld_lbl] = (float)yB;
    }

    return KERMAC_SUCCESS;
}

KERMAC_PUBLIC_DEF
KermacResult
kermac_relsprites_generate_u32(
    const KermacRelSpritesConfig* cfg,
    KermacTensor images,
    KermacTensor labels
) {
    if (!cfg) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (cfg->w < 8 || cfg->h < 8 || cfg->n <= 0) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (cfg->distractors < 0) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (cfg->noise < 0.0f || cfg->noise > 1.0f) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (images.data_type != KERMAC_DATA_TYPE_UINT || labels.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (images.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_HOST ||
        labels.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_HOST) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    int channels = (cfg->mode == KERMAC_RELSPRITES_MODE_COLOR) ? 3 : 1;
    int64_t plane = (int64_t)cfg->w * (int64_t)cfg->h;

    if (images.extent[0] < cfg->n || images.extent[1] < plane) {
        _KERMAC_ERROR( KERMAC_ERROR_SIZE_MISMATCH );
    }
    if (labels.extent[0] < cfg->n || labels.extent[1] < REL_LABEL_COUNT) {
        _KERMAC_ERROR( KERMAC_ERROR_SIZE_MISMATCH );
    }

    uint32_t* img_ptr = NULL;
    float* label_ptr = NULL;
    _KERMAC_CHECK_RET( kermac_memory_pointer(images.memory, (void**)&img_ptr) );
    _KERMAC_CHECK_RET( kermac_memory_pointer(labels.memory, (void**)&label_ptr) );

    int64_t ld_img = images.stride[1];
    int64_t ld_lbl = labels.stride[1];

    g_rng = (cfg->seed == 0u) ? 1u : cfg->seed;

    int mindim = (cfg->w < cfg->h) ? cfg->w : cfg->h;
    int base = mindim / 8;
    if (base < 2) base = 2;

    uint32_t clear = pack_rgba(0, 0, 0);

    for (int i = 0; i < cfg->n; i++) {
        uint32_t* sample = img_ptr + i;
        for (int64_t d = 0; d < plane; d++) {
            sample[d * ld_img] = clear;
        }

        int rA = base + irand(-1, 1);
        int rB = base + irand(-1, 1);
        if (rA < 1) rA = 1;
        if (rB < 1) rB = 1;

        int margin = (rA > rB ? rA : rB) + 1;
        int xA = irand(margin, cfg->w - 1 - margin);
        int yA = irand(margin, cfg->h - 1 - margin);

        int xB = 0, yB = 0;
        int tries = 0;
        int min_d = rA + rB + 2;
        int min_d2 = min_d * min_d;
        do {
            xB = irand(margin, cfg->w - 1 - margin);
            yB = irand(margin, cfg->h - 1 - margin);
            int dx = xB - xA;
            int dy = yB - yA;
            if (dx * dx + dy * dy >= min_d2) break;
            tries++;
        } while (tries < 2000);

        uint8_t col_circle[3], col_square[3];
        pick_key_colors_u8(cfg->mode, col_circle, col_square);
        draw_circle_u32(sample, ld_img, cfg->w, cfg->h, xA, yA, rA, col_circle);
        draw_square_u32(sample, ld_img, cfg->w, cfg->h, xB, yB, rB, col_square);

        for (int d = 0; d < cfg->distractors; d++) {
            uint8_t col[3];
            random_distractor_color_u8(cfg->mode, col);

            int t = irand(0, 2);
            if (t == 0 || t == 1) {
                int rr = clampi(base - 1 + irand(-1, 1), 1, base + 1);
                int m2 = rr + 1;
                int x = 0, y = 0;
                int min_dA = (rA + rr + 1);
                int min_dB = (rB + rr + 1);
                int min_d2_A = min_dA * min_dA;
                int min_d2_B = min_dB * min_dB;
                int ok = 0;
                for (int k = 0; k < 200; k++) {
                    x = irand(m2, cfg->w - 1 - m2);
                    y = irand(m2, cfg->h - 1 - m2);
                    if (far_enough(x, y, xA, yA, xB, yB, min_d2_A, min_d2_B)) { ok = 1; break; }
                }
                if (!ok) {
                    x = irand(m2, cfg->w - 1 - m2);
                    y = irand(m2, cfg->h - 1 - m2);
                }
                if (t == 0) {
                    draw_circle_u32(sample, ld_img, cfg->w, cfg->h, x, y, rr, col);
                } else {
                    draw_square_u32(sample, ld_img, cfg->w, cfg->h, x, y, rr, col);
                }
            } else {
                int x0 = irand(0, cfg->w - 1);
                int y0 = irand(0, cfg->h - 1);
                int x1 = irand(0, cfg->w - 1);
                int y1 = irand(0, cfg->h - 1);
                draw_line_u32(sample, ld_img, cfg->w, cfg->h, x0, y0, x1, y1, col);
            }
        }

        add_noise_u32(sample, ld_img, cfg->w, cfg->h, channels, cfg->noise);

        if (cfg->mode == KERMAC_RELSPRITES_MODE_BINARY) {
            threshold_binary_u32(sample, ld_img, cfg->w, cfg->h, channels, 127);
        }

        int dx, dy, lr, ud, quad, angle8, dist_bin;
        float dist;
        compute_labels(cfg->w, cfg->h, xA, yA, xB, yB,
                       &dx, &dy, &dist,
                       &lr, &ud, &quad, &angle8, &dist_bin);

        float* lbl = label_ptr + i;
        lbl[REL_LABEL_DX * ld_lbl] = (float)dx;
        lbl[REL_LABEL_DY * ld_lbl] = (float)dy;
        lbl[REL_LABEL_DIST * ld_lbl] = dist;
        lbl[REL_LABEL_LR * ld_lbl] = (float)lr;
        lbl[REL_LABEL_UD * ld_lbl] = (float)ud;
        lbl[REL_LABEL_QUAD * ld_lbl] = (float)quad;
        lbl[REL_LABEL_ANGLE8 * ld_lbl] = (float)angle8;
        lbl[REL_LABEL_DIST_BIN * ld_lbl] = (float)dist_bin;
        lbl[REL_LABEL_RA * ld_lbl] = (float)rA;
        lbl[REL_LABEL_RB * ld_lbl] = (float)rB;
        lbl[REL_LABEL_XA * ld_lbl] = (float)xA;
        lbl[REL_LABEL_YA * ld_lbl] = (float)yA;
        lbl[REL_LABEL_XB * ld_lbl] = (float)xB;
        lbl[REL_LABEL_YB * ld_lbl] = (float)yB;
    }

    return KERMAC_SUCCESS;
}

KERMAC_PUBLIC_DEF
KermacResult
kermac_relsprites_generate_u8(
    const KermacRelSpritesConfig* cfg,
    KermacTensor images,
    KermacTensor labels
) {
    if (!cfg) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (cfg->w < 8 || cfg->h < 8 || cfg->n <= 0) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (cfg->distractors < 0) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (cfg->noise < 0.0f || cfg->noise > 1.0f) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (cfg->mode == KERMAC_RELSPRITES_MODE_COLOR) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    if (images.data_type != KERMAC_DATA_TYPE_BYTE || labels.data_type != KERMAC_DATA_TYPE_FLOAT) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_DATA_TYPE );
    }
    if (images.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_HOST ||
        labels.memory.stack_allocator->memory_space != KERMAC_MEMORY_SPACE_HOST) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    int64_t plane = (int64_t)cfg->w * (int64_t)cfg->h;
    if (images.extent[0] < cfg->n || images.extent[1] < plane) {
        _KERMAC_ERROR( KERMAC_ERROR_SIZE_MISMATCH );
    }
    if (labels.extent[0] < cfg->n || labels.extent[1] < REL_LABEL_COUNT) {
        _KERMAC_ERROR( KERMAC_ERROR_SIZE_MISMATCH );
    }

    uint8_t* img_ptr = NULL;
    float* label_ptr = NULL;
    _KERMAC_CHECK_RET( kermac_memory_pointer(images.memory, (void**)&img_ptr) );
    _KERMAC_CHECK_RET( kermac_memory_pointer(labels.memory, (void**)&label_ptr) );

    int64_t ld_img = images.stride[1];
    int64_t ld_lbl = labels.stride[1];

    g_rng = (cfg->seed == 0u) ? 1u : cfg->seed;

    int mindim = (cfg->w < cfg->h) ? cfg->w : cfg->h;
    int base = mindim / 8;
    if (base < 2) base = 2;

    for (int i = 0; i < cfg->n; i++) {
        uint8_t* sample = img_ptr + i;
        for (int64_t d = 0; d < plane; d++) {
            sample[d * ld_img] = 0;
        }

        int rA = base + irand(-1, 1);
        int rB = base + irand(-1, 1);
        if (rA < 1) rA = 1;
        if (rB < 1) rB = 1;

        int margin = (rA > rB ? rA : rB) + 1;
        int xA = irand(margin, cfg->w - 1 - margin);
        int yA = irand(margin, cfg->h - 1 - margin);

        int xB = 0, yB = 0;
        int tries = 0;
        int min_d = rA + rB + 2;
        int min_d2 = min_d * min_d;
        do {
            xB = irand(margin, cfg->w - 1 - margin);
            yB = irand(margin, cfg->h - 1 - margin);
            int dx = xB - xA;
            int dy = yB - yA;
            if (dx * dx + dy * dy >= min_d2) break;
            tries++;
        } while (tries < 2000);

        uint8_t col_circle[3], col_square[3];
        pick_key_colors_u8(cfg->mode, col_circle, col_square);
        draw_circle_u8(sample, ld_img, cfg->w, cfg->h, xA, yA, rA, col_circle[0]);
        draw_square_u8(sample, ld_img, cfg->w, cfg->h, xB, yB, rB, col_square[0]);

        for (int d = 0; d < cfg->distractors; d++) {
            uint8_t col[3];
            random_distractor_color_u8(cfg->mode, col);

            int t = irand(0, 2);
            if (t == 0 || t == 1) {
                int rr = clampi(base - 1 + irand(-1, 1), 1, base + 1);
                int m2 = rr + 1;
                int x = 0, y = 0;
                int min_dA = (rA + rr + 1);
                int min_dB = (rB + rr + 1);
                int min_d2_A = min_dA * min_dA;
                int min_d2_B = min_dB * min_dB;
                int ok = 0;
                for (int k = 0; k < 200; k++) {
                    x = irand(m2, cfg->w - 1 - m2);
                    y = irand(m2, cfg->h - 1 - m2);
                    if (far_enough(x, y, xA, yA, xB, yB, min_d2_A, min_d2_B)) { ok = 1; break; }
                }
                if (!ok) {
                    x = irand(m2, cfg->w - 1 - m2);
                    y = irand(m2, cfg->h - 1 - m2);
                }
                if (t == 0) {
                    draw_circle_u8(sample, ld_img, cfg->w, cfg->h, x, y, rr, col[0]);
                } else {
                    draw_square_u8(sample, ld_img, cfg->w, cfg->h, x, y, rr, col[0]);
                }
            } else {
                int x0 = irand(0, cfg->w - 1);
                int y0 = irand(0, cfg->h - 1);
                int x1 = irand(0, cfg->w - 1);
                int y1 = irand(0, cfg->h - 1);
                draw_line_u8(sample, ld_img, cfg->w, cfg->h, x0, y0, x1, y1, col[0]);
            }
        }

        add_noise_u8(sample, ld_img, cfg->w, cfg->h, cfg->noise);

        if (cfg->mode == KERMAC_RELSPRITES_MODE_BINARY) {
            threshold_binary_u8(sample, ld_img, cfg->w, cfg->h, 127);
        }

        int dx, dy, lr, ud, quad, angle8, dist_bin;
        float dist;
        compute_labels(cfg->w, cfg->h, xA, yA, xB, yB,
                       &dx, &dy, &dist,
                       &lr, &ud, &quad, &angle8, &dist_bin);

        float* lbl = label_ptr + i;
        lbl[REL_LABEL_DX * ld_lbl] = (float)dx;
        lbl[REL_LABEL_DY * ld_lbl] = (float)dy;
        lbl[REL_LABEL_DIST * ld_lbl] = dist;
        lbl[REL_LABEL_LR * ld_lbl] = (float)lr;
        lbl[REL_LABEL_UD * ld_lbl] = (float)ud;
        lbl[REL_LABEL_QUAD * ld_lbl] = (float)quad;
        lbl[REL_LABEL_ANGLE8 * ld_lbl] = (float)angle8;
        lbl[REL_LABEL_DIST_BIN * ld_lbl] = (float)dist_bin;
        lbl[REL_LABEL_RA * ld_lbl] = (float)rA;
        lbl[REL_LABEL_RB * ld_lbl] = (float)rB;
        lbl[REL_LABEL_XA * ld_lbl] = (float)xA;
        lbl[REL_LABEL_YA * ld_lbl] = (float)yA;
        lbl[REL_LABEL_XB * ld_lbl] = (float)xB;
        lbl[REL_LABEL_YB * ld_lbl] = (float)yB;
    }

    return KERMAC_SUCCESS;
}
