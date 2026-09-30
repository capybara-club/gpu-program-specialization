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
/* Compiled only against the archived settings checkout, with hidden symbols. */
#include "secant.h"
#include "legacy_bridge.h"
#include <stdlib.h>
typedef struct Plan {
  void *storage;
  SecantCubinPlan *plan;
} Plan;
static SecantCubinDynamicLeafSSERecipe recipe(const LegacyShape *s){
  SecantCubinDynamicLeafSSERecipe r = secant_cubin_dynamic_leaf_sse_recipe_init();
  r.num_input_columns = s->columns;
  r.num_static_input_columns = 0;
  r.num_dynamic_leaves = s->slots;
  r.num_kernels = s->kernels;
  r.asts_per_kernel = s->packed;
  r.num_targets = 1;
  r.tile_rows = s->tile;
  r.threads_per_block = s->threads;
  r.patch_capacity_instructions = s->patch;
  return r;
}
static int
source(const LegacyShape *s, char **out)
{
  SecantCubinDynamicLeafSSERecipe r = recipe(s);
  size_t n;
  int e;
  *out = NULL;
  e = secant_cubin_source_size(&r.header, &n);
  if (e)
    return e;
  *out = malloc(n);
  if (!*out)
    return -1;
  e = secant_cubin_source_write(&r.header, *out, n);
  if (e) {
    free(*out);
    *out = NULL;
  } return e;
}
static int
plan(const LegacyShape *s, const void *cubin, size_t bytes, void **out)
{
  SecantCubinDynamicLeafSSERecipe r = recipe(s);
  Plan *p = NULL;
  size_t n;
  int e;
  *out = NULL;
  e = secant_cubin_plan_storage_size(&r.header, cubin, bytes, &n);
  if (e)
    return e;
  p = calloc(1, sizeof(*p));
  if (!p)
    return -1;
  p->storage = malloc(n);
  if (!p->storage) {
    free(p);
    return -1;
  }
  e = secant_cubin_plan_init(&r.header, cubin, bytes, p->storage, n, &p->plan);
  if (e) {
    free(p->storage);
    free(p);
    return e;
  } *out = p;
  return 0;
}
static int
specialize(void *p, const uint8_t * const *asts, size_t count, void *out, size_t bytes)
{
  SecantAstProgramSet programs = {0};
  programs.asts.items = asts;
  programs.asts.count = count;
  return secant_cubin_specialize_into(((Plan *)p)->plan, &programs, out, bytes);
}
static void
destroy(void *p)
{
  if (p) {
    free(((Plan *)p)->storage);
    free(p);
  }
}
__attribute__((visibility("default")))
const LegacyAPI *
secant_bench_legacy_v2(void)
{
  static const LegacyAPI api = {2, source, plan, specialize, destroy};
  return &api;
}
