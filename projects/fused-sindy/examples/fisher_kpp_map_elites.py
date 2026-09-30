#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Charles Durham
# SPDX-License-Identifier: MIT
#
# MIT License
#
# Copyright (c) 2026 Charles Durham
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

import fused_sindy as isindy
from map_elites_common import (
    FEATURES,
    LEAVES,
    CudaMapElitesSettingsGenerator,
    active_indices,
    inherited_settings_tensors,
    word_to_f32,
)


PRIMITIVE_NAMES = ("u", "u_x", "u_xx")


@dataclass
class Candidate:
    asts: list[isindy.BinaryAst]
    inherited_terms: int
    inherited_masks: np.ndarray
    inherited_words: np.ndarray


@dataclass
class Elite:
    active_count: int
    mse: float
    cohort_idx: int
    setting_idx: int
    sweep_idx: int
    asts: list[isindy.BinaryAst]
    leaf_mask: np.ndarray
    leaf_words: np.ndarray
    active_mask: int
    beta: np.ndarray
    intercept: float
    robustness: int = 1


def make_data(seed: int, rows: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    u = rng.uniform(-1.0, 1.0, rows).astype(np.float32)
    ux = rng.normal(0.0, 0.7, rows).astype(np.float32)
    uxx = rng.normal(0.0, 0.9, rows).astype(np.float32)
    target = (np.float32(0.2) * uxx + np.float32(1.3) * u - np.float32(1.3) * u * u).astype(np.float32)
    primitive = np.stack([u, ux, uxx]).astype(np.float32)
    return primitive, target.reshape(1, rows)


def random_ast(rng: np.random.Generator) -> isindy.BinaryAst:
    unary_choices = np.asarray(
        [
            int(isindy.UnaryOp.IDENTITY),
            int(isindy.UnaryOp.IDENTITY),
            int(isindy.UnaryOp.IDENTITY),
            int(isindy.UnaryOp.IDENTITY),
            int(isindy.UnaryOp.IDENTITY),
            int(isindy.UnaryOp.SQUARE_F32),
            int(isindy.UnaryOp.SQUARE_F32),
            int(isindy.UnaryOp.NEG_FTZ_F32),
        ],
        dtype=np.int32,
    )
    binary_choices = np.asarray(
        [
            int(isindy.BinaryOp.KEEP_LEFT),
            int(isindy.BinaryOp.KEEP_LEFT),
            int(isindy.BinaryOp.KEEP_RIGHT),
            int(isindy.BinaryOp.KEEP_RIGHT),
            int(isindy.BinaryOp.MUL_FTZ_F32),
            int(isindy.BinaryOp.MUL_FTZ_F32),
            int(isindy.BinaryOp.MUL_FTZ_F32),
            int(isindy.BinaryOp.ADD_FTZ_F32),
            int(isindy.BinaryOp.SUB_FTZ_F32),
        ],
        dtype=np.int32,
    )
    unary = rng.choice(unary_choices, size=isindy.NUM_UNARY_OPS, replace=True)
    binary = rng.choice(binary_choices, size=isindy.NUM_BINARY_OPS, replace=True)
    return isindy.binary_ast(unary.tolist(), binary.tolist())


def empty_inherited_settings() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.zeros((FEATURES,), dtype=np.int32),
        np.zeros((FEATURES, LEAVES), dtype=np.int32),
    )


def random_candidate(rng: np.random.Generator) -> Candidate:
    inherited_masks, inherited_words = empty_inherited_settings()
    return Candidate([random_ast(rng) for _ in range(FEATURES)], 0, inherited_masks, inherited_words)


def merged_candidate(rng: np.random.Generator, elites: list[Elite]) -> Candidate:
    if not elites:
        return random_candidate(rng)

    parent_count = min(len(elites), 2)
    parents = list(rng.choice(elites, size=parent_count, replace=len(elites) < parent_count))
    asts: list[isindy.BinaryAst] = []
    inherited_masks, inherited_words = empty_inherited_settings()

    for parent in parents:
        for feature_idx in active_indices(parent.active_mask):
            inherited_idx = len(asts)
            asts.append(parent.asts[feature_idx])
            inherited_masks[inherited_idx] = np.int32(parent.leaf_mask[feature_idx])
            inherited_words[inherited_idx, :] = parent.leaf_words[feature_idx]
            if len(asts) == FEATURES:
                break
        if len(asts) == FEATURES:
            break

    inherited_terms = len(asts)
    while len(asts) < FEATURES:
        asts.append(random_ast(rng))
    return Candidate(asts, inherited_terms, inherited_masks, inherited_words)


def raw_beta_and_intercept(solve) -> tuple[np.ndarray, np.ndarray]:
    beta_standardized = solve.beta_standardized.detach().cpu().numpy()
    x_scale = solve.x_scale.detach().cpu().numpy()
    x_mean = solve.x_mean.detach().cpu().numpy()
    y_mean = solve.y_mean.detach().cpu().numpy()
    beta = beta_standardized / x_scale[None, :, None, :]
    intercept = y_mean[None, :, :] - np.sum(x_mean[None, :, None, :] * beta, axis=-1)
    return beta, intercept


def evaluate_candidates(
    candidates: list[Candidate],
    *,
    settings_generator: CudaMapElitesSettingsGenerator,
    settings: int,
    generation: int,
    seed: int,
    feature_leaf_probability: float,
    train_primitive_t,
    train_targets_t,
    validation_primitive_t,
    validation_targets_t,
    alphas_t,
    thresholds_t,
) -> tuple[list[tuple[Candidate, object, object, object]], list[float]]:
    all_asts: list[isindy.BinaryAst] = []
    for candidate in candidates:
        all_asts.extend(candidate.asts)

    results = []
    compile_begin = time.perf_counter()
    with isindy.compile_gram_kernels(
        all_asts,
        kernels_per_module=isindy.KernelsPerModule.FOUR,
        worker_count=0,
    ) as gram_kernels, isindy.RidgeSolver() as solver:
        compile_ms = (time.perf_counter() - compile_begin) * 1000.0
        stream = torch.cuda.current_stream(device=train_primitive_t.device)
        leaf_masks_batch_t = torch.empty((len(candidates), settings, FEATURES), device=train_primitive_t.device, dtype=torch.int32)
        leaf_words_batch_t = torch.empty((len(candidates), settings, FEATURES, LEAVES), device=train_primitive_t.device, dtype=torch.int32)
        inherited_masks_t, inherited_words_t, inherited_terms_t = inherited_settings_tensors(torch, candidates, train_primitive_t.device)
        settings_generator.generate(
            leaf_masks_batch_t,
            leaf_words_batch_t,
            inherited_masks_t,
            inherited_words_t,
            inherited_terms_t,
            stream,
            primitive_cols=len(PRIMITIVE_NAMES),
            seed=seed,
            generation=generation,
            start_cohort_seed=0,
            feature_leaf_probability=feature_leaf_probability,
        )

        for cohort_idx, candidate in enumerate(candidates):
            leaf_masks_t = leaf_masks_batch_t[cohort_idx]
            leaf_words_t = leaf_words_batch_t[cohort_idx]
            train_raw = gram_kernels.run_gram(
                train_primitive_t,
                train_targets_t,
                leaf_masks_t,
                leaf_words_t,
                cohort_index=cohort_idx,
            )
            validation_raw = gram_kernels.run_gram(
                validation_primitive_t,
                validation_targets_t,
                leaf_masks_t,
                leaf_words_t,
                cohort_index=cohort_idx,
            )
            solve = solver.solve_stlsq(train_raw, alphas_t, thresholds_t, row_count=train_primitive_t.shape[1])
            mse = solver.score_validation_mse(
                validation_raw,
                solve,
                validation_row_count=validation_primitive_t.shape[1],
            )
            results.append((candidate, solve, mse, leaf_masks_t, leaf_words_t))
        torch.cuda.synchronize()
    return results, compile_ms


def update_elites(
    elites: dict[int, Elite],
    results: list[tuple[Candidate, object, object, object, object]],
    *,
    generation: int,
) -> tuple[float, int]:
    best_mse = np.inf
    best_active_count = 0
    for cohort_idx, (candidate, solve, mse, leaf_masks_t, leaf_words_t) in enumerate(results):
        beta, intercept = raw_beta_and_intercept(solve)
        leaf_masks = leaf_masks_t.detach().cpu().numpy()
        leaf_words = leaf_words_t.detach().cpu().numpy()
        mse_np = mse.detach().cpu().numpy()[:, :, 0]
        solve_info = solve.solve_info.detach().cpu().numpy()
        active_counts = solve.active_counts.detach().cpu().numpy()[:, :, 0]
        active_masks = solve.active_masks.detach().cpu().numpy()[:, :, 0]
        mse_np = np.where(solve_info == 0, mse_np, np.inf)

        for sweep_idx in range(mse_np.shape[0]):
            for setting_idx in range(mse_np.shape[1]):
                mse_value = float(mse_np[sweep_idx, setting_idx])
                if not np.isfinite(mse_value):
                    continue
                active_count = int(active_counts[sweep_idx, setting_idx])
                if active_count <= 0:
                    continue
                if mse_value < best_mse:
                    best_mse = mse_value
                    best_active_count = active_count

                previous = elites.get(active_count)
                if previous is not None and mse_value >= previous.mse:
                    previous.robustness += int(mse_value <= previous.mse * 1.05 + 1.0e-9)
                    continue
                elites[active_count] = Elite(
                    active_count=active_count,
                    mse=mse_value,
                    cohort_idx=cohort_idx,
                    setting_idx=setting_idx,
                    sweep_idx=sweep_idx,
                    asts=candidate.asts,
                    leaf_mask=leaf_masks[setting_idx].copy(),
                    leaf_words=leaf_words[setting_idx].copy(),
                    active_mask=int(active_masks[sweep_idx, setting_idx]),
                    beta=beta[sweep_idx, setting_idx, 0].copy(),
                    intercept=float(intercept[sweep_idx, setting_idx, 0]),
                )
    print(f"generation={generation} best_mse={best_mse:.8e} best_active_count={best_active_count}")
    return best_mse, best_active_count


def render_feature(ast: isindy.BinaryAst, leaf_mask: np.ndarray, leaf_words: np.ndarray) -> str:
    values = []
    for leaf_idx in range(LEAVES):
        word = int(leaf_words[leaf_idx])
        if ((int(leaf_mask) >> leaf_idx) & 1) != 0:
            text = PRIMITIVE_NAMES[word] if 0 <= word < len(PRIMITIVE_NAMES) else f"p{word}"
        else:
            text = f"{word_to_f32(word):.4g}"
        values.append(apply_unary_text(ast.unary[leaf_idx], text))
    for ast_idx in range(LEAVES, isindy.NUM_UNARY_OPS):
        binary_idx, lhs_idx, rhs_idx = children(ast_idx)
        text = apply_binary_text(ast.binary[binary_idx], values[lhs_idx], values[rhs_idx])
        values.append(apply_unary_text(ast.unary[ast_idx], text))
    return values[-1]


def children(ast_idx: int) -> tuple[int, int, int]:
    if 8 <= ast_idx < 12:
        local = ast_idx - 8
        return local, local * 2, local * 2 + 1
    if 12 <= ast_idx < 14:
        local = ast_idx - 12
        return 4 + local, 8 + local * 2, 8 + local * 2 + 1
    if ast_idx == 14:
        return 6, 12, 13
    raise ValueError(f"node {ast_idx} does not have children")


def apply_unary_text(op: int, text: str) -> str:
    op = int(isindy.UnaryOp(op))
    if op == isindy.UnaryOp.IDENTITY:
        return text
    if op == isindy.UnaryOp.SQUARE_F32:
        return f"({text})^2"
    if op == isindy.UnaryOp.CUBE_F32:
        return f"({text})^3"
    if op == isindy.UnaryOp.NEG_FTZ_F32:
        return f"-({text})"
    if op == isindy.UnaryOp.ABS_FTZ_F32:
        return f"abs({text})"
    if op == isindy.UnaryOp.SAFE_RCP_F32:
        return f"safe_rcp({text})"
    if op == isindy.UnaryOp.SAFE_SQRT_F32:
        return f"safe_sqrt({text})"
    return f"{isindy.UnaryOp(op).name.lower()}({text})"


def apply_binary_text(op: int, lhs: str, rhs: str) -> str:
    op = int(isindy.BinaryOp(op))
    if op == isindy.BinaryOp.KEEP_LEFT:
        return lhs
    if op == isindy.BinaryOp.KEEP_RIGHT:
        return rhs
    if op == isindy.BinaryOp.ADD_FTZ_F32:
        return f"({lhs} + {rhs})"
    if op == isindy.BinaryOp.SUB_FTZ_F32:
        return f"({lhs} - {rhs})"
    if op == isindy.BinaryOp.MUL_FTZ_F32:
        return f"({lhs} * {rhs})"
    if op == isindy.BinaryOp.SAFE_DIV_F32:
        return f"safe_div({lhs}, {rhs})"
    return f"{isindy.BinaryOp(op).name.lower()}({lhs}, {rhs})"


def print_best_elites(elites: dict[int, Elite]) -> None:
    if not elites:
        print("no elites found")
        return
    best = min(elites.values(), key=lambda elite: (elite.mse + 1.0e-8 * elite.active_count, elite.active_count))
    print()
    print("best sparse-objective elite")
    print(f"  active_count={best.active_count} mse={best.mse:.8e} robustness={best.robustness}")
    if abs(best.intercept) >= 1.0e-3:
        print(f"  intercept={best.intercept:+.8g}")
    for feature_idx in active_indices(best.active_mask):
        coef = float(best.beta[feature_idx])
        if abs(coef) < 1.0e-3:
            continue
        expr = render_feature(best.asts[feature_idx], best.leaf_mask[feature_idx], best.leaf_words[feature_idx])
        print(f"  f{feature_idx:02d}: coef={coef:+.8g} expr={expr}")

    print()
    print("map elites by active count")
    for active_count in sorted(elites):
        elite = elites[active_count]
        print(f"  active={active_count:02d} mse={elite.mse:.8e} robustness={elite.robustness}")


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is not available")

    rng = np.random.default_rng(17)
    train_primitive, train_targets = make_data(101, 8192)
    validation_primitive, validation_targets = make_data(202, 4096)

    device = torch.device("cuda")
    train_primitive_t = torch.tensor(train_primitive, device=device)
    train_targets_t = torch.tensor(train_targets, device=device)
    validation_primitive_t = torch.tensor(validation_primitive, device=device)
    validation_targets_t = torch.tensor(validation_targets, device=device)
    alphas_t = torch.tensor([1.0e-8, 1.0e-7, 1.0e-6, 1.0e-5], device=device, dtype=torch.float32)
    thresholds_t = torch.tensor([1.0e-3, 5.0e-3, 1.0e-2, 5.0e-2], device=device, dtype=torch.float32)

    generations = 4
    cohorts_per_generation = 4
    settings = 256
    feature_leaf_probability = 0.72
    elites: dict[int, Elite] = {}

    print("synthetic Fisher-KPP MAP-Elites dry run")
    print("  hidden target used to generate data: u_t = 0.2*u_xx + 1.3*u - 1.3*u^2")
    print(f"  generations={generations} cohorts/generation={cohorts_per_generation} settings/cohort={settings}")

    with CudaMapElitesSettingsGenerator(torch, device) as settings_generator:
        print(f"  settings_jit_ms={settings_generator.jit_ms:.2f}")
        for generation in range(generations):
            candidates = []
            for cohort_idx in range(cohorts_per_generation):
                if generation == 0 or cohort_idx == 0 or not elites:
                    candidates.append(random_candidate(rng))
                else:
                    candidates.append(merged_candidate(rng, list(elites.values())))
            inherited = sum(candidate.inherited_terms for candidate in candidates)
            print(f"generation={generation} inherited_terms={inherited}")
            results, compile_ms = evaluate_candidates(
                candidates,
                settings_generator=settings_generator,
                settings=settings,
                generation=generation,
                seed=17,
                feature_leaf_probability=feature_leaf_probability,
                train_primitive_t=train_primitive_t,
                train_targets_t=train_targets_t,
                validation_primitive_t=validation_primitive_t,
                validation_targets_t=validation_targets_t,
                alphas_t=alphas_t,
                thresholds_t=thresholds_t,
            )
            print(f"generation={generation} compile_ms={compile_ms:.2f}")
            update_elites(elites, results, generation=generation)

    print_best_elites(elites)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
