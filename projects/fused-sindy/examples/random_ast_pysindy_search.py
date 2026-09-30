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

import argparse
import math
import os
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np

import fused_sindy as isindy


FEATURES = 32
LEAVES = 8


@dataclass
class SearchResult:
    mse_sum: float = math.inf
    objective: float = math.inf
    active_count: int = 0
    cohort_idx: int = -1
    setting_idx: int = -1
    sweep_idx: int = -1
    alpha: float = math.nan
    threshold: float = math.nan
    beta: np.ndarray | None = None
    intercept: np.ndarray | None = None
    feature_text: list[str] | None = None
    rhs_mse: np.ndarray | None = None


@dataclass
class CompiledBatch:
    batch_idx: int
    start_cohort: int
    batch_cohorts: int
    asts: list[isindy.BinaryAst]
    gram_kernels: object
    compile_ms: float


@dataclass
class PendingCohort:
    global_cohort_idx: int
    local_cohort_idx: int
    asts: list[isindy.BinaryAst]
    settings_np: tuple[np.ndarray, np.ndarray]
    leaf_masks_t: object
    leaf_words_t: object
    train_raw: object
    validation_raw: object
    stlsq: object
    validation_mse: object


@dataclass
class PendingBatch:
    compiled: CompiledBatch
    stream: object
    start_event: object
    end_event: object
    cohorts: list[PendingCohort]
    enqueue_ms: float


def f32_word(value: float) -> np.int32:
    return np.asarray([value], dtype=np.float32).view(np.int32)[0]


def word_to_f32(value: np.int32 | int) -> np.float32:
    return np.asarray([value], dtype=np.int32).view(np.float32)[0]


def make_trajectory_data(
    *,
    seed: int,
    trajectories: int,
    samples_per_trajectory: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows = trajectories * samples_per_trajectory
    primitive = np.empty((2, rows), dtype=np.float32)
    targets = np.empty((2, rows), dtype=np.float32)

    for trajectory in range(trajectories):
        x0 = np.float32(rng.uniform(0.35, 2.0))
        y0 = np.float32(rng.uniform(0.25, 1.5))
        for sample in range(samples_per_trajectory):
            row = trajectory * samples_per_trajectory + sample
            t = np.float32(sample / max(samples_per_trajectory - 1, 1))
            x = np.float32(x0 * np.exp(np.float32(-2.0) * t))
            y = np.float32(y0 * np.exp(t))
            primitive[0, row] = x
            primitive[1, row] = y
            targets[0, row] = np.float32(-2.0) * x
            targets[1, row] = y
    return primitive, targets


def random_simple_ast_cohort(
    rng: np.random.Generator,
    count: int = FEATURES,
) -> list[isindy.BinaryAst]:
    unary_choices = np.asarray(
        [
            int(isindy.UnaryOp.IDENTITY),
            int(isindy.UnaryOp.SQUARE_F32),
            int(isindy.UnaryOp.CUBE_F32),
            int(isindy.UnaryOp.NEG_FTZ_F32),
        ],
        dtype=np.int32,
    )
    unary_prob = np.asarray([0.86, 0.07, 0.03, 0.04], dtype=np.float64)
    binary_choices = np.asarray(
        [
            int(isindy.BinaryOp.KEEP_LEFT),
            int(isindy.BinaryOp.KEEP_RIGHT),
            int(isindy.BinaryOp.ADD_FTZ_F32),
            int(isindy.BinaryOp.SUB_FTZ_F32),
            int(isindy.BinaryOp.MUL_FTZ_F32),
        ],
        dtype=np.int32,
    )
    binary_prob = np.asarray([0.34, 0.34, 0.12, 0.08, 0.12], dtype=np.float64)

    asts: list[isindy.BinaryAst] = []
    for _ in range(count):
        unary = rng.choice(unary_choices, size=isindy.NUM_UNARY_OPS, replace=True, p=unary_prob)
        binary = rng.choice(binary_choices, size=isindy.NUM_BINARY_OPS, replace=True, p=binary_prob)
        asts.append(isindy.binary_ast(unary.tolist(), binary.tolist()))
    return asts


def random_leaf_settings(
    rng: np.random.Generator,
    settings: int,
    *,
    feature_leaf_probability: float = 0.82,
) -> tuple[np.ndarray, np.ndarray]:
    masks = np.zeros((settings, FEATURES), dtype=np.int32)
    words = np.empty((settings, FEATURES, LEAVES), dtype=np.int32)
    constants = np.asarray(
        [
            f32_word(-2.0),
            f32_word(-1.0),
            f32_word(-0.5),
            f32_word(0.0),
            f32_word(0.5),
            f32_word(1.0),
            f32_word(2.0),
        ],
        dtype=np.int32,
    )

    for leaf_idx in range(LEAVES):
        use_feature = rng.random((settings, FEATURES)) < feature_leaf_probability
        feature_cols = rng.integers(0, 2, size=(settings, FEATURES), dtype=np.int32)
        constant_words = rng.choice(constants, size=(settings, FEATURES), replace=True)
        words[:, :, leaf_idx] = np.where(use_feature, feature_cols, constant_words)
        masks |= np.where(use_feature, np.int32(1 << leaf_idx), np.int32(0))
    return masks, words


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
    if op == isindy.UnaryOp.RCP_APPROX_FTZ_F32:
        return f"rcp({text})"
    if op == isindy.UnaryOp.SQRT_APPROX_FTZ_F32:
        return f"sqrt({text})"
    if op == isindy.UnaryOp.RSQRT_APPROX_FTZ_F32:
        return f"rsqrt({text})"
    if op == isindy.UnaryOp.SIN_APPROX_FTZ_F32:
        return f"sin({text})"
    if op == isindy.UnaryOp.COS_APPROX_FTZ_F32:
        return f"cos({text})"
    if op == isindy.UnaryOp.EX2_APPROX_FTZ_F32:
        return f"ex2({text})"
    if op == isindy.UnaryOp.EXP_APPROX_FTZ_F32:
        return f"exp({text})"
    if op == isindy.UnaryOp.LOG2_APPROX_FTZ_F32:
        return f"log2({text})"
    if op == isindy.UnaryOp.LOG10_APPROX_FTZ_F32:
        return f"log10({text})"
    if op == isindy.UnaryOp.SAFE_RCP_F32:
        return f"safe_rcp({text})"
    if op == isindy.UnaryOp.SAFE_SQRT_F32:
        return f"safe_sqrt({text})"
    if op == isindy.UnaryOp.SAFE_RSQRT_F32:
        return f"safe_rsqrt({text})"
    if op == isindy.UnaryOp.SAFE_EX2_F32:
        return f"safe_ex2({text})"
    if op == isindy.UnaryOp.SAFE_EXP_F32:
        return f"safe_exp({text})"
    if op == isindy.UnaryOp.SAFE_LOG2_F32:
        return f"safe_log2({text})"
    if op == isindy.UnaryOp.SAFE_LOG10_F32:
        return f"safe_log10({text})"
    raise ValueError(f"unsupported unary op {op}")


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
    if op == isindy.BinaryOp.DIV_APPROX_FTZ_F32:
        return f"({lhs} / {rhs})"
    if op == isindy.BinaryOp.MIN_FTZ_F32:
        return f"min({lhs}, {rhs})"
    if op == isindy.BinaryOp.MAX_FTZ_F32:
        return f"max({lhs}, {rhs})"
    if op == isindy.BinaryOp.SAFE_DIV_F32:
        return f"safe_div({lhs}, {rhs})"
    raise ValueError(f"unsupported binary op {op}")


def feature_texts(
    asts: Sequence[isindy.BinaryAst],
    leaf_masks: np.ndarray,
    leaf_words: np.ndarray,
    setting_idx: int,
) -> list[str]:
    out: list[str] = []
    for feature_idx, ast in enumerate(asts):
        mask = int(np.uint32(leaf_masks[setting_idx, feature_idx]))
        values: list[str] = []
        for leaf_idx in range(LEAVES):
            word = leaf_words[setting_idx, feature_idx, leaf_idx]
            if ((mask >> leaf_idx) & 1) != 0:
                text = "x" if int(word) == 0 else "y" if int(word) == 1 else f"u{int(word)}"
            else:
                text = f"{float(word_to_f32(word)):.6g}"
            values.append(apply_unary_text(ast.unary[leaf_idx], text))

        for ast_idx in range(LEAVES, isindy.NUM_UNARY_OPS):
            binary_idx, lhs_idx, rhs_idx = children(ast_idx)
            text = apply_binary_text(ast.binary[binary_idx], values[lhs_idx], values[rhs_idx])
            values.append(apply_unary_text(ast.unary[ast_idx], text))
        out.append(values[-1])
    return out


def raw_beta_and_intercept(solve) -> tuple[np.ndarray, np.ndarray]:
    beta_standardized = solve.beta_standardized.detach().cpu().numpy()
    x_scale = solve.x_scale.detach().cpu().numpy()
    x_mean = solve.x_mean.detach().cpu().numpy()
    y_mean = solve.y_mean.detach().cpu().numpy()
    beta = beta_standardized / x_scale[None, :, None, :]
    intercept = y_mean[None, :, :] - np.sum(x_mean[None, :, None, :] * beta, axis=-1)
    return beta, intercept


def equation_text(
    lhs: str,
    intercept: float,
    beta: np.ndarray,
    feature_text: Sequence[str],
    *,
    tol: float,
) -> str:
    terms: list[tuple[int, float, str]] = []
    if abs(intercept) >= tol:
        terms.append((-1, float(intercept), "1"))
    for idx, coef in enumerate(beta):
        coef_f = float(coef)
        if abs(coef_f) >= tol:
            terms.append((idx, coef_f, feature_text[idx]))
    if not terms:
        return f"{lhs} = 0"
    rhs = " + ".join(f"{coef:+.6g}*{text}" for _, coef, text in terms)
    return f"{lhs} = {rhs}"


def update_best(
    best: SearchResult,
    *,
    cohort_idx: int,
    settings_np: tuple[np.ndarray, np.ndarray],
    asts: Sequence[isindy.BinaryAst],
    stlsq,
    mse,
    solve_info,
    alphas: np.ndarray,
    thresholds: np.ndarray,
    coefficient_tol: float,
    sparsity_penalty: float,
) -> SearchResult:
    beta, intercept = raw_beta_and_intercept(stlsq)
    active = (np.abs(beta) >= coefficient_tol).sum(axis=(2, 3))
    active += (np.abs(intercept) >= coefficient_tol).sum(axis=2)
    total = mse.sum(axis=-1)
    invalid = solve_info != 0
    total = np.where(invalid, np.inf, total)
    objective = total + sparsity_penalty * active
    flat_idx = int(np.argmin(objective))
    sweep_idx, setting_idx = np.unravel_index(flat_idx, total.shape)
    mse_sum = float(total[sweep_idx, setting_idx])
    objective_value = float(objective[sweep_idx, setting_idx])
    if not np.isfinite(mse_sum) or objective_value >= best.objective:
        return best

    leaf_masks, leaf_words = settings_np
    text = feature_texts(asts, leaf_masks, leaf_words, setting_idx)
    return SearchResult(
        mse_sum=mse_sum,
        objective=objective_value,
        active_count=int(active[sweep_idx, setting_idx]),
        cohort_idx=cohort_idx,
        setting_idx=int(setting_idx),
        sweep_idx=int(sweep_idx),
        alpha=float(alphas[sweep_idx]),
        threshold=float(thresholds[sweep_idx]),
        beta=beta[sweep_idx, setting_idx].copy(),
        intercept=intercept[sweep_idx, setting_idx].copy(),
        feature_text=text,
        rhs_mse=mse[sweep_idx, setting_idx].copy(),
    )


def make_batch_asts(
    rng: np.random.Generator,
    batch_cohorts: int,
) -> list[isindy.BinaryAst]:
    asts: list[isindy.BinaryAst] = []
    for _ in range(batch_cohorts):
        asts.extend(random_simple_ast_cohort(rng, FEATURES))
    return asts


def compile_batch(
    *,
    batch_idx: int,
    start_cohort: int,
    batch_cohorts: int,
    asts: list[isindy.BinaryAst],
    worker_count: int,
) -> CompiledBatch:
    compile_begin = time.perf_counter()
    gram_kernels = isindy.compile_gram_kernels(
        asts,
        kernels_per_module=isindy.KernelsPerModule.FOUR,
        worker_count=worker_count,
    )
    compile_ms = (time.perf_counter() - compile_begin) * 1000.0
    return CompiledBatch(
        batch_idx=batch_idx,
        start_cohort=start_cohort,
        batch_cohorts=batch_cohorts,
        asts=asts,
        gram_kernels=gram_kernels,
        compile_ms=compile_ms,
    )


def enqueue_batch(
    *,
    torch,
    rng: np.random.Generator,
    solver,
    compiled: CompiledBatch,
    train_primitive_t,
    train_targets_t,
    validation_primitive_t,
    validation_targets_t,
    alphas_t,
    thresholds_t,
    train_rows: int,
    validation_rows: int,
    settings: int,
    feature_leaf_probability: float,
) -> PendingBatch:
    stream = torch.cuda.Stream(device=compiled.gram_kernels.device)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    cohorts: list[PendingCohort] = []
    enqueue_begin = time.perf_counter()

    stream.wait_stream(torch.cuda.current_stream(device=compiled.gram_kernels.device))
    with torch.cuda.device(compiled.gram_kernels.device), torch.cuda.stream(stream):
        start_event.record(stream)
        for local_cohort_idx in range(compiled.batch_cohorts):
            global_cohort_idx = compiled.start_cohort + local_cohort_idx
            cohort_asts = compiled.asts[local_cohort_idx * FEATURES : (local_cohort_idx + 1) * FEATURES]
            leaf_masks, leaf_words = random_leaf_settings(
                rng,
                settings,
                feature_leaf_probability=feature_leaf_probability,
            )
            leaf_masks_t = torch.tensor(leaf_masks, device=compiled.gram_kernels.device, dtype=torch.int32)
            leaf_words_t = torch.tensor(leaf_words, device=compiled.gram_kernels.device, dtype=torch.int32)

            train_raw = compiled.gram_kernels.run_gram(
                train_primitive_t,
                train_targets_t,
                leaf_masks_t,
                leaf_words_t,
                cohort_index=local_cohort_idx,
                stream=stream,
            )
            validation_raw = compiled.gram_kernels.run_gram(
                validation_primitive_t,
                validation_targets_t,
                leaf_masks_t,
                leaf_words_t,
                cohort_index=local_cohort_idx,
                stream=stream,
            )
            stlsq = solver.solve_stlsq(
                train_raw,
                alphas_t,
                thresholds_t,
                row_count=train_rows,
                stream=stream,
            )
            validation_mse = solver.score_validation_mse(
                validation_raw,
                stlsq,
                validation_row_count=validation_rows,
                stream=stream,
            )
            cohorts.append(
                PendingCohort(
                    global_cohort_idx=global_cohort_idx,
                    local_cohort_idx=local_cohort_idx,
                    asts=cohort_asts,
                    settings_np=(leaf_masks, leaf_words),
                    leaf_masks_t=leaf_masks_t,
                    leaf_words_t=leaf_words_t,
                    train_raw=train_raw,
                    validation_raw=validation_raw,
                    stlsq=stlsq,
                    validation_mse=validation_mse,
                )
            )
        end_event.record(stream)

    return PendingBatch(
        compiled=compiled,
        stream=stream,
        start_event=start_event,
        end_event=end_event,
        cohorts=cohorts,
        enqueue_ms=(time.perf_counter() - enqueue_begin) * 1000.0,
    )


def evaluate_pending_batch(
    *,
    pending: PendingBatch,
    best: SearchResult,
    alphas_np: np.ndarray,
    thresholds_np: np.ndarray,
    coefficient_tol: float,
    sparsity_penalty: float,
    print_cohorts: bool,
) -> SearchResult:
    previous_objective = best.objective
    wait_begin = time.perf_counter()
    pending.stream.synchronize()
    wait_ms = (time.perf_counter() - wait_begin) * 1000.0
    gpu_ms = float(pending.start_event.elapsed_time(pending.end_event))
    module_count = pending.compiled.gram_kernels.num_cubins
    batch_best_mse = math.inf

    try:
        for cohort in pending.cohorts:
            mse_np = cohort.validation_mse.detach().cpu().numpy()
            solve_info_np = cohort.stlsq.solve_info.detach().cpu().numpy()
            best = update_best(
                best,
                cohort_idx=cohort.global_cohort_idx,
                settings_np=cohort.settings_np,
                asts=cohort.asts,
                stlsq=cohort.stlsq,
                mse=mse_np,
                solve_info=solve_info_np,
                alphas=alphas_np,
                thresholds=thresholds_np,
                coefficient_tol=coefficient_tol,
                sparsity_penalty=sparsity_penalty,
            )
            cohort_total = mse_np.sum(axis=-1)
            cohort_total = np.where(solve_info_np != 0, np.inf, cohort_total)
            cohort_best = float(np.min(cohort_total))
            batch_best_mse = min(batch_best_mse, cohort_best)
            if print_cohorts:
                print(
                    f"  cohort {cohort.global_cohort_idx:03d}: "
                    f"cohort_best_mse={cohort_best:.8e} global_best_mse={best.mse_sum:.8e}",
                    flush=True,
                )
    finally:
        pending.compiled.gram_kernels.close()

    improved = best.objective < previous_objective
    print(
        f"  batch {pending.compiled.batch_idx:03d}: "
        f"cohorts={pending.compiled.batch_cohorts} modules={module_count} "
        f"compile_ms={pending.compiled.compile_ms:.2f} enqueue_ms={pending.enqueue_ms:.2f} "
        f"gpu_ms={gpu_ms:.2f} sync_wait_ms={wait_ms:.2f} "
        f"batch_best_mse={batch_best_mse:.8e} global_best_mse={best.mse_sum:.8e} "
        f"improved={'yes' if improved else 'no'}",
        flush=True,
    )
    return best


def print_best(best: SearchResult, *, coefficient_tol: float) -> None:
    if best.beta is None or best.intercept is None or best.feature_text is None or best.rhs_mse is None:
        print("no finite candidate found")
        return
    print()
    print("best candidate")
    print(f"  cohort={best.cohort_idx} setting={best.setting_idx} sweep={best.sweep_idx}")
    print(f"  alpha={best.alpha:.3g} threshold={best.threshold:.3g}")
    print(f"  validation_mse_sum={best.mse_sum:.8e} rhs_mse={best.rhs_mse}")
    print(f"  objective={best.objective:.8e} active_count={best.active_count}")
    print("  " + equation_text("x_dot", float(best.intercept[0]), best.beta[0], best.feature_text, tol=coefficient_tol))
    print("  " + equation_text("y_dot", float(best.intercept[1]), best.beta[1], best.feature_text, tol=coefficient_tol))
    print("  active features")
    active = sorted(
        {
            idx
            for idx in range(FEATURES)
            if abs(float(best.beta[0, idx])) >= coefficient_tol or abs(float(best.beta[1, idx])) >= coefficient_tol
        }
    )
    for idx in active:
        print(f"    f{idx:02d}: {best.feature_text[idx]}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cohorts", type=int, default=8)
    parser.add_argument("--settings", type=int, default=512)
    parser.add_argument("--cohorts-per-compile", type=int, default=4)
    parser.add_argument("--modules-per-compile", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=0)
    parser.add_argument("--train-trajectories", type=int, default=24)
    parser.add_argument("--validation-trajectories", type=int, default=12)
    parser.add_argument("--samples-per-trajectory", type=int, default=96)
    parser.add_argument("--feature-leaf-probability", type=float, default=0.82)
    parser.add_argument("--coefficient-tol", type=float, default=1.0e-3)
    parser.add_argument("--sparsity-penalty", type=float, default=1.0e-8)
    parser.add_argument("--print-cohorts", action="store_true")
    args = parser.parse_args()

    if args.modules_per_compile < 0:
        raise ValueError("modules-per-compile must be non-negative")
    if args.cohorts <= 0 or args.settings <= 0 or args.cohorts_per_compile <= 0:
        raise ValueError("cohorts, settings, and cohorts-per-compile must be positive")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is not available")

    kernels_per_module = int(isindy.KernelsPerModule.FOUR)
    if args.modules_per_compile > 0:
        cohorts_per_compile = args.modules_per_compile * kernels_per_module
    else:
        cohorts_per_compile = args.cohorts_per_compile
    if cohorts_per_compile <= 0:
        raise ValueError("compiled cohort batch size must be positive")

    rng = np.random.default_rng(args.seed)
    train_primitive, train_targets = make_trajectory_data(
        seed=args.seed + 101,
        trajectories=args.train_trajectories,
        samples_per_trajectory=args.samples_per_trajectory,
    )
    validation_primitive, validation_targets = make_trajectory_data(
        seed=args.seed + 202,
        trajectories=args.validation_trajectories,
        samples_per_trajectory=args.samples_per_trajectory,
    )

    train_primitive_t = torch.tensor(train_primitive, device="cuda")
    train_targets_t = torch.tensor(train_targets, device="cuda")
    validation_primitive_t = torch.tensor(validation_primitive, device="cuda")
    validation_targets_t = torch.tensor(validation_targets, device="cuda")
    alphas_np = np.asarray([1.0e-8, 1.0e-8, 1.0e-6, 1.0e-6, 1.0e-6, 1.0e-4], dtype=np.float32)
    thresholds_np = np.asarray([1.0e-4, 1.0e-3, 1.0e-2, 5.0e-2, 1.0e-1, 2.0e-1], dtype=np.float32)
    alphas_t = torch.tensor(alphas_np, device="cuda")
    thresholds_t = torch.tensor(thresholds_np, device="cuda")

    print("random AST pySINDy-style search")
    print(f"  target: x_dot = -2*x, y_dot = y")
    print(f"  cohorts={args.cohorts} settings/cohort={args.settings}")
    print(
        f"  modules/compile={math.ceil(cohorts_per_compile / kernels_per_module)} "
        f"cohorts/compile={cohorts_per_compile} worker_count={args.worker_count or (os.cpu_count() or 1)}"
    )
    print(f"  train_rows={train_primitive.shape[1]} validation_rows={validation_primitive.shape[1]}")

    best = SearchResult()
    start_time = time.perf_counter()
    generated = 0
    batch_idx = 0
    pending: PendingBatch | None = None
    with isindy.RidgeSolver() as solver:
        def compile_next() -> CompiledBatch | None:
            nonlocal batch_idx, generated
            if generated >= args.cohorts:
                return None
            batch_cohorts = min(cohorts_per_compile, args.cohorts - generated)
            asts = make_batch_asts(rng, batch_cohorts)
            compiled = compile_batch(
                batch_idx=batch_idx,
                start_cohort=generated,
                batch_cohorts=batch_cohorts,
                asts=asts,
                worker_count=args.worker_count,
            )
            generated += batch_cohorts
            batch_idx += 1
            return compiled

        compiled = compile_next()
        if compiled is None:
            raise RuntimeError("no cohorts to compile")
        pending = enqueue_batch(
            torch=torch,
            rng=rng,
            solver=solver,
            compiled=compiled,
            train_primitive_t=train_primitive_t,
            train_targets_t=train_targets_t,
            validation_primitive_t=validation_primitive_t,
            validation_targets_t=validation_targets_t,
            alphas_t=alphas_t,
            thresholds_t=thresholds_t,
            train_rows=train_primitive.shape[1],
            validation_rows=validation_primitive.shape[1],
            settings=args.settings,
            feature_leaf_probability=args.feature_leaf_probability,
        )

        while generated < args.cohorts:
            compiled = compile_next()
            if pending is None or compiled is None:
                raise RuntimeError("internal pipeline state error")
            best = evaluate_pending_batch(
                pending=pending,
                best=best,
                alphas_np=alphas_np,
                thresholds_np=thresholds_np,
                coefficient_tol=args.coefficient_tol,
                sparsity_penalty=args.sparsity_penalty,
                print_cohorts=args.print_cohorts,
            )
            pending = enqueue_batch(
                torch=torch,
                rng=rng,
                solver=solver,
                compiled=compiled,
                train_primitive_t=train_primitive_t,
                train_targets_t=train_targets_t,
                validation_primitive_t=validation_primitive_t,
                validation_targets_t=validation_targets_t,
                alphas_t=alphas_t,
                thresholds_t=thresholds_t,
                train_rows=train_primitive.shape[1],
                validation_rows=validation_primitive.shape[1],
                settings=args.settings,
                feature_leaf_probability=args.feature_leaf_probability,
            )

        if pending is not None:
            best = evaluate_pending_batch(
                pending=pending,
                best=best,
                alphas_np=alphas_np,
                thresholds_np=thresholds_np,
                coefficient_tol=args.coefficient_tol,
                sparsity_penalty=args.sparsity_penalty,
                print_cohorts=args.print_cohorts,
            )

    elapsed = time.perf_counter() - start_time
    print(f"elapsed_s={elapsed:.3f}")
    print_best(best, coefficient_tol=args.coefficient_tol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
