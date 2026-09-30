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


RHS_NAMES = ("u_t", "v_t")
PRIMITIVE_NAMES = ("u", "v", "u_x", "v_x", "u_xx", "v_xx")


@dataclass
class Candidate:
    asts: list[isindy.BinaryAst]
    inherited_terms: int
    inherited_masks: np.ndarray
    inherited_words: np.ndarray


@dataclass
class Elite:
    active_count: int
    objective: float
    mse: float
    generation: int
    cohort_idx: int
    setting_idx: int
    sweep_idx: int
    asts: list[isindy.BinaryAst]
    leaf_mask: np.ndarray
    leaf_words: np.ndarray
    active_mask: int
    active_masks_by_rhs: np.ndarray
    beta: np.ndarray
    intercept: np.ndarray
    robustness: int = 1


def target_description(feed: float, kill: float, diffusivity_u: float, diffusivity_v: float) -> str:
    return (
        "u_t = D_u*u_xx - u*v^2 + F*(1 - u), "
        "v_t = D_v*v_xx + u*v^2 - (F + k)*v "
        f"with D_u={diffusivity_u:g}, D_v={diffusivity_v:g}, F={feed:g}, k={kill:g}"
    )


def make_targets(
    u: np.ndarray,
    v: np.ndarray,
    uxx: np.ndarray,
    vxx: np.ndarray,
    *,
    feed: float,
    kill: float,
    diffusivity_u: float,
    diffusivity_v: float,
) -> np.ndarray:
    reaction = u * v * v
    u_t = np.float32(diffusivity_u) * uxx - reaction + np.float32(feed) * (np.float32(1.0) - u)
    v_t = np.float32(diffusivity_v) * vxx + reaction - np.float32(feed + kill) * v
    return np.stack([u_t, v_t]).astype(np.float32)


def make_fourier_pair_data(
    seed: int,
    rows: int,
    modes: int,
    *,
    feed: float,
    kill: float,
    diffusivity_u: float,
    diffusivity_v: float,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 2.0 * np.pi, rows).astype(np.float32)
    u = np.full(rows, np.float32(1.0), dtype=np.float32)
    v = np.full(rows, np.float32(0.5), dtype=np.float32)
    ux = np.zeros(rows, dtype=np.float32)
    vx = np.zeros(rows, dtype=np.float32)
    uxx = np.zeros(rows, dtype=np.float32)
    vxx = np.zeros(rows, dtype=np.float32)

    for k in range(1, modes + 1):
        scale = np.float32(0.55 / (k * k))
        au = np.float32(rng.normal()) * scale
        bu = np.float32(rng.normal()) * scale
        av = np.float32(rng.normal()) * scale
        bv = np.float32(rng.normal()) * scale
        phase_u = np.float32(rng.uniform(0.0, 2.0 * np.pi))
        phase_v = np.float32(rng.uniform(0.0, 2.0 * np.pi))
        kf = np.float32(k)

        theta_u = kf * x + phase_u
        su = np.sin(theta_u).astype(np.float32)
        cu = np.cos(theta_u).astype(np.float32)
        u += au * su + bu * cu
        ux += kf * au * cu - kf * bu * su
        uxx += -(kf * kf) * au * su - (kf * kf) * bu * cu

        theta_v = kf * x + phase_v
        sv = np.sin(theta_v).astype(np.float32)
        cv = np.cos(theta_v).astype(np.float32)
        v += av * sv + bv * cv
        vx += kf * av * cv - kf * bv * sv
        vxx += -(kf * kf) * av * sv - (kf * kf) * bv * cv

    primitive = np.stack([u, v, ux, vx, uxx, vxx]).astype(np.float32)
    targets = make_targets(
        u,
        v,
        uxx,
        vxx,
        feed=feed,
        kill=kill,
        diffusivity_u=diffusivity_u,
        diffusivity_v=diffusivity_v,
    )
    return primitive, targets


def random_ast(rng: np.random.Generator, *, broad_ops: bool) -> isindy.BinaryAst:
    unary_ops = [
        int(isindy.UnaryOp.IDENTITY),
        int(isindy.UnaryOp.IDENTITY),
        int(isindy.UnaryOp.IDENTITY),
        int(isindy.UnaryOp.IDENTITY),
        int(isindy.UnaryOp.IDENTITY),
        int(isindy.UnaryOp.SQUARE_F32),
        int(isindy.UnaryOp.SQUARE_F32),
        int(isindy.UnaryOp.NEG_FTZ_F32),
    ]
    binary_ops = [
        int(isindy.BinaryOp.KEEP_LEFT),
        int(isindy.BinaryOp.KEEP_LEFT),
        int(isindy.BinaryOp.KEEP_RIGHT),
        int(isindy.BinaryOp.KEEP_RIGHT),
        int(isindy.BinaryOp.MUL_FTZ_F32),
        int(isindy.BinaryOp.MUL_FTZ_F32),
        int(isindy.BinaryOp.MUL_FTZ_F32),
        int(isindy.BinaryOp.ADD_FTZ_F32),
        int(isindy.BinaryOp.SUB_FTZ_F32),
    ]
    if broad_ops:
        unary_ops.extend(
            [
                int(isindy.UnaryOp.CUBE_F32),
                int(isindy.UnaryOp.ABS_FTZ_F32),
                int(isindy.UnaryOp.SAFE_RCP_F32),
                int(isindy.UnaryOp.SAFE_SQRT_F32),
            ]
        )
        binary_ops.extend([int(isindy.BinaryOp.SAFE_DIV_F32), int(isindy.BinaryOp.MIN_FTZ_F32), int(isindy.BinaryOp.MAX_FTZ_F32)])

    unary = rng.choice(np.asarray(unary_ops, dtype=np.int32), size=isindy.NUM_UNARY_OPS, replace=True)
    binary = rng.choice(np.asarray(binary_ops, dtype=np.int32), size=isindy.NUM_BINARY_OPS, replace=True)
    return isindy.binary_ast(unary.tolist(), binary.tolist())


def empty_inherited_settings() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.zeros((FEATURES,), dtype=np.int32),
        np.zeros((FEATURES, LEAVES), dtype=np.int32),
    )


def random_candidate(rng: np.random.Generator, broad_ops: bool) -> Candidate:
    inherited_masks, inherited_words = empty_inherited_settings()
    return Candidate([random_ast(rng, broad_ops=broad_ops) for _ in range(FEATURES)], 0, inherited_masks, inherited_words)


def merged_candidate(
    rng: np.random.Generator,
    elites: list[Elite],
    broad_ops: bool,
) -> Candidate:
    if not elites:
        return random_candidate(rng, broad_ops)

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
        asts.append(random_ast(rng, broad_ops=broad_ops))
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
) -> tuple[list[tuple[Candidate, object, object, object]], float, float]:
    all_asts: list[isindy.BinaryAst] = []
    for candidate in candidates:
        all_asts.extend(candidate.asts)

    compile_begin = time.perf_counter()
    results = []
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
        gpu_begin = torch.cuda.Event(enable_timing=True)
        gpu_end = torch.cuda.Event(enable_timing=True)
        gpu_begin.record()
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
        gpu_end.record()
        torch.cuda.synchronize()
        gpu_ms = float(gpu_begin.elapsed_time(gpu_end))
    return results, compile_ms, gpu_ms


def update_elites(
    elites: dict[int, Elite],
    results: list[tuple[Candidate, object, object, object, object]],
    *,
    generation: int,
    sparsity_penalty: float,
) -> tuple[float, int]:
    best_mse = np.inf
    best_active_count = 0
    for cohort_idx, (candidate, solve, mse, leaf_masks_t, leaf_words_t) in enumerate(results):
        beta, intercept = raw_beta_and_intercept(solve)
        leaf_masks = leaf_masks_t.detach().cpu().numpy()
        leaf_words = leaf_words_t.detach().cpu().numpy()
        mse_np = mse.detach().cpu().numpy()
        combined_mse = np.mean(mse_np, axis=2)
        solve_info = solve.solve_info.detach().cpu().numpy()
        active_masks = solve.active_masks.detach().cpu().numpy()
        combined_mse = np.where(solve_info == 0, combined_mse, np.inf)

        for sweep_idx in range(combined_mse.shape[0]):
            for setting_idx in range(combined_mse.shape[1]):
                mse_value = float(combined_mse[sweep_idx, setting_idx])
                if not np.isfinite(mse_value):
                    continue
                union_mask = 0
                for rhs_idx in range(active_masks.shape[2]):
                    union_mask |= int(active_masks[sweep_idx, setting_idx, rhs_idx])
                active_count = int(union_mask.bit_count())
                if active_count <= 0:
                    continue
                objective = mse_value + sparsity_penalty * active_count
                if mse_value < best_mse:
                    best_mse = mse_value
                    best_active_count = active_count

                previous = elites.get(active_count)
                if previous is not None and objective >= previous.objective:
                    previous.robustness += int(mse_value <= previous.mse * 1.05 + 1.0e-9)
                    continue
                elites[active_count] = Elite(
                    active_count=active_count,
                    objective=objective,
                    mse=mse_value,
                    generation=generation,
                    cohort_idx=cohort_idx,
                    setting_idx=setting_idx,
                    sweep_idx=sweep_idx,
                    asts=candidate.asts,
                    leaf_mask=leaf_masks[setting_idx].copy(),
                    leaf_words=leaf_words[setting_idx].copy(),
                    active_mask=union_mask,
                    active_masks_by_rhs=active_masks[sweep_idx, setting_idx].copy(),
                    beta=beta[sweep_idx, setting_idx].copy(),
                    intercept=intercept[sweep_idx, setting_idx].copy(),
                )
    return best_mse, best_active_count


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
    if op == isindy.BinaryOp.MIN_FTZ_F32:
        return f"min({lhs}, {rhs})"
    if op == isindy.BinaryOp.MAX_FTZ_F32:
        return f"max({lhs}, {rhs})"
    return f"{isindy.BinaryOp(op).name.lower()}({lhs}, {rhs})"


def print_best_elites(elites: dict[int, Elite]) -> None:
    if not elites:
        print("no elites found")
        return
    best = min(elites.values(), key=lambda elite: (elite.objective, elite.active_count))
    print()
    print("best sparse-objective elite")
    print(
        f"  active_count={best.active_count} mean_validation_mse={best.mse:.8e} "
        f"objective={best.objective:.8e} generation={best.generation} robustness={best.robustness}"
    )
    for rhs_idx, rhs_name in enumerate(RHS_NAMES):
        print(f"  {rhs_name}:")
        if abs(float(best.intercept[rhs_idx])) >= 1.0e-4:
            print(f"    intercept={float(best.intercept[rhs_idx]):+.8g}")
        rhs_mask = int(best.active_masks_by_rhs[rhs_idx])
        for feature_idx in active_indices(rhs_mask):
            coef = float(best.beta[rhs_idx, feature_idx])
            if abs(coef) < 1.0e-4:
                continue
            expr = render_feature(best.asts[feature_idx], best.leaf_mask[feature_idx], best.leaf_words[feature_idx])
            print(f"    f{feature_idx:02d}: coef={coef:+.8g} expr={expr}")

    print()
    print("map elites by union active count")
    for active_count in sorted(elites):
        elite = elites[active_count]
        print(f"  active={active_count:02d} mean_validation_mse={elite.mse:.8e} robustness={elite.robustness}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--cohorts-per-generation", type=int, default=8)
    parser.add_argument("--settings", type=int, default=512)
    parser.add_argument("--train-rows", type=int, default=16384)
    parser.add_argument("--validation-rows", type=int, default=8192)
    parser.add_argument("--modes", type=int, default=7)
    parser.add_argument("--feature-leaf-probability", type=float, default=0.82)
    parser.add_argument("--sparsity-penalty", type=float, default=2.0e-8)
    parser.add_argument("--feed", type=float, default=0.04)
    parser.add_argument("--kill", type=float, default=0.06)
    parser.add_argument("--diffusivity-u", type=float, default=0.08)
    parser.add_argument("--diffusivity-v", type=float, default=0.04)
    parser.add_argument("--broad-ops", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is not available")
    if args.cohorts_per_generation <= 0 or args.settings <= 0 or args.generations <= 0:
        raise ValueError("generations, cohorts-per-generation, and settings must be positive")

    rng = np.random.default_rng(args.seed)
    train_primitive, train_targets = make_fourier_pair_data(
        args.seed + 101,
        args.train_rows,
        args.modes,
        feed=args.feed,
        kill=args.kill,
        diffusivity_u=args.diffusivity_u,
        diffusivity_v=args.diffusivity_v,
    )
    validation_primitive, validation_targets = make_fourier_pair_data(
        args.seed + 202,
        args.validation_rows,
        args.modes,
        feed=args.feed,
        kill=args.kill,
        diffusivity_u=args.diffusivity_u,
        diffusivity_v=args.diffusivity_v,
    )

    device = torch.device("cuda")
    train_primitive_t = torch.tensor(train_primitive, device=device)
    train_targets_t = torch.tensor(train_targets, device=device)
    validation_primitive_t = torch.tensor(validation_primitive, device=device)
    validation_targets_t = torch.tensor(validation_targets, device=device)
    alphas_t = torch.tensor([1.0e-8, 1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4], device=device, dtype=torch.float32)
    thresholds_t = torch.tensor([1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2], device=device, dtype=torch.float32)

    elites: dict[int, Elite] = {}
    print("synthetic Gray-Scott reaction-diffusion MAP-Elites dry run")
    print(f"  hidden target used to generate data: {target_description(args.feed, args.kill, args.diffusivity_u, args.diffusivity_v)}")
    print(
        f"  generations={args.generations} cohorts/generation={args.cohorts_per_generation} "
        f"settings/cohort={args.settings}"
    )
    print(
        f"  train_rows={args.train_rows} validation_rows={args.validation_rows} modes={args.modes} "
        f"broad_ops={args.broad_ops}"
    )

    total_compile_ms = 0.0
    total_gpu_ms = 0.0
    total_inherited = 0
    with CudaMapElitesSettingsGenerator(torch, device) as settings_generator:
        print(f"  settings_jit_ms={settings_generator.jit_ms:.2f}")
        for generation in range(args.generations):
            candidates = []
            for cohort_idx in range(args.cohorts_per_generation):
                if generation == 0 or cohort_idx == 0 or not elites:
                    candidates.append(random_candidate(rng, args.broad_ops))
                else:
                    candidates.append(
                        merged_candidate(
                            rng,
                            list(elites.values()),
                            args.broad_ops,
                        )
                    )
            inherited = sum(candidate.inherited_terms for candidate in candidates)
            total_inherited += inherited
            results, compile_ms, gpu_ms = evaluate_candidates(
                candidates,
                settings_generator=settings_generator,
                settings=args.settings,
                generation=generation,
                seed=args.seed,
                feature_leaf_probability=args.feature_leaf_probability,
                train_primitive_t=train_primitive_t,
                train_targets_t=train_targets_t,
                validation_primitive_t=validation_primitive_t,
                validation_targets_t=validation_targets_t,
                alphas_t=alphas_t,
                thresholds_t=thresholds_t,
            )
            total_compile_ms += compile_ms
            total_gpu_ms += gpu_ms
            best_mse, best_active_count = update_elites(
                elites,
                results,
                generation=generation,
                sparsity_penalty=args.sparsity_penalty,
            )
            print(
                f"generation={generation} inherited_terms={inherited} compile_ms={compile_ms:.2f} "
                f"gpu_ms={gpu_ms:.2f} best_mean_validation_mse={best_mse:.8e} "
                f"best_union_active_count={best_active_count}",
                flush=True,
            )

    total_candidates = args.generations * args.cohorts_per_generation * args.settings * int(alphas_t.shape[0])
    print(
        f"total_compile_ms={total_compile_ms:.2f} total_gpu_ms={total_gpu_ms:.2f} "
        f"total_solve_candidates={total_candidates} inherited_terms={total_inherited}"
    )
    print_best_elites(elites)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
