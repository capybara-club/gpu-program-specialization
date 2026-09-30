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
import argparse
import os
import time

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:
    torch = None
    _TORCH_AVAILABLE = False


def torch_available():
    return _TORCH_AVAILABLE


def is_torch_tensor(value):
    return _TORCH_AVAILABLE and torch.is_tensor(value)


def euclidean_distances(samples, centers, squared=True):
    if is_torch_tensor(samples):
        samples_norm = torch.sum(samples**2, dim=1, keepdim=True)
        if samples is centers:
            centers_norm = samples_norm
        else:
            centers_norm = torch.sum(centers**2, dim=1, keepdim=True)
        centers_norm = torch.reshape(centers_norm, (1, -1))

        distances = samples.mm(torch.t(centers))
        distances.mul_(-2)
        distances.add_(samples_norm)
        distances.add_(centers_norm)

        if not squared:
            distances.clamp_(min=0)
            distances.sqrt_()

        return distances

    same_inputs = samples is centers
    samples_np = np.asarray(samples, dtype=np.float32)
    centers_np = samples_np if same_inputs else np.asarray(centers, dtype=np.float32)

    samples_norm = np.sum(samples_np * samples_np, axis=1, keepdims=True)
    if same_inputs:
        centers_norm = samples_norm
    else:
        centers_norm = np.sum(centers_np * centers_np, axis=1, keepdims=True)
    centers_norm = centers_norm.reshape(1, -1)

    distances = samples_np @ centers_np.T
    distances = distances * -2.0
    distances = distances + samples_norm + centers_norm

    if not squared:
        distances = np.sqrt(np.maximum(distances, 0.0))

    return distances


def euclidean_distances_M(samples, centers, M, squared=True):
    if is_torch_tensor(samples):
        samples_norm = (samples @ M) * samples
        samples_norm = torch.sum(samples_norm, dim=1, keepdim=True)

        if samples is centers:
            centers_norm = samples_norm
        else:
            centers_norm = (centers @ M) * centers
            centers_norm = torch.sum(centers_norm, dim=1, keepdim=True)

        centers_norm = torch.reshape(centers_norm, (1, -1))

        distances = samples.mm(M @ torch.t(centers))
        distances.mul_(-2)
        distances.add_(samples_norm)
        distances.add_(centers_norm)

        if not squared:
            distances.clamp_(min=0)
            distances.sqrt_()

        return distances

    same_inputs = samples is centers
    samples_np = np.asarray(samples, dtype=np.float32)
    centers_np = samples_np if same_inputs else np.asarray(centers, dtype=np.float32)
    M_np = np.asarray(M, dtype=np.float32)

    samples_norm = (samples_np @ M_np) * samples_np
    samples_norm = np.sum(samples_norm, axis=1, keepdims=True)

    if same_inputs:
        centers_norm = samples_norm
    else:
        centers_norm = (centers_np @ M_np) * centers_np
        centers_norm = np.sum(centers_norm, axis=1, keepdims=True)

    centers_norm = centers_norm.reshape(1, -1)

    distances = samples_np @ (M_np @ centers_np.T)
    distances = distances * -2.0
    distances = distances + samples_norm + centers_norm

    if not squared:
        distances = np.sqrt(np.maximum(distances, 0.0))

    return distances


def gaussian(samples, centers, bandwidth):
    assert bandwidth > 0
    kernel_mat = euclidean_distances(samples, centers)
    gamma = 1.0 / (2.0 * bandwidth**2)
    if is_torch_tensor(kernel_mat):
        kernel_mat.clamp_(min=0)
        kernel_mat.mul_(-gamma)
        kernel_mat.exp_()
    else:
        kernel_mat = np.maximum(kernel_mat, 0.0)
        kernel_mat = np.exp(kernel_mat * -gamma)

    return kernel_mat


def laplacian(samples, centers, bandwidth):
    assert bandwidth > 0
    kernel_mat = euclidean_distances(samples, centers, squared=False)
    gamma = 1.0 / bandwidth
    if is_torch_tensor(kernel_mat):
        kernel_mat.clamp_(min=0)
        kernel_mat.mul_(-gamma)
        kernel_mat.exp_()
    else:
        kernel_mat = np.maximum(kernel_mat, 0.0)
        kernel_mat = np.exp(kernel_mat * -gamma)
    return kernel_mat


def laplacian_M(samples, centers, bandwidth, M):
    assert bandwidth > 0
    kernel_mat = euclidean_distances_M(samples, centers, M, squared=False)
    gamma = 1.0 / bandwidth
    if is_torch_tensor(kernel_mat):
        kernel_mat.clamp_(min=0)
        kernel_mat.mul_(-gamma)
        kernel_mat.exp_()
    else:
        kernel_mat = np.maximum(kernel_mat, 0.0)
        kernel_mat = np.exp(kernel_mat * -gamma)
    return kernel_mat


def laplace_kernel_M(pair1, pair2, bandwidth, M, grad=False):
    K = laplacian_M(pair1, pair2, bandwidth, M)
    if not grad:
        return K

    dist = euclidean_distances_M(pair1, pair2, M, squared=False)
    if is_torch_tensor(dist):
        dist = torch.where(dist < 1e-10, torch.zeros(1, device=pair1.device), dist)
        K = K / dist
        K[K == float("Inf")] = 0.0
    else:
        mask = dist < 1e-10
        K = np.divide(K, dist, out=np.zeros_like(K), where=~mask)
        K[~np.isfinite(K)] = 0.0

    return K


def get_grads(X, sol, bandwidth, P, K=None, X_M=None, return_intermediates=False):
    if is_torch_tensor(X):
        if K is None:
            K = laplace_kernel_M(X, X, bandwidth, P, grad=True)
        if X_M is None:
            X_M = X @ P

        step2 = torch.einsum("mn,nc,nd->mcd", K, sol, X_M)
        step3 = torch.einsum("mn,nc,md->mcd", K, sol, X_M)

        G = (step2 - step3) * (-1.0 / bandwidth)
        M = torch.einsum("mcd,mce->ed", G, G) / len(G)

        if return_intermediates:
            return M, K, X_M
        return M

    X_np = np.asarray(X, dtype=np.float32)
    P_np = np.asarray(P, dtype=np.float32)
    if K is None:
        K = laplace_kernel_M(X_np, X_np, bandwidth, P_np, grad=True)
    if X_M is None:
        X_M = X_np @ P_np

    step2 = np.einsum("mn,nc,nd->mcd", K, sol, X_M)
    step3 = np.einsum("mn,nc,md->mcd", K, sol, X_M)

    G = (step2 - step3) * (-1.0 / bandwidth)
    M = np.einsum("mcd,mce->ed", G, G) / len(G)

    if return_intermediates:
        return M, K, X_M
    return M


def rfm(X_train, y_train, X_test, y_test, bandwidth, device, iters=3, reg=1e-3):
    n, d = X_train.shape
    if is_torch_tensor(X_train):
        M = torch.eye(d, device=device)

        for i in range(iters + 1):
            K_train = laplace_kernel_M(X_train, X_train, bandwidth, M)
            sol = torch.linalg.solve(
                K_train + reg * torch.eye(len(K_train), device=device),
                y_train,
            )
            K_test = laplace_kernel_M(X_train, X_test, bandwidth, M)
            preds = K_test.T @ sol
            mse = torch.mean(torch.square(preds - y_test)).item()
            print(f"Round {i} MSE: {mse}")
            if i == iters:
                return M, mse
            M = get_grads(X_train, sol, bandwidth, M)
        return M, mse

    X_train_np = np.asarray(X_train, dtype=np.float32)
    X_test_np = np.asarray(X_test, dtype=np.float32)
    y_train_np = np.asarray(y_train, dtype=np.float32)
    y_test_np = np.asarray(y_test, dtype=np.float32)
    M = np.eye(d, dtype=np.float32)

    for i in range(iters + 1):
        K_train = laplace_kernel_M(X_train_np, X_train_np, bandwidth, M)
        K_train = K_train + reg * np.eye(len(K_train), dtype=np.float32)
        sol = np.linalg.solve(K_train, y_train_np)
        K_test = laplace_kernel_M(X_train_np, X_test_np, bandwidth, M)
        preds = K_test.T @ sol
        mse = float(np.mean(np.square(preds - y_test_np)))
        print(f"Round {i} MSE: {mse}")
        if i == iters:
            return M, mse
        M = get_grads(X_train_np, sol, bandwidth, M)
    return M, mse


def save_tensor(filepath, arr):
    directory_path = os.path.dirname(filepath)
    if directory_path:
        os.makedirs(directory_path, exist_ok=True)

    if is_torch_tensor(arr):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr, dtype=np.float32)

    sizes = np.array(arr.shape, dtype=np.int64)
    dtype_size = np.array([arr.dtype.itemsize], dtype=np.int64)
    desc = np.concatenate([dtype_size, sizes])
    desc.tofile(filepath + ".desc")
    arr.ravel(order="F").tofile(filepath + ".dat")


def fstar(X):
    if is_torch_tensor(X):
        mask = (X[:, 5] > 0).to(X.dtype)
        return (X[:, 0] * X[:, 1] * mask).unsqueeze(1)
    X_np = np.asarray(X, dtype=np.float32)
    mask = (X_np[:, 5] > 0).astype(X_np.dtype)
    return (X_np[:, 0] * X_np[:, 1] * mask)[:, None]


def make_rng(rng_name, seed):
    if rng_name == "numpy":
        return np.random.default_rng(seed)
    if rng_name == "legacy":
        return np.random.RandomState(seed)
    if rng_name == "torch":
        if not _TORCH_AVAILABLE:
            raise ValueError("Torch RNG requested but torch is not available.")
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(seed)
        return gen
    raise ValueError(f"Unknown rng '{rng_name}'")


def sample_normal(rng, rng_name, shape, scale, use_torch):
    if rng_name in ("numpy", "legacy"):
        samples = rng.normal(loc=0.0, scale=scale, size=shape).astype(np.float32)
        if use_torch:
            return torch.from_numpy(samples)
        return samples
    if rng_name == "torch":
        if not use_torch:
            raise ValueError("Torch RNG requested but torch backend is disabled.")
        return torch.randn(shape, generator=rng, dtype=torch.float32) * scale
    raise ValueError(f"Unknown rng '{rng_name}'")


def export_snapshot(X_train, y_train, bandwidth, reg, export_dir):
    n, d = X_train.shape
    if is_torch_tensor(X_train):
        M = torch.eye(d, device=X_train.device)
        K_train = laplace_kernel_M(X_train, X_train, bandwidth, M)
        sol_nc = torch.linalg.solve(
            K_train + reg * torch.eye(len(K_train), device=X_train.device),
            y_train,
        )
        M_out, K_grad, X_M = get_grads(
            X_train, sol_nc, bandwidth, M, return_intermediates=True
        )
        diag = torch.diagonal(M_out).contiguous()
    else:
        X_train_np = np.asarray(X_train, dtype=np.float32)
        y_train_np = np.asarray(y_train, dtype=np.float32)
        M = np.eye(d, dtype=np.float32)
        K_train = laplace_kernel_M(X_train_np, X_train_np, bandwidth, M)
        K_train = K_train + reg * np.eye(len(K_train), dtype=np.float32)
        sol_nc = np.linalg.solve(K_train, y_train_np)
        M_out, K_grad, X_M = get_grads(
            X_train_np, sol_nc, bandwidth, M, return_intermediates=True
        )
        diag = np.diag(M_out)

    save_tensor(os.path.join(export_dir, "kernel_matrix"), K_grad)
    save_tensor(os.path.join(export_dir, "solution"), sol_nc)
    save_tensor(os.path.join(export_dir, "features"), X_M)
    save_tensor(os.path.join(export_dir, "feature_matrix"), M_out)
    save_tensor(os.path.join(export_dir, "feature_matrix_diag"), diag)


def parse_args():
    parser = argparse.ArgumentParser(description="Low-rank Laplace RFM")
    parser.add_argument("--n", type=int, default=4000, help="Train/test sample count.")
    parser.add_argument("--d", type=int, default=20, help="Feature dimension.")
    parser.add_argument("--iters", type=int, default=5, help="Number of RFM iterations.")
    parser.add_argument("--reg", type=float, default=1e-3, help="Kernel regularizer.")
    parser.add_argument("--bandwidth", type=float, default=10.0, help="Laplacian bandwidth.")
    parser.add_argument(
        "--rng",
        choices=["numpy", "legacy", "torch"],
        default="numpy",
        help="Pseudo RNG backend.",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="Torch device.",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default="",
        help="Directory to export kernel/solution/features/M.",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Export snapshot and skip training loop.",
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Report elapsed time and peak CUDA memory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.export_only and not args.export_dir:
        raise ValueError("--export-only requires --export-dir")

    use_torch = _TORCH_AVAILABLE
    if not use_torch:
        if args.rng == "torch":
            print("Torch not available; switching RNG to numpy.")
            args.rng = "numpy"
        if args.device == "cuda":
            print("Torch not available; forcing device to cpu.")
            args.device = "cpu"
    elif args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU.")
        args.device = "cpu"

    if args.seed is not None:
        np.random.seed(args.seed)
        if _TORCH_AVAILABLE:
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)

    rng = make_rng(args.rng, args.seed)
    X_train = sample_normal(rng, args.rng, (args.n, args.d), scale=0.5, use_torch=use_torch)
    X_test = sample_normal(rng, args.rng, (args.n, args.d), scale=0.5, use_torch=use_torch)
    y_train = fstar(X_train)
    y_test = fstar(X_test)

    if use_torch:
        device = torch.device(args.device)
        X_train = X_train.to(device)
        X_test = X_test.to(device)
        y_train = y_train.to(device)
        y_test = y_test.to(device)
    else:
        device = None
        X_train = np.asarray(X_train, dtype=np.float32)
        X_test = np.asarray(X_test, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)
        y_test = np.asarray(y_test, dtype=np.float32)

    if args.export_dir:
        export_snapshot(X_train, y_train, args.bandwidth, args.reg, args.export_dir)
        if args.export_only:
            return

    if args.timing and use_torch and device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        rfm(X_train, y_train, X_test, y_test, args.bandwidth, device, args.iters, args.reg)
        stop.record()
        stop.synchronize()
        print(f"time: {start.elapsed_time(stop)} ms")
        print(torch.cuda.max_memory_allocated())
    else:
        start = time.time()
        rfm(X_train, y_train, X_test, y_test, args.bandwidth, device, args.iters, args.reg)
        print(f"time: {(time.time() - start) * 1000.0} ms")


if __name__ == "__main__":
    main()
