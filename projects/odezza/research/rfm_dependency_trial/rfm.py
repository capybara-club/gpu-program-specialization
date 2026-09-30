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
"""Gaussian Recursive Feature Machine with explicit AGOP metric updates.

All calculations use float64. Exact dense kernels, no sample cap or approximation.
Ridge is added to the kernel diagonal (not multiplied by sample count).
"""
import time
import torch


def kernel(x, centers, metric, bandwidth):
    xm = x @ metric
    d2 = (xm * x).sum(1)[:, None] + ((centers @ metric) * centers).sum(1)[None, :] - 2 * xm @ centers.T
    return torch.exp(-d2.clamp_min(0) / (2 * bandwidth ** 2))


def fit(x, y, metric, bandwidth, ridge):
    k = kernel(x, x, metric, bandwidth)
    k.diagonal().add_(ridge)
    return torch.linalg.solve(k, y)


def predict(x, centers, alpha, metric, bandwidth):
    return kernel(x, centers, metric, bandwidth) @ alpha


def gradient(x, centers, alpha, metric, bandwidth):
    w = kernel(x, centers, metric, bandwidth) * alpha[None, :]
    return ((w @ centers - w.sum(1)[:, None] * x) @ metric) / bandwidth ** 2


def agop(x, alpha, metric, bandwidth):
    g = gradient(x, x, alpha, metric, bandwidth)
    return g.T @ g / len(x)


def update(m, diagonal, floor):
    if diagonal: m = torch.diag(m.diagonal())
    trace = m.trace()
    if trace.item() < 1e-24: return torch.eye(len(m), device=m.device, dtype=m.dtype)
    m = m * len(m) / trace
    return (1 - floor) * m + floor * torch.eye(len(m), device=m.device, dtype=m.dtype)


def learn(x, y, xv, yv, cfg):
    started = time.perf_counter()
    n = x.shape[1]
    identity = torch.eye(n, device=x.device, dtype=x.dtype)
    bandwidth = cfg['bandwidth_factor'] * n ** 0.5
    alpha0 = fit(x, y, identity, bandwidth, cfg['ridge'])
    m0 = agop(x, alpha0, identity, bandwidth)
    if x.is_cuda:
        done = torch.cuda.Event(); done.record(); done.synchronize()
    shared_seconds = time.perf_counter() - started
    results = {}
    for mode in ('fixed', 'diagonal', 'full'):
        mode_started = time.perf_counter()
        metric, alpha, relevance = identity, alpha0, m0
        for _ in range(0 if mode == 'fixed' else cfg['rfm_updates']):
            metric = update(relevance, mode == 'diagonal', cfg['metric_floor'])
            alpha = fit(x, y, metric, bandwidth, cfg['ridge'])
            relevance = agop(x, alpha, metric, bandwidth)
        prediction = predict(xv, x, alpha, metric, bandwidth)
        results[mode] = {'matrix': relevance.cpu().tolist(),
                         'validation_mse_scaled': ((prediction - yv) ** 2).mean().item()}
        results[mode]['incremental_seconds'] = time.perf_counter() - mode_started
        results[mode]['shared_initial_seconds'] = shared_seconds
    return results
