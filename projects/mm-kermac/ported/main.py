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
import torch

import numpy as np

import torch.nn.functional as F
import torchvision.transforms as transforms

from torchvision import datasets

EPSILON = 1e-12

def euclidean(samples: torch.Tensor, centers: torch.Tensor,
              squared: bool = True) -> torch.Tensor:
    '''Calculate the pointwise distance.

    Args:
        samples: of shape (n_sample, n_feature).
        centers: of shape (n_center, n_feature).
        squared: boolean.

    Returns:
        pointwise distances (n_sample, n_center).
    '''
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

def laplacian(samples: torch.Tensor, centers: torch.Tensor,
              bandwidth: float) -> torch.Tensor:
    '''Laplacian kernel.

    Args:
        samples: of shape (n_sample, n_feature).
        centers: of shape (n_center, n_feature).
        bandwidth: kernel bandwidth.

    Returns:
        kernel matrix of shape (n_sample, n_center).
    '''
    assert bandwidth > 0
    kernel_mat = euclidean(samples, centers, squared=False)
    kernel_mat.clamp_(min=0)
    gamma = 1. / bandwidth
    kernel_mat.mul_(-gamma)
    kernel_mat.exp_()
    return kernel_mat

def load_fmnist(data_root, n_train, n_test, shuffle=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.FashionMNIST(root=data_root, train=True, download=True, transform=transform)
    n_train_ = min(n_train, len(train_dataset))
    if shuffle:
        train_indices = np.random.choice(len(train_dataset), n_train_, replace=False)
    else:
        train_indices = list(range(n_train_))
    X_train = train_dataset.data[train_indices].reshape(-1, 28*28)/255.0
    Y_train = F.one_hot(train_dataset.targets[train_indices].long())
    
    test_dataset = datasets.FashionMNIST(root=data_root, train=False, download=True, transform=transform)
    n_test_ = min(n_test, len(test_dataset))
    if shuffle:
        test_indices = np.random.choice(len(test_dataset), n_test_, replace=False)
    else:
        test_indices = list(range(n_test_))

    X_test = test_dataset.data[test_indices].reshape(-1,28*28)/255.0
    Y_test = F.one_hot(test_dataset.targets[test_indices].long())

    return X_train, X_test, Y_train, Y_test

x_train, x_test, y_train, y_test = load_fmnist('', 60000, 10000, False)
x_train = x_train.cuda()
print(x_train)
print(x_train.shape)
print(x_train @ x_train.T)

print(laplacian(x_train, x_train, 10.0))