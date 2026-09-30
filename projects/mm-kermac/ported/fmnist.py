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
import numpy as np

import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets

from numpy.linalg import solve

def load_fmnist(data_root, n_train, n_test):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.FashionMNIST(root=data_root, train=True, download=True, transform=transform)
    n_train_ = min(n_train, len(train_dataset))
    train_indices = np.random.choice(len(train_dataset), n_train_, replace=False)
    X_train = train_dataset.data[train_indices].reshape(-1, 28*28)/255.0
    Y_train = F.one_hot(train_dataset.targets[train_indices].long())
    Y_train = Y_train.float()
    
    test_dataset = datasets.FashionMNIST(root=data_root, train=False, download=True, transform=transform)
    n_test_ = min(n_test, len(test_dataset))
    test_indices = np.random.choice(len(test_dataset), n_test_, replace=False)
    X_test = test_dataset.data[test_indices].reshape(-1,28*28)/255.0
    Y_test = F.one_hot(test_dataset.targets[test_indices].long())
    Y_test = Y_test.float()

    return X_train, X_test, Y_train, Y_test

import os
def save(filepath, arr):
    directory_path = os.path.dirname(filepath)
    os.makedirs(directory_path, exist_ok=True)
    sizes=np.array(arr.shape)

    dtype_size = np.array([arr.dtype.itemsize])
    desc = np.concatenate([dtype_size, sizes])
    desc.tofile(filepath+'.desc')
    if (isinstance(arr, np.ndarray)):
        arr.tofile(filepath+'.dat')
    else:
        arr.numpy().tofile(filepath+'.dat')

import torch
import kernels
import  time
if __name__ == '__main__':

    np.set_printoptions(threshold=500)
    np.set_printoptions(linewidth=100)
    np.set_printoptions(edgeitems=30)
    torch.set_printoptions(edgeitems=30)
    torch.set_printoptions(threshold=1000)
    torch.set_printoptions(linewidth=2000)

    start = time.time()

    np.random.seed(1)

    data_path = "./__data__/fmnist"
    n_train = 100_000
    n_test = 100_000
    X_train, X_test, Y_train, Y_test = load_fmnist(data_path, n_train, n_test)
    print(X_train[1])
    # exit()
    print(X_train.dtype)
    print(Y_train.dtype)

    end = time.time()
    print(end - start)

    save('data/fmnist/X_train', X_train)
    save('data/fmnist/X_test', X_test)
    save('data/fmnist/Y_train', Y_train)
    save('data/fmnist/Y_test', Y_test)

    X_train_view = X_train[:10]
    Y_train_view = Y_train[:10]
    print(X_train_view)

    print(X_train_view.shape)
    print(X_train_view @ X_train_view.T)


    kernel_matrix = kernels.laplacian(X_train_view, X_train_view, 10.0)
    print(kernel_matrix)

    K_train = kernels.laplacian(X_train_view, X_train_view, 10.0).numpy()
    reg = 1e-3
    sol = solve(K_train + reg * np.eye(len(K_train)), Y_train_view)
    print(sol)

    print(Y_test[0])
    print(Y_train[:10])

    print(X_train.shape)
    print(Y_train.shape)