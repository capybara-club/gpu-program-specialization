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
import torch
import torchvision
import torchvision.transforms as transforms

def pre_process(torchset,n_samples,num_classes=10):
    n_samples = min(n_samples, len(torchset))
    indices = list(np.random.choice(len(torchset),n_samples))

    trainset = []
    for ix in indices:
        x,y = torchset[ix]
        ohe_y = torch.zeros(num_classes)
        ohe_y[y] = 1
        trainset.append(((x/np.linalg.norm(x)).reshape(-1),ohe_y))
    return trainset

def load_svhn(data_root, n_train, n_test):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = torchvision.datasets.SVHN(root=data_root,
                                    split = "train",
                                    transform=transform,
                                    download=True)
    
    test_dataset = torchvision.datasets.SVHN(root=data_root,
                                    split = "test",
                                    transform=transform,
                                    download=True)
    trainset = pre_process(train_dataset, n_samples=n_train, num_classes=10)
    X_train = [example[0] for example in trainset]
    X_train = torch.stack(X_train)
    Y_train = [example[1] for example in trainset]
    Y_train = torch.stack(Y_train)
    testset = pre_process(test_dataset, n_samples=n_test, num_classes=10)
    X_test = [example[0] for example in testset]
    X_test = torch.stack(X_test)
    Y_test = [example[1] for example in testset]
    Y_test = torch.stack(Y_test)
    
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

import time
if __name__ == '__main__':
    np.set_printoptions(threshold=500)
    np.set_printoptions(linewidth=100)
    np.set_printoptions(edgeitems=30)
    torch.set_printoptions(edgeitems=30)
    torch.set_printoptions(threshold=1000)
    torch.set_printoptions(linewidth=2000)

    start = time.time()

    np.random.seed(1)

    data_path = "./__data__/svhn"
    n_train = 100_000
    n_test = 100_000
    # X_train, X_test, Y_train, Y_test = load_svhn(data_path, n_train, n_test)
    X_train, X_test, Y_train, Y_test = load_svhn(data_path, n_train, n_test)

    print(X_train[1])
    # exit()
    print(X_train.dtype)
    print(Y_train.dtype)

    print(X_train.shape)
    print(Y_train.shape)

    save('data/svhn/X_train', X_train)
    save('data/svhn/X_test', X_test)
    save('data/svhn/Y_train', Y_train)
    save('data/svhn/Y_test', Y_test)
