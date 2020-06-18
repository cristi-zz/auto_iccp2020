import torch
import dataprocessing
import numpy as np


def test_Itemizer_regular():
    F = 14
    N = 60
    future_len = 25
    sample = np.random.rand(N, F)
    itemizer = dataprocessing.HeatingItemiser(future_len)
    out_tuple = itemizer.encodes(sample)
    x12, y = out_tuple
    x1, x2 = x12
    assert type(x1) == torch.Tensor
    assert type(x2) == torch.Tensor
    assert type(y) == torch.Tensor
    assert x1.size(0) == F
    assert x1.size(1) == N - future_len
    assert x2.size(0) == future_len
    assert y.size(0) == future_len
    assert np.allclose(x2.numpy(), sample[-future_len:, itemizer.command_column])


def test_Itemizer_transformer():
    F = 14
    N = 60
    future_len = 25
    sample = np.random.rand(N, F)
    itemizer = dataprocessing.HeatingItemizerTrans(future_len)
    out_tuple = itemizer.encodes(sample)
    x12, y = out_tuple
    x1, x2 = x12
    assert type(x1) == torch.Tensor
    assert type(x2) == torch.Tensor
    assert type(y) == torch.Tensor
    assert x1.size(0) == F
    assert x1.size(1) == N - future_len
    assert x2.size(1) == future_len
    assert x2.size(0) == 2
    assert y.size(0) == future_len
    assert np.allclose(x2[0,:].numpy(), sample[-future_len:, itemizer.command_column])
