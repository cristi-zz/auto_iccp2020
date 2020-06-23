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
    itemizer = dataprocessing.HeatingItemiser(future_len, transformer=True)
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


def test_Itemizer_encoder_target_offseting():
    F = 3
    N = 60
    future_len = 25
    sample = np.random.rand(N, F)
    sample[:, 2] *= 10
    itemizer = dataprocessing.HeatingItemiser(future_len, command_column=1, temperature_column=2)
    out_tuple = itemizer.encodes(sample)
    x12, y = out_tuple
    y = y.numpy()
    actual_target = sample[-future_len:, 2]
    delta = np.abs(actual_target - y)
    assert np.std(delta) < 0.1 # All offsets are the same


def test_Itemizer_encoder_decoder_throgh_learner():
    from fastai2.data.all import TfmdLists, DataLoaders
    from fastai2.learner import Learner, MSELossFlat
    import network_definitions
    F = 3
    N = 60
    future_len = 10
    past_len = N - future_len
    itemizer = dataprocessing.HeatingItemiser(future_len, command_column=1, temperature_column=2)
    samples = []
    for k in range(100):
        sample = np.random.rand(N, F)
        sample[:, 2] = 10 * sample[:, 2] + 5
        samples.append(sample)

    tls_train = TfmdLists(samples, itemizer)
    dloader = DataLoaders.from_dsets(tls_train, tls_train, bs=8).cuda()
    model_Linear= network_definitions.LinearModel(in_features=F, past_steps=past_len, future_steps=future_len).cuda()
    learner_Linear = Learner(dloader, model_Linear, loss_func=MSELossFlat())
    learner_Linear.fit_one_cycle(1)
    decoded_output, _, model_output = learner_Linear.predict(sample, with_input=False)
    assert np.average(decoded_output) > np.average(model_output), "the model's output is not decoded right"

