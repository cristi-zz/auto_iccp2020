import torch
import torch.nn
import numpy as np
from fastai.data.transforms import TfmdLists, DataLoaders
import fastai.losses
import fastai.learner
import fastai.optimizer
import fastai.callback.schedule

import network_definitions


def test_LogGauss_1_kernel_per_feature():
    no_batch = 16
    no_feats = 4
    input_ts = 201
    output_ts = 101
    one_kernel_per_feature=True

    filter = network_definitions.LogGauss(no_feats, input_ts, output_ts, one_kernel_per_feature)
    some_input = torch.rand(no_batch, no_feats, input_ts, requires_grad=True)
    some_output = filter(some_input)
    loss = torch.mean(some_output)
    loss.backward()
    assert some_input.grad.shape == some_input.shape, "The gradient of the input is missing"
    assert some_output.shape == some_input.shape
    assert filter.translate.shape ==  (no_feats,)
    assert filter.scale.shape == (no_feats,)


def test_LogGauss_1_kernel_per_cell_same_len():
    no_batch = 16
    no_feats = 4
    input_ts = 201
    output_ts = 101
    one_kernel_per_feature=False

    filter = network_definitions.LogGauss(no_feats, input_ts, output_ts, one_kernel_per_feature)
    some_input = torch.rand(no_batch, no_feats, input_ts, requires_grad=True)
    some_output = filter(some_input)
    loss = torch.mean(some_output)
    loss.backward()
    assert some_input.grad.shape == some_input.shape, "The gradient of the input is missing"
    assert some_output.shape == (no_batch, no_feats, output_ts)
    assert filter.translate.shape == (no_feats * output_ts,)
    assert filter.scale.shape == (no_feats * output_ts,)
    assert filter.bn.act.weight.shape == (no_feats * output_ts, )



def test_Gabor_1_kernel_per_feature():
    no_batch = 16
    no_feats = 4
    input_ts = 201
    output_ts = 101
    one_kernel_per_feature=True

    filter = network_definitions.GaborFilter(no_feats, input_ts, output_ts, one_kernel_per_feature)
    some_input = torch.rand(no_batch, no_feats, input_ts, requires_grad=True)
    some_output, _ = filter(some_input)
    loss = torch.mean(some_output)
    loss.backward()
    assert some_input.grad.shape == some_input.shape, "The gradient of the input is missing"
    assert some_output.shape == some_input.shape
    assert filter.frequency.shape == (no_feats,)
    assert filter.mu.shape == (no_feats,)


def test_Gabor_1_kernel_per_cell_same_len():
    no_batch = 16
    no_feats = 4
    input_ts = 201
    output_ts = 101
    one_kernel_per_feature=False

    filter = network_definitions.GaborFilter(no_feats, input_ts, output_ts, one_kernel_per_feature)
    some_input = torch.rand(no_batch, no_feats, input_ts, requires_grad=True)
    some_output, _ = filter(some_input)
    loss = torch.mean(some_output)
    loss.backward()
    assert some_input.grad.shape == some_input.shape, "The gradient of the input is missing"
    assert some_output.shape == (no_batch, no_feats, output_ts)
    assert filter.frequency.shape == (no_feats * output_ts,)
    assert filter.mu.shape == (no_feats * output_ts,)
    assert filter.bn_mag.act.weight.shape == (no_feats * output_ts, )
    assert filter.bn_orient.act.weight.shape == (no_feats * output_ts,)


def test_IdentityTransform_1_kernel_per_feature():
    no_batch = 16
    no_feats = 4
    input_ts = 201
    output_ts = 101
    one_kernel_per_feature=True

    filter = network_definitions.IdentityTransform(no_feats, input_ts, output_ts, one_kernel_per_feature)
    some_input = torch.rand(no_batch, no_feats, input_ts, requires_grad=True)
    some_output = filter(some_input)
    loss = torch.mean(some_output)
    loss.backward()
    assert some_input.grad.shape == some_input.shape, "The gradient of the input is missing"
    assert some_output.shape == some_input.shape


def test_IdentityTransform_1_kernel_per_cell_same_len():
    no_batch = 16
    no_feats = 4
    input_ts = 201
    output_ts = 101
    one_kernel_per_feature = False

    filter = network_definitions.IdentityTransform(no_feats, input_ts, output_ts, one_kernel_per_feature)
    some_input = torch.rand(no_batch, no_feats, input_ts, requires_grad=True)
    some_output = filter(some_input)
    loss = torch.mean(some_output)
    loss.backward()
    assert some_input.grad.shape == some_input.shape, "The gradient of the input is missing"
    assert some_output.shape == (no_batch, no_feats, output_ts)




def test_Gauss_1_kernel_per_feature():
    no_batch = 16
    no_feats = 4
    input_ts = 201
    output_ts = 101
    one_kernel_per_feature=True

    filter = network_definitions.GaussFilter(no_feats, input_ts, output_ts, one_kernel_per_feature)
    some_input = torch.rand(no_batch, no_feats, input_ts, requires_grad=True)
    some_output = filter(some_input)
    loss = torch.mean(some_output)
    loss.backward()
    assert some_input.grad.shape == some_input.shape, "The gradient of the input is missing"
    assert some_output.shape == some_input.shape
    assert filter.sigma.shape ==  (no_feats,)
    assert filter.mu.shape == (no_feats,)


def test_Gauss_1_kernel_per_cell_same_len():
    no_batch = 16
    no_feats = 4
    input_ts = 201
    output_ts = 101
    one_kernel_per_feature=False

    filter = network_definitions.GaussFilter(no_feats, input_ts, output_ts, one_kernel_per_feature)
    some_input = torch.rand(no_batch, no_feats, input_ts, requires_grad=True)
    some_output = filter(some_input)
    loss = torch.mean(some_output)
    loss.backward()
    assert some_input.grad.shape == some_input.shape, "The gradient of the input is missing"
    assert some_output.shape == (no_batch, no_feats, output_ts)
    assert filter.sigma.shape == (no_feats * output_ts,)
    assert filter.mu.shape == (no_feats * output_ts,)
    assert filter.bn.act.weight.shape == (no_feats * output_ts, )



def test_AffineTransform_1_kernel_per_feature():
    no_batch = 16
    no_feats = 4
    input_ts = 201
    output_ts = 101
    one_kernel_per_feature=True

    filter = network_definitions.AffineTransform(no_feats, input_ts, output_ts, one_kernel_per_feature)
    some_input = torch.rand(no_batch, no_feats, input_ts, requires_grad=True)
    some_output = filter(some_input)
    loss = torch.mean(some_output)
    loss.backward()
    assert some_input.grad.shape == some_input.shape, "The gradient of the input is missing"
    assert some_output.shape == some_input.shape
    assert filter.translate.shape == (no_feats,)
    assert filter.scale.shape == (no_feats,)


def test_AffineTransform_1_kernel_per_cell_same_len():
    no_batch = 16
    no_feats = 4
    input_ts = 201
    output_ts = 101
    one_kernel_per_feature=False

    filter = network_definitions.AffineTransform(no_feats, input_ts, output_ts, one_kernel_per_feature)
    some_input = torch.rand(no_batch, no_feats, input_ts, requires_grad=True)
    some_output = filter(some_input)
    loss = torch.mean(some_output)
    loss.backward()
    assert some_input.grad.shape == some_input.shape, "The gradient of the input is missing"
    assert some_output.shape == (no_batch, no_feats, output_ts)
    assert filter.translate.shape == (no_feats, )
    assert filter.scale.shape == (no_feats, )
    assert filter.bn.act.weight.shape == (no_feats, )


def test_batch_vs_instancenorm():
    no_batch = 16
    no_feats = 4
    input_ts = 128
    some_input = torch.rand(no_batch, no_feats, input_ts, requires_grad=True)
    bn = torch.nn.BatchNorm1d(no_feats)
    ibn = torch.nn.InstanceNorm1d(no_feats, affine=True, track_running_stats=True)
    bno = bn(some_input)
    ibno = ibn(some_input)

    assert not torch.allclose(bno, ibno)


def test_LogGauss_output_dist():
    bs = 8
    nf = 20
    tlen = 400
    rnd_input = torch.randn(bs, nf, tlen)
    print(f"Input shape: {rnd_input.shape}")
    layer = network_definitions.LogGauss(nf, tlen, None, False)
    out = layer.forward(rnd_input)
    mean_out = out.mean()
    std_out = out.std()
    assert (mean_out - 0).abs() < 0.1, "Mean activations is not zero"
    assert (std_out - 1).abs() < 0.1, "Std of activations is not 1"


def test_AffineTransform_output_dist():
    bs = 8
    nf = 20
    tlen = 400
    rnd_input = torch.randn(bs, nf, tlen)
    print(f"Input shape: {rnd_input.shape}")
    layer = network_definitions.LogGauss(nf, tlen, None, False)
    out = layer.forward(rnd_input)
    mean_out = out.mean()
    std_out = out.std()
    assert (mean_out - 0).abs() < 0.1, "Mean activations is not zero"
    assert (std_out - 1).abs() < 0.1, "Std of activations is not 1"


def test_GaborFilter_output_dist():
    bs = 8
    nf = 20
    tlen = 400
    rnd_input = torch.randn(bs, nf, tlen)
    print(f"Input shape: {rnd_input.shape}")
    layer = network_definitions.GaborFilter(nf, tlen, None, False, False)
    out , _= layer.forward(rnd_input)
    std_out = out.std()
    assert (std_out - 1).abs() < 0.15, "Std of activations is not 1"


def test_GaborFilter_output_distribution():
    bs = 8
    nf = 20
    tlen = 400
    rnd_input = torch.randn(bs, nf, tlen)
    print(f"Input shape: {rnd_input.shape}")
    layer = network_definitions.GaborFilter(nf, tlen, None, one_kernel_per_feature=False)
    out, orient = layer.forward(rnd_input)
    mean_out = out.mean()
    std_out = out.std()
    assert (mean_out - 0).abs() < 0.1, "Mean activations is not zero"
    assert (std_out - 1).abs() < 0.1, "Std of activations is not 1"


def test_GaborFilter_output_the_orientation():
    bs = 8
    nf = 20
    tlen = 400
    rnd_input = torch.randn(bs, nf, tlen)
    print(f"Input shape: {rnd_input.shape}")
    layer = network_definitions.GaborFilter(nf, tlen, None, one_kernel_per_feature=False)
    out, orient = layer.forward(rnd_input)
    assert orient.shape == out.shape


def test_GaussFilter_output_dist_imag():
    bs = 8
    nf = 20
    tlen = 400
    rnd_input = torch.randn(bs, nf, tlen)
    print(f"Input shape: {rnd_input.shape}")
    layer = network_definitions.GaussFilter(nf, tlen, None, True)
    out = layer.forward(rnd_input)
    mean_out = out.mean()
    std_out = out.std()
    assert (mean_out - 0).abs() < 0.1, "Mean activations is not zero"
    assert (std_out - 1).abs() < 0.1, "Std of activations is not 1"


def test_CausalTemporalConv():
    inp_len = 100
    out_len = 40
    ran_imp = torch.rand((1, 5, inp_len)) # Batch size, features, time len
    filter = network_definitions.CausalTemporalConv(5, inp_len, out_len, one_kernel_per_feature=False)
    out = filter.forward(ran_imp)
    assert out.shape[2] == out_len


def test_CausalTemporalConv_same_size():
    inp_len = 30
    out_len = 30
    ran_imp = torch.rand((1, 5, inp_len)) # Batch size, features, time len
    filter = network_definitions.CausalTemporalConv(5, inp_len, out_len, one_kernel_per_feature=False)
    out = filter.forward(ran_imp)
    assert out.shape[2] == out_len


def test_selectable_bn():
    bs = 8
    nf = 20
    tlen = 400
    rnd_input = torch.randn(bs, nf, tlen)
    print(f"Input shape: {rnd_input.shape}")
    bn = network_definitions.SelectableBN(nf, 2)
    out=bn(rnd_input)
    assert out.shape == rnd_input.shape


def test_ConstructDelayNet_forward():
    lo_hi_layers = ["AffineTransform", "LogGauss", "GaborFilter", "IdentityTransform", "GaussFilter"]
    temporal_agg = ["AffineTransform", "LogGauss",                "IdentityTransform", "GaussFilter", "CausalTemporalConv"]
    N = 512
    F = 4
    T1 = 50
    T2 = 50
    rnd_data = np.random.rand(N, (T1 + T2), F).tolist()
    rnd_data = [np.array(a) for a in rnd_data]
    itemizer = network_definitions.FeatureItemizer(T2, list(range(F)), [2], [3], [], 0.1)
    tls = TfmdLists(rnd_data, [itemizer])
    dloaders = DataLoaders.from_dsets(tls, tls, bs=16, drop_last=True, shuffle=True, num_workers=0, device=torch.device('cuda'))
    t_agg = "CausalTemporalConv"
    high = "GaussFilter"
    for low in lo_hi_layers:
        model = network_definitions.ConstructDelayNet(F, 1, 1, T1, T2, low, 1, True, 1, 1, 4, t_agg,
                                        high, 1, True, 1, 1)
        learner = fastai.learner.Learner(dloaders, model, opt_func=fastai.optimizer.Adam,
                                         loss_func=fastai.losses.L1LossFlat())
        learner.fit_one_cycle(2, 1e-5)
        train_loss = learner.recorder.values[0][0]
        assert not np.isnan(train_loss), f"NaN in train, for {low}-{t_agg}-{high}"

    low = "GaussFilter"
    for t_agg in temporal_agg:
        model = network_definitions.ConstructDelayNet(F, 1, 1, T1, T2, low, 1, True, 1, 1, 4, t_agg,
                                        high, 1, True, 1, 1)
        learner = fastai.learner.Learner(dloaders, model, opt_func=fastai.optimizer.Adam,
                                         loss_func=fastai.losses.L1LossFlat())
        learner.fit_one_cycle(2, 1e-5)
        train_loss = learner.recorder.values[0][0]
        assert not np.isnan(train_loss), f"NaN in train, for {low}-{t_agg}-{high}"



def test_run_data_AttentionModel():
    bs = 32
    in_features = 14
    past_steps = 20
    future_steps = 30
    t1 = torch.rand(bs, in_features, past_steps)
    t2 = torch.rand(bs, future_steps)
    model = network_definitions.AttentionModel(in_features, past_steps,future_steps, hidden_size=64, num_layers=3).cpu()
    model.eval()
    out = model((t1, t2))
    assert out is not None
    assert out.shape == (bs, future_steps)

