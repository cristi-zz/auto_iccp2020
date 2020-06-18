import torch
import network_definitions


def test_run_data_LinearModel():
    bs = 32
    in_features = 14
    past_steps = 20
    future_steps = 30
    t1 = torch.rand(bs, in_features, past_steps)
    t2 = torch.rand(bs, future_steps)
    model = network_definitions.LinearModel(in_features, past_steps,future_steps).cpu()
    model.eval()
    out = model((t1, t2))
    assert out is not None
    assert out.shape == (bs, future_steps)


def test_run_data_CnnFcn():
    bs = 32
    in_features = 14
    past_steps = 20
    future_steps = 30
    t1 = torch.rand(bs, in_features, past_steps)
    t2 = torch.rand(bs, future_steps)
    model = network_definitions.CnnFcn(in_features, past_steps,future_steps, cnn_stack_len=2, fcn_stack_len=2,
                                       cnn_filters=8, avg_pool_len=10, fcn_ratio=0.8).cpu()
    model.eval()
    out = model((t1, t2))
    assert out is not None
    assert out.shape == (bs, future_steps)


def test_run_data_EncDec():
    bs = 32
    in_features = 14
    past_steps = 20
    future_steps = 30
    t1 = torch.rand(bs, in_features, past_steps)
    t2 = torch.rand(bs, future_steps)
    model = network_definitions.EncoderDecoder(in_features, past_steps,future_steps, hidden_size=64).cpu()
    model.eval()
    out = model((t1, t2))
    assert out is not None
    assert out.shape == (bs, future_steps)


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


def test_run_data_SimpleTransformer():
    bs = 32
    in_features = 14
    past_steps = 20
    future_steps = 30
    t1 = torch.rand(bs, in_features, past_steps)
    t2 = torch.rand(bs, future_steps)
    model = network_definitions.SimpleTransformer(in_features, past_steps,future_steps, d_model=64, dim_feedforward=128).cpu()
    model.eval()
    out = model((t1, t2))
    assert out is not None
    assert out.shape == (bs, future_steps)


def test_TransformerFull_training_loop():
    bs = 8
    in_features = 14
    target_features = 2
    past_steps = 90
    future_steps = 40
    d_model = 128
    dim_feedforward = 32
    t1 = torch.rand(bs, in_features, past_steps)
    t2 = torch.rand(bs, target_features, future_steps)
    net = network_definitions.TransformerFull(in_features, past_steps ,future_steps, target_features, d_model, dim_feedforward).cpu()
    net.train()
    out = net((t1, t2))
    assert out is not None
    assert out.shape == (bs, future_steps)


def test_TransformerFull_evaluation_loop():
    bs = 8
    in_features = 14
    target_features = 2
    past_steps = 90
    future_steps = 40
    d_model = 128
    dim_feedforward = 32
    t1 = torch.rand(bs, in_features, past_steps)
    t2 = torch.rand(bs, target_features, future_steps)
    net = network_definitions.TransformerFull(in_features, past_steps, future_steps, target_features, d_model,
                                              dim_feedforward).cpu()
    net.eval()
    out = net((t1, t2))
    assert out is not None
    assert out.shape == (bs, future_steps)