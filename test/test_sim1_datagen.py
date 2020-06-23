import sim1_datagen
import numpy as np

def test_crude_gauss_gen_small_len_3():
    gauss = sim1_datagen.crude_gauss_gen(3)
    assert abs(np.sum(gauss) - 1) < 1e-3


def test_crude_gauss_gen_small_len_1():
    gauss = sim1_datagen.crude_gauss_gen(1)
    assert abs(np.sum(gauss) - 1) < 1e-3


def test_filter_signal():
    N = 20
    signal = np.zeros(N, dtype=np.float32)
    signal[5] = 1
    out_signal = sim1_datagen.filter_signal(signal,5, 10)
    assert np.sum(out_signal[0:10]) <= 1e-5, "Empty part of the signal should be filled with zero"
    assert abs(np.sum(out_signal) - 1) < 1e-2, "There must be no scaling of the signal"
    assert abs(np.argmax(out_signal) - (5 + 10))  <= 1, "The shift is not working properly"


def test_filter_signal_zero_k():
    N = 5
    signal = np.zeros(N, dtype=np.float32)
    signal[2] = 1
    out_signal = sim1_datagen.filter_signal(signal,1, 0)
    assert abs(out_signal[2] - 1) < 1e-2


def test_compute_heating_equation():
    N = 20
    t1 = np.ones(N) * 5
    t2 = np.ones(N)
    rez = sim1_datagen.compute_heating_equation(t1, t2, 1, 0, 1, 1, 0, 1, initial_temperature=5)
    assert rez.shape[0] == N, "There should be no delays in signal"


def test_compute_heating_equation_delays():
    N = 20
    t1 = np.ones(N) * 5
    t2 = np.ones(N)
    rez = sim1_datagen.compute_heating_equation(t1, t2, 1, 2, 1, 1, 3, 1, initial_temperature=5)
    assert rez[4] == 5, "The output signal must be delayed"
    assert rez[5] != 5, "The output signal must be delayed"


def test_generate_heating_data():
    N = 100
    outside, commands, result = sim1_datagen.generate_heating_data(N, 1, cmd_k=3, cmd_T=5, cmd_alpha=1,
                                                                   out_k=3, out_T=5, out_alpha=1)
    assert outside.shape[0] == N
    assert commands.shape[0] == N
    assert result.shape[0] == N
