import numpy as np
import pandas as pd
from fastcore.all import L

ONE_DAY_MINUTES = 24 * 60

def crude_gauss_gen(N):
    """
    Crude gauss kernel generator

    :param N:
    :return:
    """
    N = int(N)
    if N % 2 == 0:
        N += 1
    k = int(N / 2)
    sigma = N / 6
    arr = np.linspace(-k, k, num=N, endpoint=True, dtype=np.float32)
    arr = -1 * np.square(arr / sigma) / 2
    arr = np.exp(arr)
    arr /= np.sum(arr)
    return arr


def generate_sin_wave(N):
    """
    Generates a sinusoidal signal mixed with another, higher frequency sinusoid.

    The larger sinusoid is a yearly variation and the smaller one (higher frequency), the daily variation

    :param N: Length of the signal (in minutes)
    :return:
    """
    minutes = np.arange(0, N)
    output = np.sin(minutes / (365 * ONE_DAY_MINUTES) * 2 * np.pi) * 30 + np.sin(minutes / (ONE_DAY_MINUTES) * 2 * np.pi) * 10
    return output


def generate_random_commands(N, proba=0.9995, command_length=80,seed=None):
    """
    Generate a bunch of commands, randomly started and with fixed duration

    :param N: How many minutes to generate?
    :param proba: Probability of no command for current timestamp
    :param command_length: How many minutes to keep the command on?
    :param seed: Random seed, use None for no fixed randomization
    :return:
    """
    if seed is not None:
        np.random.seed(seed)
    cmds_rnd = np.random.rand(N) > proba
    cmds_where = np.where(cmds_rnd)[0]
    cmds = np.zeros(N)
    for k in range(cmds_where.shape[0]):
        a = cmds_where[k]
        cmds[a:a + command_length] = 1
    return cmds


def filter_signal(signal, gauss_k, delay_T):
    """
    Filters a signal with a Gauss kernel of size gauss_k and apply a delay delay_T

    :param signal: 1D ndarray with the signal to be filtered
    :param gauss_k: Spatial width of the kernel (sigma will be ~ k/6)
    :param delay_T: Delay to be applied to the signal. First values will be filled with zero
    :return:
    """
    N = signal.shape[0]
    gauss_kernel = crude_gauss_gen(gauss_k)
    filtered = np.convolve(signal, gauss_kernel, mode="same")
    output = np.zeros(N)
    if delay_T > 0:
        output[delay_T:] = filtered[0:-delay_T]
    else:
        output = filtered
    return output


def compute_heating_equation(commands, outside_temp, cmd_k, cmd_T, cmd_alpha, out_k, out_T, out_alpha, initial_temperature=25, heating_temperature=60):
    """
    Computes a crude heating equation from commands and outside_temp. Each signal is low-pass filtered, delayed and scaled.
    The end result is a summation of resulted values.

    :param commands: 1D ndarray with the commands for internal heater (1 == on)
    :param outside_temp: 1D ndarray with outside temperature. Same length as commands
    :param cmd_k: Gauss kernel to filter commands
    :param cmd_T: Delay to be applied to internal heating
    :param cmd_alpha: Ratio on heater influence
    :param out_k: Gauss kernel to filter outside temperature
    :param out_T: Delay to be applied to outside influences
    :param out_alpha: Ratio on outside influences
    :param initial_temperature: The output temperature at t_0
    :return: 1D ndarray with valid data only (smaller than the input arrays)
    """
    assert commands.shape == outside_temp.shape, "The input arrays must match in shape"
    N = commands.shape[0]
    filtered_cmds = filter_signal(commands, cmd_k, cmd_T)
    filtered_out = filter_signal(outside_temp, out_k, out_T)
    vaid_start = cmd_T + out_T
    resulted_temperature = np.zeros(N)
    resulted_temperature[0:vaid_start] = initial_temperature
    for k in range(vaid_start, N):
        t1 = resulted_temperature[k - 1]
        outside_influence = out_alpha * (filtered_out[k] - t1)
        inside_heating = cmd_alpha * (heating_temperature - t1) * filtered_cmds[k]
        t2 = t1 + outside_influence + inside_heating
        resulted_temperature[k] = t2
    return resulted_temperature


def generate_heating_data(N, command_length=80, command_seed=None, cmd_k=1.5*60, cmd_T=60, cmd_alpha=1e-1, out_k=12*60, out_T=6*60, out_alpha=1e-3):
    """
    Given a desired length N and some parameters, generate the outside temperature, commands and resulting inside temperature

    :param N: The number of timestamps in the output
    :param command_length: how much time a command is ON (minutes)
    :param command_seed: random seed for the commands
    :param cmd_k:  See compute_heating_equation
    :param cmd_T: See compute_heating_equation
    :param cmd_alpha: See compute_heating_equation
    :param out_k: See compute_heating_equation
    :param out_T: See compute_heating_equation
    :param out_alpha: See compute_heating_equation
    :return:
    """
    vaid_start = cmd_T + out_T
    N1 = N + vaid_start
    commands = generate_random_commands(N1, command_length=command_length, seed=command_seed)
    outside = generate_sin_wave(N1)
    result = compute_heating_equation(commands, outside, cmd_k, cmd_T, cmd_alpha, out_k, out_T, out_alpha)
    commands = commands[vaid_start:]
    outside = outside[vaid_start:]
    result = result[vaid_start:]
    return outside, commands, result


def generate_feature_block(outside, commands, target, no_features, positions=(8, 12, 13)):
    """
    Generates a 2D array with features, and inserts at positions, the outside,  commands and target arrays.
    The rest of the features are randomized

    :param outside: 1D ndarray
    :param commands: 1D ndarray
    :param target: 1D ndarray
    :param no_features: Number of columns in output array
    :param positions: Where to insert the 3 input arrays. Must have length 3
    :return:
    """
    assert len(positions) == 3
    N = outside.shape[0]
    features = np.random.rand(N, no_features)
    # Smooth the randomness a bit, and scale up the values
    for k in range(features.shape[1]):
        features[:, k] = filter_signal(features[:, k], 7, 0) * 30
    features[:, positions[0]] = outside
    features[:, positions[1]] = commands
    features[:, positions[2]] = target
    return features

def generate_samples(df, intervals, aggregate_interval_mins=10, stride_step = 0.5, sample_size=100, columns=None):
    """
    Cut out samples of sample_size len from full_df, respecting intervals tuple list

    :param df: Continuous dataframe
    :param intervals: List of tuples denoting what intervals we want to extract
    :param aggregate_interval_mins: How to aggregate the time
    :param stride_step: How many samples to skip before next window.
    :param sample_size: Length of the cutted window
    :param columns: None ==> all columns from dataframe
    :return:
    """

    if columns is None:
        columns = df.columns
    valid_segments = []
    for iv in intervals:
        sel_date = (iv[0] < df.index) & (df.index < iv[1])
        selected_data = df.loc[sel_date, columns]
        selected_data_filled = selected_data.fillna(method='ffill', inplace=False)
        selected_data_filled_agg = selected_data_filled.resample(f"{aggregate_interval_mins}T", axis=0, kind='timestamp').mean()
        valid_segments.append(selected_data_filled_agg)
    samples = []
    for iv in valid_segments:
        N = iv.shape[0]
        for k in range(0,N - sample_size, int(sample_size * stride_step)):
            sample = iv.iloc[k:(k+sample_size), :].values
            if not np.any(np.isnan(sample)):
                samples.append(sample)
    return L(samples)