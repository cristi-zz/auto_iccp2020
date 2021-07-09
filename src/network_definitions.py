import functools
import numpy as np
import torch
import torch.nn.functional as F
from fastai.data.transforms import ItemTransform
from torch import nn as nn
from torch.nn import functional as F


"""
Definitions for the architectures. See paper for more details.

The NN modules in this section assume that the I/O data have the following form:

In:  Batch size x no_features x input_time_len
Out: Batch size x no_features x output_time_len
 
As in paper, x1 is the known past and x2 is the known future. 
 
"""


class SelectableBN(nn.Module):
    """
    Allows one to select what flavour of BN to apply

    Selects between:
      0 no BN
      1 Regular BatchNorm
      2 just a learnable scale+bias without normalizing first the inputs.

    """
    def __init__(self, no_features, style=1):
        super().__init__()
        self.no_features=no_features
        self.style = style
        if style == 0:
            self.act = nn.Identity()
        elif style == 1:
            self.act = nn.BatchNorm1d(self.no_features)
        else:
            self.weight = nn.Parameter(torch.ones(1, self.no_features, 1))
            self.bias = nn.Parameter(torch.zeros(1, self.no_features, 1))

    def forward(self, data):
        if self.style in [0, 1]:
            out = self.act(data)
        if self.style == 2:
            batch_size = data.size(0)
            time_len = data.size(2)
            out = data * self.weight.expand(batch_size, self.no_features, time_len) +\
                  self.bias.expand(batch_size, self.no_features, time_len)
        return out


class AffineTransform(nn.Module):
    """
    An affine transform using some learned parameters.

    The one_kernel_per_feature parameter is ignored.

    """
    def __init__(self, no_features, input_time_len, output_time_len=None, one_kernel_per_feature=None, bn_type=1):
        super().__init__()
        if output_time_len is None:
            output_time_len = input_time_len
        if one_kernel_per_feature == True:
            output_time_len = input_time_len
        self.no_features = no_features
        self.input_time_len = input_time_len
        self.output_time_len = output_time_len
        self.translate = nn.Parameter(torch.zeros(self.no_features))
        self.scale = nn.Parameter(torch.zeros(self.no_features))
        self.bn = SelectableBN(self.no_features, bn_type)
        self.init_params()

    def init_params(self):
        self.translate.data = torch.randn(self.no_features) * 0.1
        self.scale.data = torch.randn(self.no_features) * 0.15


    def forward(self, data):
        batch_size = data.shape[0]
        affine = torch.zeros(self.no_features, 2, 3, dtype=torch.float32, device=self.scale.device)
        # Scale, then translate
        scale_factor = torch.exp(self.scale / 5)
        affine[:, 0, 0] = scale_factor
        affine[:, 1, 1] = 1
        affine[:, 0, 2] = scale_factor * self.translate
        grid_tensor = torch.nn.functional.affine_grid(affine, [self.no_features, 1, 1, self.output_time_len],
                                                          align_corners=False)
        grid_tensor_rep = grid_tensor.repeat(batch_size, 1, 1, 1)
        data_rep = data.reshape(-1, 1, self.input_time_len).unsqueeze(2)
        filtered_signal = torch.nn.functional.grid_sample(data_rep, grid_tensor_rep, mode='bilinear', align_corners=False)
        filtered_signal.squeeze(2)
        filtered_signal = filtered_signal.view(batch_size, self.no_features, self.output_time_len)
        scaled = self.bn(filtered_signal)
        return scaled


class LogGauss(nn.Module):
    """
    A fixed logNormal kernel is affine transformed using some learned parameters.

    """
    def __init__(self, no_features, input_time_len, output_time_len=None, one_kernel_per_feature=True, bn_type=1):
        super().__init__()
        if one_kernel_per_feature == False:
            if output_time_len is None:
                output_time_len = input_time_len
        self.no_features = no_features
        self.input_time_len = input_time_len
        self.one_kernel_per_feature = one_kernel_per_feature
        self.output_time_len = output_time_len
        self.kernel_width, self.pad = LogGauss.get_kernel_size_and_padding(self.input_time_len, one_kernel_per_feature)
        if one_kernel_per_feature == False:
            self.internal_no_features = self.no_features * self.output_time_len
        else:
            self.internal_no_features = self.no_features
        self.register_buffer("mother_kernel", self.gen_base_kernel())
        self.translate = nn.Parameter(torch.zeros(self.internal_no_features))
        self.scale = nn.Parameter(torch.zeros(self.internal_no_features))
        self.bn = SelectableBN(self.internal_no_features, bn_type)
        self.init_params()

    def init_params(self):
        self.translate.data = torch.randn(self.internal_no_features) * 0.1
        self.scale.data = torch.randn(self.internal_no_features) * 0.5
        if self.one_kernel_per_feature == False:
            wid_div = 0.9
            self.translate.data = self.translate.data - \
                                  torch.linspace(-wid_div, wid_div, self.output_time_len).repeat(self.no_features)

    def gen_base_kernel(self):
        no_samples = 100
        max_range = 20
        base = torch.linspace(1e-5, max_range, no_samples, requires_grad=False, dtype=torch.float32)
        loggauss = 1 / ((base)) * torch.exp(-np.square(np.log(base)) / 2)
        loggauss[torch.isnan(loggauss)] = 0
        filter_mode = no_samples / max_range * np.exp(-1)
        new_start_pos = int(no_samples / 2 - filter_mode) - 1
        loggauss = torch.roll(loggauss, new_start_pos)
        loggauss[:new_start_pos] = 0
        loggauss = loggauss.flip(0)
        return loggauss

    def forward(self, data):
        batch_size = data.shape[0]
        scaled_filters = self.get_scaled_filters()
        filtered_signal = nn.functional.conv1d(data, scaled_filters, padding=self.pad, groups=self.no_features)
        scaled = self.bn(filtered_signal)
        if not self.one_kernel_per_feature:
            scaled = scaled.view(batch_size, self.no_features, self.output_time_len)
        return scaled

    def get_scaled_filters(self):
        affine = torch.zeros(self.internal_no_features, 2, 3, dtype=torch.float32, device=self.scale.device)
        scale_factor = torch.exp(self.scale) + 3
        affine[:, 0, 0] = scale_factor
        affine[:, 1, 1] = 1
        affine[:, 0, 2] = scale_factor * self.translate
        grid_tensor = torch.nn.functional.affine_grid(affine, [self.internal_no_features, 1, 1, self.kernel_width],
                                                      align_corners=False)
        base_kern = self.mother_kernel
        base_kern = base_kern.expand(self.internal_no_features, 1, 1, base_kern.shape[0])
        scaled_kern = torch.nn.functional.grid_sample(base_kern, grid_tensor, mode='bilinear', align_corners=False)
        scaled_kern = scaled_kern.squeeze(2)  # Remove W dimension
        scaled_kern = scaled_kern / (torch.sum(scaled_kern, dim=2).view(-1, 1, 1) + 1e-7)
        return scaled_kern

    @staticmethod
    def get_kernel_size_and_padding(input_size, one_kernel_per_feature=True):
        if one_kernel_per_feature:
            offset = (input_size + 1) % 2
            nk = input_size + offset
            pad = int((input_size - input_size - 1 + nk) / 2)  # compute padding so out signal len == in signal len
        else:
            assert input_size is not None, "There must be a non null output size specified!"
            nk = input_size   # The conv1d will return ONE value per convolution
            pad = 0
        return nk, pad



class GaborFilter(nn.Module):
    """
    Gabor filters with fixed bandwidth. They output the magnitute and orientation of the complex convolution.

    Unstable for now in deeper architectures.

    """
    def __init__(self, no_features, input_time_len, output_time_len=None, one_kernel_per_feature=True,  bn_type=1, bandwidth=2.5,):
        super().__init__()
        self.one_kernel_per_feature = one_kernel_per_feature
        self.no_features = no_features
        self.input_time_len = input_time_len
        self.internal_no_features = 1
        self.bandwidth = bandwidth
        dtype = torch.float32
        if output_time_len is None:
            output_time_len = input_time_len
        self.output_time_len = output_time_len
        if self.one_kernel_per_feature:
            self.internal_no_features = no_features
        else:
            self.internal_no_features = no_features * output_time_len
        self.phase = 0
        self.nk, self.pad = GaborFilter.get_kernel_size_and_padding(self.input_time_len, self.one_kernel_per_feature)
        #         print(f"nk:{self.nk} pad: { self.pad}")
        self.support_range = max(self.nk / 2, 6)  # To avoid aliasing, but limits the offset range.
        self.frequency = nn.Parameter(torch.zeros(self.internal_no_features, dtype=dtype, requires_grad=True))
        self.mu = nn.Parameter(torch.zeros(self.internal_no_features, dtype=dtype, requires_grad=True))
        self.bn_mag = SelectableBN(self.internal_no_features, bn_type)
        self.bn_orient = SelectableBN(self.internal_no_features, bn_type)

        self.kernel_support = nn.Parameter(torch.zeros(self.nk, dtype=dtype, requires_grad=False))
        self.init_params()

    def compute_kernel(self):
        mu = self.mu
        frequency = torch.sigmoid(self.frequency)*(0.5 - 1.0/self.nk) + 1.0/self.nk  # min frequency =1/T, max frequency 1/2
        mu = mu.view(-1, 1)
        frequency = frequency.view(-1, 1)
        sigma = self.bw_constant / frequency
        real_comp = torch.cos((self.kernel_support - mu) * frequency * self.two_pi_constant)
        imag_comp = torch.sin((self.kernel_support - mu) * frequency * self.two_pi_constant)
        exp_comp = torch.exp(-torch.square((self.kernel_support - mu) / sigma) / 2)
        gabor_kernels_real = real_comp * exp_comp
        gabor_kernels_imag = imag_comp * exp_comp

        gabor_kernels_real = gabor_kernels_real.flip(1).unsqueeze(1)
        gabor_kernels_imag = gabor_kernels_imag.flip(1).unsqueeze(1)
        return gabor_kernels_real, gabor_kernels_imag

    def forward(self, data):
        batch_size = data.shape[0]
        gabor_kernels_real, gabor_kernels_imag = self.compute_kernel()
        if torch.isnan(gabor_kernels_real).any():
            print(f"NaNs detected in gabor_kernel_real")
        if torch.isnan(gabor_kernels_imag).any():
            print(f"NaNs detected in gabor_kernels_imag")

        filtered_signal_real = nn.functional.conv1d(data, gabor_kernels_real, padding=self.pad, groups=self.no_features)
        filtered_signal_imag = nn.functional.conv1d(data, gabor_kernels_imag, padding=self.pad, groups=self.no_features)
        magnitude = torch.sqrt(torch.square(filtered_signal_real) + torch.square(filtered_signal_imag))
        orientation = torch.atan2(filtered_signal_imag, filtered_signal_real)

        scaled_magnitude = self.bn_mag(magnitude)
        scaled_orientation = self.bn_orient(orientation)
        if not self.one_kernel_per_feature:
            scaled_magnitude = scaled_magnitude.view(batch_size, self.no_features, self.output_time_len)
            scaled_orientation = scaled_orientation.view(batch_size, self.no_features, self.output_time_len)
        return scaled_magnitude, scaled_orientation

    def init_params(self):
        self.kernel_support.data = torch.linspace(-self.support_range, self.support_range, self.nk)
        self.frequency.data = torch.randn(self.internal_no_features) * 1 + 2
        self.mu.data = (torch.rand(self.internal_no_features) - 0.5) * 0.1 * self.support_range
        if self.one_kernel_per_feature == False:
            wid_div = self.support_range * 0.9
            self.mu.data -= torch.linspace(-wid_div, wid_div, self.output_time_len).repeat(self.no_features)
        two_df = np.power(2, self.bandwidth)
        self.bw_constant = np.sqrt(np.log(2) / np.pi) * (two_df + 1)/(two_df - 1)
        self.two_pi_constant = np.pi * 2

    @staticmethod
    def get_kernel_size_and_padding(input_size, one_kernel_per_feature=True):
        if one_kernel_per_feature:
            offset = (input_size + 1) % 2
            nk = input_size + offset
            pad = int((input_size - input_size - 1 + nk) / 2)  # compute padding so out signal len == in signal len
        else:
            assert input_size is not None, "There must be a non null output size specified!"
            nk = input_size   # The conv1d will return ONE value per convolution
            pad = 0
        return nk, pad


class IdentityTransform(nn.Module):
    """
    An identity transform that forwards the input to the output.

    If one_kernel_per_feature parameter is False and the output_time_len < input_time_len the output is a selected
    view of the input. Only the last part of the input signal is sent to the output.

    No batch norm is perfomed. (bn_type is ignored)
    """

    def __init__(self, no_features, input_time_len, output_time_len=None, one_kernel_per_feature=True, bn_type=None):
        super().__init__()
        if output_time_len is None:
            output_time_len = input_time_len
        self.no_features = no_features
        self.input_time_len = input_time_len
        self.output_time_len = output_time_len
        self.one_kernel_per_feature = one_kernel_per_feature

    def forward(self, data):
        batch_size = data.shape[0]
        if (self.one_kernel_per_feature == False) and (self.input_time_len != self.output_time_len):
            return data[:, :, -self.output_time_len:]
        return data


class CausalTemporalConv(nn.Module):
    """
    Causal temporal convolution.

    It will "shrink" the input, along time, from input_time_len to output_time_len.

    There is a convolution layer that performs this transformation. Each feature will have its own learnable
    convolution kernel. There is no crossover between featuers.

    The one_kernel_per_feature must be false for compatibility with other filters.

    """
    def __init__(self, no_features, input_time_len, output_time_len, one_kernel_per_feature=False):
        super().__init__()
        assert (one_kernel_per_feature == False), "one_kernel_per_feature must be false so output_time_len makes sense"
        assert (output_time_len <= input_time_len), "The desired output must be smaller than the input"
        kernel_size = input_time_len - output_time_len + 1
        self.conv = nn.Conv1d(in_channels=no_features, out_channels=no_features, kernel_size=kernel_size, padding=0,
                              groups=no_features)

    def forward(self, data):
        out = self.conv(data)
        return out


class GaussFilter(nn.Module):
    """
    Gaussian filter.

    """
    def __init__(self, no_features, input_time_len, output_time_len=None, one_kernel_per_feature=True, bn_type=1):
        super().__init__()
        self.one_kernel_per_feature = one_kernel_per_feature
        self.no_features = no_features
        self.input_time_len = input_time_len
        self.internal_no_features = 1
        dtype = torch.float32
        if output_time_len is None:
            output_time_len = input_time_len
        self.output_time_len = output_time_len
        if self.one_kernel_per_feature:
            self.internal_no_features = no_features
        else:
            self.internal_no_features = no_features * output_time_len
        self.nk, self.pad = GaborFilter.get_kernel_size_and_padding(self.input_time_len, self.one_kernel_per_feature)
        self.support_range = max(self.nk / 2, 3)  # To avoid aliasing, but limits the offset range.
        self.sigma = nn.Parameter(torch.zeros(self.internal_no_features, dtype=dtype, requires_grad=True))
        self.mu = nn.Parameter(torch.zeros(self.internal_no_features, dtype=dtype, requires_grad=True))
        self.bn = SelectableBN(self.internal_no_features, bn_type)
        self.kernel_support = nn.Parameter(torch.zeros(self.nk, dtype=dtype, requires_grad=False))
        self.init_params()

    def forward(self, data):
        batch_size = data.shape[0]
        gauss_kernel = self.compute_kernel()
        filtered_signal = nn.functional.conv1d(data, gauss_kernel, padding=self.pad, groups=self.no_features)
        scaled = self.bn(filtered_signal)
        if not self.one_kernel_per_feature:
            scaled = scaled.view(batch_size, self.no_features, self.output_time_len)
        return scaled

    def init_params(self):
        self.kernel_support.data = torch.linspace(-self.support_range, self.support_range, self.nk)
        self.sigma.data = torch.randn(self.internal_no_features) * 0.1
        self.mu.data = (torch.randn(self.internal_no_features)) * 0.01 * self.support_range
        if self.one_kernel_per_feature == False:
            wid_div = self.support_range * 0.9
            self.mu.data -= torch.linspace(-wid_div, wid_div, self.output_time_len).repeat(self.no_features)

    def compute_kernel(self):
        mu = self.mu
        sigma = self.sigma
        mu = mu.view(-1, 1)
        sigma = sigma.view(-1, 1)
        gauss_kernel = torch.exp(-torch.square((self.kernel_support - mu) / (torch.exp(sigma))) / 2)
        gauss_kernel = gauss_kernel.flip(1).unsqueeze(1)
        return gauss_kernel

    @staticmethod
    def get_kernel_size_and_padding(input_size, one_kernel_per_feature=True):
        if one_kernel_per_feature:
            offset = (input_size + 1) % 2
            nk = input_size + offset
            pad = int((input_size - input_size - 1 + nk) / 2)  # compute padding so out signal len == in signal len
        else:
            assert input_size is not None, "There must be a non null output size specified!"
            nk = input_size   # The conv1d will return ONE value per convolution
            pad = 0
        return nk, pad


class BankedFilters(nn.Module):
    """
    One will want more filters per feature and this class organize these filters.

    All filters here have the same meta-properties (eg filtering type) but different learnable parameters.

    The only parameters that is not forwarded is number_of_filters that specify how many filters are in this bank.

    """
    def __init__(self, no_features, input_time_steps, filter_class, number_of_filters=8, one_kernel_per_feature=True, bn_type=1):
        super().__init__()
        assert number_of_filters >= 1, "There must be at least one filter in the filter bank"
        filter_bank = []
        assert filter_class in [IdentityTransform, GaussFilter, AffineTransform, GaborFilter, LogGauss]
        self.filter_class = filter_class
        self.no_features = no_features
        self.number_of_filters = number_of_filters
        for k in range(number_of_filters):
            f1 = self.filter_class(no_features, input_time_steps, input_time_steps, one_kernel_per_feature=one_kernel_per_feature, bn_type=bn_type)
            filter_bank.append(f1)
        self.filter_bank = nn.ModuleList(filter_bank)

    def get_number_of_out_features(self):
        no_out_features = self.no_features * self.number_of_filters
        if self.filter_class == GaborFilter:
            return 2 * no_out_features
        return no_out_features

    def forward(self, data):
        """
        data have shape: Batch x Features x Time
        """
        acc_features = []
        if self.filter_class == GaborFilter:
            for gf in self.filter_bank:
                mag, orient = gf(data)
                acc_features.extend([mag, orient])
        else:
            for gf in self.filter_bank:
                filter_out = gf(data)
                acc_features.append(filter_out)
        out = torch.cat(acc_features, dim=1)
        return out


class FeatureAggregationStack(nn.Module):
    """
    Transforms a set of features into an output using a fully connected network.

    The input is of shape batch x no_features x time_len and the FCN is applied only accross features.
    The weights are shared for all temporal steps (along time_len dimension)

    The FCN is implemented using a series of 1D convolutional kernels.

    One can select the number of linear layers that do the transform and the intermediate size of those layers.
    The number of layers is no_layers+1:

    If no_layers is set to zero, only a linear operation is applied.

    If no_layers is one, two linear transformations will be present.

    Between the layers is leaky_relu nonlinearity

    No activation is performed on the output. So, if no_layers is 0, this module will perform a linear transformation.

    An optional (do_initial_bn) BatchNorm is performed on the input.

    Inputs:  Batch x no_features x time_len
    Outputs: Batch x no_out_features x time_len

    """
    def __init__(self, no_features, no_out_features, no_intermediate_features, no_layers, do_initial_bn=True):
        super().__init__()
        self.do_initial_bn = do_initial_bn
        if self.do_initial_bn:
            self.bn = nn.BatchNorm1d(no_features)
        self.no_layers = no_layers

        if no_layers <= 0:
            self.input = nn.Conv1d(no_features, no_out_features, kernel_size=1, bias=True)
        else:
            self.input = nn.Conv1d(no_features, no_intermediate_features, kernel_size=1, bias=False)
            self.output = nn.Conv1d(no_intermediate_features, no_out_features, kernel_size=1, bias=True)
            self.interm_layers = nn.ModuleList([nn.Conv1d(no_intermediate_features, no_intermediate_features, kernel_size=1, bias=True) for _ in range(no_layers - 1)])

    def forward(self, data):
        data_bn = data
        if self.do_initial_bn:
            data_bn = self.bn(data)
        if self.no_layers == 0:
            out = self.input(data_bn)
        else:
            interm_data = F.leaky_relu(self.input(data_bn))
            for k,layer in enumerate(self.interm_layers):
                interm_data = F.leaky_relu(layer(interm_data))
            out = self.output(interm_data)
        return out


class ConstructDelayNet(nn.Module):
    """
    Instantiate a heavily customizable Delay type architecture.

    The Delay architecture (shown approx in Fig. 5 in the paper) have several blocks, each block with parameters.
    This class allows one to instantiate everything in one go.

    Parameters like no_features, no_commands, no_targets, past_length, future_length are specific to the inputs and outputs,
    the rest of the parameters are sent to their specific blocks:

    filter_* to BankedFilters class
    aggregator_* to TimeWiseLinearStack
    temporal_contractor to CausalTemporalConv or directly to one of the filters, having one_kernel_per_feature == False
    The number of I/O features for the filters is deduced automatically based on other parameters.

    aggregator_low_out_bottleneck is the **Fc** parameter in the paper. It decides how many features the aggregator_low
    block will output.

    The bn_type_* allows one to experiment with the batch normalization after filtering. Setting it to 1 is a sensitive choice.
    Check SelectableBN for some options.

    out_interpolator allows some interpolation for the final output, slightly reducing the network capacity. A sensitive
    choice is to not use it.

    The input shapes for this networ are:

    In:  A tuple tensor: (Batch size x no_features x past_length , Batch size x no_commands x (past_length + future_length) )
    Out: A tensor: Batch size x no_targets x future_length

    """
    def __init__(self, no_features, no_commands, no_targets, past_length, future_length,
                 filter_low_classname, filter_low_filter_count, filter_low_one_kernel_per_feature,
                 aggregator_low_expansion, aggregator_low_layers, aggregator_low_out_bottleneck,
                 temporal_contractor_classname,
                 filter_high_classname, filter_high_filter_count, filter_high_one_kernel_per_feature,
                 aggregator_high_expansion, aggregator_high_layers,
                 bn_type_low=1, bn_type_temporal=1, bn_type_high=1, out_interpolator=None):
        super().__init__()
        self.no_features = no_features
        self.no_commands = no_commands
        self.no_targets = no_targets
        self.past_timelen = past_length
        self.future_timelen = future_length
        self.total_time = past_length + future_length
        if out_interpolator is None:
            out_interpolator = self.future_timelen

        self.internal_future_timelen = out_interpolator

        self.filter_low_classname = filter_low_classname
        self.filter_low_filter_count = filter_low_filter_count
        self.aggregator_low_expansion = aggregator_low_expansion
        self.aggregator_low_layers = aggregator_low_layers
        self.aggregator_low_out_bottleneck = aggregator_low_out_bottleneck
        self.temporal_contractor_classname = temporal_contractor_classname
        self.filter_high_classname = filter_high_classname
        self.filter_high_filter_count = filter_high_filter_count
        self.aggregator_high_expansion = aggregator_high_expansion
        self.aggregator_high_layers = aggregator_high_layers

        self.filter_low = BankedFilters(self.no_features, self.past_timelen, eval(self.filter_low_classname),
                                        self.filter_low_filter_count, filter_low_one_kernel_per_feature, bn_type=bn_type_low)
        no_features_after_expander_low = self.filter_low.get_number_of_out_features()
        self.aggregator_low = FeatureAggregationStack(no_features_after_expander_low,
                                                      self.aggregator_low_out_bottleneck,
                                                      int(no_features_after_expander_low * self.aggregator_low_expansion),
                                                      self.aggregator_low_layers, do_initial_bn=False)
        if self.temporal_contractor_classname in ["IdentityTransform", "GaussFilter", "AffineTransform", "GaborFilter", "LogGauss"]:
            self.temporal_contractor = eval(self.temporal_contractor_classname)(self.aggregator_low_out_bottleneck,
                                                                                self.past_timelen, self.internal_future_timelen,
                                                                                one_kernel_per_feature=False, bn_type=bn_type_temporal)
        elif self.temporal_contractor_classname == "CausalTemporalConv":
            self.temporal_contractor = CausalTemporalConv(self.aggregator_low_out_bottleneck, self.past_timelen, self.internal_future_timelen)
        else:
            assert False, "temporal_contractor_classname is wrong. Check documentation"
        self.high_feature_count = self.aggregator_low_out_bottleneck + self.no_commands
        self.filter_high = BankedFilters(self.high_feature_count, self.internal_future_timelen, eval(self.filter_high_classname),
                                         self.filter_high_filter_count, filter_high_one_kernel_per_feature, bn_type=bn_type_high)
        self.high_feature_count_expanded = self.filter_high.get_number_of_out_features()
        self.aggregator_high = FeatureAggregationStack(self.high_feature_count_expanded, self.no_targets,
                                                       int(self.high_feature_count_expanded * self.aggregator_high_expansion),
                                                       self.aggregator_high_layers, do_initial_bn=False)

    def forward(self, *data):
        feature_block = data[0]
        command_block = data[1]
        batch_size = feature_block.shape[0]

        feat_delay_low = self.filter_low(feature_block)
        feature_mid = F.leaky_relu(self.aggregator_low(feat_delay_low))
        feature_mid_future = F.leaky_relu((self.temporal_contractor(feature_mid)))
        if self.internal_future_timelen != self.future_timelen:
            command_block = nn.functional.interpolate(command_block, size=(self.internal_future_timelen, ), mode='linear', align_corners=False)
        feature_mid_all = torch.cat([feature_mid_future, command_block], dim=1)
        feature_high_delayed = F.leaky_relu(self.filter_high(feature_mid_all))
        out_aggregated = self.aggregator_high(feature_high_delayed)
        if self.internal_future_timelen != self.future_timelen:
            out_aggregated = nn.functional.interpolate(out_aggregated.unsqueeze(1), size=(1, self.future_timelen, ), mode='bilinear', align_corners=False)
            out_aggregated = out_aggregated.squeeze(1)
        return out_aggregated


class ICCP_Wrap_AttentionModel(nn.Module):
    """
    This model is a small wrapper around the previously developed Attention model.

    The models developed in previous iterations had fixed sizes (of one) for the number of command features and
    number of target features. In current iteration, one can have more than one command and can predict more than one
    target.

    When there are more commands and targets, only the first command and target are sent to the Attention network.

    """
    def __init__(self, no_features, no_commands, no_targets, past_length, future_length,
                 hidden_size, num_layers=1, dropout=0.5):
        super().__init__()
        in_features = no_features
        self.base_network = AttentionModel(in_features, past_length, future_length,
                                                        hidden_size, num_layers, dropout)

    def forward(self, *data):
        feature_block = data[0]
        command_block = data[1]
        out = self.base_network((feature_block, command_block.squeeze(1)))
        out = out.unsqueeze(1)
        return out


class FeatureItemizer(ItemTransform):
    """
    Generates the features, commands and ground truth data from list of continuous sequences of data.

    The data list is parsed by the fastai Transform and the encodes() function receive one sample at a time.
    One assumes that the input is a blend of features, commands, and targets (eg output of the system).
    Up to past_time these are known. The commands are known for the future_time too.

    Input for encodes(): Time x K array, where K is no of features + no of commands + no of targets
    Output: Several arrays:
        len(known_set) x past_time representing the known features+commands+targets
        len(comands_set) x future_len representing the future proposed commands
        len(target_set) x future_time, future targets, for the future period. Used as ground truth in learning

    In setups(), the method will calculate avg and std for each group in feature_groups. If some features are not in
    feature_groups, each one will be treated as separate groups.

    If average_range is >0, then the last average_range values from the known past, of the target, are averaged and this
    average is then substracted from all the features in the first feature group. So, make sure that the target_set is
    in the feature_groups[0]

    If average_range == 0 this step is ignored.

    Use decode_predictions_from_the_network() to reconstruct the samples

    Constructor parameters: future_len, comands_set, target_set are lists of column indices that will be exported to
    the output (eg vectors X1, X2, Y from paper)

    """

    def __init__(self, future_len, known_set, comands_set, target_set, feature_groups, average_range=0.2):
        super().__init__()
        self.future_len = future_len
        self.known_set = known_set
        self.comands_set = comands_set
        self.target_set = target_set
        self.feature_groups = feature_groups
        self.average_range = average_range
        self.avgs = None

    def encodes(self, input):
        # input of shape Time x Features
        features_input = input - self.avgs  # Only one block of inputs
        features_input /= self.stds
        N = features_input.shape[0]
        past_len = N - self.future_len

        if self.average_range > 0:
            count_to_subst = int(past_len *(1-self.average_range))
            val_to_substract = np.average(features_input[count_to_subst:past_len, self.target_set])
            features_input[:, self.feature_groups[0]] -= val_to_substract

        known_past = features_input[:past_len, self.known_set]
        future_commands = features_input[past_len:N, self.comands_set]
        targets_future = features_input[past_len:N, self.target_set]

        return torch.tensor(known_past, dtype=torch.float).transpose(0, 1), \
               torch.tensor(future_commands, dtype=torch.float).transpose(0, 1), \
               torch.tensor(targets_future, dtype=torch.float).transpose(0, 1),

    def setups(self, items):
        if self.avgs is not None:  # avoid re-setup, to be refactored with dataset_index
            return
        fg_new = []  # lone features that will be grouped
        item = items[0]
        no_features = item.shape[1]
        items_arr = np.reshape(np.array(items), (-1, no_features))
        grouped_features = functools.reduce(set.union, self.feature_groups, set())
        for k in range(no_features):
            if k not in grouped_features:
                fg_new.append((k,))
        self.feature_groups.extend(fg_new)
        self.avgs = np.zeros(no_features)
        self.stds = np.ones(no_features)
        for group in self.feature_groups:
            group = list(group)
            features = items_arr[:, group].ravel()
            avgs = np.average(features)
            stds = np.std(features)
            self.avgs[group] = avgs
            self.stds[group] = stds


def decode_predictions_from_the_network(network_output, data_sample_array, itemizer):
    """
    Takes a network output generated by the  fastai get_preds() method (either target or preds) and the list of samples (in an array form)
    and the itemizer instance that coded the instances.

    It returns the network output scaled back and with eventual extracted average added.

    See the demo notebook on how to use it.

    """
    N = data_sample_array.shape[1]
    past_len = N - itemizer.future_len
    val_to_substract = 0
    if itemizer.average_range > 0:
        count_to_subst = int(past_len *(1-itemizer.average_range))
        to_avg = data_sample_array[:, count_to_subst:past_len, itemizer.target_set]
        a1 = np.average(to_avg, axis=1)
        a2 = itemizer.avgs[itemizer.target_set]
        val_to_substract = a1 - a2
    stds = itemizer.stds[itemizer.target_set]
    avgs = itemizer.avgs[itemizer.target_set]
    pred_set_np = network_output.squeeze(1).detach().cpu().numpy()
    decoded = stds * pred_set_np + avgs + val_to_substract
    return decoded






############################
#
# Att_* network definitions. Used indirectly, through ICCP_Wrap_AttentionModel()
#
############################



class AttentionEncoding(nn.Module):
    """
    The encoder from Attention architecture. It works only on the input x1

    """
    def __init__(self, in_features, sample_len, hidden_size, num_layers, dropout):
        super().__init__()
        self.in_features = in_features
        self.sample_len = sample_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if num_layers == 1:
            dropout=0
        self.bn = nn.BatchNorm1d(num_features=in_features)
        self.gru = nn.LSTM(in_features, hidden_size, num_layers=num_layers, dropout=dropout,  batch_first=True)

    def forward(self, data):
        bn = self.bn(data)          # B, F, S
        bn_t = bn.transpose(2, 1)   # B, S, F, needed for LSTM
        output, hidden = self.gru(bn_t) # Output: (B, S, H),  2 x (num_layers, B, H)
        return output, hidden


class AttentionDecoding(nn.Module):
    """
    The decoder, from the architecture. Implements the attention mechanism.

    It expects one "output" at a time
    """
    def __init__(self, hidden_size, sample_len, input_features, output_size, num_layers, dropout):
        """

        :param hidden_size: Same as for encoder
        :param sample_len: The "future" size, T
        :param input_features:  The number of input features, F2 (here, == 2)
        :param output_size: The output size, here == 1
        :param num_layers:  Same as for encoder.
        :param dropout: Ignored if num_layers == 1
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.sample_len = sample_len
        self.input_features = input_features
        self.num_layers = num_layers
        self.hidden_flat = hidden_size * num_layers
        self.output_size = output_size

        self.attn = nn.Linear(self.input_features + self.hidden_flat, self.sample_len)
        self.attn_combine = nn.Linear(self.input_features + self.hidden_size, self.hidden_size)
        self.bn_input_attn = nn.BatchNorm1d(num_features=self.input_features+self.hidden_size)
        self.gru = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, encoder_outputs, hidden):
        """

        :param input: size B, 1, input_features, the previous prediction
        :param encoder_outputs: BS, sample_len, hidden_size , the otuput of the encoder
        :param hidden: *  the output of the encoder.
        :return:
        """
        bs = input.size(0)              # batch size B
        if type(self.gru) is nn.LSTM:
            hidden_for_cat = hidden[0]      # Because this is where LSTM keeps its hidden states
        else:
            hidden_for_cat = hidden         # GRU does not output a tuple for the hidden states.
        # We flatten the hidden states, batch-wise
        hidden_for_cat = hidden_for_cat.view(1, bs, self.hidden_flat) # 1, B, num_layers * H
        # Attach the previous known output
        cat_hidden_input = torch.cat((input.squeeze(1), hidden_for_cat.squeeze(0)), 1) # B, num_layers * H + input_features
        # Compute the weights for the attention. One weight for each past timestamp
        attn_weights = self.attn(cat_hidden_input) # B, S
        # One of the past items is relevant, so softmax to find it.
        attn_weights_smax = torch.softmax(attn_weights, dim=1) # attn_weights_smax shape (B, S)
        # Apply the attention to the encoder outputs and outuput the relevant past information
        attn_applied = torch.bmm(attn_weights_smax.unsqueeze(1),
                                 encoder_outputs)  #attn_applied shape: (BS, 1, Hidden)
        # Attach the previous output to the relevant past information
        att_plus_input = torch.cat((input.squeeze(1), attn_applied.squeeze(1)), 1)
        # Regularize the responses, otherwise there is no learning for num_layers > 1
        att_plus_input = self.bn_input_attn(att_plus_input)
        # Generate the input for LSTM, with a linear layer
        att_plus_input_comb = self.attn_combine(att_plus_input)
        att_plus_input_comb = torch.sigmoid(att_plus_input_comb).unsqueeze(1)
        output, hidden_out = self.gru(att_plus_input_comb, hidden) # att_plus_input_comb shape (BS, 1, H),hidden: 2 x (num_layers, BS, H)
        output = torch.tanh(output)
        output = self.out(output.squeeze(1))
        # output shape (B, output_size)
        # hidden_out shape 2  x (num_layers, B, Hidden) -> This will be injected back
        return output, hidden_out


class AttentionModel(nn.Module):
    """
    Main attention module. Will call AttentionEncoding and AttentionDecoding

    Handles a bit of housekeeping and data iteration.

    Does NOT implement teacher forcing, each future time step is based on the network's previous output.

    """

    def __init__(self, in_features, past_steps, future_steps, hidden_size, num_layers=1, dropout=0.5):
        """

        :param in_features: F
        :param past_steps:  S
        :param future_steps:  T
        :param hidden_size:  H
        :param num_layers:  Number of LSTM layers, same for encoder and decoder
        :param dropout: Dropout between LSTM layers. Ignored if num_layers == 1
        """
        super().__init__()
        if num_layers == 1:
            dropout=0
        self.in_features = in_features
        self.past_steps = past_steps
        self.hidden_size = hidden_size
        self.future_steps = future_steps
        self.num_layers = num_layers
        self.encoder = AttentionEncoding(in_features, past_steps, hidden_size, num_layers, dropout)
        self.decoder = AttentionDecoding(hidden_size, past_steps, input_features=2, output_size=1,
                                         num_layers=num_layers, dropout=dropout)

    def forward(self, input):
        x1, x2 = input
        encoded, hidden_encoded = self.encoder(x1)
        # We collect the outputs
        output_lst = []
        dec_hidden = hidden_encoded
        dec_out_recursive = torch.zeros((x1.size(0), 1, 2), requires_grad=False, device=x1.device)
        for k in range(self.future_steps):
            # First feature in F2 is the previous prediciton
            # Second feature in F2 is the known future at current step.
            dec_out_recursive[:, 0, 1] = (x2[:, k]).detach()
            dec_out, dec_hidden = self.decoder(dec_out_recursive, encoded, dec_hidden)
            output_lst.append(dec_out)
            # The gradient will not propagate back from the input of the next step
            dec_out_detached = dec_out.detach()
            # Write the current prediction for the next step
            dec_out_recursive[:, :, 0] = dec_out_detached
        output = torch.cat(output_lst, axis=1)
        output = output.squeeze(1)
        return output

def compute_model_param_count(model):
    """
    Computes the total number of *trainable* weights/params into a model.

    For some reason, default parameter count in pytorch skips some layers.

    """
    net_params = filter(lambda p: p.requires_grad, model.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count

def print_model_weights_rec(model, crt_level=0, max_level=2):
    """
    Pretty prints a network structure showing the weights at each sub-level.

    """
    crt_sum = 0
    prefix = "      " * crt_level
    network_name = str(type(model))
    no_params = compute_model_param_count(model)
    print(f"{prefix}{network_name} with {no_params} parameters:")
    for cm in model.children():
        if crt_level < max_level:
            print_model_weights_rec(cm, crt_level+1, max_level)
    print(f"{prefix}Parameters: {no_params}\n{prefix}------------------")
