from fastai2.vision.all import *
import torch
"""
Definitions for the architectures. See paper for more details.
F is the number of features, usuallu dentoted with  in_features
S is the number of past timestamps, past_steps
T is the future of future time steps. Either the number of future steps that need predicted or the length of vector x2.
Usually future_steps here.
 
The number of output features is assumed to be 1.
 
As in paper, x1 is the known past and x2 is the known future. 
 
"""


class LinearModel(nn.Module):
    """
    Simple linear architecture. Batch norm of the input data then a linear layer.
    Each timestamp from future_steps is a linear combination of the normalized inputs.
    There is no activation.

    """
    def __init__(self, in_features, past_steps, future_steps):
        """
        Constructor

        :param in_features: F
        :param past_steps:
        :param future_steps:
        """
        super().__init__()
        self.in_features = in_features
        self.future_steps = future_steps
        self.past_steps = past_steps
        self.bn = nn.BatchNorm1d(num_features=in_features)
        self.linear_out = nn.Linear(in_features=in_features * past_steps + future_steps, out_features=future_steps)

    def forward(self, data):
        """
        data is a tuple: x1, x2
        x1: B, F, S
        x2: B, T
        :param data:
        :return:
        """
        x1, x2 = data # x1: B, F, S   x2: B, T
        bs = x1.size(0)
        x1_norm = self.bn(x1)
        x1_flatten = x1_norm.reshape(bs, self.in_features * self.past_steps)
        concatenaded_input = torch.cat([x1_flatten, x2], axis=1)
        linear_out = self.linear_out(concatenaded_input)
        return linear_out



class CnnFcn(nn.Module):
    """
    A sanwich of CNNs followed by a layer of FCNs.
    Between them, a pooling layer reduces the dimensionality and concatenates the known future (x2).

    There will always be one CNN layer and one FCN layer.

    See fastai2 documentation for ConvLayer and LinBnDrop

    """
    def __init__(self, in_features, past_steps, future_steps, cnn_stack_len, fcn_stack_len, cnn_filters, avg_pool_len,
                 fcn_ratio):
        """

        :param in_features:  F
        :param past_steps: S
        :param future_steps: T
        :param cnn_stack_len: Number of CNN layers
        :param fcn_stack_len: Number of FCN layers
        :param cnn_filters: No of kernel channels in the CNN stack
        :param avg_pool_len: The length of the signal after the Pooling
        :param fcn_ratio: Ratio between the size of FCN layers and the signal after Pooling. Channels are included.
        """
        super().__init__()
        self.cnn_filters = cnn_filters
        self.avg_pool_len = avg_pool_len
        self.hidden_fcn = int(self.cnn_filters * self.avg_pool_len * 2 * fcn_ratio)
        self.cnn_stack_len = cnn_stack_len
        self.fcn_stack_len = fcn_stack_len
        self.in_features = in_features
        self.future_steps = future_steps
        self.past_steps = past_steps

        self.cnn1 = ConvLayer(in_features, self.cnn_filters, ks=7, ndim=1, bn_1st=True)
        cnn_stack = [nn.Identity()]
        for k in range(cnn_stack_len):
            cnn_stack.append(ConvLayer(self.cnn_filters, self.cnn_filters, ks=3, ndim=1, bn_1st=True))
        self.cnn_stack = nn.Sequential(*cnn_stack)

        self.pool = AdaptiveConcatPool1d(size=self.avg_pool_len)
        self.flat1 = Flatten()
        self.fcn1 = LinBnDrop(self.cnn_filters * self.avg_pool_len * 2 + future_steps, self.hidden_fcn, p=0,
                              act=nn.ReLU())
        fcn_list = [nn.Identity()]
        for k in range(fcn_stack_len):
            fcn_list.append(LinBnDrop(self.hidden_fcn, self.hidden_fcn, bn=False, p=0.2, act=nn.ReLU()))
        self.fcn_stack = nn.Sequential(*fcn_list)
        self.out_linear = LinBnDrop(self.hidden_fcn, future_steps, bn=False, p=0.1, act=None)

    def forward(self, data):
        """
        data = tuple of (B, F, S), (B, T)
        B is batch size,
        F is no of features
        S is train timestamp length
        T is future timestamp length
        :param data:
        :return:
        """
        x1, x2 = data # x1: B, F, S   x2: B, T
        conv1 = self.cnn1(x1)             # conv1: B, cnn_filters, S
        cnn_stack = self.cnn_stack(conv1) # B, cnn_filters, S
        pooling = self.pool(cnn_stack)    # B, 2 * cnn_filters, avg_pool_len
        encoder_output = self.flat1(pooling)

        decoer_input = torch.cat([encoder_output, x2], axis=1)
        fcn1 = self.fcn1(decoer_input)          # B, hidden_fcn
        fcn_stack = self.fcn_stack(fcn1)        # B, hidden_fcn
        out_layer = self.out_linear(fcn_stack)  # B, T
        return out_layer


class EncoderDecoder(nn.Module):
    """
    LSTM based encoder + decoder.

    """
    def __init__(self, in_features, past_steps, future_steps, hidden_size, num_layers=1, dropout=0.5):
        """

        :param in_features:  F
        :param past_steps: S
        :param future_steps: T
        :param hidden_size: Number of hidden units in LSTM
        :param num_layers: Number of layers in LSTM
        :param dropout: If num_layers > 1, the dropout between LSTM layers

        The output is assumed having one feature

        """
        super().__init__()
        self.in_features = in_features
        self.past_steps = past_steps
        self.hidden_size = hidden_size
        self.future_steps = future_steps
        if num_layers == 1:
            dropout=0
        self.bn = nn.BatchNorm1d(num_features=in_features)
        self.rnn_encoder = nn.LSTM(in_features, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.rnn_decoder= nn.LSTM(1, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.out_shrink = nn.Linear(hidden_size, 1)

    def forward(self, input):
        x1, x2 = input         # x1: B, F, S   x2: B, T
        x2_t = x2.unsqueeze(2) # x2: (B, T, 1)
        bn = self.bn(x1)       # Apply a batch normalization
        bn_t = bn.transpose(2, 1)  # B, S, F, needed for LSTM input
        _, hidden_encoded = self.rnn_encoder(bn_t)# output_enc (B, S, H),  2 x (num_layers, B, Hidden)
        decoder_out, _ = self.rnn_decoder(x2_t, hidden_encoded) # output: B, T, H
        output = self.out_shrink(decoder_out).squeeze(2) # Output: B, T
        return output



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



class PositionalEncoding(nn.Module):
    """
    Taken from pytorch: https://github.com/pytorch/tutorials/blob/f7d736060e5150d185ac5c75ef8a3625edebec60/beginner_source/transformer_tutorial.py#L100

    """
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        xlen = x.size(2)
        xfeat = x.size(1)
        pe = (self.pe[:xfeat, :xlen]).unsqueeze(0)
        x = x + pe
        return self.dropout(x)


class SimpleTransformer(nn.Module):
    """
    Simple transformer module. TransS from the paper

    """
    def __init__(self, in_features, past_steps, future_steps, d_model, dim_feedforward):
        """
        :param in_features: F
        :param past_steps:  S
        :param future_steps:  T
        :param d_model: Number of internal features in the Transformer
        :param dim_feedforward: No of units in the last layer of Transformer
        """
        super().__init__()
        self.in_features = in_features
        self.sample_len = past_steps
        self.output_size = future_steps
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model, max_len=max(past_steps, future_steps))
        self.trans = nn.Transformer(d_model, nhead=4, num_encoder_layers=3, num_decoder_layers=3,
                                    dim_feedforward=dim_feedforward,)
        self.input_expander = ConvLayer(ni=in_features, nf=d_model, ks=1, ndim=1)
        self.target_expander = ConvLayer(ni=1, nf=d_model, ks=1, ndim=1)
        self.out_contractor = nn.Conv1d(in_channels=d_model, out_channels=1, kernel_size=1)

    def forward(self, input):
        """
        data_in is a tuple x1, x2:
        x1 have size (BS, E1, S)
        x2 have size (BS, E2, T)
        where BS is batch size, Ex is the feature count E1 == in_features E2 == target_features
        S is sample_len and T is future_steps

        """
        x1, x2 = input  # x1: (B, F, S) x2:(B, T)
        x2_t = x2.unsqueeze(1) # x2_t: (B, 1, T)
        xinput, xtarget = (x1, x2_t)
        # Expand the input with a 1D convlayer applied featurewise
        xinput_exp = self.input_expander(xinput) # (B, d_model, S)
        xtarget_exp = self.target_expander(xtarget) # (B, d_model, T)
        # Add the positional encoding
        xinput_enc = self.pos_encoder(xinput_exp * math.sqrt(self.d_model))
        xtarget_enc = self.pos_encoder(xtarget_exp * math.sqrt(self.d_model))
        # Prepare for transformer
        xinput_enc = xinput_enc.permute(2, 0, 1) # (S, B, d_model)
        xtarget_enc = xtarget_enc.permute(2, 0, 1) # (T, B, d_model)
        output_trans = self.trans(xinput_enc, xtarget_enc) # (T, B, d_model)
        output_trans = output_trans.permute(1, 2, 0) # (B, d_model, T)
        # Reduce the output features to 1
        output = self.out_contractor(output_trans).squeeze(1) # (B, T)
        return output


class TransformerFull(nn.Module):
    def __init__(self, in_features, past_steps, future_steps, target_features, d_model, dim_feedforward):
        """
        Full transformer network. The input and the target is expanded to d_model feature size before being fed to the
        transformer block.

        At training, the known future is 1:1 but the target is shifted and masked.
        At evaluation the target contains zero and it is iteratively produced. That is, one element at a time.

        :param in_features: F
        :param past_steps:  S
        :param future_steps:  T
        :param target_features: F2, the number of target features (known and unknown) that will be fed into decoder. It is NOT y!
        :param d_model: Number of internal features in the Transformer
        :param dim_feedforward: No of units in the last layer of Transformer
        """
        super().__init__()
        self.in_features = in_features
        self.sample_len = past_steps
        self.output_size = future_steps
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model, max_len=past_steps)
        self.trans = nn.Transformer(d_model, nhead=4, num_encoder_layers=3, num_decoder_layers=3,
                                    dim_feedforward=dim_feedforward, )
        self.input_expander = ConvLayer(ni=in_features, nf=d_model, ks=1, ndim=1)
        self.target_expander = ConvLayer(ni=target_features, nf=d_model, ks=1, ndim=1)
        self.out_contractor = nn.Conv1d(in_channels=d_model, out_channels=1, kernel_size=1)

    def forward(self, data_in):
        """
        data_in is a tuple x1, x2:
        x1 have size (B, F, S)
        x2 have size (B, F2, T)
        where BS is batch size, Ex is the feature count E1 == in_features E2 == target_features
        S is sample_len and T is future_steps

        """
        xinput, xtarget = data_in  # x1 and augmented x2
        T = xtarget.size(2)
        BS = xtarget.size(0)
        xinput_exp = self.input_expander(xinput) # (B, d_model, S)
        xtarget_exp = self.target_expander(xtarget) # (B, d_model, T)
        xinput_enc = self.pos_encoder(xinput_exp * math.sqrt(self.d_model))
        xtarget_enc = self.pos_encoder(xtarget_exp * math.sqrt(self.d_model))
        xinput_enc = xinput_enc.permute(2, 0, 1) # (S, B, d_model)
        xtarget_enc = xtarget_enc.permute(2, 0, 1) # (T, B, d_model)
        tgt_mask = TransformerFull.gen_nopeek_mask(xtarget_enc.size(0)).to(xtarget.device)
        if self.training:
            output_trans = self.trans(xinput_enc, xtarget_enc, tgt_mask=tgt_mask) # (T, B, d_model)
        else:
            output_trans = torch.zeros(T, BS, self.d_model).to(xinput.device)
            temp_target_in = torch.zeros(T, BS, self.d_model).to(xinput.device)
            temp_target_in[0,:,:] = -1
            for k in range(T):
                temp_output = self.trans(xinput_enc, temp_target_in, tgt_mask=tgt_mask)
                output_trans[k,:, :] = temp_output[k,:,:]
                if k < T-1:
                    temp_target_in[k + 1,:,:] = temp_output[k,:,:].detach()
        output_trans = output_trans.permute(1, 2, 0) # (B, d_model, T)
        output = self.out_contractor(output_trans).squeeze(1) # (B, T)
        return output

    @staticmethod
    def gen_nopeek_mask(length):
        """
        Taken from pytorch: https://github.com/pytorch/tutorials/blob/f7d736060e5150d185ac5c75ef8a3625edebec60/beginner_source/transformer_tutorial.py

        :param length:
        :return:
        """
        mask = (torch.triu(torch.ones(length, length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

