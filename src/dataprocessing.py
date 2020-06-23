from fastai2.vision.all import *
import torch
"""
Data Loaders and preprocessing

"""

class HeatingItemiser(ItemTransform):
    """
    Itemizer, used internally by fast.ai

    Returns (x1, x2), y. Where x1 have shape: (F, S), x2: (T), y2: (T) or x2:(2, T) if transformer is True.

    """
    def __init__(self, future_len, command_column=12, temperature_column=13, transformer=False):
        """

        :param future_len: How many samples, at the end of each input, is set as "unknown"
        :param command_column: What is the known future? That is x2 value
        :param temperature_column: What is the unknown future? That needs to be predicted
        :param transformer: If true, the x2 will have two features, with the target appended and shifted to the right.
        """
        super().__init__()
        self.future_len = future_len
        self.temperature_column = temperature_column
        self.command_column = command_column
        self.average_over = 0.4
        self.transformer=transformer

    def encodes(self, input):
        """
        From a 2D sample, cuts out (x1, x2) vectors and target y. Creates tensors from them and returns them.

        Used internally by fastai2 library

        :param input:
        :return:
        """
        x1 = input[:-self.future_len, :]
        x2 = input[-self.future_len:, self.command_column]
        past_temperature = x1[:, self.temperature_column]
        future_temperature = input[-self.future_len:, self.temperature_column]
        avg_temp = self.get_value_to_substract(past_temperature)
        target_y = future_temperature - avg_temp
        if self.transformer:
            target_shift = np.copy(target_y)
            target_shift[1:] = target_shift[0:-1]
            target_shift[0] = 0
            x2_out = np.vstack([x2, target_shift])
        else:
            x2_out = x2
        x1t = tensor(x1).float().transpose(0, 1)
        x2t = tensor(x2_out).float()
        yt = tensor(target_y).float()
        return ((x1t, x2t), yt)

    def decodes(self, data):
        """
        From a tuple of tensors ((x1, x2), y) get back a tuple of tensors but with some inverse modifications
        from encodes().

        Transpose x1,
        add to y the average of last x% temperature values from x1.

        :param data:
        :return:
        """
        x12, y = data
        x1 = x12[0].transpose(0, 1).numpy()
        x2 = x12[1].numpy()
        y = y.numpy()
        past_temperature = x1[:, self.temperature_column]
        avg_temp = self.get_value_to_substract(past_temperature)
        y_out = y + avg_temp
        return (x1, x2), y_out

    def get_value_to_substract(self, past_data):
        N = past_data.shape[0]
        RG_len = int(N * self.average_over)
        RG = N - RG_len
        avg_temp = 0
        avg_temp = np.nanmean(past_data[RG:N])
        return avg_temp

