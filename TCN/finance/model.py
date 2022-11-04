import torch
from torch import nn
# import sys
# sys.path.append("../../")
from TCN.tcn import TemporalConvNet


class TCN(nn.Module):

    def __init__(self, input_size, output_size, num_channels,
                 kernel_size=2, dropout=0.3,  tied_weights=False):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)

        self.decoder = nn.Linear(num_channels[-1], output_size)
        if tied_weights:
            if num_channels[-1] != input_size:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight
            print("Weight tied")
        self.init_weights()

    def init_weights(self):
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(0, 0.01)

    def forward(self, input):
        """Input ought to have dimension (N, C_in, L_in), where L_in is the seq_len; here the input is (N, L, C)"""
        y = self.tcn(input)
        y = self.decoder(y.transpose(1,2))
        return y.contiguous().squeeze(-1)

