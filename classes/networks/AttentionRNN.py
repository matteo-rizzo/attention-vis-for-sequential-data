import math
from typing import Union

import torch.nn.functional as F
from torch import nn

from params import *

""" Reference: "Attention Is All You Need" <https://arxiv.org/pdf/1706.03762.pdf> """


class AttentionRNN(nn.Module):

    def __init__(self, return_attention: bool = False):
        super().__init__()

        self.hidden_size = 2 * HIDDEN_SIZE if BIDIRECTIONAL else HIDDEN_SIZE
        self.num_layers = NUM_LAYERS
        self.scale = 1. / math.sqrt(2 * self.hidden_size if BIDIRECTIONAL else self.hidden_size)
        self.return_attention = return_attention

        rnn_cell = getattr(nn, RNN_TYPE)
        self.encoder = rnn_cell(input_size=INPUT_SIZE,
                                hidden_size=self.hidden_size,
                                num_layers=self.num_layers,
                                dropout=DROPOUT,
                                bidirectional=BIDIRECTIONAL,
                                batch_first=True)

        self.decoder = nn.Linear(2 * self.hidden_size if BIDIRECTIONAL else self.hidden_size, OUTPUT_SIZE)

    def __compute_attention(self, query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> tuple:
        """
        Attention assuming q_dim == k_dim (dot product attention)

        Let:
          - B: size of batch
          - T: number of timesteps
          - Q: size of query
          - K: size of keys
          - V: size of values

        @param query: [B x Q]
        @param keys: [B x T x K]
        @param values: [B x T x V]
        @returns linear combination: [B x V]
        """

        # [B x Q] -> [B x 1 x Q]
        query = query.unsqueeze(1)

        # [B x T x K] -> [T x B x K] -> [B x K x T]
        keys = keys.permute(1, 0, 2).transpose(0, 1).transpose(1, 2)

        # [B x 1 x Q] x [B x K x T] -> [B x 1 x T]
        a = torch.bmm(query, keys)

        # Scale and normalize ([T x B])
        a = F.softmax(a.mul_(self.scale), dim=2)

        # [B x T x V] -> [T x B x V] -> [B x T x V]
        values = values.permute(1, 0, 2).transpose(0, 1)

        # [B x 1 x T] x [B x T x V] -> [B x V]
        linear_combination = torch.bmm(a, values).squeeze(1)

        return a, linear_combination

    def init_state(self, batch_size: int) -> Union[tuple, torch.Tensor]:
        h = torch.zeros(2 * self.num_layers if BIDIRECTIONAL else self.num_layers, batch_size, self.hidden_size)
        return (h, h) if RNN_TYPE == "LSTM" else h

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Union[tuple, torch.Tensor]:
        """
        @param x: input timesteps of shape [batch_size, timesteps, num_features]
        @param h: hidden state
        """

        e, h = self.encoder(x, h)

        # Take cell state of LSTM
        if RNN_TYPE == "LSTM":
            h = h[1]

        # Concat the last 2 hidden layers for bidirectional encoder
        h = torch.cat([h[-1], h[-2]], dim=1) if BIDIRECTIONAL else h[-1]

        a, linear_combination = self.__compute_attention(h, e, e)
        o = self.decoder(linear_combination)

        if self.return_attention:
            return a, o

        return o
