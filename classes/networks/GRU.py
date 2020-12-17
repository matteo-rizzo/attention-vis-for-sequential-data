from torch import nn

from params import *


class GRU(nn.Module):

    def __init__(self):
        super().__init__()

        self.hidden_size = HIDDEN_SIZE
        self.num_layers = NUM_LAYERS

        self.gru = nn.GRU(input_size=INPUT_SIZE,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          bidirectional=BIDIRECTIONAL,
                          dropout=DROPOUT,
                          batch_first=True)

        self.fc = nn.Linear(2 * self.hidden_size if BIDIRECTIONAL else self.hidden_size, OUTPUT_SIZE)

    def init_state(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(2 * self.num_layers if BIDIRECTIONAL else self.num_layers, batch_size, self.hidden_size)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        @param x: input sequence of shape [batch_size, sequence_length, num_features]
        @param h: hidden state of shape [num_layers (2x if bidirectional), batch_size, hidden_size]
        """
        o, _ = self.gru(x, h)
        return self.fc(o)[:, -1, :]
