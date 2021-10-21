import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMModule(nn.Module):
    def __init__(
            self,
            window_size: int,
            input_feature: int = 1,
            target_size: int = 1,
            batch_size: int = 64,
            hidden_size: int = 20,
            num_lstm: int = 1,
            device: str = 'cpu',
            dropout: float = 0,
            bidirectional: bool = False):
        super(LSTMModule, self).__init__()
        self.LSTM = nn.LSTM(
            input_feature,
            hidden_size,
            num_lstm,
            dropout=dropout,
            bidirectional=bidirectional)
        self.hidden_memory, self.cell_memory = torch.zeros(
            num_lstm, batch_size, hidden_size).to(device), torch.zeros(
            num_lstm, batch_size, hidden_size).to(device)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(
            hidden_size * window_size, int(hidden_size * window_size / 2))
        self.linear2 = nn.Linear(
            int(hidden_size * window_size / 2), target_size)

    def forward(self, x):
        output, (_, _) = self.LSTM(x.transpose(0, 1),
                                   (self.hidden_memory, self.cell_memory))
        result = self.linear2(
            F.relu(
                self.linear1(
                    self.flatten(
                        output.transpose(
                            0,
                            1)))))
        return result.unsqueeze(-1)


if __name__ == '__main__':
    windows_size = 120
    target_size = 12
    input = torch.randn(64, windows_size, 1)
    net = LSTMModule(windows_size, 1, 48, 64, 200, 2)
    out = net(input)
    print(net)
    print(out.shape)
