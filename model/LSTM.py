import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(
        self,
        window_size,
        input_feature=1,
        target_size=1,
        batch_size=64,
        hidden_size=20,
        num_lstm=1,
            device='cpu'):
        super(LSTM, self).__init__()
        self.LSTM = nn.LSTM(input_feature, hidden_size, num_lstm)
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
        output = self.linear2(
            F.relu(
                self.linear1(
                    self.flatten(
                        output.transpose(
                            0,
                            1)))))
        return output


if __name__ == '__main__':
    input = torch.randn(64, 12, 1)
    net = LSTM(window_size=12)
    out = net(input)
    print(out.shape)
