import torch
import torch.nn as nn


class LSTMModule(nn.Module):
    def __init__(
            self,
            moid = 'RNN',
            window_size: int = 12,
            input_feature: int = 1,
            target_size: int = 1,
            hidden_size: int = 20,
            num_lstm: int = 1,
            dropout: float = 0.01,
            bidirectional:bool = False):
        super(LSTMModule, self).__init__()
        assert moid in ['RNN','LSTM','GRU']
        if moid == 'LSTM':
            self.RNNS_model = nn.Sequential(
            nn.LSTM(input_feature,hidden_size,num_lstm,bidirectional=bidirectional)
        )
        elif moid == 'GRU':
            self.RNNS_model = nn.Sequential(
                nn.GRU(input_feature, hidden_size, num_lstm, bidirectional=bidirectional)
            )
        else:
            self.RNNS_model = nn.Sequential(
                nn.LSTM(input_feature, hidden_size, num_lstm, bidirectional=bidirectional)
            )
        self.fc1 = nn.Flatten()


        self.linear_model = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(window_size * hidden_size,int(window_size*hidden_size / 2)),
            nn.ReLU(),
            nn.Linear(int(window_size*hidden_size / 2),target_size),
        )

    def forward(self, x):
        output_rnn,_ = self.RNNS_model(x.transpose(0,1))
        output = self.linear_model(self.fc1(output_rnn.transpose(0,1)))
        return output


if __name__ == '__main__':
    windows_size = 120
    target_size = 12
    moid = 'LSTM'
    input = torch.randn(64,windows_size, 1)
    net = LSTMModule('LSTM',windows_size,1,48,200,3,0.5)
    out = net(input)
    print(out.shape)