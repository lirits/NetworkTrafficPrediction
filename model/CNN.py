import torch
from torch import nn
import torch.nn.functional as F
from NetworkTrafficPrediction.utils.tools import compute_dim
# Unfinished
# [b,windows_size,embed_dim]


class CNN1d(nn.Module):

    def __init__(
            self,
            windows_size,
            input_dim=1,
            output_dim=1,
            kernel_size=2,
            stride=2,
            padding=0,
            kernel_size_Maxpool=3,
            stride_Maxpool=2):
        super(CNN1d, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=int(
                input_dim * 2),
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)
        self.conv2 = nn.Conv1d(
            in_channels=int(
                input_dim * 2),
            out_channels=int(
                input_dim * 4),
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=stride)
        self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=stride)
        self.flatten = nn.Flatten()
        self.dim = compute_dim(
            compute_dim(
                compute_dim(
                    compute_dim(
                        windows_size,
                        padding,
                        kernel_size,
                        stride),
                    padding,
                    kernel_size_Maxpool,
                    stride_Maxpool),
                padding,
                kernel_size,
                stride),
            padding,
            kernel_size_Maxpool,
            stride_Maxpool)
        # print(self.dim,type(self.dim))
        self.fc1 = nn.Linear(in_features=self.dim *
                             int(input_dim * 4), out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=output_dim)

    def forward(self, x):
        x = self.maxpool2(
            F.relu(
                self.conv2(
                    self.maxpool1(
                        F.relu(
                            self.conv1(
                                x.transpose(
                                    1,
                                    2)))))))
        out = self.fc3(F.relu(self.fc2(F.relu(self.fc1(self.flatten(x))))))
        return out


if __name__ == '__main__':
    windows_size = 48
    cnn1 = CNN1d(windows_size=windows_size)
    input = torch.rand(64, windows_size, 1)
    output1 = cnn1(input)
    print(output1.shape)
