import torch
import torch.nn as nn


class DNNModule(nn.Module):
    def __init__(self, windows_size, target_size):
        super(DNNModule, self).__init__()
        self.windows_size = windows_size
        self.linear_stack = nn.Sequential(
            nn.Linear(self.windows_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, target_size),
        )

    def forward(self, x):
        x = torch.squeeze(x, dim=-1)
        target = self.linear_stack(x)
        return target