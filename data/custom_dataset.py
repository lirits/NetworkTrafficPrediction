import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from utils.tools import sliding_windows, preprocessing
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


class LteTrafficDataset(Dataset):
    def __init__(
            self,
            file_path: str,
            CellName: str,
            windows_size: int,
            target_size: int,
            transform: bool = True,
            target_transform: bool = True):
        self.df = pd.read_csv(file_path)
        self.CellName = CellName
        self.transform = transform
        self.target_transform = target_transform
        self.windows_size = windows_size
        self.target_size = target_size
        self.dataframe = preprocessing(self.df, self.CellName)
        self.history_values, self.pred_values = sliding_windows(
            self.dataframe, self.dataframe.values, self.windows_size, self.target_size)

    def __len__(self):
        return len(self.history_values)

    def __getitem__(self, idx):
        history_value = self.history_values[idx]
        pred_value = self.pred_values[idx]
        if self.transform:
            history_value = torch.tensor(
                history_value, dtype=torch.float32) / self.dataframe.values.max()
        if self.target_transform:
            pred_value = torch.tensor(
                pred_value, dtype=torch.float32) / self.dataframe.values.max()
        return history_value, pred_value


def LteTrafficDataloader(file_path: str,
                         CellName: str,
                         windows_size: int,
                         target_size: int,
                         transform: bool = True,
                         target_transform: bool = True,
                         split_rate: float = 0.2,
                         batch_size: int = 64,
                         drop_last: bool = True):
    dataset = LteTrafficDataset(file_path, CellName, windows_size,
                                target_size, transform, target_transform)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(split_rate * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        sampler=SubsetRandomSampler(train_indices))
    validation_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        sampler=SubsetRandomSampler(val_indices))

    return train_loader, validation_loader


if __name__ == '__main__':
    file_path = '../dataset/LteTraffic/lte.train.csv'
    training_data = LteTrafficDataset(
        file_path,
        CellName='Cell_003781',
        windows_size=12,
        target_size=1,
        transform=True,
        target_transform=True)
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    X, y = next(iter(train_dataloader))
    print(X.shape, y.shape)
