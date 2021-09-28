import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from NetworkTrafficPrediction.utils.tools import sliding_windows, preprocessing


class LteTrafficDataset(Dataset):
    def __init__(
            self,
            file_path: str,
            CellName: str,
            windows_size: int,
            target_size: int,
            transform=None,
            target_transform=None):
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
