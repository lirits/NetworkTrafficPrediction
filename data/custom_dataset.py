import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import sys
sys.path.append("..")
from utils.tools import sliding_windows, preprocessing


class LteTrafficDataset(Dataset):
    def __init__(self, file_path, CellName, windows_size, pred_size, transform=None, target_transform=None):
        self.df = pd.read_csv(file_path)
        self.CellName = CellName
        self.transform = transform
        self.target_transform = target_transform
        self.windows_size = windows_size
        self.pred_size = pred_size
        self.dataframe = preprocessing(self.df,self.CellName)
        self.history_values, self.pred_values = sliding_windows(self.dataframe, self.dataframe.values,
                                                                self.windows_size, self.pred_size)

    def __len__(self):
        return len(self.history_values)

    def __getitem__(self, idx):
        history_value = self.history_values[idx]
        pred_value = self.pred_values[idx]
        if self.transform:
            history_value = torch.tensor(history_value,dtype=torch.float32) / self.dataframe.max()
        if self.target_transform:
            pred_value = torch.tensor(pred_value,dtype=torch.float32) / self.dataframe.max()
        return history_value,pred_value
