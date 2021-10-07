import pandas as pd
import torch


def sliding_windows(
        time_values,
        data_values,
        windows_size: int,
        target_size: int,
        start_index=0) -> list:
    X = []
    y = []
    for i in range(len(time_values)):
        X.append(data_values[start_index:start_index + windows_size])
        y.append(data_values[start_index +
                             windows_size:start_index +
                             windows_size +
                             target_size])
        start_index += 1
        if len(time_values) - start_index < windows_size or len(time_values) - start_index-windows_size<target_size:
          break
    return X, y


def preprocessing(data_frame, CellName):
    data_frame = data_frame[data_frame['CellName'] == CellName].copy()
    data_frame.loc[:, 'Date'] = pd.to_datetime(data_frame.Date.astype(str))
    data_frame.loc[:, 'Hour'] = pd.to_timedelta(data_frame.Hour, unit='h')
    data_frame.loc[:, 'Date_time'] = pd.to_datetime(
        data_frame.Date + data_frame.Hour)
    data_frame = data_frame.drop(['Date', 'Hour', 'CellName'], axis=1)
    data_frame = data_frame.set_index('Date_time')
    data_frame = data_frame.sort_values(by='Date_time')
    return data_frame

def compute_dim(windows_size,padding,kernel_size,stride):
    """
    compute convolutional dimension
    :param windows_size:
    :param padding:
    :param kernel_size:
    :param stride:
    :return:
    """
    return int(((windows_size + 2 * padding - 1*(kernel_size-1) -1)/stride)+1)


def train_loop(datalodaer, model, loss_fn, optimizer,device):
    size = len(datalodaer.dataset)
    for batch, (X, y) in enumerate(datalodaer):
        X, y = X.to(device), y.to(device)
        # compute prediction and loss
        pred = model(X)
        loss = torch.sqrt(loss_fn(pred, y))
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn,device):
    size = len(dataloader.dataset)
    num_batch = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += torch.sqrt(loss_fn(pred, y)).item()

    test_loss /= num_batch

    print(f"Test Error: \n  Avg Mse loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    import numpy as np
    x = np.arange(100)
    X, y = sliding_windows(x, x, 12, 1)
    print('Test-sliding-windows-function')
    print(len(X), '\n', X[0], y[0], '\n', X[1], y[1])
