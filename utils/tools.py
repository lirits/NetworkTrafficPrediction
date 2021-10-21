import pandas as pd
import torch
import wandb


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
        if len(time_values) - start_index < windows_size or len(time_values) - \
                start_index - windows_size < target_size:
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


def compute_dim(windows_size, padding, kernel_size, stride):
    """
    compute convolutional dimension
    :param windows_size:
    :param padding:
    :param kernel_size:
    :param stride:
    :return:
    """
    return int(((windows_size + 2 * padding - 1 *
               (kernel_size - 1) - 1) / stride) + 1)


def train_loop(
        loader,
        model,
        loss_fn,
        optimizer,
        device,
        usd_wandb: bool = False):
    # size = len(datalodaer.dataset)
    for batch, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)
        # compute prediction and loss
        pred = model(X)
        # loss = torch.sqrt(loss_fn(pred, y))
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            if usd_wandb:
                wandb.log({'train_loss': loss.item()})
            print(
                f"loss: {loss:>7f}") #,[{current:>5d}/{size:>5d}]


def test_loop(dataloader, model, loss_fn, device, usd_wandb: bool = False,Tr_rate:float = 0):
    num_batch = len(dataloader)
    test_loss = 0
    acc = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += torch.sqrt(loss_fn(pred, y)).item()
            if Tr_rate != 0:
                acc  += Tr_accuracy(pred,y,Tr_rate)

    test_loss /= num_batch
    if usd_wandb:
        wandb.log({'test_loss': test_loss.item(),'Tr_accuracy': acc/num_batch })
    print(f"Test Error: \n  Avg Mse loss: {test_loss:>8f} \n Tr Acc : {acc/num_batch} \n")


def Tr_accuracy(y_pred: float, y_true: float, tolerate_rate: int) -> float:
    return ((y_true * (1 - tolerate_rate) < y_pred) == (y_pred < (1 + tolerate_rate)
            * y_true)).sum() / (y_pred.shape[0] * y_pred.shape[1] * y_pred.shape[2])


def preprocess_prediction(net,dataloader,device):
    Y = []
    Y_pred = []
    with torch.no_grad():
        for X,y in dataloader:
            y_pred = net(X.to(device))
            for i in y[:,-1,0]:
                Y.append(i.to('cpu').detach())
            for j in y_pred[:,-1,0]:
                Y_pred.append(j.to('cpu').detach())
    return Y,Y_pred








if __name__ == '__main__':
    # import numpy as np
    # x = np.arange(100)
    # X, y = sliding_windows(x, x, 12, 1)
    # print('Test-sliding-windows-function')
    # print(len(X), '\n', X[0], y[0], '\n', X[1], y[1])
    # y_pred = torch.rand(64, 24, 1)
    y_true = torch.rand(64, 24, 1)
    # # print(Tr_accuracy(y_pred, y_true, 0.1))
    # print(y_pred[:,-1,0])

