import pandas as pd
import torch
import numpy as np


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
                import wandb
                wandb.log({'train_loss': loss.item()})
            print(
                f"loss: {loss:>7f}")  # ,[{current:>5d}/{size:>5d}]


def test_loop(
        dataloader,
        model,
        loss_fn,
        device,
        usd_wandb: bool = False,
        Tr_rate: float = 0):
    num_batch = len(dataloader)
    test_loss = 0
    acc = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            if Tr_rate != 0:
                acc += Tr_accuracy(pred, y, Tr_rate)

    test_loss /= num_batch
    if usd_wandb:
        import wandb
        wandb.log({'test_loss': test_loss.item(),
                  'Tr_accuracy': acc / num_batch})
    print(
        f"Test Error: \n  Avg Mse loss: {test_loss:>8f} \n Tr Acc : {acc/num_batch} \n")


def Tr_accuracy(y_pred: float, y_true: float, tolerate_rate: int) -> float:
    return ((y_true * (1 - tolerate_rate) < y_pred) == (y_pred < (1 + tolerate_rate)
            * y_true)).sum() / (y_pred.shape[0] * y_pred.shape[1] * y_pred.shape[2])

# copyright @
# https://github.com/Bjarten/early-stopping-pytorch/blob/master/MNIST_Early_Stopping_example.ipynb


def train_model(train_loader,
                valid_loader,
                net,
                optimizer,
                train_loss_fn,
                test_loss_fn,
                device,
                n_epochs,
                patience,
                use_acc: bool = False,
                Tr_rate: float = 0.2):
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    if use_acc:
        valid_acc = []
        avg_valid_acc = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(1, n_epochs + 1):

        ###################
        # train the model #
        ###################
        net.train()  # prep model for training
        for batch, (X, y) in enumerate(train_loader, 1):
            X, y = X.to(device), y.to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the
            # model
            output = net(X)
            # calculate the loss
            loss = train_loss_fn(output, y)
            # backward pass: compute gradient of the loss with respect to model
            # parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # record training loss
            train_losses.append(loss.item())

        ######################
        # validate the model #
        ######################
        net.eval()  # prep model for evaluation
        for X2, y2 in valid_loader:
            X2, y2 = X2.to(device), y2.to(device)
            # forward pass: compute predicted outputs by passing inputs to the
            # model
            output = net(X2)
            # calculate the loss
            loss = torch.sqrt(test_loss_fn(output, y2))
            # record validation loss
            valid_losses.append(loss.item())
            if use_acc:
                acc = Tr_accuracy(output, y2, Tr_rate)
                valid_acc.append(acc.item())

        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        valid_accuracy = np.average(valid_acc)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        if use_acc:
            avg_valid_acc.append(valid_accuracy)

        epoch_len = len(str(n_epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')

        print(print_msg)
        if use_acc:
            print(f'{Tr_rate*100}% ACC:{valid_accuracy:.5f}')

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        valid_acc = []

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, net)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # load the last checkpoint with the best model
    net.load_state_dict(torch.load('checkpoint.pt'))

    return net, avg_train_losses, avg_valid_losses, avg_valid_acc


def preprocess_prediction(net, dataloader, device):
    Y = []
    Y_pred = []
    net.eval()
    with torch.no_grad():
        for X, y in dataloader:
            y_pred = net(X.to(device))
            for i in y[:, -1, 0]:
                Y.append(i.to('cpu').detach())
            for j in y_pred[:, -1, 0]:
                Y_pred.append(j.to('cpu').detach())
    return Y, Y_pred

# copyright @ https://github.com/Bjarten/early-stopping-pytorch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
            self,
            patience=7,
            verbose=False,
            delta=0,
            path='checkpoint.pt',
            trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss



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