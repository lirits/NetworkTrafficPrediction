import wandb
import torch
from data.custom_dataset import LteTrafficDataloader
from model.Transformer import TransformerModule
from utils.tools import EarlyStopping,Tr_accuracy
import numpy as np
import math


sweep_config = {'method': 'random'}

metric = {
    'name': 'loss',
    'goal': 'minimize'
    }
sweep_config['metric'] = metric

parameters_dict = {
    'batch_size': {
        'distribution': 'q_log_uniform',
        'q': 1,
        'min': math.log(32),
        'max': math.log(256),
    },
    'windows_size' :{
        "values":[72,96,120,144,168]
    },
    'target_size':{
        'value':36
    },
    'learning_rate' :{
        'values':[1e-1,1e-2,1e-3,1e-4,1e-5]
    }
}
sweep_config['parameters'] = parameters_dict

parameters_dict.update({
    'optimizer':{
        'values': ['adam','sgd','radam']
    },
    'dropout':{
        'values': [0.05,0.1,0.15,0.5]
    },
    'input_dim':{
        'value': 1
    },
    'output_dim':{
        "min":0,
        "max":32
    },
    "embed_dim":{
        'values':[256,512,1024,1536,2048]
    },
    "nhead":{
        'values':[1,2,4,8,16]
    },
    "dim_hid":{
        'values':[256,512,1024,1536,2048]
    },
    "num_encoder":{
        'values': [1,2,3,4,5]
    },
    "num_decoder":{
        'values':[1,2,3,4,5]
    },
    "activation":{
        'values':['relu','gelu']
    },

    })
train_file_path = 'NetworkTrafficPrediction/data/dataset/LteTraffic/lte.train.csv'
CellName= 'Cell_003781'
device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
patience=10
Tr_rate=0.2
epochs=200
def train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        train_loader,valid_loader = LteTrafficDataloader(train_file_path, CellName, windows_size=config.windows_size, target_size=config.target_size, batch_size=config.batch_size)

        network = TransformerModule(config.input_dim, config.output_dim, config.embed_dim, config.nhead, config.dim_hid,
                                config.num_encoder, config.num_decoder, config.dropout, config.target_size,
                                config.activation).to(device)
        optimizer = build_optimizer(network, config.optimizer, config.learning_rate)


        # for epoch in range(config.epochs):
        model = train_model(train_loader,valid_loader,network,  optimizer,device,epochs,patience,Tr_rate)
        #     wandb.log({"loss": avg_loss, "epoch": epoch})
        return model



def build_optimizer(network, optimizer, learning_rate):
    if optimizer == "sgd":
        optimizer = torch.optim.SGD(network.parameters(),
                              lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = torch.optim.Adam(network.parameters(),
                               lr=learning_rate)
    elif optimizer == 'radam':
        from NetworkTrafficPrediction.utils.radam import RAdam
        optimizer = RAdam(network.parameters(),
                          lr=learning_rate)
    return optimizer


def train_model(train_loader,
                valid_loader,
                net,
                optimizer,
                device,
                n_epochs,
                patience,
                Tr_rate: float = 0.2):


    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(1, n_epochs + 1):
        train_losses = 0
        valid_losses = 0
        valid_acc = 0

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
            loss = torch.nn.L1Loss()(output, y)
            # backward pass: compute gradient of the loss with respect to model
            # parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # record training loss
            train_losses+=loss.item()

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
            loss = torch.sqrt(torch.nn.MSELoss()(output, y2))
            # record validation loss
            valid_losses +=loss.item()

            acc = Tr_accuracy(output, y2, Tr_rate)
            valid_acc+=acc.item()

        # print training/validation statistics
        # calculate average loss over an epoch
        wandb.log({'train_loss':train_losses/len(train_loader),'valid_loss':valid_losses/len(valid_loader),'Tr-accuracy':valid_acc/len(valid_loader)})

        epoch_len = len(str(n_epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'MAE: {(train_losses/len(train_loader)):.5f} ' +
                     f'MSE: {(valid_losses/len(valid_loader)):.5f}')

        print(print_msg)

        print(f'{Tr_rate*100}% ACC:{(valid_acc/len(valid_loader)):.5f}')

        # clear lists to track next epoch


        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping((valid_losses/len(valid_loader)), net)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # load the last checkpoint with the best model
    net.load_state_dict(torch.load('checkpoint.pt'))

    return net
if __name__ == '__main__':
    import pprint

    pprint.pprint(sweep_config)
    sweep_id = wandb.sweep(sweep_config, project="Network-Traffic-Prediction-sweeps")
    wandb.agent(sweep_id, train, count=1)