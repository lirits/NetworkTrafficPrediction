import torch
import torch.nn as nn
from data.custom_dataset import LteTrafficDataloader
from utils.tools import train_model
import warnings
warnings.filterwarnings('ignore')


# global args
data_config = {
    'file_path': 'dataset/LteTraffic/lte.train.csv',
    'CellName': 'Cell_003781',
    'batch_size': 64,
    'windows_size': 128,
    'target_size': 32,
    'learning_rate': 1e-4,
    'epochs': 200,
    'transform': True,
    'target_transform': True,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'validation_split': .2,
    'patience': 10,
    'use_acc': True,
    'tr_rate': 0.2,
    'train_loss_fn': 'mae',  # mse,mae
    'test_loss_fn': 'rmse',  # mse,mae
    'optimizer': 'radam'  # Adam,sgd,AdamW
}

transformer_config = {
    'input_dim': 1,
    'output_dim': 12,  # input_size of decoder
    'embed_dim': 1024,  # ValueEmbedding
    'nhead': 8,
    'dim_hid': 1024,  # the dimension of the feedforward network model
    'num_encoder': 2,
    'num_decoder': 1,
    'dropout': 0.05,
    'activation': 'gelu',
    'pred_size': 32}

simple_Rnns_config = {
    'mode': 'RNN',
    'window_size': 12,
    'input_feature': 1,
    'target_size': 1,
    'hidden_size': 20,
    'num_lstm': 1,
    'dropout': 0.01,
    'bidirectional': False}


def load_model(mode, model_config, device):
    assert mode in ['Transformer', 'LSTM', 'RNN', 'GRU']
    if mode == 'Transformer':
        from model.Transformer import TransformerModule

        net = TransformerModule(
            model_config['input_dim'],
            model_config['output_dim'],
            model_config['embed_dim'],
            model_config['nhead'],
            model_config['dim_hid'],
            model_config['num_encoder'],
            model_config['num_decoder'],
            model_config['dropout'],
            model_config['pred_size'],
            model_config['activation']).to(device)
    else:
        from model.Simple_RNNs import RnnsModule
        net = RnnsModule(
            model_config['mode'],
            model_config['window_size'],
            model_config['input_feature'],
            model_config['target_size'],
            model_config['hidden_size'],
            model_config['num_lstm'],
            model_config['dropout'],
            model_config['bidirectional']
        ).to(device)
    return net


def build_model(model, data_config):
    assert data_config['optimizer'] in ['adam', 'sgd', 'adamW', 'radam']
    assert data_config['train_loss_fn'] in ['rmse', 'mae']
    assert data_config['test_loss_fn'] in ['rmse', 'mae']
    if data_config['optimizer'] == 'radam':
        from utils.radam import RAdam
        optimizer = RAdam(lr=data_config['learning_rate'])
    elif data_config['optimizer'] == 'adamW':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=data_config['learning_rate'])
    elif data_config['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=data_config['learning_rate'])
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=data_config['learning_rate'])

    train_loss_fn = torch.nn.L1Loss(
    ) if data_config['train_loss_fn'] == 'mae' else torch.nn.MSELoss()
    test_loss_fn = torch.nn.L1Loss(
    ) if data_config['test_loss_fn'] == 'mae' else torch.nn.MSELoss()
    return optimizer, train_loss_fn, test_loss_fn


def train(
        mode,
        data_config,
        model_config):
    train_loader, validation_loader = LteTrafficDataloader(data_config['file_path'], data_config['CellName'], data_config['windows_size'], data_config[
                                                           'target_size'], data_config['transform'], data_config['target_transform'], data_config['validation_split'], data_config['batch_size'])
    net = load_model(mode, model_config, data_config['device'])
    optimizer, train_loss_fn, test_loss_fn = build_model(net, data_config)
    # optimizer = torch.optim.Adam(
    #     net.parameters(),
    #     lr=data_config['learning_rate'])
    # train_loss_fn = nn.L1Loss()
    # valid_loss_fn = nn.MSELoss()
    model, train_loss, valid_loss, valid_acc = train_model(train_loader, validation_loader, net, optimizer, train_loss_fn, test_loss_fn,
                                                           data_config['device'], data_config['epochs'], data_config['patience'], data_config['use_acc'], data_config['tr_rate'])
    return model, train_loss, valid_loss, valid_acc, train_loader, validation_loader


if __name__ == '__main__':
    model, train_loss, valid_loss, valid_acc, train_loader, validation_loader = train(
        'transformer', data_config, transformer_config)
