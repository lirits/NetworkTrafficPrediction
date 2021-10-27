import torch
import torch.nn as nn
from data.custom_dataset import LteTrafficDataloader
from utils.tools import train_model
import warnings
warnings.filterwarnings('ignore')


# global args
date_config = {
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
    'tr_rate': 0.2
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


def load_model(moid, config_file, device):
    assert moid in ['transformer', 'lstm', 'rnn', 'gru']
    if moid == 'transformer':
        from model.Transformer import TransformerModule

        net = TransformerModule(
            config_file['input_dim'],
            config_file['output_dim'],
            config_file['embed_dim'],
            config_file['nhead'],
            config_file['dim_hid'],
            config_file['num_encoder'],
            config_file['num_decoder'],
            config_file['dropout'],
            config_file['pred_size'],
            config_file['activation']).to(device)

    return net


def main(
        moid,
        data_config,
        model_config):
    train_loader, validation_loader = LteTrafficDataloader(data_config['file_path'], data_config['CellName'], data_config['windows_size'], data_config[
                                                           'target_size'], data_config['transform'], data_config['target_transform'], data_config['validation_split'], data_config['batch_size'])
    net = load_model(moid, model_config, data_config['device'])
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=data_config['learning_rate'])
    train_loss_fn = nn.L1Loss()
    valid_loss_fn = nn.MSELoss()
    model, train_loss, valid_loss, valid_acc = train_model(train_loader, validation_loader, net, optimizer, train_loss_fn, valid_loss_fn,
                                                           data_config['device'], data_config['epochs'], data_config['patience'], data_config['use_acc'], data_config['tr_rate'])
    return model, train_loss, valid_loss, valid_acc


if __name__ == '__main__':
    model, train_loss, valid_loss, valid_acc = main(
        'transformer', date_config, transformer_config)
