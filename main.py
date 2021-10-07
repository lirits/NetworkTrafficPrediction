import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.custom_dataset import LteTrafficDataset
from utils.tools import train_loop,test_loop

# global args
train_file_path = '/tmp/data/lte.test.csv' # path of training dataset
test_file_path = '/tmp/data/lte.test.csv' # path of training dataset
CellName = 'Cell_003781' # Name of Lte Cell
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform,target_transform = True,True # Regularization
batch_size = 64
window_size = 12
target_size = 1
input_feature = 1
transform,target_transform = True,True
learning_rate = 1e-4
epochs= 50

# LSTM model args
model = 'Transformer'
hidden_size = 200
num_lstm = 1

# Transformer model args
input_dim = 1
output_dim = target_size -1 if target_size != 1 else target_size
embed_dim = 512
nhead = 8
dim_hid = 512
num_encoder = 2
num_decoder = 1
dropout = 0.05

assert model in ['Transformer','LSTM','DNN']

if model == 'LSTM':
  from model.LSTM import LSTMModule
  net = LSTMModule(window_size=window_size,input_feature=input_feature,target_size=target_size,batch_size=batch_size,hidden_size=200,num_lstm=num_lstm,device=device)
elif model == 'Transformer':
  from model.Transformer import TransformerModule
  net = TransformerModule(input_dim,output_dim,embed_dim,nhead,dim_hid,num_encoder,num_decoder,dropout)
elif model == 'DNN':
  from model.DNN import DNNModule
  net = DNNModule(window_size,target_size)


training_data = LteTrafficDataset(train_file_path,CellName=CellName,windows_size=window_size,target_size=target_size,transform=transform,target_transform=target_transform)
train_dataloader = DataLoader(training_data,batch_size=batch_size,shuffle=True,drop_last=True)
test_data = LteTrafficDataset(test_file_path,CellName,window_size,target_size,transform,target_transform)
test_dataloader = DataLoader(test_data,batch_size=batch_size,shuffle=False,drop_last=True)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)

if __name__ == '__main__':
  for t in range(epochs):
    print(f'Epoch {t+1}\n-------------------------------------')
    train_loop(train_dataloader,net,loss_fn,optimizer,device)
    test_loop(test_dataloader,net,loss_fn,device)