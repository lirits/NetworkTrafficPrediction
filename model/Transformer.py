import torch
import torch.nn as nn
import math
from torch import Tensor

# input_shape:[batch,windows_size,values]


class Transformer(nn.Module):
    def __init__(self,input_dim,output_dim,embed_dim,nhead,dim_hid,num_encoder,num_decoder,dropout):
        super(Transformer,self).__init__()
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.encoder_embedding = ValueEmbedding(input_dim,embed_dim,'encoder')
        self.decoder_embedding = ValueEmbedding(input_dim,embed_dim,'decoder',seq_decoder=output_dim)
        self.pos_embedding = PositionalEncoding(embed_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim,nhead=nhead,dim_feedforward=dim_hid,dropout=dropout)
        self.Encoder = nn.TransformerEncoder(encoder_layers,num_layers=num_encoder)
        decoder_layers = nn.TransformerDecoderLayer(d_model=embed_dim,nhead=nhead,dim_feedforward=dim_hid,dropout=dropout)
        self.Decoder =nn.TransformerDecoder(decoder_layers,num_decoder)
        self.fc1 = nn.Linear(embed_dim*output_dim,1)



    def forward(self,x):
        x_encoder = self.pos_embedding(self.encoder_embedding(x).transpose(0,1))
        x_decoder = self.pos_embedding(self.decoder_embedding(x).transpose(0,1))
        out_encoder = self.Encoder(x_encoder)
        out_decoder = self.Decoder(x_decoder,out_encoder)
        out = self.fc1(out_decoder.transpose(0,1).reshape(-1,self.embed_dim*self.output_dim))
        return out


class ValueEmbedding(nn.Module):
    def __init__(self,input_dim,embed_dim,coder,seq_decoder=2):
        super(ValueEmbedding,self).__init__()
        assert coder in ['encoder','decoder']
        self.coder = coder
        self.seq_decoder = seq_decoder
        self.embedding = nn.Linear(input_dim,embed_dim)

    def forward(self,x):
        if self.coder is 'decoder':
            x = x[:, :self.seq_decoder, :]
        x = self.embedding(x)
        return x

class PositionalEncoding(nn.Module):

    def __init__(
            self,
            d_model: int,
            dropout: float = 0.1,
            max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
