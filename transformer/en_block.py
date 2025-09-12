import torch.nn as nn
from transformer_block import TransformerBlock
import torch
class Encoder(nn.Module):
    def __init__(self,vocab_size,e_dim, n_layers,dropout,device,n_heads,forward_expansion,max_len):
        super(Encoder,self).__init__()
        self.e_dim = e_dim
        self.device = device
        self.word_embed = nn.Embedding(vocab_size,e_dim)
        self.pos_embed = nn.Embedding(max_len,e_dim)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(e_dim=e_dim,dropout=dropout,forward_expansion=forward_expansion,n_heads=n_heads)
                for _ in range(n_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,mask):
        N,seq_len = x.shape
        positions = torch.arange(0,seq_len).expand(N,seq_len).to(device=self.device)
        out = self.dropout(self.word_embed(x)+self.pos_embed(positions))

        for layer in self.layers:
            out = layer(out,out,out,mask)
        return out