import torch
import torch.nn as nn
from multihead_attention import MultiHeadAttention 
from transformer_block import TransformerBlock
class DecoderBlock(nn.Module):
    def __init__(self,e_dim,n_heads,forward_expansion,dropout,device):
        super(DecoderBlock,self).__init__()
        self.attention = MultiHeadAttention(e_dim,n_heads)
        self.norm = nn.LayerNorm(e_dim)
        self.transformerblock = TransformerBlock(e_dim,dropout,forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,key,value,src_mask,target_mask):
        attention = self.attention(x,x,x,target_mask)
        query = self.dropout(self.norm(attention + x))
        output = self.transformerblock(value,key,query,src_mask)
        return output


class MainDecoder(nn.Module):
    def __init__(self,traget_vocab,e_dim,n_layers,n_heads,froward_exapansion,dropout,device,max_len):
        super(MainDecoder,self).__init__()
        self.device = device
        self.word_embed = nn.Embedding(traget_vocab,e_dim)
        self.positions = nn.Embedding(max_len,e_dim)
        self.layers = nn.ModuleList(
            [
                DecoderBlock(e_dim,n_heads,froward_exapansion,dropout,device)
                for _ in range(n_layers)
            ]
        )
        self.ffn = nn.Linear(e_dim,traget_vocab)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,enc_output,src_mask,target_mask):
        N,seq_len = x.shape()
        positions = torch.arange(0,seq_len).expand(N,seq_len).to(device=self.device)
        x = self.dropout((self.word_embed(x)+self.positions(positions)))

        for layer in self.layers:
            x = layer(x,enc_output,enc_output,src_mask,target_mask)
    
        out = self.ffn(x)

        return out