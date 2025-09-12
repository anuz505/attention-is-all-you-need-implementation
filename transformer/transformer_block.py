import torch.nn as nn
from multihead_attention import MultiHeadAttention 

class TransformerBlock(nn.Module):
    def __init__(self,e_dim,dropout,forward_expansion,n_heads):
        super(TransformerBlock,self).__init__()
        self.multiheadattention = MultiHeadAttention(e_dim, n_heads)
        self.norm1 = nn.LayerNorm(e_dim)
        self.norm2 = nn.LayerNorm(e_dim)

        self.FFN = nn.Sequential(
            nn.Linear(e_dim,forward_expansion * e_dim),
            nn.ReLU(),
            nn.Linear(e_dim * forward_expansion, e_dim)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,query,key,value,mask):
        attention = self.multiheadattention(value,key,query,mask)
        # shape = (N,seq_len,e_dim)
        n_norm1 = self.dropout(self.norm1(attention + query))
        FFN = self.FFN(n_norm1)
        output = self.dropout(self.norm2(FFN + n_norm1))
        return output