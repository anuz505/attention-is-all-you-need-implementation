import torch.nn as nn
from multihead_attention import MultiHeadAttention 

class TransformerBlock:
    def __init__(self,e_dim,dropout,forward_expansion):
        super(TransformerBlock,self).__init__()
        self.multiheadattention = MultiHeadAttention()
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()

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
        FFN = self.FFN()
        output = self.dropout(self.norm2(FFN + n_norm1))
        return output