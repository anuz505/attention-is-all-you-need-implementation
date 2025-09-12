import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, e_dim, n_heads):
        super(MultiHeadAttention,self).__init__()
        self.e_dim = e_dim # original paper 512
        self.n_heads = n_heads
        self.head_dim = e_dim // n_heads

        assert (self.head_dim * n_heads == e_dim), "Embed size needs to be div by heads"

        self.values = nn.Linear(e_dim, e_dim, bias=False)
        self.keys = nn.Linear(e_dim, e_dim, bias=False)
        self.queries = nn.Linear(e_dim, e_dim, bias=False)
        self.fc_output = nn.Linear(n_heads * self.head_dim, e_dim)
    
    def forward(self,value,key,query,mask):
        N = query.shape[0] # batch
        value_seq_len, key_seq_len, query_seq_len = value.shape[1],key.shape[1],query.shape[1]

        values = self.values(value) # (N, seq_len, e_dim)
        keys = self.keys(key)
        queries = self.queries(query)

        # split embeddings into self.heads pieces       
        values = values.reshape(N, value_seq_len, self.n_heads, self.head_dim) 
        keys = keys.reshape(N, key_seq_len, self.n_heads, self.head_dim)
        queries = queries.reshape(N, query_seq_len, self.n_heads, self.head_dim)

        # attention score calc
        q_k_mult = torch.einsum("nqhd,nkhd->nhqk",[queries,keys]) #nhqk each query compares itself with all keys

        if mask is not None:
            q_k_mult = q_k_mult.masked_fill(mask ==0, float("-1e20")) #set the score to -inf.

        attention_score = torch.softmax(q_k_mult /(self.e_dim ** (1/2)),dim=3)  # scaling and softmax
        # attention_score_shape = (N,n_heads,query_seq_len,key_seq_len)
        # Since key_seq_len == value_seq_len, we can use k for both
        output = torch.einsum("nhqk,nkhd->nqhd",[attention_score,values]).reshape(N,query_seq_len,self.n_heads * self.head_dim)
        # output_shape (N,query_seq_len,n_head,head_dim)
        # we reshape and flatten the last two dim
        # 
        output = self.fc_output(output)
        # final_shape = (N,query_len,e_dim)

        return output  
    

