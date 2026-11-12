import torch
import einops
import cs336_basics.Myfunctions as Mf
import torch.nn as nn
from cs336_basics.fushuRoPE import RoPE as ROPE
class mutihead_self_attention(torch.nn.Module):
    def __init__(self, d_model:int,num_heads:int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads 
        self.W_q = nn.Linear(d_model, d_model,bias=False)
        self.W_k = nn.Linear(d_model, d_model,bias=False)
        self.W_v = nn.Linear(d_model, d_model,bias=False)
        self.W_o = nn.Linear(d_model, d_model,bias=False)

    def forward(self,x:torch.tensor,mask=None,):
        
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        Q = einops.rearrange(Q, '... t (h d) -> ... h t d', h=self.num_heads)
        K = einops.rearrange(K, '... t (h d) -> ... h t d', h=self.num_heads)
        V = einops.rearrange(V, '... t (h d) -> ... h t d', h=self.num_heads)
        mask = torch.tril(torch.ones(Q.shape[-2],Q.shape[-2],dtype=torch.bool))
        context=Mf.scaled_dot_product_attention(Q,K,V,mask)
        out = einops.rearrange(context, '... h t d -> ... t (h d)')
        return self.W_o(out)

class mutihead_self_attention_with_rope(torch.nn.Module):
    def __init__(self, d_model:int,num_heads:int,max_seq_len:int,theta:float):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.max_seq_len = max_seq_len 
        self.theta = theta
        #self.token_positions = token_positions
        self.W_q = nn.Linear(d_model, d_model,bias=False)
        self.W_k = nn.Linear(d_model, d_model,bias=False)
        self.W_v = nn.Linear(d_model, d_model,bias=False)
        self.W_o = nn.Linear(d_model, d_model,bias=False)
        self.rope_f = ROPE(theta=self.theta,d_k=self.d_k,max_seq_len=self.max_seq_len)

    def forward(self,x:torch.tensor,mask=None,token_positions=None):
        #rope_f = ROPE(theta=self.theta,d_k=self.d_k,max_seq_len=self.max_seq_len)
        if token_positions is None:
            token_positions = torch.arange(x.shape[-2], device=x.device)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        Q = einops.rearrange(Q, '... t (h d) -> ... h t d', h=self.num_heads)
        K = einops.rearrange(K, '... t (h d) -> ... h t d', h=self.num_heads)
        V = einops.rearrange(V, '... t (h d) -> ... h t d', h=self.num_heads)
        Q = self.rope_f.forward(Q,token_positions)
        K = self.rope_f.forward(K,token_positions)
        mask = torch.tril(torch.ones(Q.shape[-2],Q.shape[-2],dtype=torch.bool,device=x.device))
        context=Mf.scaled_dot_product_attention(Q,K,V,mask)
        out = einops.rearrange(context, '... h t d -> ... t (h d)')
        return self.W_o(out)