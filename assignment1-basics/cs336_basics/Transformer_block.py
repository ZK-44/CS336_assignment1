from cs336_basics.mutihead_self_attention import mutihead_self_attention_with_rope 
import cs336_basics.Myfunctions as Mf
import torch
from cs336_basics.forward import custumSWIGLU
from cs336_basics.fushuRoPE import RoPE 
from cs336_basics.MyRMSNorm import myRMSNorm
from cs336_basics.MyEmbedding import myEmbedding
class transformer(torch.nn.Module):
    def __init__(self,d_model:int ,num_heads:int ,d_ff:int,d_k,theta,max_seq_len,):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        #self.rope = RoPE(theta=theta,d_k=d_k,max_seq_len=max_seq_len)
        self.multi_r = mutihead_self_attention_with_rope(self.d_model,self.num_heads,max_seq_len=max_seq_len,theta=theta)
        self.rms_norm_1 = myRMSNorm(d_model=d_model)
        self.rms_norm_2 = myRMSNorm(d_model=d_model)
        self.swiglu = custumSWIGLU(d_model=d_model,d_ff=d_ff)
        
    def forward(self,x):
        #sublayer 1:Residual multiself_attention_with_rope(rms(x))
        y_1 = x + self.multi_r.forward(self.rms_norm_1.forward(x))
        #sublayer 2 :FFN
        y_2 = y_1 + self.swiglu.forward(self.rms_norm_2.forward(y_1))
        return y_2



class transformer_lm(torch.nn.Module):
    def __init__(self,vocab_size:int,context_length:int,d_model:int ,num_heads:int ,d_ff:int,d_k,theta,num_layers):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.embedding = myEmbedding(num_embeddings=vocab_size,embedding_dim=d_model)
        self.transformer_blocks = torch.nn.ModuleList()
        for i in range(num_layers):
            self.transformer_blocks.append(transformer(d_model=d_model,num_heads=num_heads,d_ff=d_ff,d_k=d_k,theta=theta,max_seq_len=context_length))
        self.norm = myRMSNorm(d_model=d_model)
        self.linear = torch.nn.Linear(d_model,vocab_size,bias=False)

    def forward(self,x):
        x = self.embedding.forward(x)
        for i in range(self.num_layers):
            x = self.transformer_blocks[i].forward(x)
        x = self.norm.forward(x)
        x = self.linear.forward(x)
        #x = Mf.safe_softmax(self.linear.forward(x))
        return x