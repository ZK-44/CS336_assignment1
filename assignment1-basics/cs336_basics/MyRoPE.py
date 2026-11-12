import torch
import math
import einops
class myRoPE(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        self.theta_inv = torch.zeros(max_seq_len, d_k // 2)
        for i in range(max_seq_len):
            for k in range(d_k // 2):
                self.theta_inv[i, k] = (i*theta**(-(2*k/d_k))) 

        self.R = torch.zeros(max_seq_len, d_k, d_k,device=device)
        for i in range(max_seq_len):
            for j in range(0,d_k,2):
                self.R[i, j, j] = torch.cos(self.theta_inv[i, j // 2])
                self.R[i,j,j+1] = -torch.sin(self.theta_inv[i, j // 2])
                self.R[i,j+1,j] = torch.sin(self.theta_inv[i, j // 2])
                self.R[i,j+1,j+1] = torch.cos(self.theta_inv[i, j // 2])
        self.register_buffer("R_matrix", self.R)
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        R = self.R_matrix[token_positions]       # (..., seq_len, d_k, d_k)
        in_dtype = x.dtype
        x = x.to(torch.float32)                  # 与 R 的 dtype 对齐
        out = einops.einsum(R, x, '... i j, ... j -> ... i')
        return out.to(in_dtype)
