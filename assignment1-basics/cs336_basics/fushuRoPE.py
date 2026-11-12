import torch
import torch.nn as nn


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        assert d_k % 2 == 0
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # 唯一需要持久化的东西：[d_k/2]，theta^(-2k/d_k)
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=True)

        # cos/sin 缓存，按需扩展，不进 state_dict
        self.register_buffer("cos_cached", torch.zeros(0, d_k // 2), persistent=False)
        self.register_buffer("sin_cached", torch.zeros(0, d_k // 2), persistent=False)

    @torch.no_grad()
    def _build_cache(self, seq_len: int, device: torch.device):
        """缓存不足时增量扩展，最多扩到 max_seq_len"""
        if seq_len <= self.cos_cached.shape[0] and self.cos_cached.device == device:
            return
        t = torch.arange(self.cos_cached.shape[0], seq_len,
                         device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))   # [new, d_k/2]
        self.cos_cached = torch.cat([self.cos_cached.to(device), freqs.cos()])
        self.sin_cached = torch.cat([self.sin_cached.to(device), freqs.sin()])

    

    def forward(self,x,token_positions):
        seq_len = x.shape[-2]
        assert seq_len <= self.max_seq_len
        self._build_cache(seq_len, x.device)
        cos = self.cos_cached[token_positions]   # (..., seq_len, d_k/2)
        sin = self.sin_cached[token_positions]
        in_dtype = x.dtype
        x = x.to(torch.float32)
        x_even = x[..., 0::2]   # 偶数维 x_{2k}
        x_odd = x[..., 1::2]    # 奇数维 x_{2k+1}
        
        # 对应 R @ [x_even, x_odd] 的逐元素展开
        out_even = x_even * cos - x_odd * sin
        out_odd  = x_even * sin + x_odd * cos
        
        out = torch.stack([out_even, out_odd], dim=-1).flatten(-2)
        return out.to(in_dtype)