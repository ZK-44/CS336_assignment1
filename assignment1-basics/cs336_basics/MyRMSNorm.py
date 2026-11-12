import torch
class myRMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        result = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) 
        return (x * result * self.weight).to(in_dtype)