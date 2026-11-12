import torch
def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

def glu(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return x1 * torch.sigmoid(x2)


class custumGLU(torch.nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.linear1 = torch.nn.Linear(d_model,  d_model)
        self.linear2 = torch.nn.Linear(d_model,  d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.linear1(x)
        x2 = self.linear2(x)
        return x1 * torch.sigmoid(x2)

class custumSWIGLU(torch.nn.Module):
    def __init__(self, d_model: int,d_ff: int = None):
        super().__init__()
        if d_ff is not None:
            self.dff = d_ff
        else:
            self.dff = int(d_model * 8/3)
        self.w1 = torch.nn.Linear( d_model,self.dff,bias=False)
        self.w2 = torch.nn.Linear(self.dff,  d_model,bias=False)
        self.w3 = torch.nn.Linear( d_model,self.dff,bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = silu(self.w1(x))
        x3 = self.w3(x)
        return self.w2(x1 * x3)
           