import einops
import torch
import torch.nn as nn
import numpy as np
import math

class myLinear(torch.nn.Module):
    def __init__(self,in_features,out_features,device=None,dtype=None):
        super(myLinear,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        std = math.sqrt(2.0 / (in_features + out_features))
        #weight = torch.empty(out_features, in_features )
        weight = torch.empty(out_features, in_features, device=device, dtype=dtype)
        nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3.0*std, b=3.0*std)
        self.weight = torch.nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(x.shape, self.weight.shape)
        return einops.einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")
        #return self.weight @ x