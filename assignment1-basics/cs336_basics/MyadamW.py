import math
import torch
from torch.optim import Optimizer


class myAdamW(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            lam = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("不支持稀疏梯度")

                state = self.state[p]
                if len(state) == 0:                     # 伪代码第 2、3 行
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p)    # 一阶矩向量 m
                    state['v'] = torch.zeros_like(p)    # 二阶矩向量 v

                m, v = state['m'], state['v']
                state['step'] += 1                      # 伪代码第 4 行: t
                t = state['step']

                # 伪代码第 7 行: α_t ← α·√(1-β₂ᵗ)/(1-β₁ᵗ)
                alpha_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)

                # 伪代码第 8 行: θ ← θ - αλθ
                if lam != 0:
                    p.mul_(1 - lr * lam)

                # 伪代码第 9、10 行: 矩估计更新
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                # 伪代码第 11 行: θ ← θ - α_t·m/(√v + ε)
                denom = v.sqrt().add_(eps)
                p.addcdiv_(m, denom, value=-alpha_t)

        return loss