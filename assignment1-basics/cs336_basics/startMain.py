from __future__ import annotations
import os
import math
import time
from collections.abc import Iterable
from typing import IO, Any, BinaryIO
import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from cs336_basics.bpe_tokenization import bpe_tokenizer
from cs336_basics.MyLinear import myLinear
from cs336_basics.MyEmbedding import myEmbedding
from cs336_basics.MyRMSNorm import myRMSNorm
import cs336_basics.forward as Ff
from cs336_basics.fushuRoPE import RoPE
import cs336_basics.Myfunctions as Mf
from cs336_basics.mutihead_self_attention import mutihead_self_attention
from cs336_basics.mutihead_self_attention import mutihead_self_attention_with_rope
from cs336_basics.Transformer_block import transformer
from cs336_basics.Transformer_block import transformer_lm
from cs336_basics.MyadamW import myAdamW
import numpy as np
from pathlib import Path
# ---------------------------------------------------------------
# 超参数（截图 7.2.1 的推荐初始值，调参时改这里）
# ---------------------------------------------------------------
VOCAB_SIZE = 10_000
CONTEXT_LENGTH = 256
D_MODEL = 512
D_FF = 1344
NUM_LAYERS = 4
NUM_HEADS = 16
ROPE_THETA = 10_000.0
TOTAL_TOKENS = 327_680_000          # 批量×步数×上下文长度应约等于此值

BATCH_SIZE = 32                     # 按显存调整，B * T * steps ≈ TOTAL_TOKENS
MAX_ITERS = TOTAL_TOKENS // (BATCH_SIZE * CONTEXT_LENGTH)   # ≈ 40000
WARMUP_ITERS = 100                  # warmup 步数，自行试错
MAX_LR = 3e-4                       # 自行试错
MIN_LR = 3e-5                       # 通常取 max_lr 的 1/10
WEIGHT_DECAY = 0.1
BETA1, BETA2, EPS = 0.9, 0.95, 1e-8 # Kingma & Ba 论文典型取值
GRAD_CLIP_NORM = 1.0

LOG_INTERVAL = 100                  # 每多少步打印一次
CHECKPOINT_DIR = Path("checkpoints")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NO_DECAY_PARAM_NAMES = ("norm", "rms_norm", "embedding")

def get_batch(dataset: np.ndarray, batch_size: int, context_length: int, device: str):
    """采样一个 batch 输入 x 和右移一位的标签 y shape 均为 (B, T)"""
    return Mf.data_loading(dataset, batch_size, context_length, device)


def get_lr(it: int) -> float:
    """线性 warmup + cosine 退火"""
    return Mf.learning_rate_schedule(it, MAX_LR, MIN_LR, WARMUP_ITERS, MAX_ITERS)

@torch.no_grad()
def evaluate(model, dataset, batch_size, context_length, device, eval_iters=20):
    """验证集上估算平均 loss"""
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        x, y = get_batch(dataset, batch_size, context_length, device)
        logits = model(x)
        losses[k] = Mf.cross_entropy_loss(
            logits.view(-1, logits.size(-1)), y.reshape(-1)
        )
    model.train()
    return losses.mean().item()


def main():
    torch.manual_seed(1337)
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    # ---------------------------------------------------------------
    # 1. 准备数据：加载训练好的 BPE 分词器，对语料分词并保存为 .npy
    #    （只需做一次，之后直接 np.load）
    # ---------------------------------------------------------------
    train_tokens_path = Path("data/TinyStoriesV2-GPT4-train.npy")
    if train_tokens_path.exists():
        train_ids = np.load(train_tokens_path)
    else:
        tokenizer = bpe_tokenizer(input_file_path="data/TinyStoriesV2-GPT4-train.txt", special_token_list=["<|endoftext|>"])
        tokenizer.run_pre_tokenization()
        _, vocab, merges = tokenizer.train_bpe(VOCAB_SIZE)
        with open("data/TinyStoriesV2-GPT4-train.txt", "r", encoding="utf-8") as f:
            text = f.read()
        train_ids = np.array(tokenizer.encode(text), dtype=np.int64)
        np.save(train_tokens_path, train_ids)


    # 2.build model
    model = transformer_lm(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        d_k=D_MODEL // NUM_HEADS,
        theta=ROPE_THETA,
        num_layers=NUM_LAYERS,
    ).to(DEVICE)

    # ---------------------------------------------------------------
    #   3. 构建优化器，区分有无 weight decay 的参数
    decay_params = [p for n, p in model.named_parameters()
                    if p.requires_grad and not any(k in n for k in NO_DECAY_PARAM_NAMES)]
    no_decay_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and any(k in n for k in NO_DECAY_PARAM_NAMES)]
    optimizer = myAdamW(
        [
            {"params": decay_params, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=MAX_LR, betas=(BETA1, BETA2), eps=EPS,
    )

    # ---------------------------------------------------------------
    # 5. 训练主循环
    # ---------------------------------------------------------------
    model.train()
    t0 = time.time()
    running_loss = 0.0
    print(torch.cuda.is_available())
    
    #for it in range(10):
    for it in range(0, MAX_ITERS):
        # -- 5.1 更新学习率（每步调用，warmup + cosine） --
        lr = get_lr(it)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # -- 5.2 取 batch，前向计算 loss --
        x, y = get_batch(train_ids, BATCH_SIZE, CONTEXT_LENGTH, DEVICE)
        logits = model(x)                                  # (B, T, V)
        loss = Mf.cross_entropy_loss(
            logits.view(-1, logits.size(-1)), y.reshape(-1)  # 展平成 (B*T, V) 和 (B*T,)
        )

        # -- 5.3 反向传播 + 梯度裁剪 + 优化器步进 --
        loss.backward()
        Mf.gradient_clipping(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # -- 5.4 日志 --
        #change !!!!
        running_loss += loss.item()
        if (it + 1) % 100 == 0:
            avg = running_loss / 100
            elapsed = time.time() - t0
            tokens_seen = (it + 1) * BATCH_SIZE * CONTEXT_LENGTH
            
            print(f"iter {it+1}/{MAX_ITERS} | lr {lr:.2e} | "
                  f"train loss {avg:.4f} | ppl {math.exp(avg):.2f} | "
                  f"{tokens_seen / elapsed:,.0f} tok/s")
            running_loss = 0.0

        # -- 5.5 定期保存检查点 --
        #if (it + 1) % 2000 == 0 or (it + 1) == MAX_ITERS:
        #    Mf.save_checkpoint(model, optimizer, it + 1, ckpt_path)

    # ---------------------------------------------------------------
    # 6. 训练结束：最终验证 loss + 保存最终模型
    # ---------------------------------------------------------------
    final_loss = evaluate(model, train_ids, BATCH_SIZE, CONTEXT_LENGTH, DEVICE)
    print(f"final eval loss: {final_loss:.4f} | ppl {math.exp(final_loss):.2f}")
    Mf.save_checkpoint(model, optimizer, MAX_ITERS, CHECKPOINT_DIR / "final.pt")

if __name__ == "__main__":
    main()


