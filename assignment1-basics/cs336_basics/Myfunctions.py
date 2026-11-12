import torch
import einops
import math
import numpy as np
def safe_softmax(x, dim=-1):
    max_val, _ = torch.max(x, dim=dim, keepdim=True)
    exp_x = torch.exp(x - max_val)
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp

def temperature_scaled_softmax(x, temperature=1.0, dim=-1):
    scaled_x = x / temperature
    return safe_softmax(scaled_x, dim=dim)


def scaled_dot_product_attention(Q,K,V,mask=None):
    d = math.sqrt(K.shape[-1])
    scores = torch.matmul(Q, K.transpose(-2, -1)) / d
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn_weights = safe_softmax(scores, dim=-1)
    context = torch.matmul(attn_weights, V)
    return context

def cross_entropy_loss(logits, targets):
    loss =torch.nn.functional.cross_entropy(logits, targets)
    return loss

def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

def learning_rate_schedule(t,alpha_max,alpha_min,T_w,T_c):
    if t < T_w:
        return alpha_max * t / T_w
    elif t < T_c:
        return alpha_min + (alpha_max - alpha_min) * (1 + math.cos(math.pi * (t - T_w) / (T_c - T_w))) / 2
    else:
        return alpha_min

def gradient_clipping(parameters, max_norm):
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    clip_coef = max_norm / (total_norm + 1e-6)
    if total_norm > max_norm:
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)

        
def data_loading(x:np.ndarray, batch_size, context_length,device):
    # 1. 随机选择 batch_size 个起始索引
    # 为了保证截取 context_length 长度时不越界，最大起始索引为 len(x) - context_length
    max_start_index = len(x) - context_length
    start_indices = np.random.randint(0, max_start_index, size=batch_size)
    
    # 2. 构建输入序列 (inputs) 和目标序列 (targets)
    # 目标序列是输入序列整体向右平移一位（Next-token prediction）
    inputs = np.stack([x[i : i + context_length] for i in start_indices])
    targets = np.stack([x[i + 1 : i + 1 + context_length] for i in start_indices])
    
    # 3. 将 NumPy 数组转换为 PyTorch 张量，并移动到指定设备
    inputs_tensor = torch.from_numpy(inputs).to(device)
    targets_tensor = torch.from_numpy(targets).to(device)
    
    return inputs_tensor, targets_tensor

    
def save_checkpoint(model, optimizer, iteration, out ):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
    }
    torch.save(checkpoint, out)

def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iteration = checkpoint['iteration']
    return iteration

def nuclues_sampling(logits: torch.Tensor, p: float) -> torch.Tensor:
    #p is the cumulative probability threshold for nucleus sampling
    sorted_probs, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    mask = (cumulative_probs-sorted_probs) >= p
    sorted_probs = sorted_probs.masked_fill(mask, 0.0)
    filtered = torch.zeros_like(sorted_probs).scatter_(-1, sorted_indices, sorted_probs)
    return filtered/filtered.sum(dim=-1, keepdim=True)


def SampleNextToken(x:torch.tensor,temperature,p,generator=None): 
    #x.shape = ( context_length,vocab_size)
    # 1. 对每个时间步的 logits 进行 softmax，得到概率分布
    # 可以根据需要调整温度参数
    last_ = x[:, -1, :] if x.dim() == 3 else x[-1, :]
    probabilities = temperature_scaled_softmax(last_, temperature=temperature, dim=-1)
    probabilities = nuclues_sampling(probabilities, p)
    return torch.multinomial(probabilities, num_samples=1, generator=generator)

@torch.no_grad()
def decode(
    model,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 1.0,
    end_of_text_id: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    自回归解码：为用户的 prompt 生成续写，直到遇到 <|endoftext|> 或达到最大长度。

    参数:
        model:          语言模型，输入 (batch, seq_len) 的 token id
                        输出 (batch, seq_len, vocab_size) 的 logits
        prompt_ids:     输入 prompt 的 token id 形状 (batch, seq_len)
        max_new_tokens: 最多生成的词元数
        temperature:    温度 τ，采样前对下一词元分布做 softmax 温度缩放
        top_p:          top-p 核采样阈值
        end_of_text_id: <|endoftext|> 的 token id为 None 时不在中途停止
        generator:      可选的随机数生成器

    返回:
        拼接 prompt 与生成结果的完整序列，形状 (batch, seq_len + 已生成词元数)
    """
    was_training = model.training
    model.eval()

    ids = prompt_ids
    for _ in range(max_new_tokens):
        logits = model(ids)                      # (batch, seq_len, vocab_size)
        next_id = SampleNextToken(             # (batch,)
            logits, temperature=temperature, p=top_p, generator=generator
        )
        ids = torch.cat([ids, next_id], dim=-1)
        # 遇到 <|endoftext|> 且所有序列都结束则停止
        if end_of_text_id is not None and (next_id == end_of_text_id).all():
            break

    if was_training:
        model.train()
    return ids
    
    
