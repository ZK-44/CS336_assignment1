import torch
from cs336_basics import Myfunctions as Mf
from cs336_basics.Transformer_block import transformer_lm
from pathlib import Path
from cs336_basics.MyadamW import myAdamW
from cs336_basics.bpe_tokenization import bpe_tokenizer
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


tokenizer = bpe_tokenizer(input_file_path="data/TinyStoriesV2-GPT4-train.txt", special_token_list=["<|endoftext|>"])
tokenizer.run_pre_tokenization()
_, vocab, merges = tokenizer.train_bpe(VOCAB_SIZE)

#build model
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
Mf.load_checkpoint(model, optimizer, "checkpoints/final.pt")
 
model.eval()
