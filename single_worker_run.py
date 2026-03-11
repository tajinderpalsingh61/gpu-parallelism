"""
Single-Process Training (No Parallelism) — Baseline for Comparison
===================================================================
Same model architecture as tp_distributed.py but runs on a single process.
Use this to verify that the distributed TP version produces matching results.

Run:  python tp_single.py
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Model config (must match tp_distributed.py) ---
batch_size = 2
embed_dim = 4
attn_heads = 2
ffn_dim = 16
vocab_size = 8
seq_len = 3
L = 2
NUM_TRAIN_STEPS = 10
LR = 1e-2


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, attn_heads, bias=False, batch_first=True)
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x):
        residual = x
        x = self.attn_norm(x)
        x, _ = self.attention(x, x, x)
        x = x + residual
        residual = x
        x = self.ffn_norm(x)
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.linear2(x)
        return x + residual


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformers = nn.ModuleList([Transformer() for _ in range(L)])
        self.output = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, token_ids):
        x = self.embedding(token_ids)
        for t in self.transformers:
            x = t(x)
        return self.output(x)


if __name__ == "__main__":
    # Same seed as tp_distributed.py for identical initial weights
    torch.manual_seed(42)
    model = Model()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"\n{'=' * 60}")
    print(f"SINGLE-PROCESS TRAINING (no parallelism)")
    print(f"{'=' * 60}")
    print(f"\n  Config: {NUM_TRAIN_STEPS} steps, lr={LR}, batch={batch_size}, seq_len={seq_len}")
    print(f"  Trainable params: {sum(p.numel() for p in model.parameters())}")
    print(f"\n  {'Step':<6} {'Loss':<12} {'Grad Norm':<12}")
    print(f"  {'-'*30}")

    # Same seed as tp_distributed.py training loop
    torch.manual_seed(123)

    start_time = time.time()

    for step in range(NUM_TRAIN_STEPS):
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))

        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))

        optimizer.zero_grad()
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
        optimizer.step()

        print(f"  {step:<6} {loss.item():<12.4f} {grad_norm.item():<12.4f}")

    total_time = time.time() - start_time
    print(f"\n  Total training time: {total_time:.4f}s")

    # Verification output (same test input as tp_distributed.py)
    print(f"\n{'=' * 60}")
    print("VERIFICATION")
    print(f"{'=' * 60}")

    with torch.no_grad():
        torch.manual_seed(999)
        test_ids = torch.randint(0, vocab_size, (1, seq_len))
        output = model(test_ids)
        print(f"  Test input:  {test_ids.tolist()}")
        print(f"  Output logits (first token): {output[0, 0].tolist()}")
        print(f"  Predicted tokens: {output.argmax(dim=-1).tolist()}")
