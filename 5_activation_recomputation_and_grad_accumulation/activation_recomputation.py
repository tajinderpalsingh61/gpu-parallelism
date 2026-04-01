# datasets

"""
Activation Recomputation (Gradient Checkpointing)
==================================================

During a normal forward pass, every layer's intermediate activations are stored
in memory so that backward can compute gradients. For a model with L layers,
this means O(L) activation tensors live simultaneously in GPU memory.

Activation recomputation trades compute for memory:
  - Forward pass: discard intermediate activations (don't store them)
  - Backward pass: re-run the forward for each layer to recompute activations
    just before they're needed for the gradient calculation

Memory savings:
  - Without checkpointing: all L layers' activations stored  → O(L) memory
  - With checkpointing:    only checkpoint boundaries stored → O(sqrt(L)) memory
    (PyTorch recomputes activations between checkpoints on the fly)

Cost:
  - ~33% more compute (one extra forward pass per checkpointed segment)

This is essential for training large models where activation memory dominates.
For a 7B parameter model with seq_len=2048, activations can use 10-60 GB
depending on batch size — often more than the model weights themselves.

PyTorch API:
  torch.utils.checkpoint.checkpoint(fn, *args) — wraps a function so its
  intermediate activations are discarded and recomputed during backward.

Run:  python 5_activation_recomputation_and_grad_accumulation/activation_recomputation.py
"""

import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from datasets import load_from_disk, load_dataset
from torch.utils.data import DataLoader, Dataset

# --- Model config ---
batch_size = 32
embed_dim = 768
attn_heads = 12
ffn_dim = 3072
vocab_size = 50257
seq_len = 512
num_layers = 12
NUM_TRAIN_STEPS = 50
LR = 3e-4
MAX_GRAD_NORM = 1.0
USE_REAL_DATA = True
DATA_CACHE_DIR = "./data/wikitext-103"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


DEVICE = get_device()


# --- Data loading ---
def load_wikitext():
    if os.path.exists(DATA_CACHE_DIR):
        return load_from_disk(DATA_CACHE_DIR)
    else:
        print("  Downloading Wikitext-103...")
        dataset = load_dataset("wikitext", "wikitext-103-v1")
        dataset.save_to_disk(DATA_CACHE_DIR)
        return dataset


class TokenizedDataset(Dataset):
    def __init__(self, texts, vocab_size=50257, seq_len=512):
        self.texts = [t for t in texts if t.strip()]
        self.vocab_size = vocab_size
        self.seq_len = seq_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = [hash(word) % self.vocab_size for word in text.split()]
        if len(tokens) < self.seq_len:
            tokens = tokens + [0] * (self.seq_len - len(tokens))
        else:
            tokens = tokens[: self.seq_len]
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        targets = torch.tensor(tokens[1:], dtype=torch.long)
        return input_ids, targets


def get_data_batch(use_real=True):
    if not use_real:
        while True:
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len - 1))
            targets = torch.randint(0, vocab_size, (batch_size, seq_len - 1))
            yield input_ids, targets
    else:
        dataset = load_wikitext()
        if dataset is None:
            yield from get_data_batch(use_real=False)
            return
        texts = dataset["train"]["text"]
        wikitext_dataset = TokenizedDataset(texts, vocab_size, seq_len)
        loader = DataLoader(wikitext_dataset, batch_size=batch_size, shuffle=True)
        while True:
            for batch in loader:
                yield batch


# --- Transformer blocks ---
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim, attn_heads, bias=False, batch_first=True
        )
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
    """Standard model — stores all activations during forward."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformers = nn.ModuleList(
            [Transformer() for _ in range(num_layers)]
        )
        self.output = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, token_ids):
        x = self.embedding(token_ids)
        for layer in self.transformers:
            x = layer(x)
        return self.output(x)


class ModelWithActivationRecomputation(nn.Module):
    """
    Same architecture, but wraps each transformer layer in
    torch.utils.checkpoint.checkpoint so activations are discarded
    during forward and recomputed during backward.

    use_reentrant=False is the recommended setting (PyTorch >= 2.0).
    It supports all autograd features and gives clearer error messages.
    """

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformers = nn.ModuleList(
            [Transformer() for _ in range(num_layers)]
        )
        self.output = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, token_ids):
        x = self.embedding(token_ids)
        for layer in self.transformers:
            # checkpoint: don't save activations for this layer,
            # recompute them during the backward pass instead
            x = checkpoint(layer, x, use_reentrant=False)
        return self.output(x)


# --- Memory measurement ---
def get_memory_mb():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    # MPS and CPU don't have reliable memory tracking

    return 0.0


def get_peak_memory_mb():
    """Get peak GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**2
    return 0.0


def reset_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# --- Training ---
def train(model_class, name: str):
    torch.manual_seed(42)
    model = model_class().to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    data_gen = get_data_batch(use_real=USE_REAL_DATA)

    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(f"{'=' * 70}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Config: {NUM_TRAIN_STEPS} steps, batch={batch_size}, seq_len={seq_len}")
    print(f"\n  {'Step':<6} {'Loss':<12} {'Grad Norm':<12}")
    print(f"  {'-' * 30}")

    reset_peak_memory()
    start_time = time.time()

    for step in range(NUM_TRAIN_STEPS):
        input_ids, targets = next(data_gen)
        input_ids = input_ids.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), MAX_GRAD_NORM
        )
        optimizer.step()

        if step % 10 == 0:
            print(f"  {step:<6} {loss.item():<12.4f} {grad_norm.item():<12.4f}")

    total_time = time.time() - start_time
    peak_mem = get_peak_memory_mb()

    print(f"\n  Training time: {total_time:.2f}s")
    if peak_mem > 0:
        print(f"  Peak GPU memory: {peak_mem:.1f} MB")

    return {"name": name, "time": total_time, "peak_mem": peak_mem}


def main():
    print(f"\n{'#' * 70}")
    print("# ACTIVATION RECOMPUTATION (GRADIENT CHECKPOINTING)")
    print(f"{'#' * 70}")

    if torch.cuda.is_available():
        print(f"\n  Device: CUDA — {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        print(f"\n  Device: MPS (macOS)")
        print("  Note: memory tracking requires CUDA; memory stats will show 0 on MPS")
    else:
        print(f"\n  Device: CPU")

    results = []

    # 1. Baseline: all activations stored
    results.append(train(Model, "BASELINE (all activations stored)"))

    # 2. Activation recomputation: activations discarded and recomputed
    results.append(
        train(
            ModelWithActivationRecomputation,
            "ACTIVATION RECOMPUTATION (checkpoint per layer)",
        )
    )

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print("  COMPARISON")
    print(f"{'=' * 70}")

    baseline = results[0]
    for r in results:
        speedup = baseline["time"] / r["time"] if r["time"] > 0 else 0
        print(f"  {r['name']:<55}")
        print(f"    Time: {r['time']:.2f}s  ({speedup:.2f}x vs baseline)")
        if r["peak_mem"] > 0:
            saving = (1 - r["peak_mem"] / baseline["peak_mem"]) * 100 if baseline["peak_mem"] > 0 else 0
            print(f"    Peak memory: {r['peak_mem']:.1f} MB  ({saving:+.1f}% vs baseline)")
        print()

    print(f"\n{'#' * 70}")
    print("# DONE")
    print(f"{'#' * 70}\n")


if __name__ == "__main__":
    main()



# root@1f23dc01020d:/# python activation_recomputation.py 

# ######################################################################
# # ACTIVATION RECOMPUTATION (GRADIENT CHECKPOINTING)
# ######################################################################

#   Device: CUDA — NVIDIA GeForce RTX 5090

# ======================================================================
#   BASELINE (all activations stored)
# ======================================================================
#   Params: 162,212,352
#   Config: 50 steps, batch=32, seq_len=512

#   Step   Loss         Grad Norm   
#   ------------------------------
#   0      12.6002      80.1880     
#   10     1.6615       2.0349      
#   20     1.2963       0.7120      
#   30     1.3515       0.4727      
#   40     1.0113       0.3115      

#   Training time: 28.93s
#   Peak GPU memory: 28853.3 MB

# ======================================================================
#   ACTIVATION RECOMPUTATION (checkpoint per layer)
# ======================================================================
#   Params: 162,212,352
#   Config: 50 steps, batch=32, seq_len=512

#   Step   Loss         Grad Norm   
#   ------------------------------
#   0      12.6002      80.1880     
#   10     1.6615       2.0349      
#   20     1.2963       0.7120      
#   30     1.3515       0.4727      
#   40     1.0113       0.3115      

#   Training time: 31.02s
#   Peak GPU memory: 15046.0 MB

# ======================================================================
#   COMPARISON
# ======================================================================
#   BASELINE (all activations stored)                      
#     Time: 28.93s  (1.00x vs baseline)
#     Peak memory: 28853.3 MB  (+0.0% vs baseline)

#   ACTIVATION RECOMPUTATION (checkpoint per layer)        
#     Time: 31.02s  (0.93x vs baseline)
#     Peak memory: 15046.0 MB  (+47.9% vs baseline)


# ######################################################################
# # DONE
# ######################################################################
