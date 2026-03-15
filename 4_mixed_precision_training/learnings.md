# GPU Parallelism & Mixed Precision Training - Key Learnings

## 1. Mixed Precision Training Performance

### When Mixed Precision HELPS
- **NVIDIA GPUs with Tensor Cores** (V100, A100, H100, RTX 3090/4090/5090)
  - FP16 operations 2-8x faster than FP32
  - Memory usage reduced by ~50% (FP16 = 2 bytes vs FP32 = 4 bytes)
  - Enables larger batch sizes and longer sequences

### When Mixed Precision DOESN'T Help
- **Apple Silicon (MPS)** - No dedicated tensor hardware, same throughput for FP16/FP32
- **CPU** - Actually slower due to overhead of autocast + GradScaler
- **Small models** - Overhead dominates, negligible speedup

### Real Performance Metrics (Measured)

**162M parameter transformer, 100 steps, batch=32, seq_len=512**

| Hardware | FP32 Baseline | Mixed Precision (FP16) | Speedup |
|----------|--------------|----------------------|---------|
| Apple M4 Max 64 GB -series (MPS) | 486.09s | 383.61s | **1.27x** |
| NVIDIA RTX 5090 (CUDA) | 51.61s | 35.19s | **1.47x** |

- **RTX 5090 is ~9.4x faster than MPS** in absolute FP32 time
- **RTX 5090 is ~10.9x faster than MPS** in absolute FP16 time
- Mixed precision gives a bigger relative boost on CUDA (Tensor Cores) vs MPS
- Even MPS sees ~21% speedup — likely from reduced memory bandwidth with FP16

## 2. Performance Bottlenecks Encountered

### Problem 1: Profiler Overhead
```python
# BAD: Wrapping every step in profiler
with profile(activities=[...]) as prof:
    for step in range(NUM_TRAIN_STEPS):
        with record_function(f"step_{step}"):  # Creates massive overhead
            loss, grad_norm = trainer.train_step(...)
```
**Impact**: Added 10-30% overhead. Removed entirely for true benchmarking.

### Problem 2: Batch Size Too Small
- Started with `batch_size=4` → GPU idle, memory underutilized
- RTX 5090 with 24GB VRAM needs `batch_size=32-128` for good utilization
- Larger batches = better GPU saturation = clearer speedup comparison

### Problem 3: Learning Rate Instability
- Initial `LR=0.01` caused divergence (loss: 11 → 1044 in 10 steps)
- **Fix**: `LR=3e-4` (30x lower) for 162M param model
- **Lesson**: Transformer scale requires careful hyperparameter tuning
  - Small models: `LR=1e-2` to `1e-3`
  - 100M+ params: `LR=1e-4` to `3e-4`

### Problem 4: Gradient Overflow in FP16
```python
# BAD: No actual clipping
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))

# GOOD: Proper gradient clipping
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```
**Impact**: Gradients exploded to `inf` in mixed precision without clipping.

### Problem 5: GradScaler on CPU/MPS
- `GradScaler` only needed for CUDA with FP16
- On CPU/MPS, it's a no-op but adds overhead
- Warning: "GradScaler is enabled, but CUDA is not available"

## 3. Mixed Precision Implementation Details

### Correct Flow (PyTorch AMP)
```python
with autocast(device_type="cuda", dtype=torch.float16):
    logits = model(x)              # Forward in FP16
    loss = loss_fn(logits, target) # Loss computed in FP32 automatically

scaler.scale(loss).backward()       # Scale loss to prevent gradient underflow
scaler.unscale_(optimizer)          # Unscale before gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)              # Step with scaling
scaler.update()                     # Update scale for next step
```

### Key Points
- **Autocast handles casting inside context manager automatically**
- **Loss is computed in FP32** even inside autocast (by default)
- **GradScaler** prevents gradient underflow by scaling loss during backward
- **Unscale before clipping** to avoid clipping scaled gradients
- **Update scale dynamically** to prevent overflow/underflow

## 4. Real-World Data vs Random Data

### Wikitext-103 Integration
- Added automatic download and caching to `./data/wikitext-103`
- Simple hash-based tokenization (in production, use BPE: tiktoken/transformers)
- Realistic text distribution vs random token sequences
- **Better for detecting training divergence issues**

### When to Use Random vs Real Data
| Metric | Random | Real |
|--------|--------|------|
| **Speed** | Fast (no I/O) | Slower (data loading) |
| **Debugging** | Good for algorithm testing | Good for convergence testing |
| **Batch Size** | Can be arbitrarily large | Limited by dataset size |
| **Realistic** | No (uniform distribution) | Yes (Zipfian, long-tail) |

## 5. Device Detection Strategy

### Proper Device Priority
```python
# CUDA > MPS (macOS) > CPU
if torch.cuda.is_available():
    device = "cuda"      # Best performance
elif torch.backends.mps.is_available():
    device = "mps"       # Apple Silicon (M1/M2/M3)
else:
    device = "cpu"       # Fallback
```

### Device-Specific Considerations
- **CUDA**: Full mixed precision support, GradScaler works
- **MPS**: No tensor cores, FP16 no faster than FP32
- **CPU**: FP16 slower than FP32 due to overhead

## 6. Benchmarking Best Practices

### What We Did Wrong
- ✗ Profiler overhead masked real performance
- ✗ Too few training steps (10-100)
- ✗ Batch size too small for GPU
- ✗ Wrong learning rate (divergence)
- ✗ Random data (unrealistic)

### Best Practices
- ✓ Remove profiler for final benchmarks
- ✓ Run 1000+ steps for stable measurements
- ✓ Use batch sizes that saturate GPU memory (typically 32-256)
- ✓ Tune learning rate properly before comparing
- ✓ Use real data for realistic results
- ✓ Warm up GPU (run 100 steps, discard) before timing
- ✓ Measure throughput: `steps / total_time` (tokens/second is better)

## 7. Expected Speedups on Different Hardware

### NVIDIA GPUs (with Tensor Cores)
**Measured on RTX 5090:**
| Model | FP32 Time | FP16 Time | Speedup |
|-------|-----------|-----------|---------|
| 162M params (measured) | 51.61s | 35.19s | **1.47x** |
| Extrapolated: scales similarly | - | - | **~1.45-1.50x** |

### Apple Silicon (MPS)
**Measured on M-series:**
| Model | FP32 Time | FP16 Time | Speedup |
|-------|-----------|-----------|---------|
| 162M params (measured) | 486.09s | 383.61s | **1.27x** |

- **No dedicated Tensor Cores**, but still 27% speedup
- Benefit likely from reduced memory bandwidth (FP16 = 2 bytes vs FP32 = 4 bytes)
- More modest than NVIDIA but still meaningful
- **RTX 5090 is ~9.4x faster than M-series in FP32**

### CPU
- FP16 slower than FP32 (overhead without hardware support)
- Speedup: **0.8-0.95x** (worse)

## 8. Key Takeaways

1. **Mixed precision helps on all GPUs, but NVIDIA Tensor Cores get the biggest boost**
   - NVIDIA RTX 5090: 1.47x speedup
   - Apple M-series MPS: 1.27x speedup
   - CPU: Don't use (slower)

2. **Remove profiler overhead before final benchmarking**
3. **Proper hyperparameter tuning (LR, grad clipping) is essential**
4. **Batch size matters more than precision for throughput**
5. **Use real data to catch convergence issues**
6. **Memory savings from FP16 are universal (50%), speedup is hardware-dependent**
7. **GradScaler prevents FP16 underflow, only needed for CUDA**
8. **100 training steps was enough to see stable speedup measurements**

## 9. Code Configuration Reference

```python
# Optimal settings for 162M param transformer on RTX 5090
batch_size = 32              # Saturate GPU memory
embed_dim = 768
attn_heads = 12
ffn_dim = 3072
seq_len = 512
num_layers = 12
NUM_TRAIN_STEPS = 1000       # At least 1000 for stable benchmarks
LR = 3e-4                    # Proper learning rate
MAX_GRAD_NORM = 1.0          # Gradient clipping
USE_REAL_DATA = True         # Wikitext-103
```

## 10. Further Optimization Opportunities

1. **Gradient Accumulation** - Process larger effective batch sizes with memory constraints
2. **DeepSpeed/FSDP** - Distributed training across multiple GPUs
3. **Flash Attention** - Faster attention computation (xformers, triton)
4. **Custom CUDA Kernels** - Fused operations (LayerNorm + dropout)
5. **Quantization** - INT8/INT4 post-training or during training
6. **Knowledge Distillation** - Train smaller models from larger ones
