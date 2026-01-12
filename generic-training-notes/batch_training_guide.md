# Training Modes and Batching Cheat Sheet

## Core Concepts

### Dataset Terminology
- **Dataset size (N)**: Total number of training samples
- **Batch size (B)**: Number of samples processed together before updating weights
- **Iteration/Step**: One forward pass + backward pass on a batch
- **Epoch**: One complete pass through the **entire** training dataset

---

## Training Modes

### 1. Batch Gradient Descent (Full Batch)

**Definition:** Use the **entire dataset** in a single batch to compute gradients.

```
Batch size = Dataset size (B = N)
Iterations per epoch = 1
```

**Algorithm:**
```python
for epoch in range(num_epochs):
    # Use ALL samples at once
    gradients = compute_gradient(entire_dataset)  
    weights = weights - learning_rate * gradients
    # ↑ One update per epoch
```

**Example:**
```
Dataset: 10,000 samples
Batch size: 10,000
Iterations per epoch: 10,000 / 10,000 = 1

Epoch 1: Process all 10,000 → update once
Epoch 2: Process all 10,000 → update once
...
```

**Characteristics:**
- ✅ Stable gradient estimates (uses all data)
- ✅ Smooth convergence
- ✅ Guaranteed to converge for convex problems
- ❌ Very slow per iteration (processes entire dataset)
- ❌ Requires massive memory (all samples in GPU at once)
- ❌ Can get stuck in local minima
- ❌ No regularization effect from noise

**When to use:**
- Small datasets (< 10,000 samples)
- When you have abundant memory
- Convex optimization problems

---

### 2. Stochastic Gradient Descent (SGD / Online Learning)

**Definition:** Use **one sample at a time** to compute gradients.

```
Batch size = 1 (B = 1)
Iterations per epoch = Dataset size (N)
```

**Algorithm:**
```python
for epoch in range(num_epochs):
    shuffle(dataset)
    for sample in dataset:
        # Use ONE sample
        gradient = compute_gradient(sample)
        weights = weights - learning_rate * gradient
        # ↑ Update after EVERY sample
```

**Example:**
```
Dataset: 10,000 samples
Batch size: 1
Iterations per epoch: 10,000 / 1 = 10,000

Epoch 1: 
  Sample 1 → update
  Sample 2 → update
  ...
  Sample 10,000 → update
  (10,000 updates total)
```

**Characteristics:**
- ✅ Very fast iterations (processes one sample)
- ✅ Low memory requirements
- ✅ Can escape local minima (due to noise)
- ✅ Online learning capability (can learn from streaming data)
- ❌ Very noisy gradient estimates
- ❌ Erratic convergence path
- ❌ May never truly converge (oscillates around minimum)
- ❌ Inefficient on modern hardware (GPUs underutilized)

**When to use:**
- Extremely large datasets that don't fit in memory
- Online learning scenarios
- When you need to escape sharp local minima
- Rarely used in modern deep learning (mini-batch is preferred)

---

### 3. Mini-Batch Gradient Descent ⭐ (Most Common)

**Definition:** Use **small batches** of samples to compute gradients.

```
Batch size = B (typically 32, 64, 128, 256)
Iterations per epoch = ⌈N / B⌉
```

**Algorithm:**
```python
for epoch in range(num_epochs):
    shuffle(dataset)
    for batch in get_batches(dataset, batch_size):
        # Use a BATCH of samples
        gradients = compute_gradient(batch)
        weights = weights - learning_rate * gradients
        # ↑ Update after each batch
```

**Example:**
```
Dataset: 10,000 samples
Batch size: 32
Iterations per epoch: 10,000 / 32 = 312.5 ≈ 313

Epoch 1:
  Batch 1 (samples 1-32) → update
  Batch 2 (samples 33-64) → update
  ...
  Batch 313 (samples 9985-10000) → update
  (313 updates total)
```

**Characteristics:**
- ✅ **Best balance** between stability and speed
- ✅ Efficient GPU utilization (parallel processing)
- ✅ More stable than SGD, faster than batch GD
- ✅ Provides regularization through noise
- ✅ Can escape shallow local minima
- ✅ Predictable memory usage
- ⚠️ Requires tuning batch size

**When to use:**
- **Default choice for modern deep learning**
- Large datasets
- When using GPUs/TPUs
- Almost all scenarios (CNNs, Transformers, etc.)

---

## Detailed Example: One Epoch Breakdown

**Setup:**
```
Dataset: 1,000 samples
```

### Batch GD (Batch = 1000)
```
Epoch 1:
├─ Iteration 1: samples [1...1000] → update weights
└─ Total: 1 iteration, 1 update

Memory: Load all 1,000 samples at once
Time per epoch: Slow (large computation)
Updates per epoch: 1
```

### SGD (Batch = 1)
```
Epoch 1:
├─ Iteration 1: sample [1] → update weights
├─ Iteration 2: sample [2] → update weights
├─ Iteration 3: sample [3] → update weights
├─ ...
├─ Iteration 1000: sample [1000] → update weights
└─ Total: 1,000 iterations, 1,000 updates

Memory: Load 1 sample at a time
Time per epoch: Fast per iteration, many iterations
Updates per epoch: 1,000
```

### Mini-Batch GD (Batch = 100)
```
Epoch 1:
├─ Iteration 1: samples [1...100] → update weights
├─ Iteration 2: samples [101...200] → update weights
├─ Iteration 3: samples [201...300] → update weights
├─ ...
├─ Iteration 10: samples [901...1000] → update weights
└─ Total: 10 iterations, 10 updates

Memory: Load 100 samples at a time
Time per epoch: Balanced
Updates per epoch: 10
```

---

## Key Formula

```
Iterations per Epoch = ⌈Dataset Size / Batch Size⌉
                     = ⌈N / B⌉
```

**Examples:**
```
N = 10,000, B = 32  → Iterations = ⌈10,000/32⌉ = 313
N = 10,000, B = 1   → Iterations = ⌈10,000/1⌉ = 10,000
N = 10,000, B = 100 → Iterations = ⌈10,000/100⌉ = 100
N = 10,000, B = 10,000 → Iterations = ⌈10,000/10,000⌉ = 1
```

---

## Relationship Between Epochs, Iterations, and Batches

```
Total Updates = Epochs × Iterations per Epoch
              = Epochs × ⌈N / B⌉

Total Samples Seen = Epochs × Dataset Size
                   = Epochs × N
```

**Example:**
```
Dataset: 50,000 samples
Batch size: 256
Epochs: 10

Iterations per epoch = 50,000 / 256 ≈ 196
Total iterations = 10 × 196 = 1,960
Total updates = 1,960
Total samples processed = 10 × 50,000 = 500,000
  (but many are repeats across epochs)
```

---

## Visual Comparison

```
Dataset: [■■■■■■■■■■] (10 samples)

Batch GD (B=10):
Epoch 1: [■■■■■■■■■■] → update
         └─── 1 iteration ───┘

SGD (B=1):
Epoch 1: [■]→update [■]→update [■]→update ... [■]→update
         └─────────── 10 iterations ──────────┘

Mini-Batch (B=2):
Epoch 1: [■■]→update [■■]→update [■■]→update [■■]→update [■■]→update
         └────────────── 5 iterations ─────────────┘
```

---

## Choosing Batch Size

### Small Batches (1-32)
- ✅ Better generalization (more noise)
- ✅ Can escape sharp minima
- ✅ Lower memory usage
- ❌ Slower training (more iterations)
- ❌ Noisy gradients
- ❌ Underutilizes GPU

### Medium Batches (32-256) ⭐ Most Common
- ✅ Good balance
- ✅ Efficient GPU usage
- ✅ Stable training
- ✅ Reasonable generalization

### Large Batches (256-2048+)
- ✅ Faster training (fewer iterations)
- ✅ Very stable gradients
- ✅ Full GPU utilization
- ❌ May generalize worse
- ❌ Can get stuck in sharp minima
- ❌ Requires more memory
- ⚠️ Often needs learning rate scaling

### Rule of Thumb
- **Start with 32 or 64** for most problems
- **Increase** if you have memory and want speed
- **Decrease** if running out of memory
- **Powers of 2** (32, 64, 128, 256) for hardware efficiency

---

## Common Batch Sizes by Domain

| Task | Typical Batch Size | Reason |
|------|-------------------|--------|
| **Image Classification** | 32-256 | Balance speed/memory |
| **Object Detection** | 8-32 | Large images, high memory |
| **Language Models (BERT)** | 16-32 | Long sequences, memory intensive |
| **GPT Training** | 512-2048+ | Massive scale, distributed training |
| **GANs** | 64-128 | Stability in adversarial training |
| **Reinforcement Learning** | 32-256 | Depends on environment |

---

## Other Training Strategies

### 1. Data Shuffling
```python
for epoch in range(num_epochs):
    shuffle(dataset)  # ← Important!
    for batch in dataset:
        train(batch)
```
- **Why?** Prevents model from learning order patterns
- **When?** Almost always (except for time-series)

### 2. Learning Rate Scheduling
Adjust learning rate based on epoch/iteration:
- **Step decay**: Reduce LR every N epochs
- **Cosine annealing**: Smooth reduction
- **Warmup**: Gradually increase LR at start

### 3. Gradient Accumulation
Simulate larger batches with limited memory:
```python
for i, batch in enumerate(dataset):
    loss = forward(batch)
    loss.backward()  # Accumulate gradients
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()  # Update after N batches
        optimizer.zero_grad()
```
**Effective batch size = batch_size × accumulation_steps**

### 4. Distributed Training
Split batches across multiple GPUs:
- **Data Parallel**: Each GPU processes different batch
- **Model Parallel**: Split model across GPUs
- **Effective batch size = batch_size × num_gpus**

---

## Quick Reference

```
Epoch = One complete pass through dataset

Iteration = One weight update
          = Processing one batch

Batch Size = Samples per iteration

Key Equation:
  Iterations per Epoch = Dataset Size / Batch Size
```

**Memory vs Speed Trade-off:**
```
Batch Size ↑ → Memory ↑, Speed ↑, Noise ↓, Generalization ↓
Batch Size ↓ → Memory ↓, Speed ↓, Noise ↑, Generalization ↑
```

**Modern Best Practice:**
- Use **mini-batch GD** (batch size 32-256)
- **Shuffle** data each epoch
- Use **learning rate warmup** for large batches
- Consider **gradient accumulation** if memory-limited