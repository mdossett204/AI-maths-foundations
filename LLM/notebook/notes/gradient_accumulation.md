# Gradient Accumulation

**Gradient Accumulation** is a technique used in deep learning to effectively train with a larger batch size than what your GPU memory can physically hold.

## The Problem: Limited GPU Memory

Training Large Language Models (LLMs) requires storing:
1.  **Model Weights**
2.  **Optimizer States**
3.  **Activations** (intermediate outputs of each layer for the backward pass)

A large **Batch Size** is often desirable for stable convergence, but increasing the batch size linearly increases the memory required for activations. If you set the batch size too high, you run out of memory (OOM).

## The Solution: Accumulate Gradients

Instead of updating the model weights after every single small batch (which might be noisy), we **accumulate** the calculated gradients over several smaller batches and only perform the weight update once we've processed enough data to meet our "effective" target batch size.

### How it Works (Step-by-Step)

Imagine you want a batch size of **32**, but your GPU can only fit a batch size of **4**. You can set your **Gradient Accumulation Steps** to **8** (`32 / 4 = 8`).

1.  **Step 1:** Load a mini-batch of 4 samples.
2.  **Forward Pass:** Compute the loss.
3.  **Backward Pass:** Compute gradients. **Do NOT update weights yet.** Add these gradients to a holding buffer.
4.  **Repeat:** Do this 7 more times (Steps 2-8), adding the new gradients to the existing ones in the buffer.
5.  **Optimizer Step:** Now that we have processed 32 samples (4 * 8), use the *summed* (or averaged) gradients in the buffer to update the model weights.
6.  **Zero Gradients:** Clear the buffer and start over.

### Mathematical Equivalence

*   **Physical Batch Size:** The number of samples processed in one forward/backward pass (e.g., 4).
*   **Accumulation Steps:** The number of passes before a weight update (e.g., 8).
*   **Effective Batch Size:** `Physical Batch Size * Accumulation Steps` (e.g., 32).

Mathematically, gradient accumulation is almost identical to training with the larger batch size, assuming the batch normalization statistics (if used) are handled correctly.

### Pros and Cons

**Pros:**
*   Allows training large models on consumer hardware (e.g., training a 7B model on a 16GB GPU).
*   Enables large effective batch sizes which can stabilize training.

**Cons:**
*   **Slower Training Speed:** You are performing the same number of compute operations, but the optimizer step happens less frequently. (Actually, it can be slightly faster per epoch because you have fewer costly optimizer updates, but the total time to convergence is usually comparable).
*   **Complexity:** Requires slight modification to the training loop (though libraries like Hugging Face `Trainer` handle this automatically via the `gradient_accumulation_steps` argument).
