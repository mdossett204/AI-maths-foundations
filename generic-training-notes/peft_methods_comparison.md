# Comparison of PEFT Methods

**Source:** "Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning" (Lialin et al.)

This document summarizes the extensive experimental comparison of 14 Parameter-Efficient Fine-Tuning (PEFT) methods across various dimensions.

## 1. Evaluation Framework
The comparison benchmarks methods on five key dimensions:
1.  **Storage Efficiency:** Disk space required to store the fine-tuned parameters.
2.  **Memory Efficiency:** Peak GPU RAM usage during training.
3.  **Computational Efficiency:** Training throughput (tokens processed per second).
4.  **Inference Overhead:** Inference throughput (tokens processed per second).
5.  **Downstream Performance:** Accuracy (for NLU tasks) or ROUGE-L scores (for Summarization).

## 2. Experimental Setup
*   **Models:** Comparison performed on **T5** models of varying scales:
    *   **T5-Large (0.7B)**
    *   **T5-3B**
    *   **T5-11B** (Crucial for testing realistic memory constraints where full fine-tuning often fails).
*   **Datasets:**
    *   **NLU:** SuperGLUE (BoolQ, RTE, COPA).
    *   **NLG:** CNN/Dailymail (Abstractive Summarization).

## 3. Key Findings

### Downstream Performance
*   **Top Tier:** **Houlsby Adapters** and **LoRA** were the only methods to consistently achieve or exceed full fine-tuning performance without extensive hyperparameter tuning.
*   **The Surprise:** **Layer Norm (LN) Tuning** proved unexpectedly competitive, performing very close to full fine-tuning on T5-Large and T5-11B despite its simplicity.
*   **Weaknesses:**
    *   **Prompt Tuning** performed poorly in compute-limited settings and showed high sensitivity to random seeds.
    *   **Hybrid Methods** (e.g., UniPELT, MAM) were difficult to tune and often underperformed simpler methods in this specific setup.
    *   **Pfeiffer Adapters** lagged significantly behind Houlsby Adapters (up to 15 points lower), contradicting some prior studies.

### Efficiency & Speed
*   **Memory vs. Parameters:**
    *   **Diminishing Returns:** "Sparse" methods (like Compacter/KronA) use drastically fewer parameters than LoRA but yield **negligible additional RAM savings** (e.g., both use ~28GB for T5-11B).
    *   *Reason:* Memory usage is dominated by activations and the frozen base model weights, not just the optimizer states of the trainable parameters.
*   **Training Speed:**
    *   On smaller models (T5-Large), some PEFT methods (like Adapters) were **20-40% slower** than full fine-tuning due to the computational overhead of the added modules.
*   **Inference Speed:**
    *   **Additive Methods (Adapters, IA³):** Introduce permanent latency overhead.
    *   **Reparameterization (LoRA, KronA):** Can be merged into the base model weights after training, resulting in **zero inference overhead**.

## 4. Practical Recommendations
1.  **Go-to Methods:** Use **LoRA** or **Houlsby Adapters** for the most reliable performance/efficiency balance.
2.  **Strong Baseline:** Always check **Layer Norm Tuning** as a baseline; it is extremely simple (tuning only `gamma`/`beta`) and surprisingly effective.
3.  **Metric Awareness:** Do not rely solely on "parameter count" as a proxy for efficiency. Measure actual **Peak RAM** and **Throughput**, as these often tell a different story.
