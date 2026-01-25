# Hybrid PEFT Methods

Hybrid parameter-efficient fine-tuning (PEFT) approaches combine techniques from the three main categories—**Additive**, **Selective**, and **Reparametrization-based**—to leverage their respective strengths while mitigating their weaknesses. By mixing these strategies, hybrid methods often aim to optimize specific trade-offs between trainable parameter counts, memory efficiency, training speed, and downstream performance.

## 1. SparseAdapter

**Concept:**
SparseAdapter (He et al., 2022b) modifies the standard Adapter approach by using a **Large-Sparse** strategy. Instead of a small bottleneck dimension, it uses a large hidden dimension but prunes approximately 40% of the values at initialization.

**Key Features:**
-   **Large-Sparse Strategy:** Uses a larger capacity initially but enforces sparsity.
-   **Performance:** Consistently outperforms non-sparse adapters with the same number of trainable parameters.
-   **Trade-off:** Training and inference costs can be higher without specific hardware support for sparse operations.

## 2. MAM Adapter (Mix-And-Match)

**Concept:**
Proposed by He et al. (2022a), the MAM Adapter is a result of a systematic investigation into adapter placement and soft prompts. It combines **Scaled Parallel Adapters** with **Soft Prompts**.

**Architecture:**
1.  **Parallel Adapter:** A scaled adapter placed in parallel to the Feed-Forward Network (FFN) layer.
2.  **Soft Prompt:** Learnable continuous tokens prepended to the input, which efficiently modify attention mechanisms.

**Why it works:**
-   Parallel adapters generally outperform sequential ones.
-   Adapters parallel to the FFN layer outperform those parallel to Multi-Head Attention (MHA).
-   Soft prompts are highly efficient at modifying attention behavior (changing only ~0.1% of parameters).

**Pseudocode:**
```python
def transformer_block_mam(x):
    # Soft Prompt
    x = concat([x, soft_prompt], dim=seq)
    
    residual = x
    x = SelfAttention(x)
    x = LN(x + residual)
    
    # Parallel Adapter for FFN
    x_a = FFN(x) 
    x_a = scale * x_a
    
    x = LN(x + x_adapter)
    return x
```

## 3. UniPELT

**Concept:**
UniPELT (Mao et al., 2021) is a unified framework that combines **LoRA**, **Prefix-tuning**, and **Adapters** via a gating mechanism. It activates different sub-modules for different parts of the transformer block.

**Architecture:**
-   **LoRA:** Applied to $W_Q$ (Query) and $W_V$ (Value) attention matrices.
-   **Prefix-tuning:** Applied to Keys ($K$) and Values ($V$) at each layer.
-   **Adapters:** Added after the Feed-Forward (FFN) layer.
-   **Gating:** A learnable gating function (linear layer + sigmoid) controls the activation strength of each method.

**Benefits:**
-   **Robustness:** Performs particularly well in low-data scenarios (e.g., 100 examples).
-   **Performance:** Often surpasses individual methods by learning which combination works best for a specific task.
-   **Cost:** Uses slightly more parameters (~1.3% for BERT) compared to singular methods.

**Pseudocode (simplified):**
```python
def unipelt_self_attention(x):
    k, q, v = x @ W_k, x @ W_q, x @ W_v
    
    # LoRA for queries and values with gating
    lora_gate = gate(x)
    q += lora_gate * (x @ W_qA @ W_aB)
    v += lora_gate * (x @ W_vA @ W_vB)
    
    # Prefix tuning with gating
    pt_gate = gate(x)
    q_prefix = pt_gate * P_q
    k_prefix = pt_gate * P_k
    
    return softmax(q @ k.T) @ v
```

## 4. Compacter

**Concept:**
Compacter (Karimi Mahabadi et al., 2021) is a method that heavily utilizes **Kronecker products**, **low-rank matrices**, and **parameter sharing** to construct ultra-efficient adapter weights.

**Mechanism:**
-   It replaces the standard linear layers in adapters with **Parametrized Hypercomplex Multiplication (PHM)** layers.
-   Weights are decomposed into a sum of Kronecker products: $W = \sum A_i \otimes B_i$.
-   **Compacter++:** A variant that uses a single adapter after the FFN layer (similar to the Pfeiffer adapter configuration).

**Efficiency:**
-   Extremely parameter-efficient (e.g., 0.05% additional parameters).
-   Achieves comparable performance to standard adapters despite the massive reduction in parameters.

**Pseudocode:**
```python
def lphm_forward(x):
    # Decompose weight W into Kronecker product of A and B
    B = B_d @ B_u  # Low-rank B
    W = batched_kronecker_product(A, B)
    W = sum(W, dim=0)
    return x @ W + b
```

## 5. S4 (Search for Sparse and Soft Subspace)

**Concept:**
S4 (Chen et al., 2023) is a search-based approach that automatically finds the optimal combination of PEFT techniques (**Adapters**, **Prefix-tuning**, **BitFit**, **LoRA**) for different layers of the network.

**Strategy:**
-   **Spindle Pattern:** Divides layers into four groups ($G_1, G_2, G_3, G_4$). It allocates more trainable parameters to the middle groups and fewer to the top/bottom layers.
-   **Combinatorial Search:** Different groups utilize different sets of PEFT methods.
    -   Example: $G_1$ might use Adapters + LoRA, while $G_2$ uses Adapters + Prefix-tuning.

**Performance:**
-   Consistently outperforms individual methods (like BitFit, LoRA, Prefix Tuning) across various architectures (T5, RoBERTa) by optimizing "where" and "how" to fine-tune.

---

**Summary Table**

| Method | Composition | Key Strengths |
| :--- | :--- | :--- |
| **SparseAdapter** | Adapters + Pruning | High performance via large initial capacity. |
| **MAM Adapter** | Parallel Adapter + Soft Prompt | Optimal placement for FFN and Attention tuning. |
| **UniPELT** | LoRA + Prefix + Adapter + Gating | Robustness in low-resource settings; learns optimal mix. |
| **Compacter** | Adapters + Kronecker Products | Extreme parameter efficiency. |
| **S4** | Neural Architecture Search over PEFTs | Automated optimization of method per layer group. |
