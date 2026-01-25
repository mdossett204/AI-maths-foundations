# Reparametrization-based PEFT Methods

Reparametrization-based Parameter-Efficient Fine-Tuning (PEFT) methods leverage low-rank representations to minimize the number of trainable parameters. Instead of training the full weight matrices directly, these methods reparametrize the weights (or weight updates) using low-rank transformations or matrix factorizations. This allows for efficient updates to high-dimensional matrices while keeping the trainable parameter count low.

## Key Methods

### 1. Intrinsic SAID (Section 9.1)
*   **Concept:** Investigates the intrinsic dimensionality of fine-tuning, demonstrating that optimization can effectively occur in low-rank subspaces.
*   **Mechanism:** Utilizes the **Fastfood transform** to reparametrize the update to the model weights. The update is defined as $\theta = \theta_0 + F(\theta^d)$, where $F$ is the transform and $\theta^d$ are the low-dimensional parameters.
*   **Pros:** Theoretical foundation for low-rank fine-tuning.
*   **Cons:** Updates *all* model parameters and has high memory complexity ($O(D)$), making it impractical for very large networks compared to newer methods.

### 2. LoRA (Low-Rank Adaptation) (Section 9.2)
*   **Concept:** Decomposes the weight update matrix $\Delta W$ into a product of two low-rank matrices.
*   **Mechanism:** The update is defined as $\Delta W = W_A W_B$, where $W_A \in \mathbb{R}^{d_{in} \times r}$ and $W_B \in \mathbb{R}^{r \times d_{out}}$. The original weight $W$ is frozen.
*   **Pros:** 
    *   Highly parameter-efficient (rank $r$ is typically small, e.g., 4, 8, 16).
    *   No inference latency overhead (weights can be merged: $W_{new} = W + W_A W_B$).
    *   Consistently strong performance across various tasks and model sizes.

### 3. KronA (Section 9.3)
*   **Concept:** Utilizes the **Kronecker product** for matrix factorization instead of standard matrix multiplication.
*   **Mechanism:** The update is defined as $\Delta W = W_A \otimes W_B$.
*   **Pros:** 
    *   Preserves the rank of the original matrices better than matrix multiplication ($\text{rank}(A \otimes B) = \text{rank}(A) \cdot \text{rank}(B)$).
    *   Offers a better rank-to-parameter tradeoff.
    *   Computational speedups via efficient Kronecker product-vector operations.

### 4. DoRA (Weight-Decomposed LoRA) (Section 9.4)
*   **Concept:** Decouples the **magnitude** and **direction** of the weight updates to mimic the learning dynamics of full fine-tuning.
*   **Mechanism:** Decomposes weights into a magnitude vector $m$ and a directional matrix $V$ ($W = m \frac{V}{||V||}$). LoRA is applied specifically to the directional component $V$, while $m$ is trained separately.
*   **Pros:** 
    *   Often outperforms standard LoRA.
    *   Robust to hyperparameter changes.
    *   Achieves high performance even with very low ranks.

### 5. GLoRA (Generalized LoRA) (Section 9.5)
*   **Concept:** A generalized formulation that extends LoRA by adding more learnable parameters to scale and shift parameters or activations.
*   **Mechanism:** Uses a generalized update formula: $W = W_0 + W_0 A + B$, where additional matrices $A$ and $B$ are learned (often also factorized).
*   **Pros:** 
    *   Greater expressivity for difficult tasks.
    *   Maintains efficiency through low-rank approximations of the added components.

### 6. AdaLoRA (Section 9.6)
*   **Concept:** Adaptively allocates the rank $r$ across different layers (pruning ranks in less important layers) rather than using a fixed rank globally.
*   **Mechanism:** Formulates updates as SVD-like triplets: $W = W + W_A \Lambda W_B$. It prunes the singular values in the diagonal matrix $\Lambda$ based on importance scores derived from sensitivity.
*   **Pros:** 
    *   Allocates parameter budget more effectively (more params where needed).
    *   Better performance-to-parameter ratio than fixed-rank LoRA.

### 7. GaLore (Section 9.7)
*   **Concept:** Gradient Low-Rank Projection. Unlike others that low-rank the *weights*, GaLore trains *all* parameters but projects the **gradients** into a low-rank subspace.
*   **Mechanism:** Projects gradients into a low-rank form $G = P G_{low} Q^T$, updates the optimizer states in this low-rank space, and then projects back to update full weights.
*   **Pros:** 
    *    significantly reduces **optimizer state memory** (by up to 65%).
    *   Allows full-parameter learning dynamics on consumer hardware (e.g., pre-training 7B models on 24GB GPUs).

### 8. Quantization and LoRA (Section 9.8)
*   **Concept:** Combines LoRA with **Backbone Quantization** to further reduce memory footprint. Since the frozen pre-trained model takes up the majority of memory, quantizing it (e.g., to 4-bit) is highly effective.
*   **Key Method: QLoRA (Quantized LoRA)**
    *   **Mechanism:**
        1.  **4-bit NormalFloat (NF4):** A data type optimal for normally distributed weights.
        2.  **Double Quantization:** Quantizing the quantization constants themselves to save even more memory.
        3.  **Paged Optimizers:** Offloads optimizer states to CPU RAM when GPU runs out of memory.
        4.  **On-the-fly Dequantization:** Weights are **stored** in 4-bit to save memory but are **dequantized** to 16-bit (BF16/FP16) for the actual matrix multiplication (forward/backward pass). This ensures gradients for the adapters are computed with sufficient precision.
    *   **Pros:** Enables fine-tuning of massive models (e.g., 65B) on a single GPU without performance degradation.

### 9. Challenge: Quantizing the Final Model (Inference)
While QLoRA makes *training* efficient, the resulting LoRA adapters are still in high precision (16-bit).
*   **The Issue:** If you merge the adapters into the base model and try to **quantize the final result** (Post-Training Quantization) for faster *inference*, performance often degrades significantly. The adapters were not trained to survive this final compression.
*   **Solutions (Quantization-Aware PEFT):**
    *   **QA-LoRA, LoftQ, IR-QLoRA, L4Q:** These methods integrate quantization awareness *into* the training process itself (or initialize weights smartly). This ensures the final model remains accurate even when quantized for deployment.

## Mathematical Foundations

### Matrix Factorization & Low-Rank Transformation
Matrix factorization approximates a large matrix $W$ of dimensions $m \times n$ by the product of two smaller matrices $A$ ($m \times r$) and $B$ ($r \times n$), where $r$ is the **rank** and $r \ll \min(m, n)$.

$$ W \approx A \times B $$

*   **Goal:** To capture the most important information of $W$ using fewer parameters.
*   **Parameter Reduction:** A full matrix requires $m \times n$ parameters. The low-rank form requires $r(m + n)$.
    *   *Example:* For a $1000 \times 1000$ matrix ($1,000,000$ params) with rank $r=8$, the low-rank form needs only $8(1000 + 1000) = 16,000$ params (a ~98% reduction).

### Singular Value Decomposition (SVD)
SVD is a specific method to perform matrix factorization. It decomposes a real matrix $W$ into three matrices:

$$ W = U \Sigma V^T $$

1.  **$U$ (Left Singular Vectors):** Orthogonal matrix capturing row patterns.
2.  **$\Sigma$ (Singular Values):** Diagonal matrix with non-negative values ($\sigma_1, \sigma_2, ...$) sorted in descending order. These represent the "strength" or "importance" of each dimension.
3.  **$V^T$ (Right Singular Vectors):** Orthogonal matrix capturing column patterns.

**Low-Rank Approximation via SVD:** By keeping only the top-$r$ largest singular values (and setting the rest to zero), we get the optimal rank-$r$ approximation of $W$ (in terms of L2 norm). This is the mathematical basis for why methods like LoRA and AdaLoRA work—neural network weight changes often exist in a low intrinsic rank.

### Post-Training Quantization (PTQ)
Quantization reduces the precision of model weights (e.g., from 32-bit floating point to 8-bit integer) to save memory and computation *after* the model has been trained.

$$ W_{quant} = \text{round}\left(\frac{W_{fp32}}{S}\right) + Z $$

*   **$W_{fp32}$:** Original high-precision weight.
*   **$S$ (Scale):** A factor that maps the range of float values to the integer range.
*   **$Z$ (Zero-Point):** Shifts the values so the zero in float space maps to a specific integer.
*   **Result:** The storage cost drops (e.g., 32 bits $\to$ 8 bits = 4x compression), but small errors are introduced. QLoRA manages these errors effectively even at 4-bit precision.
