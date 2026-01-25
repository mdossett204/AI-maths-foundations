# Additive PEFT Methods: Summary & Intuition

**Parameter-Efficient Fine-Tuning (PEFT)** aims to adapt large pre-trained models (LLMs) to specific tasks without retraining all parameters. **Additive Methods** are a primary category of PEFT where **new, trainable parameters are added** to the model (or input) while the original pre-trained weights remain **frozen**.

## 1. Adapters
**Mechanism:**
Adapters introduce small, fully connected networks (bottleneck layers) inside the existing transformer architecture, typically after the multi-head attention and feed-forward blocks.
*   **Structure:** Down-projection (Input dimension $d$ $\to$ bottleneck dimension $r$, where $r \ll d$) $\to$ Non-linearity $\to$ Up-projection ($r \to d$).
*   **Residual Connection:** The adapter output is usually added to the original activation (skip connection), ensuring the model defaults to its pre-trained behavior if the adapter weights are near zero.

**Intuition:**
Think of the pre-trained model as a universal engine. Instead of rebuilding the engine for a specific car (full fine-tuning), adapters act as custom "connectors" or "gearboxes" inserted into the mechanism. They intercept the flow of information, tweak the features slightly to align with the new task, and pass them along. Because they are inserted *internally*, they have fine-grained control over the model's reasoning process.

## 2. Soft Prompts (Prompt Tuning & Prefix Tuning)
**Mechanism:**
Instead of modifying the model architecture, these methods modify the **context** passed to the model by adding trainable "virtual tokens" (continuous vectors).
*   **Prompt Tuning:** Prepends trainable embeddings to the *input layer* only.
*   **Prefix Tuning:** Prepends trainable tensors (prefixes) to the keys and values at *every layer* of the transformer stack.

**Intuition:**
This is akin to finding the perfect "magic words" to whisper to the model to get it to do what you want—but instead of searching for English words (like "Translate this:"), we search for the optimal numerical vectors in the model's latent space.
*   *Why it works:* It steers the model's internal activation trajectory. Prefix tuning is generally more expressive/powerful than Prompt tuning because it can influence the model at every stage of reasoning, not just the start.

## 3. Ladder Side Tuning (LST)
**Mechanism:**
LST creates a separate, lightweight "side network" (the ladder) that runs in parallel to the massive frozen backbone.
*   It takes intermediate activations from the backbone layers via "ladder connections."
*   The side network processes these features and generates the final output.
*   Crucially, backpropagation only happens through the small side network, not the giant backbone.

**Intuition:**
This uses the LLM as a fixed, high-quality feature extractor. The side network is like a small, agile student observing the master (the LLM). The student takes the master's intermediate thoughts (activations), processes them quickly, and produces the answer. It avoids the massive memory cost of calculating gradients for the master model.

## 4. (IA)³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)
*Note: Sometimes classified as reparameterization, but functionally additive in terms of parameters.*
**Mechanism:**
Injects learned vectors that element-wise multiply (scale) the activations in the Key, Value, and Feed-Forward layers.

**Intuition:**
Instead of rewriting the knowledge, (IA)³ acts like a complex equalizer board. It learns to "turn up" or "turn down" specific feature channels in the model's internal processing to suit the new task.

---

## Why Do Additive Methods Work? (The Core Intuition)

The success of these methods relies on the hypothesis of **Low Intrinsic Dimension**.

1.  **Over-parameterization:** Large models (LLMs) are vastly over-parameterized. They contain a "Super-knowledge" space that covers a wide variety of tasks.
2.  **Subspace Optimization:** To perform a *specific* new task (like medical summarization), the model doesn't need to change its entire brain. The solution to that specific task likely exists within a very small, low-dimensional subspace of the model's total parameter space.
3.  **Steering vs. Re-learning:** Additive methods simply learn the coordinates to "steer" the pre-trained model into that specific subspace. By adding a small number of parameters (the steering wheel), we can navigate the vast knowledge of the frozen model without needing to rebuild the map.
