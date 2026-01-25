# Selection-Based PEFT Methods

**Selection-based methods** fine-tune a subset of the **existing** parameters of the model. Unlike additive methods (Adapters) or reparametrization (LoRA), these methods do not introduce new matrices or layers; they simply identify "important" weights in the pre-trained model and update only those.

## 1. The Inference Efficiency Question

> **User Question:** _"Since it is not additive, is the inference gain good?"_

**Answer:** **Yes, but with a caveat regarding implementation.**

- **Latency (Speed):** **Excellent (No Overhead).** Because you are updating existing parameters, the model architecture remains identical to the original.
  - _Mechanism:_ You can permanently merge the learned sparse updates into the original weights: $W_{final} = W_{frozen} + \Delta W_{sparse}$.
  - _Result:_ The model runs at the exact same speed as the original base model. There are no extra adapter layers to calculate.
- **Storage:** **Excellent.** You only need to save the sparse "diff" (the changed values) and their positions (indices). This allows storing hundreds of task-specific versions on a small device (like a phone) while sharing one frozen backbone.
- **The Caveat:** If you _don't_ merge weights and try to run the sparse operation dynamically (calculating $W + \text{mask} \odot \Delta W$ on the fly), it can actually be **slower** than full fine-tuning because standard GPUs are not optimized for sparse math (scattering/gathering non-zero values).

---

## 2. Key Selection Methods

### A. BitFit (Bias Tuning)

- **Concept:** Freeze all weights ($W$), train **only the bias vectors** ($b$).
- **Selection Rationale:** Hypothesis that biases control the "threshold" of neuron activation. Modifying thresholds is enough to shift the domain for simple tasks.
- **Pros:** Easiest to implement; tiny parameter count (<0.1%).
- **Cons:**
  - **Capacity Gap:** Works well for small models (BERT) but **fails** for Large Language Models (>1B params) or complex generation tasks. Biases simply don't provide enough "steering" power for massive models.
  - **Architecture dependent:** Some models (like LLaMA) don't use biases, making this method impossible. T5 uses "relative attention biases" which can be tuned.

### B. DiffPruning

- **Concept:** Learn a **sparse update vector** ($\delta$) that is added to the weights.
- **Math:** $\delta = z \circ \Delta W$
  - $\Delta W$: The weight update.
  - $z$: A learnable binary mask (0 or 1).
  - $\circ$: **Hadamard Product** (Element-wise multiplication).
- **Selection Rationale:** Uses **$L_0$ regularization**. The loss function includes a penalty for every non-zero parameter. The model "learns" which parameters are necessary to update and kills the rest (sets them to 0).
- **Pros:** Extremely efficient for storage.
- **Cons:** **Training is heavy.** You must compute gradients for _all_ parameters during training to decide which ones to keep, even if you only end up keeping 0.5%.

#### 📝 Note: What is L0 Regularization?

- **L2 Norm (Ridge):** Sum of **squared** values ($\sum w^2$). Pushes weights to be small (e.g., 0.0001) but rarely zero.
- **L1 Norm (Lasso):** Sum of **absolute** values ($\sum |w|$). Pushes weights toward zero, encouraging sparsity.
- **L0 Norm:** Count of **non-zero** values.
  - _Effect:_ It doesn't care if a weight is 100 or 0.01; it penalizes them equally. It strictly wants to minimize the **number** of active parameters.
  - _Challenge:_ L0 is non-differentiable (you can't calculate the gradient of a "count"). DiffPruning approximates this (often using a "relaxed" L0 via a stretched sigmoid) to allow gradient descent to work.

### C. FishMask

- **Concept:** Select the top-$p$ parameters to train based on their **Fisher Information**.
- **Selection Rationale:** **Fisher Information** approximates how sensitive the loss function is to a specific parameter.
  - _High Fisher Score_ = Changing this weight impacts the output significantly (Important).
  - _Low Fisher Score_ = Changing this weight does nothing (Ignore).
- **Pros:** Theoretically grounded selection criterion.
- **Cons:** Computationally expensive start. You must run a "diagnostic" pass to calculate gradients and Fisher info for the whole model before training begins.

#### 📝 Note: Intuition on Fisher Information

Imagine the loss landscape as a terrain:

- **Steep Cliffs (High Fisher):** If a weight is on a cliff edge, a tiny change causes the loss to skyrocket or plummet. This weight is **critical**.
- **Flat Plains (Low Fisher):** If a weight is on a flat plain, you can change it significantly without affecting the loss. This weight is **unimportant**.
  FishMask effectively finds the "cliffs" and only trains the parameters located there.

### D. FAR (Freeze and Reconfigure)

- **Concept:** A **structured** selection method. Instead of picking random dots (weights), it selects entire **columns or rows** of the weight matrix.
- **Selection Rationale:** **Magnitude of Change ($L_1$ Norm)**.
  1.  Fine-tune the whole model for a few steps (warm-up).
  2.  Measure which weights moved the farthest from their starting point.
  3.  Group these into columns/rows.
  4.  Keep the top-$r$ most changed groups trainable, freeze the rest.
- **Pros:** **Hardware Friendly.** GPUs prefer processing dense blocks (rows/columns) rather than random sparse dots (BitFit/DiffPruning).

#### 📝 Note: L1 in FAR vs. L1 in Loss

- **L1 in Loss (e.g., Lasso):** Used as a _penalty_ to force weights to zero during training.
- **L1 in FAR:** Used as a **Ranking Metric** (measuring stick).
  - FAR calculates $|W_{current} - W_{frozen}|$ to see how much a weight "wanted" to move.
  - It assumes that if a weight moved a lot during the warm-up, it is important for the task. It does not use L1 to penalize the model; it uses it to _select_ the winners.

---

## 3. Summary of Rationales (How they choose what to train)

| Method          | Selection Criterion       | Logic                                                                                     |
| :-------------- | :------------------------ | :---------------------------------------------------------------------------------------- |
| **BitFit**      | Architecture-based        | "Biases are the easiest levers to pull."                                                  |
| **FishMask**    | Gradient Sensitivity      | "Train the weights that the loss function cares about most (High Fisher Info/Curvature)." |
| **DiffPruning** | Optimization ($L_0$ Norm) | "Let the optimizer learn to mask out updates (approximate L0)."                           |
| **FAR**         | Magnitude ($L_1$ Norm)    | "If it moved a lot during a test run, it must be important."                              |
