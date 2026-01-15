# DDIM Paper — Single-Page Summary & Comparison

This document provides:

1. A one-page conceptual summary of _Denoising Diffusion Implicit Models_
2. A structured comparison between DDPM and DDIM

---

## 1. DDIM — Single-Page Summary

### Problem with DDPM

Denoising Diffusion Probabilistic Models (DDPMs):

- Define a **Markovian forward diffusion**
- Require **stochastic reverse sampling**
- Need **hundreds to thousands of timesteps**
- Are slow at inference

---

### Key Observation

The DDPM training objective:
$$
\mathbb{E}\|\epsilon - \epsilon_\theta(x_t)\|^2
$$

depends **only on the marginal distribution**:
$$
q(x_t \mid x_0) = \mathcal{N}(\sqrt{\alpha_t}x_0, (1-\alpha_t)I)
$$

It does **not** depend on:

- Markov structure
- Forward transition variance
- How \( x_t \) was generated

---

### Core Insight

Because only marginals matter:

- Many different forward processes are valid
- The reverse process can be deterministic
- Sampling does not need injected noise

DDPM training implicitly learns a **family of generative models**.

---

### Non-Markovian Forward Processes

DDIM introduces:

- A non-Markovian inference process
- Conditional transitions that depend on \( x_0 \)
- The same marginals as DDPM

This preserves the training objective exactly.

---

### DDIM Sampling

DDIM uses the update:
$$
x_{t-1} = \sqrt{\alpha_{t-1}}\,\hat{x}_0 + \sqrt{1-\alpha_{t-1}}\,\epsilon_\theta(x_t)
$$

Properties:

- Deterministic
- No injected noise
- Same trained model

---

### Accelerated Generation

Because:

- The model is trained for all noise levels
- Sampling follows a deterministic trajectory

We can:

- Skip timesteps
- Use 10–50 steps instead of 1000
- Preserve sample quality

---

### ODE Interpretation

DDIM corresponds to:

- The probability flow ODE
- Deterministic integration of a learned vector field

This explains:

- Invertibility
- Reconstruction
- Fast sampling

---

### Final Takeaway

> DDIM reinterprets diffusion models as deterministic generative processes,
> enabling fast sampling without retraining.

---

## 2. DDPM vs DDIM Comparison Table

| Aspect                         | DDPM                   | DDIM                                 |
| ------------------------------ | ---------------------- | ------------------------------------ |
| Training objective             | Noise prediction (MSE) | Same as DDPM                         |
| Forward process                | Markovian diffusion    | Non-Markovian (implicit)             |
| Reverse process                | Stochastic             | Deterministic                        |
| Noise injected during sampling | Yes                    | No                                   |
| Sampling variance              | Fixed, > 0             | 0                                    |
| Sampling type                  | Random walk            | Deterministic trajectory             |
| Number of steps                | ~1000                  | 10–50                                |
| Can skip timesteps             | No                     | Yes                                  |
| Model retraining required      | No                     | No                                   |
| Sample diversity source        | Sampling noise         | Initial noise only                   |
| Interpretation                 | Diffusion / Langevin   | ODE / flow-like                      |
| Invertibility                  | No                     | Yes (theoretical)                    |
| Relation to score models       | Discretized SDE        | Probability flow ODE                 |
| Speed                          | Slow                   | Fast                                 |
| Image quality                  | High                   | Comparable / higher at equal compute |

---

## 3. Mental Model Comparison

### DDPM

- Add noise gradually
- Remove noise stochastically
- Requires small steps for correctness

### DDIM

- Predict clean image
- Rescale noise analytically
- Follow a deterministic path

---

## 4. One-Sentence Summary

> DDPM learns how to denoise; DDIM decides how to use that knowledge efficiently.