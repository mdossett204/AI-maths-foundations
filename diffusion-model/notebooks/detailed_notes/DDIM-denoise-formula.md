# DDPM / DDIM — Section 4 Cheatsheet

## Denoising Diffusion Implicit Models (Sampling)

This cheatsheet summarizes **Section 4** of the DDIM paper.
It explains how the deterministic DDIM sampler is constructed,
how timestep skipping works, and how this connects to DDPM.

---

## Section 4 Overview

Goal:
Construct a **fast sampling procedure** using a model trained with DDPM,
without changing training or retraining the network.

Key idea:

- Use the non-Markovian formulation from Section 3
- Choose a particular noise level $\sigma_t$
- Recover DDPM or DDIM as special cases

---

## 4.1 Denoising Diffusion Implicit Models

### Recap: Predicting the Clean Image

From Section 3 (Equation 9):

$$
\hat{x}_0 =
\frac{x_t - \sqrt{1-\alpha_t}\,\epsilon_\theta(x_t)}{\sqrt{\alpha_t}}
$$

This equation is reused directly in DDIM.

---

## Core DDIM Sampling Equation (Equation 12)

$$
x_{t-1}
=
\sqrt{\alpha_{t-1}}\,\hat{x}_0
+ \sqrt{1-\alpha_{t-1}-\sigma_t^2}\,\epsilon_\theta(x_t)
+ \sigma_t z
\quad,\quad z \sim \mathcal{N}(0, I)
$$

This defines a **family of samplers** indexed by $\sigma_t$.

---

## Special Cases

### DDPM Sampling

Choose:

$$
\sigma_t^2 =
\tilde{\beta}_t
$$

Then:

- Sampling is stochastic
- Matches original DDPM reverse process
- Requires many steps

---

### DDIM Sampling

Choose:

$$
\sigma_t = 0
$$

Then:

$$
x_{t-1}
=
\sqrt{\alpha_{t-1}}\,\hat{x}_0
+ \sqrt{1-\alpha_{t-1}}\,\epsilon_\theta(x_t)
$$

Properties:

- Deterministic
- No injected noise
- Same model, different sampler

---

## Interpretation of the DDIM Update

Each $x_t$ is decomposed as:

$$
x_t =
\sqrt{\alpha_t} x_0
+ \sqrt{1-\alpha_t} \epsilon
$$

DDIM performs:

1. **Projection** onto predicted clean image $\hat{x}_0$
2. **Reconstruction** at lower noise level $t-1$
3. **No randomness introduced**

This traces a deterministic trajectory.

---

## Mapping to Implementation (Python)

Given:

- `ab[t]` = $\alpha_t$
- `pred_noise` = $\epsilon_\theta(x_t)$

DDIM update:

```python
# 1. Predict clean image x_0
# (x_t - sqrt(1-alpha_t) * eps) / sqrt(alpha_t)
x0_pred = (x - sqrt(1 - ab) * pred_noise) / sqrt(ab)

# 2. Calculate direction to x_t
# sqrt(1-alpha_{t-1}) * eps
dir_xt = sqrt(1 - ab_prev) * pred_noise

# 3. Combine to get x_{t-1}
x_prev = sqrt(ab_prev) * x0_pred + dir_xt
```
