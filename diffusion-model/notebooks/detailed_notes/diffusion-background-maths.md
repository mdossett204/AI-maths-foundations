# Diffusion Models – Math Cheatsheet

This cheatsheet covers the minimum statistical background needed to understand
Section 2 of DDPM / DDIM, plus optional concepts that clarify later sections.

---

## 1. Gaussian Distributions & Reparameterization

### Multivariate Gaussian

A Gaussian in $\mathbb{R}^d$:

$$
\mathcal{N}(x \mid \mu, \Sigma)
$$

Special case (used in diffusion):

$$
\mathcal{N}(x \mid \mu, \sigma^2 I)
$$

Mean controls **location**, variance controls **noise magnitude**.

---

### Reparameterization Trick

Sampling:

$$
x \sim \mathcal{N}(\mu, \sigma^2 I)
$$

Equivalent to:

$$
x = \mu + \sigma \epsilon,\quad \epsilon \sim \mathcal{N}(0, I)
$$

Used heavily in diffusion to:

- Separate **signal** and **noise**
- Enable deterministic transformations

---

### Key Identity Used in DDPM

For constants $a, b$:

$$
\mathcal{N}(a x_0, b I) \Rightarrow x = a x_0 + \sqrt{b}\epsilon
$$

---

## 2. Variational Inference & ELBO (Intuition)

Goal:
Approximate an unknown data distribution $q(x_0)$ with a model $p_\theta(x_0)$.

---

### Latent Variable Model

Introduce latent variables $x_1, \dots, x_T$:

$$
p_\theta(x_0) = \int p_\theta(x_{0:T}) dx_{1:T}
$$

Exact marginal likelihood is intractable.

---

### Evidence Lower Bound (ELBO)

We maximize:

$$
\log p_\theta(x_0) \ge
\mathbb{E}_{q(x_{1:T} \mid x_0)}
\left[
\log p_\theta(x_{0:T}) - \log q(x_{1:T} \mid x_0)
\right]
$$

Interpretation:

- Encourage model to match reverse of inference process
- Penalize mismatch between true and learned transitions

---

### Key DDPM Property

The ELBO simplifies because:

- $q(x_{1:T} \mid x_0)$ is **fixed**
- All transitions are **Gaussian**

---

## 3. DDPM Basics

### Forward Process (Noising)

Defined as a Markov chain:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}(\sqrt{\alpha_t} x_{t-1}, (1 - \alpha_t) I)
$$

Noise is added gradually:

$$
x_0 \rightarrow x_1 \rightarrow \dots \rightarrow x_T
$$

---

### Closed-Form Marginal

Important identity:

$$
q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t)I)
$$

Where:

$$
\bar{\alpha}_t = \prod_{i=1}^t \alpha_i
$$

---

### Reparameterized Form

$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon
$$

This is the **core equation** behind DDPM and DDIM.

---

## 4. Noise-Prediction Objective

Instead of predicting $x_0$, DDPM predicts noise:

$$
\epsilon \sim \mathcal{N}(0, I)
$$

Training objective:

$$
\mathbb{E}_{x_0, \epsilon, t}
\left[
\|\epsilon - \epsilon_\theta(x_t, t)\|^2
\right]
$$

Why predict noise?

- Stable optimization
- Equivalent to score matching
- Allows analytic posterior means

---

### Recovering $x_0$ from Noise

$$
\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\,\epsilon_\theta(x_t)}{\sqrt{\bar{\alpha}_t}}
$$

Used directly in DDIM sampling.

---

## 5. Conditional Gaussian Sampling

General rule:
If:

$$
p(x \mid y) = \mathcal{N}(\mu(y), \Sigma)
$$

Then sampling:

$$
x = \mu(y) + \Sigma^{1/2} \epsilon
$$

---

### In DDPM

Reverse transitions:

$$
p_\theta(x_{t-1} \mid x_t)
$$

Are Gaussian with:

- Mean computed from $\epsilon_\theta$
- Fixed variance

Requires **fresh noise each step**.

---

### In DDIM

Variance is set to zero:

$$
\sigma_t = 0
$$

Sampling becomes deterministic.

---

## 6. (Optional) Score Matching

Score:

$$
\nabla_x \log p(x)
$$

DDPM noise predictor approximates:

$$
\epsilon_\theta(x_t) \approx -\sigma_t \nabla_{x_t} \log q(x_t)
$$

This connects diffusion models to:

- Score-based generative modeling
- SDE formulations

---

## 7. (Optional) ODE Intuition

DDIM corresponds to solving:

$$
\frac{dx}{dt} = f_\theta(x, t)
$$

Instead of sampling from a stochastic process.

Key ideas:

- Deterministic trajectory
- Fewer steps needed
- Same endpoint distribution (in theory)

---

## 8. (Optional) Langevin Dynamics

Langevin update:

$$
x_{k+1} = x_k + \eta \nabla \log p(x_k) + \sqrt{2\eta}\epsilon
$$

Properties:

- Stochastic sampling
- Requires many steps
- Used in early score-based models

DDPM ≈ discretized Langevin with annealed noise  
DDIM ≈ removing stochastic term

---

## 9. Big Picture Summary

- DDPM: stochastic reverse Markov chain
- Training depends only on marginals
- DDIM: deterministic path using same model
- Skipping steps works because noise levels are learned independently
