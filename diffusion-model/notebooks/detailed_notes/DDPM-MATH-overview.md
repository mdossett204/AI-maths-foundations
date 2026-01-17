# DDPM / DDIM — Section 2 Annotated Walkthrough

This document annotates **Section 2 (Background)** of the DDIM paper.
Each equation is explained in terms of probabilistic modeling assumptions
and linked to the required statistical concepts.

---

## Section 2 Overview

Goal:
Learn a generative model $p_\theta(x_0)$ that approximates the data distribution $q(x_0)$.

Approach:
Introduce a sequence of latent variables $x_1, \dots, x_T$ and define
a **forward noising process** and a **reverse denoising process**.

---

## Equation (1): Latent Variable Model

$$
p_\theta(x_0) =
\int p_\theta(x_{0:T}) dx_{1:T}
$$

with:

$$
p_\theta(x_{0:T}) := p_\theta(x_T) \prod_{t=1}^T p_\theta^{(t)}(x_{t-1} \mid x_t)
$$

### Interpretation

- This is a **latent variable model**
- $x_1, \dots, x_T$ are latent variables of the same dimension as $x_0$
- Sampling proceeds **backwards** from noise to data

### Key assumption

The generative process is a **Markov chain**:

$$
x_t \rightarrow x_{t-1}
$$

---

## Equation (2): Variational Lower Bound (ELBO)

$$
\max_\theta \mathbb{E}_{q(x_0)}[\log p_\theta(x_0)]
\le
\max_\theta \mathbb{E}_{q(x_{0:T})}
\left[
\log p_\theta(x_{0:T}) - \log q(x_{1:T} \mid x_0)
\right]
$$

### Interpretation

- Exact likelihood is intractable
- Introduce a **fixed inference distribution** $q(x_{1:T} \mid x_0)$
- Optimize a lower bound (ELBO)

### Key DDPM design choice

- $q$ is **not learned**
- Only $p_\theta$ is learned

---

## Equation (3): Forward Diffusion Process

$$
q(x_t \mid x_{t-1}) =
\mathcal{N}
\left(
\sqrt{\frac{\alpha_t}{\alpha_{t-1}}} x_{t-1},
\left(1 - \frac{\alpha_t}{\alpha_{t-1}}\right) I
\right)
$$

### Interpretation

- Forward process is a **Gaussian Markov chain**
- Noise is gradually added
- Variance is strictly **positive**

### Consequence

- Noise accumulates slowly
- At large $t$, $x_t \approx \mathcal{N}(0, I)$

This assumption forces:

- Many timesteps
- Sequential sampling

---

## Key Property: Closed-Form Marginals

The forward process satisfies:

$$
q(x_t \mid x_0) =
\mathcal{N}
\left(
\sqrt{\alpha_t} x_0,
(1 - \alpha_t) I
\right)
$$

### Why this matters

- You can sample $x_t$ **directly** from $x_0$
- No need to simulate all intermediate steps

---

## Equation (4): Reparameterization

$$
x_t = \sqrt{\alpha_t} x_0 + \sqrt{1 - \alpha_t} \epsilon,
\quad \epsilon \sim \mathcal{N}(0, I)
$$

### Interpretation

- Decomposes $x_t$ into:
  - Signal: $\sqrt{\alpha_t} x_0$
  - Noise: $\sqrt{1 - \alpha_t} \epsilon$

This is the **core identity** of diffusion models.

---

## Prior Choice

When $\alpha_T \rightarrow 0$:

$$
q(x_T \mid x_0) \rightarrow \mathcal{N}(0, I)
$$

So define:

$$
p_\theta(x_T) := \mathcal{N}(0, I)
$$

### Interpretation

- Final latent variable is pure noise
- Sampling starts from Gaussian noise

---

## Equation (5): Noise-Prediction Objective

$$
\mathcal{L}_\gamma(\epsilon_\theta)
=
\sum_{t=1}^T
\gamma_t
\mathbb{E}_{x_0, \epsilon}
\left[
\|\epsilon_\theta^{(t)}(
\sqrt{\alpha_t} x_0 + \sqrt{1-\alpha_t}\epsilon
)
- \epsilon
  \|^2
  \right]
$$

### Interpretation

- Model predicts **noise**, not $x_0$
- Equivalent to ELBO under Gaussian assumptions
- Equivalent to score matching

### Important insight

The loss depends **only on**:

$$
q(x_t \mid x_0)
$$

NOT on:

$$
q(x_t \mid x_{t-1})
$$

This is the key fact DDIM exploits later.

---

## Sampling in DDPM (Implicit Assumption)

Sampling requires:

$$
x_{t-1} \sim p_\theta(x_{t-1} \mid x_t)
$$

Each step:

- Adds fresh noise
- Is stochastic
- Cannot be skipped

---

## Section 2 Summary (Critical Takeaways)

1. DDPM defines a **fixed Gaussian forward process**
2. Training depends only on **marginal distributions**
3. Reverse process is **stochastic**
4. Positive variance forces:
   - Markov sampling
   - Many steps

---

## Why This Enables DDIM (Preview)

Since:

$$
\mathcal{L} \text{ depends only on } q(x_t \mid x_0)
$$

We are free to:

- Change the **joint forward process**
- Remove stochasticity
- Skip timesteps

As long as marginals stay the same.

This is exactly what Section 3 formalizes.

---

## Mental Model Going Forward

Think of Section 2 as establishing:

- What is **necessary** (marginals)
- What is **incidental** (Markov chain, noise variance)

DDIM removes the incidental parts.
