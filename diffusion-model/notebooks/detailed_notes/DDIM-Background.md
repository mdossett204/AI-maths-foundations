# DDPM / DDIM — Section 3 Annotated Walkthrough

## Variational Inference for Non-Markovian Forward Processes

This document explains **Section 3** of the DDIM paper.
This section contains the _core theoretical insight_ behind DDIM.

---

## Section 3 Overview

Key claim of this section:

> The DDPM training objective depends **only on the marginals** $q(x_t \mid x_0)$, not on how those marginals are generated.

Therefore:

- The forward process does **not** need to be Markovian
- We can define alternative inference processes
- These lead to different (and faster) generative models
- Training remains unchanged

---

## Why This Section Exists

From Section 2:

- DDPM uses a Markov forward diffusion
- Reverse process must mirror it
- Sampling is slow

Observation:

- The loss in Equation (5) never uses:
  $$
  q(x_t \mid x_{t-1})
  $$
- It only uses:
  $$
  q(x_t \mid x_0)
  $$

This section formalizes that observation.

---

## Section 3.1 — Non-Markovian Forward Processes

### Equation (6): New Inference Factorization

$$
q_\sigma(x_{1:T} \mid x_0)
=
q_\sigma(x_T \mid x_0)
\prod_{t=2}^T q_\sigma(x_{t-1} \mid x_t, x_0)
$$

### What changed?

- Forward process is **no longer Markovian**
- Each $x_{t-1}$ may depend on:
  - $x_t$
  - $x_0$

This breaks the diffusion assumption.

---

### Why introduce $x_0$ dependency?

Because we want:
$$
q_\sigma(x_t \mid x_0)
=
\mathcal{N}
(\sqrt{\alpha_t} x_0, (1-\alpha_t)I)
$$

for **all t**, exactly as in DDPM.

---

## Equation (7): Conditional Gaussian Definition

$$
q_\sigma(x_{t-1} \mid x_t, x_0)
=
\mathcal{N}
\left(
\mu_\sigma(x_t, x_0),
\sigma_t^2 I
\right)
$$

Mean is chosen so that:

- Marginals match DDPM
- Noise schedule is preserved

### Important

This equation is _constructed_, not derived from physics.
It exists purely to preserve marginals.

---

## Key Lemma (Lemma 1)

Despite being non-Markovian:

$$
q_\sigma(x_t \mid x_0)
=
\mathcal{N}
(\sqrt{\alpha_t} x_0, (1-\alpha_t)I)
$$

### Meaning

The same closed-form marginal as DDPM.

This is the **only requirement** for the training loss.

---

## Section 3.2 — Generative Process Construction

Now define a **learned reverse process**.

---

### Equation (9): Predict Clean Image

$$
\hat{x}_0 =
f_\theta^{(t)}(x_t)
=
\frac{x_t - \sqrt{1-\alpha_t}\,\epsilon_\theta(x_t)}{\sqrt{\alpha_t}}
$$

This is identical to DDPM.

The model still predicts noise.

---

## Equation (10): Reverse Process Definition

$$
p_\theta^{(t)}(x_{t-1} \mid x_t)
=
q_\sigma(x_{t-1} \mid x_t, \hat{x}_0)
$$

### Interpretation

- Replace true $x_0$ with predicted $\hat{x}_0$
- Use the same Gaussian as the inference model
- This mirrors standard variational inference

---

## Equation (11): New Variational Objective

$$
J_\sigma(\epsilon_\theta)
=
\mathbb{E}_{q_\sigma(x_{0:T})}
[
\log q_\sigma(x_{1:T} \mid x_0)
- \log p_\theta(x_{0:T})
  ]
$$

### At first glance

Looks like a _different_ objective for each $\sigma$.

---

## Theorem 1 (Critical Result)

> For any $\sigma > 0$,
> $J_\sigma = \mathcal{L}_\gamma + C$

Where:

- $\mathcal{L}_\gamma$ is the DDPM noise loss
- $C$ is a constant independent of $\theta$

---

## What Theorem 1 Means

1. Different forward processes
2. Different joint distributions
3. **Same training objective**

Therefore:

- A DDPM model is _already trained_ for all $\sigma$
- No retraining needed
- Sampling procedure is free to change

---

## Why This Is So Powerful

The DDPM loss:
$$
\mathbb{E}\|\epsilon - \epsilon_\theta(x_t)\|^2
$$

does **not care**:

- How $x_t$ was reached
- Whether noise was injected gradually
- Whether the process was Markovian

It only cares that:
$$
x_t \sim \mathcal{N}(\sqrt{\alpha_t}x_0, (1-\alpha_t)I)
$$

---

## Transition to DDIM

By choosing:
$$
\sigma_t = 0
$$

We get:

- Deterministic forward process
- Deterministic reverse process
- Implicit generative model

This leads directly to Section 4.

---

## Section 3 Summary (Key Takeaways)

1. DDPM training depends only on marginals
2. Joint inference process is flexible
3. Non-Markovian inference is allowed
4. Deterministic generative models are valid
5. DDIM is one member of a large family

---

## Mental Model

Think of DDPM training as learning:

- A denoising **vector field**

Section 3 shows:

- Many paths can follow that field
- Markov diffusion is only one choice

DDIM picks the straightest one.
