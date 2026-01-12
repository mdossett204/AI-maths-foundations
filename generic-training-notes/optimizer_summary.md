# Optimization Algorithms Summary

## 1. Vanilla SGD (Stochastic Gradient Descent)

### Concept
The simplest optimizer - takes steps directly in the direction of the negative gradient.

### Formula
```
θ_{t+1} = θ_t - η · g_t
```

Where:
- `θ_t` = parameters at step t
- `η` = learning rate
- `g_t` = gradient at step t

### Characteristics
- ✅ Simple and straightforward
- ❌ Noisy updates (follows gradient exactly)
- ❌ No adaptation to parameter scales
- ❌ Can oscillate in narrow valleys
- ❌ Fixed learning rate for all parameters

---

## 2. SGD with Momentum

### Concept
Adds "velocity" by maintaining an exponential moving average of past gradients. This smooths out noisy gradients and accelerates learning in consistent directions.

### Formula
```
m_t = β · m_{t-1} + (1 - β) · g_t
θ_{t+1} = θ_t - η · m_t
```

Where:
- `m_t` = momentum (exponential moving average of gradients)
- `β` = momentum coefficient (typically 0.9)
- `η` = learning rate
- `g_t` = gradient at step t

### Characteristics
- ✅ Smooths noisy gradients
- ✅ Accelerates in consistent directions
- ✅ Helps escape shallow local minima
- ✅ Reduces oscillations
- ❌ Still uses fixed learning rate for all parameters
- ❌ No adaptation to gradient magnitude

### Intuition
- **β = 0.9** means effective memory of ~10 steps: `1/(1-0.9) = 10`
- 90% from history, 10% from current gradient
- Like a ball rolling downhill - builds momentum in consistent directions

---

## 3. Adam (Adaptive Moment Estimation)

### Concept
Combines momentum with adaptive learning rates per parameter. Tracks both the first moment (mean) and second moment (variance) of gradients to adaptively scale learning rates.

### Full Algorithm

**Initialize:**
```
m_0 = 0  (first moment vector)
v_0 = 0  (second moment vector)
t = 0    (timestep)
```

**At each step t:**
```
t = t + 1
g_t = ∇θ L(θ_{t-1})  (compute gradient)

# Update biased first moment estimate
m_t = β₁ · m_{t-1} + (1 - β₁) · g_t

# Update biased second raw moment estimate
v_t = β₂ · v_{t-1} + (1 - β₂) · g_t²

# Compute bias-corrected first moment estimate
m̂_t = m_t / (1 - β₁^t)

# Compute bias-corrected second raw moment estimate
v̂_t = v_t / (1 - β₂^t)

# Update parameters
θ_t = θ_{t-1} - η · m̂_t / (√v̂_t + ε)
```

### Hyperparameters (Default Values)
- `η` = learning rate (typically 0.001)
- `β₁` = 0.9 (exponential decay rate for first moment)
- `β₂` = 0.999 (exponential decay rate for second moment)
- `ε` = 1e-8 (small constant for numerical stability)

### Component Breakdown

#### First Moment (m_t) - Momentum/Direction
```
m_t = β₁ · m_{t-1} + (1 - β₁) · g_t
```
- Exponential moving average of gradients
- Tells us **which direction** to move
- Memory window: ~10 steps (1/(1-0.9))

#### Second Moment (v_t) - Scale/Variance
```
v_t = β₂ · v_{t-1} + (1 - β₂) · g_t²
```
- Exponential moving average of **squared** gradients
- Tracks **how noisy/large** gradients are
- Memory window: ~1000 steps (1/(1-0.999))
- High v_t → noisy/uncertain → need smaller steps
- Low v_t → stable/confident → can take larger steps

#### Bias Correction
```
m̂_t = m_t / (1 - β₁^t)
v̂_t = v_t / (1 - β₂^t)
```
- Corrects initialization bias (starting from m_0 = 0, v_0 = 0)
- At t=1: correction factor = 1/0.1 = 10 (large correction)
- At t→∞: correction factor → 1 (no correction needed)

#### Final Update
```
θ_t = θ_{t-1} - η · m̂_t / (√v̂_t + ε)
                      ↑         ↑
                  direction   scale
```
- **Numerator (m̂_t)**: smoothed gradient direction
- **Denominator (√v̂_t + ε)**: adaptive scaling per parameter
- **ε**: prevents division by zero when gradients vanish

### Characteristics
- ✅ Combines benefits of momentum and adaptive learning rates
- ✅ Per-parameter learning rates (adapts to each parameter's gradient history)
- ✅ Robust to noisy gradients
- ✅ Works well with sparse gradients
- ✅ Requires minimal hyperparameter tuning
- ✅ Generally converges faster than SGD
- ⚠️ Can generalize slightly worse than SGD on some tasks
- ⚠️ More memory overhead (stores m_t and v_t for each parameter)

### Intuition

**Why β₁ = 0.9 and β₂ = 0.999?**
- β₁ needs shorter memory for **direction** (adapt quickly to changes)
- β₂ needs longer memory for **variance** (stable noise estimate)

**The adaptive learning rate:**
- Parameters with large/noisy gradients → √v̂_t is large → smaller effective learning rate
- Parameters with small/stable gradients → √v̂_t is small → larger effective learning rate

**Think of it as:**
- **m̂_t**: "Where should I go?" (signal)
- **√v̂_t**: "How confident am I?" (noise/uncertainty)
- High uncertainty → take small, cautious steps
- High confidence → take larger steps

---

## Quick Comparison Table

| Feature | SGD | SGD + Momentum | Adam |
|---------|-----|----------------|------|
| **Smoothing** | None | ✓ (via momentum) | ✓ (via first moment) |
| **Adaptive LR** | ✗ | ✗ | ✓ (per parameter) |
| **Memory** | None | m_t | m_t + v_t |
| **Hyperparams** | η | η, β | η, β₁, β₂, ε |
| **Tuning needed** | High | Medium | Low |
| **Convergence speed** | Slow | Medium | Fast |
| **Best for** | Simple problems | Most problems | Default choice |

---

## When to Use Each?

**SGD:**
- When you need simplicity
- When you have time to tune learning rate schedules
- Rarely used in practice without momentum

**SGD + Momentum:**
- Classic choice for computer vision
- When generalization is critical
- With good learning rate scheduling

**Adam:**
- **Default choice** for most deep learning tasks
- NLP, transformers, LLMs
- When you want fast convergence with minimal tuning
- When dealing with sparse gradients