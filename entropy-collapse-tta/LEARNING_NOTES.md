# Learning Notes: Test-Time Entropy Minimisation & Collapse Dynamics

*A research guide and learning companion for understanding Test-Time Adaptation (TTA), collapse dynamics, and information-theoretic stabilization.*

---

## 1. What is Test-Time Adaptation (TTA)?
In classic supervised learning, a deep neural network is trained on a source dataset $(X_{train}, Y_{train})$ and frozen during deployment. However, real-world deployment data inevitably experiences **distribution shifts**:
- Weather variations (fog, rain, snow)
- Sensor degradation (lens blur, noise, compression artifacts)
- Domain shifts (different hospitals, lighting, cameras)

When shifted test samples $x \sim P_{target}(x)$ arrive, model accuracy typically degrades drastically. 

**Test-Time Adaptation (TTA)** allows the model to update its parameters $\theta$ *on-the-fly* using only the incoming stream of unlabeled test inputs $\{x_t\}_{t=1}^T$, without requiring any ground-truth labels $y_t$ or access to the original training data.

---

## 2. The Core Mechanism of TENT
Wang et al. (ICLR 2021) introduced **TENT** (Test-Time Entropy Minimisation).
The intuitive premise:
- When a model is accurate, its softmax predictions are sharp and confident (low Shannon entropy).
- When a model is confused by corrupted inputs, its predictions are diffuse and uncertain (high Shannon entropy).

Therefore, TENT minimizes the sample-wise prediction entropy:
$$\min_{\theta} \mathcal{L}_{\text{TENT}} = \mathbb{E}_{x \sim P_{\text{test}}} \left[ H(p_{\theta}(\cdot \mid x)) \right]$$
where:
$$H(p_{\theta}(\cdot \mid x)) = - \sum_{k=1}^K p_{\theta}(y=k \mid x) \log p_{\theta}(y=k \mid x)$$

To avoid destroying learned representation filters, TENT restricts the gradient updates strictly to the affine scale $\gamma$ and shift $\beta$ parameters of the **Batch Normalisation (BN)** layers.

---

## 3. Why Does Pure Entropy Minimisation Collapse?

### The Mathematical Trap: Trivial Minima
The minimum possible value of Shannon entropy is $0$ nats. Consider what happens if a model outputs a constant logit vector where class $c$ is always larger than all other classes, regardless of the input $x$:
$$p_{\theta}(y=c \mid x) = 1.0, \quad p_{\theta}(y \neq c \mid x) = 0.0 \quad \forall x$$

Evaluating the conditional entropy:
$$H(p_{\theta}(\cdot \mid x)) = - (1.0 \log 1.0 + 0) = 0$$

Thus, **a degenerate model that blindly predicts class $c$ for every single test image achieves the global theoretical minimum ($\mathcal{L}_{\text{TENT}} = 0$)!**

### The Dynamical Cause: The "Rich-Get-Richer" Gradient Drift
Why doesn't TENT collapse immediately on batch 1, but collapses after many batches in continual streams?
1. Suppose on the first few corrupted batches, class $k^*$ by pure chance has slightly higher average probability than other classes (e.g. $25\%$ vs $10\%$).
2. The gradient of conditional entropy with respect to the output logits $z$ pulls confident logits up and pushes uncertain logits down.
3. This creates a positive feedback loop: the bias $b_{k^*}$ (or BN shift parameter $\beta_{k^*}$) grows larger.
4. On the next batch, even more inputs are classified as class $k^*$.
5. Within dozens of batches, class $k^*$ captures $>95\%$ of all predictions. The model accuracy drops to chance ($10\%$ on CIFAR-10), and once in this state, gradient descent cannot escape.

---

## 4. The Information-Theoretic Decomposition

We can dissect this collapse using Shannon Information Theory.
Let:
- $p_{\theta}(y \mid x)$ be the predicted conditional class probability for sample $x$.
- $\bar{p} = \mathbb{E}_{x}[p_{\theta}(y \mid x)]$ be the **marginal prediction distribution** over the test stream.

Recall the fundamental identity relating conditional entropy, marginal entropy, and mutual information:
$$I(X; Y_{\theta}) = H(\bar{p}) - \mathbb{E}_x \left[ H(p_{\theta}(\cdot \mid x)) \right]$$

Rearranging for the TENT objective:
$$\mathbb{E}_x \left[ H(p_{\theta}(\cdot \mid x)) \right] = \underbrace{H(\bar{p})}_{\text{Diversity of predictions}} - \underbrace{I(X; Y_{\theta})}_{\text{Dependence between input and output}}$$

This equation makes the problem crystal clear:
To make the TENT loss $\mathbb{E}[H(p)]$ small, the optimizer has **two independent options**:
1. **Increase $I(X; Y_{\theta})$** (Good): Make predictions genuinely informative of the input $x$.
2. **Decrease $H(\bar{p})$** (Bad): Collapse the marginal distribution so that only one class is ever predicted, driving $H(\bar{p}) \to 0$!

Standard TENT lacks any term constraining $H(\bar{p})$, so it frequently falls down the second path.

---

## 5. The Fix: Information Maximisation (IM)

To eliminate the degenerate collapse mode, we add a marginal entropy penalty:
$$\min_{\theta} \mathcal{L}_{\text{IM}} = \mathbb{E}_x \left[ H(p_{\theta}(\cdot \mid x)) \right] - \lambda H(\bar{p})$$

When $\lambda = 1.0$:
$$\mathcal{L}_{\text{IM}} = - I(X; Y_{\theta})$$
Minimising $\mathcal{L}_{\text{IM}}$ directly maximises the Mutual Information between inputs and outputs:
- $\mathbb{E}[H(p)]$ pushes individual predictions to be sharp.
- $-\lambda H(\bar{p})$ penalises any deviation from a balanced, uniform marginal class distribution, creating a restorative force that prevents the single-class runaway feedback loop.

---

## 6. Numerical Implementation: Avoiding NaNs

In theoretical formulas, $\lim_{p \to 0} p \log p = 0$. However, in standard IEEE floating-point arithmetic on GPUs:
- When a logit is very negative (e.g. $-500$), `torch.exp(logit)` evaluates to `0.0`.
- `torch.log(0.0)` evaluates to `-inf`.
- Multiplying `0.0 * -inf` produces `NaN`!

### The Solution
In `src/metrics/entropy.py`:
```python
log_probs = F.log_softmax(logits, dim=-1)
probs = torch.exp(log_probs)
# Guard limit: if probs == 0.0, replace with 0.0 directly
plogp = torch.where(probs > 0.0, probs * log_probs, torch.zeros_like(probs))
entropy = -torch.sum(plogp, dim=-1)
```
This guarantees numerical stability without gradient distortion even under extreme logits.
