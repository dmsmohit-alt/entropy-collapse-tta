# Theoretical Analysis: Test-Time Entropy Minimisation & Dynamical Collapse

This document provides the mathematical foundation for analyzing when and why test-time entropy minimisation collapses, and how marginal-entropy regularisation restores asymptotic stability.

---

## 1. System Formalism & Notation

### Model Definition [ASSUMPTION]
Consider a tractable linear-softmax classification model with $K$ classes:
$$f_{\theta}(x) = z(x) = W x + b, \quad x \in \mathbb{R}^d, \; W \in \mathbb{R}^{K \times d}, \; b \in \mathbb{R}^K$$
The conditional class probabilities are given by the softmax function:
$$p_k(x) \equiv p(y=k \mid x; \theta) = \frac{\exp(z_k(x))}{\sum_{j=1}^K \exp(z_j(x))}$$

### Data Distribution [ASSUMPTION]
The test distribution is modeled as a mixture of $K$ isotropic Gaussians with equal prior probability $\pi_k = \frac{1}{K}$:
$$P(x) = \frac{1}{K} \sum_{k=1}^K \mathcal{N}(x; \mu_k, \sigma^2 I_d)$$
where class centroids $\mu_k$ are symmetrically distributed with separation $\|\mu_i - \mu_j\| > 0$.

### Objectives
- **Standard TENT Objective**:
  $$\mathcal{L}_{\text{TENT}}(\theta) = \mathbb{E}_{x \sim P(x)} \left[ H(p(\cdot \mid x)) \right] = - \mathbb{E}_x \left[ \sum_{k=1}^K p_k(x) \ln p_k(x) \right]$$
- **Information-Maximisation (IM) Objective**:
  $$\mathcal{L}_{\text{IM}}(\theta) = \mathbb{E}_{x \sim P(x)} \left[ H(p(\cdot \mid x)) \right] - \lambda H(\bar{p})$$
  where $\bar{p} = \mathbb{E}_{x}[p(\cdot \mid x)] \in \Delta^{K-1}$ is the marginal class prediction distribution.

---

## 2. Gradient Dynamics on Logits and Bias

### Gradient of Conditional Entropy w.r.t. Logits [DERIVED]
Let $H(p(x)) = -\sum_{j=1}^K p_j(x) \ln p_j(x)$.
Using the softmax Jacobian:
$$\frac{\partial p_j}{\partial z_k} = p_k (\delta_{jk} - p_j)$$
Differentiating $H(p)$ with respect to logit $z_k$:
$$\begin{aligned}
\frac{\partial H(p)}{\partial z_k} &= - \sum_{j=1}^K \left[ (1 + \ln p_j) \frac{\partial p_j}{\partial z_k} \right] \\
&= - \sum_{j=1}^K (1 + \ln p_j) p_k (\delta_{jk} - p_j) \\
&= - p_k (1 + \ln p_k) + p_k \sum_{j=1}^K p_j (1 + \ln p_j) \\
&= - p_k - p_k \ln p_k + p_k (1 - H(p)) \\
&= - p_k (\ln p_k + H(p))
\end{aligned}$$
$$\boxed{\frac{\partial H(p)}{\partial z_k} = - p_k(x) \left( \ln p_k(x) + H(p(x)) \right)}$$

### Continuous-Time Bias Gradient Flow $\dot{b}_k$ [DERIVED]
When adapting the bias $b$ alone under gradient flow $\dot{b} = - \nabla_b \mathcal{L}_{\text{TENT}}$:
$$\dot{b}_k = - \frac{\partial \mathcal{L}_{\text{TENT}}}{\partial b_k} = - \mathbb{E}_x \left[ \frac{\partial H(p(x))}{\partial z_k} \right]$$
Substituting the logit gradient:
$$\boxed{\dot{b}_k = \mathbb{E}_x \left[ p_k(x) \left( \ln p_k(x) + H(p(x)) \right) \right]}$$

---

## 3. The "Rich-Get-Richer" Instability Analysis

### Perturbation Around Uniform State [DERIVED]
Consider the balanced state where all classes are symmetric: $b_1 = b_2 = \dots = b_K = 0$.
Here, $\mathbb{E}[p_k] = \frac{1}{K}$ and $\ln p_k \approx -\ln K$, while $H(p) \approx \ln K$.
Hence, at the perfectly balanced uniform point:
$$\ln p_k + H(p) \approx -\ln K + \ln K = 0 \implies \dot{b}_k \approx 0$$
Thus, $b^* = 0$ is a critical point (equilibrium).

However, consider an infinitesimal perturbation $\delta b_1 > 0$ favoring class 1:
- For inputs $x$ where class 1 is already competitive, $p_1(x)$ increases above $\frac{1}{K}$.
- The term $\ln p_1(x)$ increases (becomes less negative) faster than the global entropy $H(p(x))$ decreases.
- Consequently, the product $p_1(x) (\ln p_1(x) + H(p(x)))$ becomes strictly positive in the neighbourhood of class 1's decision boundary.
- This produces $\dot{b}_1 > 0$, driving $b_1$ even higher!

### Basin Boundary and Threshold Phenomenon [NUMERICAL]
- If initial perturbation $|\delta b| < \epsilon_{\text{threshold}}$, the restoring forces from the input data geometry may maintain balanced predictions.
- If $|\delta b| \ge \epsilon_{\text{threshold}}$, the rich-get-richer positive feedback drives $b_1 \to +\infty$ and $b_{k \neq 1} \to -\infty$, producing complete collapse $H(\bar{p}) \to 0$.
*(The exact numerical boundary will be mapped systematically in Phase 1).*

---

## 4. Information-Maximisation Fixed Point & Stability

### Marginal Entropy Gradient [DERIVED]
Recall the marginal distribution $\bar{p}_k = \mathbb{E}_x [p_k(x)]$.
The entropy of the marginal is:
$$H(\bar{p}) = - \sum_{k=1}^K \bar{p}_k \ln \bar{p}_k$$
Its derivative with respect to bias $b_k$ via the chain rule:
$$\frac{\partial H(\bar{p})}{\partial b_k} = \sum_{j=1}^K \frac{\partial H(\bar{p})}{\partial \bar{p}_j} \frac{\partial \bar{p}_j}{\partial b_k} = - \sum_{j=1}^K (1 + \ln \bar{p}_j) \mathbb{E}_x \left[ p_k(x)(\delta_{jk} - p_j(x)) \right]$$
At the uniform marginal fixed point $\bar{p}_j = \frac{1}{K}$:
$1 + \ln \bar{p}_j = 1 - \ln K$ is constant for all $j$.
Since $\sum_{j=1}^K (\delta_{jk} - p_j(x)) = 1 - \sum_j p_j(x) = 0$, the constant term vanishes.
When class 1 is perturbed positively ($\bar{p}_1 > \frac{1}{K}$):
$$\frac{\partial H(\bar{p})}{\partial b_1} < 0 \implies - \lambda \frac{\partial (-H(\bar{p}))}{\partial b_1} < 0$$
The marginal term acts as an **opposing restorative force** pulling $b_1$ back toward the origin.

### Linearized Stability and Step Size Bound $\eta^*$ [DERIVED]
Under the regularised dynamics:
$$\dot{b} = - \nabla_b \mathcal{L}_{\text{IM}}(b)$$
Linearising around the uniform fixed point $b^* = 0$:
$$\delta \dot{b} = - J_{\text{IM}} \delta b$$
where $J_{\text{IM}} = \nabla_b^2 \mathcal{L}_{\text{IM}}(b^*)$ is the Hessian at the equilibrium.
For continuous gradient flow, asymptotic stability requires that all eigenvalues of $-J_{\text{IM}}$ have negative real parts:
$$\operatorname{Re}(\lambda_i(-J_{\text{IM}})) < 0 \iff \operatorname{Re}(\lambda_i(J_{\text{IM}})) > 0$$

For discrete gradient descent with step size $\eta$:
$$b^{(t+1)} = b^{(t)} - \eta \nabla_b \mathcal{L}_{\text{IM}}(b^{(t)}) \implies \delta b^{(t+1)} = (I - \eta J_{\text{IM}}) \delta b^{(t)}$$
Stability requires spectral radius $\rho(I - \eta J_{\text{IM}}) < 1$:
$$|1 - \eta \lambda_{\max}(J_{\text{IM}})| < 1 \iff \boxed{\eta < \eta^* = \frac{2}{\lambda_{\max}(J_{\text{IM}})}}$$
If $\eta > \eta^*$, updates overshoot and induce numerical divergence or oscillatory collapse even with the regulariser.
*(Analytical and numerical Jacobian eigenvalue validation will be performed in Phase 2).*
