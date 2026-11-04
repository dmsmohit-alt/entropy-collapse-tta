# Phase 1 Notes: Numerical Dynamics & Basin Boundary Observations

*Status: Empirical simulation records from actual execution runs on synthetic Gaussian mixture linear-softmax models.*
*Rule: All claims here are labeled as numerical observations, not proven theorems.*

---

## 1. Experimental Setup & Definitions

### Model & Data Parameters
- **Architecture**: Linear-softmax classifier $p(y=k \mid x) = \text{softmax}(W x + b)$, with $K=3$ classes, dimension $d=2$.
- **Data Generator**: Balanced 3-class Gaussian mixture, centroids placed regularly on a circle of radius $R=1.0$, with isotropic covariance $\sigma^2 I$ ($\sigma=1.0$).
- **Test-Time Adaptation**: Full-batch gradient descent on TENT objective $\min_b \mathbb{E}_{x}[H(p(\cdot \mid x))]$, adapting **bias only** ($W$ fixed, initialized from centroid directions with scale $1.0$).
- **Step Sizes Evaluated**: $\eta \in \{0.01, 0.05, 0.20\}$.
- **Initial Bias Perturbation**: $b_0(0) = \delta$, $b_{1}(0) = b_{2}(0) = 0$.
- **Sample Statistics**: All reported standard deviations use sample standard deviation (`ddof=1`).

### Definitions of Collapse Evaluated
1. **Def A (`top_frac_0.9`)**: Top-class prediction fraction $\ge 0.90$.
2. **Def B (`h_pbar_0.3`)**: Marginal entropy $H(\bar{p}) < 0.30 \times \ln(K)$ ($0.3296$ nats for $K=3$).
3. **Def C (`h_pbar_0.1`)**: Marginal entropy $H(\bar{p}) < 0.10 \times \ln(K)$ ($0.1098$ nats for $K=3$).
4. **Standard (Both)**: `top_frac >= 0.90` **AND** `marginal_entropy < 0.30 * ln(K)`.

---

## 2. Robustness & Dynamical Sensitivity Findings

### Investigation 1: Horizon Sensitivity & The Disappearance of the Basin [NUMERICAL OBSERVATION, NOT PROVEN]
We swept optimization horizon $T \in \{500, 2000, 10000, 50000\}$ steps at fixed $\eta = 0.05$ across 5 random seeds (42--46):

| Optimization Horizon ($T$) | Flow-Time $T_{\text{flow}} = \eta \cdot T$ | Critical Bias $\delta^*(T)$ (Mean $\pm$ Std, `ddof=1`) | Per-Seed Values (Seeds 42 to 46) | State at $\delta = 0$ |
| :---: | :---: | :---: | :---: | :---: |
| **500 steps** | $25.0$ | **$0.8340 \pm 0.0493$** | [0.7637, 0.8418, 0.8613, 0.8926, 0.8105] | Non-collapsed ($H=1.098$ nats) |
| **2,000 steps** | $100.0$ | **$0.0020 \pm 0.0000$** | [0.0020, 0.0020, 0.0020, 0.0020, 0.0020] | **Collapsed** (at step 1400) |
| **10,000 steps** | $500.0$ | **$0.0020 \pm 0.0000$** | [0.0020, 0.0020, 0.0020, 0.0020, 0.0020] | **Collapsed** ($H=0.0026$ nats) |
| **50,000 steps** | $2500.0$ | **$0.0020 \pm 0.0000$** | [0.0020, 0.0020, 0.0020, 0.0020, 0.0020] | **Collapsed** ($H \approx 0.0$ nats) |

*(Note: $0.0020$ is the lower numerical bisection search boundary corresponding to $\delta = 0$ within tolerance $\text{tol}=0.005$).*

#### Honest Scientific Assessment:
- **Does $\delta^*$ drift toward 0? YES, strictly.**
- **Dynamical Meaning**: Under pure entropy minimisation, the uniform marginal fixed point $b^* = 0$ has **no true asymptotic basin of attraction** in the presence of distribution shift and finite-sample noise.
- What appeared to be a stable basin boundary at $T \le 500$ is a **finite-time metastable regime**:
  The collapse time scales inversely with perturbation size:
  $$T_{\text{collapse}}(\delta) \sim C_0 + \frac{1}{\lambda_{\max}} \ln\left(\frac{1}{\delta}\right)$$
  When $\delta$ is small, the exponential growth of the perturbation takes many steps to cross the macroscopic threshold $\text{top\_frac} > 0.90$. On short test sequences ($T \le 500$), the system appears stable. However, on long streams ($T \ge 2000$), even infinitesimal perturbations ($\delta \to 0$) or finite-sample stochastic variance spontaneously amplify into total single-class collapse.

---

### Investigation 2: Flow-Time Invariance ($\eta \in \{0.01, 0.05, 0.20\}$) [NUMERICAL OBSERVATION, NOT PROVEN]
We tested whether the collapse boundary depends on the discrete learning rate $\eta$ or solely on continuous gradient flow-time $T_{\text{flow}} = \eta \cdot \text{num\_steps}$:

| Flow-Time $T_{\text{flow}}$ | Step Size $\eta$ | Required Steps ($N$) | Critical Bias $\delta^*$ (Mean $\pm$ Std, `ddof=1`) |
| :---: | :---: | :---: | :---: |
| **$T_{\text{flow}} = 25.0$** | $0.01$ | 2,500 | **$0.8324 \pm 0.0487$** |
| **$T_{\text{flow}} = 25.0$** | $0.05$ | 500 | **$0.8340 \pm 0.0493$** |
| **$T_{\text{flow}} = 25.0$** | $0.20$ | 125 | **$0.8387 \pm 0.0505$** |
| **$T_{\text{flow}} = 100.0$** | $0.01$ | 10,000 | **$0.0020 \pm 0.0000$** |
| **$T_{\text{flow}} = 100.0$** | $0.05$ | 2,000 | **$0.0020 \pm 0.0000$** |
| **$T_{\text{flow}} = 100.0$** | $0.20$ | 500 | **$0.0020 \pm 0.0000$** |

#### Conclusion:
Across a 20-fold variation in step size ($\eta = 0.01$ vs $0.20$), the collapse boundary is invariant at matching continuous flow-times ($0.8324 \approx 0.8340 \approx 0.8387$). This confirms that collapse is fundamentally governed by continuous gradient flow dynamics $\dot{b} = -\nabla_b \mathcal{L}$, and is not an artifact of discrete step-size discretization.

---

### Investigation 3: Convergence & Gradient Norm Dynamics $\|\nabla_b \mathcal{L}\|_2$ [NUMERICAL OBSERVATION, NOT PROVEN]
We tracked the exact gradient norm $\|\nabla_b \mathcal{L}\|_2(t)$ over a 10,000-step horizon for initial biases $\delta \in [0.0, 0.3, 0.6, 0.9, 1.2, 1.8]$:

| Initial Bias $\delta$ | Collapsed? | Collapse Step ($t_{\text{coll}}$) | Final $H(\bar{p})$ | Final Top Fraction | Final $\|\nabla_b \mathcal{L}\|_2$ | Dynamical Outcome |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0.0** | **Yes** | 1,400 | 0.0026 nats | 1.0000 | $2.32 \times 10^{-3}$ | Spontaneous collapse; asymptotically stationary |
| **0.3** | **Yes** | 750 | 0.0024 nats | 1.0000 | $2.15 \times 10^{-3}$ | Accelerated collapse; asymptotically stationary |
| **0.6** | **Yes** | 600 | 0.0024 nats | 1.0000 | $2.11 \times 10^{-3}$ | Accelerated collapse; asymptotically stationary |
| **0.9** | **Yes** | 500 | 0.0024 nats | 1.0000 | $2.09 \times 10^{-3}$ | Accelerated collapse; asymptotically stationary |
| **1.2** | **Yes** | 400 | 0.0023 nats | 1.0000 | $2.08 \times 10^{-3}$ | Rapid collapse; asymptotically stationary |
| **1.8** | **Yes** | 300 | 0.0023 nats | 1.0000 | $2.06 \times 10^{-3}$ | Rapid collapse; asymptotically stationary |

#### Stationary Point Characterization:
- As $b_0 \to +\infty$ and $b_{1,2} \to -\infty$, $p_0(x) \to 1.0$.
- Because $\lim_{p \to 1} p(\ln p + H) = 0$, the gradient norm $\|\nabla_b \mathcal{L}\|_2$ decays monotonically toward 0 (reaching $2 \times 10^{-3}$ by step 10,000).
- **The collapsed state is an asymptotic fixed point of the dynamics**, representing the global minimum of the unconstrained conditional entropy loss ($\mathcal{L} = 0$).

---

### Investigation 4: Collapse Definition Robustness [NUMERICAL OBSERVATION, NOT PROVEN]
We compared critical $\delta^*$ across definitions at finite horizon ($T = 500$) and long horizon ($T = 2000$):

| Collapse Definition | Formula / Cutoff | Finite Horizon ($T=500$) $\delta^*$ (Mean $\pm$ Std) | Long Horizon ($T=2000$) $\delta^*$ (Mean $\pm$ Std) |
| :--- | :--- | :---: | :---: |
| **Def A** | $\text{top\_frac} \ge 0.90$ | **$0.4816 \pm 0.0544$** | **$0.0020 \pm 0.0000$** |
| **Def B** | $H(\bar{p}) < 0.30 \ln K$ | **$0.8340 \pm 0.0493$** | **$0.0020 \pm 0.0000$** |
| **Standard** | Def A **AND** Def B | **$0.8340 \pm 0.0493$** | **$0.0020 \pm 0.0000$** |
| **Def C** | $H(\bar{p}) < 0.10 \ln K$ | **$2.7012 \pm 0.1594$** | **$0.0020 \pm 0.0000$** |

#### Hierarchy of Collapse:
- In the transient phase ($T = 500$), the metrics reflect sequential stages of collapse:
  1. Prediction skew appears first: $\text{top\_frac} \ge 0.90$ is triggered at $\delta^* = 0.4816$.
  2. Loss of class diversity follows: $H(\bar{p}) < 0.3 \ln K$ is triggered at $\delta^* = 0.8340$.
  3. Severe logit saturation occurs last: $H(\bar{p}) < 0.1 \ln K$ requires $\delta^* = 2.7012$.
- At long horizons ($T \ge 2000$), all definitions converge identically to $\delta^* \to 0$.

---

## 3. Neutral Terminology Clarification
In earlier working drafts, the regime $\delta \in [0.6, 0.9]$ at $T=400$ was colloquially termed "sub-threshold margin compression". To maintain scientific neutrality, this phenomenon is formally documented as:
> **Intermediate Non-Collapsed State (Transient Class Skew)**: A state in which class predictions are skewed toward class 0 but have not yet crossed the formal collapse thresholds within the designated finite evaluation horizon.

---

## 4. Artifact Registry

- **Logs & Summaries**:
  - `results/logs/phase1_robustness.log`
  - `results/logs/phase1_robustness_summary.json`
  - `results/logs/phase1_theory.log`
  - `results/logs/phase1_summary.json`
- **Figures Generated**:
  - `results/figures/phase1_horizon_sensitivity.png`
  - `results/figures/phase1_flow_time_check.png`
  - `results/figures/phase1_gradient_norm_convergence.png`
  - `results/figures/phase1_definition_robustness.png`
  - `results/figures/phase1_h_pbar_trajectories.png`
  - `results/figures/phase1_bias_dynamics.png`
  - `results/figures/phase1_basin_1d.png`
  - `results/figures/phase1_basin_phase_diagram_2d.png`
