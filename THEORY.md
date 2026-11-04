# Theory Notes

> **Status: Phase 1 complete; Phase 2 complete (local stability analysis);
> Phase 3 not started.** Claims are tagged as Derived / Numerically observed /
> Assumption / Open. The Phase 2 analysis is local at the symmetric point
> `W = 0, b = 0`; it is not a derivation of the Phase 1 basin boundary.

## Notation

For input `x`, parameters `theta = (W, b)`, and `K` classes:

```text
p_theta(y | x) = softmax(W x + b)
p_bar           = E_x[p_theta(. | x)]        # marginal distribution
H(p)              = -sum_y p(y) log p(y)     (nats)
```

The two adaptation objectives considered by the project are:

```text
Entropy minimisation:       L_ent  = E_x[H(p_theta(. | x))]
Information maximisation:   L_info = E_x[H(p_theta(. | x))] - lambda H(p_bar)
```

Throughout Phase 2 the empirical mean over `N` inputs is used for `E_x`.

## Exact identity (**Derived**)

The following identity is exact for the empirical input distribution used to
form `p_bar`:

```text
E_x[H(p_theta(. | x))] = H(p_bar) - I(X; Y_theta).
```

It follows directly from the definition of mutual information. The repository
unit-tests this identity against a direct computation on random logits. It is a
known decomposition, not a novel result of this project.

## Phase 1 numerical protocol

Phase 1 is a numerical experiment, not a derivation. The protocol was fixed in
`configs/theory_phase1.yaml` before the saved run:

- Balanced `K = 3` Gaussian mixture in `d = 3`, class means
  `separation * e_k` with `separation = 3.0`, identity covariance.
- 600 training and 1,000 evaluation samples per class. Train and evaluation
  splits are drawn independently.
- Linear-softmax model `softmax(W x + b)`. **Both `W` and `b` are adapted**, by
  full-batch SGD on mean predictive entropy at learning rate `0.03` for 2,500
  steps.
- Initial `W` is the class-prototype matrix (the first `K` coordinates of the
  identity); initial `b = delta * e_0`.
- `delta` is swept over `0.0, 0.05, ..., 3.60` (73 values) for three data seeds
  `[0, 1, 2]`.
- Collapse is declared only after `H(p_bar) <= 0.10` **and**
  top-class fraction `>= 0.95` hold for 20 consecutive evaluations (every 10
  steps), i.e. 200 adaptation steps.
- Labels are used only for diagnostics, never in the adaptation objective.
- Mean and standard deviation are computed across the three data seeds.

### Why both parameters are adapted

With `W` frozen, the balanced solution is not a stable non-collapsed attractor;
an exploratory pilot (not saved as a result) showed every trajectory eventually
collapsing. With `W` free, the model can either sharpen a class-separating
solution while keeping `p_bar` balanced, or shrink the useful weight directions
and let the bias dominate. The basin boundary therefore exists for the
both-adapted model and is the object Phase 2 must analyse.

## Phase 1 numerical results (**Numerically observed**)

Saved run:

```text
results/theory_phase1/20261104T173154.734687Z_theory_phase1_seed0_4f7a72f0/
```

Source model over three seeds: accuracy `0.968778 ± 0.004286`,
`H(p_bar) = 1.098604 ± 0.000007`.

| Initial bias | Collapse rate | Final `H(p_bar)` | Final top-class fraction | Final accuracy |
|---:|---:|---:|---:|---:|
| 3.10 | 0/3 | 1.09847 ± 0.00011 | 0.3391 ± 0.0024 | 0.9649 ± 0.0044 |
| 3.15 | 1/3 | 0.58319 ± 0.54388 | 0.6721 ± 0.3295 | 0.6442 ± 0.3153 |
| 3.20 | 2/3 | 0.22071 ± 0.35855 | 0.8929 ± 0.1855 | 0.4343 ± 0.1749 |
| 3.25 | 3/3 | 0.01323 ± 0.00073 | 1.0000 ± 0.0000 | 0.3333 ± 0.0000 |

The empirical 50% crossing is `3.175`, bracketed by the tested values
`[3.15, 3.20]`. This crossing is **descriptive**, not a derived threshold.

Illustrative trajectories:

- Bias `3.0`: no seed collapsed. Seed 0 ended at `H(p_bar) = 1.098535`,
  accuracy `0.9620`.
- Bias `3.5`: every seed collapsed at step `390`. Seed 0 ended at
  `H(p_bar) = 0.012007`, top-class fraction `1.0`, accuracy `1/3`.

**Open:** the run reports a finite-horizon boundary. It does not prove that
`H(p_bar)` reaches exactly zero asymptotically, and the boundary is specific to
this toy configuration, initialisation, optimiser, step size, and horizon.

---

## Phase 2: local stability analysis

### 2.1 Exact gradients (**Derived**)

Let `z_i = W x_i + b`, `p_i = softmax(z_i)`, and

```text
C_i = diag(p_i) - p_i p_i^T.
```

The matrix `C_i` is the softmax Jacobian and is positive semidefinite, with
`C_i 1 = 0`.

For `L_ent = (1/N) sum_i H(p_i)`:

```text
d L_ent / d z_i = -(1/N) C_i (log p_i + H(p_i) 1).
```

For `L_info = L_ent - lambda H(p_bar)` with `p_bar = (1/N) sum_i p_i`:

```text
d L_info / d z_i
    = -(1/N) C_i (log p_i + H(p_i) 1 - lambda log p_bar).
```

The parameter gradients follow from the chain rule:

```text
d L / d b = sum_i d L / d z_i
d L / d W = sum_i (d L / d z_i) x_i^T
```

Both logit gradients sum to zero over classes. This is required by softmax
shift invariance and is checked numerically.

### 2.2 Reduced `W = 0` bias flow (**Derived**)

If `W = 0`, every sample has the same logits `b`, so `p = softmax(b)` and
`p_bar = p`. The objective collapses to

```text
L_info = (1 - lambda) H(p).
```

The continuous-time gradient flow is

```text
db/dt = -dL_info/db
      = (1 - lambda) p * (b - (p . b) 1).
```

At `b = 0`, write `b = eps * u` with `u` in the sum-zero subspace. Linearising:

```text
d(eps u)/dt = ((1 - lambda) / K) eps u.
```

Therefore the uniform point is:

- **unstable for `lambda < 1`** — the bias spread grows and `H(p)` falls;
- **neutral for `lambda = 1`** — `L_info` is identically zero;
- **locally stable for `lambda > 1`** — the bias spread shrinks and
  `H(p)` returns to `log K`.

For discrete gradient descent with step size `eta`, the Jacobian eigenvalue on
the sum-zero subspace is

```text
1 - eta (lambda - 1) / K,
```

so the local stability condition is

```text
0 < eta < 2K / (lambda - 1)      (lambda > 1).
```

This is an exact statement about the reduced `W = 0` model. It is **not** the
full Phase 1 model, which also adapts `W`.

### 2.3 Symmetric point of the full model (**Derived**)

At `W = 0, b = 0`, every `p_i` equals `1/K`. Differentiating the objective with
respect to `b` gives

```text
d^2 L_info / db^2 = (lambda - 1) C,
C = diag(1/K) - (1/K^2) 1 1^T.
```

The non-zero eigenvalues of `C` are `1/K`, with multiplicity `K - 1`. Hence the
marginal-mode curvature is

```text
(lambda - 1) / K,
```

the same value as in the reduced model. The `b`-block result does not depend on
the number of samples or the feature dimension.

For the full `(W, b)` Hessian with **centred features**, a `W` perturbation
produces only sample-contrast logit changes. On those modes the
`lambda`-dependent contribution cancels, leaving the negative semidefinite
entropy curvature. Consequently the only possible positive eigenvalue is the
marginal mode `(lambda - 1) / K`.

The symmetric point is therefore:

- unstable in the marginal direction for `lambda < 1`;
- neutral in the marginal direction for `lambda = 1`;
- locally stable in the marginal direction for `lambda > 1`.

It remains a **saddle**, not a minimum, because per-sample sharpening
directions still have negative curvature. This is the intended behaviour of the
information objective: it penalises marginal collapse, not confidence itself.
The discrete step bound for the marginal mode is again
`eta < 2K / (lambda - 1)` for `lambda > 1`.

### 2.4 Scope and limitations (**Open / Assumption**)

- The analysis is local at `W = 0, b = 0`. Phase 1 initialises `W = I` and
  `b = delta e_0`; therefore the measured `3.175` crossing is **not** predicted
  by `lambda = 1` or by this local Hessian.
- The full-model symmetric point is a saddle. The analysis does not prove
  global convergence, nor does it characterise the Phase 1 basin boundary.
- The reduced `W = 0` model is a mechanism check, not the full experiment.
  Adapting `W` is what creates the Phase 1 balanced non-collapsed solution.
- No claim about deep-network or CIFAR-10-C behaviour is made here. That is
  Phase 3.

## Phase 2 numerical verification (**Numerically observed**)

Saved run:

```text
results/theory_phase2/20261104T184229.827471Z_theory_phase2_seed0_6b788892/
```

The check used `K = 3`, `d = 2`, `N = 6`, float64, and three verification
seeds `[0, 1, 2]`. The analytic gradients and Hessians were compared with
autograd; the reduced flow was simulated directly from its exact update.

### Gradient and Hessian checks

| Check | Max absolute error | Tolerance |
|---|---:|---:|
| Entropy / information logit gradients | `5.551e-16` | `1e-9` |
| Bias-block Hessian vs `(lambda - 1) C` | `2.220e-16` | `1e-9` |
| Bias-block eigenvalues vs `(lambda - 1)/K` | `4.441e-16` | `1e-9` |
| Centred full-Hessian maximum vs `max(0, (lambda-1)/K)` | `8.882e-16` | `1e-9` |

All checks passed. The maximum errors are over the three verification seeds.

### Reduced `W = 0` flow

Initial bias spread `0.014142`; initial `H(p) = 1.098579`; `log 3 = 1.098612`.

| `lambda` | Final `H(p)` | Final bias spread | Behaviour |
|---:|---:|---:|---|
| 0.0 | 0.004725 | 6.756954 | collapse |
| 0.5 | 0.010671 | 6.006260 | collapse |
| 0.9 | 0.184525 | 3.229866 | collapse |
| 1.0 | 1.098579 | 0.014142 | neutral |
| 1.1 | 1.098612 | 0.000018 | balanced |
| 2.0 | 1.098612 | < `1e-6` | balanced |

The simulated flow crosses from collapse to balance at `lambda = 1`, as derived.
The `lambda = 0.9` run is a slow collapse in the 2,000-step horizon, not an
instantaneous one.

### Discrete step-size check

At `lambda = 2`, `K = 3`, the derived bound is `eta < 6`. The one-step factor is
`1 - eta/3`; the table reports its magnitude for comparison with the observed
spread ratio.

| `eta` | Predicted factor | Observed factor | Stable? |
|---:|---:|---:|---|
| 1.0 | 0.666667 | 0.666667 | yes |
| 3.0 | 0.000000 | 0.000000 | yes |
| 5.8 | 0.933333 | 0.933334 | yes |
| 6.2 | 1.066667 | 1.066668 | no |
| 7.0 | 1.333333 | 1.333334 | no |

The largest factor error is `1.167e-6`, consistent with the small finite
perturbation used in the one-step check.

### Figures

```text
results/theory_phase2/20261104T184229.827471Z_theory_phase2_seed0_6b788892/figures/stability_spectrum.png
results/theory_phase2/20261104T184229.827471Z_theory_phase2_seed0_6b788892/figures/reduced_dynamics.png
```

## Correction to the earlier draft (**Open**)

An earlier bias-only draft in this file is superseded. It used
`dH/dz = p - e_y`, which is the cross-entropy gradient for a one-hot label, not
the entropy gradient; it also violated softmax shift invariance and asserted a
step-size bound without deriving it from the implemented update. The correct
gradient and sum-zero flow are given in Sections 2.1 and 2.2. The incorrect
draft is not used as evidence.

## Tagging conventions

Every theoretical statement added later will be tagged as one of:

- **Derived** — follows from the stated model and assumptions.
- **Numerically observed** — measured in a saved run, with the config and log
  referenced.
- **Assumption** — required for the argument but not proved here.
- **Open** — unresolved and explicitly flagged.

No result will be promoted from "numerically observed" to "derived" without a
proof, and no citation will be added until its bibliographic details have been
checked.
