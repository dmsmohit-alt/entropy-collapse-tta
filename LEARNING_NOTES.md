# Learning Notes

These notes explain the design choices in the repository. They are written for
a B.Tech CSE reader who knows basic probability and PyTorch but may not have
seen test-time adaptation before.

## Phase 0 — what was built and why

### 1. Entropy in nats

Shannon entropy is

```text
H(p) = -sum_k p_k log p_k.
```

If the logarithm is natural, the unit is **nats**. Using nats consistently
avoids conversion mistakes when comparing analytic derivatives with PyTorch
values. The code does not mix bits and nats.

The implementation uses `torch.log_softmax` and `torch.special.xlogy`:

- `log_softmax` avoids computing `log(0)` for extremely confident logits.
- `xlogy(0, 0)` is defined as zero, which is the correct limiting value in
  entropy.

### 2. Per-sample entropy versus marginal entropy

For a batch of predictions:

```text
mean predictive entropy = (1/N) sum_i H(p_i)
marginal distribution    = p_bar = (1/N) sum_i p_i
marginal entropy         = H(p_bar)
```

These answer different questions:

- Low mean predictive entropy means each individual prediction is confident.
- Low `H(p_bar)` means the **average** prediction is concentrated on one class.
  In the extreme, it means every image is effectively predicted as the same
  class.

A model can be confidently wrong on every input and still have high `H(p_bar)`
if its mistakes are spread across classes. Conversely, a model can have
moderately confident individual predictions but collapse its marginal.

### 3. Why the identity matters

The exact decomposition is

```text
mean predictive entropy = H(p_bar) - I(X; Y_theta).
```

Therefore minimising only the mean predictive entropy does not explicitly
protect `H(p_bar)`. If the mutual-information term also falls, the marginal can
become one-hot. The planned stabilised loss is

```text
L = mean predictive entropy - lambda * H(p_bar).
```

Gradient descent on this loss rewards a high-entropy marginal while still
encouraging confident predictions. This is the information-maximisation idea
used by SHOT-style methods. Phase 2 must establish the local stability
condition; that claim is not being made yet.

### 4. Why a toy linear-softmax model

A deep network has millions of parameters, BatchNorm statistics, and
non-stationary activations. A linear-softmax model with a Gaussian mixture has
few enough parameters to write the gradient flow and inspect fixed points. It
is not expected to reproduce every detail of a deep network. Its role is to
make a mechanism visible and falsifiable.

### 5. Config system

Every experiment reads YAML. Child configs can inherit from `configs/base.yaml`
using `_base_`. Dotted overrides such as `train.batch_size=32` are parsed as
YAML scalars, so `false` becomes a boolean rather than the string `"false"`.

The fully resolved config is saved in each run directory. A SHA-256 hash of the
canonical JSON representation is included in the directory name and metadata.
This makes accidental config changes visible even if two runs have similar
names.

### 6. Seeding

One integer seed controls:

- Python's `random`;
- NumPy's global RNG;
- PyTorch's CPU and (when available) CUDA RNGs;
- deterministic kernel flags when requested.

`PYTHONHASHSEED` only affects child processes because hash randomisation is
fixed when Python starts. This limitation is documented in the code rather
than hidden.

### 7. Run metadata and JSONL logs

Each run writes:

- the resolved config;
- a metadata JSON file with versions and platform;
- append-only JSONL metrics.

JSONL is used because a long adaptation stream may be interrupted. A completed
line remains readable even if the process dies. Non-finite values are rejected
so that a silent NaN cannot contaminate a later aggregate.

### 8. Tests

The Phase 0 tests check:

- uniform logits give `log(K)`;
- confident but balanced predictions give `H(p_bar) = log(2)` while the mean
  predictive entropy is near zero;
- the entropy identity matches a direct mutual-information calculation;
- shift invariance and zero-probability handling;
- finite entropy gradients;
- config merging, overrides, cycle detection, and hashing;
- reproducible RNG streams;
- JSONL round-tripping and NaN rejection;
- run metadata and saved configs.

These are software and numerical checks, not evidence for the research claim.

## Phase 1 — toy-model experiment

### Modelling choice: adapt both `W` and `b`

The toy classifier is `softmax(W x + b)`. Phase 1 adapts **both** `W` and `b`
by full-batch gradient descent. This is deliberate. If `W` is frozen, the
balanced solution is not a stable non-collapsed attractor, so every run
eventually collapses regardless of the initial bias. With `W` free, the model
can either:

- sharpen a class-separating solution while keeping `p_bar` balanced, or
- shrink the useful weight directions and let the bias dominate, collapsing
  all predictions to one class.

The initial weight is the class-prototype matrix. The initial bias is
`delta * e_0`; `delta` is the parameter scanned. Labels are used only for
diagnostics, never in the adaptation objective.

### Data

Balanced `K=3` Gaussian mixture in `d=3`:

```text
class means: separation * e_k
covariance:  I
```

Train and evaluation samples are generated independently. Adaptation uses
unlabeled training samples only; all reported entropy, concentration, and
accuracy values use the held-out split.

### Collapse definition

Collapse is declared only when both conditions hold for `persistence`
consecutive evaluations:

```text
H(p_bar) <= marginal_entropy_threshold
top_class_fraction >= top_class_fraction_threshold
```

The same detector is used for the real-network track. The configured Phase 1
values are in `configs/theory_phase1.yaml`. A single high-confidence batch is
not called collapse.

### Basin scan

For each data seed, the initial bias is swept over a fixed grid and the full
entropy-minimisation trajectory is run. The scan records final marginal
entropy, final top-class fraction, final accuracy, and time-to-collapse.
Aggregation across seeds reports mean and standard deviation. The empirical
50% crossing is descriptive; it is not a derived threshold.

### Why a finite horizon

The simulation reports a finite-horizon boundary. It does not prove that
`H(p_bar)` reaches exactly zero asymptotically. The trajectory plot and the
finite collapse time are reported as numerical observations; Phase 2 handles
the analytic dynamics.

### What the Phase 1 run actually shows

From the saved run
`results/theory_phase1/20261104T173154.734687Z_theory_phase1_seed0_4f7a72f0/`:

- The unadapted source model is strong: accuracy `0.968778 ± 0.004286` and
  `H(p_bar) = 1.098604 ± 0.000007`.
- No seed collapsed at initial bias `3.10`; all three collapsed at `3.25`.
- The empirical 50% crossing is `3.175`, bracketed by `[3.15, 3.20]`. This is
  descriptive, not derived.
- At bias `3.25`, final `H(p_bar) = 0.013227 ± 0.000732`, top-class fraction
  `1.0`, and held-out accuracy is exactly chance (`1/3`).
- At bias `3.5`, every seed collapsed at step `390`; seed 0 ended at
  `H(p_bar) = 0.012007` with accuracy `1/3`.

The boundary is finite-horizon and specific to this configuration. The run
does not prove an asymptotic limit of exactly zero.

### What went wrong in the discarded `THEORY.md` draft

The draft used `dH/dz = pi - e_y`, which is the gradient of cross-entropy with
a one-hot label, not entropy. The correct entropy gradient is
`dH/dz_k = -p_k (log p_k + H(p))`, which sums to zero and respects softmax
shift invariance. The draft also described a bias-only model while the code
adapts both `W` and `b`, and asserted an `eta*` bound without deriving it from
the actual update. `THEORY.md` now contains the corrected derivation and a short
correction note; the incorrect draft is no longer used as evidence. This is a
reminder to check every gradient against autograd before building on it.

## Phase 2 — local stability analysis

### The correct entropy gradient

For one sample, let `p = softmax(z)` and `C = diag(p) - p p^T`. The softmax
Jacobian is `C`, and `C 1 = 0`. Differentiating

```text
H(p) = -sum_k p_k log p_k
```

gives

```text
dH/dz = -C (log p + H(p) 1).
```

The extra `H(p) 1` term is easy to miss. It appears because changing a logit
changes both `p` and the normaliser inside `log p`. The result sums to zero over
classes, which is required because adding the same constant to every logit does
not change `p`. Comparing this expression with autograd caught the earlier
mistake.

For the information objective

```text
L_info = (1/N) sum_i H(p_i) - lambda H(p_bar),   p_bar = (1/N) sum_i p_i,
```

the logit gradient is

```text
dL_info/dz_i = -(1/N) C_i (log p_i + H(p_i) 1 - lambda log p_bar).
```

### Reduced `W = 0` flow

Setting `W = 0` makes every sample share the logits `b`, so `p_bar = p` and

```text
L_info = (1 - lambda) H(softmax(b)).
```

Gradient flow then gives

```text
db/dt = (1 - lambda) p * (b - (p . b) 1).
```

This is a useful mechanism check because it has a closed-form linearisation. Near
`b = 0`, on the sum-zero subspace, the eigenvalue is `(lambda - 1)/K`:

- `lambda < 1`: positive eigenvalue, so the uniform bias is unstable and the
  bias spreads out, collapsing the marginal;
- `lambda = 1`: the objective is identically zero, so the flow is neutral;
- `lambda > 1`: negative eigenvalue, so the uniform bias is locally stable.

The saved run reproduces this transition at `lambda = 1`. At `lambda = 0.9` the
collapse is simply slower than at `lambda = 0`, which is expected because the
driving coefficient `1 - lambda` is smaller.

### Why the full model is still a saddle

At `W = 0, b = 0`, every `p_i = 1/K`. The bias block of the Hessian is

```text
d^2 L_info/db^2 = (lambda - 1) C,   C = diag(1/K) - (1/K^2) 1 1^T,
```

whose non-zero eigenvalues are `(lambda - 1)/K` on the sum-zero subspace. So
`lambda > 1` does stabilise marginal collapse. But a `W` perturbation changes
each sample's logits in a sample-specific way, and on those contrast modes the
`lambda` term cancels. Those directions retain negative entropy curvature, so
the symmetric point is a **saddle**, not a minimum. This matches the intent of
the objective: it discourages a one-hot marginal, not confidence itself.

The practical consequence is that local stability of the marginal mode does not
imply global convergence of the full model. It also does not predict the Phase 1
basin boundary at `3.175`, because Phase 1 starts from `W = I` and a large bias,
far from the symmetric point.

### Discrete step bound

For gradient descent with step size `eta`, the marginal-mode multiplier is

```text
1 - eta (lambda - 1)/K.
```

It is stable when its magnitude is below one, giving

```text
0 < eta < 2K/(lambda - 1)      (lambda > 1).
```

At `lambda = 2`, `K = 3`, the bound is `eta < 6`. The saved one-step check is
stable at `5.8` and unstable at `6.2` and `7.0`, with the observed factor
matching `|1 - eta/3|` to about `1e-6`. The sign of the factor alternates when
`eta > K/(lambda - 1)`, which is why the table compares magnitudes.

### Scope limits of Phase 2

- The analysis is local at `W = 0, b = 0`, not a global convergence proof.
- The reduced `W = 0` model is a mechanism check; adapting `W` is what creates
  the Phase 1 balanced non-collapsed solution.
- No deep-network or CIFAR-10-C claim is made. That is Phase 3.

## Phase 3 planning notes

### Environment pinning

`requirements.lock.txt` records exact versions instead of the ranges in
`requirements.txt`. The earlier saved runs exposed a real problem: the run
metadata read the torch package version through `importlib.metadata`, which at
one point reported `2.15.0` while `torch.__version__` reported `2.9.1+cu126`.
Recording only one of those values is not enough. Run metadata now records both,
together with CUDA and cuDNN versions, so a mismatch is visible in the artifact
rather than discovered later.

### Determinism and the GPU tolerance

`seed_everything(..., deterministic=True)` now also disables TF32 and sets
`cudnn.deterministic=True` with `cudnn.benchmark=False`. That removes the largest
sources of run-to-run drift on an Ampere-class GPU, but CUDA reductions are
still not guaranteed to be bitwise reproducible. The project will therefore:

1. run the same smoke configuration twice with the same seed;
2. measure the maximum absolute difference in the logged metrics;
3. record that measured tolerance in the run metadata and README;
4. compare later runs using the tolerance, not exact equality.

No tolerance value is claimed before that measurement exists.

### Why ResNet-20 and batch size 64

The target GPU is an RTX 2050 with 4 GiB VRAM. ResNet-20 for CIFAR-10 has about
0.27 M parameters and is small enough to leave room for activations and the
adaptation graph. Batch size starts at 64 and is configurable; the loader catches
CUDA out-of-memory errors and retries at 32. A larger model such as
WideResNet-28-10 is not appropriate for this hardware.

### Data plan

Only two dataset downloads are needed: clean CIFAR-10 (official Toronto python
archive, 170.5 MB) and CIFAR-10-C (Zenodo, 2.92 GB). The source classifier is a
1.09 MiB ResNet-20 checkpoint trained on clean CIFAR-10. Exact URLs, checksums,
extracted sizes, and the peak disk budget are in `DATA_SOURCES.md`. No data has
been downloaded yet.

## What is not established yet

- The Phase 2 derivation is local at `W = 0, b = 0`; it does not explain the
  Phase 1 basin boundary or prove global convergence.
- The full-model symmetric point is a saddle for `lambda > 1`, so stabilising
  the marginal mode is not sufficient for the whole model.
- No real network has been adapted, so there is no claim about deep-network
  behaviour.
- No literature novelty claim has been checked.

## Commands

From `entropy-collapse-tta/`:

```bash
python -m pytest -q
python scripts/phase0_smoke.py --config configs/phase0_smoke.yaml
python scripts/run_theory.py --config configs/theory_phase1.yaml
python scripts/make_plots.py \
  --run-dir results/theory_phase1/20261104T173154.734687Z_theory_phase1_seed0_4f7a72f0 \
  --trajectory-seed 0
python scripts/run_stability.py --config configs/theory_phase2.yaml
python scripts/make_stability_plots.py \
  --run-dir results/theory_phase2/20261104T184229.827471Z_theory_phase2_seed0_6b788892
```

## Phase 0 checklist

- [x] Repository scaffold matching the requested layout
- [x] Installable package metadata and requirements
- [x] YAML config inheritance and overrides
- [x] Stable config hashing
- [x] Central seeding
- [x] Append-only logging and run metadata
- [x] Predictive and marginal entropy functions
- [x] Entropy identity and gradient tests
- [x] Reproduction instructions
- [x] Phase 1 toy model, basin scan, saved run, and figures

## Phase 1 checklist

- [x] Balanced Gaussian-mixture generator with reproducible seeding
- [x] Linear-softmax model with adaptable `W` and `b`
- [x] Full-batch entropy-minimisation dynamics
- [x] Persistent collapse detector shared with the real-network track
- [x] Configurable bias grid and multi-seed scan
- [x] Per-run CSV, aggregate CSV, JSONL trajectories, and `summary.json`
- [x] Figures generated from saved artifacts, not typed numbers
- [x] Unit tests for data balance, model shapes, collapse detection,
      aggregation, and CSV output
- [x] Three-seed saved run and README results
- [x] Analytic bias dynamics and stability condition (moved to Phase 2)

## Phase 2 checklist

- [x] Correct entropy logit gradient derived and checked against autograd
- [x] Information-objective logit and parameter gradients checked against autograd
- [x] Symmetric-point bias-block Hessian equals `(lambda - 1) C`
- [x] Centred-feature full Hessian has only the marginal mode positive
- [x] Reduced `W = 0` flow derived with closed-form linearisation
- [x] `lambda = 1` collapse/balance transition measured at three verification seeds
- [x] Discrete step bound `eta < 2K/(lambda - 1)` derived and measured
- [x] Saved run with `summary.json`, Hessian CSV, trajectories, and figures
- [x] `THEORY.md` updated with derivations, scope limits, and draft correction
- [x] Phase 2 unit tests
- [x] README and learning notes updated from saved artifacts

## Phase 3 checklist (in progress)

- [x] Exact environment lock file
- [x] GPU/CUDA/cuDNN and batch-size metadata capture
- [x] Dataset and checkpoint sources, sizes, and disk budget documented
- [ ] Download and checksum CIFAR-10 and CIFAR-10-C
- [ ] Load the ResNet-20 checkpoint and measure clean CIFAR-10 accuracy
- [ ] Two identical smoke runs and a measured determinism tolerance
- [ ] Smoke adaptation run at batch size 64 with 32 fallback
- [ ] Small CIFAR-10-C stream
- [ ] Full corruption sweep if runtime allows
