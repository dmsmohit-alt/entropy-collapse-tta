# When Does Entropy Minimisation at Test Time Collapse?

A research project on the dynamics of entropy-minimising test-time adaptation
(TTA), with a tractable linear-softmax analysis and controlled experiments on
corrupted-image streams.

> **Status: Phase 2 complete.** The scaffold, the toy-model experiments, and the
> local stability analysis are implemented. Phase 1 and Phase 2 results below
> come from saved runs with their configs and logs. No real-network experiment
> has been run yet, so there are **no CIFAR or TTA performance claims** in this
> README.

## Research question

Entropy-minimising TTA such as TENT adapts a model on unlabeled test data.
On long or changing streams it can collapse to a single predicted class. This
project asks:

1. In a tractable linear-softmax model, when does the marginal prediction
   distribution become one-hot under entropy minimisation?
2. What is the local stability condition for the information-maximisation
   objective `E[H(p)] - lambda * H(p_bar)`?
3. Do the same qualitative trajectories appear when adapting BatchNorm
   affine parameters of a real CIFAR-10 classifier?

The explanatory target is the marginal entropy `H(p_bar)` and the exact
decomposition

```text
E_x[H(p_theta(.|x))] = H(p_bar) - I(X; Y_theta).
```

This identity is exact; it is not claimed as a new contribution. The planned
contribution is a sharper dynamical account of when the marginal term vanishes
and a finite-step-size stability condition for the stabilised objective.

## Phase status

| Phase | Status | Scope |
|---|---|---|
| 0 | Complete | Scaffold, configs, seeding, logging, entropy metrics, tests |
| 1 | Complete | Toy Gaussian-mixture model, collapse dynamics, basin map |
| 2 | Complete | Analytic gradients, reduced bias flow, symmetric-point Hessian, step bound |
| 3 | In progress | ResNet-20, CIFAR-10-C streams, Source/TENT; smoke before full sweep |
| 4 | Not started | Marginal-entropy stabiliser and RDumb reset baseline |
| 5 | Not started | Analysis, ablations, toy-vs-real comparison |
| 6 | Optional | Streamlit demonstration |
| 7 | Not started | Final README and interview preparation notes |

Do not treat the table above as evidence that an experiment succeeded. It only
tracks implementation progress.

## Repository layout

```text
entropy-collapse-tta/
  README.md
  THEORY.md                 # verified Phase 1 protocol/results and Phase 2 derivations
  LEARNING_NOTES.md
  DATA_SOURCES.md           # Phase 3 dataset/checkpoint sources, sizes, disk budget
  requirements.txt
  requirements.lock.txt
  pyproject.toml
  configs/
    base.yaml
    phase0_smoke.yaml
    theory_phase1.yaml
    theory_phase2.yaml
  src/
    theory/                 # Gaussian mixture, toy dynamics, basin scan, stability
      basin.py
      toy_model.py
      stability.py          # Phase 2 gradients, Hessian, reduced flow, step bound
    data/                   # synthetic and CIFAR-10-C streams (Phase 3)
    models/                 # pretrained loading and BN selection (Phase 3)
    tta/                    # Source, TENT, stabiliser, reset (Phase 3/4)
    metrics/
      entropy.py            # predictive entropy, marginal entropy, gap
      collapse.py           # persistent collapse detector
    utils/                  # config, seeding, logging, run directories
  scripts/
    phase0_smoke.py
    run_theory.py
    make_plots.py
    run_stability.py        # Phase 2 verification runner
    make_stability_plots.py # Phase 2 figures
    check_environment.py    # Verify requirements.lock.txt and print GPU details
  results/                  # run logs and figures from real executions
  tests/
  app/                      # optional demo (Phase 6)
```

## Installation

Phase 0 and the Phase 1 toy model are CPU-safe. From the project directory:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

An editable install is optional because the test and smoke-script paths are
configured explicitly:

```bash
python -m pip install -e ".[dev]"
```

For a reproducible Phase 3 environment, use the exact pins in
`requirements.lock.txt`. The CUDA wheel must be installed from the PyTorch
index first:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.9.1+cu126
python -m pip install -r requirements.lock.txt
```

The lock file records the environment used for planning: Python 3.14.5,
torch 2.9.1+cu126, CUDA 12.6, cuDNN 9.10.2, and an NVIDIA GeForce RTX 2050
with 4 GiB VRAM. Verify a fresh install with:

```bash
python scripts/check_environment.py
```

Dataset and checkpoint sources are documented in
`DATA_SOURCES.md`; nothing is downloaded automatically.

The saved Phase 1 run recorded Python 3.14.5, NumPy 2.5.4, and PyTorch
package metadata 2.15.0 in its `run_metadata.json`; the run itself used
`device: cpu`. The saved Phase 2 run recorded the same package metadata. The
current environment reports torch `2.9.1+cu126` consistently from both
`torch.__version__` and package metadata; run metadata now records both values
so a future mismatch is visible. Because the older runs were made before the
lock file existed, re-check installed versions before comparing a new run with
them.

## Run the Phase 0 checks

Run the unit tests from the repository root:

```bash
python -m pytest -q
```

Run the end-to-end wiring check:

```bash
python scripts/phase0_smoke.py --config configs/phase0_smoke.yaml
```

This writes a new directory under `results/phase0_smoke/` containing:

- `config.yaml` — the fully resolved config, including base defaults;
- `run_metadata.json` — seed, config hash, Python/platform/package versions,
  and invocation arguments;
- `metrics.jsonl` — the synthetic diagnostic record.

The smoke check is explicitly labelled as synthetic wiring, not a research
result. Overrides work as follows:

```bash
python scripts/phase0_smoke.py \
  --config configs/phase0_smoke.yaml \
  --override seed=1 \
  --override logging.level=DEBUG
```

## What Phase 0 verifies

- Entropy is computed in nats with stable `log_softmax`/`xlogy` operations.
- `H(p_bar) - mean H(p)` numerically matches direct mutual information.
- Entropy gradients are finite.
- Config inheritance, dotted overrides, validation, and stable hashing work.
- Python, NumPy, and PyTorch seeds reproduce the same random streams.
- JSONL logs reject NaN values and are readable immediately after each write.
- Every run directory records its exact resolved config and environment.

## Run the Phase 1 toy experiment

The Phase 1 configuration sweeps 73 initial-bias values over three data seeds,
using 2,500 full-batch adaptation steps per trajectory:

```bash
python scripts/run_theory.py --config configs/theory_phase1.yaml
python scripts/make_plots.py \
  --run-dir results/theory_phase1/20261104T173154.734687Z_theory_phase1_seed0_4f7a72f0 \
  --trajectory-seed 0
```

Each run directory contains the resolved config, run metadata, every per-run
row in `basin_scan.csv`, the aggregate table, JSONL trajectories, and figures.
The figures for the saved run below are stored under that run's `figures/`
directory.

## Phase 1 results

These values are from the saved run

```text
results/theory_phase1/20261104T173154.734687Z_theory_phase1_seed0_4f7a72f0/
```

with `K=3`, `d=3`, class separation `3.0`, 600 training and 1,000 evaluation
samples per class, full-batch SGD at learning rate `0.03`, and 2,500 steps.
Both `W` and `b` were adapted. Entropies are in nats.

### Source model

The unadapted class-prototype model (`W=I`, `b=0`) over three seeds:

| Metric | Mean ± std |
|---|---|
| Accuracy | 0.9688 ± 0.0043 |
| `H(p_bar)` | 1.098604 ± 0.000007 |
| Mean predictive entropy | 0.4426 ± 0.0022 |
| Top-class fraction | 0.3359 ± 0.0005 |

### Basin boundary

The empirical 50% crossing is at initial bias **3.175**, bracketed by the tested
values `[3.15, 3.20]`. No seed collapsed at bias `3.10`; all three collapsed at
bias `3.25`.

| Initial bias | Collapse rate | Final `H(p_bar)` | Final top-class fraction | Final accuracy |
|---:|---:|---:|---:|---:|
| 3.10 | 0/3 | 1.09847 ± 0.00011 | 0.3391 ± 0.0024 | 0.9649 ± 0.0044 |
| 3.15 | 1/3 | 0.58319 ± 0.54388 | 0.6721 ± 0.3295 | 0.6442 ± 0.3153 |
| 3.20 | 2/3 | 0.22071 ± 0.35855 | 0.8929 ± 0.1855 | 0.4343 ± 0.1749 |
| 3.25 | 3/3 | 0.01323 ± 0.00073 | 1.0000 ± 0.0000 | 0.3333 ± 0.0000 |

At bias `3.25`, therefore, the marginal entropy falls from the source value
`1.098604` to `0.01323`, the top-class fraction reaches `1.0`, and held-out
accuracy is exactly chance (`1/3`). This is a finite-horizon numerical result,
not a proof that the limit is exactly zero.

### Illustrative trajectories

Across all three seeds:

- Bias `3.0`: no seed collapsed. Seed 0 ended at `H(p_bar)=1.098535` with
  accuracy `0.9620`.
- Bias `3.5`: every seed collapsed at step `390`. Seed 0 ended at
  `H(p_bar)=0.012007`, top-class fraction `1.0`, accuracy `0.3333`.

The numerical boundary is specific to this toy configuration, initialisation,
optimiser, step size, and horizon. It is **not** claimed as a universal
threshold. Phase 2 analyses local stability separately, as described below.

Figures:

```text
results/theory_phase1/20261104T173154.734687Z_theory_phase1_seed0_4f7a72f0/figures/basin_boundary.png
results/theory_phase1/20261104T173154.734687Z_theory_phase1_seed0_4f7a72f0/figures/trajectories.png
```

## Run the Phase 2 stability checks

The Phase 2 configuration uses `K=3`, `d=2`, `N=6`, float64, three
verification seeds `[0, 1, 2]`, and seven `lambda` values:

```bash
python scripts/run_stability.py --config configs/theory_phase2.yaml
python scripts/make_stability_plots.py \
  --run-dir results/theory_phase2/20261104T184229.827471Z_theory_phase2_seed0_6b788892
```

`run_stability.py` checks four derived claims:

1. The analytic entropy and information gradients match autograd.
2. The symmetric-point bias-block Hessian equals `(lambda - 1) C`, with
   `C = diag(1/K) - (1/K^2) 1 1^T`.
3. With centred features, the only positive full-Hessian eigenvalue is the
   marginal mode `(lambda - 1) / K`.
4. The reduced `W = 0` bias flow switches from collapse to balance at
   `lambda = 1`, and the one-step spread factor is `1 - eta (lambda - 1)/K`.

Each run directory contains `summary.json`, `hessian_spectrum.csv`,
`gradient_checks.json`, `reduced_trajectories.jsonl`, and generated figures.

## Phase 2 results

These values are from the saved run

```text
results/theory_phase2/20261104T184229.827471Z_theory_phase2_seed0_6b788892/
```

The run used CPU and float64 and took `1.78` seconds. Entropies are in nats.

### Gradient and Hessian checks

| Check | Max absolute error | Tolerance |
|---|---:|---:|
| Entropy / information logit gradients | `5.551e-16` | `1e-9` |
| Bias-block Hessian vs `(lambda - 1) C` | `2.220e-16` | `1e-9` |
| Bias-block eigenvalues vs `(lambda - 1)/K` | `4.441e-16` | `1e-9` |
| Centred full-Hessian maximum vs `max(0, (lambda-1)/K)` | `8.882e-16` | `1e-9` |

All checks passed. Errors are maxima over the three verification seeds.

### Reduced `W = 0` bias flow

Initial bias spread `0.014142`; initial `H(p) = 1.098579`; `log 3 = 1.098612`.
The simulation uses the exact reduced update for 2,000 steps at learning rate
`0.1`.

| `lambda` | Final `H(p)` | Final bias spread | Behaviour |
|---:|---:|---:|---|
| 0.0 | 0.004725 | 6.756954 | collapse |
| 0.5 | 0.010671 | 6.006260 | collapse |
| 0.9 | 0.184525 | 3.229866 | collapse |
| 1.0 | 1.098579 | 0.014142 | neutral |
| 1.1 | 1.098612 | 0.000018 | balanced |
| 2.0 | 1.098612 | `< 1e-6` | balanced |

The observed transition is at `lambda = 1`, matching the derived threshold. The
`lambda = 0.9` case is a slow collapse within the 2,000-step horizon, not an
instantaneous one.

### Discrete step-size check

At `lambda = 2`, `K = 3`, the derived bound is `eta < 6`. The table reports the
magnitude of the predicted one-step factor because the sign alternates when the
factor is negative.

| `eta` | Predicted factor | Observed factor | Stable? |
|---:|---:|---:|---|
| 1.0 | 0.666667 | 0.666667 | yes |
| 3.0 | 0.000000 | 0.000000 | yes |
| 5.8 | 0.933333 | 0.933334 | yes |
| 6.2 | 1.066667 | 1.066668 | no |
| 7.0 | 1.333333 | 1.333334 | no |

The largest factor error is `1.167e-6`, consistent with the small finite
perturbation used in the one-step test.

### What Phase 2 does and does not show

- **Derived:** the exact entropy/information logit gradients, the reduced
  `W = 0` flow, the `lambda = 1` threshold, the marginal curvature
  `(lambda - 1)/K`, and the discrete bound `eta < 2K/(lambda - 1)` for
  `lambda > 1`.
- **Numerically observed:** all four checks above, on the saved run.
- **Scope limit:** the analysis is local at `W = 0, b = 0`. It does **not**
  predict the Phase 1 crossing at `3.175`, which is a finite-horizon nonlinear
  result from a different initialisation.
- **Saddle caveat:** the full-model symmetric point remains a saddle for
  `lambda > 1`; only the marginal-collapse direction is stabilised, while
  per-sample sharpening directions still have negative curvature.

Figures:

```text
results/theory_phase2/20261104T184229.827471Z_theory_phase2_seed0_6b788892/figures/stability_spectrum.png
results/theory_phase2/20261104T184229.827471Z_theory_phase2_seed0_6b788892/figures/reduced_dynamics.png
```

## Reproducibility conventions

- YAML is the source of truth; each run saves a resolved copy.
- Seeds are integers in `[0, 2**32)` and are stored in the run metadata.
- Main experiments will use at least three seeds and report mean ± standard
  deviation.
- Results are written as append-only JSONL, with figures generated from those
  files rather than typed manually.
- Hard rule for later phases: any README number without a saved log/config is
  written as **TBD**, not estimated from memory.

## Compute

Phase 0 and Phase 1 run on CPU; the Phase 1 basin scan took 486.64 seconds in
the saved run. The Phase 2 stability run took 1.78 seconds on CPU. The target
for Phases 3–5 is **TBD**: either a Colab/Kaggle T4 or a CPU-only laptop. Batch
size, stream length, corruption count, and adaptation frequency will be
configurable; small CPU smoke configurations will be provided before any full
run.

## Next phase

Phase 2 is complete. Phase 3 is approved but has not downloaded any data yet.
It will adapt BatchNorm affine parameters of a pretrained ResNet-20 on
CIFAR-10-C streams, comparing Source and TENT. The local target is an NVIDIA
GeForce RTX 2050 with 4 GiB VRAM; the plan starts at batch size 64, falls back
to 32 on OOM, and runs a smoke test before a small stream and only then a full
corruption sweep. Exact dataset and checkpoint sources, sizes, checksums, and
the disk budget are in `DATA_SOURCES.md`. No real-network claims will be made
until a saved run exists.
