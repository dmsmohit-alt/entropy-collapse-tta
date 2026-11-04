# Phase 3 data sources and disk budget

> **Status: planning only.** No dataset or checkpoint has been downloaded yet.
> Sizes and checksums below were read on 2026-11-05 from the official file
> metadata (HTTP `HEAD` / the Zenodo and GitHub APIs), not from a local copy.

## 1. Clean CIFAR-10

| Item | Value |
|---|---|
| Source | Alex Krizhevsky, University of Toronto |
| Page | <https://www.cs.toronto.edu/~kriz/cifar.html> |
| File | `cifar-10-python.tar.gz` |
| URL | <https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz> |
| Archive size | `170,498,071` bytes (162.6 MiB; the page rounds to 163 MB) |
| MD5 | `c58f30108f718f92721af3b95e74349a` |
| Extracted layout | `cifar-10-batches-py/` with 5 training batches, 1 test batch, `batches.meta` |
| Extracted size | about `184.5 MB` (`6 x 30,720,000` bytes plus pickle overhead) |
| Planned location | `data/cifar10/` |
| Citation | Krizhevsky, *Learning Multiple Layers of Features from Tiny Images*, 2009 |

The python archive is used because it needs no image decoder: each batch is a
pickled `uint8` array of shape `10000 x 3072`, which NumPy can read directly.

## 2. CIFAR-10-C

| Item | Value |
|---|---|
| Source | Zenodo record 2535967 (Hendrycks & Dietterich) |
| Record page | <https://zenodo.org/records/2535967> |
| File | `CIFAR-10-C.tar` |
| URL | <https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1> |
| Archive size | `2,918,471,680` bytes (2.72 GiB) |
| MD5 | `56bf5dcef84df0e2308c6dcbcbbd8499` |
| Extracted layout | `CIFAR-10-C/<corruption>.npy` (15 corruptions) and `labels.npy` |
| Extracted size | about `2.304 GB` (`15 x 50,000 x 3,072` uint8 bytes plus labels) |
| Planned location | `data/cifar10c/` |
| Citation | Hendrycks & Dietterich, *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations*, ICLR 2019 |

Each corruption file holds all five severities in order: 50,000 rows =
5 severities x 10,000 test images. The labels file has 50,000 entries and is
repeated per severity by the loader. The dataset is evaluation-only; it is never
used for source training or hyperparameter selection.

## 3. Pretrained CIFAR-10 classifier

Primary choice: a ResNet-20 checkpoint trained on clean CIFAR-10.

| Item | Value |
|---|---|
| Source | `chenyaofo/pytorch-cifar-models`, release tag `resnet` |
| Repository | <https://github.com/chenyaofo/pytorch-cifar-models> |
| License | BSD-3-Clause |
| File | `cifar10_resnet20-4118986f.pt` |
| URL | <https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet20-4118986f.pt> |
| Size | `1,139,055` bytes (1.09 MiB) |
| Checksum | none published; compute SHA-256 after download and record it |
| Planned location | `data/checkpoints/cifar10_resnet20-4118986f.pt` |
| Parameters | about 0.27 M, which fits the 4 GiB GPU comfortably |

We will verify the checkpoint before using it:

1. load it into a ResNet-20 implementation matching the CIFAR architecture
   (3x3 stem, three stages of 16/32/64 channels, no max-pool);
2. evaluate on the clean CIFAR-10 test split;
3. record the measured accuracy, checkpoint SHA-256, and model definition in the
   source-evaluation run directory.

No accuracy number is claimed in this file before that measurement exists.

Fallback if the external checkpoint cannot be fetched or does not load: train
ResNet-20 from scratch on clean CIFAR-10 with a fixed seed and save the source
checkpoint under `data/checkpoints/`. This downloads no extra weights and is
fully local, at the cost of training time.

ImageNet-pretrained torchvision ResNet-18 is **not** the primary choice: it is
not a CIFAR-10 classifier, and fine-tuning it would add a separate training
stage. Its weight file is `46,830,571` bytes if that route is ever needed.

## 4. Disk budget

| Item | Download | On disk after extraction |
|---|---:|---:|
| CIFAR-10 archive + extracted | 170.5 MB | 184.5 MB |
| CIFAR-10-C archive + extracted | 2,918.5 MB | 2,304.0 MB |
| ResNet-20 checkpoint | 1.1 MB | 1.1 MB |
| **Total** | **3,090.1 MB (2.88 GiB)** | **2,489.6 MB (2.32 GiB)** |

Peak usage if both archives are kept while extracting is about **5.6 GB**.
Extracting one dataset at a time and deleting its archive lowers the steady
state to about **2.5 GB**. Add roughly 1–2 GB for source-evaluation logs,
adaptation logs, and figures.

`D:` currently has **220.9 GB free**, so the datasets fit easily. A 6 GB
working reserve is enough for the full Phase 3 sweep.

## 5. Reproducibility rules for downloads

- Record the source URL, final URL, byte size, and checksum in the run metadata.
- Verify the CIFAR-10 and CIFAR-10-C MD5 values before extraction.
- Compute and record the checkpoint SHA-256; the upstream release publishes
  only the filename hash fragment.
- Never silently re-download or overwrite a dataset. Cache under `data/` and
  fail loudly on a checksum mismatch.
