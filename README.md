# Image Colorization with Pix2Pix (Classical + Regression)

ABD Thesis Submission — **KUANG ZIXUAN**

This repository contains the thesis manuscript, source-code listing, and all code/checkpoints used in the thesis project on image colorization with Pix2Pix, including:

- a **classical classification-based pipeline** using 313-bin `ab` quantization, and
- a **regression-based pipeline** using direct 2-channel `ab` prediction.

---

## Table of Contents

- [1. Project Structure](#1-project-structure)
- [2. Hardware Dependencies](#2-hardware-dependencies)
- [3. Software Dependencies](#3-software-dependencies)
- [4. Dataset](#4-dataset)
- [5. How to Install](#5-how-to-install)
- [6. How to Run](#6-how-to-run)
- [7. Notes / Troubleshooting](#7-notes--troubleshooting)

---

## 1. Project Structure

```text
KUANG_ZIXUAN/
├── ABD_Thesis.pdf
├── Code_Listing.pdf
├── README.md
└── Final_report_code/
    ├── requirements.txt
    ├── benchmark_two_models.py
    ├── benchmark_sweep.py
    ├── Classical/
    │   ├── train_pix2pix_from_npy.py
    │   ├── load_lab_npy_data.py
    │   ├── val.py
    │   ├── (full)main-model.pt
    │   └── (VL)main-model.pt
    ├── Regression/
    │   ├── train_pix2pix_from_npy.py
    │   ├── load_lab_npy_data.py
    │   ├── val.py
    │   ├── (low-lr)main-model.pt
    │   ├── (only-regression)main-model.pt
    │   └── (TV-loss)main-model.pt
    └── archive/
        ├── l/gray_scale.npy
        ├── ab/ab/ab1.npy
        ├── ab/ab/ab2.npy
        ├── ab/ab/ab3.npy
        └── pts_in_hull.npy
```

### Root directory

- **`ABD_Thesis.pdf`** — the full thesis manuscript.
- **`Code_Listing.pdf`** — formatted source-code listing (appendix to the thesis).
- **`README.md`** — this file.
- **`Final_report_code/`** — all source code and model checkpoints used in the thesis.

### `Final_report_code/`

- **`requirements.txt`** — Python dependencies (pip).
- **`benchmark_two_models.py`** — runs both trained models on the validation split and saves rendered RGB images to disk.
- **`benchmark_sweep.py`** — sweeps the benchmark over multiple sample counts and plots average time per image.

### `Final_report_code/Classical/`

Classification pipeline using 313-bin `ab` quantization, cross-entropy, and GAN.

- **`train_pix2pix_from_npy.py`** — training entry point.
- **`load_lab_npy_data.py`** — dataset / DataLoader.
- **`val.py`** — visual validation of one random or indexed sample, including annealed-mean decoding.
- **`(full)main-model.pt`** — checkpoint: full run.
- **`(VL)main-model.pt`** — checkpoint: best variant reported in the thesis.

### `Final_report_code/Regression/`

Regression pipeline using direct 2-channel `ab` prediction, L1 + GAN, and optional TV loss.

- **`train_pix2pix_from_npy.py`** — training entry point.
- **`load_lab_npy_data.py`** — dataset / DataLoader.
- **`val.py`** — visual validation.
- **`(low-lr)main-model.pt`** — checkpoint: low-LR variant (default used by the benchmark).
- **`(only-regression)main-model.pt`** — regression-only variant (no GAN).
- **`(TV-loss)main-model.pt`** — regression + TV loss.

### `Final_report_code/archive/`

Dataset expected by the code (not redistributed; see Section 4).

- **`l/gray_scale.npy`** — L-channel inputs.
- **`ab/ab/ab1.npy`** — `ab` targets, part 1.
- **`ab/ab/ab2.npy`** — `ab` targets, part 2.
- **`ab/ab/ab3.npy`** — `ab` targets, part 3.
- **`pts_in_hull.npy`** — the 313 `ab`-bin centres used by the classification pipeline. Auto-generated on first run if missing.

---

## 2. Hardware Dependencies

### Training

- A CUDA-capable NVIDIA GPU with **≥ 16 GB VRAM** is strongly recommended.
- Models were trained on an **RTX A6000 (48 GB)** with batch size 64 @ 224×224.
- Training will also run on smaller GPUs by reducing `batch_size` in `TrainConfig`; see `oom_auto_shrink_batch`.
- CPU-only training is technically possible but impractically slow and is not recommended.

### Inference / validation / benchmark

- Works on any single CUDA GPU (≥ 6 GB VRAM is sufficient at batch size 4–8).
- Falls back to CPU automatically via `get_device()`.
- A full 500-image benchmark on CPU takes tens of minutes.

### Disk

- ≈ **2.5 GB** for the provided model checkpoints.
- ≈ **2.6 GB** for the `gray_scale.npy` + 3× `ab*.npy` dataset arrays (see Section 4 — not included in this repository/archive).

---

## 3. Software Dependencies

### Operating system

- Linux (developed/tested on Ubuntu 24.04, kernel 6.8)
- macOS and Windows should work for CPU inference
- GPU training on Windows requires a matching CUDA toolkit and NVIDIA driver

### Core

- Python **3.10 or newer** (3.10 / 3.11 tested)
- pip **23+** (for `--extra-index-url` support)
- NVIDIA driver **≥ 550** and a CUDA runtime compatible with the installed PyTorch wheel

### Python packages

Pinned in `Final_report_code/requirements.txt`:

| Package        | Version              | Notes                                         |
| -------------- | -------------------- | --------------------------------------------- |
| `torch`        | 2.7.1 + cu128        | CUDA 12.8 / Blackwell build                   |
| `torchvision`  | 0.22.1 + cu128       |                                               |
| `numpy`        | 2.2.6                |                                               |
| `matplotlib`   | 3.10.8               |                                               |
| `scikit-image` | 0.25.2               |                                               |
| `pillow`       | 12.0.0               |                                               |
| `tqdm`         | 4.67.3               |                                               |
| `fastai`       | 2.8.7                | for `DynamicUnet` + ResNet18 backbone         |
| `ipython`      | 8.38.0                | implicit dependency of fastai / fastprogress |

### GPU wheel selection

The default wheels target **CUDA 12.8**, as required by NVIDIA Blackwell (RTX 50-series, `sm_120`).

For other environments, edit `requirements.txt` as noted in its header:

**CUDA 11.8**

```text
--extra-index-url .../whl/cu118
torch==2.7.1+cu118
torchvision==0.22.1+cu118
```

**CUDA 12.1**

```text
--extra-index-url .../whl/cu121
torch==2.7.1+cu121
torchvision==0.22.1+cu121
```

**CPU only**

```text
--extra-index-url .../whl/cpu
torch==2.7.1
torchvision==0.22.1
```

No IDE is required. Any terminal with Python installed is sufficient. (VS Code was used during development; a `.vscode/` folder is included but is purely optional.)

---

## 4. Dataset

> **Note:** The dataset is **not included** in this repository/archive.

The code expects the "Image Colorization" LAB dataset originally published on Kaggle:

<https://www.kaggle.com/datasets/shravankumar9892/image-colorization>

After download, place the four `.npy` files into `Final_report_code/archive/` exactly as:

```text
Final_report_code/archive/l/gray_scale.npy
Final_report_code/archive/ab/ab/ab1.npy
Final_report_code/archive/ab/ab/ab2.npy
Final_report_code/archive/ab/ab/ab3.npy
```

### About `pts_in_hull.npy`

`pts_in_hull.npy` does **not** need to be supplied manually. It contains the 313 `ab`-bin centres used by the classification pipeline. If it is missing, `load_lab_npy_data.py` will generate it automatically by k-means on the first run and cache it to:

```text
archive/pts_in_hull.npy
```

---

## 5. How to Install

The code is pure Python — no compilation step is required.

### 1. Create an isolated environment

**Option A — `venv`** (ships with Python, no extra install):

```bash
python3 -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows PowerShell
```

**Option B — `conda` / `mamba`** (preferred if you want to pin the Python version or manage CUDA toolkits).

Install one of the following distributions first:

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- [Miniforge](https://github.com/conda-forge/miniforge)
- [Anaconda](https://www.anaconda.com/download)

Then create and activate an env:

```bash
conda create -n colorize python=3.10 -y
conda activate colorize
```

### 2. Upgrade pip

```bash
pip install --upgrade pip
```

### 3. Install dependencies

```bash
cd Final_report_code
pip install -r requirements.txt
```

### 4. If not using CUDA 12.8

Edit `requirements.txt` according to the instructions in Section 3 before installing.

### Quick sanity check

```bash
python -c "import torch; print(torch.__version__, 'cuda=', torch.cuda.is_available())"
```

---

## 6. How to Run

All commands below assume the working directory is:

```text
Final_report_code/
```

### (a) Reproduce the side-by-side benchmark used in the thesis

Renders 500 validation images with each of the two reported models and prints timings.

```bash
python benchmark_two_models.py
```

**Outputs**

- `test1/img_000.png … img_499.png` — Regression model
- `test2/img_000.png … img_499.png` — Classical model

### (b) Reproduce the throughput-vs-sample-count sweep figure

```bash
python benchmark_sweep.py
```

**Output**: `benchmark_sweep.png`

### (c) Visualize a single validation sample

Shows the grayscale input, prediction, and ground truth side-by-side.

Classification model with annealed-mean decoding:

```bash
cd Classical && python val.py && cd ..
```

Regression model:

```bash
cd Regression && python val.py && cd ..
```

The target checkpoint and sample-selection mode are configured at the top of each `val.py`:

- `MODEL_PATH`
- `VISUALIZE_SELECT_MODE`
- `SELECTED_SAMPLE_IDX`
- `ANNEALED_TEMPERATURE`

The rendered figure is saved as `validation_result_idx_<N>.png` alongside the script.

### (d) Retrain from scratch (optional)

Checkpoints are already provided, but you can retrain if needed.

```bash
cd Classical && python train_pix2pix_from_npy.py
cd Regression && python train_pix2pix_from_npy.py
```

Training hyper-parameters live in the `TrainConfig` dataclass at the top of each training script. Defaults match the runs reported in the thesis:

- 20 epochs
- batch size 64
- image size 224×224
- ResNet18 + DynamicUnet backbone

---

## 7. Notes / Troubleshooting

**`FileNotFoundError: No AB npy files found`**

The dataset has not been placed under `Final_report_code/archive/` in the layout described in Section 4.

**`RuntimeError: CUDA error: no kernel image`**

The installed PyTorch wheel does not match your GPU/CUDA environment. Reinstall with the correct index URL and wheel version as described in Section 3.

**Very slow first run of the classical pipeline**

This is the one-off k-means computation used to generate `pts_in_hull.npy`. Subsequent runs are much faster.

**Out-of-memory on smaller GPUs**

Reduce `batch_size` in `TrainConfig`, or set:

```python
oom_auto_shrink_batch = True
```

to halve the batch size automatically on OOM.