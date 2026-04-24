================================================================
 Image Colorization with Pix2Pix (Classical + Regression)
 ABD Thesis Submission — KUANG ZIXUAN
================================================================

------------------------------------------------------------
1. CONTENTS OF THIS TAR FILE
------------------------------------------------------------
KUANG_ZIXUAN/
├── ABD_Thesis.pdf               The full thesis manuscript.
├── Code_Listing.pdf             Formatted source-code listing
                                  (appendix to the thesis).
├── readme.txt                   This file.
└── Final_report_code/           All source code and model
                                  checkpoints used in the thesis.
    │
    ├── requirements.txt         Python dependencies (pip).
    ├── benchmark_two_models.py  Runs both trained models on
    │                            the validation split and saves
    │                            rendered RGB images to disk.
    ├── benchmark_sweep.py       Sweeps the benchmark over
    │                            multiple sample counts and
    │                            plots avg-time-per-image.
    │
    ├── Classical/               Classification pipeline
    │                            (313-bin ab quantization,
    │                            cross-entropy + GAN).
    │   ├── train_pix2pix_from_npy.py   Training entry point.
    │   ├── load_lab_npy_data.py        Dataset/DataLoader.
    │   ├── val.py                      Visual validation of
    │                                   one random/indexed
    │                                   sample (incl. annealed-
    │                                   mean decoding).
    │   ├── (full)main-model.pt         Checkpoint: full run.
    │   └── (VL)main-model.pt           Checkpoint: best
    │                                   variant reported in
    │                                   the thesis.
    │
    ├── Regression/              Regression pipeline (direct
    │                            2-channel ab prediction, L1
    │                            + GAN, optional TV loss).
    │   ├── train_pix2pix_from_npy.py   Training entry point.
    │   ├── load_lab_npy_data.py        Dataset/DataLoader.
    │   ├── val.py                      Visual validation.
    │   ├── (low-lr)main-model.pt       Checkpoint: low-LR
    │                                   variant (default used
    │                                   by the benchmark).
    │   ├── (only-regression)main-model.pt   Regression-only
    │                                        (no GAN).
    │   └── (TV-loss)main-model.pt      Regression + TV loss.
    │
    └── archive/                 Dataset expected by the code
                                  (not redistributed — see §4).
        ├── l/gray_scale.npy          L-channel inputs.
        ├── ab/ab/ab1.npy             ab targets, part 1.
        ├── ab/ab/ab2.npy             ab targets, part 2.
        ├── ab/ab/ab3.npy             ab targets, part 3.
        └── pts_in_hull.npy           313 ab-bin centres (auto-
                                       generated on first run if
                                       missing; see §4).

------------------------------------------------------------
2. HARDWARE DEPENDENCIES
------------------------------------------------------------
Training:
  - A CUDA-capable NVIDIA GPU with >= 16 GB VRAM is strongly
    recommended (models were trained on an RTX A6000, 48 GB,
    with batch size 64 @ 224x224).
  - Training will also run on smaller GPUs by reducing
    `batch_size` in `TrainConfig` — see `oom_auto_shrink_batch`.
  - CPU-only training is technically possible but impractically
    slow and is not recommended.

Inference / validation / benchmark:
  - Works on any single CUDA GPU (>= 6 GB VRAM is sufficient
    at batch size 4–8).
  - Falls back to CPU automatically via `get_device()`; a
    full 500-image benchmark on CPU takes tens of minutes.

Disk:
  - ~2.5 GB for the provided model checkpoints.
  - ~2.6 GB for the gray_scale.npy + 3x ab*.npy dataset
    arrays (see §4 — not included in this tar).

------------------------------------------------------------
3. SOFTWARE DEPENDENCIES
------------------------------------------------------------
Operating system
  - Linux (developed/tested on Ubuntu 24.04, kernel 6.8).
  - macOS and Windows should work for CPU inference;
    GPU training on Windows requires a matching CUDA
    toolkit + NVIDIA driver.

Core
  - Python 3.10 or newer (3.10 / 3.11 tested).
  - pip 23+ (for --extra-index-url support).
  - NVIDIA driver >= 550 and a CUDA runtime compatible
    with the PyTorch wheel installed (see below).

Python packages (pinned in Final_report_code/requirements.txt)
  - torch           2.7.1 + cu128    (CUDA 12.8 / Blackwell build)
  - torchvision     0.22.1 + cu128
  - numpy           2.2.6
  - matplotlib      3.10.8
  - scikit-image    0.25.2
  - pillow          12.0.0
  - tqdm            4.67.3
  - fastai          2.8.7   (for DynamicUnet + ResNet18 backbone)
  - ipython         8.38.0  (implicit dep of fastai/fastprogress)

GPU-wheel selection
  The default wheels target CUDA 12.8, as required by
  NVIDIA Blackwell (RTX 50-series, sm_120). For other
  environments, edit requirements.txt as noted in its header:
    CUDA 11.8 : --extra-index-url .../whl/cu118
                torch==2.7.1+cu118  torchvision==0.22.1+cu118
    CUDA 12.1 : --extra-index-url .../whl/cu121
                torch==2.7.1+cu121  torchvision==0.22.1+cu121
    CPU only  : --extra-index-url .../whl/cpu
                torch==2.7.1        torchvision==0.22.1

No IDE is required. Any terminal with Python installed is
sufficient. (VS Code was used during development; a
`.vscode/` folder is included but is purely optional.)

------------------------------------------------------------
4. DATASET (NOT INCLUDED IN THE TAR)
------------------------------------------------------------
The code expects the "Image Colorization" LAB dataset
originally published on Kaggle:
  https://www.kaggle.com/datasets/shravankumar9892/image-colorization

After download, place the four .npy files into the
`Final_report_code/archive/` directory exactly as:
  Final_report_code/archive/l/gray_scale.npy
  Final_report_code/archive/ab/ab/ab1.npy
  Final_report_code/archive/ab/ab/ab2.npy
  Final_report_code/archive/ab/ab/ab3.npy

`pts_in_hull.npy` (the 313 ab-bin centres used by the
classification pipeline) does NOT need to be supplied:
`load_lab_npy_data.py` will generate it automatically by
k-means on the first run if it is missing, and cache it
to `archive/pts_in_hull.npy`.

------------------------------------------------------------
5. HOW TO INSTALL
------------------------------------------------------------
The code is pure Python — no compilation step is required.

  # 1. (Recommended) create an isolated environment
  python3 -m venv .venv
  source .venv/bin/activate          # Linux / macOS
  # .venv\Scripts\activate           # Windows PowerShell

  # 2. Upgrade pip
  pip install --upgrade pip

  # 3. Install dependencies
  cd Final_report_code
  pip install -r requirements.txt

  # 4. (If not using CUDA 12.8) edit requirements.txt per §3
  #    before running step 3.

A quick sanity check:
  python -c "import torch; print(torch.__version__, \
             'cuda=', torch.cuda.is_available())"

------------------------------------------------------------
6. HOW TO RUN
------------------------------------------------------------
All commands below assume the working directory is
`Final_report_code/`.

(a) Reproduce the side-by-side benchmark used in the thesis
    (renders 500 validation images with each of the two
    reported models, prints timings):

      python benchmark_two_models.py

    Outputs:
      test1/img_000.png ... img_499.png   (Regression model)
      test2/img_000.png ... img_499.png   (Classical model)

(b) Reproduce the throughput-vs-sample-count sweep figure:

      python benchmark_sweep.py

    Outputs:
      benchmark_sweep.png

(c) Visualize a single validation sample (gray input,
    prediction, and ground truth shown side-by-side):

      # Classification model with annealed-mean decoding
      cd Classical && python val.py && cd ..

      # Regression model
      cd Regression && python val.py && cd ..

    The target checkpoint and sample selection mode are
    configured at the top of each `val.py` (`MODEL_PATH`,
    `VISUALIZE_SELECT_MODE`, `SELECTED_SAMPLE_IDX`,
    `ANNEALED_TEMPERATURE`). The rendered figure is saved
    as `validation_result_idx_<N>.png` alongside the script.

(d) Retrain from scratch (optional — checkpoints are
    already provided):

      cd Classical && python train_pix2pix_from_npy.py
      cd Regression && python train_pix2pix_from_npy.py

    Training hyper-parameters live in the `TrainConfig`
    dataclass at the top of each training script. Defaults
    match the runs reported in the thesis (20 epochs, batch
    64, 224x224, ResNet18+DynamicUnet backbone).

------------------------------------------------------------
7. NOTES / TROUBLESHOOTING
------------------------------------------------------------
- "FileNotFoundError: No AB npy files found" — the dataset
  has not been placed under `Final_report_code/archive/`
  in the layout described in §4.
- "RuntimeError: CUDA error: no kernel image" — the torch
  wheel installed does not match your GPU/CUDA. Re-install
  with the matching index URL (see §3).
- Very slow first run of the classical pipeline — this is
  the one-off k-means computation that produces
  `pts_in_hull.npy`. Subsequent runs are instant.
- On smaller GPUs, either reduce `batch_size` in
  `TrainConfig`, or set `oom_auto_shrink_batch = True` to
  halve the batch size automatically on OOM.

------------------------------------------------------------
8. CONTACT
------------------------------------------------------------
Author : KUANG ZIXUAN
Thesis : See ABD_Thesis.pdf in the root of this archive.
