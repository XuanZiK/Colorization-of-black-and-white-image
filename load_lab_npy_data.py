"""
Only use 
Load paired LAB training data from local .npy files for pix2pix colorization.

This script is designed for the current project structure:
- L channel:  /home/xzk/thesis/archive/l/gray_scale.npy
- ab channel: /home/xzk/thesis/archive/ab/ab/*.npy

Output sample format is compatible with training code that expects:
- data['L']  -> shape (B, 1, H, W), range around [-1, 1]
- data['ab'] -> shape (B, 2, H, W), range around [-1, 1]
"""

# ============================== 1) Imports ===============================
import glob
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ============================== 2) Config ================================
@dataclass
class DataConfig:
    gray_scale_path: str = "/home/xzk/thesis/archive/l/gray_scale.npy"
    ab_glob: str = "/home/xzk/thesis/archive/ab/ab/*.npy"
    color_bins_path: str = "/home/xzk/thesis/archive/pts_in_hull.npy"

    external_data_size: int = 25000
    train_size: int = 20000
    batch_size: int = 32

    use_mmap: bool = True
    num_workers: int = 0
    pin_memory: bool = True
    random_seed: int = 42


# ======================== 3) File Loading Helpers ========================
def _load_npy(path: str, use_mmap: bool):
    mode = "r" if use_mmap else None
    return np.load(path, mmap_mode=mode)


def load_local_lab_data(cfg: DataConfig):
    """Load gray and ab memmaps and do basic consistency checks."""
    gray_data = _load_npy(cfg.gray_scale_path, cfg.use_mmap)

    ab_paths = sorted(glob.glob(cfg.ab_glob))
    if not ab_paths:
        raise FileNotFoundError(f"No AB npy files found: {cfg.ab_glob}")

    ab_data_parts = [_load_npy(p, cfg.use_mmap) for p in ab_paths]

    ab_total = sum(arr.shape[0] for arr in ab_data_parts)
    num_samples = min(gray_data.shape[0], ab_total)

    if num_samples < 2:
        raise ValueError("Not enough paired samples to build train/val loaders.")

    print("=" * 70)
    print("[Data Overview]")
    print(f"gray shape: {gray_data.shape}, dtype: {gray_data.dtype}")
    print(f"ab chunks: {len(ab_data_parts)}, total samples: {ab_total}")
    print(f"usable paired samples: {num_samples}")
    print("=" * 70)

    return gray_data, ab_data_parts, num_samples


def load_color_bins(cfg: DataConfig, ab_data_parts: List[np.ndarray]) -> np.ndarray:
    """Load 313 LAB color bins from file or estimate a fallback from AB samples."""
    color_bins_path = getattr(cfg, "color_bins_path", "")
    if color_bins_path and os.path.exists(color_bins_path):
        bins = np.load(color_bins_path).astype(np.float32)
        if bins.ndim != 2 or bins.shape[1] != 2:
            raise ValueError(f"Invalid color bins shape: {bins.shape}")
        return bins

    sampled_chunks = []
    max_pixels = 200000
    remaining = max_pixels

    for part in ab_data_parts:
        if remaining <= 0:
            break

        arr = np.asarray(part)
        if arr.ndim != 4:
            continue

        if arr.shape[1] == 2:
            flat = np.transpose(arr, (0, 2, 3, 1)).reshape(-1, 2)
        elif arr.shape[-1] == 2:
            flat = arr.reshape(-1, 2)
        else:
            continue

        take = min(len(flat), remaining)
        if take <= 0:
            continue

        sampled_chunks.append(flat[:take])
        remaining -= take

    if not sampled_chunks:
        raise ValueError("Unable to estimate color bins from AB data")

    bins = np.concatenate(sampled_chunks, axis=0).astype(np.float32)
    bins_min = float(bins.min())
    bins_max = float(bins.max())

    if bins_max <= 1.5 and bins_min >= -1.5:
        bins = bins * 128.0
    elif bins_min >= 0.0 and bins_max <= 255.0:
        bins = bins - 128.0

    if len(bins) < 313:
        repeat = int(np.ceil(313 / max(len(bins), 1)))
        bins = np.tile(bins, (repeat, 1))

    step = max(len(bins) // 313, 1)
    bins = bins[::step][:313].astype(np.float32)
    return bins


# =========================== 4) Index Splitting ===========================
def build_train_val_indices(cfg: DataConfig, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Build shuffled train/val indices over paired samples."""
    max_samples = min(cfg.external_data_size, num_samples)
    if max_samples < 2:
        raise ValueError("max_samples < 2; cannot split train/val")

    rng = np.random.default_rng(cfg.random_seed)
    chosen = rng.choice(num_samples, size=max_samples, replace=False)
    perm = rng.permutation(max_samples)

    train_count = min(cfg.train_size, max_samples - 1)
    train_indices = chosen[perm[:train_count]]
    val_indices = chosen[perm[train_count:]]

    if len(val_indices) == 0:
        raise ValueError("Validation set is empty. Reduce train_size or increase external_data_size.")

    print(f"train samples: {len(train_indices)} | val samples: {len(val_indices)}")
    return train_indices, val_indices


# ============================= 5) Dataset ================================
class NpyLabDataset(Dataset):
    """Load paired L/ab samples from local npy files."""

    def __init__(self, gray_memmap, ab_memmaps: List[np.ndarray], indices: np.ndarray, train: bool = True):
        self.gray = gray_memmap
        self.ab_parts = ab_memmaps
        self.indices = np.asarray(indices)
        self.train = train

        lengths = [arr.shape[0] for arr in self.ab_parts]
        self.ab_cumsum = np.cumsum(lengths)

    def __len__(self) -> int:
        return len(self.indices)

    def _get_ab_by_global_idx(self, global_idx: int) -> np.ndarray:
        chunk_id = int(np.searchsorted(self.ab_cumsum, global_idx, side="right"))
        prev = 0 if chunk_id == 0 else int(self.ab_cumsum[chunk_id - 1])
        local_idx = global_idx - prev
        return self.ab_parts[chunk_id][local_idx]

    def _normalize_l(self, l_array: np.ndarray) -> torch.Tensor:
        # L is expected from gray uint8 [0,255], mapped to [-1,1] for the model.
        L = torch.from_numpy(l_array.astype(np.float32)).unsqueeze(0)
        return (L / 255.0) * 2.0 - 1.0

    def _normalize_ab(self, ab_array: np.ndarray) -> torch.Tensor:
        ab = torch.from_numpy(ab_array.astype(np.float32))

        if ab.ndim == 3 and ab.shape[0] == 2:
            pass
        elif ab.ndim == 3 and ab.shape[-1] == 2:
            ab = ab.permute(2, 0, 1)
        else:
            raise ValueError(f"Unsupported ab shape: {tuple(ab.shape)}")

        # Compatible normalization for common storage formats:
        # - real LAB ab in [-128, 127] -> divide by 128
        # - uint8-like [0, 255]         -> map to [-1, 1]
        ab_min = float(ab.min())
        ab_max = float(ab.max())

        if ab_max > 1.5 or ab_min < -1.5:
            if ab_min >= 0.0 and ab_max <= 255.0:
                ab = (ab / 255.0) * 2.0 - 1.0
            else:
                ab = ab / 128.0

        return torch.clamp(ab, -1.0, 1.0)

    def __getitem__(self, idx: int):
        gidx = int(self.indices[idx])

        L = self._normalize_l(self.gray[gidx])
        ab = self._normalize_ab(self._get_ab_by_global_idx(gidx))

        # Paired augmentation: flip both L and ab together.
        if self.train and torch.rand(1).item() < 0.5:
            L = torch.flip(L, dims=[2])
            ab = torch.flip(ab, dims=[2])

        return {"L": L, "ab": ab}


# ========================= 6) DataLoader Builder ==========================
def build_dataloaders(cfg: DataConfig):
    gray_data, ab_data_parts, num_samples = load_local_lab_data(cfg)
    train_indices, val_indices = build_train_val_indices(cfg, num_samples)

    train_data = NpyLabDataset(
        gray_memmap=gray_data,
        ab_memmaps=ab_data_parts,
        indices=train_indices,
        train=True,
    )
    valid_data = NpyLabDataset(
        gray_memmap=gray_data,
        ab_memmaps=ab_data_parts,
        indices=val_indices,
        train=False,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=cfg.pin_memory,
        num_workers=cfg.num_workers,
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=cfg.batch_size,
        shuffle=False,
        pin_memory=cfg.pin_memory,
        num_workers=cfg.num_workers,
    )

    return train_loader, valid_loader


# =========================== 7) Quick Smoke Test ==========================
def main():
    cfg = DataConfig()
    train_loader, valid_loader = build_dataloaders(cfg)

    sample_batch = next(iter(train_loader))
    print("=" * 70)
    print("[Batch Check]")
    print("L batch shape:", sample_batch["L"].shape, "range:",
          (sample_batch["L"].min().item(), sample_batch["L"].max().item()))
    print("ab batch shape:", sample_batch["ab"].shape, "range:",
          (sample_batch["ab"].min().item(), sample_batch["ab"].max().item()))
    print(f"train batches: {len(train_loader)} | valid batches: {len(valid_loader)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
