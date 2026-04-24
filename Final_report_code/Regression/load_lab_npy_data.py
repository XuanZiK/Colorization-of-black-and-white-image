# ============================== 1) Imports ===============================
import glob
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ============================== 2) Config ================================
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ARCHIVE_DIR = os.path.join(_PROJECT_ROOT, "archive")


@dataclass
class DataConfig:
    gray_scale_path: str = os.path.join(_ARCHIVE_DIR, "l", "gray_scale.npy")
    ab_glob: str = os.path.join(_ARCHIVE_DIR, "ab", "ab", "*.npy")
    color_bins_path: str = os.path.join(_ARCHIVE_DIR, "pts_in_hull.npy")
    color_bin_count: int = 313
    auto_create_color_bins: bool = True
    color_bins_sample_points: int = 200000
    color_bins_kmeans_iters: int = 20

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


def _ab_to_chw_and_lab_range(ab_array: np.ndarray) -> np.ndarray:
    """Convert ab sample to CHW in LAB coordinate range (about [-128,127])."""
    ab = ab_array.astype(np.float32)
    if ab.ndim == 3 and ab.shape[0] == 2:
        pass
    elif ab.ndim == 3 and ab.shape[-1] == 2:
        ab = np.transpose(ab, (2, 0, 1))
    else:
        raise ValueError(f"Unsupported ab shape: {tuple(ab.shape)}")

    ab_min = float(ab.min())
    ab_max = float(ab.max())
    if ab_min >= 0.0 and ab_max <= 255.0:
        ab = ab - 128.0
    elif ab_min >= -1.5 and ab_max <= 1.5:
        ab = ab * 128.0
    return np.clip(ab, -128.0, 127.0)


def _sample_ab_points(ab_data_parts: List[np.ndarray], sample_points: int, seed: int) -> np.ndarray:
    """Randomly sample ab pixels from dataset for color bin estimation."""
    if sample_points <= 0:
        raise ValueError("color_bins_sample_points must be > 0")

    rng = np.random.default_rng(seed)
    lengths = [arr.shape[0] for arr in ab_data_parts]
    total_imgs = int(sum(lengths))
    if total_imgs == 0:
        raise ValueError("No AB images available for color bins generation")

    cumsum = np.cumsum(lengths)

    def get_ab_by_global_idx(global_idx: int) -> np.ndarray:
        chunk_id = int(np.searchsorted(cumsum, global_idx, side="right"))
        prev = 0 if chunk_id == 0 else int(cumsum[chunk_id - 1])
        local_idx = global_idx - prev
        return ab_data_parts[chunk_id][local_idx]

    points = []
    target = sample_points
    # 每张图采样固定像素，减少 I/O 次数。
    pixels_per_img = 256
    needed_imgs = max(1, target // pixels_per_img)
    img_indices = rng.integers(0, total_imgs, size=needed_imgs)

    for gidx in img_indices:
        ab = _ab_to_chw_and_lab_range(get_ab_by_global_idx(int(gidx)))
        h, w = ab.shape[1], ab.shape[2]
        n = min(pixels_per_img, h * w)
        choose = rng.choice(h * w, size=n, replace=False)
        flat = np.transpose(ab, (1, 2, 0)).reshape(-1, 2)
        points.append(flat[choose])

        if sum(p.shape[0] for p in points) >= target:
            break

    sampled = np.concatenate(points, axis=0)[:target]
    return sampled.astype(np.float32)


def _kmeans(points: np.ndarray, k: int, iters: int, seed: int) -> np.ndarray:
    """Simple numpy kmeans for generating color bins."""
    if points.shape[0] < k:
        raise ValueError(f"Not enough sampled points ({points.shape[0]}) for k={k}")

    rng = np.random.default_rng(seed)
    centers = points[rng.choice(points.shape[0], size=k, replace=False)].copy()

    for _ in range(max(1, iters)):
        dists = np.sum((points[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        assign = np.argmin(dists, axis=1)

        for ci in range(k):
            mask = assign == ci
            if np.any(mask):
                centers[ci] = points[mask].mean(axis=0)
            else:
                centers[ci] = points[rng.integers(0, points.shape[0])]

    return np.clip(centers.astype(np.float32), -128.0, 127.0)


def load_color_bins(cfg: DataConfig, ab_data_parts: List[np.ndarray]) -> np.ndarray:
    """Load existing color bins, or auto-create them from AB data if missing."""
    path = cfg.color_bins_path
    if not path:
        raise ValueError("color_bins_path is empty")

    if os.path.exists(path):
        bins = np.load(path)
        if bins.ndim != 2 or bins.shape[1] != 2:
            raise ValueError(f"pts_in_hull.npy should have shape (N,2), got {bins.shape}")
        if bins.shape[0] != cfg.color_bin_count:
            raise ValueError(f"Expected {cfg.color_bin_count} color bins, got {bins.shape[0]}")
        return bins.astype(np.float32)

    if not cfg.auto_create_color_bins:
        raise FileNotFoundError(
            f"Color bins file not found: {path}. "
            "Set auto_create_color_bins=True to estimate bins from AB data."
        )

    print(f"[Warn] 未找到颜色中心文件: {path}")
    print("[Info] 正在从 AB 数据自动估计 313 个颜色中心（一次性耗时操作）...")
    points = _sample_ab_points(ab_data_parts, cfg.color_bins_sample_points, cfg.random_seed)
    bins = _kmeans(points, cfg.color_bin_count, cfg.color_bins_kmeans_iters, cfg.random_seed)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, bins)
    print(f"[Done] 已自动生成颜色中心并保存: {path}")
    return bins


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

    def __init__(
        self,
        gray_memmap,
        ab_memmaps: List[np.ndarray],
        indices: np.ndarray,
        train: bool = True,
    ):
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

    def _to_ab_lab_range(self, ab_array: np.ndarray) -> np.ndarray:
        """Convert ab arrays to LAB coordinate range (about [-128, 127])."""
        return _ab_to_chw_and_lab_range(ab_array)

    def _normalize_ab(self, ab_array: np.ndarray) -> torch.Tensor:
        """Normalize ab arrays to the regression target range [-1, 1]."""
        ab_lab = self._to_ab_lab_range(ab_array)
        ab = torch.from_numpy(ab_lab.astype(np.float32)) / 128.0
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
