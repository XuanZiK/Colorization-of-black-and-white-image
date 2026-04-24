import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import numpy as np
from PIL import Image

# Ensure local project modules in subfolders are importable.
PROJECT_ROOT = Path(__file__).resolve().parent
REGRESSION_DIR = str(PROJECT_ROOT / "Regression")
CLASSICAL_DIR = str(PROJECT_ROOT / "Classical")
for p in (REGRESSION_DIR, CLASSICAL_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from train_pix2pix_from_npy import (
    TrainConfig,
    build_generator,
    get_device,
    set_seed,
    lab_to_rgb,
)
from load_lab_npy_data import (
    DataConfig,
    build_dataloaders,
    load_local_lab_data,
    load_color_bins,
)

# Models to benchmark: (save_dir will hold rendered RGB images)
MODEL_RUNS = [
    {"name": "Regression", "path": "Regression/(low-lr)main-model.pt", "save_dir": "test1"},
    {"name": "Classical", "path": "Classical/(VL)main-model.pt", "save_dir": "test2"},
]

TOTAL_SAMPLES = 500  # default fallback; overridable via arguments
BATCH_SIZE_FOR_BENCH = 8
RANDOM_SEED = 42


def fix_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Strip optional generator.* prefix and drop non-generator keys."""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("generator."):
            new_state_dict[k[len("generator."):]] = v
        elif not k.startswith("discriminator.") and not k.startswith("GANloss."):
            new_state_dict[k] = v
    return new_state_dict


def infer_output_channels(state_dict: Dict[str, torch.Tensor]) -> int:
    preferred_keys = ["layers.12.0.weight", "layers.10.0.weight", "final_conv.weight"]
    for k in preferred_keys:
        if k in state_dict and state_dict[k].ndim == 4:
            return int(state_dict[k].shape[0])

    candidates = []
    for k, v in state_dict.items():
        if k.endswith(".weight") and isinstance(v, torch.Tensor) and v.ndim == 4 and v.shape[2:] == (1, 1):
            candidates.append((k, int(v.shape[0])))

    for _, out_c in candidates:
        if out_c in (2, 313):
            return out_c

    if candidates:
        candidates.sort(key=lambda x: x[1])
        return candidates[0][1]

    raise RuntimeError("Cannot infer output channels from checkpoint")


def load_color_bins_tensor(path: str, device: torch.device) -> torch.Tensor:
    cfg = DataConfig(color_bins_path=path)
    _, ab_data_parts, _ = load_local_lab_data(cfg)
    bins = load_color_bins(cfg, ab_data_parts).astype(np.float32)
    return torch.from_numpy(bins).to(device)


def class_idx_to_ab(class_idx: torch.Tensor, color_bins_lab: torch.Tensor) -> torch.Tensor:
    b, h, w = class_idx.shape
    flat = class_idx.reshape(-1).long()
    ab_lab = color_bins_lab.index_select(0, flat).reshape(b, h, w, 2)
    ab_lab = ab_lab.permute(0, 3, 1, 2).contiguous()
    return torch.clamp(ab_lab / 128.0, -1.0, 1.0)


def logits_to_ab(logits: torch.Tensor, color_bins_lab: torch.Tensor) -> torch.Tensor:
    class_idx = torch.argmax(logits, dim=1)
    return class_idx_to_ab(class_idx, color_bins_lab)


def save_rgb_image(rgb: np.ndarray, path: Path) -> None:
    rgb_uint8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(rgb_uint8).save(path)


def prepare_generator(model_path: str, device: torch.device, data_cfg: DataConfig):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    cleaned_dict = fix_state_dict_keys(checkpoint) if any(k.startswith("generator.") for k in checkpoint.keys()) else checkpoint

    out_channels = infer_output_channels(cleaned_dict)
    cfg = TrainConfig(
        image_size_1=224,
        image_size_2=224,
        use_backbone_pretrain=True,
        output_channels=out_channels,
    )

    generator = build_generator(cfg, device)
    generator.load_state_dict(cleaned_dict, strict=True)
    generator.eval()

    color_bins_lab = None
    if out_channels == 313:
        color_bins_lab = load_color_bins_tensor(data_cfg.color_bins_path, device)

    return generator, color_bins_lab, out_channels


def run_benchmark(model_path: str, output_dir: Path, device: torch.device, data_cfg: DataConfig, valid_loader, total_samples: int) -> Tuple[int, float, float]:
    generator, color_bins_lab, out_channels = prepare_generator(model_path, device, data_cfg)
    os.makedirs(output_dir, exist_ok=True)

    total = min(total_samples, len(valid_loader.dataset))
    processed = 0
    start = time.perf_counter()

    with torch.no_grad():
        for batch in valid_loader:
            L = batch["L"].to(device)
            output = generator(L)
            if out_channels == 2:
                ab_pred = torch.clamp(output, -1.0, 1.0)
            else:
                ab_pred = logits_to_ab(output, color_bins_lab)

            rgb_batch = lab_to_rgb(L.cpu(), ab_pred.cpu())
            for i in range(rgb_batch.shape[0]):
                if processed >= total:
                    break
                save_rgb_image(rgb_batch[i], output_dir / f"img_{processed:03d}.png")
                processed += 1

            if processed >= total:
                break

    elapsed = time.perf_counter() - start
    avg = elapsed / processed if processed else float("inf")
    return processed, elapsed, avg


def benchmark_models(total_samples: int = TOTAL_SAMPLES):
    set_seed(RANDOM_SEED)
    device = get_device()

    data_cfg = DataConfig(
        external_data_size=25000,
        train_size=20000,
        batch_size=BATCH_SIZE_FOR_BENCH,
        num_workers=0,
        pin_memory=False,
    )
    _, valid_loader = build_dataloaders(data_cfg)

    results = []
    for item in MODEL_RUNS:
        processed, elapsed, avg = run_benchmark(
            model_path=item["path"],
            output_dir=Path(item["save_dir"]),
            device=device,
            data_cfg=data_cfg,
            valid_loader=valid_loader,
            total_samples=total_samples,
        )
        results.append((item["name"], processed, elapsed, avg, item["save_dir"]))

    print("\n========== Benchmark Summary ==========")
    for name, processed, elapsed, avg, save_dir in results:
        print(f"Model: {name}")
        print(f"  Samples rendered: {processed}")
        print(f"  Total time: {elapsed:.2f} s")
        print(f"  Avg time per image: {avg:.4f} s")
        print(f"  Saved to: {save_dir}\n")

    return results


def main(total_samples: int = TOTAL_SAMPLES):
    benchmark_models(total_samples=total_samples)


if __name__ == "__main__":
    main()
