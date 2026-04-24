import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

from train_pix2pix_from_npy import (
    TrainConfig, 
    build_generator, 
    get_device, 
    set_seed,
    lab_to_rgb,
)
from load_lab_npy_data import DataConfig, build_dataloaders, load_local_lab_data, load_color_bins

# ================= 配置区域 =================
# MODEL_PATH = "/home/xzk/thesis/classical/(full)main-model.pt"
# MODEL_PATH = "/home/xzk/thesis/classical/(VL)main-model.pt"
# MODEL_PATH = "(only-regression)main-model.pt"    
MODEL_PATH = "(low-lr)main-model.pt"    
# MODEL_PATH = "(TV-loss)main-model.pt"      
 
  
SAVE_FIG_PREFIX = "validation_result_idx" 
VISUALIZE_SELECT_MODE = "random"  # 可选: "random" 或 "by_index"
SELECTED_SAMPLE_IDX = 0            # 当模式为 by_index 时生效
RANDOM_SEED = 42
BATCH_SIZE_FOR_VAL = 4
# ===========================================

def load_batch_for_visualization(data_loader, device, select_mode="random", selected_idx=0):
    dataset = data_loader.dataset
    if select_mode == "random":
        idx = random.randint(0, len(dataset) - 1)
        print(f"🎲 随机选中的样本索引: {idx}")
    elif select_mode == "by_index":
        idx = int(selected_idx)
        if idx < 0 or idx >= len(dataset):
            raise ValueError(f"SELECTED_SAMPLE_IDX 超出范围: {idx}, 合法范围 [0, {len(dataset)-1}]")
        print(f"🎯 按指定索引选中样本: {idx}")
    else:
        raise ValueError("VISUALIZE_SELECT_MODE 仅支持 'random' 或 'by_index'")
    
    sample = dataset[idx]
    L = sample['L'].unsqueeze(0).to(device)
    ab = sample['ab'].unsqueeze(0).to(device)
    
    return {'L': L, 'ab': ab}, idx

def fix_state_dict_keys(state_dict):

    new_state_dict = {}
    count = 0
    for k, v in state_dict.items():
        if k.startswith('generator.'):
            new_key = k[len('generator.'):] 
            new_state_dict[new_key] = v
            count += 1
        elif not k.startswith('discriminator.') and not k.startswith('GANloss.'):
            new_state_dict[k] = v
            count += 1
            
    print(f"🔧 已处理权重键: 原始 {len(state_dict)} 个 -> 提取生成器权重 {count} 个")
    return new_state_dict


def infer_output_channels(state_dict) -> int:
    """Infer generator output channels from checkpoint's final 1x1 conv weight."""
    preferred_keys = ["layers.12.0.weight", "layers.10.0.weight", "final_conv.weight"]
    for k in preferred_keys:
        if k in state_dict and state_dict[k].ndim == 4:
            return int(state_dict[k].shape[0])

    candidates = []
    for k, v in state_dict.items():
        if k.endswith(".weight") and isinstance(v, torch.Tensor) and v.ndim == 4 and v.shape[2:] == (1, 1):
            candidates.append((k, int(v.shape[0]), int(v.shape[1])))

    for _, out_c, _ in candidates:
        if out_c in (2, 313):
            return out_c

    if candidates:
        candidates.sort(key=lambda x: x[1])
        return candidates[0][1]

    raise RuntimeError("无法从 checkpoint 推断生成器输出通道数")


def load_color_bins_tensor(path: str, device: torch.device) -> torch.Tensor:
    """Local helper for decoding old classification checkpoints in validation."""
    cfg = DataConfig(color_bins_path=path)
    _, ab_data_parts, _ = load_local_lab_data(cfg)
    bins = load_color_bins(cfg, ab_data_parts).astype(np.float32)
    return torch.from_numpy(bins).to(device)


def class_idx_to_ab(class_idx: torch.Tensor, color_bins_lab: torch.Tensor) -> torch.Tensor:
    """Map class index map (B,H,W) to normalized ab map (B,2,H,W) in [-1,1]."""
    b, h, w = class_idx.shape
    flat = class_idx.reshape(-1).long()
    ab_lab = color_bins_lab.index_select(0, flat).reshape(b, h, w, 2)
    ab_lab = ab_lab.permute(0, 3, 1, 2).contiguous()
    return torch.clamp(ab_lab / 128.0, -1.0, 1.0)


def logits_to_ab(logits: torch.Tensor, color_bins_lab: torch.Tensor) -> torch.Tensor:
    class_idx = torch.argmax(logits, dim=1)
    return class_idx_to_ab(class_idx, color_bins_lab)

def main():
    set_seed(RANDOM_SEED)
    device = get_device()
    print(f"🚀 使用设备: {device}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误: 找不到模型文件 '{MODEL_PATH}'")
        return

    base_cfg = TrainConfig(
        image_size_1=224,
        image_size_2=224,
        use_backbone_pretrain=True,
    )
    
    data_cfg = DataConfig(
        external_data_size=25000,
        train_size=20000,
        batch_size=BATCH_SIZE_FOR_VAL,
        num_workers=0,
        pin_memory=False
    )
    
    print("📂 正在加载数据集...")
    _, valid_loader = build_dataloaders(data_cfg)
    print(f"✅ 数据集加载完成。总样本数: {len(valid_loader.dataset)}")
    
    print(f"📥 正在加载并修复权重: {MODEL_PATH} ...")
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        
        if any(k.startswith('generator.') for k in checkpoint.keys()):
            print("   检测到 'generator.' 前缀，正在提取并清洗权重...")
            cleaned_dict = fix_state_dict_keys(checkpoint)
            
        else:
            print("   检测到干净的权重，直接加载...")

            cleaned_dict = checkpoint

        out_channels = infer_output_channels(cleaned_dict)
        mode = "regression" if out_channels == 2 else "classification" if out_channels == 313 else f"unknown({out_channels})"
        print(f"🧠 检测到权重输出通道数: {out_channels} ({mode})")

        cfg = TrainConfig(
            image_size_1=224,
            image_size_2=224,
            use_backbone_pretrain=True,
            output_channels=out_channels,
        )

        print(f"🏗️  正在按 out_channels={out_channels} 构建生成器...")
        generator = build_generator(cfg, device)
        generator.load_state_dict(cleaned_dict, strict=True)

        color_bins_lab = None
        if out_channels == 313:
            color_bins_lab = load_color_bins_tensor(data_cfg.color_bins_path, device)
            
        print("✅ 模型权重加载成功！")
            
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        print("💡 提示: 请检查模型架构定义是否与训练时完全一致。")
        return

    generator.eval()
    print("✨ 模型准备就绪 (Eval Mode)。")

    batch_data, sample_idx = load_batch_for_visualization(
        valid_loader,
        device,
        select_mode=VISUALIZE_SELECT_MODE,
        selected_idx=SELECTED_SAMPLE_IDX,
    )
    L_input = batch_data['L']
    ab_real = batch_data['ab']

    print("🎨 正在进行推理...")
    with torch.no_grad():
        output = generator(L_input)
        if output.shape[1] == 2:
            ab_pred = torch.clamp(output, -1.0, 1.0)
            pred_title = "Predicted Color (Regression, 2ch)"
        else:
            ab_pred = logits_to_ab(output, color_bins_lab)
            pred_title = "Predicted Color (Classification, 313ch)"
    
    # 可视化结果
    L_cpu = L_input.cpu()
    ab_pred_cpu = ab_pred.cpu()
    ab_real_cpu = ab_real.cpu()
    
    rgb_fake = lab_to_rgb(L_cpu, ab_pred_cpu)[0]
    rgb_real = lab_to_rgb(L_cpu, ab_real_cpu)[0]
    gray_img = ((L_cpu[0, 0] + 1) / 2).cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(gray_img, cmap='gray')
    axes[0].set_title(f"Input Gray (Idx: {sample_idx})", fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(rgb_fake)
    axes[1].set_title(pred_title, fontsize=14)
    axes[1].axis('off')
    
    axes[2].imshow(rgb_real)
    axes[2].set_title("Ground Truth (Real)", fontsize=14)
    axes[2].axis('off')
    
    save_fig_path = f"{SAVE_FIG_PREFIX}_{sample_idx}.png"
    plt.tight_layout()
    plt.savefig(save_fig_path)
    print(f"💾 验证结果已保存至: {os.path.abspath(save_fig_path)}")
    plt.show()

if __name__ == "__main__":
    main()