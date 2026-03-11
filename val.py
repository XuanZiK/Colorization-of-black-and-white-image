"""
验证脚本 (修复版): 
1. 自动去除 saved state_dict 中的 'generator.' 前缀。
2. 兼容 MainModel 保存格式。
3. 自动兼容两种输出头：
    - 回归模型: 2 通道 (直接输出 ab)
    - 分类模型: 313 通道 (argmax + pts_in_hull 查表还原 ab)
4. 从 npy 数据中随机抽取一张图进行上色验证。
"""

import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

# 导入训练脚本中的配置、模型构建函数和数据加载配置
# 确保本文件与 train_pix2pix_from_npy.py 在同一目录
from train_pix2pix_from_npy import (
    TrainConfig, 
    build_generator, 
    get_device, 
    set_seed,
    lab_to_rgb,
    load_color_bins_tensor,
    class_idx_to_ab,
    logits_to_ab,
)
from load_lab_npy_data import DataConfig, build_dataloaders

# ================= 配置区域 =================    
MODEL_PATH = "(VL)main-model.pt"       
SAVE_FIG_PATH = "validation_result.png" 
RANDOM_SEED = 42
BATCH_SIZE_FOR_VAL = 4
# ===========================================

def load_random_batch(data_loader, device):
    dataset = data_loader.dataset
    idx = random.randint(0, len(dataset) - 1)
    print(f"🎲 随机选中的样本索引: {idx}")
    
    sample = dataset[idx]
    L = sample['L'].unsqueeze(0).to(device)
    ab = sample['ab'].unsqueeze(0).to(device)
    
    return {'L': L, 'ab': ab}, idx

def fix_state_dict_keys(state_dict):
    """
    修复 state_dict 键名：
    1. 如果键以 'generator.' 开头，去掉它。
    2. 过滤掉不属于 generator 的键 (如 discriminator.*, GANloss.*)。
    """
    new_state_dict = {}
    count = 0
    for k, v in state_dict.items():
        if k.startswith('generator.'):
            new_key = k[len('generator.'):] # 去掉前缀
            new_state_dict[new_key] = v
            count += 1
        elif not k.startswith('discriminator.') and not k.startswith('GANloss.'):
            # 如果没有前缀且不是其他组件，直接保留 (以防万一)
            new_state_dict[k] = v
            count += 1
            
    print(f"🔧 已处理权重键: 原始 {len(state_dict)} 个 -> 提取生成器权重 {count} 个")
    return new_state_dict


def infer_output_channels(state_dict) -> int:
    """Infer generator output channels from checkpoint's final 1x1 conv weight."""
    # DynamicUnet 常见最终层键名。
    preferred_keys = ["layers.12.0.weight", "layers.10.0.weight", "final_conv.weight"]
    for k in preferred_keys:
        if k in state_dict and state_dict[k].ndim == 4:
            return int(state_dict[k].shape[0])

    # 回退：扫描所有 1x1 conv 权重，优先返回 2 或 313。
    candidates = []
    for k, v in state_dict.items():
        if k.endswith(".weight") and isinstance(v, torch.Tensor) and v.ndim == 4 and v.shape[2:] == (1, 1):
            candidates.append((k, int(v.shape[0]), int(v.shape[1])))

    for _, out_c, _ in candidates:
        if out_c in (2, 313):
            return out_c

    if candidates:
        # 通常最后层 out_channels 最小，这里保守取最小值。
        candidates.sort(key=lambda x: x[1])
        return candidates[0][1]

    raise RuntimeError("无法从 checkpoint 推断生成器输出通道数")

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
        
        # 情况 A: 加载的是整个 MainModel 的 state_dict (带有 generator. 前缀)
        if any(k.startswith('generator.') for k in checkpoint.keys()):
            print("   检测到 'generator.' 前缀，正在提取并清洗权重...")
            cleaned_dict = fix_state_dict_keys(checkpoint)
            
        # 情况 B: 加载的已经是干净的 generator state_dict
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
            color_bins_path=base_cfg.color_bins_path,
        )

        print(f"🏗️  正在按 out_channels={out_channels} 构建生成器...")
        generator = build_generator(cfg, device)
        generator.load_state_dict(cleaned_dict, strict=True)

        color_bins_lab = load_color_bins_tensor(cfg.color_bins_path, device)
            
        print("✅ 模型权重加载成功！")
            
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        print("💡 提示: 请检查模型架构定义是否与训练时完全一致。")
        return

    generator.eval()
    print("✨ 模型准备就绪 (Eval Mode)。")

    # 获取随机数据并推理
    batch_data, sample_idx = load_random_batch(valid_loader, device)
    L_input = batch_data['L']
    ab_real = batch_data['ab']

    print("🎨 正在进行推理...")
    with torch.no_grad():
        output = generator(L_input)
        if output.shape[1] == 2:
            # 旧回归模型直接输出 ab
            ab_pred = torch.clamp(output, -1.0, 1.0)
            pred_title = "Predicted Color (Regression, 2ch)"
        else:
            # 新分类模型输出 logits
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
    
    plt.tight_layout()
    plt.savefig(SAVE_FIG_PATH)
    print(f"💾 验证结果已保存至: {os.path.abspath(SAVE_FIG_PATH)}")
    plt.show()

if __name__ == "__main__":
    main()