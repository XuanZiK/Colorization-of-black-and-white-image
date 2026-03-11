"""
从 notebook 中提取的 pix2pix 训练主代码（不含数据读取细节）。

说明：
1) 数据部分已拆分到 `load_lab_npy_data.py`，本文件直接复用其中的 DataLoader。
2) 本文件尽量保留 notebook 的训练逻辑，并补充大量中文注释，便于后续维护。
3) 默认不会直接开始长时间训练，你可以在 main() 中按需开启。
"""

# ============================== 1) 基础导入 ===============================
import gc
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.color import lab2rgb
from torch import nn, optim
from torchvision.models.resnet import resnet18
from tqdm import tqdm

# 数据加载来自你已经验证通过的脚本
from load_lab_npy_data import DataConfig, build_dataloaders, load_local_lab_data, load_color_bins


# ========================== 2) 可选 FastAI 导入 ===========================
# notebook 中使用了 create_body + DynamicUnet 作为预训练骨干。
# 这里做成可选依赖：有 fastai 就走原方案；没有就回退到纯 PyTorch U-Net。
try:
    from fastai.vision.learner import create_body
    from fastai.vision.models.unet import DynamicUnet

    FASTAI_AVAILABLE = True
except Exception:
    FASTAI_AVAILABLE = False


# ============================= 3) 训练配置区 ==============================
@dataclass
class TrainConfig:
    # -------------------- 模型结构相关 --------------------
    image_size_1: int = 224
    image_size_2: int = 224
    input_channels: int = 1
    output_channels: int = 313
    color_bins_path: str = "/home/xzk/thesis/archive/pts_in_hull.npy"

    # 这里给 7 更稳妥（224 尺寸通常比 8 层下采样更稳定）
    unet_n_down: int = 7
    unet_num_filters: int = 64

    # -------------------- 通用超参数 --------------------
    # A6000(40GB) 默认可用更大的 batch；如需更激进可尝试 48/64。
    batch_size: int = 64
    epochs: int = 20
    display_every: int = 100
    pretrain_max_steps_per_epoch: int = 0  # 0 表示跑完整个 epoch
    gan_max_steps_per_epoch: int = 0       # 0 表示跑完整个 epoch

    # -------------------- 优化器超参数 --------------------
    gen_lr: float = 2e-4
    disc_lr: float = 2e-4
    pretrain_lr: float = 1e-4
    beta1: float = 0.5
    beta2: float = 0.999

    # -------------------- 损失相关 --------------------
    lambda_ce: float = 1.0
    gan_mode: str = "vanilla"  # vanilla / lsgan

    # -------------------- 卷积层通用参数 --------------------
    kernel_size: int = 4
    stride: int = 2
    padding: int = 1
    leaky_relu_slope: float = 0.2
    dropout: float = 0.5

    # -------------------- 骨干网络设置 --------------------
    # 大显存默认开启 ResNet18 + DynamicUnet 方案。
    use_backbone_pretrain: bool = True
    layers_to_cut: int = -2

    # -------------------- 显存设置 --------------------
    use_amp: bool = True
    # A6000 默认不自动降 batch；若再次遇到 OOM 可改回 True。
    oom_auto_shrink_batch: bool = False
    min_batch_size: int = 2

    # -------------------- 运行行为控制 --------------------
    run_generator_pretrain: bool = True
    run_gan_training: bool = False

    # -------------------- CPU 负载控制 --------------------
    # True: 默认降低数据与 DataLoader 负载；False: 使用更激进的全量训练设置。
    cpu_friendly_mode: bool = True
    data_external_size: int = 12000
    data_train_size: int = 10000
    data_num_workers: int = 2
    data_pin_memory: bool = False


# ========================= 4) 设备与随机数设置 ============================
def get_device() -> torch.device:
    """统一设备入口，方便后续替换多卡逻辑。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42):
    """固定随机种子，尽量提升实验可复现性。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================ 5) 工具类与函数 =============================
class AverageMeter:
    """用于统计 loss 的当前值、累计值和平均值。"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.count, self.avg, self.sum = [0.0] * 3

    def update(self, val: float, count: int = 1):
        self.count += count
        self.sum += count * val
        self.avg = self.sum / max(self.count, 1e-12)


def create_loss_meters() -> Dict[str, AverageMeter]:
    """创建 GAN 训练所需的各项 loss 统计器。"""
    return {
        "disc_loss_gen": AverageMeter(),
        "disc_loss_real": AverageMeter(),
        "disc_loss": AverageMeter(),
        "loss_G_GAN": AverageMeter(),
        "loss_G_CE": AverageMeter(),
        "loss_G": AverageMeter(),
    }


def update_losses(model, loss_meter_dict: Dict[str, AverageMeter], count: int):
    """把当前 step 的模型 loss 写入统计器。"""
    for loss_name, loss_meter in loss_meter_dict.items():
        loss = getattr(model, loss_name)
        loss_meter.update(loss.item(), count=count)


def log_results(loss_meter_dict: Dict[str, AverageMeter]):
    """打印当前阶段 loss 平均值，便于观察训练是否稳定。"""
    for loss_name, loss_meter in loss_meter_dict.items():
        print(f"{loss_name}: {loss_meter.avg:.5f}")


def lab_to_rgb(L: torch.Tensor, ab: torch.Tensor) -> np.ndarray:
    """
    把一批 LAB 张量转换成 RGB 方便可视化。

    输入约定：
    - L:  [-1, 1]
    - ab: [-1, 1]（或近似）
    """
    # 与 notebook 保持一致：L 反归一化到 [0,100]，ab 放大到典型范围。
    L = (L + 1.0) * 50.0
    ab = ab * 128.0

    lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()

    rgb_imgs = []
    for img in lab:
        rgb_imgs.append(lab2rgb(img))
    return np.stack(rgb_imgs, axis=0)


def load_color_bins_tensor(path: str, device: torch.device) -> torch.Tensor:
    """Load 313 color centers from pts_in_hull.npy and move to device."""
    if not path:
        raise ValueError("color_bins_path is empty")

    if os.path.exists(path):
        bins = np.load(path).astype(np.float32)
    else:
        # 兜底：若外部未提前生成颜色中心，则从 AB 数据自动估计。
        cfg = DataConfig(color_bins_path=path)
        _, ab_data_parts, _ = load_local_lab_data(cfg)
        bins = load_color_bins(cfg, ab_data_parts)

    if bins.ndim != 2 or bins.shape[1] != 2:
        raise ValueError(f"pts_in_hull.npy should have shape (N,2), got {bins.shape}")
    if bins.shape[0] != 313:
        raise ValueError(f"Expected 313 color bins, got {bins.shape[0]}")
    return torch.from_numpy(bins).to(device)


def class_idx_to_ab(class_idx: torch.Tensor, color_bins_lab: torch.Tensor) -> torch.Tensor:
    """Map class index map (B,H,W) to normalized ab map (B,2,H,W) in [-1,1]."""
    if class_idx.ndim != 3:
        raise ValueError(f"class_idx should be (B,H,W), got {tuple(class_idx.shape)}")

    b, h, w = class_idx.shape
    flat = class_idx.reshape(-1).long()
    ab_lab = color_bins_lab.index_select(0, flat).reshape(b, h, w, 2)
    ab_lab = ab_lab.permute(0, 3, 1, 2).contiguous()
    return torch.clamp(ab_lab / 128.0, -1.0, 1.0)


def ab_to_class_idx(ab: torch.Tensor, color_bins_lab: torch.Tensor) -> torch.Tensor:
    """Quantize normalized ab map (B,2,H,W) to nearest 313-bin class indices (B,H,W)."""
    if ab.ndim != 4 or ab.shape[1] != 2:
        raise ValueError(f"ab should be (B,2,H,W), got {tuple(ab.shape)}")

    b, _, h, w = ab.shape
    ab_lab = (ab * 128.0).permute(0, 2, 3, 1).reshape(-1, 2)
    dist = torch.cdist(ab_lab, color_bins_lab)
    class_idx = torch.argmin(dist, dim=1)
    return class_idx.reshape(b, h, w)


def logits_to_ab(logits: torch.Tensor, color_bins_lab: torch.Tensor) -> torch.Tensor:
    """Decode generator logits (B,313,H,W) to normalized ab map for visualization/GAN."""
    class_idx = torch.argmax(logits, dim=1)
    return class_idx_to_ab(class_idx, color_bins_lab)


def visualize(model, data, save: bool = False):
    """可视化灰度输入、生成结果与真实结果。"""
    model.generator.eval()
    with torch.no_grad():
        model.prepare_input(data)
        model.forward()

    fake_color = logits_to_ab(model.gen_output.detach(), model.color_bins_lab)
    real_color = class_idx_to_ab(model.class_idx, model.color_bins_lab)
    L = model.L

    fake_imgs = lab_to_rgb(L, fake_color)
    real_imgs = lab_to_rgb(L, real_color)

    fig = plt.figure(figsize=(15, 8))
    show_n = min(5, L.size(0))

    for i in range(show_n):
        ax = plt.subplot(3, 5, i + 1)
        ax.imshow(L[i][0].cpu(), cmap="gray")
        ax.axis("off")

        ax = plt.subplot(3, 5, i + 1 + 5)
        ax.imshow(fake_imgs[i])
        ax.axis("off")

        ax = plt.subplot(3, 5, i + 1 + 10)
        ax.imshow(real_imgs[i])
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    if save:
        fig.savefig(f"colorization_{time.time()}.png")


# =========================== 6) GAN Loss 定义 ============================
class GANLoss(nn.Module):
    """支持 vanilla GAN 与 LSGAN 两种损失。"""

    def __init__(self, gan_mode: str = "vanilla", real_label: float = 1.0, fake_label: float = 0.0):
        super().__init__()
        self.register_buffer("real_label", torch.tensor(real_label))
        self.register_buffer("fake_label", torch.tensor(fake_label))

        if gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported gan_mode: {gan_mode}")

    def get_labels(self, preds: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        labels = self.real_label if target_is_real else self.fake_label
        return labels.expand_as(preds)

    def __call__(self, preds: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        labels = self.get_labels(preds, target_is_real)
        return self.loss(preds, labels)


# ======================== 7) 生成器 U-Net 模块定义 ========================
class UnetBlock(nn.Module):
    """用于构建 U-Net 的可嵌套 block（最外层/中间层/最内层）。"""

    def __init__(
        self,
        cfg: TrainConfig,
        nf: int,
        ni: int,
        submodule: Optional[nn.Module] = None,
        input_channels: Optional[int] = None,
        dropout: bool = False,
        innermost: bool = False,
        outermost: bool = False,
    ):
        super().__init__()
        self.outermost = outermost

        if input_channels is None:
            input_channels = nf

        downconv = nn.Conv2d(
            input_channels,
            ni,
            kernel_size=cfg.kernel_size,
            stride=cfg.stride,
            padding=cfg.padding,
            bias=False,
        )
        downrelu = nn.LeakyReLU(cfg.leaky_relu_slope, True)
        downnorm = nn.BatchNorm2d(ni)

        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(nf)

        if outermost:
            upconv = nn.ConvTranspose2d(
                ni * 2,
                nf,
                kernel_size=cfg.kernel_size,
                stride=cfg.stride,
                padding=cfg.padding,
            )
            down = [downconv]
            up = [uprelu, upconv]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(
                ni,
                nf,
                kernel_size=cfg.kernel_size,
                stride=cfg.stride,
                padding=cfg.padding,
                bias=False,
            )
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(
                ni * 2,
                nf,
                kernel_size=cfg.kernel_size,
                stride=cfg.stride,
                padding=cfg.padding,
                bias=False,
            )
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if dropout:
                up += [nn.Dropout(cfg.dropout)]
            model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.outermost:
            return self.model(x)
        y = self.model(x)

        # 关键修复：当输入尺寸不是 2^n 的严格倍数（例如 224）时，
        # 下采样/上采样链路会在某些层出现 1 个像素的尺寸偏差。
        # 这里在 skip-connection 拼接前做一次尺寸对齐，避免 cat 时报错。
        if y.shape[2:] != x.shape[2:]:
            y = torch.nn.functional.interpolate(y, size=x.shape[2:], mode="bilinear", align_corners=False)

        return torch.cat([x, y], 1)


class Unet(nn.Module):
    """纯 PyTorch U-Net 生成器。"""

    def __init__(self, cfg: TrainConfig):
        super().__init__()

        n_down = cfg.unet_n_down
        num_filters = cfg.unet_num_filters

        # 最内层
        unet_block = UnetBlock(cfg, num_filters * 8, num_filters * 8, innermost=True)

        # 中间若干层
        for _ in range(max(n_down - 5, 0)):
            unet_block = UnetBlock(cfg, num_filters * 8, num_filters * 8, submodule=unet_block, dropout=True)

        # 逐步向外层扩展
        out_filters = num_filters * 8
        for _ in range(3):
            unet_block = UnetBlock(cfg, out_filters // 2, out_filters, submodule=unet_block)
            out_filters //= 2

        # 最外层：输入 1 通道，输出 313 类 logits
        self.model = UnetBlock(
            cfg,
            cfg.output_channels,
            out_filters,
            input_channels=cfg.input_channels,
            submodule=unet_block,
            outermost=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# =========================== 8) 判别器定义 ===============================
class Discriminator(nn.Module):
    """PatchGAN 判别器：判断局部 patch 的真伪。"""

    def __init__(self, cfg: TrainConfig, input_channels: int, num_filters: int = 64, n_down: int = 3):
        super().__init__()

        model = [self.get_layers(cfg, input_channels, num_filters, norm=False)]
        model += [
            self.get_layers(
                cfg,
                num_filters * 2 ** i,
                num_filters * 2 ** (i + 1),
                stride=1 if i == (n_down - 1) else 2,
            )
            for i in range(n_down)
        ]
        model += [self.get_layers(cfg, num_filters * 2 ** n_down, 1, stride=1, norm=False, activation=False)]
        self.model = nn.Sequential(*model)

    def get_layers(
        self,
        cfg: TrainConfig,
        ni: int,
        nf: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        norm: bool = True,
        activation: bool = True,
    ) -> nn.Sequential:
        layers = [nn.Conv2d(ni, nf, kernel_size, stride, padding, bias=not norm)]
        if norm:
            layers += [nn.BatchNorm2d(nf)]
        if activation:
            layers += [nn.LeakyReLU(cfg.leaky_relu_slope, True)]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ========================= 9) 初始化与骨干构建 ============================
def init_weights(net: nn.Module, init: str = "norm", gain: float = 0.02) -> nn.Module:
    """按 notebook 方案初始化网络权重。"""

    def init_func(m):
        classname = m.__class__.__name__

        if hasattr(m, "weight") and "Conv" in classname:
            if init == "norm":
                nn.init.normal_(m.weight.data, mean=0.0, std=gain)
            elif init == "xavier":
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init == "kaiming":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")

            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif "BatchNorm2d" in classname:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)
    print(f"model initialized with {init} initialization")
    return net


def init_model(model: nn.Module, device: torch.device) -> nn.Module:
    """把模型移动到设备并完成初始化。"""
    model = model.to(device)
    model = init_weights(model)
    return model


def build_backbone_unet(cfg: TrainConfig, device: torch.device) -> nn.Module:
    """
    使用 fastai 的 DynamicUnet + ResNet18 backbone 构建生成器。
    如果 fastai 不可用会抛错，由上层决定是否回退。
    """
    if not FASTAI_AVAILABLE:
        raise ImportError("fastai is not available, cannot build backbone DynamicUnet")

    # fastai 的 create_body 需要的是“模型实例”而不是函数对象。
    # 这里兼容不同 torchvision 版本的预训练权重参数写法。
    try:
        backbone = resnet18(weights="IMAGENET1K_V1")
    except Exception:
        backbone = resnet18(pretrained=True)

    body = create_body(backbone, n_in=cfg.input_channels, cut=cfg.layers_to_cut)
    generator = DynamicUnet(body, cfg.output_channels, (cfg.image_size_1, cfg.image_size_2)).to(device)
    return generator


def build_generator(cfg: TrainConfig, device: torch.device) -> nn.Module:
    """根据配置选择生成器实现。"""
    if cfg.use_backbone_pretrain:
        try:
            print("[Info] 尝试构建 ResNet18 + DynamicUnet 生成器...")
            return build_backbone_unet(cfg, device)
        except Exception as e:
            print(f"[Warn] 骨干 U-Net 构建失败，回退到纯 U-Net: {e}")

    print("[Info] 使用纯 PyTorch U-Net 生成器")
    return init_model(Unet(cfg), device)


# ====================== 9.5) Total Variation Loss ==========================
def total_variation_loss(img: torch.Tensor, tv_weight: float = 1e-6) -> torch.Tensor:
    """Calculate Total Variation Loss to encourage spatial smoothness and reduce artifacts."""
    b, c, h, w = img.size()
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    return tv_weight * (tv_h + tv_w) / (b * c * h * w)


# ====================== 10) 生成器预训练（L1） ==========================
def pretrain_generator(
    generator: nn.Module,
    train_dl,
    opt: optim.Optimizer,
    criterion: nn.Module,
    epochs: int,
    device: torch.device,
    color_bins_lab: torch.Tensor,
    use_amp: bool = True,
    max_steps_per_epoch: int = 0,
):
    """先用监督式 CE 预训练生成器，帮助后续 GAN 更稳定。"""
    generator.train()
    amp_enabled = use_amp and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    for e in range(epochs):
        loss_meter = AverageMeter()

        for step_i, data in enumerate(tqdm(train_dl, desc=f"[Pretrain] Epoch {e + 1}/{epochs}"), start=1):
            if max_steps_per_epoch > 0 and step_i > max_steps_per_epoch:
                break

            try:
                L = data["L"].to(device)
                real_ab = data["ab"].to(device)
                class_idx = ab_to_class_idx(real_ab, color_bins_lab)

                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    preds = generator(L)
                    ce_loss = criterion(preds, class_idx)
                    tv_loss = total_variation_loss(preds)
                    loss = ce_loss + tv_loss

                opt.zero_grad(set_to_none=True)
                if amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()
            except torch.OutOfMemoryError:
                # 清理缓存后把异常抛给上层，由上层统一执行 batch 缩减与重试。
                opt.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
                raise

            loss_meter.update(loss.item(), L.size(0))

        print(f"Epoch {e + 1}/{epochs}")
        print(f"CE Loss: {loss_meter.avg:.5f}, TV Loss: {tv_loss.item():.5f}")


# ========================= 11) 主模型（GAN） =============================
class MainModel(nn.Module):
    """整合生成器、判别器、损失与优化过程。"""

    def __init__(self, cfg: TrainConfig, generator: Optional[nn.Module] = None):
        super().__init__()
        self.cfg = cfg
        self.device = get_device()
        self.lambda_ce = cfg.lambda_ce
        self.color_bins_lab = load_color_bins_tensor(cfg.color_bins_path, self.device)

        if generator is None:
            self.generator = build_generator(cfg, self.device)
        else:
            self.generator = generator.to(self.device)

        self.discriminator = init_model(Discriminator(cfg, input_channels=3, num_filters=64, n_down=3), self.device)

        self.GANloss = GANLoss(gan_mode=cfg.gan_mode).to(self.device)
        self.CEloss = nn.CrossEntropyLoss()

        self.gen_optim = optim.Adam(self.generator.parameters(), lr=cfg.gen_lr, betas=(cfg.beta1, cfg.beta2))
        self.disc_optim = optim.Adam(self.discriminator.parameters(), lr=cfg.disc_lr, betas=(cfg.beta1, cfg.beta2))

    def requires_grad(self, model: nn.Module, requires_grad: bool = True):
        """控制参数是否参与梯度计算，减少不必要的开销。"""
        for p in model.parameters():
            p.requires_grad = requires_grad

    def prepare_input(self, data):
        """把 dataloader 给的 batch 搬到当前设备。"""
        self.L = data["L"].to(self.device)
        self.real_ab = data["ab"].to(self.device)
        self.class_idx = ab_to_class_idx(self.real_ab, self.color_bins_lab)

    def forward(self):
        """生成器前向：L -> 预测颜色类别 logits。"""
        self.gen_output = self.generator(self.L)
        self.gen_ab = logits_to_ab(self.gen_output, self.color_bins_lab)

    def disc_backward(self):
        """判别器反向传播：同时看 fake 与 real。"""
        gen_image = torch.cat([self.L, self.gen_ab], dim=1)
        gen_image_preds = self.discriminator(gen_image.detach())
        self.disc_loss_gen = self.GANloss(gen_image_preds, False)

        real_image = torch.cat([self.L, self.real_ab], dim=1)
        real_preds = self.discriminator(real_image)
        self.disc_loss_real = self.GANloss(real_preds, True)

        self.disc_loss = (self.disc_loss_gen + self.disc_loss_real) * 0.5
        self.disc_loss.backward()

    def gen_backward(self):
        """生成器反向传播：GAN 损失 + CE 分类损失。"""
        gen_image = torch.cat([self.L, self.gen_ab], dim=1)
        gen_image_preds = self.discriminator(gen_image)
        self.loss_G_GAN = self.GANloss(gen_image_preds, True)

        self.loss_G_CE = self.CEloss(self.gen_output, self.class_idx) * self.lambda_ce
        self.loss_G = self.loss_G_GAN + self.loss_G_CE
        self.loss_G.backward()

    def optimize(self):
        """一次完整优化：先判别器，再生成器。"""
        self.forward()

        # 1) 更新判别器
        self.discriminator.train()
        self.requires_grad(self.discriminator, True)
        self.disc_optim.zero_grad()
        self.disc_backward()
        self.disc_optim.step()

        # 2) 更新生成器（冻结判别器参数）
        self.generator.train()
        self.requires_grad(self.discriminator, False)
        self.gen_optim.zero_grad()
        self.gen_backward()
        self.gen_optim.step()


# =========================== 12) GAN 训练循环 ============================
def train_model(
    model: MainModel,
    train_loader,
    epochs: int,
    display: int = 100,
    max_steps_per_epoch: int = 0,
):
    """按 notebook 风格执行 GAN 训练，并定期可视化。"""
    for epoch in range(epochs):
        loss_meter_dict = create_loss_meters()
        i = 0

        for data in tqdm(train_loader, desc=f"[GAN] Epoch {epoch + 1}/{epochs}"):
            if max_steps_per_epoch > 0 and i >= max_steps_per_epoch:
                break

            model.prepare_input(data)
            model.optimize()
            update_losses(model, loss_meter_dict, count=data["L"].size(0))

            i += 1
            if i % display == 0:
                print(f"\nEpoch {epoch + 1}/{epochs}")
                print(f"Iteration {i}/{len(train_loader)}")
                log_results(loss_meter_dict)
                visualize(model, data, save=False)


# ============================ 13) 推理辅助函数 ============================
def infer_one_batch(model: MainModel, batch):
    """对一个 batch 做推理并返回可视化用 RGB。"""
    model.eval()
    with torch.no_grad():
        L = batch["L"].to(model.device)
        logits = model.generator(L)
        preds_ab = logits_to_ab(logits, model.color_bins_lab)
    return lab_to_rgb(L.cpu(), preds_ab.cpu())


# ============================ 14) 运行入口 ================================
def main():
    """
    建议执行流程：
    1) 先跑数据加载 + 1 个 batch 验证（你已经完成）。
    2) 再执行本文件，先观察是否能完成模型构建和 1 个前向。
    3) 最后再开启完整训练。
    """
    set_seed(42)
    device = get_device()
    print(f"[Device] {device}")

    # -------------------- 14.1 训练参数 --------------------
    cfg = TrainConfig(
        image_size_1=224,
        image_size_2=224,
    )

    if cfg.cpu_friendly_mode:
        print("[Info] CPU 友好模式已开启：减少 DataLoader 线程与每轮样本量")

    # -------------------- 14.2 构建数据 --------------------
    data_cfg = DataConfig(
        external_data_size=cfg.data_external_size,
        train_size=cfg.data_train_size,
        batch_size=cfg.batch_size,
        color_bins_path=cfg.color_bins_path,
        num_workers=cfg.data_num_workers,
        pin_memory=cfg.data_pin_memory,
    )
    train_loader, valid_loader = build_dataloaders(data_cfg)
    print(f"[Data] train batches={len(train_loader)}, valid batches={len(valid_loader)}")

    # -------------------- 14.3 构建生成器 --------------------
    generator = build_generator(cfg, device)

    # -------------------- 14.4 可选：先做 CE 预训练 --------------------
    if cfg.run_generator_pretrain:
        print("\n[Stage] 开始生成器 CE 预训练")
        ce_loss = nn.CrossEntropyLoss()
        color_bins_lab = load_color_bins_tensor(cfg.color_bins_path, device)

        cur_bs = data_cfg.batch_size
        while True:
            try:
                pretrain_opt = optim.Adam(generator.parameters(), lr=cfg.pretrain_lr)
                pretrain_generator(
                    generator,
                    train_loader,
                    pretrain_opt,
                    ce_loss,
                    cfg.epochs,
                    device,
                    color_bins_lab,
                    use_amp=cfg.use_amp,
                    max_steps_per_epoch=cfg.pretrain_max_steps_per_epoch,
                )
                break
            except torch.OutOfMemoryError as e:
                if not cfg.oom_auto_shrink_batch:
                    raise

                new_bs = max(cfg.min_batch_size, cur_bs // 2)
                if new_bs == cur_bs:
                    print("[Error] 已达到最小 batch size，仍然 OOM。")
                    raise

                print(f"[Warn] 发生 OOM，batch_size: {cur_bs} -> {new_bs}，自动重试。")
                print(f"[Warn] 原始错误: {e}")

                cur_bs = new_bs
                data_cfg.batch_size = cur_bs
                train_loader, valid_loader = build_dataloaders(data_cfg)
                print(f"[Data] train batches={len(train_loader)}, valid batches={len(valid_loader)}")

                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

        # 保存与重载：与 notebook 行为保持一致，方便断点续训。
        torch.save(generator.state_dict(), "res18-unet.pt")
        generator.load_state_dict(torch.load("res18-unet.pt", map_location=device))

    # -------------------- 14.5 构建 GAN 主模型 --------------------
    model = MainModel(cfg=cfg, generator=generator)

    # -------------------- 14.6 可选：执行 GAN 训练 --------------------
    if cfg.run_gan_training:
        print("\n[Stage] 开始 GAN 训练")
        train_model(
            model,
            train_loader,
            cfg.epochs,
            display=cfg.display_every,
            max_steps_per_epoch=cfg.gan_max_steps_per_epoch,
        )
        torch.save(model.state_dict(), "main-model.pt")
        print("[Done] main-model.pt 已保存")
    else:
        print("\n[Info] 已跳过 GAN 训练（cfg.run_gan_training=False）")
        print("[Info] 你可以把 run_gan_training 改为 True 后再正式开训")


if __name__ == "__main__":
    main()
