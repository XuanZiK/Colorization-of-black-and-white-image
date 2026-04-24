# ============================== 1) 基础导入 ===============================
import gc
import os
import time
import warnings
from dataclasses import dataclass
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from skimage.color import lab2rgb  # type: ignore
except ImportError:
    # Minimal LAB->RGB fallback to avoid hard dependency on scikit-image.
    def lab2rgb(lab: np.ndarray) -> np.ndarray:
        lab = lab.astype(np.float32)
        L = lab[..., 0]
        a = lab[..., 1]
        b = lab[..., 2]

        y = (L + 16.0) / 116.0
        x = y + (a / 500.0)
        z = y - (b / 200.0)

        epsilon = 0.008856
        kappa = 903.3

        def f_inv(t):
            return np.where(t ** 3 > epsilon, t ** 3, (116.0 * t - 16.0) / kappa)

        Xn, Yn, Zn = 0.95047, 1.0, 1.08883
        X = Xn * f_inv(x)
        Y = Yn * f_inv(y)
        Z = Zn * f_inv(z)

        xyz = np.stack([X, Y, Z], axis=-1)
        mat = np.array(
            [
                [3.2406, -1.5372, -0.4986],
                [-0.9689, 1.8758, 0.0415],
                [0.0557, -0.2040, 1.0570],
            ],
            dtype=np.float32,
        )
        rgb = xyz @ mat.T

        mask = rgb > 0.0031308
        rgb = np.where(mask, 1.055 * np.power(rgb, 1.0 / 2.4) - 0.055, 12.92 * rgb)
        return np.clip(rgb, 0.0, 1.0)
from torch import nn, optim
from torchvision.models.resnet import resnet18
from tqdm import tqdm
from load_lab_npy_data import DataConfig, build_dataloaders

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
    output_channels: int = 2

    # 这里给 7 更稳妥（224 尺寸通常比 8 层下采样更稳定）
    unet_n_down: int = 7
    unet_num_filters: int = 64

    # -------------------- 通用超参数 --------------------
    # A6000(40GB) 默认可用更大的 batch；如需更激进可尝试 48/64。
    batch_size: int = 64
    epochs: int = 20
    display_every: int = 100
    pretrain_max_steps_per_epoch: int = 0  
    gan_max_steps_per_epoch: int = 0      

    # -------------------- 优化器超参数 --------------------
    # 2e-4,5e-5
    gen_lr: float = 5e-5
    disc_lr: float = 2e-4
    pretrain_lr: float = 1e-4
    beta1: float = 0.5
    beta2: float = 0.999

    # -------------------- 损失相关 --------------------
    lambda_l1: float = 100.0
    lambda_tv: float = 0.1
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
    run_gan_training: bool = True

    # -------------------- CPU 负载控制 --------------------
    # True: 默认降低数据与 DataLoader 负载；False: 使用更激进的全量训练设置。
    cpu_friendly_mode: bool = True
    data_external_size: int = 12000
    data_train_size: int = 10000
    data_num_workers: int = 2
    data_pin_memory: bool = False


# ========================= 4) 设备与随机数设置 ============================
def get_device() -> torch.device:
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
        "loss_G_L1": AverageMeter(),
        "loss_G_TV": AverageMeter(),
        "loss_G": AverageMeter(),
    }


def update_losses(model, loss_meter_dict: Dict[str, AverageMeter], count: int):
    for loss_name, loss_meter in loss_meter_dict.items():
        loss = getattr(model, loss_name)
        loss_meter.update(loss.item(), count=count)


def log_results(loss_meter_dict: Dict[str, AverageMeter]):
    for loss_name, loss_meter in loss_meter_dict.items():
        print(f"{loss_name}: {loss_meter.avg:.5f}")


def lab_to_rgb(L: torch.Tensor, ab: torch.Tensor) -> np.ndarray:
    L = torch.clamp((L + 1.0) * 50.0, 0.0, 100.0)
    ab = torch.clamp(ab * 128.0, -128.0, 127.0)

    lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()

    rgb_imgs = []
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Conversion from CIE-LAB, via XYZ to sRGB color space resulted in .* clipped to zero",
            category=UserWarning,
        )
        for img in lab:
            rgb_imgs.append(np.clip(lab2rgb(img), 0.0, 1.0))
    return np.stack(rgb_imgs, axis=0)


def visualize(model, data, save: bool = False):
    model.generator.eval()
    with torch.no_grad():
        model.prepare_input(data)
        model.forward()

    fake_color = model.gen_output.detach()
    real_color = model.ab
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


class TotalVariationLoss(nn.Module):
    """Encourage spatial smoothness in generated ab maps to reduce speckle artifacts."""

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError(f"TV loss expects 4D input, got {tuple(image.shape)}")

        tv_h = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]).mean()
        tv_w = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]).mean()
        return tv_h + tv_w


# ======================== 7) 生成器 U-Net 模块定义 ========================
class UnetBlock(nn.Module):

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
            up = [uprelu, upconv, nn.Tanh()]
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
        if y.shape[2:] != x.shape[2:]:
            y = torch.nn.functional.interpolate(y, size=x.shape[2:], mode="bilinear", align_corners=False)

        return torch.cat([x, y], 1)


class Unet(nn.Module):

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

        # 最外层：输入 1 通道，输出 2 通道 ab
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

    if not FASTAI_AVAILABLE:
        raise ImportError("fastai is not available, cannot build backbone DynamicUnet")

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


# ====================== 10) 生成器预训练（L1） ==========================
def pretrain_generator(
    generator: nn.Module,
    train_dl,
    opt: optim.Optimizer,
    criterion: nn.Module,
    epochs: int,
    device: torch.device,
    use_amp: bool = True,
    max_steps_per_epoch: int = 0,
):
    """先用监督式 L1 预训练生成器，帮助后续 GAN 更稳定。"""
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
                ab = data["ab"].to(device)

                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    preds = generator(L)
                    loss = criterion(preds, ab)

                opt.zero_grad(set_to_none=True)
                if amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()
            except torch.OutOfMemoryError:
                opt.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
                raise

            loss_meter.update(loss.item(), L.size(0))

        print(f"Epoch {e + 1}/{epochs}")
        print(f"L1 Loss: {loss_meter.avg:.5f}")


# ========================= 11) 主模型（GAN） =============================
class MainModel(nn.Module):
    """整合生成器、判别器、损失与优化过程。"""

    def __init__(self, cfg: TrainConfig, generator: Optional[nn.Module] = None):
        super().__init__()
        self.cfg = cfg
        self.device = get_device()
        self.lambda_l1 = cfg.lambda_l1
        self.lambda_tv = cfg.lambda_tv

        if generator is None:
            self.generator = build_generator(cfg, self.device)
        else:
            self.generator = generator.to(self.device)

        self.discriminator = init_model(Discriminator(cfg, input_channels=3, num_filters=64, n_down=3), self.device)

        self.GANloss = GANLoss(gan_mode=cfg.gan_mode).to(self.device)
        self.L1loss = nn.L1Loss()
        self.TVloss = TotalVariationLoss().to(self.device)

        self.gen_optim = optim.Adam(self.generator.parameters(), lr=cfg.gen_lr, betas=(cfg.beta1, cfg.beta2))
        self.disc_optim = optim.Adam(self.discriminator.parameters(), lr=cfg.disc_lr, betas=(cfg.beta1, cfg.beta2))

    def requires_grad(self, model: nn.Module, requires_grad: bool = True):
        """控制参数是否参与梯度计算，减少不必要的开销。"""
        for p in model.parameters():
            p.requires_grad = requires_grad

    def prepare_input(self, data):
        """把 dataloader 给的 batch 搬到当前设备。"""
        self.L = data["L"].to(self.device)
        self.ab = data["ab"].to(self.device)

    def forward(self):
        """生成器前向：L -> 预测 ab。"""
        self.gen_output = self.generator(self.L)

    def disc_backward(self):
        """判别器反向传播：同时看 fake 与 real。"""
        gen_image = torch.cat([self.L, self.gen_output], dim=1)
        gen_image_preds = self.discriminator(gen_image.detach())
        self.disc_loss_gen = self.GANloss(gen_image_preds, False)

        real_image = torch.cat([self.L, self.ab], dim=1)
        real_preds = self.discriminator(real_image)
        self.disc_loss_real = self.GANloss(real_preds, True)

        self.disc_loss = (self.disc_loss_gen + self.disc_loss_real) * 0.5
        self.disc_loss.backward()

    def gen_backward(self):
        """生成器反向传播：GAN 损失 + L1 重建损失 + TV 平滑正则。"""
        gen_image = torch.cat([self.L, self.gen_output], dim=1)
        gen_image_preds = self.discriminator(gen_image)
        self.loss_G_GAN = self.GANloss(gen_image_preds, True)

        self.loss_G_L1 = self.L1loss(self.gen_output, self.ab) * self.lambda_l1
        self.loss_G_TV = self.TVloss(self.gen_output) * self.lambda_tv
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_TV
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
        preds = model.generator(L)
    return lab_to_rgb(L.cpu(), preds.cpu())


# ============================ 14) 运行入口 ================================
def main():

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
        num_workers=cfg.data_num_workers,
        pin_memory=cfg.data_pin_memory,
    )
    train_loader, valid_loader = build_dataloaders(data_cfg)
    print(f"[Data] train batches={len(train_loader)}, valid batches={len(valid_loader)}")

    # -------------------- 14.3 构建生成器 --------------------
    generator = build_generator(cfg, device)

    # -------------------- 14.4 可选：先做 L1 预训练 --------------------
    if cfg.run_generator_pretrain:
        print("\n[Stage] 开始生成器 L1 预训练")
        l1_loss = nn.L1Loss()

        cur_bs = data_cfg.batch_size
        while True:
            try:
                pretrain_opt = optim.Adam(generator.parameters(), lr=cfg.pretrain_lr)
                pretrain_generator(
                    generator,
                    train_loader,
                    pretrain_opt,
                    l1_loss,
                    cfg.epochs,
                    device,
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
