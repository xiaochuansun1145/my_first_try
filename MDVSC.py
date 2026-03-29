import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
import time
import matplotlib.pyplot as plt
from collections import deque
from pytorch_msssim import ms_ssim


# --------------------------
# 熵图生成模块
# --------------------------
class EntropyModel(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(latent_dim // 2, latent_dim // 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(latent_dim // 4, latent_dim // 8, kernel_size=3, stride=2, padding=1),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim // 8, latent_dim // 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(latent_dim // 4, latent_dim // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(latent_dim // 2, latent_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # 输出0-1之间的熵值
        )

    def forward(self, x):
        encoded = self.encoder(x)
        entropy_map = self.decoder(encoded)
        return entropy_map


# --------------------------
# 残差块模块
# --------------------------
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x += residual
        return self.relu(x)


# --------------------------
# 核心模型架构 (基于百分比掩码)
# --------------------------
class MDVSCv3(nn.Module):
    def __init__(self, in_channels=3, latent_dim=64, gop_size=6,
                 common_ratio=0.4, individual_ratio=0.2, mask_temperature=0.1):
        super().__init__()
        self.gop_size = gop_size
        self.common_ratio = common_ratio  # 公共特征保留比例
        self.individual_ratio = individual_ratio  # 个体特征保留比例
        self.mask_temperature = mask_temperature  # soft top-k温度，越小越接近hard top-k

        # --------------------------
        # 发送端架构
        # --------------------------
        # 潜在空间转换器
        self.latent_transformer = nn.Sequential(
            nn.Conv2d(in_channels, latent_dim, kernel_size=5, stride=2, padding=2),
            *[ResBlock(latent_dim) for _ in range(2)]
        )

        # JSCC编码器
        self.jscc_encoder = nn.Sequential(
            self._make_encoder_block(latent_dim),
            self._make_encoder_block(latent_dim),
            self._make_encoder_block(latent_dim)
        )

        # 公共特征提取器
        self.common_feature_extractor = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # 熵模型
        self.entropy_model_common = EntropyModel(latent_dim)
        self.entropy_model_individual = nn.ModuleList([
            EntropyModel(latent_dim) for _ in range(gop_size)
        ])

        # --------------------------
        # 接收端架构
        # --------------------------
        # JSCC解码器
        self.jscc_decoder = nn.Sequential(
            self._make_decoder_block(latent_dim),
            self._make_decoder_block(latent_dim),
            self._make_decoder_block(latent_dim)
        )

        # 潜在空间逆转换器
        self.latent_inversion = nn.Sequential(
            *[ResBlock(latent_dim) for _ in range(2)],
            nn.ConvTranspose2d(latent_dim, in_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Tanh()  # 约束输出到[-1, 1]，与输入归一化范围保持一致
        )

    def _make_encoder_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
            *[ResBlock(channels) for _ in range(2)]
        )

    def _make_decoder_block(self, channels):
        return nn.Sequential(
            nn.ConvTranspose2d(channels, channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            *[ResBlock(channels) for _ in range(2)]
        )

    def create_mask(self, entropy_map, ratio, temperature=None):
        """
        根据熵图和保留比例创建掩码。

        训练阶段：
            使用 straight-through estimator (STE)。
            forward 仍然是 hard top-k，backward 通过 soft mask 传播梯度。

        推理阶段：
            直接使用 hard top-k。
        """
        b, c, h, w = entropy_map.shape
        flat_entropy = entropy_map.reshape(b, -1)
        total_elements = c * h * w
        k = max(1, min(total_elements, int(ratio * total_elements)))

        # hard top-k mask：保证前向选择逻辑和原实现一致
        top_values, top_indices = torch.topk(flat_entropy, k, dim=1)
        hard_mask = torch.zeros_like(flat_entropy)
        hard_mask.scatter_(1, top_indices, 1.0)

        if (not self.training):
            return hard_mask.view(b, c, h, w)

        if temperature is None:
            temperature = self.mask_temperature

        # soft mask：用第k大值作为连续门限，给entropy分支提供梯度
        kth_value = top_values[:, -1:].detach()
        soft_mask = torch.sigmoid((flat_entropy - kth_value) / temperature)

        # 可选：把soft mask的平均激活数拉近k，减少hard/soft偏差
        soft_sum = soft_mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
        soft_mask = soft_mask * (k / soft_sum)
        soft_mask = soft_mask.clamp(0.0, 1.0)

        # STE: forward=hard, backward=soft
        mask = hard_mask.detach() - soft_mask.detach() + soft_mask
        return mask.view(b, c, h, w)

    def create_hard_mask(self, entropy_map, ratio):
        """仅用于统计/评估的hard top-k mask。"""
        b, c, h, w = entropy_map.shape
        flat_entropy = entropy_map.reshape(b, -1)
        total_elements = c * h * w
        k = max(1, min(total_elements, int(ratio * total_elements)))
        _, top_indices = torch.topk(flat_entropy, k, dim=1)
        mask = torch.zeros_like(flat_entropy)
        mask.scatter_(1, top_indices, 1.0)
        return mask.view(b, c, h, w)

    def forward(self, gop, return_details=False, channel_snr=15):
        """
        处理一个GOP（6帧视频）

        参数:
        gop: (batch_size, gop_size, C, H, W)

        返回:
        recovered_gop: 重建的GOP (batch_size, gop_size, C, H, W)
        cbr: 通道带宽比 (标量)
        """
        batch_size, gop_size, C, H, W = gop.shape

        # --------------------------
        # 发送端处理
        # --------------------------
        # 转换到潜在空间
        latent_frames = []
        for i in range(gop_size):
            frame = gop[:, i]
            latent = self.latent_transformer(frame)
            latent_frames.append(latent)
        latent_gop = torch.stack(latent_frames, dim=1)  # (B, T, C, H, W)

        # JSCC编码
        encoded_frames = []
        for i in range(gop_size):
            frame = latent_gop[:, i]
            encoded = self.jscc_encoder(frame)
            encoded_frames.append(encoded)
        encoded_gop = torch.stack(encoded_frames, dim=1)  # (B, T, C, H, W)

        # 公共特征提取
        mean_features = torch.mean(encoded_gop, dim=1)  # 平均特征作为先验
        common_features = self.common_feature_extractor(mean_features)

        # 个体特征计算
        individual_features = encoded_gop - common_features.unsqueeze(1)

        # 熵值图生成
        # 公共特征熵值图
        entropy_common = self.entropy_model_common(common_features)

        # 个体特征熵值图
        entropy_individual = []
        for i in range(gop_size):
            entropy_ind = self.entropy_model_individual[i](individual_features[:, i])
            entropy_individual.append(entropy_ind)

        # 创建可微掩码（训练时STE，推理时hard）
        common_mask = self.create_mask(entropy_common, self.common_ratio)
        individual_masks = []
        for i in range(gop_size):
            mask = self.create_mask(entropy_individual[i], self.individual_ratio)
            individual_masks.append(mask)

        # 单独生成hard mask用于统计CBR，避免STE软梯度影响统计值
        common_mask_hard = self.create_hard_mask(entropy_common, self.common_ratio)
        individual_masks_hard = []
        for i in range(gop_size):
            hard_mask = self.create_hard_mask(entropy_individual[i], self.individual_ratio)
            individual_masks_hard.append(hard_mask)

        # 应用掩码压缩
        coded_common = common_features * common_mask
        coded_individual = []
        for i in range(gop_size):
            coded_ind = individual_features[:, i] * individual_masks[i]
            coded_individual.append(coded_ind)

        # 计算CBR (通道带宽比)
        # 原始数据量 (像素数)
        original_pixels = batch_size * gop_size * H * W * C

        # 传输数据量 (元素数)
        # 公共特征
        common_elements = torch.sum(common_mask_hard).item()
        #print('mask size',common_mask.shape,'origin',original_pixels)
        # 个体特征
        individual_elements = sum(torch.sum(mask).item() for mask in individual_masks_hard)
        # 总传输元素数
        transmitted_elements = common_elements + individual_elements

        # CBR = 传输元素数 / 原始像素数
        cbr = transmitted_elements / original_pixels
        #print(common_elements,transmitted_elements,original_pixels)
        # 统计信息
        stats = {
            "cbr": cbr,
            "common_elements": common_elements,
            "individual_elements": individual_elements,
            "common_ratio": self.common_ratio,
            "individual_ratio": self.individual_ratio
        }

        # --------------------------
        # 信道传输（添加AWGN噪声）
        # --------------------------
        noisy_common = self.awgn_channel(coded_common, snr=channel_snr)
        noisy_individual = [self.awgn_channel(ind, snr=channel_snr) for ind in coded_individual]

        # --------------------------
        # 接收端处理
        # --------------------------
        # JSCC解码
        decoded_frames = []
        for i in range(gop_size):
            # 组合公共特征和个体特征
            combined = noisy_common + noisy_individual[i]
            decoded = self.jscc_decoder(combined)
            decoded_frames.append(decoded)

        # 潜在空间逆转换
        recovered_frames = []
        for i in range(gop_size):
            frame = self.latent_inversion(decoded_frames[i])
            recovered_frames.append(frame)

        recovered_gop = torch.stack(recovered_frames, dim=1)

        if return_details:
            details = {
                "entropy_common": entropy_common,
                "entropy_individual": entropy_individual,
                "common_mask_hard": common_mask_hard,
                "individual_masks_hard": individual_masks_hard,
                "common_features": common_features,
                "individual_features": individual_features
            }
            return recovered_gop, cbr, stats, details

        return recovered_gop, cbr, stats

    def awgn_channel(self, x, snr=15):
        """
        加性高斯白噪声信道
        """
        signal_power = torch.mean(x ** 2)
        noise_power = signal_power / (10 ** (snr / 10))
        noise = torch.randn_like(x) * torch.sqrt(noise_power)
        return x + noise


# --------------------------
# 视频数据集处理
# --------------------------
class VideoDataset(Dataset):
    def __init__(self, video_path, gop_size=6, frame_size=(128, 128)):
        self.gop_size = gop_size
        self.frame_size = frame_size
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        if os.path.isdir(video_path):
            self.frames = self._load_from_folder(video_path)
        else:
            self.frames = self._load_from_video(video_path)

        # 创建GOP索引
        self.gop_indices = []
        num_gops = len(self.frames) // gop_size
        for i in range(num_gops):
            start_idx = i * gop_size
            end_idx = start_idx + gop_size
            self.gop_indices.append((start_idx, end_idx))

    def _load_from_folder(self, folder_path):
        """从图片文件夹加载帧"""
        frame_files = sorted([f for f in os.listdir(folder_path)
                              if f.endswith(('.png', '.jpg', '.jpeg'))])
        frames = []
        for file in frame_files:
            img = cv2.imread(os.path.join(folder_path, file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.frame_size)
            frames.append(img)
        return frames

    def _load_from_video(self, video_path):
        """从视频文件加载帧"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.frame_size)
            frames.append(frame)
        cap.release()
        return frames

    def __len__(self):
        return len(self.gop_indices)

    def __getitem__(self, idx):
        start, end = self.gop_indices[idx]
        gop = self.frames[start:end]

        # 转换为张量并归一化
        gop_tensors = []
        for frame in gop:
            tensor = self.transform(frame)
            gop_tensors.append(tensor)

        return torch.stack(gop_tensors)  # (T, C, H, W)


# --------------------------
# 可视化工具
# --------------------------
def visualize_comparison(original, reconstructed, epoch, batch_idx, save_dir="reconstructions"):
    """
    可视化原始帧与重建帧的对比
    """
    os.makedirs(save_dir, exist_ok=True)

    # 取batch中第一个样本
    orig_frames = original[0].detach().cpu().numpy()
    recon_frames = reconstructed[0].detach().cpu().numpy()

    # 反归一化
    orig_frames = (orig_frames * 0.5 + 0.5).clip(0, 1)
    recon_frames = (recon_frames * 0.5 + 0.5).clip(0, 1)

    # 创建对比图
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    plt.suptitle(f'Epoch {epoch + 1}, Batch {batch_idx} - Original vs Reconstructed')

    for i in range(6):
        # 原始帧
        axes[0, i].imshow(np.transpose(orig_frames[i], (1, 2, 0)))
        axes[0, i].set_title(f'Original Frame {i + 1}')
        axes[0, i].axis('off')

        # 重建帧
        axes[1, i].imshow(np.transpose(recon_frames[i], (1, 2, 0)))
        axes[1, i].set_title(f'Reconstructed Frame {i + 1}')
        axes[1, i].axis('off')

    # 保存并关闭
    plt.savefig(f'{save_dir}/epoch_{epoch + 1}_batch_{batch_idx}.png')
    plt.close()

    print(f"Saved visualization: epoch_{epoch + 1}_batch_{batch_idx}.png")



def _to_display_image(frame_tensor):
    """将[-1,1]范围的张量转换为可显示图像。"""
    img = frame_tensor.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = (img * 0.5 + 0.5).clip(0, 1)
    return img


def _to_heatmap(feature_tensor, sample_idx=0, channel_reduce="mean"):
    """将(C,H,W)特征图转换为2D热力图。"""
    feat = feature_tensor[sample_idx].detach().cpu().numpy()
    if channel_reduce == "first":
        return feat[0]
    return feat.mean(axis=0)


def visualize_entropy_mask_reconstruction(model, gop, epoch, batch_idx,
                                          frame_idx=0, sample_idx=0,
                                          channel_snr=15,
                                          save_dir="entropy_mask_reconstruction"):
    """
    生成 Original / Reconstruction / Entropy Map / Mask 多图对比。
    """
    os.makedirs(save_dir, exist_ok=True)
    was_training = model.training
    model.eval()

    with torch.no_grad():
        recovered_gop, _, _, details = model(gop, return_details=True, channel_snr=channel_snr)

    original_img = _to_display_image(gop[sample_idx, frame_idx])
    recon_img = _to_display_image(recovered_gop[sample_idx, frame_idx])
    common_entropy_map = _to_heatmap(details["entropy_common"], sample_idx=sample_idx)
    common_mask_map = _to_heatmap(details["common_mask_hard"], sample_idx=sample_idx)
    individual_entropy_map = _to_heatmap(details["entropy_individual"][frame_idx], sample_idx=sample_idx)
    individual_mask_map = _to_heatmap(details["individual_masks_hard"][frame_idx], sample_idx=sample_idx)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f'Epoch {epoch + 1}, Batch {batch_idx}, Frame {frame_idx + 1} - Entropy/Mask/Reconstruction')

    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(recon_img)
    axes[0, 1].set_title('Reconstruction')
    axes[0, 1].axis('off')

    im0 = axes[0, 2].imshow(common_entropy_map, cmap='hot')
    axes[0, 2].set_title('Common Entropy Map')
    axes[0, 2].axis('off')
    fig.colorbar(im0, ax=axes[0, 2], fraction=0.046, pad=0.04)

    im1 = axes[1, 0].imshow(common_mask_map, cmap='gray')
    axes[1, 0].set_title('Common Top-K Mask')
    axes[1, 0].axis('off')
    fig.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im2 = axes[1, 1].imshow(individual_entropy_map, cmap='hot')
    axes[1, 1].set_title(f'Individual Entropy Map (Frame {frame_idx + 1})')
    axes[1, 1].axis('off')
    fig.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im3 = axes[1, 2].imshow(individual_mask_map, cmap='gray')
    axes[1, 2].set_title(f'Individual Top-K Mask (Frame {frame_idx + 1})')
    axes[1, 2].axis('off')
    fig.colorbar(im3, ax=axes[1, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'entropy_mask_recon_epoch_{epoch + 1}_batch_{batch_idx}_frame_{frame_idx + 1}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved entropy/mask visualization to {save_path}')

    if was_training:
        model.train()



def visualize_original_reconstruction_error(original, reconstructed, epoch, batch_idx,
                                            frame_idx=0, sample_idx=0,
                                            save_dir="error_analysis"):
    """
    对指定epoch和batch保存原图、重建图、误差图对比。
    """
    os.makedirs(save_dir, exist_ok=True)

    original_img = _to_display_image(original[sample_idx, frame_idx])
    recon_img = _to_display_image(reconstructed[sample_idx, frame_idx])
    error_map = np.abs(original_img - recon_img).mean(axis=2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Epoch {epoch + 1}, Batch {batch_idx}, Frame {frame_idx + 1} - Original/Reconstruction/Error')

    axes[0].imshow(original_img)
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(recon_img)
    axes[1].set_title('Reconstruction')
    axes[1].axis('off')

    im = axes[2].imshow(error_map, cmap='jet')
    axes[2].set_title('Absolute Error Map')
    axes[2].axis('off')
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'error_map_epoch_{epoch + 1}_batch_{batch_idx}_frame_{frame_idx + 1}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved original/reconstruction/error visualization to {save_path}')



def plot_snr_impact(model, gop, epoch, batch_idx,
                    snr_values=None, frame_idx=0, sample_idx=0,
                    save_dir="snr_analysis"):
    """
    生成信道噪声影响图：SNR vs PSNR。
    """
    if snr_values is None:
        snr_values = [0, 5, 10, 15, 20, 25, 30]

    os.makedirs(save_dir, exist_ok=True)
    was_training = model.training
    model.eval()

    psnr_values = []
    with torch.no_grad():
        for snr in snr_values:
            recovered_gop, _, _ = model(gop, channel_snr=snr)
            gop_01 = (gop + 1) / 2
            recovered_gop_01 = ((recovered_gop + 1) / 2).clamp(0, 1)
            mse = F.mse_loss(gop_01, recovered_gop_01).item()
            psnr = 10 * np.log10(1.0 / max(mse, 1e-12))
            psnr_values.append(psnr)

    plt.figure(figsize=(8, 5))
    plt.plot(snr_values, psnr_values, marker='o')
    plt.title(f'SNR Impact on Reconstruction Quality (Epoch {epoch + 1}, Batch {batch_idx})')
    plt.xlabel('SNR (dB)')
    plt.ylabel('PSNR (dB)')
    plt.grid(True)
    save_path = os.path.join(save_dir, f'snr_impact_epoch_{epoch + 1}_batch_{batch_idx}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved SNR impact plot to {save_path}')

    if was_training:
        model.train()


# --------------------------
# 训练函数 (只优化MSE)
# --------------------------
# 修改后的训练函数 (MS-SSIM + PSNR加权损失)
# 修改后的训练函数 (包含损失记录和可视化增强)
# 训练函数 - 使用MS-SSIM和PSNR加权损失
def train_mdvsc(model, dataloader, epochs=10, lr=1e-4, device='cuda', vis_freq=5,
                alpha=0.85, beta=0.15,
                target_visual_epoch=None, target_visual_batch=None,
                target_frame_idx=0, target_sample_idx=0,
                snr_values=None):
    """
    alpha: MS-SSIM损失权重
    beta: PSNR损失权重
    """
    model.train()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(dataloader)
    )

    # 历史记录 - 保存所有数据点
    history = {
        'batch_index': [],
        'total_loss': [],
        'ms_ssim_loss': [],
        'psnr_loss': [],
        'ms_ssim': [],
        'psnr': [],
        'cbr': [],
        'lr': []
    }

    # 新增：用于记录每个epoch的平均指标
    epoch_history = {
        'epoch': [],
        'avg_total_loss': [],
        'avg_ms_ssim': [],
        'avg_psnr': []
    }

    # 训练开始时间
    start_time = time.time()

    for epoch in range(epochs):
        epoch_losses = []
        epoch_ms_ssim = []
        epoch_psnr = []

        for batch_idx, gop in enumerate(dataloader):
            batch_start_time = time.time()
            gop = gop.to(device)

            # 前向传播
            recovered_gop, cbr, stats = model(gop)

            # 将图像转换到[0,1]范围
            gop_01 = (gop + 1) / 2
            recovered_gop_01 = ((recovered_gop + 1) / 2).clamp(0, 1)

            # 计算MS-SSIM及其损失
            ms_ssim_val = ms_ssim(
                gop_01.view(-1, *gop_01.shape[2:]),
                recovered_gop_01.view(-1, *recovered_gop_01.shape[2:]),
                data_range=1.0,
                win_size=11,
                size_average=True
            )
            ms_ssim_loss_val = 1 - ms_ssim_val  # MS-SSIM损失 (0到1范围)

            # 计算PSNR及其损失
            mse = F.mse_loss(gop_01, recovered_gop_01)  # 均方误差
            psnr_val = 10 * torch.log10(1.0 / mse)  # PSNR值
            psnr_loss_val = 1 / (psnr_val.detach() + 1e-8)  # 使用PSNR倒数作为损失项

            # 加权损失组合
            total_loss = alpha * ms_ssim_loss_val + beta * mse

            # 记录epoch指标
            epoch_losses.append(total_loss.item())
            epoch_ms_ssim.append(ms_ssim_val.item())
            epoch_psnr.append(psnr_val.item())

            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()

            # 梯度裁剪防止异常峰值
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # 更新学习率调度器
            scheduler.step()

            # 获取当前学习率
            current_lr = optimizer.param_groups[0]['lr']

            # 计算批处理时间
            batch_time = time.time() - batch_start_time

            # 更新历史记录
            batch_index = len(history['batch_index'])
            history['batch_index'].append(batch_index + 1)
            history['total_loss'].append(total_loss.item())
            history['ms_ssim_loss'].append(ms_ssim_loss_val.item())
            history['psnr_loss'].append(psnr_loss_val.item())
            history['ms_ssim'].append(ms_ssim_val.item())
            history['psnr'].append(psnr_val.item())
            history['cbr'].append(cbr)
            history['lr'].append(current_lr)

            # 每10个batch打印一次性能
            if batch_idx % 10 == 0:
                avg_total = np.mean(history['total_loss'][-50:]) if history['total_loss'] else 0
                avg_ms_ssim = np.mean(history['ms_ssim'][-50:]) if history['ms_ssim'] else 0
                avg_psnr = np.mean(history['psnr'][-50:]) if history['psnr'] else 0
                avg_cbr = np.mean(history['cbr'][-50:]) if history['cbr'] else 0

                print(f'Epoch [{epoch + 1}/{epochs}] Batch [{batch_idx}/{len(dataloader)}]')
                print(
                    f'  Total Loss: {total_loss.item():.4f}, MS-SSIM: {ms_ssim_val.item():.4f}, PSNR: {psnr_val.item():.2f} dB')
                print(
                    f'  Component Loss: MS-SSIM Loss={alpha * ms_ssim_loss_val.item():.4f}, PSNR Loss={beta * psnr_loss_val:.4f}')
                print(f'  CBR: {cbr:.6f}, Common: {stats["common_ratio"]}, Individual: {stats["individual_ratio"]}')
                print(f'  Avg Loss: {avg_total:.4f}, Avg MS-SSIM: {avg_ms_ssim:.4f}, Avg PSNR: {avg_psnr:.2f} dB')
                print(f'  Batch Time: {batch_time:.3f}s, LR: {current_lr:.6f}')

            # 常规原图/重建图可视化
            if vis_freq > 0 and batch_idx % vis_freq == 0:
                visualize_comparison(gop, recovered_gop, epoch, batch_idx)

            # 对指定epoch和batch生成论文图
            if (target_visual_epoch is not None and target_visual_batch is not None and
                    (epoch + 1) == target_visual_epoch and batch_idx == target_visual_batch):
                visualize_entropy_mask_reconstruction(
                    model, gop, epoch, batch_idx,
                    frame_idx=target_frame_idx,
                    sample_idx=target_sample_idx
                )
                visualize_original_reconstruction_error(
                    gop, recovered_gop, epoch, batch_idx,
                    frame_idx=target_frame_idx,
                    sample_idx=target_sample_idx
                )
                plot_snr_impact(
                    model, gop, epoch, batch_idx,
                    snr_values=snr_values,
                    frame_idx=target_frame_idx,
                    sample_idx=target_sample_idx
                )

        # 计算并记录当前epoch的平均指标
        avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else 0
        avg_epoch_ms_ssim = np.mean(epoch_ms_ssim) if epoch_ms_ssim else 0
        avg_epoch_psnr = np.mean(epoch_psnr) if epoch_psnr else 0

        epoch_history['epoch'].append(epoch + 1)
        epoch_history['avg_total_loss'].append(avg_epoch_loss)
        epoch_history['avg_ms_ssim'].append(avg_epoch_ms_ssim)
        epoch_history['avg_psnr'].append(avg_epoch_psnr)

        # 打印epoch摘要
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Avg Total Loss: {avg_epoch_loss:.4f}")
        print(f"  Avg MS-SSIM: {avg_epoch_ms_ssim:.4f}")
        print(f"  Avg PSNR: {avg_epoch_psnr:.2f} dB\n")

        # 保存epoch数据
        save_epoch_data(epoch_history)
        plot_epoch_history(epoch_history)

    # 训练结束保存最终数据
    save_training_data(history, epochs, final=True)
    print("Training completed!")
    return model, history, epoch_history


# 保存训练数据到CSV
def save_training_data(history, epoch, final=False):
    """将训练数据保存到CSV文件"""
    filename = "training_data_final.csv" if final else f"training_data_epoch_{epoch}.csv"

    data = {
        'Batch': history['batch_index'],
        'Total_Loss': history['total_loss'],
        'MS_SSIM_Loss': history['ms_ssim_loss'],
        'PSNR_Loss': history['psnr_loss'],
        'MS_SSIM': history['ms_ssim'],
        'PSNR': history['psnr'],
        'CBR': history['cbr'],
        'Learning_Rate': history['lr']
    }

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Saved training data to {filename}")


# 新增：保存epoch数据
def save_epoch_data(epoch_history):
    """将epoch数据保存到CSV文件"""
    filename = "epoch_history.csv"

    data = {
        'Epoch': epoch_history['epoch'],
        'Avg_Total_Loss': epoch_history['avg_total_loss'],
        'Avg_MS_SSIM': epoch_history['avg_ms_ssim'],
        'Avg_PSNR': epoch_history['avg_psnr']
    }

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Saved epoch data to {filename}")


# 新增：绘制epoch历史图表
def plot_epoch_history(epoch_history):
    """绘制每个epoch的平均指标变化"""
    epochs = epoch_history['epoch']

    # 创建包含三个子图的图表
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    plt.suptitle('Training Performance per Epoch')

    # 总损失变化
    ax1.plot(epochs, epoch_history['avg_total_loss'], 'b-o')
    ax1.set_title('Average Total Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)

    # MS-SSIM变化
    ax2.plot(epochs, epoch_history['avg_ms_ssim'], 'g-s')
    ax2.set_title('Average MS-SSIM')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MS-SSIM')
    ax2.grid(True)

    # PSNR变化
    ax3.plot(epochs, epoch_history['avg_psnr'], 'r-^')
    ax3.set_title('Average PSNR')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('PSNR (dB)')
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig('epoch_performance.png')
    plt.close()
    print("Saved epoch performance visualization to epoch_performance.png")


# 使用示例
if __name__ == "__main__":
    # 设置训练轮数
    total_epochs = 150
s
    # 指定需要额外保存论文图的epoch和batch（epoch从1开始，batch按训练日志中的编号）
    target_visual_epoch = 1
    target_visual_batch = 0
    target_frame_idx = 0
    target_sample_idx = 0
    snr_values = [0, 5, 10, 15, 20, 25, 30]

    # 1. 初始化模型
    model = MDVSCv3(
        in_channels=3,
        latent_dim=64,
        gop_size=6,
        common_ratio=0.2,
        individual_ratio=0.1
    )

    # 2. 准备数据
    dataset = VideoDataset(
        video_path="2.mp4",
        gop_size=6,
        frame_size=(1024, 1024)
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

    # 3. 训练模型
    trained_model, history, epoch_history = train_mdvsc(
        model,
        dataloader,
        epochs=total_epochs,
        lr=1e-3,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        vis_freq=5,
        target_visual_epoch=target_visual_epoch,
        target_visual_batch=target_visual_batch,
        target_frame_idx=target_frame_idx,
        target_sample_idx=target_sample_idx,
        snr_values=snr_values
    )

    # 4. 保存模型
    torch.save(trained_model.state_dict(), "mdvsc_v3_model.pth")

    # 5. 训练后分析
    print("\nTraining Summary:")
    print(f"Final MS-SSIM: {epoch_history['avg_ms_ssim'][-1]:.6f}")
    print(f"Final PSNR: {epoch_history['avg_psnr'][-1]:.2f} dB")
    print(f"Final Loss: {epoch_history['avg_total_loss'][-1]:.6f}")
    print(f"Compression Ratio: {1 / np.mean(history['cbr']):.1f}x")

    # 6. 展示最终重建效果与补充论文图
    trained_model.eval()
    with torch.no_grad():
        test_batch = next(iter(dataloader))
        test_batch = test_batch.to('cuda' if torch.cuda.is_available() else 'cpu')
        recovered, cbr, _ = trained_model(test_batch)
        visualize_comparison(test_batch, recovered, total_epochs, 0, "final_results")
        visualize_original_reconstruction_error(
            test_batch, recovered, total_epochs, 0,
            frame_idx=target_frame_idx,
            sample_idx=target_sample_idx,
            save_dir="final_results"
        )

    visualize_entropy_mask_reconstruction(
        trained_model, test_batch, total_epochs, 0,
        frame_idx=target_frame_idx,
        sample_idx=target_sample_idx,
        save_dir="final_results"
    )
    plot_snr_impact(
        trained_model, test_batch, total_epochs, 0,
        snr_values=snr_values,
        frame_idx=target_frame_idx,
        sample_idx=target_sample_idx,
        save_dir="final_results"
    )
