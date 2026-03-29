from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LevelTransmissionStats:
    level: int
    latent_dim: int
    block_size: int
    common_active_ratio: float
    individual_active_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MDVSCOutput:
    restored_sequences: list[torch.Tensor]
    reconstructed_frames: torch.Tensor
    level_stats: list[LevelTransmissionStats]
    common_masks: list[torch.Tensor]
    individual_masks: list[torch.Tensor]

    def stats_dict(self) -> list[dict[str, Any]]:
        return [stat.to_dict() for stat in self.level_stats]


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.activation = nn.GELU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        outputs = self.activation(self.conv1(inputs))
        outputs = self.conv2(outputs)
        return self.activation(outputs + residual)


class ChannelImportanceGate(nn.Module):
    def __init__(self, keep_ratio: float, temperature: float = 0.1):
        super().__init__()
        self.keep_ratio = keep_ratio
        self.temperature = temperature

    def forward(self, features: torch.Tensor, enabled: bool) -> torch.Tensor:
        batch_size, channels, _, _ = features.shape
        if not enabled:
            return torch.ones((batch_size, channels, 1, 1), device=features.device, dtype=features.dtype)

        scores = features.abs().mean(dim=(2, 3))
        flat_mask = self._build_topk_mask(scores)
        return flat_mask.view(batch_size, channels, 1, 1)

    def _build_topk_mask(self, scores: torch.Tensor) -> torch.Tensor:
        total = scores.shape[1]
        keep = max(1, min(total, int(round(total * self.keep_ratio))))
        top_values, top_indices = torch.topk(scores, keep, dim=1)
        hard_mask = torch.zeros_like(scores)
        hard_mask.scatter_(1, top_indices, 1.0)

        if not self.training:
            return hard_mask

        threshold = top_values[:, -1:].detach()
        soft_mask = torch.sigmoid((scores - threshold) / self.temperature)
        soft_mask = soft_mask * (keep / soft_mask.sum(dim=1, keepdim=True).clamp_min(1e-6))
        soft_mask = soft_mask.clamp(0.0, 1.0)
        return hard_mask.detach() - soft_mask.detach() + soft_mask


class BlockImportanceGate(nn.Module):
    def __init__(self, keep_ratio: float, block_size: int, temperature: float = 0.1):
        super().__init__()
        self.keep_ratio = keep_ratio
        self.block_size = block_size
        self.temperature = temperature

    def forward(self, features: torch.Tensor, enabled: bool) -> torch.Tensor:
        batch_size, _, height, width = features.shape
        if not enabled:
            return torch.ones((batch_size, 1, height, width), device=features.device, dtype=features.dtype)

        pad_height = (self.block_size - (height % self.block_size)) % self.block_size
        pad_width = (self.block_size - (width % self.block_size)) % self.block_size
        pooled_source = F.pad(features.abs().mean(dim=1, keepdim=True), (0, pad_width, 0, pad_height))
        pooled = F.avg_pool2d(pooled_source, kernel_size=self.block_size, stride=self.block_size)
        flat_scores = pooled.flatten(1)
        flat_mask = self._build_topk_mask(flat_scores)
        block_mask = flat_mask.view(batch_size, 1, pooled.shape[-2], pooled.shape[-1])
        full_mask = block_mask.repeat_interleave(self.block_size, dim=2).repeat_interleave(self.block_size, dim=3)
        return full_mask[:, :, :height, :width]

    def _build_topk_mask(self, scores: torch.Tensor) -> torch.Tensor:
        total = scores.shape[1]
        keep = max(1, min(total, int(round(total * self.keep_ratio))))
        top_values, top_indices = torch.topk(scores, keep, dim=1)
        hard_mask = torch.zeros_like(scores)
        hard_mask.scatter_(1, top_indices, 1.0)

        if not self.training:
            return hard_mask

        threshold = top_values[:, -1:].detach()
        soft_mask = torch.sigmoid((scores - threshold) / self.temperature)
        soft_mask = soft_mask * (keep / soft_mask.sum(dim=1, keepdim=True).clamp_min(1e-6))
        soft_mask = soft_mask.clamp(0.0, 1.0)
        return hard_mask.detach() - soft_mask.detach() + soft_mask


class PerLevelMDVSC(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        common_keep_ratio: float,
        individual_keep_ratio: float,
        block_size: int,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.block_size = block_size

        self.feature_adaptor = nn.Sequential(
            nn.Conv2d(in_channels, latent_dim, kernel_size=1),
            nn.GELU(),
            ResidualBlock(latent_dim),
        )
        self.latent_encoder = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1),
            nn.GELU(),
            ResidualBlock(latent_dim),
        )
        self.latent_decoder = nn.Sequential(
            ResidualBlock(latent_dim),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.feature_decoder = nn.Sequential(
            ResidualBlock(latent_dim),
            nn.Conv2d(latent_dim, in_channels, kernel_size=1),
        )
        self.refinement = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GELU(),
            ResidualBlock(in_channels),
        )
        self.common_gate = ChannelImportanceGate(common_keep_ratio)
        self.individual_gate = BlockImportanceGate(individual_keep_ratio, block_size)

    def forward(
        self,
        feature_sequence: torch.Tensor,
        apply_masks: bool,
        channel_mode: str,
        snr_db: float | None,
        level_index: int,
    ) -> tuple[torch.Tensor, LevelTransmissionStats, torch.Tensor, torch.Tensor]:
        batch_size, time_steps, channels, height, width = feature_sequence.shape
        flat_sequence = feature_sequence.reshape(batch_size * time_steps, channels, height, width)
        latent_sequence = self.latent_encoder(self.feature_adaptor(flat_sequence))
        latent_sequence = latent_sequence.view(batch_size, time_steps, self.latent_dim, height, width)

        common_latent = latent_sequence.mean(dim=1)
        individual_latent = latent_sequence - common_latent.unsqueeze(1)

        common_mask = self.common_gate(common_latent, enabled=apply_masks)
        transmitted_common = self._transmit(common_latent * common_mask, channel_mode, snr_db)

        restored_frames: list[torch.Tensor] = []
        individual_masks: list[torch.Tensor] = []
        for frame_index in range(time_steps):
            frame_residual = individual_latent[:, frame_index]
            individual_mask = self.individual_gate(frame_residual, enabled=apply_masks)
            individual_masks.append(individual_mask)
            transmitted_individual = self._transmit(frame_residual * individual_mask, channel_mode, snr_db)
            decoded = self.latent_decoder(transmitted_common + transmitted_individual)
            restored = self.refinement(self.feature_decoder(decoded))
            restored_frames.append(restored)

        restored_sequence = torch.stack(restored_frames, dim=1)
        individual_mask_tensor = torch.stack(individual_masks, dim=1)
        stats = LevelTransmissionStats(
            level=level_index,
            latent_dim=self.latent_dim,
            block_size=self.block_size,
            common_active_ratio=float(common_mask.detach().mean().item()),
            individual_active_ratio=float(individual_mask_tensor.detach().mean().item()),
        )
        return restored_sequence, stats, common_mask, individual_mask_tensor

    @staticmethod
    def _transmit(features: torch.Tensor, channel_mode: str, snr_db: float | None) -> torch.Tensor:
        if channel_mode == "identity" or snr_db is None:
            return features
        if channel_mode != "awgn":
            raise ValueError(f"Unsupported channel mode: {channel_mode}")

        signal_power = features.pow(2).mean(dim=(1, 2, 3), keepdim=True)
        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_power = signal_power / max(snr_linear, 1e-6)
        noise = torch.randn_like(features) * noise_power.clamp_min(1e-8).sqrt()
        return features + noise


class ReconstructionHead(nn.Module):
    def __init__(self, feature_channels: list[int], hidden_channels: int = 128):
        super().__init__()
        if len(feature_channels) != 3:
            raise ValueError("ReconstructionHead expects exactly three feature levels")

        self.level2_proj = nn.Sequential(
            nn.Conv2d(feature_channels[2], hidden_channels, kernel_size=1),
            ResidualBlock(hidden_channels),
        )
        self.level1_proj = nn.Sequential(
            nn.Conv2d(feature_channels[1], hidden_channels, kernel_size=1),
            ResidualBlock(hidden_channels),
        )
        self.level0_proj = nn.Sequential(
            nn.Conv2d(feature_channels[0], hidden_channels, kernel_size=1),
            ResidualBlock(hidden_channels),
        )
        self.mid_fusion = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            ResidualBlock(hidden_channels),
        )
        self.low_fusion = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            ResidualBlock(hidden_channels),
        )
        self.output_layer = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels // 2, 3, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, feature_maps: list[torch.Tensor], output_size: tuple[int, int]) -> torch.Tensor:
        level2 = self.level2_proj(feature_maps[2])
        level1 = self.level1_proj(feature_maps[1])
        level0 = self.level0_proj(feature_maps[0])

        fused = F.interpolate(level2, size=level1.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.mid_fusion(torch.cat([fused, level1], dim=1))
        fused = F.interpolate(fused, size=level0.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.low_fusion(torch.cat([fused, level0], dim=1))
        fused = F.interpolate(fused, size=output_size, mode="bilinear", align_corners=False)
        return self.output_layer(fused)


class ProjectMDVSC(nn.Module):
    def __init__(
        self,
        feature_channels: list[int] | tuple[int, ...] = (256, 256, 256),
        latent_dims: list[int] | tuple[int, ...] = (48, 64, 96),
        common_keep_ratios: list[float] | tuple[float, ...] = (0.5, 0.625, 0.75),
        individual_keep_ratios: list[float] | tuple[float, ...] = (0.125, 0.1875, 0.25),
        block_sizes: list[int] | tuple[int, ...] = (8, 4, 2),
    ):
        super().__init__()
        self.feature_channels = list(feature_channels)
        self.latent_dims = list(latent_dims)
        self.common_keep_ratios = list(common_keep_ratios)
        self.individual_keep_ratios = list(individual_keep_ratios)
        self.block_sizes = list(block_sizes)

        if not (
            len(self.feature_channels)
            == len(self.latent_dims)
            == len(self.common_keep_ratios)
            == len(self.individual_keep_ratios)
            == len(self.block_sizes)
        ):
            raise ValueError("All MDVSC per-level configuration lists must have the same length")

        self.level_modules = nn.ModuleList(
            PerLevelMDVSC(
                in_channels=in_channels,
                latent_dim=latent_dim,
                common_keep_ratio=common_keep_ratio,
                individual_keep_ratio=individual_keep_ratio,
                block_size=block_size,
            )
            for in_channels, latent_dim, common_keep_ratio, individual_keep_ratio, block_size in zip(
                self.feature_channels,
                self.latent_dims,
                self.common_keep_ratios,
                self.individual_keep_ratios,
                self.block_sizes,
            )
        )
        self.reconstruction_head = ReconstructionHead(self.feature_channels)

    def forward(
        self,
        feature_sequences: list[torch.Tensor],
        output_size: tuple[int, int],
        apply_masks: bool = True,
        channel_mode: str = "identity",
        snr_db: float | None = None,
    ) -> MDVSCOutput:
        if len(feature_sequences) != len(self.level_modules):
            raise ValueError("feature_sequences must match the configured number of levels")

        restored_sequences: list[torch.Tensor] = []
        level_stats: list[LevelTransmissionStats] = []
        common_masks: list[torch.Tensor] = []
        individual_masks: list[torch.Tensor] = []
        for level_index, (level_module, feature_sequence) in enumerate(zip(self.level_modules, feature_sequences)):
            restored_sequence, stats, common_mask, individual_mask = level_module(
                feature_sequence,
                apply_masks=apply_masks,
                channel_mode=channel_mode,
                snr_db=snr_db,
                level_index=level_index,
            )
            restored_sequences.append(restored_sequence)
            level_stats.append(stats)
            common_masks.append(common_mask)
            individual_masks.append(individual_mask)

        time_steps = restored_sequences[0].shape[1]
        reconstructed_frames = torch.stack(
            [
                self.reconstruction_head(
                    [level_sequence[:, frame_index] for level_sequence in restored_sequences],
                    output_size=output_size,
                )
                for frame_index in range(time_steps)
            ],
            dim=1,
        )
        return MDVSCOutput(
            restored_sequences=restored_sequences,
            reconstructed_frames=reconstructed_frames,
            level_stats=level_stats,
            common_masks=common_masks,
            individual_masks=individual_masks,
        )