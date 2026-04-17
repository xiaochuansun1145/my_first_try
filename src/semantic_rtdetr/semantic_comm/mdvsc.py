from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint_util


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
    detection_sequences: list[torch.Tensor]
    reconstruction_sequences: list[torch.Tensor]
    reconstructed_frames: torch.Tensor
    reconstructed_base_frames: torch.Tensor
    reconstructed_high_frequency_residuals: torch.Tensor
    level_stats: list[LevelTransmissionStats]
    common_masks: list[torch.Tensor]
    individual_masks: list[torch.Tensor]

    def stats_dict(self) -> list[dict[str, Any]]:
        return [stat.to_dict() for stat in self.level_stats]


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        hidden_channels = channels
        groups = _group_count(channels)
        self.norm = SafeGroupNorm(groups, channels)
        self.expand = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.depthwise = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, groups=hidden_channels)
        self.project = nn.Conv2d(hidden_channels, channels, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        outputs = self.expand(F.gelu(self.norm(inputs)))
        outputs = self.depthwise(F.gelu(outputs))
        outputs = self.project(outputs)
        return residual + outputs


def _group_count(channels: int, preferred_groups: int = 8) -> int:
    for groups in range(min(preferred_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class SafeGroupNorm(nn.Module):
    def __init__(self, num_groups: int, num_channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 4 and inputs.shape[0] * inputs.shape[2] * inputs.shape[3] <= 1:
            return inputs
        return self.norm(inputs)


class ReconstructionResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        hidden_channels = channels
        groups = _group_count(channels)
        self.norm1 = SafeGroupNorm(groups, channels)
        self.norm2 = SafeGroupNorm(_group_count(hidden_channels), hidden_channels)
        self.conv1 = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, groups=hidden_channels)
        self.conv3 = nn.Conv2d(hidden_channels, channels, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        outputs = self.conv1(F.gelu(self.norm1(inputs)))
        outputs = self.conv2(F.gelu(self.norm2(outputs)))
        outputs = self.conv3(outputs)
        return outputs + residual


class FusionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        groups = _group_count(out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            SafeGroupNorm(groups, out_channels),
            nn.GELU(),
            ReconstructionResidualBlock(out_channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class TaskAdaptationBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        bottleneck_channels = max(channels // 4, 32)
        groups = _group_count(bottleneck_channels)
        self.block = nn.Sequential(
            nn.Conv2d(channels, bottleneck_channels, kernel_size=1),
            SafeGroupNorm(groups, bottleneck_channels),
            nn.GELU(),
            nn.Conv2d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=3,
                padding=1,
                groups=bottleneck_channels,
            ),
            nn.GELU(),
            nn.Conv2d(bottleneck_channels, channels, kernel_size=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.block(inputs)


class UpsampleRefineBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        groups = _group_count(out_channels)
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            SafeGroupNorm(groups, out_channels),
            nn.GELU(),
            ReconstructionResidualBlock(out_channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class ChannelBlockImportanceGate(nn.Module):
    def __init__(self, keep_ratio: float, block_size: int, temperature: float = 0.1):
        super().__init__()
        self.keep_ratio = keep_ratio
        self.block_size = block_size
        self.temperature = temperature

    def forward(self, features: torch.Tensor, enabled: bool) -> torch.Tensor:
        batch_size, channels, height, width = features.shape
        if not enabled:
            return torch.ones((batch_size, channels, height, width), device=features.device, dtype=features.dtype)

        pad_height = (self.block_size - (height % self.block_size)) % self.block_size
        pad_width = (self.block_size - (width % self.block_size)) % self.block_size
        pooled_source = F.pad(features.abs(), (0, pad_width, 0, pad_height))
        pooled = F.avg_pool2d(
            pooled_source.reshape(batch_size * channels, 1, height + pad_height, width + pad_width),
            kernel_size=self.block_size,
            stride=self.block_size,
        )
        pooled_height, pooled_width = pooled.shape[-2:]
        scores = pooled.view(batch_size, channels, pooled_height * pooled_width)
        flat_mask = self._build_topk_mask(scores)
        block_mask = flat_mask.view(batch_size, channels, pooled_height, pooled_width)
        full_mask = block_mask.repeat_interleave(self.block_size, dim=2).repeat_interleave(self.block_size, dim=3)
        return full_mask[:, :, :height, :width]

    def _build_topk_mask(self, scores: torch.Tensor) -> torch.Tensor:
        total = scores.shape[-1]
        keep = max(1, min(total, int(round(total * self.keep_ratio))))
        top_values, top_indices = torch.topk(scores, keep, dim=-1)
        hard_mask = torch.zeros_like(scores)
        hard_mask.scatter_(-1, top_indices, 1.0)

        if not self.training:
            return hard_mask

        threshold = top_values[..., -1:].detach()
        soft_mask = torch.sigmoid((scores - threshold) / self.temperature)
        soft_mask = soft_mask * (keep / soft_mask.sum(dim=-1, keepdim=True).clamp_min(1e-6))
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
        self.latent_encoder = nn.Sequential(ResidualBlock(latent_dim), nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1))
        self.latent_decoder = nn.Sequential(
            ResidualBlock(latent_dim),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.feature_decoder = nn.Sequential(
            nn.Conv2d(latent_dim, in_channels, kernel_size=1),
            nn.GELU(),
        )
        self.refinement = TaskAdaptationBlock(in_channels)
        self.common_gate = ChannelBlockImportanceGate(common_keep_ratio, block_size)
        self.individual_gate = ChannelBlockImportanceGate(individual_keep_ratio, block_size)

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


class LightUpsampleBlock(nn.Module):
    """Bilinear 2x upsample + depth-wise separable conv.

    Much lighter than ``UpsampleRefineBlock`` (no full 3x3 conv or ResBlock
    at high resolution) and more parameter-efficient than PixelShuffle
    (no 4x channel expansion in the conv).  The DW-separable structure
    still provides learned spatial refinement at each scale.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        groups = _group_count(out_channels)
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels),
            SafeGroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class LightReconstructionHead(nn.Module):
    """Lightweight reconstruction head with additive FPN and DW-separable upsampling.

    Compared to ``ReconstructionHead``:
    * ~55 % fewer parameters (additive FPN, no separate detail branch,
      DW-separable upsample instead of Conv3x3 + ResBlock)
    * Significantly less peak VRAM (no heavy convolutions at 320x320 / 640x640)
    * Learnable high-frequency residual scale (initial 0.2 instead of fixed 0.15)
    * Optional gradient-checkpointed upsampling stages for further VRAM savings
    """

    def __init__(
        self,
        feature_channels: list[int],
        hidden_channels: int = 128,
        detail_channels: int = 48,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        if len(feature_channels) != 3:
            raise ValueError("LightReconstructionHead expects exactly three feature levels")
        self.use_checkpoint = use_checkpoint

        # ---------- 1x1 projections to a common channel dimension ----------
        self.proj0 = nn.Conv2d(feature_channels[0], hidden_channels, kernel_size=1)
        self.proj1 = nn.Conv2d(feature_channels[1], hidden_channels, kernel_size=1)
        self.proj2 = nn.Conv2d(feature_channels[2], hidden_channels, kernel_size=1)

        # ---------- Post-fusion depth-wise separable refinement (stride 8) ----------
        groups = _group_count(hidden_channels)
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, groups=hidden_channels),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            SafeGroupNorm(groups, hidden_channels),
            nn.GELU(),
        )

        # ---------- Lightweight 8x upsampling: stride-8 -> stride-1 ----------
        self.mid_channels = max(hidden_channels // 2, detail_channels)
        self.output_channels = max(detail_channels, 32)
        self.up1 = LightUpsampleBlock(hidden_channels, self.mid_channels)
        self.up2 = LightUpsampleBlock(self.mid_channels, self.output_channels)
        self.up3 = LightUpsampleBlock(self.output_channels, self.output_channels)

        # ---------- Output heads ----------
        self.base_head = nn.Sequential(
            nn.Conv2d(self.output_channels, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.detail_head = nn.Conv2d(self.output_channels, 3, kernel_size=3, padding=1)
        self.hf_scale = nn.Parameter(torch.tensor(0.2))

    def decode_components(
        self,
        feature_maps: list[torch.Tensor],
        output_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Project all levels to the same channel dimension
        f0 = self.proj0(feature_maps[0])  # stride 8
        f1 = self.proj1(feature_maps[1])  # stride 16
        f2 = self.proj2(feature_maps[2])  # stride 32

        # Top-down additive FPN
        f1 = f1 + F.interpolate(f2, size=f1.shape[-2:], mode="nearest")
        f0 = f0 + F.interpolate(f1, size=f0.shape[-2:], mode="nearest")

        # Refine at stride 8
        fused = self.refine(f0)

        # Progressive lightweight upsampling
        if self.use_checkpoint and self.training:
            x = checkpoint_util.checkpoint(self.up1, fused, use_reentrant=False)
            x = checkpoint_util.checkpoint(self.up2, x, use_reentrant=False)
            x = checkpoint_util.checkpoint(self.up3, x, use_reentrant=False)
        else:
            x = self.up1(fused)
            x = self.up2(x)
            x = self.up3(x)

        if x.shape[-2:] != output_size:
            x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)

        base = self.base_head(x)
        high_frequency_residual = self.hf_scale * torch.tanh(self.detail_head(x))
        reconstructed = (base + high_frequency_residual).clamp(0.0, 1.0)
        return reconstructed, base, high_frequency_residual

    def forward(self, feature_maps: list[torch.Tensor], output_size: tuple[int, int]) -> torch.Tensor:
        reconstructed, _, _ = self.decode_components(feature_maps, output_size)
        return reconstructed


class ReconstructionHead(nn.Module):
    def __init__(
        self,
        feature_channels: list[int],
        hidden_channels: int = 256,
        detail_channels: int = 128,
    ):
        super().__init__()
        if len(feature_channels) != 3:
            raise ValueError("ReconstructionHead expects exactly three feature levels")

        self.top_channels = hidden_channels
        self.mid_channels = max(hidden_channels // 2, detail_channels)
        self.low_channels = max(self.mid_channels // 2, detail_channels)
        self.output_channels = max(detail_channels // 2, 32)

        self.level2_semantic_proj = FusionBlock(feature_channels[2], self.top_channels)
        self.level1_semantic_proj = FusionBlock(feature_channels[1], self.mid_channels)
        self.level0_semantic_proj = FusionBlock(feature_channels[0], self.low_channels)
        self.level0_detail_proj = FusionBlock(feature_channels[0], detail_channels)

        self.fuse_16 = FusionBlock(self.top_channels + self.mid_channels, self.mid_channels)
        self.fuse_8 = FusionBlock(self.mid_channels + self.low_channels, self.low_channels)
        self.detail_fuse_8 = FusionBlock(self.low_channels + detail_channels, self.low_channels)

        self.up_stage1 = UpsampleRefineBlock(self.low_channels, self.mid_channels)
        self.up_stage2 = UpsampleRefineBlock(self.mid_channels, self.low_channels)
        self.up_stage3 = UpsampleRefineBlock(self.low_channels, self.output_channels)
        self.output_refinement = ReconstructionResidualBlock(self.output_channels)
        self.high_frequency_refinement = ReconstructionResidualBlock(self.output_channels)
        self.base_layer = nn.Sequential(
            nn.Conv2d(self.output_channels, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.detail_layer = nn.Conv2d(self.output_channels, 3, kernel_size=3, padding=1)

    def decode_components(
        self,
        feature_maps: list[torch.Tensor],
        output_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        level2 = self.level2_semantic_proj(feature_maps[2])
        level1 = self.level1_semantic_proj(feature_maps[1])
        level0_semantic = self.level0_semantic_proj(feature_maps[0])
        level0_detail = self.level0_detail_proj(feature_maps[0])

        fused_16 = self.fuse_16(
            torch.cat(
                [F.interpolate(level2, size=level1.shape[-2:], mode="nearest"), level1],
                dim=1,
            )
        )
        fused_8 = self.fuse_8(
            torch.cat(
                [F.interpolate(fused_16, size=level0_semantic.shape[-2:], mode="nearest"), level0_semantic],
                dim=1,
            )
        )
        fused_8 = self.detail_fuse_8(torch.cat([fused_8, level0_detail], dim=1))

        decoded = self.up_stage1(fused_8)
        decoded = self.up_stage2(decoded)
        decoded = self.up_stage3(decoded)
        if decoded.shape[-2:] != output_size:
            decoded = F.interpolate(decoded, size=output_size, mode="bilinear", align_corners=False)
        decoded = self.output_refinement(decoded)
        base = self.base_layer(decoded)
        high_frequency_features = self.high_frequency_refinement(decoded)
        high_frequency_residual = 0.15 * torch.tanh(self.detail_layer(high_frequency_features))
        reconstructed = (base + high_frequency_residual).clamp(0.0, 1.0)
        return reconstructed, base, high_frequency_residual

    def forward(self, feature_maps: list[torch.Tensor], output_size: tuple[int, int]) -> torch.Tensor:
        reconstructed, _, _ = self.decode_components(feature_maps, output_size)
        return reconstructed


class ProjectMDVSC(nn.Module):
    def __init__(
        self,
        feature_channels: list[int] | tuple[int, ...] = (256, 256, 256),
        latent_dims: list[int] | tuple[int, ...] = (48, 64, 96),
        common_keep_ratios: list[float] | tuple[float, ...] = (0.5, 0.625, 0.75),
        individual_keep_ratios: list[float] | tuple[float, ...] = (0.125, 0.1875, 0.25),
        block_sizes: list[int] | tuple[int, ...] = (8, 4, 2),
        reconstruction_hidden_channels: int = 256,
        reconstruction_detail_channels: int = 128,
        reconstruction_head_type: str = "light",
        reconstruction_use_checkpoint: bool = False,
    ):
        super().__init__()
        self.feature_channels = list(feature_channels)
        self.latent_dims = list(latent_dims)
        self.common_keep_ratios = list(common_keep_ratios)
        self.individual_keep_ratios = list(individual_keep_ratios)
        self.block_sizes = list(block_sizes)
        self.reconstruction_hidden_channels = reconstruction_hidden_channels
        self.reconstruction_detail_channels = reconstruction_detail_channels
        self.reconstruction_head_type = reconstruction_head_type
        self.reconstruction_use_checkpoint = reconstruction_use_checkpoint

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
        if self.reconstruction_head_type == "light":
            self.reconstruction_head = LightReconstructionHead(
                self.feature_channels,
                hidden_channels=self.reconstruction_hidden_channels,
                detail_channels=self.reconstruction_detail_channels,
                use_checkpoint=self.reconstruction_use_checkpoint,
            )
        else:
            self.reconstruction_head = ReconstructionHead(
                self.feature_channels,
                hidden_channels=self.reconstruction_hidden_channels,
                detail_channels=self.reconstruction_detail_channels,
            )
        self.detection_refinement_heads = nn.ModuleList(
            TaskAdaptationBlock(channels) for channels in self.feature_channels
        )
        self.reconstruction_refinement_heads = nn.ModuleList(
            TaskAdaptationBlock(channels) for channels in self.feature_channels
        )

    @staticmethod
    def _apply_refinement(
        feature_sequences: list[torch.Tensor],
        refinement_heads: nn.ModuleList,
    ) -> list[torch.Tensor]:
        refined_sequences: list[torch.Tensor] = []
        for feature_sequence, refinement_head in zip(feature_sequences, refinement_heads):
            batch_size, time_steps, channels, height, width = feature_sequence.shape
            flattened = feature_sequence.reshape(batch_size * time_steps, channels, height, width)
            refined = refinement_head(flattened)
            refined_sequences.append(refined.reshape(batch_size, time_steps, channels, height, width))
        return refined_sequences

    def _decode_reconstruction_sequences(
        self,
        feature_sequences: list[torch.Tensor],
        output_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        time_steps = feature_sequences[0].shape[1]
        reconstructed_frames: list[torch.Tensor] = []
        reconstructed_base_frames: list[torch.Tensor] = []
        reconstructed_high_frequency_residuals: list[torch.Tensor] = []
        for frame_index in range(time_steps):
            reconstructed, base, high_frequency_residual = self.reconstruction_head.decode_components(
                [level_sequence[:, frame_index] for level_sequence in feature_sequences],
                output_size=output_size,
            )
            reconstructed_frames.append(reconstructed)
            reconstructed_base_frames.append(base)
            reconstructed_high_frequency_residuals.append(high_frequency_residual)

        return (
            torch.stack(reconstructed_frames, dim=1),
            torch.stack(reconstructed_base_frames, dim=1),
            torch.stack(reconstructed_high_frequency_residuals, dim=1),
        )

    def reconstruct_from_feature_sequences(
        self,
        feature_sequences: list[torch.Tensor],
        output_size: tuple[int, int],
    ) -> MDVSCOutput:
        if len(feature_sequences) != len(self.level_modules):
            raise ValueError("feature_sequences must match the configured number of levels")

        detection_sequences = self._apply_refinement(feature_sequences, self.detection_refinement_heads)
        reconstruction_sequences = self._apply_refinement(feature_sequences, self.reconstruction_refinement_heads)
        reconstructed_frames, reconstructed_base_frames, reconstructed_high_frequency_residuals = self._decode_reconstruction_sequences(
            reconstruction_sequences,
            output_size=output_size,
        )

        level_stats: list[LevelTransmissionStats] = []
        common_masks: list[torch.Tensor] = []
        individual_masks: list[torch.Tensor] = []
        for level_index, feature_sequence in enumerate(feature_sequences):
            batch_size, time_steps, _channels, height, width = feature_sequence.shape
            common_masks.append(
                torch.ones(
                    (batch_size, self.latent_dims[level_index], height, width),
                    device=feature_sequence.device,
                    dtype=feature_sequence.dtype,
                )
            )
            individual_masks.append(
                torch.ones(
                    (batch_size, time_steps, self.latent_dims[level_index], height, width),
                    device=feature_sequence.device,
                    dtype=feature_sequence.dtype,
                )
            )
            level_stats.append(
                LevelTransmissionStats(
                    level=level_index,
                    latent_dim=self.latent_dims[level_index],
                    block_size=self.block_sizes[level_index],
                    common_active_ratio=1.0,
                    individual_active_ratio=1.0,
                )
            )

        return MDVSCOutput(
            restored_sequences=list(feature_sequences),
            detection_sequences=detection_sequences,
            reconstruction_sequences=reconstruction_sequences,
            reconstructed_frames=reconstructed_frames,
            reconstructed_base_frames=reconstructed_base_frames,
            reconstructed_high_frequency_residuals=reconstructed_high_frequency_residuals,
            level_stats=level_stats,
            common_masks=common_masks,
            individual_masks=individual_masks,
        )

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

        detection_sequences = self._apply_refinement(restored_sequences, self.detection_refinement_heads)
        reconstruction_sequences = self._apply_refinement(restored_sequences, self.reconstruction_refinement_heads)
        reconstructed_frames, reconstructed_base_frames, reconstructed_high_frequency_residuals = self._decode_reconstruction_sequences(
            reconstruction_sequences,
            output_size=output_size,
        )
        return MDVSCOutput(
            restored_sequences=restored_sequences,
            detection_sequences=detection_sequences,
            reconstruction_sequences=reconstruction_sequences,
            reconstructed_frames=reconstructed_frames,
            reconstructed_base_frames=reconstructed_base_frames,
            reconstructed_high_frequency_residuals=reconstructed_high_frequency_residuals,
            level_stats=level_stats,
            common_masks=common_masks,
            individual_masks=individual_masks,
        )