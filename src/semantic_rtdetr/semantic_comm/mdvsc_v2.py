"""MDVSC v2 – Progressive compression with skip connections, learnable temporal
decomposition, and entropy-based importance masking.

Key differences from v1 (``mdvsc.py``):
* **Progressive U-Net encoder/decoder** per level with skip connections.
* **Learnable temporal attention** for common/individual decomposition instead
  of hard mean.
* **Entropy model network** that predicts per-element entropy scores; top-k is
  applied on these entropy scores (lower entropy → more important → keep)
  instead of on the raw feature magnitudes.
* **SE channel attention** in each encoder stage.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _group_count(channels: int, preferred: int = 8) -> int:
    for g in range(min(preferred, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class SafeGroupNorm(nn.Module):
    def __init__(self, num_groups: int, num_channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4 and x.shape[0] * x.shape[2] * x.shape[3] <= 1:
            return x
        return self.norm(x)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc1 = nn.Conv2d(channels, mid, 1)
        self.fc2 = nn.Conv2d(mid, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(2, 3), keepdim=True)
        s = torch.sigmoid(self.fc2(F.gelu(self.fc1(s))))
        return x * s


class ResBlock(nn.Module):
    """Depthwise-separable residual block with GroupNorm."""

    def __init__(self, channels: int):
        super().__init__()
        g = _group_count(channels)
        self.body = nn.Sequential(
            SafeGroupNorm(g, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class DownBlock(nn.Module):
    """Halve channels: conv1x1 + 2×ResBlock + SE."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, 1)
        self.res1 = ResBlock(out_ch)
        self.res2 = ResBlock(out_ch)
        self.se = SEBlock(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.res2(self.res1(x))
        return self.se(x)


class UpBlock(nn.Module):
    """Double channels back: concat skip → conv1x1 → 2×ResBlock."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch + skip_ch, out_ch, 1)
        self.res1 = ResBlock(out_ch)
        self.res2 = ResBlock(out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, skip], dim=1)
        x = self.proj(x)
        return self.res2(self.res1(x))


# ---------------------------------------------------------------------------
# Progressive encoder / decoder with skip connections
# ---------------------------------------------------------------------------

class ProgressiveEncoder(nn.Module):
    """``in_channels → mid → latent_dim`` with skip connections at each stage."""

    def __init__(self, in_channels: int, latent_dim: int):
        super().__init__()
        mid = max((in_channels + latent_dim) // 2, latent_dim)
        # stage 0: in_channels → mid
        self.down0 = DownBlock(in_channels, mid)
        # stage 1: mid → latent_dim
        self.down1 = DownBlock(mid, latent_dim)
        # bottleneck
        self.bottleneck = ResBlock(latent_dim)
        self.skip0_channels = mid

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (latent, skip0)."""
        s0 = self.down0(x)         # [B, mid, H, W]
        z = self.down1(s0)         # [B, latent, H, W]
        z = self.bottleneck(z)
        return z, s0


class ProgressiveDecoder(nn.Module):
    """``latent_dim → mid → in_channels`` consuming skip connections."""

    def __init__(self, in_channels: int, latent_dim: int):
        super().__init__()
        mid = max((in_channels + latent_dim) // 2, latent_dim)
        self.up0 = UpBlock(latent_dim, mid, mid)
        self.up1 = nn.Sequential(
            nn.Conv2d(mid, in_channels, 1),
            ResBlock(in_channels),
            ResBlock(in_channels),
        )

    def forward(self, z: torch.Tensor, skip0: torch.Tensor) -> torch.Tensor:
        x = self.up0(z, skip0)   # [B, mid, H, W]
        x = self.up1(x)          # [B, in_channels, H, W]
        return x


# ---------------------------------------------------------------------------
# Learnable temporal decomposition
# ---------------------------------------------------------------------------

class TemporalAttentionDecomposer(nn.Module):
    """Produce per-frame attention weights via a lightweight 1×1 conv,
    then compute ``common = weighted_sum`` and ``individual = x - common``.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Conv2d(channels, max(channels // 4, 8), 1),
            nn.GELU(),
            nn.Conv2d(max(channels // 4, 8), 1, 1),
        )

    def forward(self, latent_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            latent_seq: [B, T, C, H, W]
        Returns:
            common:     [B, C, H, W]
            individual: [B, T, C, H, W]
        """
        B, T, C, H, W = latent_seq.shape
        flat = latent_seq.reshape(B * T, C, H, W)
        scores = self.score_net(flat)                         # [B*T, 1, H, W]
        scores = scores.view(B, T, 1, H, W)
        weights = torch.softmax(scores, dim=1)                # [B, T, 1, H, W]
        common = (weights * latent_seq).sum(dim=1)            # [B, C, H, W]
        individual = latent_seq - common.unsqueeze(1)         # [B, T, C, H, W]
        return common, individual


# ---------------------------------------------------------------------------
# Entropy-based importance masking
# ---------------------------------------------------------------------------

class EntropyMaskGate(nn.Module):
    """Predict per-element *entropy score* with a small network, then keep the
    top-k elements with the **lowest** entropy (= most predictable / most
    important for reconstruction).

    During training the gate uses a soft STE approximation so gradients flow
    through the mask.
    """

    def __init__(self, channels: int, keep_ratio: float, block_size: int = 1, temperature: float = 0.1):
        super().__init__()
        self.keep_ratio = keep_ratio
        self.block_size = block_size
        self.temperature = temperature

        mid = max(channels // 4, 8)
        self.entropy_net = nn.Sequential(
            nn.Conv2d(channels, mid, 1),
            nn.GELU(),
            nn.Conv2d(mid, mid, 3, padding=1, groups=_group_count(mid)),
            nn.GELU(),
            nn.Conv2d(mid, channels, 1),
        )

    def forward(self, features: torch.Tensor, enabled: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mask, entropy_map).  ``entropy_map`` can be used for
        visualisation / auxiliary losses later.
        """
        B, C, H, W = features.shape
        if not enabled:
            ones = torch.ones_like(features)
            return ones, torch.zeros_like(features)

        entropy_scores = self.entropy_net(features)           # [B, C, H, W]

        # Pool to block level if block_size > 1
        if self.block_size > 1:
            pH = (self.block_size - H % self.block_size) % self.block_size
            pW = (self.block_size - W % self.block_size) % self.block_size
            padded = F.pad(entropy_scores, (0, pW, 0, pH))
            pooled = F.avg_pool2d(
                padded.reshape(B * C, 1, H + pH, W + pW),
                kernel_size=self.block_size,
                stride=self.block_size,
            )
            pH2, pW2 = pooled.shape[-2:]
            scores_flat = pooled.view(B, C, pH2 * pW2)
        else:
            scores_flat = entropy_scores.view(B, C, H * W)

        # Lower entropy → more important → we use *negative* entropy as importance
        importance = -scores_flat

        total = scores_flat.shape[-1]
        keep = max(1, min(total, int(round(total * self.keep_ratio))))
        top_vals, top_idx = torch.topk(importance, keep, dim=-1)
        hard_mask = torch.zeros_like(scores_flat)
        hard_mask.scatter_(-1, top_idx, 1.0)

        if self.training:
            threshold = top_vals[..., -1:].detach()
            soft_mask = torch.sigmoid((importance - threshold) / self.temperature)
            soft_mask = soft_mask * (keep / soft_mask.sum(dim=-1, keepdim=True).clamp_min(1e-6))
            soft_mask = soft_mask.clamp(0.0, 1.0)
            mask_flat = hard_mask.detach() - soft_mask.detach() + soft_mask
        else:
            mask_flat = hard_mask

        # Expand back to spatial
        if self.block_size > 1:
            block_mask = mask_flat.view(B, C, pH2, pW2)
            full_mask = block_mask.repeat_interleave(self.block_size, dim=2).repeat_interleave(self.block_size, dim=3)
            full_mask = full_mask[:, :, :H, :W]
        else:
            full_mask = mask_flat.view(B, C, H, W)

        return full_mask, entropy_scores


# ---------------------------------------------------------------------------
# Dataclasses for outputs
# ---------------------------------------------------------------------------

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
class MDVSCV2Output:
    restored_sequences: list[torch.Tensor]
    level_stats: list[LevelTransmissionStats]
    common_masks: list[torch.Tensor]
    individual_masks: list[torch.Tensor]
    common_entropy_maps: list[torch.Tensor]
    individual_entropy_maps: list[torch.Tensor]

    def stats_dict(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self.level_stats]


# ---------------------------------------------------------------------------
# Per-level v2 module
# ---------------------------------------------------------------------------

class PerLevelMDVSCV2(nn.Module):
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
        self.in_channels = in_channels

        # Progressive encoder / decoder with skip connections
        self.encoder = ProgressiveEncoder(in_channels, latent_dim)
        self.decoder = ProgressiveDecoder(in_channels, latent_dim)

        # Learnable temporal decomposition
        self.temporal_decomposer = TemporalAttentionDecomposer(latent_dim)

        # Entropy-based importance gates
        self.common_gate = EntropyMaskGate(latent_dim, common_keep_ratio, block_size)
        self.individual_gate = EntropyMaskGate(latent_dim, individual_keep_ratio, block_size)

    def forward(
        self,
        feature_sequence: torch.Tensor,
        apply_masks: bool,
        channel_mode: str,
        snr_db: float | None,
        level_index: int,
    ) -> tuple[torch.Tensor, LevelTransmissionStats, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, C, H, W = feature_sequence.shape
        flat = feature_sequence.reshape(B * T, C, H, W)

        # Encode – collect skip connections for all frames together
        latent_flat, skip0_flat = self.encoder(flat)
        latent_seq = latent_flat.view(B, T, self.latent_dim, H, W)
        skip0_seq = skip0_flat.view(B, T, self.encoder.skip0_channels, H, W)

        # Temporal decomposition (learned)
        common_latent, individual_latent = self.temporal_decomposer(latent_seq)

        # Entropy-based masking
        common_mask, common_entropy = self.common_gate(common_latent, enabled=apply_masks)
        transmitted_common = self._transmit(common_latent * common_mask, channel_mode, snr_db)

        restored_frames: list[torch.Tensor] = []
        ind_masks: list[torch.Tensor] = []
        ind_entropies: list[torch.Tensor] = []
        for t in range(T):
            frame_residual = individual_latent[:, t]
            ind_mask, ind_entropy = self.individual_gate(frame_residual, enabled=apply_masks)
            ind_masks.append(ind_mask)
            ind_entropies.append(ind_entropy)
            transmitted_ind = self._transmit(frame_residual * ind_mask, channel_mode, snr_db)

            # Decode using skip connection from the same frame
            decoded = self.decoder(transmitted_common + transmitted_ind, skip0_seq[:, t])
            restored_frames.append(decoded)

        restored_seq = torch.stack(restored_frames, dim=1)             # [B, T, C_in, H, W]
        ind_mask_tensor = torch.stack(ind_masks, dim=1)                # [B, T, C_lat, H, W]
        ind_entropy_tensor = torch.stack(ind_entropies, dim=1)

        stats = LevelTransmissionStats(
            level=level_index,
            latent_dim=self.latent_dim,
            block_size=self.block_size,
            common_active_ratio=float(common_mask.detach().mean().item()),
            individual_active_ratio=float(ind_mask_tensor.detach().mean().item()),
        )
        return restored_seq, stats, common_mask, ind_mask_tensor, common_entropy, ind_entropy_tensor

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


# ---------------------------------------------------------------------------
# Cross-level fusion (top-down)
# ---------------------------------------------------------------------------

class CrossLevelFusion(nn.Module):
    """Lightweight top-down cross-level message passing in the latent space
    *before* transmission.  Level 2 (stride-32) → Level 1 → Level 0.
    Each fusion step is a 1×1 project + add after bilinear upsample.
    """

    def __init__(self, latent_dims: list[int]):
        super().__init__()
        self.projs = nn.ModuleList()
        # Projection from level i+1 to level i
        for i in range(len(latent_dims) - 1):
            self.projs.append(nn.Conv2d(latent_dims[i + 1], latent_dims[i], 1))

    def forward(self, latents: list[torch.Tensor]) -> list[torch.Tensor]:
        """latents[i] has shape [B, C_i, H_i, W_i], ordered stride-8…stride-32."""
        out = list(latents)
        for i in reversed(range(len(self.projs))):
            coarse = out[i + 1]
            fine = out[i]
            projected = self.projs[i](coarse)
            projected = F.interpolate(projected, size=fine.shape[-2:], mode="bilinear", align_corners=False)
            out[i] = fine + projected
        return out


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class ProjectMDVSCV2(nn.Module):
    def __init__(
        self,
        feature_channels: list[int] | tuple[int, ...] = (256, 256, 256),
        latent_dims: list[int] | tuple[int, ...] = (64, 80, 96),
        common_keep_ratios: list[float] | tuple[float, ...] = (0.6, 0.7, 0.8),
        individual_keep_ratios: list[float] | tuple[float, ...] = (0.15, 0.2, 0.25),
        block_sizes: list[int] | tuple[int, ...] = (4, 2, 1),
        apply_cross_level_fusion: bool = True,
    ):
        super().__init__()
        self.feature_channels = list(feature_channels)
        self.latent_dims = list(latent_dims)
        self.common_keep_ratios = list(common_keep_ratios)
        self.individual_keep_ratios = list(individual_keep_ratios)
        self.block_sizes = list(block_sizes)
        self.apply_cross_level_fusion = apply_cross_level_fusion

        if not (
            len(self.feature_channels)
            == len(self.latent_dims)
            == len(self.common_keep_ratios)
            == len(self.individual_keep_ratios)
            == len(self.block_sizes)
        ):
            raise ValueError("All per-level config lists must have the same length")

        self.level_modules = nn.ModuleList(
            PerLevelMDVSCV2(
                in_channels=fc,
                latent_dim=ld,
                common_keep_ratio=ckr,
                individual_keep_ratio=ikr,
                block_size=bs,
            )
            for fc, ld, ckr, ikr, bs in zip(
                self.feature_channels,
                self.latent_dims,
                self.common_keep_ratios,
                self.individual_keep_ratios,
                self.block_sizes,
            )
        )

        if apply_cross_level_fusion:
            self.cross_level_fusion = CrossLevelFusion(self.latent_dims)
        else:
            self.cross_level_fusion = None

    def forward(
        self,
        feature_sequences: list[torch.Tensor],
        apply_masks: bool = True,
        channel_mode: str = "identity",
        snr_db: float | None = None,
    ) -> MDVSCV2Output:
        if len(feature_sequences) != len(self.level_modules):
            raise ValueError("feature_sequences must match the configured number of levels")

        restored_sequences: list[torch.Tensor] = []
        level_stats: list[LevelTransmissionStats] = []
        common_masks: list[torch.Tensor] = []
        individual_masks: list[torch.Tensor] = []
        common_entropy_maps: list[torch.Tensor] = []
        individual_entropy_maps: list[torch.Tensor] = []

        for idx, (mod, feat_seq) in enumerate(zip(self.level_modules, feature_sequences)):
            restored, stats, c_mask, i_mask, c_ent, i_ent = mod(
                feat_seq,
                apply_masks=apply_masks,
                channel_mode=channel_mode,
                snr_db=snr_db,
                level_index=idx,
            )
            restored_sequences.append(restored)
            level_stats.append(stats)
            common_masks.append(c_mask)
            individual_masks.append(i_mask)
            common_entropy_maps.append(c_ent)
            individual_entropy_maps.append(i_ent)

        return MDVSCV2Output(
            restored_sequences=restored_sequences,
            level_stats=level_stats,
            common_masks=common_masks,
            individual_masks=individual_masks,
            common_entropy_maps=common_entropy_maps,
            individual_entropy_maps=individual_entropy_maps,
        )
