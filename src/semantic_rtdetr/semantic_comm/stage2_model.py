"""Stage-2 model: SharedEncoder from raw backbone → dual decoder.

Architecture (no MDVSC transmission in this stage):
    backbone raw features → SharedEncoder (per-level backbone_ch→256)
    → DetRecoveryHead (per-level 256→projected 256 recovery)
    → ReconRefinement + LightReconstructionHead (256→image reconstruction)

SharedEncoder keeps layers independent (no FPN). C_shared = 256.
DetRecoveryHead is a minimal linear projection + BN per level.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.semantic_rtdetr.semantic_comm.mdvsc import (
    LightReconstructionHead,
    ReconstructionHead,
    ResidualBlock,
    TaskAdaptationBlock,
)


@dataclass
class Stage2Output:
    """Output of Stage2MDVSC forward pass."""

    shared_sequences: list[torch.Tensor]
    det_recovery_sequences: list[torch.Tensor]
    reconstruction_sequences: list[torch.Tensor]
    reconstructed_frames: torch.Tensor
    reconstructed_base_frames: torch.Tensor
    reconstructed_high_frequency_residuals: torch.Tensor


class SharedEncoder(nn.Module):
    """Per-level independent projection from raw backbone channels to shared 256.

    Each level: Conv1×1(backbone_ch → shared_ch) + GELU + ResBlock(shared_ch).
    No FPN — layers are kept independent so that DetRecoveryHead can cleanly
    recover the per-level projected features.
    """

    def __init__(self, backbone_channels: list[int], shared_channels: int = 256):
        super().__init__()
        self.shared_channels = shared_channels
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_ch, shared_channels, kernel_size=1),
                    nn.GELU(),
                    ResidualBlock(shared_channels),
                )
                for in_ch in backbone_channels
            ]
        )

    def forward(self, backbone_features: list[torch.Tensor]) -> list[torch.Tensor]:
        return [proj(feat) for proj, feat in zip(self.projections, backbone_features)]


class DetRecoveryHead(nn.Module):
    """Per-level linear recovery from shared 256 to projected 256.

    Each level: Conv1×1(256→256, bias=False) + BatchNorm2d(256).
    Matches the structure of the frozen teacher encoder_input_proj.
    """

    def __init__(self, channels: int = 256, num_levels: int = 3):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                )
                for _ in range(num_levels)
            ]
        )

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        return [head(feat) for head, feat in zip(self.heads, features)]


class Stage2MDVSC(nn.Module):
    """Stage-2 model: SharedEncoder → dual decoder (no MDVSC transmission).

    This stage validates the SharedEncoder + dual decoder architecture by
    training with lossless passthrough (no MDVSC modules). The focus is on:
    - Image reconstruction quality from backbone raw features
    - Projected 256 feature recovery precision via DetRecoveryHead
    """

    def __init__(
        self,
        backbone_channels: list[int] | tuple[int, ...] = (512, 1024, 2048),
        shared_channels: int = 256,
        reconstruction_hidden_channels: int = 160,
        reconstruction_detail_channels: int = 64,
        reconstruction_head_type: str = "light",
        reconstruction_use_checkpoint: bool = False,
    ):
        super().__init__()
        self.backbone_channels = list(backbone_channels)
        self.shared_channels = shared_channels
        num_levels = len(backbone_channels)
        feature_channels = [shared_channels] * num_levels

        self.shared_encoder = SharedEncoder(self.backbone_channels, shared_channels)
        self.det_recovery_head = DetRecoveryHead(shared_channels, num_levels)

        self.reconstruction_refinement_heads = nn.ModuleList(
            TaskAdaptationBlock(shared_channels) for _ in range(num_levels)
        )

        if reconstruction_head_type == "light":
            self.reconstruction_head: nn.Module = LightReconstructionHead(
                feature_channels,
                hidden_channels=reconstruction_hidden_channels,
                detail_channels=reconstruction_detail_channels,
                use_checkpoint=reconstruction_use_checkpoint,
            )
        else:
            self.reconstruction_head = ReconstructionHead(
                feature_channels,
                hidden_channels=reconstruction_hidden_channels,
                detail_channels=reconstruction_detail_channels,
            )

    def forward(
        self,
        backbone_sequences: list[torch.Tensor],
        output_size: tuple[int, int],
    ) -> Stage2Output:
        """Full forward pass through SharedEncoder → dual decoder.

        Args:
            backbone_sequences: Per-level raw backbone feature sequences
                ``[B, T, C_backbone, H, W]``.
            output_size: Target reconstruction size ``(H, W)``.
        """
        if len(backbone_sequences) != len(self.backbone_channels):
            raise ValueError("backbone_sequences must match the configured number of levels")

        # 1. SharedEncoder: backbone → shared 256
        shared_sequences = self._encode_sequences(backbone_sequences)

        # 2. Detection recovery: shared 256 → projected 256
        det_recovery_sequences = self._recover_sequences(shared_sequences)

        # 3. Reconstruction: shared 256 → image
        recon_sequences = self._apply_refinement(shared_sequences, self.reconstruction_refinement_heads)
        reconstructed_frames, base_frames, hf_residuals = self._decode_reconstruction(
            recon_sequences, output_size
        )

        return Stage2Output(
            shared_sequences=shared_sequences,
            det_recovery_sequences=det_recovery_sequences,
            reconstruction_sequences=recon_sequences,
            reconstructed_frames=reconstructed_frames,
            reconstructed_base_frames=base_frames,
            reconstructed_high_frequency_residuals=hf_residuals,
        )

    def _encode_sequences(self, backbone_sequences: list[torch.Tensor]) -> list[torch.Tensor]:
        shared_sequences: list[torch.Tensor] = []
        for proj, seq in zip(self.shared_encoder.projections, backbone_sequences):
            batch_size, time_steps, channels, height, width = seq.shape
            flat = seq.reshape(batch_size * time_steps, channels, height, width)
            shared = proj(flat)
            shared_sequences.append(shared.reshape(batch_size, time_steps, self.shared_channels, height, width))
        return shared_sequences

    def _recover_sequences(self, shared_sequences: list[torch.Tensor]) -> list[torch.Tensor]:
        recovered: list[torch.Tensor] = []
        for head, seq in zip(self.det_recovery_head.heads, shared_sequences):
            batch_size, time_steps, channels, height, width = seq.shape
            flat = seq.reshape(batch_size * time_steps, channels, height, width)
            rec = head(flat)
            recovered.append(rec.reshape(batch_size, time_steps, channels, height, width))
        return recovered

    @staticmethod
    def _apply_refinement(
        feature_sequences: list[torch.Tensor],
        refinement_heads: nn.ModuleList,
    ) -> list[torch.Tensor]:
        refined_sequences: list[torch.Tensor] = []
        for feature_sequence, refinement_head in zip(feature_sequences, refinement_heads):
            batch_size, time_steps, channels, height, width = feature_sequence.shape
            flat = feature_sequence.reshape(batch_size * time_steps, channels, height, width)
            refined = refinement_head(flat)
            refined_sequences.append(refined.reshape(batch_size, time_steps, channels, height, width))
        return refined_sequences

    def _decode_reconstruction(
        self,
        feature_sequences: list[torch.Tensor],
        output_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        time_steps = feature_sequences[0].shape[1]
        frames: list[torch.Tensor] = []
        bases: list[torch.Tensor] = []
        hf_residuals: list[torch.Tensor] = []
        for frame_index in range(time_steps):
            frame_features = [seq[:, frame_index] for seq in feature_sequences]
            recon, base, hf = self.reconstruction_head.decode_components(frame_features, output_size)
            frames.append(recon)
            bases.append(base)
            hf_residuals.append(hf)
        return torch.stack(frames, dim=1), torch.stack(bases, dim=1), torch.stack(hf_residuals, dim=1)
