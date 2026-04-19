"""Stage-2.1 model: Stage2 + Detail Bypass with compressed stage1 features.

Architecture:
    backbone raw features (stage2/3/4) → SharedEncoder → DetRecoveryHead
    backbone stage1 features (256, stride 4) → DetailCompressor → [transmitted]
                                              → DetailDecompressor → fused into ReconHead

The DetailCompressor aggressively compresses stage1 (256ch, H/4×W/4) to a
tiny packet (detail_latent_channels × detail_spatial_size × detail_spatial_size).
This adds ~1% transmission overhead vs the main 3-level features.
The DetailDecompressor upsamples the packet back to stride-4 resolution
and fuses it into the reconstruction path at the stride-8→stride-4 transition.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.semantic_rtdetr.semantic_comm.mdvsc import (
    LightReconstructionHead,
    ReconstructionHead,
    ResidualBlock,
    TaskAdaptationBlock,
)
from src.semantic_rtdetr.semantic_comm.stage2_model import (
    DetRecoveryHead,
    SharedEncoder,
)


@dataclass
class Stage2_1Output:
    """Output of Stage2_1MDVSC forward pass."""

    shared_sequences: list[torch.Tensor]
    det_recovery_sequences: list[torch.Tensor]
    reconstruction_sequences: list[torch.Tensor]
    reconstructed_frames: torch.Tensor
    reconstructed_base_frames: torch.Tensor
    reconstructed_high_frequency_residuals: torch.Tensor
    detail_compressed: torch.Tensor  # [B, T, C_latent, Hs, Ws]
    detail_transmission_ratio: float  # ratio of detail packet size vs main features


class DetailCompressor(nn.Module):
    """Compress stage1 features (256ch, stride 4) to a tiny latent packet.

    Pipeline:
        [B, 256, H/4, W/4]
        → Conv1×1(256 → detail_latent_channels) + GELU
        → AdaptiveAvgPool2d(detail_spatial_size)
        → ResBlock refinement
    """

    def __init__(
        self,
        in_channels: int = 256,
        detail_latent_channels: int = 32,
        detail_spatial_size: int = 20,
    ):
        super().__init__()
        self.detail_latent_channels = detail_latent_channels
        self.detail_spatial_size = detail_spatial_size
        self.compress = nn.Sequential(
            nn.Conv2d(in_channels, detail_latent_channels, kernel_size=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(detail_spatial_size),
            ResidualBlock(detail_latent_channels),
        )

    def forward(self, stage1_features: torch.Tensor) -> torch.Tensor:
        """Compress stage1 features to a tiny packet.

        Args:
            stage1_features: ``[B, 256, H/4, W/4]``

        Returns:
            Compressed detail packet ``[B, C_latent, Hs, Ws]``.
        """
        return self.compress(stage1_features)


class DetailDecompressor(nn.Module):
    """Decompress the detail packet and fuse into the reconstruction path.

    Pipeline:
        [B, C_latent, Hs, Ws]
        → Conv1×1(C_latent → hidden_channels) + GELU
        → Bilinear upsample to target spatial size
        → ResBlock refinement

    The output is additive-fused with the main reconstruction features
    at the stride-8 level before upsampling begins.
    """

    def __init__(
        self,
        detail_latent_channels: int = 32,
        hidden_channels: int = 160,
    ):
        super().__init__()
        self.expand = nn.Sequential(
            nn.Conv2d(detail_latent_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            ResidualBlock(hidden_channels),
        )

    def forward(self, detail_packet: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        """Decompress and upsample detail packet.

        Args:
            detail_packet: ``[B, C_latent, Hs, Ws]``
            target_size: ``(H, W)`` at stride 8.

        Returns:
            Detail features ``[B, hidden_channels, H, W]`` ready for additive fusion.
        """
        expanded = self.expand(detail_packet)
        if expanded.shape[-2:] != target_size:
            expanded = F.interpolate(expanded, size=target_size, mode="bilinear", align_corners=False)
        return expanded


class DetailAwareLightReconstructionHead(LightReconstructionHead):
    """LightReconstructionHead extended with optional detail bypass fusion.

    When detail_features is provided, it is additively fused with the
    stride-8 features after the additive FPN and refinement, before upsampling.
    """

    def decode_components_with_detail(
        self,
        feature_maps: list[torch.Tensor],
        output_size: tuple[int, int],
        detail_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Like decode_components but with optional detail fusion."""
        import torch.utils.checkpoint as checkpoint_util

        # Project all levels to the same channel dimension
        f0 = self.proj0(feature_maps[0])  # stride 8
        f1 = self.proj1(feature_maps[1])  # stride 16
        f2 = self.proj2(feature_maps[2])  # stride 32

        # Top-down additive FPN
        f1 = f1 + F.interpolate(f2, size=f1.shape[-2:], mode="nearest")
        f0 = f0 + F.interpolate(f1, size=f0.shape[-2:], mode="nearest")

        # Refine at stride 8
        fused = self.refine(f0)

        # Detail bypass fusion: add detail features at stride 8
        if detail_features is not None:
            if detail_features.shape[-2:] != fused.shape[-2:]:
                detail_features = F.interpolate(
                    detail_features, size=fused.shape[-2:], mode="bilinear", align_corners=False
                )
            fused = fused + detail_features

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


class Stage2_1MDVSC(nn.Module):
    """Stage-2.1 model: Stage2 + DetailBypass (compressed stage1 for reconstruction).

    Extends Stage2MDVSC with:
    - DetailCompressor: aggressively compresses stage1 (256ch) to tiny packet
    - DetailDecompressor: expands packet and fuses into reconstruction path
    - DetailAwareLightReconstructionHead: accepts detail features

    The detail bypass only helps reconstruction; detection recovery is unchanged.
    Transmission cost increase is ~1% of the main feature volume.
    """

    def __init__(
        self,
        backbone_channels: list[int] | tuple[int, ...] = (512, 1024, 2048),
        shared_channels: int = 256,
        reconstruction_hidden_channels: int = 160,
        reconstruction_detail_channels: int = 64,
        reconstruction_head_type: str = "light",
        reconstruction_use_checkpoint: bool = False,
        # Detail bypass parameters
        stage1_channels: int = 256,
        detail_latent_channels: int = 32,
        detail_spatial_size: int = 20,
    ):
        super().__init__()
        self.backbone_channels = list(backbone_channels)
        self.shared_channels = shared_channels
        self.detail_latent_channels = detail_latent_channels
        self.detail_spatial_size = detail_spatial_size
        num_levels = len(backbone_channels)
        feature_channels = [shared_channels] * num_levels

        # Reuse Stage2 components
        self.shared_encoder = SharedEncoder(self.backbone_channels, shared_channels)
        self.det_recovery_head = DetRecoveryHead(shared_channels, num_levels)

        self.reconstruction_refinement_heads = nn.ModuleList(
            TaskAdaptationBlock(shared_channels) for _ in range(num_levels)
        )

        # Detail bypass: compress and decompress stage1 features
        self.detail_compressor = DetailCompressor(
            in_channels=stage1_channels,
            detail_latent_channels=detail_latent_channels,
            detail_spatial_size=detail_spatial_size,
        )
        self.detail_decompressor = DetailDecompressor(
            detail_latent_channels=detail_latent_channels,
            hidden_channels=reconstruction_hidden_channels,
        )

        # Use detail-aware reconstruction head
        if reconstruction_head_type == "light":
            self.reconstruction_head: nn.Module = DetailAwareLightReconstructionHead(
                feature_channels,
                hidden_channels=reconstruction_hidden_channels,
                detail_channels=reconstruction_detail_channels,
                use_checkpoint=reconstruction_use_checkpoint,
            )
        else:
            raise ValueError(
                f"Stage 2.1 only supports 'light' reconstruction head, got '{reconstruction_head_type}'"
            )

    def forward(
        self,
        backbone_sequences: list[torch.Tensor],
        stage1_sequences: torch.Tensor,
        output_size: tuple[int, int],
    ) -> Stage2_1Output:
        """Full forward pass through SharedEncoder + DetailBypass → dual decoder.

        Args:
            backbone_sequences: Per-level raw backbone feature sequences (stage2/3/4)
                ``[B, T, C_backbone, H, W]``.
            stage1_sequences: Stage1 backbone features ``[B, T, 256, H/4, W/4]``.
            output_size: Target reconstruction size ``(H, W)``.
        """
        if len(backbone_sequences) != len(self.backbone_channels):
            raise ValueError("backbone_sequences must match the configured number of levels")

        batch_size, time_steps = backbone_sequences[0].shape[:2]

        # 1. SharedEncoder: backbone → shared 256
        shared_sequences = self._encode_sequences(backbone_sequences)

        # 2. Detection recovery: shared 256 → projected 256 (unchanged from stage2)
        det_recovery_sequences = self._recover_sequences(shared_sequences)

        # 3. Detail compression: stage1 → tiny packet
        detail_compressed, detail_ratio = self._compress_detail(
            stage1_sequences, backbone_sequences
        )

        # 4. Detail decompression: tiny packet → stride-8 detail features
        # Target size = stride-8 spatial from level 0
        stride8_size = shared_sequences[0].shape[-2:]  # (H/8, W/8)
        detail_features_seq = self._decompress_detail(detail_compressed, stride8_size)

        # 5. Reconstruction with detail fusion
        recon_sequences = self._apply_refinement(shared_sequences, self.reconstruction_refinement_heads)
        reconstructed_frames, base_frames, hf_residuals = self._decode_reconstruction_with_detail(
            recon_sequences, detail_features_seq, output_size
        )

        return Stage2_1Output(
            shared_sequences=shared_sequences,
            det_recovery_sequences=det_recovery_sequences,
            reconstruction_sequences=recon_sequences,
            reconstructed_frames=reconstructed_frames,
            reconstructed_base_frames=base_frames,
            reconstructed_high_frequency_residuals=hf_residuals,
            detail_compressed=detail_compressed,
            detail_transmission_ratio=detail_ratio,
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

    def _compress_detail(
        self,
        stage1_sequences: torch.Tensor,
        backbone_sequences: list[torch.Tensor],
    ) -> tuple[torch.Tensor, float]:
        """Compress stage1 features and compute transmission cost ratio."""
        batch_size, time_steps, channels, h, w = stage1_sequences.shape
        flat = stage1_sequences.reshape(batch_size * time_steps, channels, h, w)
        compressed = self.detail_compressor(flat)
        compressed_seq = compressed.reshape(batch_size, time_steps, *compressed.shape[1:])

        # Compute transmission ratio: detail_packet_size / main_features_size
        detail_numel = compressed.shape[1] * compressed.shape[2] * compressed.shape[3]
        main_numel = sum(
            seq.shape[2] * seq.shape[3] * seq.shape[4]
            for seq in backbone_sequences
        )
        ratio = detail_numel / max(main_numel, 1)

        return compressed_seq, ratio

    def _decompress_detail(
        self,
        detail_compressed: torch.Tensor,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        """Decompress detail packet sequence to stride-8 resolution."""
        batch_size, time_steps = detail_compressed.shape[:2]
        flat = detail_compressed.reshape(batch_size * time_steps, *detail_compressed.shape[2:])
        expanded = self.detail_decompressor(flat, target_size)
        return expanded.reshape(batch_size, time_steps, *expanded.shape[1:])

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

    def _decode_reconstruction_with_detail(
        self,
        feature_sequences: list[torch.Tensor],
        detail_features_seq: torch.Tensor,
        output_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        time_steps = feature_sequences[0].shape[1]
        frames: list[torch.Tensor] = []
        bases: list[torch.Tensor] = []
        hf_residuals: list[torch.Tensor] = []
        for frame_index in range(time_steps):
            frame_features = [seq[:, frame_index] for seq in feature_sequences]
            frame_detail = detail_features_seq[:, frame_index]
            assert isinstance(self.reconstruction_head, DetailAwareLightReconstructionHead)
            recon, base, hf = self.reconstruction_head.decode_components_with_detail(
                frame_features, output_size, detail_features=frame_detail,
            )
            frames.append(recon)
            bases.append(base)
            hf_residuals.append(hf)
        return torch.stack(frames, dim=1), torch.stack(bases, dim=1), torch.stack(hf_residuals, dim=1)
