"""Stage-4 model: End-to-end joint training assembling Stage 2.1 + Stage 3.

Architecture:
    backbone raw features (stage2/3/4) → SharedEncoder → shared 256
    → MDVSC v2 (feature compression) → restored 256
    → DetRecoveryHead → projected 256 (detection target)
    → TaskAdaptationBlock × 3 + DetailBypass → reconstructed frames

    backbone stage1 features (256, stride 4) → DetailCompressor → tiny packet
    → DetailDecompressor → additive fusion at stride-8 in reconstruction path

This stage jointly trains all learnable components end-to-end with combined
losses from feature compression (Stage 3) + detection recovery + image
reconstruction (Stage 2.1).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.semantic_rtdetr.semantic_comm.mdvsc import TaskAdaptationBlock
from src.semantic_rtdetr.semantic_comm.mdvsc_v2 import MDVSCV2Output, ProjectMDVSCV2
from src.semantic_rtdetr.semantic_comm.stage2_1_model import (
    DetailAwareLightReconstructionHead,
    DetailCompressor,
    DetailDecompressor,
)
from src.semantic_rtdetr.semantic_comm.stage2_model import DetRecoveryHead, SharedEncoder


@dataclass
class Stage4Output:
    """Output of Stage4MDVSC forward pass."""

    # From SharedEncoder
    shared_sequences: list[torch.Tensor]
    # From MDVSC v2
    mdvsc_output: MDVSCV2Output
    # From DetRecoveryHead
    det_recovery_sequences: list[torch.Tensor]
    # Reconstruction
    reconstruction_sequences: list[torch.Tensor]
    reconstructed_frames: torch.Tensor
    reconstructed_base_frames: torch.Tensor
    reconstructed_high_frequency_residuals: torch.Tensor
    # Detail bypass
    detail_compressed: torch.Tensor
    detail_transmission_ratio: float


class Stage4MDVSC(nn.Module):
    """Stage-4 end-to-end model: SharedEncoder + MDVSC v2 + DetRecovery + DetailBypass + Reconstruction.

    Assembles all components from Stage 2.1 and Stage 3 into a single
    differentiable pipeline for joint fine-tuning.

    Data flow:
        backbone raw [512/1024/2048] → SharedEncoder → shared [256×3]
        → MDVSC v2 (compress/transmit/restore) → restored [256×3]
        → DetRecoveryHead → projected 256 (matches teacher encoder_input_proj)
        → TaskAdaptationBlock × 3 → reconstruction features
        → DetailAwareLightReconstructionHead + detail bypass → RGB frames

        backbone stage1 [256, H/4, W/4] → DetailCompressor → tiny packet
        → DetailDecompressor → fused at stride-8
    """

    def __init__(
        self,
        # SharedEncoder + DetRecovery
        backbone_channels: list[int] | tuple[int, ...] = (512, 1024, 2048),
        shared_channels: int = 256,
        # MDVSC v2
        latent_dims: list[int] | tuple[int, ...] = (64, 80, 96),
        common_keep_ratios: list[float] | tuple[float, ...] = (0.6, 0.7, 0.8),
        individual_keep_ratios: list[float] | tuple[float, ...] = (0.15, 0.2, 0.25),
        block_sizes: list[int] | tuple[int, ...] = (4, 2, 1),
        spatial_strides: list[int] | tuple[int, ...] = (2, 2, 1),
        apply_cross_level_fusion: bool = True,
        # Reconstruction
        reconstruction_hidden_channels: int = 160,
        reconstruction_detail_channels: int = 64,
        reconstruction_use_checkpoint: bool = False,
        # Detail bypass
        stage1_channels: int = 256,
        detail_latent_channels: int = 32,
        detail_spatial_size: int = 20,
    ):
        super().__init__()
        self.backbone_channels = list(backbone_channels)
        self.shared_channels = shared_channels
        num_levels = len(backbone_channels)
        feature_channels = [shared_channels] * num_levels

        # Stage 2.1 components
        self.shared_encoder = SharedEncoder(self.backbone_channels, shared_channels)
        self.det_recovery_head = DetRecoveryHead(shared_channels, num_levels)
        self.reconstruction_refinement_heads = nn.ModuleList(
            TaskAdaptationBlock(shared_channels) for _ in range(num_levels)
        )
        self.detail_compressor = DetailCompressor(
            in_channels=stage1_channels,
            detail_latent_channels=detail_latent_channels,
            detail_spatial_size=detail_spatial_size,
        )
        self.detail_decompressor = DetailDecompressor(
            detail_latent_channels=detail_latent_channels,
            hidden_channels=reconstruction_hidden_channels,
        )
        self.reconstruction_head = DetailAwareLightReconstructionHead(
            feature_channels,
            hidden_channels=reconstruction_hidden_channels,
            detail_channels=reconstruction_detail_channels,
            use_checkpoint=reconstruction_use_checkpoint,
        )

        # Stage 3 component
        self.mdvsc_v2 = ProjectMDVSCV2(
            feature_channels=feature_channels,
            latent_dims=latent_dims,
            common_keep_ratios=common_keep_ratios,
            individual_keep_ratios=individual_keep_ratios,
            block_sizes=block_sizes,
            spatial_strides=spatial_strides,
            apply_cross_level_fusion=apply_cross_level_fusion,
        )

    def forward(
        self,
        backbone_sequences: list[torch.Tensor],
        stage1_sequences: torch.Tensor,
        output_size: tuple[int, int],
        apply_masks: bool = True,
        channel_mode: str = "identity",
        snr_db: float | None = None,
    ) -> Stage4Output:
        """Full end-to-end forward pass.

        Args:
            backbone_sequences: Per-level raw backbone feature sequences
                ``[B, T, C_backbone, H, W]`` for each level.
            stage1_sequences: Stage1 backbone features ``[B, T, 256, H/4, W/4]``.
            output_size: Target reconstruction size ``(H, W)``.
            apply_masks: Whether to apply entropy-based masks in MDVSC v2.
            channel_mode: Channel simulation mode ('identity' or 'awgn').
            snr_db: SNR in dB for AWGN channel.
        """
        if len(backbone_sequences) != len(self.backbone_channels):
            raise ValueError("backbone_sequences must match the configured number of levels")

        batch_size, time_steps = backbone_sequences[0].shape[:2]

        # 1. SharedEncoder: backbone raw → shared 256
        shared_sequences = self._encode_sequences(backbone_sequences)

        # 2. MDVSC v2: compress → transmit → restore
        mdvsc_out: MDVSCV2Output = self.mdvsc_v2(
            shared_sequences,
            apply_masks=apply_masks,
            channel_mode=channel_mode,
            snr_db=snr_db,
        )
        restored_sequences = mdvsc_out.restored_sequences

        # 3. DetRecoveryHead: restored 256 → projected 256
        det_recovery_sequences = self._recover_sequences(restored_sequences)

        # 4. Detail compression: stage1 → tiny packet
        detail_compressed, detail_ratio = self._compress_detail(
            stage1_sequences, backbone_sequences
        )

        # 5. Detail decompression → stride-8 features
        stride8_size = restored_sequences[0].shape[-2:]
        detail_features_seq = self._decompress_detail(detail_compressed, stride8_size)

        # 6. Reconstruction refinement + detail-aware decode
        recon_sequences = self._apply_refinement(restored_sequences, self.reconstruction_refinement_heads)
        reconstructed_frames, base_frames, hf_residuals = self._decode_reconstruction_with_detail(
            recon_sequences, detail_features_seq, output_size
        )

        return Stage4Output(
            shared_sequences=shared_sequences,
            mdvsc_output=mdvsc_out,
            det_recovery_sequences=det_recovery_sequences,
            reconstruction_sequences=recon_sequences,
            reconstructed_frames=reconstructed_frames,
            reconstructed_base_frames=base_frames,
            reconstructed_high_frequency_residuals=hf_residuals,
            detail_compressed=detail_compressed,
            detail_transmission_ratio=detail_ratio,
        )

    # ------------------------------------------------------------------
    # Internal helpers (same patterns as Stage 2.1)
    # ------------------------------------------------------------------

    def _encode_sequences(self, backbone_sequences: list[torch.Tensor]) -> list[torch.Tensor]:
        shared_sequences: list[torch.Tensor] = []
        for proj, seq in zip(self.shared_encoder.projections, backbone_sequences):
            B, T, C, H, W = seq.shape
            flat = seq.reshape(B * T, C, H, W)
            shared = proj(flat)
            shared_sequences.append(shared.reshape(B, T, self.shared_channels, H, W))
        return shared_sequences

    def _recover_sequences(self, feature_sequences: list[torch.Tensor]) -> list[torch.Tensor]:
        recovered: list[torch.Tensor] = []
        for head, seq in zip(self.det_recovery_head.heads, feature_sequences):
            B, T, C, H, W = seq.shape
            flat = seq.reshape(B * T, C, H, W)
            rec = head(flat)
            recovered.append(rec.reshape(B, T, C, H, W))
        return recovered

    def _compress_detail(
        self,
        stage1_sequences: torch.Tensor,
        backbone_sequences: list[torch.Tensor],
    ) -> tuple[torch.Tensor, float]:
        B, T, C, H, W = stage1_sequences.shape
        flat = stage1_sequences.reshape(B * T, C, H, W)
        compressed = self.detail_compressor(flat)
        compressed_seq = compressed.reshape(B, T, *compressed.shape[1:])
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
        B, T = detail_compressed.shape[:2]
        flat = detail_compressed.reshape(B * T, *detail_compressed.shape[2:])
        expanded = self.detail_decompressor(flat, target_size)
        return expanded.reshape(B, T, *expanded.shape[1:])

    @staticmethod
    def _apply_refinement(
        feature_sequences: list[torch.Tensor],
        refinement_heads: nn.ModuleList,
    ) -> list[torch.Tensor]:
        refined_sequences: list[torch.Tensor] = []
        for seq, head in zip(feature_sequences, refinement_heads):
            B, T, C, H, W = seq.shape
            flat = seq.reshape(B * T, C, H, W)
            refined = head(flat)
            refined_sequences.append(refined.reshape(B, T, C, H, W))
        return refined_sequences

    def _decode_reconstruction_with_detail(
        self,
        feature_sequences: list[torch.Tensor],
        detail_features_seq: torch.Tensor,
        output_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        T = feature_sequences[0].shape[1]
        frames, bases, hf_residuals = [], [], []
        for t in range(T):
            frame_features = [seq[:, t] for seq in feature_sequences]
            frame_detail = detail_features_seq[:, t]
            recon, base, hf = self.reconstruction_head.decode_components_with_detail(
                frame_features, output_size, detail_features=frame_detail,
            )
            frames.append(recon)
            bases.append(base)
            hf_residuals.append(hf)
        return torch.stack(frames, dim=1), torch.stack(bases, dim=1), torch.stack(hf_residuals, dim=1)
