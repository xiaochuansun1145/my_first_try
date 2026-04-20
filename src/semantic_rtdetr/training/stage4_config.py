"""Stage-4 configuration: end-to-end joint training config dataclass & YAML loader."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.semantic_rtdetr.training.stage1_config import (
    Stage1DataConfig,
    Stage1DetectorConfig,
    _as_float_list,
    _as_int_list,
)


@dataclass(frozen=True)
class Stage4MDVSCConfig:
    """Combined architecture config for SharedEncoder + MDVSC v2 + dual decoder."""

    # SharedEncoder
    backbone_channels: list[int] = field(default_factory=lambda: [512, 1024, 2048])
    shared_channels: int = 256

    # MDVSC v2
    latent_dims: list[int] = field(default_factory=lambda: [64, 80, 96])
    common_keep_ratios: list[float] = field(default_factory=lambda: [0.6, 0.7, 0.8])
    individual_keep_ratios: list[float] = field(default_factory=lambda: [0.15, 0.2, 0.25])
    block_sizes: list[int] = field(default_factory=lambda: [4, 2, 1])
    spatial_strides: list[int] = field(default_factory=lambda: [2, 2, 1])
    apply_cross_level_fusion: bool = True
    apply_masks: bool = True
    channel_mode: str = "identity"
    snr_db: float = 20.0

    # Reconstruction
    reconstruction_hidden_channels: int = 160
    reconstruction_detail_channels: int = 64
    reconstruction_use_checkpoint: bool = False

    # Detail bypass
    stage1_channels: int = 256
    detail_latent_channels: int = 32
    detail_spatial_size: int = 20


@dataclass(frozen=True)
class Stage4PhaseConfig:
    """Per-phase training config for progressive unfreezing."""

    epochs: int = 5
    lr: float = 1e-4
    freeze_shared_encoder: bool = True
    freeze_mdvsc_v2: bool = False
    freeze_det_recovery: bool = False
    freeze_detail_bypass: bool = False
    freeze_reconstruction: bool = False


@dataclass(frozen=True)
class Stage4OptimizationConfig:
    batch_size: int = 1
    num_workers: int = 4
    use_amp: bool = True
    amp_dtype: str = "float16"
    optimizer: str = "adamw"
    weight_decay: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    scheduler: str = "onecycle"
    onecycle_pct_start: float = 0.15
    onecycle_div_factor: float = 25.0
    onecycle_final_div_factor: float = 1000.0
    grad_clip_norm: float = 1.0
    log_every: int = 10
    save_every_epochs: int = 1
    max_steps_per_epoch: int | None = None
    seed: int = 42

    # Phase schedule
    phase1: Stage4PhaseConfig = field(default_factory=lambda: Stage4PhaseConfig(
        epochs=8,
        lr=1e-4,
        freeze_shared_encoder=True,
        freeze_mdvsc_v2=False,
        freeze_det_recovery=False,
        freeze_detail_bypass=False,
        freeze_reconstruction=False,
    ))
    phase2: Stage4PhaseConfig = field(default_factory=lambda: Stage4PhaseConfig(
        epochs=8,
        lr=5e-5,
        freeze_shared_encoder=False,
        freeze_mdvsc_v2=False,
        freeze_det_recovery=False,
        freeze_detail_bypass=False,
        freeze_reconstruction=False,
    ))
    phase3: Stage4PhaseConfig = field(default_factory=lambda: Stage4PhaseConfig(
        epochs=4,
        lr=2e-5,
        freeze_shared_encoder=False,
        freeze_mdvsc_v2=False,
        freeze_det_recovery=False,
        freeze_detail_bypass=False,
        freeze_reconstruction=False,
    ))


@dataclass(frozen=True)
class Stage4LossConfig:
    """Combined loss weights for feature + detection + reconstruction."""

    # Feature compression (from Stage 3)
    feature_loss_type: str = "smooth_l1"
    feature_loss_weight: float = 1.0
    level_loss_weights: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])

    # Detection recovery (from Stage 2.1)
    det_recovery_weight: float = 1.0
    level_recovery_weights: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])

    # Reconstruction (from Stage 2.1)
    recon_l1_weight: float = 1.0
    recon_mse_weight: float = 0.25
    recon_ssim_weight: float = 0.25
    recon_edge_weight: float = 0.2
    ssim_downsample_factor: int = 2


@dataclass(frozen=True)
class Stage4OutputConfig:
    output_dir: str = "outputs/mdvsc_stage4"
    save_visualizations: bool = True
    visualization_every_epochs: int = 1
    visualization_num_frames: int = 4


@dataclass(frozen=True)
class Stage4InitializationConfig:
    """Checkpoints for initializing Stage 4 from prior stages."""

    stage2_1_checkpoint: str | None = None
    stage3_checkpoint: str | None = None
    full_checkpoint: str | None = None
    strict: bool = False


@dataclass(frozen=True)
class MDVSCStage4TrainConfig:
    detector: Stage1DetectorConfig = field(default_factory=Stage1DetectorConfig)
    data: Stage1DataConfig = field(default_factory=Stage1DataConfig)
    mdvsc: Stage4MDVSCConfig = field(default_factory=Stage4MDVSCConfig)
    optimization: Stage4OptimizationConfig = field(default_factory=Stage4OptimizationConfig)
    loss: Stage4LossConfig = field(default_factory=Stage4LossConfig)
    output: Stage4OutputConfig = field(default_factory=Stage4OutputConfig)
    initialization: Stage4InitializationConfig = field(default_factory=Stage4InitializationConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_phase(data: dict, defaults: Stage4PhaseConfig) -> Stage4PhaseConfig:
    if not data:
        return defaults
    return Stage4PhaseConfig(
        epochs=int(data.get("epochs", defaults.epochs)),
        lr=float(data.get("lr", defaults.lr)),
        freeze_shared_encoder=bool(data.get("freeze_shared_encoder", defaults.freeze_shared_encoder)),
        freeze_mdvsc_v2=bool(data.get("freeze_mdvsc_v2", defaults.freeze_mdvsc_v2)),
        freeze_det_recovery=bool(data.get("freeze_det_recovery", defaults.freeze_det_recovery)),
        freeze_detail_bypass=bool(data.get("freeze_detail_bypass", defaults.freeze_detail_bypass)),
        freeze_reconstruction=bool(data.get("freeze_reconstruction", defaults.freeze_reconstruction)),
    )


def load_stage4_config(config_path: str | Path) -> MDVSCStage4TrainConfig:
    path = Path(config_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    detector_data = data.get("detector") or {}
    data_data = data.get("data") or {}
    mdvsc_data = data.get("mdvsc") or {}
    opt_data = data.get("optimization") or {}
    loss_data = data.get("loss") or {}
    output_data = data.get("output") or {}
    init_data = data.get("initialization") or {}

    # Parse phase configs
    default_opt = Stage4OptimizationConfig()
    phase1 = _parse_phase(opt_data.get("phase1") or {}, default_opt.phase1)
    phase2 = _parse_phase(opt_data.get("phase2") or {}, default_opt.phase2)
    phase3 = _parse_phase(opt_data.get("phase3") or {}, default_opt.phase3)

    # Remove phase keys from opt_data before passing to dataclass
    opt_clean = {k: v for k, v in opt_data.items() if k not in ("phase1", "phase2", "phase3")}

    return MDVSCStage4TrainConfig(
        detector=Stage1DetectorConfig(**detector_data),
        data=Stage1DataConfig(**data_data),
        mdvsc=Stage4MDVSCConfig(
            backbone_channels=_as_int_list(mdvsc_data.get("backbone_channels"), [512, 1024, 2048]),
            shared_channels=int(mdvsc_data.get("shared_channels", 256)),
            latent_dims=_as_int_list(mdvsc_data.get("latent_dims"), [64, 80, 96]),
            common_keep_ratios=_as_float_list(mdvsc_data.get("common_keep_ratios"), [0.6, 0.7, 0.8]),
            individual_keep_ratios=_as_float_list(mdvsc_data.get("individual_keep_ratios"), [0.15, 0.2, 0.25]),
            block_sizes=_as_int_list(mdvsc_data.get("block_sizes"), [4, 2, 1]),
            spatial_strides=_as_int_list(mdvsc_data.get("spatial_strides"), [2, 2, 1]),
            apply_cross_level_fusion=bool(mdvsc_data.get("apply_cross_level_fusion", True)),
            apply_masks=bool(mdvsc_data.get("apply_masks", True)),
            channel_mode=str(mdvsc_data.get("channel_mode", "identity")),
            snr_db=float(mdvsc_data.get("snr_db", 20.0)),
            reconstruction_hidden_channels=int(mdvsc_data.get("reconstruction_hidden_channels", 160)),
            reconstruction_detail_channels=int(mdvsc_data.get("reconstruction_detail_channels", 64)),
            reconstruction_use_checkpoint=bool(mdvsc_data.get("reconstruction_use_checkpoint", False)),
            stage1_channels=int(mdvsc_data.get("stage1_channels", 256)),
            detail_latent_channels=int(mdvsc_data.get("detail_latent_channels", 32)),
            detail_spatial_size=int(mdvsc_data.get("detail_spatial_size", 20)),
        ),
        optimization=Stage4OptimizationConfig(
            **opt_clean,
            phase1=phase1,
            phase2=phase2,
            phase3=phase3,
        ),
        loss=Stage4LossConfig(
            feature_loss_type=str(loss_data.get("feature_loss_type", "smooth_l1")),
            feature_loss_weight=float(loss_data.get("feature_loss_weight", 1.0)),
            level_loss_weights=_as_float_list(loss_data.get("level_loss_weights"), [1.0, 1.0, 1.0]),
            det_recovery_weight=float(loss_data.get("det_recovery_weight", 1.0)),
            level_recovery_weights=_as_float_list(loss_data.get("level_recovery_weights"), [1.0, 1.0, 1.0]),
            recon_l1_weight=float(loss_data.get("recon_l1_weight", 1.0)),
            recon_mse_weight=float(loss_data.get("recon_mse_weight", 0.25)),
            recon_ssim_weight=float(loss_data.get("recon_ssim_weight", 0.25)),
            recon_edge_weight=float(loss_data.get("recon_edge_weight", 0.2)),
            ssim_downsample_factor=int(loss_data.get("ssim_downsample_factor", 2)),
        ),
        output=Stage4OutputConfig(**output_data),
        initialization=Stage4InitializationConfig(**init_data),
    )
