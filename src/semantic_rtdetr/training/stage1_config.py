from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _as_int_list(values: list[Any] | tuple[Any, ...] | None, default: list[int]) -> list[int]:
    if values is None:
        return list(default)
    return [int(value) for value in values]


def _as_float_list(values: list[Any] | tuple[Any, ...] | None, default: list[float]) -> list[float]:
    if values is None:
        return list(default)
    return [float(value) for value in values]


@dataclass(frozen=True)
class Stage1DetectorConfig:
    hf_name: str = "PekingU/rtdetr_r50vd"
    local_path: str | None = "pretrained/rtdetr_r50vd"
    cache_dir: str | None = None
    device: str = "auto"


@dataclass(frozen=True)
class Stage1DataConfig:
    dataset_name: str = "generic"
    train_source_path: str | None = None
    val_source_path: str | None = None
    recursive: bool = True
    index_cache_dir: str | None = ".cache/stage1_data"
    subset_seed: int = 42
    source_fraction: float = 1.0
    sample_fraction: float = 1.0
    gop_size: int = 4
    frame_height: int = 640
    frame_width: int = 640
    frame_stride: int = 1
    gop_stride: int = 2
    max_frames_per_source: int | None = None
    max_sources: int | None = None
    max_samples: int | None = None
    train_val_split: float = 0.1


@dataclass(frozen=True)
class Stage1MDVSCConfig:
    feature_channels: list[int] = field(default_factory=lambda: [256, 256, 256])
    latent_dims: list[int] = field(default_factory=lambda: [32, 48, 64])
    common_keep_ratios: list[float] = field(default_factory=lambda: [0.5, 0.625, 0.75])
    individual_keep_ratios: list[float] = field(default_factory=lambda: [0.125, 0.1875, 0.25])
    block_sizes: list[int] = field(default_factory=lambda: [8, 4, 2])
    reconstruction_hidden_channels: int = 160
    reconstruction_detail_channels: int = 64
    reconstruction_head_type: str = "light"
    reconstruction_use_checkpoint: bool = False
    apply_masks: bool = True
    channel_mode: str = "identity"
    snr_db: float = 20.0


@dataclass(frozen=True)
class Stage1OptimizationConfig:
    batch_size: int = 1
    num_workers: int = 4
    use_amp: bool = True
    amp_dtype: str = "float16"
    optimizer: str = "adamw"
    reconstruction_pretrain_epochs: int = 3
    reconstruction_pretrain_lr: float = 3e-4
    mdvsc_bootstrap_epochs: int = 3
    mdvsc_bootstrap_lr: float = 1e-4
    epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    scheduler: str = "onecycle"
    warmup_epochs: int = 1
    warmup_start_factor: float = 0.2
    min_lr_ratio: float = 0.1
    onecycle_pct_start: float = 0.15
    onecycle_div_factor: float = 25.0
    onecycle_final_div_factor: float = 1000.0
    grad_clip_norm: float = 1.0
    log_every: int = 10
    save_every_epochs: int = 1
    max_steps_per_epoch: int | None = None
    seed: int = 42


@dataclass(frozen=True)
class Stage1LossConfig:
    feature_loss_weight: float = 1.0
    recon_l1_weight: float = 1.0
    recon_mse_weight: float = 0.25
    recon_ssim_weight: float = 0.25
    recon_edge_weight: float = 0.2
    ssim_downsample_factor: int = 2
    detection_logit_weight: float = 0.05
    detection_box_weight: float = 0.05
    level_loss_weights: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])


@dataclass(frozen=True)
class Stage1OutputConfig:
    output_dir: str = "outputs/mdvsc_stage1"
    save_visualizations: bool = True
    visualization_every_epochs: int = 1
    visualization_num_frames: int = 4


@dataclass(frozen=True)
class Stage1InitializationConfig:
    full_checkpoint: str | None = None
    reconstruction_checkpoint: str | None = None
    transmission_checkpoint: str | None = None
    strict: bool = False


@dataclass(frozen=True)
class MDVSCStage1TrainConfig:
    detector: Stage1DetectorConfig = field(default_factory=Stage1DetectorConfig)
    data: Stage1DataConfig = field(default_factory=Stage1DataConfig)
    mdvsc: Stage1MDVSCConfig = field(default_factory=Stage1MDVSCConfig)
    optimization: Stage1OptimizationConfig = field(default_factory=Stage1OptimizationConfig)
    loss: Stage1LossConfig = field(default_factory=Stage1LossConfig)
    output: Stage1OutputConfig = field(default_factory=Stage1OutputConfig)
    initialization: Stage1InitializationConfig = field(default_factory=Stage1InitializationConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_stage1_config(config_path: str | Path) -> MDVSCStage1TrainConfig:
    path = Path(config_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    detector_data = data.get("detector") or {}
    data_data = data.get("data") or {}
    mdvsc_data = data.get("mdvsc") or {}
    optimization_data = data.get("optimization") or {}
    loss_data = data.get("loss") or {}
    output_data = data.get("output") or {}
    initialization_data = data.get("initialization") or {}

    return MDVSCStage1TrainConfig(
        detector=Stage1DetectorConfig(**detector_data),
        data=Stage1DataConfig(**data_data),
        mdvsc=Stage1MDVSCConfig(
            feature_channels=_as_int_list(mdvsc_data.get("feature_channels"), [256, 256, 256]),
            latent_dims=_as_int_list(mdvsc_data.get("latent_dims"), [32, 48, 64]),
            common_keep_ratios=_as_float_list(mdvsc_data.get("common_keep_ratios"), [0.5, 0.625, 0.75]),
            individual_keep_ratios=_as_float_list(mdvsc_data.get("individual_keep_ratios"), [0.125, 0.1875, 0.25]),
            block_sizes=_as_int_list(mdvsc_data.get("block_sizes"), [8, 4, 2]),
            reconstruction_hidden_channels=int(mdvsc_data.get("reconstruction_hidden_channels", 160)),
            reconstruction_detail_channels=int(mdvsc_data.get("reconstruction_detail_channels", 64)),
            apply_masks=bool(mdvsc_data.get("apply_masks", True)),
            channel_mode=str(mdvsc_data.get("channel_mode", "identity")),
            snr_db=float(mdvsc_data.get("snr_db", 20.0)),
        ),
        optimization=Stage1OptimizationConfig(**optimization_data),
        loss=Stage1LossConfig(
            level_loss_weights=_as_float_list(loss_data.get("level_loss_weights"), [1.0, 1.0, 1.0]),
            feature_loss_weight=float(loss_data.get("feature_loss_weight", 1.0)),
            recon_l1_weight=float(loss_data.get("recon_l1_weight", 1.0)),
            recon_mse_weight=float(loss_data.get("recon_mse_weight", 0.25)),
            recon_ssim_weight=float(loss_data.get("recon_ssim_weight", 0.25)),
            recon_edge_weight=float(loss_data.get("recon_edge_weight", 0.2)),
            detection_logit_weight=float(loss_data.get("detection_logit_weight", 0.05)),
            detection_box_weight=float(loss_data.get("detection_box_weight", 0.05)),
        ),
        output=Stage1OutputConfig(**output_data),
        initialization=Stage1InitializationConfig(**initialization_data),
    )