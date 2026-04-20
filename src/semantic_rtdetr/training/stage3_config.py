from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _as_int_list(values: list[Any] | tuple[Any, ...] | None, default: list[int]) -> list[int]:
    if values is None:
        return list(default)
    return [int(v) for v in values]


def _as_float_list(values: list[Any] | tuple[Any, ...] | None, default: list[float]) -> list[float]:
    if values is None:
        return list(default)
    return [float(v) for v in values]


@dataclass(frozen=True)
class Stage3DetectorConfig:
    hf_name: str = "PekingU/rtdetr_r50vd"
    local_path: str | None = "pretrained/rtdetr_r50vd"
    cache_dir: str | None = None
    device: str = "auto"


@dataclass(frozen=True)
class Stage3DataConfig:
    dataset_name: str = "generic"
    train_source_path: str | None = None
    val_source_path: str | None = None
    recursive: bool = True
    index_cache_dir: str | None = ".cache/stage3_data"
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
class Stage3MDVSCConfig:
    backbone_channels: list[int] = field(default_factory=lambda: [512, 1024, 2048])
    feature_channels: list[int] = field(default_factory=lambda: [256, 256, 256])
    latent_dims: list[int] = field(default_factory=lambda: [64, 80, 96])
    common_keep_ratios: list[float] = field(default_factory=lambda: [0.6, 0.7, 0.8])
    individual_keep_ratios: list[float] = field(default_factory=lambda: [0.15, 0.2, 0.25])
    block_sizes: list[int] = field(default_factory=lambda: [4, 2, 1])
    spatial_strides: list[int] = field(default_factory=lambda: [2, 2, 1])
    apply_cross_level_fusion: bool = True
    apply_masks: bool = True
    channel_mode: str = "identity"
    snr_db: float = 20.0


@dataclass(frozen=True)
class Stage3OptimizationConfig:
    batch_size: int = 1
    num_workers: int = 4
    use_amp: bool = True
    amp_dtype: str = "float16"
    optimizer: str = "adamw"
    epochs: int = 20
    lr: float = 2e-4
    weight_decay: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    scheduler: str = "onecycle"
    warmup_epochs: int = 2
    warmup_start_factor: float = 0.1
    min_lr_ratio: float = 0.05
    onecycle_pct_start: float = 0.2
    onecycle_div_factor: float = 20.0
    onecycle_final_div_factor: float = 500.0
    grad_clip_norm: float = 1.0
    log_every: int = 10
    save_every_epochs: int = 1
    max_steps_per_epoch: int | None = None
    seed: int = 42


@dataclass(frozen=True)
class Stage3LossConfig:
    feature_loss_type: str = "smooth_l1"
    level_loss_weights: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])


@dataclass(frozen=True)
class Stage3OutputConfig:
    output_dir: str = "outputs/mdvsc_stage3"
    save_visualizations: bool = True
    visualization_every_epochs: int = 1
    visualization_num_frames: int = 4


@dataclass(frozen=True)
class Stage3InitializationConfig:
    stage2_checkpoint: str | None = None
    checkpoint: str | None = None
    strict: bool = False


@dataclass(frozen=True)
class MDVSCStage3TrainConfig:
    detector: Stage3DetectorConfig = field(default_factory=Stage3DetectorConfig)
    data: Stage3DataConfig = field(default_factory=Stage3DataConfig)
    mdvsc: Stage3MDVSCConfig = field(default_factory=Stage3MDVSCConfig)
    optimization: Stage3OptimizationConfig = field(default_factory=Stage3OptimizationConfig)
    loss: Stage3LossConfig = field(default_factory=Stage3LossConfig)
    output: Stage3OutputConfig = field(default_factory=Stage3OutputConfig)
    initialization: Stage3InitializationConfig = field(default_factory=Stage3InitializationConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_stage3_config(config_path: str | Path) -> MDVSCStage3TrainConfig:
    path = Path(config_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    detector_data = data.get("detector") or {}
    data_data = data.get("data") or {}
    mdvsc_data = data.get("mdvsc") or {}
    opt_data = data.get("optimization") or {}
    loss_data = data.get("loss") or {}
    output_data = data.get("output") or {}
    init_data = data.get("initialization") or {}

    return MDVSCStage3TrainConfig(
        detector=Stage3DetectorConfig(**detector_data),
        data=Stage3DataConfig(**data_data),
        mdvsc=Stage3MDVSCConfig(
            backbone_channels=_as_int_list(mdvsc_data.get("backbone_channels"), [512, 1024, 2048]),
            feature_channels=_as_int_list(mdvsc_data.get("feature_channels"), [256, 256, 256]),
            latent_dims=_as_int_list(mdvsc_data.get("latent_dims"), [64, 80, 96]),
            common_keep_ratios=_as_float_list(mdvsc_data.get("common_keep_ratios"), [0.6, 0.7, 0.8]),
            individual_keep_ratios=_as_float_list(mdvsc_data.get("individual_keep_ratios"), [0.15, 0.2, 0.25]),
            block_sizes=_as_int_list(mdvsc_data.get("block_sizes"), [4, 2, 1]),
            spatial_strides=_as_int_list(mdvsc_data.get("spatial_strides"), [2, 2, 1]),
            apply_cross_level_fusion=bool(mdvsc_data.get("apply_cross_level_fusion", True)),
            apply_masks=bool(mdvsc_data.get("apply_masks", True)),
            channel_mode=str(mdvsc_data.get("channel_mode", "identity")),
            snr_db=float(mdvsc_data.get("snr_db", 20.0)),
        ),
        optimization=Stage3OptimizationConfig(**opt_data),
        loss=Stage3LossConfig(
            feature_loss_type=str(loss_data.get("feature_loss_type", "smooth_l1")),
            level_loss_weights=_as_float_list(loss_data.get("level_loss_weights"), [1.0, 1.0, 1.0]),
        ),
        output=Stage3OutputConfig(**output_data),
        initialization=Stage3InitializationConfig(**init_data),
    )
