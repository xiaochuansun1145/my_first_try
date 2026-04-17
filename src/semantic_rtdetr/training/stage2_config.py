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
class Stage2MDVSCConfig:
    backbone_channels: list[int] = field(default_factory=lambda: [512, 1024, 2048])
    shared_channels: int = 256
    reconstruction_hidden_channels: int = 160
    reconstruction_detail_channels: int = 64
    reconstruction_head_type: str = "light"
    reconstruction_use_checkpoint: bool = False


@dataclass(frozen=True)
class Stage2OptimizationConfig:
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
class Stage2LossConfig:
    det_recovery_weight: float = 1.0
    recon_l1_weight: float = 1.0
    recon_mse_weight: float = 0.25
    recon_ssim_weight: float = 0.25
    recon_edge_weight: float = 0.2
    ssim_downsample_factor: int = 2
    level_recovery_weights: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])


@dataclass(frozen=True)
class Stage2OutputConfig:
    output_dir: str = "outputs/mdvsc_stage2"
    save_visualizations: bool = True
    visualization_every_epochs: int = 1
    visualization_num_frames: int = 4


@dataclass(frozen=True)
class Stage2InitializationConfig:
    full_checkpoint: str | None = None
    stage1_recon_checkpoint: str | None = None
    strict: bool = False


@dataclass(frozen=True)
class MDVSCStage2TrainConfig:
    detector: Stage1DetectorConfig = field(default_factory=Stage1DetectorConfig)
    data: Stage1DataConfig = field(default_factory=Stage1DataConfig)
    mdvsc: Stage2MDVSCConfig = field(default_factory=Stage2MDVSCConfig)
    optimization: Stage2OptimizationConfig = field(default_factory=Stage2OptimizationConfig)
    loss: Stage2LossConfig = field(default_factory=Stage2LossConfig)
    output: Stage2OutputConfig = field(default_factory=Stage2OutputConfig)
    initialization: Stage2InitializationConfig = field(default_factory=Stage2InitializationConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_stage2_config(config_path: str | Path) -> MDVSCStage2TrainConfig:
    path = Path(config_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    detector_data = data.get("detector") or {}
    data_data = data.get("data") or {}
    mdvsc_data = data.get("mdvsc") or {}
    optimization_data = data.get("optimization") or {}
    loss_data = data.get("loss") or {}
    output_data = data.get("output") or {}
    initialization_data = data.get("initialization") or {}

    return MDVSCStage2TrainConfig(
        detector=Stage1DetectorConfig(**detector_data),
        data=Stage1DataConfig(**data_data),
        mdvsc=Stage2MDVSCConfig(
            backbone_channels=_as_int_list(mdvsc_data.get("backbone_channels"), [512, 1024, 2048]),
            shared_channels=int(mdvsc_data.get("shared_channels", 256)),
            reconstruction_hidden_channels=int(mdvsc_data.get("reconstruction_hidden_channels", 160)),
            reconstruction_detail_channels=int(mdvsc_data.get("reconstruction_detail_channels", 64)),
            reconstruction_head_type=str(mdvsc_data.get("reconstruction_head_type", "light")),
            reconstruction_use_checkpoint=bool(mdvsc_data.get("reconstruction_use_checkpoint", False)),
        ),
        optimization=Stage2OptimizationConfig(**optimization_data),
        loss=Stage2LossConfig(
            det_recovery_weight=float(loss_data.get("det_recovery_weight", 1.0)),
            recon_l1_weight=float(loss_data.get("recon_l1_weight", 1.0)),
            recon_mse_weight=float(loss_data.get("recon_mse_weight", 0.25)),
            recon_ssim_weight=float(loss_data.get("recon_ssim_weight", 0.25)),
            recon_edge_weight=float(loss_data.get("recon_edge_weight", 0.2)),
            ssim_downsample_factor=int(loss_data.get("ssim_downsample_factor", 2)),
            level_recovery_weights=_as_float_list(loss_data.get("level_recovery_weights"), [1.0, 1.0, 1.0]),
        ),
        output=Stage2OutputConfig(**output_data),
        initialization=Stage2InitializationConfig(**initialization_data),
    )
