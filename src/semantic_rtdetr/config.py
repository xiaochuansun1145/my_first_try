from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ModelConfig:
    hf_name: str = "PekingU/rtdetr_r50vd"
    device: str = "auto"
    threshold: float = 0.3


@dataclass(frozen=True)
class InputConfig:
    image_path: str | None = None


@dataclass(frozen=True)
class OutputConfig:
    output_dir: str = "outputs/rtdetr_baseline"
    save_features: bool = True
    save_visualization: bool = True


@dataclass(frozen=True)
class ChannelConfig:
    mode: str = "identity"
    snr_db: float = 20.0
    seed: int = 0


@dataclass(frozen=True)
class SemComConfig:
    selected_levels: list[int] = field(default_factory=list)
    adaptor: str = "identity"


@dataclass(frozen=True)
class RTDetrBaselineConfig:
    model: ModelConfig
    input: InputConfig
    output: OutputConfig
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    semcom: SemComConfig = field(default_factory=SemComConfig)


def load_baseline_config(config_path: str | Path) -> RTDetrBaselineConfig:
    path = Path(config_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    return RTDetrBaselineConfig(
        model=ModelConfig(**(data.get("model") or {})),
        input=InputConfig(**(data.get("input") or {})),
        output=OutputConfig(**(data.get("output") or {})),
        channel=ChannelConfig(**(data.get("channel") or {})),
        semcom=SemComConfig(**(data.get("semcom") or {})),
    )