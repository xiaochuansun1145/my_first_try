from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.semantic_rtdetr.config import SemComConfig
from src.semantic_rtdetr.contracts import EncoderFeatureBundle


@dataclass(frozen=True)
class FeaturePacketContract:
    selected_levels: list[int]
    bypassed_levels: list[int]
    adaptor: str
    num_transmitted_levels: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FeaturePacket:
    feature_bundle: EncoderFeatureBundle
    selected_levels: list[int]
    bypassed_levels: list[int]
    adaptor: str

    def contract(self) -> FeaturePacketContract:
        return FeaturePacketContract(
            selected_levels=self.selected_levels,
            bypassed_levels=self.bypassed_levels,
            adaptor=self.adaptor,
            num_transmitted_levels=len(self.selected_levels),
        )


class FeatureSemanticCodec:
    def __init__(self, config: SemComConfig):
        self.config = config

    def resolve_selected_levels(self, feature_bundle: EncoderFeatureBundle) -> list[int]:
        if self.config.selected_levels:
            selected_levels = list(self.config.selected_levels)
        else:
            selected_levels = list(range(len(feature_bundle.feature_maps)))

        max_level = len(feature_bundle.feature_maps) - 1
        for level in selected_levels:
            if level < 0 or level > max_level:
                raise ValueError(f"Selected level {level} is outside the available range [0, {max_level}]")
        return selected_levels

    def encode(self, feature_bundle: EncoderFeatureBundle) -> FeaturePacket:
        selected_levels = self.resolve_selected_levels(feature_bundle)
        bypassed_levels = [level for level in range(len(feature_bundle.feature_maps)) if level not in selected_levels]
        transmitted_bundle = feature_bundle.select_levels(selected_levels)
        return FeaturePacket(
            feature_bundle=transmitted_bundle,
            selected_levels=selected_levels,
            bypassed_levels=bypassed_levels,
            adaptor=self.config.adaptor,
        )

    def decode(
        self,
        feature_packet: FeaturePacket,
        reference_bundle: EncoderFeatureBundle,
    ) -> EncoderFeatureBundle:
        return reference_bundle.replace_levels(
            feature_packet.selected_levels,
            feature_packet.feature_bundle.feature_maps,
        )


def build_feature_semantic_codec(config: SemComConfig) -> FeatureSemanticCodec:
    if config.adaptor != "identity":
        raise ValueError(f"Unsupported adaptor mode: {config.adaptor}")
    return FeatureSemanticCodec(config)