from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class FeatureLevelSpec:
    level: int
    batch_size: int
    channels: int
    height: int
    width: int
    dtype: str
    stride: int | None = None


@dataclass(frozen=True)
class EncoderFeatureContract:
    levels: list[FeatureLevelSpec]
    spatial_shapes: list[list[int]]
    level_start_index: list[int]
    flattened_sequence_length: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EncoderFeatureBundle:
    feature_maps: list[torch.Tensor]
    spatial_shapes: torch.Tensor
    level_start_index: torch.Tensor
    strides: list[int] | None = None

    def clone(self) -> "EncoderFeatureBundle":
        return EncoderFeatureBundle(
            feature_maps=[feature_map.clone() for feature_map in self.feature_maps],
            spatial_shapes=self.spatial_shapes.clone(),
            level_start_index=self.level_start_index.clone(),
            strides=list(self.strides) if self.strides is not None else None,
        )

    def select_levels(self, level_indices: list[int]) -> "EncoderFeatureBundle":
        if not level_indices:
            return self.clone()

        selected_feature_maps = [self.feature_maps[level_index].clone() for level_index in level_indices]
        selected_spatial_shapes = self.spatial_shapes[level_indices].clone()
        selected_strides = [self.strides[level_index] for level_index in level_indices] if self.strides else None
        level_sizes = selected_spatial_shapes.prod(1)
        level_start_index = torch.cat(
            (level_sizes.new_zeros((1,)), level_sizes.cumsum(0)[:-1])
        )
        return EncoderFeatureBundle(
            feature_maps=selected_feature_maps,
            spatial_shapes=selected_spatial_shapes,
            level_start_index=level_start_index,
            strides=selected_strides,
        )

    def replace_levels(
        self,
        level_indices: list[int],
        replacement_feature_maps: list[torch.Tensor],
    ) -> "EncoderFeatureBundle":
        if len(level_indices) != len(replacement_feature_maps):
            raise ValueError("level_indices and replacement_feature_maps must have the same length")

        updated_bundle = self.clone()
        for level_index, replacement in zip(level_indices, replacement_feature_maps):
            updated_bundle.feature_maps[level_index] = replacement.clone()
        return updated_bundle

    def contract(self) -> EncoderFeatureContract:
        if len(self.feature_maps) != int(self.spatial_shapes.shape[0]):
            raise ValueError("feature_maps and spatial_shapes must have the same number of levels")

        strides = self.strides or [None] * len(self.feature_maps)
        total_sequence_length = 0
        levels: list[FeatureLevelSpec] = []

        for index, feature_map in enumerate(self.feature_maps):
            if feature_map.ndim != 4:
                raise ValueError("Each feature map must have shape [batch, channels, height, width]")

            batch_size, channels, height, width = feature_map.shape
            total_sequence_length += height * width
            levels.append(
                FeatureLevelSpec(
                    level=index,
                    batch_size=batch_size,
                    channels=channels,
                    height=height,
                    width=width,
                    dtype=str(feature_map.dtype).replace("torch.", ""),
                    stride=strides[index],
                )
            )

        return EncoderFeatureContract(
            levels=levels,
            spatial_shapes=self.spatial_shapes.detach().cpu().tolist(),
            level_start_index=self.level_start_index.detach().cpu().tolist(),
            flattened_sequence_length=total_sequence_length,
        )

    def to_tensor_dict(self) -> dict[str, Any]:
        return {
            "feature_maps": [feature_map.detach().cpu() for feature_map in self.feature_maps],
            "spatial_shapes": self.spatial_shapes.detach().cpu(),
            "level_start_index": self.level_start_index.detach().cpu(),
            "strides": self.strides,
        }