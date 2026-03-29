from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch

from src.semantic_rtdetr.config import ChannelConfig
from src.semantic_rtdetr.semantic_comm.codec import FeaturePacket


@dataclass(frozen=True)
class FeatureTransmissionMetrics:
    channel_mode: str
    target_snr_db: float | None
    measured_snr_db: float | None
    feature_mse: float
    feature_psnr_db: float | None
    estimated_payload_bytes_fp32: int
    estimated_bits_per_input_pixel: float
    total_feature_values: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FeatureChannelResult:
    received_packet: FeaturePacket
    metrics: FeatureTransmissionMetrics


class FeatureChannel:
    def __init__(self, config: ChannelConfig):
        self.config = config

    def transmit(
        self,
        feature_packet: FeaturePacket,
        image_size: tuple[int, int],
    ) -> FeatureChannelResult:
        raise NotImplementedError

    def _build_result(
        self,
        original_packet: FeaturePacket,
        feature_maps: list[torch.Tensor],
        image_size: tuple[int, int],
    ) -> FeatureChannelResult:
        received_packet = FeaturePacket(
            feature_bundle=original_packet.feature_bundle.replace_levels(
                list(range(len(original_packet.feature_bundle.feature_maps))),
                feature_maps,
            ),
            selected_levels=list(original_packet.selected_levels),
            bypassed_levels=list(original_packet.bypassed_levels),
            adaptor=original_packet.adaptor,
        )
        metrics = compute_feature_metrics(
            original_packet,
            received_packet,
            image_size=image_size,
            channel_mode=self.config.mode,
            target_snr_db=self.config.snr_db if self.config.mode == "awgn" else None,
        )
        return FeatureChannelResult(received_packet=received_packet, metrics=metrics)


class IdentityFeatureChannel(FeatureChannel):
    def transmit(
        self,
        feature_packet: FeaturePacket,
        image_size: tuple[int, int],
    ) -> FeatureChannelResult:
        feature_maps = [feature_map.clone() for feature_map in feature_packet.feature_bundle.feature_maps]
        return self._build_result(feature_packet, feature_maps, image_size)


class AWGNFeatureChannel(FeatureChannel):
    def transmit(
        self,
        feature_packet: FeaturePacket,
        image_size: tuple[int, int],
    ) -> FeatureChannelResult:
        noisy_feature_maps: list[torch.Tensor] = []

        for level, feature_map in enumerate(feature_packet.feature_bundle.feature_maps):
            signal_power = float(feature_map.pow(2).mean().item())
            if signal_power == 0.0:
                noisy_feature_maps.append(feature_map.clone())
                continue

            snr_linear = 10.0 ** (self.config.snr_db / 10.0)
            noise_power = signal_power / snr_linear
            noise_std = math.sqrt(noise_power)
            generator = torch.Generator(device=feature_map.device.type)
            generator.manual_seed(self.config.seed + level)
            noise = torch.randn(
                feature_map.shape,
                generator=generator,
                device=feature_map.device,
                dtype=feature_map.dtype,
            ) * noise_std
            noisy_feature_maps.append(feature_map + noise)

        return self._build_result(feature_packet, noisy_feature_maps, image_size)


def build_feature_channel(config: ChannelConfig) -> FeatureChannel:
    if config.mode == "identity":
        return IdentityFeatureChannel(config)
    if config.mode == "awgn":
        return AWGNFeatureChannel(config)
    raise ValueError(f"Unsupported channel mode: {config.mode}")


def compute_feature_metrics(
    original_packet: FeaturePacket,
    received_packet: FeaturePacket,
    image_size: tuple[int, int],
    channel_mode: str,
    target_snr_db: float | None,
) -> FeatureTransmissionMetrics:
    total_feature_values = 0
    signal_energy = 0.0
    noise_energy = 0.0
    peak_abs_value = 0.0

    for original_feature, received_feature in zip(
        original_packet.feature_bundle.feature_maps,
        received_packet.feature_bundle.feature_maps,
    ):
        diff = received_feature - original_feature
        total_feature_values += original_feature.numel()
        signal_energy += float(original_feature.pow(2).sum().item())
        noise_energy += float(diff.pow(2).sum().item())
        peak_abs_value = max(
            peak_abs_value,
            float(original_feature.abs().max().item()),
            float(received_feature.abs().max().item()),
        )

    feature_mse = noise_energy / max(total_feature_values, 1)
    measured_snr_db = None
    if noise_energy > 0.0 and signal_energy > 0.0:
        measured_snr_db = 10.0 * math.log10(signal_energy / noise_energy)

    feature_psnr_db = None
    if feature_mse > 0.0 and peak_abs_value > 0.0:
        feature_psnr_db = 20.0 * math.log10(peak_abs_value) - 10.0 * math.log10(feature_mse)

    image_width, image_height = image_size
    payload_bits = total_feature_values * 32
    estimated_bits_per_input_pixel = payload_bits / max(image_width * image_height, 1)

    return FeatureTransmissionMetrics(
        channel_mode=channel_mode,
        target_snr_db=target_snr_db,
        measured_snr_db=measured_snr_db,
        feature_mse=feature_mse,
        feature_psnr_db=feature_psnr_db,
        estimated_payload_bytes_fp32=payload_bits // 8,
        estimated_bits_per_input_pixel=estimated_bits_per_input_pixel,
        total_feature_values=total_feature_values,
    )