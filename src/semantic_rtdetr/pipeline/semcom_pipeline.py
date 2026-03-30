from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from src.semantic_rtdetr.config import RTDetrBaselineConfig
from src.semantic_rtdetr.detector.rtdetr_baseline import RTDetrBaseline
from src.semantic_rtdetr.semantic_comm.channel import build_feature_channel
from src.semantic_rtdetr.semantic_comm.codec import build_feature_semantic_codec


def run_semcom_experiment(
    config: RTDetrBaselineConfig,
    image_path: str | Path,
    baseline: RTDetrBaseline | None = None,
) -> dict[str, Any]:
    resolved_image_path = Path(image_path)
    if not resolved_image_path.is_file():
        raise FileNotFoundError(f"Image not found: {resolved_image_path}")

    image = Image.open(resolved_image_path).convert("RGB")
    active_baseline = baseline or RTDetrBaseline(
        config.model.hf_name,
        device=config.model.device,
        local_path=config.model.local_path,
        cache_dir=config.model.cache_dir,
    )
    inputs = active_baseline.prepare_inputs(image)

    direct_outputs = active_baseline.predict(inputs)
    baseline_detections = active_baseline.post_process(direct_outputs, image.size, config.model.threshold)
    feature_bundle = active_baseline.extract_encoder_feature_bundle(inputs)

    semantic_codec = build_feature_semantic_codec(config.semcom)
    feature_packet = semantic_codec.encode(feature_bundle)
    channel = build_feature_channel(config.channel)
    channel_result = channel.transmit(feature_packet, image.size)
    received_bundle = semantic_codec.decode(channel_result.received_packet, feature_bundle)
    transmitted_outputs = active_baseline.predict_from_encoder_feature_bundle(inputs, received_bundle)
    transmitted_detections = active_baseline.post_process(transmitted_outputs, image.size, config.model.threshold)
    detection_delta = active_baseline.compare_outputs(direct_outputs, transmitted_outputs)

    return {
        "model_name": config.model.hf_name,
        "device": str(active_baseline.device),
        "image_path": str(resolved_image_path),
        "feature_packet": feature_packet.contract().to_dict(),
        "channel": channel_result.metrics.to_dict(),
        "feature_contract": feature_bundle.contract().to_dict(),
        "baseline": {
            "num_detections": len(baseline_detections),
            "detections": baseline_detections,
        },
        "transmitted": {
            "num_detections": len(transmitted_detections),
            "detections": transmitted_detections,
        },
        "detection_delta": detection_delta.to_dict(),
        "artifacts": {
            "image": image,
            "feature_bundle": feature_bundle,
            "feature_packet_bundle": feature_packet.feature_bundle,
            "received_bundle": received_bundle,
            "baseline_detections": baseline_detections,
            "transmitted_detections": transmitted_detections,
        },
    }


def save_semcom_artifacts(
    summary: dict[str, Any],
    config: RTDetrBaselineConfig,
    output_dir: str | Path,
) -> None:
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    persisted_summary = {key: value for key, value in summary.items() if key != "artifacts"}
    (resolved_output_dir / "summary.json").write_text(
        json.dumps(persisted_summary, indent=2),
        encoding="utf-8",
    )
    (resolved_output_dir / "feature_contract.json").write_text(
        json.dumps(summary["feature_contract"], indent=2),
        encoding="utf-8",
    )

    artifacts = summary["artifacts"]
    if config.output.save_features:
        torch.save(artifacts["feature_bundle"].to_tensor_dict(), resolved_output_dir / "encoder_features.pt")
        torch.save(
            artifacts["feature_packet_bundle"].to_tensor_dict(),
            resolved_output_dir / "transmitted_feature_packet.pt",
        )
        torch.save(artifacts["received_bundle"].to_tensor_dict(), resolved_output_dir / "received_encoder_features.pt")

    if config.output.save_visualization:
        baseline = RTDetrBaseline(
            config.model.hf_name,
            device=config.model.device,
            local_path=config.model.local_path,
            cache_dir=config.model.cache_dir,
        )
        baseline.save_visualization(
            artifacts["image"],
            artifacts["baseline_detections"],
            resolved_output_dir / "baseline_annotated.jpg",
        )
        baseline.save_visualization(
            artifacts["image"],
            artifacts["transmitted_detections"],
            resolved_output_dir / "transmitted_annotated.jpg",
        )