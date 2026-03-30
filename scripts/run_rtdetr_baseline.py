from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.semantic_rtdetr.config import load_baseline_config
from src.semantic_rtdetr.detector.rtdetr_baseline import RTDetrBaseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the split-ready RT-DETR baseline on one image.")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "rtdetr_baseline.yaml"),
        help="Path to the baseline YAML config.",
    )
    parser.add_argument("--image", help="Override the image_path from the YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_baseline_config(args.config)

    image_path = Path(args.image or config.input.image_path or "")
    if not image_path.is_file():
        raise FileNotFoundError("Provide an existing image via --image or configs/rtdetr_baseline.yaml")

    output_dir = REPO_ROOT / config.output.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    baseline = RTDetrBaseline(
        config.model.hf_name,
        device=config.model.device,
        local_path=config.model.local_path,
        cache_dir=config.model.cache_dir,
    )
    inputs = baseline.prepare_inputs(image)

    direct_outputs = baseline.predict(inputs)
    detections = baseline.post_process(direct_outputs, image.size, config.model.threshold)

    feature_bundle = baseline.extract_encoder_feature_bundle(inputs)
    roundtrip_outputs = baseline.predict_from_encoder_feature_bundle(inputs, feature_bundle)
    roundtrip = baseline.compare_outputs(direct_outputs, roundtrip_outputs)

    summary = {
        "model_name": config.model.hf_name,
        "device": str(baseline.device),
        "image_path": str(image_path),
        "num_detections": len(detections),
        "detections": detections,
        "roundtrip": roundtrip.to_dict(),
        "feature_contract": feature_bundle.contract().to_dict(),
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "feature_contract.json").write_text(
        json.dumps(feature_bundle.contract().to_dict(), indent=2),
        encoding="utf-8",
    )

    if config.output.save_features:
        import torch

        torch.save(feature_bundle.to_tensor_dict(), output_dir / "encoder_features.pt")

    if config.output.save_visualization:
        baseline.save_visualization(image, detections, output_dir / "annotated.jpg")

    print(json.dumps({
        "num_detections": len(detections),
        "roundtrip": roundtrip.to_dict(),
        "output_dir": str(output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()