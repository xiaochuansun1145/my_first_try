from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.semantic_rtdetr.config import load_baseline_config
from src.semantic_rtdetr.pipeline.semcom_pipeline import run_semcom_experiment, save_semcom_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RT-DETR + feature-channel MVP on one image.")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "rtdetr_semcom_mvp.yaml"),
        help="Path to the semcom YAML config.",
    )
    parser.add_argument("--image", help="Override the image_path from the YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_baseline_config(args.config)

    image_path = Path(args.image or config.input.image_path or "")
    if not image_path.is_file():
        raise FileNotFoundError("Provide an existing image via --image or configs/rtdetr_semcom_mvp.yaml")

    output_dir = REPO_ROOT / config.output.output_dir
    summary = run_semcom_experiment(config, image_path)
    save_semcom_artifacts(summary, config, output_dir)

    print(
        json.dumps(
            {
                "channel_mode": config.channel.mode,
                "selected_levels": summary["feature_packet"]["selected_levels"],
                "num_baseline_detections": summary["baseline"]["num_detections"],
                "num_transmitted_detections": summary["transmitted"]["num_detections"],
                "feature_mse": summary["channel"]["feature_mse"],
                "detection_delta": summary["detection_delta"],
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()