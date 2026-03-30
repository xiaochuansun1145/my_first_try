from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.semantic_rtdetr.config import load_baseline_config
from src.semantic_rtdetr.detector.rtdetr_baseline import RTDetrBaseline
from src.semantic_rtdetr.pipeline.semcom_pipeline import run_semcom_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single-image semantic communication sweep.")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "rtdetr_semcom_mvp.yaml"),
        help="Base config path.",
    )
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument(
        "--level-sets",
        nargs="+",
        default=["0,1,2", "2"],
        help="Space-separated level sets, each encoded as comma-separated integers.",
    )
    parser.add_argument(
        "--snr-db",
        nargs="+",
        type=float,
        default=[20.0, 12.0, 8.0],
        help="AWGN SNR values to evaluate.",
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "outputs" / "semcom_sweep.json"),
        help="Path to the JSON summary file.",
    )
    return parser.parse_args()


def parse_level_set(raw: str) -> list[int]:
    if not raw:
        return []
    return [int(part) for part in raw.split(",") if part]


def main() -> None:
    args = parse_args()
    base_config = load_baseline_config(args.config)
    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    results: list[dict[str, object]] = []
    level_sets = [parse_level_set(raw_levels) for raw_levels in args.level_sets]
    baseline = RTDetrBaseline(
        base_config.model.hf_name,
        device=base_config.model.device,
        local_path=base_config.model.local_path,
        cache_dir=base_config.model.cache_dir,
    )

    for level_set in level_sets:
        identity_config = replace(
            base_config,
            channel=replace(base_config.channel, mode="identity"),
            semcom=replace(base_config.semcom, selected_levels=level_set),
        )
        summary = run_semcom_experiment(identity_config, image_path, baseline=baseline)
        results.append(
            {
                "channel_mode": "identity",
                "selected_levels": level_set,
                "feature_mse": summary["channel"]["feature_mse"],
                "estimated_bits_per_input_pixel": summary["channel"]["estimated_bits_per_input_pixel"],
                "max_abs_logit_diff": summary["detection_delta"]["max_abs_logit_diff"],
                "max_abs_box_diff": summary["detection_delta"]["max_abs_box_diff"],
            }
        )

        for snr_db in args.snr_db:
            awgn_config = replace(
                base_config,
                channel=replace(base_config.channel, mode="awgn", snr_db=snr_db),
                semcom=replace(base_config.semcom, selected_levels=level_set),
            )
            summary = run_semcom_experiment(awgn_config, image_path, baseline=baseline)
            results.append(
                {
                    "channel_mode": "awgn",
                    "selected_levels": level_set,
                    "target_snr_db": snr_db,
                    "measured_snr_db": summary["channel"]["measured_snr_db"],
                    "feature_mse": summary["channel"]["feature_mse"],
                    "estimated_bits_per_input_pixel": summary["channel"]["estimated_bits_per_input_pixel"],
                    "max_abs_logit_diff": summary["detection_delta"]["max_abs_logit_diff"],
                    "max_abs_box_diff": summary["detection_delta"]["max_abs_box_diff"],
                }
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"num_runs": len(results), "output": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()