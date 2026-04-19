from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.semantic_rtdetr.training.stage3_config import load_stage3_config
from src.semantic_rtdetr.training.stage3_trainer import run_stage3_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MDVSC v2 (stage 3): feature-map reconstruction.")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "mdvsc_stage3.yaml"),
        help="Path to the stage-3 YAML config.",
    )
    parser.add_argument("--data", help="Override data.train_source_path.")
    parser.add_argument("--output", help="Override output.output_dir.")
    parser.add_argument("--stage2-ckpt", help="Override initialization.stage2_checkpoint (path to stage-2 .pt file).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_stage3_config(args.config)

    if args.data:
        config = replace(config, data=replace(config.data, train_source_path=args.data))
    if args.output:
        config = replace(config, output=replace(config.output, output_dir=args.output))
    if args.stage2_ckpt:
        config = replace(config, initialization=replace(config.initialization, stage2_checkpoint=args.stage2_ckpt))

    summary = run_stage3_training(config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
