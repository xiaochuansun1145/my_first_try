from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.semantic_rtdetr.training.stage2_1_config import load_stage2_1_config
from src.semantic_rtdetr.training.stage2_1_trainer import run_stage2_1_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the stage-2.1 MDVSC model (SharedEncoder + DetailBypass + dual decoder)."
    )
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "mdvsc_stage2_1.yaml"),
        help="Path to the stage-2.1 training YAML config.",
    )
    parser.add_argument("--data", help="Override data.train_source_path.")
    parser.add_argument("--output", help="Override output.output_dir.")
    parser.add_argument("--stage2-checkpoint", help="Initialize from a stage-2 best.pt checkpoint.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_stage2_1_config(args.config)

    if args.data:
        config = replace(config, data=replace(config.data, train_source_path=args.data))
    if args.output:
        config = replace(config, output=replace(config.output, output_dir=args.output))
    if args.stage2_checkpoint:
        config = replace(config, initialization=replace(
            config.initialization, stage2_checkpoint=args.stage2_checkpoint
        ))

    summary = run_stage2_1_training(config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
