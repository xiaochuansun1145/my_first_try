from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.semantic_rtdetr.training.stage4_config import load_stage4_config
from src.semantic_rtdetr.training.stage4_trainer import run_stage4_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Stage 4: end-to-end joint training (Stage 2.1 + Stage 3)."
    )
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "mdvsc_stage4.yaml"),
        help="Path to the stage-4 YAML config.",
    )
    parser.add_argument("--data", help="Override data.train_source_path.")
    parser.add_argument("--output", help="Override output.output_dir.")
    parser.add_argument(
        "--stage2-1-ckpt",
        help="Override initialization.stage2_1_checkpoint (stage-2.1 best.pt).",
    )
    parser.add_argument(
        "--stage3-ckpt",
        help="Override initialization.stage3_checkpoint (stage-3 best.pt).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_stage4_config(args.config)

    if args.data:
        config = replace(config, data=replace(config.data, train_source_path=args.data))
    if args.output:
        config = replace(config, output=replace(config.output, output_dir=args.output))
    if args.stage2_1_ckpt:
        config = replace(config, initialization=replace(
            config.initialization, stage2_1_checkpoint=args.stage2_1_ckpt
        ))
    if args.stage3_ckpt:
        config = replace(config, initialization=replace(
            config.initialization, stage3_checkpoint=args.stage3_ckpt
        ))

    summary = run_stage4_training(config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
