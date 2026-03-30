from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "pretrained" / "rtdetr_r50vd"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download RT-DETR weights into the project for offline reuse.")
    parser.add_argument(
        "--repo-id",
        default="PekingU/rtdetr_r50vd",
        help="Hugging Face model repo id.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory inside the project where the model snapshot will be stored.",
    )
    parser.add_argument(
        "--token",
        help="Optional Hugging Face token for private or rate-limited access.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    local_path = snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(output_dir),
        token=args.token,
        ignore_patterns=["*.h5", "*.ot", "flax_model.msgpack"],
    )

    print(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "output_dir": str(output_dir),
                "local_snapshot": str(local_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()