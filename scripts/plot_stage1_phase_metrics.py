from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot combined metrics for the three separately trained stage-1 phases.")
    parser.add_argument("--reconstruction-dir", required=True, help="Output directory of the reconstruction-only run.")
    parser.add_argument("--transmission-dir", required=True, help="Output directory of the transmission-only run.")
    parser.add_argument("--joint-dir", required=True, help="Output directory of the joint finetuning run.")
    parser.add_argument("--output-dir", help="Directory for generated plots. Defaults to <joint-dir>/phase_plots.")
    return parser.parse_args()


def _load_metrics(run_dir: Path) -> list[dict]:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    rows: list[dict] = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"Metrics file is empty: {metrics_path}")
    return rows


def _append_rows(
    rows: list[dict],
    phase_name: str,
    offset: int,
    combined_rows: list[dict],
    boundaries: list[tuple[int, str]],
) -> int:
    if combined_rows:
        boundaries.append((offset + 1, phase_name))
    for local_epoch, row in enumerate(rows, start=1):
        combined_rows.append(
            {
                "epoch": offset + local_epoch,
                "phase": phase_name,
                "lr": row.get("lr", 0.0),
                "train": row.get("train"),
                "val": row.get("val"),
            }
        )
    return offset + len(rows)


def _series(rows: list[dict], split: str, key: str) -> list[float | None]:
    values: list[float | None] = []
    for row in rows:
        split_metrics = row.get(split)
        values.append(None if split_metrics is None else split_metrics.get(key))
    return values


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.joint_dir) / "phase_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    phase_inputs = [
        ("reconstruction", Path(args.reconstruction_dir)),
        ("transmission", Path(args.transmission_dir)),
        ("joint", Path(args.joint_dir)),
    ]

    combined_rows: list[dict] = []
    boundaries: list[tuple[int, str]] = []
    offset = 0
    for phase_name, run_dir in phase_inputs:
        offset = _append_rows(_load_metrics(run_dir), phase_name, offset, combined_rows, boundaries)

    epochs = [row["epoch"] for row in combined_rows]
    learning_rates = [row["lr"] for row in combined_rows]

    import matplotlib.pyplot as plt

    def plot_train_val(ax, key: str, title: str, ylabel: str) -> None:
        train_values = _series(combined_rows, "train", key)
        val_values = _series(combined_rows, "val", key)
        ax.plot(epochs, train_values, marker="o", label="train")
        if any(value is not None for value in val_values):
            ax.plot(epochs, val_values, marker="s", label="val")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        for boundary_epoch, phase in boundaries:
            ax.axvline(boundary_epoch - 0.5, color="gray", linestyle="--", alpha=0.4)
            ax.text(boundary_epoch - 0.45, ax.get_ylim()[1], phase, rotation=90, va="top", ha="left", fontsize=8)

    figure, axes = plt.subplots(2, 3, figsize=(18, 10))
    plot_train_val(axes[0, 0], "total_loss", "Total Loss", "loss")
    plot_train_val(axes[0, 1], "feature_loss", "Feature Loss", "loss")
    plot_train_val(axes[0, 2], "recon_ssim_loss", "SSIM Loss", "loss")
    plot_train_val(axes[1, 0], "recon_l1_loss", "L1 Reconstruction", "loss")
    plot_train_val(axes[1, 1], "detection_logit_loss", "Detection Logit", "loss")
    plot_train_val(axes[1, 2], "common_active_ratio", "Common Mask Activity", "ratio")
    figure.tight_layout()
    figure.savefig(output_dir / "phase_loss_overview.png", dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(epochs, learning_rates, marker="o")
    axis.set_title("Learning Rate by Epoch")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("lr")
    axis.grid(True, alpha=0.3)
    for boundary_epoch, phase in boundaries:
        axis.axvline(boundary_epoch - 0.5, color="gray", linestyle="--", alpha=0.4)
        axis.text(boundary_epoch - 0.45, axis.get_ylim()[1], phase, rotation=90, va="top", ha="left", fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "phase_learning_rate.png", dpi=200)
    plt.close(figure)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "plots": [
                    "phase_loss_overview.png",
                    "phase_learning_rate.png",
                ],
                "boundaries": boundaries,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()