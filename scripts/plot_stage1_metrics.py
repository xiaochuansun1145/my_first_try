from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot stage-1 training metrics from metrics.jsonl.")
    parser.add_argument(
        "--metrics",
        default=str(REPO_ROOT / "outputs" / "mdvsc_stage1" / "metrics.jsonl"),
        help="Path to the stage-1 metrics.jsonl file.",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for the generated plots. Defaults to <metrics_dir>/plots.",
    )
    return parser.parse_args()


def _load_metrics(metrics_path: Path) -> list[dict]:
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


def _series(rows: list[dict], split: str, key: str) -> list[float | None]:
    values: list[float | None] = []
    for row in rows:
        split_metrics = row.get(split)
        values.append(None if split_metrics is None else split_metrics.get(key))
    return values


def _phase_boundaries(rows: list[dict]) -> list[tuple[int, str]]:
    boundaries: list[tuple[int, str]] = []
    previous_phase = None
    for row in rows:
        phase = row.get("phase")
        if previous_phase is not None and phase != previous_phase:
            boundaries.append((int(row["epoch"]), str(phase)))
        previous_phase = phase
    return boundaries


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.metrics)
    output_dir = Path(args.output_dir) if args.output_dir else metrics_path.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_metrics(metrics_path)
    epochs = [row["epoch"] for row in rows]
    learning_rates = [row["lr"] for row in rows]
    phase_boundaries = _phase_boundaries(rows)

    import matplotlib.pyplot as plt

    def plot_train_val(ax, key: str, title: str, ylabel: str) -> None:
        train_values = _series(rows, "train", key)
        val_values = _series(rows, "val", key)
        ax.plot(epochs, train_values, marker="o", label="train")
        if any(value is not None for value in val_values):
            ax.plot(epochs, val_values, marker="s", label="val")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        for boundary_epoch, phase in phase_boundaries:
            ax.axvline(boundary_epoch - 0.5, color="gray", linestyle="--", alpha=0.4)
            ax.text(boundary_epoch - 0.45, ax.get_ylim()[1], phase, rotation=90, va="top", ha="left", fontsize=8)

    figure, axes = plt.subplots(2, 3, figsize=(18, 10))
    plot_train_val(axes[0, 0], "total_loss", "Total Loss", "loss")
    plot_train_val(axes[0, 1], "feature_loss", "Feature Reconstruction Loss", "loss")
    plot_train_val(axes[0, 2], "recon_ssim_loss", "Frame Reconstruction SSIM Loss", "loss")
    plot_train_val(axes[1, 0], "recon_l1_loss", "Frame Reconstruction L1", "loss")
    plot_train_val(axes[1, 1], "recon_mse_loss", "Frame Reconstruction MSE", "loss")
    axes[1, 2].axis("off")
    figure.tight_layout()
    figure.savefig(output_dir / "loss_overview.png", dpi=200)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_train_val(axes[0], "detection_logit_loss", "Detection Logit Consistency", "loss")
    plot_train_val(axes[1], "detection_box_loss", "Detection Box Consistency", "loss")
    figure.tight_layout()
    figure.savefig(output_dir / "detection_consistency.png", dpi=200)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_train_val(axes[0], "common_active_ratio", "Common Branch Active Ratio", "ratio")
    plot_train_val(axes[1], "individual_active_ratio", "Individual Branch Active Ratio", "ratio")
    figure.tight_layout()
    figure.savefig(output_dir / "mask_activity.png", dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, learning_rates, marker="o")
    axis.set_title("Learning Rate by Epoch")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("lr")
    axis.grid(True, alpha=0.3)
    for boundary_epoch, phase in phase_boundaries:
        axis.axvline(boundary_epoch - 0.5, color="gray", linestyle="--", alpha=0.4)
        axis.text(boundary_epoch - 0.45, axis.get_ylim()[1], phase, rotation=90, va="top", ha="left", fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "learning_rate.png", dpi=200)
    plt.close(figure)

    print(
        json.dumps(
            {
                "metrics_path": str(metrics_path),
                "output_dir": str(output_dir),
                "plots": [
                    "loss_overview.png",
                    "detection_consistency.png",
                    "mask_activity.png",
                    "learning_rate.png",
                ],
                    "phase_boundaries": phase_boundaries,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()