"""Plot training metrics from stage-2 or stage-2.1 metrics.jsonl files.

Usage:
    python scripts/plot_stage2_metrics.py --metrics outputs/mdvsc_stage2/metrics.jsonl
    python scripts/plot_stage2_metrics.py --metrics outputs/mdvsc_stage2_1/metrics.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot stage-2 / stage-2.1 training metrics.")
    parser.add_argument(
        "--metrics",
        default=str(REPO_ROOT / "outputs" / "mdvsc_stage2" / "metrics.jsonl"),
        help="Path to the metrics.jsonl file.",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for generated plots. Defaults to <metrics_dir>/plots.",
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


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.metrics)
    output_dir = Path(args.output_dir) if args.output_dir else metrics_path.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_metrics(metrics_path)
    epochs = [row["epoch"] for row in rows]
    learning_rates = [row["lr"] for row in rows]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def plot_train_val(ax, key: str, title: str, ylabel: str) -> None:
        train_values = _series(rows, "train", key)
        val_values = _series(rows, "val", key)
        ax.plot(epochs, train_values, marker="o", markersize=3, label="train")
        if any(value is not None for value in val_values):
            ax.plot(epochs, val_values, marker="s", markersize=3, label="val")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()

    # --- Loss overview (2x3 grid) ---
    figure, axes = plt.subplots(2, 3, figsize=(18, 10))
    plot_train_val(axes[0, 0], "total_loss", "Total Loss", "loss")
    plot_train_val(axes[0, 1], "recon_mse_loss", "Reconstruction MSE", "loss")
    plot_train_val(axes[0, 2], "det_recovery_loss", "Det Recovery MSE", "loss")
    plot_train_val(axes[1, 0], "recon_l1_loss", "Reconstruction L1", "loss")
    plot_train_val(axes[1, 1], "recon_ssim_loss", "Reconstruction SSIM Loss", "loss")
    plot_train_val(axes[1, 2], "recon_edge_loss", "Reconstruction Edge Loss", "loss")
    figure.tight_layout()
    figure.savefig(output_dir / "loss_overview.png", dpi=200)
    plt.close(figure)

    # --- Quality metrics (PSNR + SSIM) ---
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_train_val(axes[0], "recon_psnr", "Reconstruction PSNR", "dB")
    plot_train_val(axes[1], "recon_ssim", "Reconstruction SSIM", "SSIM (0-1)")
    figure.tight_layout()
    figure.savefig(output_dir / "quality_metrics.png", dpi=200)
    plt.close(figure)

    # --- Learning rate ---
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, learning_rates, marker="o", markersize=3)
    axis.set_title("Learning Rate by Epoch")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("lr")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "learning_rate.png", dpi=200)
    plt.close(figure)

    # --- Detail transmission ratio (stage 2.1 only) ---
    detail_ratios = _series(rows, "train", "detail_tx_ratio")
    if any(v is not None for v in detail_ratios):
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.plot(epochs, detail_ratios, marker="o", markersize=3, color="tab:orange")
        axis.set_title("Detail Packet Transmission Ratio")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("ratio (detail / main)")
        axis.grid(True, alpha=0.3)
        figure.tight_layout()
        figure.savefig(output_dir / "detail_tx_ratio.png", dpi=200)
        plt.close(figure)

    plot_names = ["loss_overview.png", "quality_metrics.png", "learning_rate.png"]
    if any(v is not None for v in detail_ratios):
        plot_names.append("detail_tx_ratio.png")

    print(
        json.dumps(
            {
                "metrics_path": str(metrics_path),
                "output_dir": str(output_dir),
                "plots": plot_names,
                "total_epochs": len(rows),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
