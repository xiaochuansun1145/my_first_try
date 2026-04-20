"""Plot stage-4 training metrics from metrics.jsonl.

Generates:
  - combined_loss.png         – total loss + feature/det/recon breakdown
  - feature_loss.png          – per-level feature compression loss
  - reconstruction.png        – recon L1/MSE/SSIM/edge + PSNR/SSIM curves
  - mask_activity.png         – common / individual active ratios
  - learning_rate.png         – lr schedule with phase boundaries
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
    p = argparse.ArgumentParser(description="Plot stage-4 metrics.")
    p.add_argument(
        "--metrics",
        default=str(REPO_ROOT / "outputs" / "mdvsc_stage4" / "metrics.jsonl"),
        help="Path to metrics.jsonl.",
    )
    p.add_argument("--output-dir", help="Plot output directory (default: <metrics_dir>/plots).")
    return p.parse_args()


def _load(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"Empty: {path}")
    return rows


def _series(rows, split, key):
    out = []
    for r in rows:
        d = r.get(split)
        out.append(None if d is None else d.get(key))
    return out


def _plot_tv(ax, rows, epochs, key, title, ylabel):
    tv = _series(rows, "train", key)
    vv = _series(rows, "val", key)
    ax.plot(epochs, tv, marker="o", markersize=3, label="train")
    if any(v is not None for v in vv):
        ax.plot(epochs, vv, marker="s", markersize=3, label="val")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()


def _add_phase_boundaries(ax, rows):
    """Add vertical lines at phase transitions."""
    prev_phase = None
    for r in rows:
        phase = r.get("phase")
        if phase != prev_phase and prev_phase is not None:
            ax.axvline(x=r["epoch"] - 0.5, color="red", linestyle="--", alpha=0.5, linewidth=1)
        prev_phase = phase


def main() -> None:
    args = parse_args()
    mpath = Path(args.metrics)
    odir = Path(args.output_dir) if args.output_dir else mpath.parent / "plots"
    odir.mkdir(parents=True, exist_ok=True)

    rows = _load(mpath)
    epochs = [r["epoch"] for r in rows]
    lrs = [r["lr"] for r in rows]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- 1) Combined loss overview ----
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    _plot_tv(axes[0], rows, epochs, "total_loss", "Total Loss", "loss")
    _plot_tv(axes[1], rows, epochs, "feature_loss_loss", "Feature Loss", "loss")
    _plot_tv(axes[2], rows, epochs, "det_recovery_loss", "Det Recovery Loss", "loss")
    _plot_tv(axes[3], rows, epochs, "recon_mse_loss", "Recon MSE Loss", "loss")
    for ax in axes:
        _add_phase_boundaries(ax, rows)
    fig.tight_layout()
    fig.savefig(odir / "combined_loss.png", dpi=200)
    plt.close(fig)

    # ---- 2) Reconstruction quality ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    _plot_tv(axes[0, 0], rows, epochs, "recon_l1_loss", "Recon L1", "loss")
    _plot_tv(axes[0, 1], rows, epochs, "recon_mse_loss", "Recon MSE", "loss")
    _plot_tv(axes[0, 2], rows, epochs, "recon_ssim_loss", "Recon SSIM Loss", "loss")
    _plot_tv(axes[1, 0], rows, epochs, "recon_edge_loss", "Recon Edge", "loss")
    _plot_tv(axes[1, 1], rows, epochs, "recon_psnr", "PSNR (dB)", "dB")
    _plot_tv(axes[1, 2], rows, epochs, "recon_ssim", "SSIM", "SSIM")
    for row in axes:
        for ax in row:
            _add_phase_boundaries(ax, rows)
    fig.tight_layout()
    fig.savefig(odir / "reconstruction.png", dpi=200)
    plt.close(fig)

    # ---- 3) Mask activity ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    _plot_tv(axes[0], rows, epochs, "common_active_ratio", "Common Active Ratio", "ratio")
    _plot_tv(axes[1], rows, epochs, "individual_active_ratio", "Individual Active Ratio", "ratio")
    for ax in axes:
        _add_phase_boundaries(ax, rows)
    fig.tight_layout()
    fig.savefig(odir / "mask_activity.png", dpi=200)
    plt.close(fig)

    # ---- 4) Learning rate with phase boundaries ----
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, lrs, marker="o", markersize=3)
    ax.set_title("Learning Rate Schedule")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("lr")
    ax.grid(True, alpha=0.3)
    _add_phase_boundaries(ax, rows)
    # Add phase labels
    prev_phase = None
    for r in rows:
        phase = r.get("phase")
        if phase != prev_phase:
            ax.annotate(phase, xy=(r["epoch"], r["lr"]),
                       fontsize=9, color="red", fontweight="bold")
        prev_phase = phase
    fig.tight_layout()
    fig.savefig(odir / "learning_rate.png", dpi=200)
    plt.close(fig)

    # ---- 5) Detail transmission ratio ----
    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_tv(ax, rows, epochs, "detail_tx_ratio", "Detail Transmission Ratio", "ratio")
    _add_phase_boundaries(ax, rows)
    fig.tight_layout()
    fig.savefig(odir / "detail_tx_ratio.png", dpi=200)
    plt.close(fig)

    plots = [
        "combined_loss.png",
        "reconstruction.png",
        "mask_activity.png",
        "learning_rate.png",
        "detail_tx_ratio.png",
    ]
    print(json.dumps({
        "metrics_path": str(mpath),
        "output_dir": str(odir),
        "plots": plots,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
