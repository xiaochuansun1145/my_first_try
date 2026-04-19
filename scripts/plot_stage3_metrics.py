"""Plot stage-3 training metrics from metrics.jsonl.

Generates:
  - feature_loss.png          – total & per-level feature loss curves
  - mask_activity.png         – common / individual active ratios
  - learning_rate.png         – lr schedule
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
    p = argparse.ArgumentParser(description="Plot stage-3 metrics.")
    p.add_argument(
        "--metrics",
        default=str(REPO_ROOT / "outputs" / "mdvsc_stage3" / "metrics.jsonl"),
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

    # ---- Detect per-level keys ----
    sample_train = rows[0].get("train") or {}
    level_keys = sorted(k for k in sample_train if k.startswith("level_") and k.endswith("_loss"))

    # ---- Feature loss overview ----
    ncols = 1 + len(level_keys)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4.5))
    if ncols == 1:
        axes = [axes]
    _plot_tv(axes[0], rows, epochs, "feature_loss", "Total Feature Loss", "loss")
    for i, lk in enumerate(level_keys):
        _plot_tv(axes[i + 1], rows, epochs, lk, lk.replace("_", " ").title(), "loss")
    fig.tight_layout()
    fig.savefig(odir / "feature_loss.png", dpi=200)
    plt.close(fig)

    # ---- Mask activity ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    _plot_tv(axes[0], rows, epochs, "common_active_ratio", "Common Active Ratio", "ratio")
    _plot_tv(axes[1], rows, epochs, "individual_active_ratio", "Individual Active Ratio", "ratio")
    fig.tight_layout()
    fig.savefig(odir / "mask_activity.png", dpi=200)
    plt.close(fig)

    # ---- Learning rate ----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, lrs, marker="o", markersize=3)
    ax.set_title("Learning Rate")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("lr")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(odir / "learning_rate.png", dpi=200)
    plt.close(fig)

    print(json.dumps({
        "metrics_path": str(mpath),
        "output_dir": str(odir),
        "plots": ["feature_loss.png", "mask_activity.png", "learning_rate.png"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
