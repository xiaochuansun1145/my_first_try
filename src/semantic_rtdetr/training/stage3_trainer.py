"""Stage-3 trainer: feature-map reconstruction only, using MDVSC v2.

Pipeline:
    raw backbone features → frozen SharedEncoder (from stage-2) → shared 256
    → MDVSC v2 (trainable) → restored 256

Loss = weighted sum of per-level smooth_l1 (or mse) between MDVSC-restored
features and the frozen SharedEncoder output (teacher shared features).
Only MDVSC v2 parameters are trained; RT-DETR and SharedEncoder are frozen.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, OneCycleLR, SequentialLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.semantic_rtdetr.contracts import EncoderFeatureBundle
from src.semantic_rtdetr.detector.rtdetr_baseline import RTDetrBaseline
from src.semantic_rtdetr.semantic_comm.mdvsc_v2 import MDVSCV2Output, ProjectMDVSCV2
from src.semantic_rtdetr.semantic_comm.stage2_model import SharedEncoder
from src.semantic_rtdetr.training.stage3_config import MDVSCStage3TrainConfig
from src.semantic_rtdetr.training.stage1_data import build_train_val_datasets


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dataset_summary(dataset) -> dict[str, Any]:
    base = dataset.dataset if isinstance(dataset, Subset) else dataset
    if hasattr(base, "summary"):
        s = dict(base.summary())
        s["visible_samples"] = len(dataset)
        if isinstance(dataset, Subset):
            s["subset_type"] = "random_split"
        return s
    return {"visible_samples": len(dataset)}


def _bundle_to_sequences(bundle: EncoderFeatureBundle, batch_size: int, time_steps: int) -> list[torch.Tensor]:
    return [fm.view(batch_size, time_steps, *fm.shape[1:]) for fm in bundle.feature_maps]


def _resolve_amp_dtype(name: str) -> torch.dtype:
    n = name.lower()
    if n == "float16":
        return torch.float16
    if n == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported AMP dtype: {name}")


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy()


def _ensure_finite(name: str, t: torch.Tensor) -> None:
    if torch.isfinite(t).all():
        return
    d = t.detach().float()
    m = torch.isfinite(d)
    fv = d[m]
    lo = float(fv.min().item()) if fv.numel() else float("nan")
    hi = float(fv.max().item()) if fv.numel() else float("nan")
    raise ValueError(f"Non-finite in {name}: finite={int(m.sum())}/{d.numel()}, min={lo}, max={hi}")


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Stage3Trainer:
    def __init__(self, config: MDVSCStage3TrainConfig):
        self.config = config
        self.output_dir = Path(config.output.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _set_seed(config.optimization.seed)

        # Fixed input size → enable cuDNN autotuner & suppress plan-failure warnings
        torch.backends.cudnn.benchmark = True

        # ---- Teacher (frozen) ----
        print("[stage3] Loading RT-DETR teacher", flush=True)
        self.baseline = RTDetrBaseline(
            config.detector.hf_name,
            device=config.detector.device,
            local_path=config.detector.local_path,
            cache_dir=config.detector.cache_dir,
        )
        for p in self.baseline.model.parameters():
            p.requires_grad = False
        self.baseline.model.eval()

        # ---- SharedEncoder (frozen, loaded from stage-2 checkpoint) ----
        self.shared_encoder = SharedEncoder(
            backbone_channels=config.mdvsc.backbone_channels,
            shared_channels=config.mdvsc.feature_channels[0],
        ).to(self.baseline.device)
        self._load_shared_encoder()
        for p in self.shared_encoder.parameters():
            p.requires_grad = False
        self.shared_encoder.eval()

        # ---- MDVSC v2 model (trainable) ----
        self.model = ProjectMDVSCV2(
            feature_channels=config.mdvsc.feature_channels,
            latent_dims=config.mdvsc.latent_dims,
            common_keep_ratios=config.mdvsc.common_keep_ratios,
            individual_keep_ratios=config.mdvsc.individual_keep_ratios,
            block_sizes=config.mdvsc.block_sizes,
            spatial_strides=config.mdvsc.spatial_strides,
            apply_cross_level_fusion=config.mdvsc.apply_cross_level_fusion,
        ).to(self.baseline.device)
        self._load_init()
        self.param_counts = self._count_params()

        # ---- AMP ----
        self.amp_dtype = _resolve_amp_dtype(config.optimization.amp_dtype)
        self.amp_on = bool(config.optimization.use_amp and self.baseline.device.type == "cuda")
        self.scaler_on = bool(self.amp_on and self.amp_dtype == torch.float16)
        self.grad_scaler = torch.amp.GradScaler("cuda", enabled=self.scaler_on)
        if config.optimization.use_amp and not self.amp_on:
            print("[stage3] AMP requested but no CUDA – falling back to FP32", flush=True)
        elif self.amp_on:
            print(f"[stage3] AMP enabled: dtype={config.optimization.amp_dtype}, scaler={'on' if self.scaler_on else 'off'}", flush=True)

        # ---- Data ----
        print("[stage3] Building train/val datasets", flush=True)
        # Reuse stage1_data with a compatible config view
        data_cfg = _to_stage1_data_config(config.data)
        train_ds, val_ds = build_train_val_datasets(data_cfg, seed=config.optimization.seed)
        self.dataset_info = {
            "dataset_name": config.data.dataset_name,
            "train_num_samples": len(train_ds),
            "val_num_samples": len(val_ds) if val_ds else 0,
            "train_source_path": config.data.train_source_path,
            "val_source_path": config.data.val_source_path,
            "train_dataset": _dataset_summary(train_ds),
            "val_dataset": _dataset_summary(val_ds) if val_ds else None,
        }
        print(f"[stage3] train={self.dataset_info['train_num_samples']}, val={self.dataset_info['val_num_samples']}", flush=True)
        pin = self.baseline.device.type == "cuda"
        self.train_loader = DataLoader(train_ds, batch_size=config.optimization.batch_size, shuffle=True, num_workers=config.optimization.num_workers, pin_memory=pin)
        self.val_loader = None
        if val_ds is not None:
            self.val_loader = DataLoader(val_ds, batch_size=config.optimization.batch_size, shuffle=False, num_workers=config.optimization.num_workers, pin_memory=pin)
        print("[stage3] DataLoader ready", flush=True)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        metrics_path = self.output_dir / "metrics.jsonl"
        (self.output_dir / "resolved_config.json").write_text(json.dumps(self.config.to_dict(), indent=2), encoding="utf-8")
        (self.output_dir / "dataset_info.json").write_text(json.dumps(self.dataset_info, indent=2), encoding="utf-8")

        steps_per_epoch = self._steps_per_epoch(self.train_loader)
        optimizer = self._build_optimizer()
        scheduler, sched_per_batch = self._build_scheduler(optimizer, steps_per_epoch)

        best_val: float | None = None
        last_summary: dict[str, Any] | None = None

        for epoch in range(1, self.config.optimization.epochs + 1):
            print(f"[stage3] epoch {epoch}/{self.config.optimization.epochs}", flush=True)

            train_m = self._run_epoch(self.train_loader, True, epoch, optimizer, scheduler if sched_per_batch else None)
            val_m = None
            if self.val_loader is not None:
                val_m = self._run_epoch(self.val_loader, False, epoch)
            if scheduler is not None and not sched_per_batch:
                scheduler.step()

            summary = {"epoch": epoch, "train": train_m, "val": val_m, "lr": float(optimizer.param_groups[0]["lr"])}
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(summary, ensure_ascii=False) + "\n")

            cur = val_m["feature_loss"] if val_m else train_m["feature_loss"]
            if best_val is None or cur < best_val:
                best_val = cur
                self._save_ckpt("best.pt", epoch, summary, optimizer, scheduler)
            if epoch % self.config.optimization.save_every_epochs == 0:
                self._save_ckpt(f"epoch_{epoch:03d}.pt", epoch, summary, optimizer, scheduler)
            self._save_ckpt("latest.pt", epoch, summary, optimizer, scheduler)
            last_summary = summary

        final = {
            "output_dir": str(self.output_dir),
            "best_val_feature_loss": best_val,
            "epochs": self.config.optimization.epochs,
            "parameter_counts": self.param_counts,
            "dataset": self.dataset_info,
            "last_epoch": last_summary,
        }
        (self.output_dir / "final_summary.json").write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
        return final

    # ------------------------------------------------------------------
    # One epoch
    # ------------------------------------------------------------------

    def _run_epoch(
        self,
        loader: DataLoader,
        training: bool,
        epoch: int,
        optimizer=None,
        step_scheduler=None,
    ) -> dict[str, float]:
        self.model.train(training)
        sums: defaultdict[str, float] = defaultdict(float)
        max_steps = self.config.optimization.max_steps_per_epoch
        steps = 0
        desc = f"epoch {epoch} {'train' if training else 'val'}"
        progress = tqdm(loader, desc=desc, leave=False)

        for bi, frames in enumerate(progress, 1):
            if max_steps is not None and bi > max_steps:
                break

            frames = frames.to(self.baseline.device, non_blocking=True)
            B, T, _, H, W = frames.shape
            flat = frames.view(B * T, *frames.shape[2:])

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=self.baseline.device.type, dtype=self.amp_dtype, enabled=self.amp_on):
                with torch.no_grad():
                    det_in = self.baseline.prepare_frame_tensor_batch(flat)
                    raw_features, _proj_bundle = self.baseline.extract_backbone_and_projected_features(det_in)
                    # Run frozen SharedEncoder to get target sequences
                    backbone_seqs = [feat.view(B, T, *feat.shape[1:]) for feat in raw_features]
                    shared_seqs = self._encode_with_shared_encoder(backbone_seqs)

                target_seqs = shared_seqs  # teacher: frozen SharedEncoder output
                out: MDVSCV2Output = self.model(
                    target_seqs,
                    apply_masks=self.config.mdvsc.apply_masks,
                    channel_mode=self.config.mdvsc.channel_mode,
                    snr_db=self.config.mdvsc.snr_db,
                )
                loss_dict = self._compute_loss(target_seqs, out)

            _ensure_finite("feature_loss", loss_dict["feature_loss"])

            if training:
                total = loss_dict["feature_loss"]
                if self.scaler_on:
                    self.grad_scaler.scale(total).backward()
                    self.grad_scaler.unscale_(optimizer)
                else:
                    total.backward()
                clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], self.config.optimization.grad_clip_norm)
                if self.scaler_on:
                    self.grad_scaler.step(optimizer)
                    self.grad_scaler.update()
                else:
                    optimizer.step()
                if step_scheduler is not None:
                    step_scheduler.step()

            det = {k: float(v.detach().item()) for k, v in loss_dict.items()}
            for k, v in det.items():
                sums[k] += v
            sums["common_active_ratio"] += float(np.mean([s.common_active_ratio for s in out.level_stats]))
            sums["individual_active_ratio"] += float(np.mean([s.individual_active_ratio for s in out.level_stats]))
            steps += 1

            if bi == 1 and self._should_vis(epoch):
                self._save_vis(epoch, "train" if training else "val", target_seqs, out)

            if training and bi % self.config.optimization.log_every == 0:
                progress.set_postfix(feat=f"{det['feature_loss']:.4f}")

        if steps == 0:
            raise ValueError("No batches processed")

        per_level_keys = [k for k in sums if k.startswith("level_")]
        result = {
            "feature_loss": sums["feature_loss"] / steps,
            "common_active_ratio": sums["common_active_ratio"] / steps,
            "individual_active_ratio": sums["individual_active_ratio"] / steps,
        }
        for k in per_level_keys:
            result[k] = sums[k] / steps
        return result

    # ------------------------------------------------------------------
    # Loss (feature reconstruction only)
    # ------------------------------------------------------------------

    def _compute_loss(
        self,
        target_seqs: list[torch.Tensor],
        out: MDVSCV2Output,
    ) -> dict[str, torch.Tensor]:
        dev = target_seqs[0].device
        total = torch.zeros((), device=dev, dtype=torch.float32)
        loss_fn = F.smooth_l1_loss if self.config.loss.feature_loss_type == "smooth_l1" else F.mse_loss
        per_level: dict[str, torch.Tensor] = {}

        for idx, (w, restored, target) in enumerate(
            zip(self.config.loss.level_loss_weights, out.restored_sequences, target_seqs)
        ):
            lv = loss_fn(restored.float(), target.float())
            per_level[f"level_{idx}_loss"] = lv
            total = total + float(w) * lv

        result = {"feature_loss": total}
        result.update(per_level)
        return result

    # ------------------------------------------------------------------
    # Optimizer / scheduler
    # ------------------------------------------------------------------

    def _build_optimizer(self) -> torch.optim.Optimizer:
        cfg = self.config.optimization
        params = [p for p in self.model.parameters() if p.requires_grad]
        betas = (cfg.adam_beta1, cfg.adam_beta2)
        if cfg.optimizer.lower() == "adamw":
            return torch.optim.AdamW(params, lr=cfg.lr, betas=betas, weight_decay=cfg.weight_decay)
        if cfg.optimizer.lower() == "adam":
            return torch.optim.Adam(params, lr=cfg.lr, betas=betas, weight_decay=cfg.weight_decay)
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")

    def _build_scheduler(self, optimizer, steps_per_epoch):
        cfg = self.config.optimization
        stype = cfg.scheduler.lower()
        if stype == "constant":
            return None, False
        if stype == "onecycle":
            return OneCycleLR(
                optimizer,
                max_lr=cfg.lr,
                epochs=max(cfg.epochs, 1),
                steps_per_epoch=max(steps_per_epoch, 1),
                pct_start=cfg.onecycle_pct_start,
                div_factor=cfg.onecycle_div_factor,
                final_div_factor=cfg.onecycle_final_div_factor,
                anneal_strategy="cos",
            ), True
        if stype == "cosine":
            total = max(cfg.epochs, 1)
            warm = max(0, min(cfg.warmup_epochs, total - 1))
            eta_min = cfg.lr * cfg.min_lr_ratio
            if warm == 0:
                return CosineAnnealingLR(optimizer, T_max=total, eta_min=eta_min), False
            warmup = LinearLR(optimizer, start_factor=cfg.warmup_start_factor, end_factor=1.0, total_iters=warm)
            cosine = CosineAnnealingLR(optimizer, T_max=max(total - warm, 1), eta_min=eta_min)
            return SequentialLR(optimizer, [warmup, cosine], milestones=[warm]), False
        raise ValueError(f"Unsupported scheduler: {cfg.scheduler}")

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _save_ckpt(self, name, epoch, summary, optimizer, scheduler):
        torch.save({
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "grad_scaler_state": self.grad_scaler.state_dict() if self.scaler_on else None,
            "config": self.config.to_dict(),
            "summary": summary,
        }, self.output_dir / name)

    def _load_shared_encoder(self) -> None:
        """Load SharedEncoder weights from a stage-2 checkpoint."""
        ckpt_path = self.config.initialization.stage2_checkpoint
        if not ckpt_path:
            print("[stage3] WARNING: No stage2_checkpoint specified – SharedEncoder uses random init", flush=True)
            return
        raw = torch.load(ckpt_path, map_location="cpu")
        state = raw.get("model_state", raw) if isinstance(raw, dict) else raw
        # Extract only shared_encoder.* keys
        prefix = "shared_encoder."
        se_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        if not se_state:
            raise ValueError(f"No shared_encoder parameters found in {ckpt_path}")
        self.shared_encoder.load_state_dict(se_state, strict=True)
        print(f"[stage3] Loaded SharedEncoder ({len(se_state)} params) from {ckpt_path}", flush=True)

    def _encode_with_shared_encoder(self, backbone_seqs: list[torch.Tensor]) -> list[torch.Tensor]:
        """Run frozen SharedEncoder on per-level backbone sequences → shared 256 sequences."""
        shared_seqs: list[torch.Tensor] = []
        for proj, seq in zip(self.shared_encoder.projections, backbone_seqs):
            B, T, C, H, W = seq.shape
            flat = seq.reshape(B * T, C, H, W)
            shared = proj(flat)
            shared_seqs.append(shared.reshape(B, T, self.shared_encoder.shared_channels, H, W))
        return shared_seqs

    def _load_init(self):
        ckpt_path = self.config.initialization.checkpoint
        if not ckpt_path:
            return
        raw = torch.load(ckpt_path, map_location="cpu")
        state = raw.get("model_state", raw) if isinstance(raw, dict) else raw
        self.model.load_state_dict(state, strict=self.config.initialization.strict)
        print(f"[stage3] Loaded MDVSC v2 checkpoint: {ckpt_path}", flush=True)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def _steps_per_epoch(self, loader):
        if self.config.optimization.max_steps_per_epoch is None:
            return len(loader)
        return min(len(loader), self.config.optimization.max_steps_per_epoch)

    def _count_params(self) -> dict[str, int]:
        counts = {
            "mdvsc_v2_total": sum(p.numel() for p in self.model.parameters()),
            "mdvsc_v2_trainable": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "level_modules": sum(p.numel() for p in self.model.level_modules.parameters()),
            "cross_level_fusion": sum(p.numel() for p in self.model.cross_level_fusion.parameters()) if self.model.cross_level_fusion else 0,
            "shared_encoder_frozen": sum(p.numel() for p in self.shared_encoder.parameters()),
        }
        print(f"[stage3] Parameters: MDVSC v2 trainable={counts['mdvsc_v2_trainable']:,}, SharedEncoder frozen={counts['shared_encoder_frozen']:,}", flush=True)
        return counts

    def _should_vis(self, epoch):
        if not self.config.output.save_visualizations:
            return False
        every = max(self.config.output.visualization_every_epochs, 1)
        return epoch % every == 0

    def _save_vis(
        self,
        epoch: int,
        split: str,
        target_seqs: list[torch.Tensor],
        out: MDVSCV2Output,
    ) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        vis_dir = self.output_dir / "visualizations" / split
        vis_dir.mkdir(parents=True, exist_ok=True)

        num_levels = len(target_seqs)

        # --- 1) Feature maps: teacher / restored / abs-diff per level ---
        fig, axes = plt.subplots(num_levels, 3, figsize=(12, 4 * num_levels))
        if num_levels == 1:
            axes = np.array([axes])
        for lv in range(num_levels):
            teacher = _to_numpy(target_seqs[lv][0, 0].mean(dim=0))
            student = _to_numpy(out.restored_sequences[lv][0, 0].mean(dim=0))
            diff = np.abs(student - teacher)
            axes[lv, 0].imshow(teacher, cmap="viridis")
            axes[lv, 0].set_title(f"lv{lv} teacher")
            axes[lv, 1].imshow(student, cmap="viridis")
            axes[lv, 1].set_title(f"lv{lv} restored")
            axes[lv, 2].imshow(diff, cmap="magma")
            axes[lv, 2].set_title(f"lv{lv} |diff|")
            for c in range(3):
                axes[lv, c].axis("off")
        fig.tight_layout()
        fig.savefig(vis_dir / f"epoch_{epoch:03d}_feature_maps.png", dpi=180)
        plt.close(fig)

        # --- 2) Entropy maps & masks per level ---
        fig, axes = plt.subplots(num_levels, 4, figsize=(16, 3.5 * num_levels))
        if num_levels == 1:
            axes = np.array([axes])
        for lv in range(num_levels):
            c_mask = _to_numpy(out.common_masks[lv][0].mean(dim=0))
            i_mask = _to_numpy(out.individual_masks[lv][0, 0].mean(dim=0))
            c_ent = _to_numpy(out.common_entropy_maps[lv][0].mean(dim=0))
            i_ent = _to_numpy(out.individual_entropy_maps[lv][0, 0].mean(dim=0))
            axes[lv, 0].imshow(c_ent, cmap="inferno")
            axes[lv, 0].set_title(f"lv{lv} common entropy")
            axes[lv, 1].imshow(c_mask, cmap="gray", vmin=0, vmax=1)
            axes[lv, 1].set_title(f"lv{lv} common mask")
            axes[lv, 2].imshow(i_ent, cmap="inferno")
            axes[lv, 2].set_title(f"lv{lv} indiv entropy t=0")
            axes[lv, 3].imshow(i_mask, cmap="gray", vmin=0, vmax=1)
            axes[lv, 3].set_title(f"lv{lv} indiv mask t=0")
            for c in range(4):
                axes[lv, c].axis("off")
        fig.tight_layout()
        fig.savefig(vis_dir / f"epoch_{epoch:03d}_entropy_masks.png", dpi=180)
        plt.close(fig)

        # --- 3) Per-level feature cosine-similarity map ---
        fig, axes = plt.subplots(1, num_levels, figsize=(5 * num_levels, 4))
        if num_levels == 1:
            axes = [axes]
        for lv in range(num_levels):
            t = target_seqs[lv][0, 0]           # [C,H,W]
            r = out.restored_sequences[lv][0, 0]
            cos = F.cosine_similarity(t.unsqueeze(0), r.unsqueeze(0), dim=1)  # [1,H,W]
            axes[lv].imshow(_to_numpy(cos[0]), cmap="RdYlGn", vmin=0, vmax=1)
            axes[lv].set_title(f"lv{lv} cosine sim")
            axes[lv].axis("off")
        fig.tight_layout()
        fig.savefig(vis_dir / f"epoch_{epoch:03d}_cosine_sim.png", dpi=180)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Adapter: Stage3DataConfig → Stage1DataConfig for dataset reuse
# ---------------------------------------------------------------------------

def _to_stage1_data_config(cfg):
    from src.semantic_rtdetr.training.stage1_config import Stage1DataConfig
    return Stage1DataConfig(
        dataset_name=cfg.dataset_name,
        train_source_path=cfg.train_source_path,
        val_source_path=cfg.val_source_path,
        recursive=cfg.recursive,
        index_cache_dir=cfg.index_cache_dir,
        subset_seed=cfg.subset_seed,
        source_fraction=cfg.source_fraction,
        sample_fraction=cfg.sample_fraction,
        gop_size=cfg.gop_size,
        frame_height=cfg.frame_height,
        frame_width=cfg.frame_width,
        frame_stride=cfg.frame_stride,
        gop_stride=cfg.gop_stride,
        max_frames_per_source=cfg.max_frames_per_source,
        max_sources=cfg.max_sources,
        max_samples=cfg.max_samples,
        train_val_split=cfg.train_val_split,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_stage3_training(config: MDVSCStage3TrainConfig) -> dict[str, Any]:
    trainer = Stage3Trainer(config)
    return trainer.run()
