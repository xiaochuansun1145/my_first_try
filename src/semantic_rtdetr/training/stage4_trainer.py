"""Stage-4 trainer: end-to-end joint training of SharedEncoder + MDVSC v2 + dual decoder.

Combines Stage 2.1 (det recovery + reconstruction) and Stage 3 (feature compression)
into a single training loop with progressive unfreezing across 3 phases:

Phase 1: Freeze SharedEncoder, train MDVSC v2 + DetRecovery + Detail + Reconstruction
Phase 2: Unfreeze SharedEncoder, all modules trainable at lower LR
Phase 3: Fine-tune all modules at minimal LR
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
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.semantic_rtdetr.detector.rtdetr_baseline import RTDetrBaseline
from src.semantic_rtdetr.semantic_comm.stage4_model import Stage4MDVSC, Stage4Output
from src.semantic_rtdetr.training.stage4_config import (
    MDVSCStage4TrainConfig,
    Stage4PhaseConfig,
)
from src.semantic_rtdetr.training.stage1_data import build_train_val_datasets
from src.semantic_rtdetr.training.stage2_trainer import (
    _dataset_summary,
    _ensure_finite_tensor,
    _gradient_edge_loss,
    _psnr_from_mse,
    _resolve_amp_dtype,
    _set_seed,
    _ssim_loss,
    _to_numpy_float_array,
)


class Stage4Trainer:
    def __init__(self, config: MDVSCStage4TrainConfig):
        self.config = config
        self.output_dir = Path(config.output.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _set_seed(config.optimization.seed)

        torch.backends.cudnn.benchmark = True

        # ---- Teacher (frozen) ----
        print("[stage4] Loading RT-DETR teacher", flush=True)
        self.baseline = RTDetrBaseline(
            config.detector.hf_name,
            device=config.detector.device,
            local_path=config.detector.local_path,
            cache_dir=config.detector.cache_dir,
        )
        for p in self.baseline.model.parameters():
            p.requires_grad = False
        self.baseline.model.eval()

        # ---- Stage 4 model ----
        print("[stage4] Building Stage4MDVSC model", flush=True)
        self.model = Stage4MDVSC(
            backbone_channels=config.mdvsc.backbone_channels,
            shared_channels=config.mdvsc.shared_channels,
            latent_dims=config.mdvsc.latent_dims,
            common_keep_ratios=config.mdvsc.common_keep_ratios,
            individual_keep_ratios=config.mdvsc.individual_keep_ratios,
            block_sizes=config.mdvsc.block_sizes,
            spatial_strides=config.mdvsc.spatial_strides,
            apply_cross_level_fusion=config.mdvsc.apply_cross_level_fusion,
            reconstruction_hidden_channels=config.mdvsc.reconstruction_hidden_channels,
            reconstruction_detail_channels=config.mdvsc.reconstruction_detail_channels,
            reconstruction_use_checkpoint=config.mdvsc.reconstruction_use_checkpoint,
            stage1_channels=config.mdvsc.stage1_channels,
            detail_latent_channels=config.mdvsc.detail_latent_channels,
            detail_spatial_size=config.mdvsc.detail_spatial_size,
        ).to(self.baseline.device)
        self._load_initialization()
        self.param_counts = self._summarize_params()

        # ---- AMP ----
        self.amp_dtype = _resolve_amp_dtype(config.optimization.amp_dtype)
        self.amp_on = bool(config.optimization.use_amp and self.baseline.device.type == "cuda")
        self.scaler_on = bool(self.amp_on and self.amp_dtype == torch.float16)
        self.grad_scaler = torch.amp.GradScaler("cuda", enabled=self.scaler_on)
        if config.optimization.use_amp and not self.amp_on:
            print("[stage4] AMP requested but no CUDA – falling back to FP32", flush=True)
        elif self.amp_on:
            print(f"[stage4] AMP enabled: dtype={config.optimization.amp_dtype}, scaler={'on' if self.scaler_on else 'off'}", flush=True)

        # ---- Data ----
        print("[stage4] Building train/val datasets", flush=True)
        train_ds, val_ds = build_train_val_datasets(config.data, seed=config.optimization.seed)
        self.dataset_info = {
            "dataset_name": config.data.dataset_name,
            "train_num_samples": len(train_ds),
            "val_num_samples": len(val_ds) if val_ds else 0,
            "train_source_path": config.data.train_source_path,
            "val_source_path": config.data.val_source_path,
            "train_dataset": _dataset_summary(train_ds),
            "val_dataset": _dataset_summary(val_ds) if val_ds else None,
        }
        print(f"[stage4] train={self.dataset_info['train_num_samples']}, val={self.dataset_info['val_num_samples']}", flush=True)
        pin = self.baseline.device.type == "cuda"
        self.train_loader = DataLoader(train_ds, batch_size=config.optimization.batch_size, shuffle=True, num_workers=config.optimization.num_workers, pin_memory=pin)
        self.val_loader = None
        if val_ds is not None:
            self.val_loader = DataLoader(val_ds, batch_size=config.optimization.batch_size, shuffle=False, num_workers=config.optimization.num_workers, pin_memory=pin)
        print("[stage4] DataLoader ready", flush=True)

    # ------------------------------------------------------------------
    # Main loop: 3-phase progressive training
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        metrics_path = self.output_dir / "metrics.jsonl"
        (self.output_dir / "resolved_config.json").write_text(
            json.dumps(self.config.to_dict(), indent=2), encoding="utf-8"
        )
        (self.output_dir / "dataset_info.json").write_text(
            json.dumps(self.dataset_info, indent=2), encoding="utf-8"
        )

        phases = [
            ("phase1", self.config.optimization.phase1),
            ("phase2", self.config.optimization.phase2),
            ("phase3", self.config.optimization.phase3),
        ]

        best_val: float | None = None
        last_summary: dict[str, Any] | None = None
        global_epoch = 0

        for phase_name, phase_cfg in phases:
            if phase_cfg.epochs <= 0:
                print(f"[stage4] Skipping {phase_name} (epochs=0)", flush=True)
                continue
            print(f"[stage4] === {phase_name}: {phase_cfg.epochs} epochs, lr={phase_cfg.lr} ===", flush=True)

            # Apply freeze strategy
            self._apply_freeze(phase_cfg)
            trainable = [p for p in self.model.parameters() if p.requires_grad]
            n_trainable = sum(p.numel() for p in trainable)
            print(f"[stage4] {phase_name}: {n_trainable:,} trainable parameters", flush=True)

            if not trainable:
                print(f"[stage4] WARNING: No trainable parameters in {phase_name}, skipping", flush=True)
                continue

            # Build optimizer + scheduler for this phase
            optimizer = self._build_optimizer(trainable, phase_cfg.lr)
            steps_per_epoch = self._steps_per_epoch(self.train_loader)
            scheduler = OneCycleLR(
                optimizer,
                max_lr=phase_cfg.lr,
                epochs=max(phase_cfg.epochs, 1),
                steps_per_epoch=max(steps_per_epoch, 1),
                pct_start=self.config.optimization.onecycle_pct_start,
                div_factor=self.config.optimization.onecycle_div_factor,
                final_div_factor=self.config.optimization.onecycle_final_div_factor,
                anneal_strategy="cos",
            )

            for local_epoch in range(1, phase_cfg.epochs + 1):
                global_epoch += 1
                print(f"[stage4] {phase_name} epoch {local_epoch}/{phase_cfg.epochs} (global {global_epoch})", flush=True)

                train_m = self._run_epoch(self.train_loader, True, global_epoch, optimizer, scheduler)
                val_m = None
                if self.val_loader is not None:
                    val_m = self._run_epoch(self.val_loader, False, global_epoch)

                summary = {
                    "epoch": global_epoch,
                    "phase": phase_name,
                    "local_epoch": local_epoch,
                    "train": train_m,
                    "val": val_m,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                }
                with metrics_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(summary, ensure_ascii=False) + "\n")

                cur = val_m["total_loss"] if val_m else train_m["total_loss"]
                if best_val is None or cur < best_val:
                    best_val = cur
                    self._save_ckpt("best.pt", global_epoch, summary, optimizer, scheduler)
                if global_epoch % self.config.optimization.save_every_epochs == 0:
                    self._save_ckpt(f"epoch_{global_epoch:03d}.pt", global_epoch, summary, optimizer, scheduler)
                self._save_ckpt("latest.pt", global_epoch, summary, optimizer, scheduler)
                last_summary = summary

        total_epochs = global_epoch
        final = {
            "output_dir": str(self.output_dir),
            "best_val_loss": best_val,
            "total_epochs": total_epochs,
            "parameter_counts": self.param_counts,
            "dataset": self.dataset_info,
            "last_epoch": last_summary,
        }
        (self.output_dir / "final_summary.json").write_text(
            json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8"
        )
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
        scheduler=None,
    ) -> dict[str, float]:
        self.model.train(training)
        sums: defaultdict[str, float] = defaultdict(float)
        max_steps = self.config.optimization.max_steps_per_epoch
        steps = 0
        desc = f"epoch {epoch} {'train' if training else 'val'}"
        progress = tqdm(loader, desc=desc, leave=False)
        num_levels = len(self.config.mdvsc.backbone_channels)

        for bi, frames in enumerate(progress, 1):
            if max_steps is not None and bi > max_steps:
                break

            frames = frames.to(self.baseline.device, non_blocking=True)
            B, T, _, H, W = frames.shape
            flat = frames.view(B * T, *frames.shape[2:])

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=self.baseline.device.type, dtype=self.amp_dtype, enabled=self.amp_on):
                # Extract teacher features (frozen)
                with torch.no_grad():
                    det_in = self.baseline.prepare_frame_tensor_batch(flat)
                    stage1_flat, raw_features, teacher_bundle = self.baseline.extract_all_backbone_features(det_in)

                # Reshape to sequences
                backbone_sequences = [feat.view(B, T, *feat.shape[1:]) for feat in raw_features]
                projected_sequences = [feat.view(B, T, *feat.shape[1:]) for feat in teacher_bundle.feature_maps]
                stage1_sequences = stage1_flat.view(B, T, *stage1_flat.shape[1:])

                # Forward: end-to-end
                out: Stage4Output = self.model(
                    backbone_sequences,
                    stage1_sequences,
                    output_size=(H, W),
                    apply_masks=self.config.mdvsc.apply_masks,
                    channel_mode=self.config.mdvsc.channel_mode,
                    snr_db=self.config.mdvsc.snr_db,
                )

                # Compute combined loss
                loss_dict = self._compute_losses(
                    frames=frames,
                    projected_sequences=projected_sequences,
                    shared_sequences=out.shared_sequences,
                    output=out,
                    num_levels=num_levels,
                )

            _ensure_finite_tensor("total", loss_dict["total"])

            if training:
                total = loss_dict["total"]
                if self.scaler_on:
                    self.grad_scaler.scale(total).backward()
                    self.grad_scaler.unscale_(optimizer)
                else:
                    total.backward()
                trainable = [p for p in self.model.parameters() if p.requires_grad]
                clip_grad_norm_(trainable, self.config.optimization.grad_clip_norm)
                if self.scaler_on:
                    self.grad_scaler.step(optimizer)
                    self.grad_scaler.update()
                else:
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            det = {k: float(v.detach().item()) for k, v in loss_dict.items()}
            det["detail_tx_ratio"] = out.detail_transmission_ratio
            mdvsc_stats = out.mdvsc_output.level_stats
            det["common_active_ratio"] = float(np.mean([s.common_active_ratio for s in mdvsc_stats]))
            det["individual_active_ratio"] = float(np.mean([s.individual_active_ratio for s in mdvsc_stats]))

            for k, v in det.items():
                sums[k] += v
            steps += 1

            if bi == 1 and self._should_vis(epoch):
                self._save_visualizations(
                    epoch=epoch,
                    split="train" if training else "val",
                    frames=frames,
                    projected_sequences=projected_sequences,
                    shared_sequences=out.shared_sequences,
                    output=out,
                )

            if training and bi % self.config.optimization.log_every == 0:
                progress.set_postfix(
                    total=f"{det['total']:.4f}",
                    feat=f"{det['feature_loss']:.4f}",
                    det=f"{det['det_recovery']:.4f}",
                    recon=f"{det['recon_mse']:.4f}",
                )

        if steps == 0:
            raise ValueError("No batches processed")

        result: dict[str, float] = {}
        for key in sums:
            if key in ("detail_tx_ratio", "common_active_ratio", "individual_active_ratio"):
                result[key] = sums[key] / steps
            else:
                result[f"{key}_loss" if key != "total" else "total_loss"] = sums[key] / steps

        # Derived quality metrics
        avg_recon_mse = sums["recon_mse"] / steps
        result["recon_psnr"] = _psnr_from_mse(avg_recon_mse)
        result["recon_ssim"] = 1.0 - sums["recon_ssim"] / steps
        return result

    # ------------------------------------------------------------------
    # Combined loss
    # ------------------------------------------------------------------

    def _compute_losses(
        self,
        frames: torch.Tensor,
        projected_sequences: list[torch.Tensor],
        shared_sequences: list[torch.Tensor],
        output: Stage4Output,
        num_levels: int,
    ) -> dict[str, torch.Tensor]:
        device = frames.device
        lcfg = self.config.loss

        # -- Feature compression loss (Stage 3 objective) --
        feature_loss = torch.zeros((), device=device, dtype=torch.float32)
        loss_fn = F.smooth_l1_loss if lcfg.feature_loss_type == "smooth_l1" else F.mse_loss
        for idx in range(num_levels):
            lv = loss_fn(
                output.mdvsc_output.restored_sequences[idx].float(),
                shared_sequences[idx].float(),
            )
            feature_loss = feature_loss + float(lcfg.level_loss_weights[idx]) * lv

        # -- Detection recovery loss (Stage 2.1 objective) --
        det_recovery = torch.zeros((), device=device, dtype=torch.float32)
        for idx in range(num_levels):
            level_mse = F.mse_loss(
                output.det_recovery_sequences[idx].float(),
                projected_sequences[idx].float(),
            )
            det_recovery = det_recovery + float(lcfg.level_recovery_weights[idx]) * level_mse

        # -- Reconstruction losses (Stage 2.1 objective) --
        frames_fp32 = frames.float()
        reconstructed = output.reconstructed_frames.float()
        recon_l1 = F.l1_loss(reconstructed, frames_fp32)
        recon_mse = F.mse_loss(reconstructed, frames_fp32)
        recon_ssim = _ssim_loss(reconstructed, frames_fp32, downsample_factor=lcfg.ssim_downsample_factor)
        recon_edge = _gradient_edge_loss(reconstructed, frames_fp32)

        # -- Total --
        total = (
            lcfg.feature_loss_weight * feature_loss
            + lcfg.det_recovery_weight * det_recovery
            + lcfg.recon_l1_weight * recon_l1
            + lcfg.recon_mse_weight * recon_mse
            + lcfg.recon_ssim_weight * recon_ssim
            + lcfg.recon_edge_weight * recon_edge
        )

        return {
            "total": total,
            "feature_loss": feature_loss,
            "det_recovery": det_recovery,
            "recon_l1": recon_l1,
            "recon_mse": recon_mse,
            "recon_ssim": recon_ssim,
            "recon_edge": recon_edge,
        }

    # ------------------------------------------------------------------
    # Freeze strategy
    # ------------------------------------------------------------------

    def _apply_freeze(self, phase: Stage4PhaseConfig) -> None:
        """Apply per-phase freeze settings."""
        def _set_requires_grad(module: torch.nn.Module, requires: bool) -> None:
            for p in module.parameters():
                p.requires_grad = requires

        _set_requires_grad(self.model.shared_encoder, not phase.freeze_shared_encoder)
        _set_requires_grad(self.model.mdvsc_v2, not phase.freeze_mdvsc_v2)
        _set_requires_grad(self.model.det_recovery_head, not phase.freeze_det_recovery)
        _set_requires_grad(self.model.detail_compressor, not phase.freeze_detail_bypass)
        _set_requires_grad(self.model.detail_decompressor, not phase.freeze_detail_bypass)
        _set_requires_grad(self.model.reconstruction_refinement_heads, not phase.freeze_reconstruction)
        _set_requires_grad(self.model.reconstruction_head, not phase.freeze_reconstruction)

    # ------------------------------------------------------------------
    # Optimizer / scheduler
    # ------------------------------------------------------------------

    def _build_optimizer(self, parameters: list[torch.nn.Parameter], lr: float) -> torch.optim.Optimizer:
        cfg = self.config.optimization
        betas = (cfg.adam_beta1, cfg.adam_beta2)
        if cfg.optimizer.lower() == "adamw":
            return torch.optim.AdamW(parameters, lr=lr, betas=betas, weight_decay=cfg.weight_decay)
        if cfg.optimizer.lower() == "adam":
            return torch.optim.Adam(parameters, lr=lr, betas=betas, weight_decay=cfg.weight_decay)
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")

    def _steps_per_epoch(self, loader: DataLoader) -> int:
        if self.config.optimization.max_steps_per_epoch is None:
            return len(loader)
        return min(len(loader), self.config.optimization.max_steps_per_epoch)

    # ------------------------------------------------------------------
    # Checkpoint / initialization
    # ------------------------------------------------------------------

    def _save_ckpt(self, name: str, epoch: int, summary: dict, optimizer, scheduler) -> None:
        torch.save({
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "grad_scaler_state": self.grad_scaler.state_dict() if self.scaler_on else None,
            "config": self.config.to_dict(),
            "summary": summary,
        }, self.output_dir / name)

    def _load_initialization(self) -> None:
        init = self.config.initialization

        # Option 1: full checkpoint — load directly
        if init.full_checkpoint:
            state = self._load_model_state(init.full_checkpoint)
            self.model.load_state_dict(state, strict=init.strict)
            print(f"[stage4] Loaded full checkpoint: {init.full_checkpoint}", flush=True)
            return

        loaded_keys: set[str] = set()

        # Option 2: assemble from Stage 2.1 + Stage 3
        if init.stage2_1_checkpoint:
            s21_state = self._load_model_state(init.stage2_1_checkpoint)
            self._load_stage2_1_weights(s21_state, loaded_keys)

        if init.stage3_checkpoint:
            s3_state = self._load_model_state(init.stage3_checkpoint)
            self._load_stage3_weights(s3_state, loaded_keys)

    def _load_stage2_1_weights(self, state: dict[str, torch.Tensor], loaded_keys: set[str]) -> None:
        """Load Stage 2.1 components: SharedEncoder, DetRecovery, Detail*, Reconstruction."""
        current = self.model.state_dict()
        compatible_prefixes = (
            "shared_encoder.",
            "det_recovery_head.",
            "reconstruction_refinement_heads.",
            "reconstruction_head.",
            "detail_compressor.",
            "detail_decompressor.",
        )
        mapped = {}
        for key, value in state.items():
            if any(key.startswith(p) for p in compatible_prefixes):
                if key in current and current[key].shape == value.shape:
                    mapped[key] = value
        if mapped:
            self.model.load_state_dict(mapped, strict=False)
            loaded_keys.update(mapped.keys())
            print(f"[stage4] Loaded {len(mapped)} params from stage-2.1 checkpoint", flush=True)
        else:
            print("[stage4] WARNING: No compatible stage-2.1 params found", flush=True)

    def _load_stage3_weights(self, state: dict[str, torch.Tensor], loaded_keys: set[str]) -> None:
        """Load Stage 3 components: MDVSC v2 (mapped from model.* → mdvsc_v2.*)."""
        current = self.model.state_dict()
        mapped = {}
        for key, value in state.items():
            # Stage 3 checkpoint keys are like 'level_modules.*', 'cross_level_fusion.*'
            # In Stage 4, these live under 'mdvsc_v2.*'
            target_key = f"mdvsc_v2.{key}"
            if target_key in current and current[target_key].shape == value.shape:
                mapped[target_key] = value
        if mapped:
            self.model.load_state_dict(mapped, strict=False)
            loaded_keys.update(mapped.keys())
            print(f"[stage4] Loaded {len(mapped)} params from stage-3 checkpoint", flush=True)
        else:
            print("[stage4] WARNING: No compatible stage-3 params found", flush=True)

    @staticmethod
    def _load_model_state(path: str) -> dict[str, torch.Tensor]:
        ckpt = torch.load(path, map_location="cpu")
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            return ckpt["model_state"]
        if isinstance(ckpt, dict):
            return ckpt
        raise ValueError(f"Unsupported checkpoint: {path}")

    # ------------------------------------------------------------------
    # Parameter summary
    # ------------------------------------------------------------------

    def _summarize_params(self) -> dict[str, int]:
        counts = {
            "total": sum(p.numel() for p in self.model.parameters()),
            "shared_encoder": sum(p.numel() for p in self.model.shared_encoder.parameters()),
            "mdvsc_v2": sum(p.numel() for p in self.model.mdvsc_v2.parameters()),
            "det_recovery_head": sum(p.numel() for p in self.model.det_recovery_head.parameters()),
            "detail_compressor": sum(p.numel() for p in self.model.detail_compressor.parameters()),
            "detail_decompressor": sum(p.numel() for p in self.model.detail_decompressor.parameters()),
            "reconstruction_refinement_heads": sum(p.numel() for p in self.model.reconstruction_refinement_heads.parameters()),
            "reconstruction_head": sum(p.numel() for p in self.model.reconstruction_head.parameters()),
        }
        print(f"[stage4] Parameters: {json.dumps(counts, indent=2)}", flush=True)
        return counts

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def _should_vis(self, epoch: int) -> bool:
        if not self.config.output.save_visualizations:
            return False
        every = max(self.config.output.visualization_every_epochs, 1)
        return epoch % every == 0

    def _save_visualizations(
        self,
        epoch: int,
        split: str,
        frames: torch.Tensor,
        projected_sequences: list[torch.Tensor],
        shared_sequences: list[torch.Tensor],
        output: Stage4Output,
    ) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        vis_dir = self.output_dir / "visualizations" / split
        vis_dir.mkdir(parents=True, exist_ok=True)

        num_levels = len(shared_sequences)
        num_frames = min(self.config.output.visualization_num_frames, frames.shape[1])

        # --- 1) Reconstruction comparison ---
        fig, axes = plt.subplots(num_frames, 3, figsize=(12, 4 * num_frames))
        if num_frames == 1:
            axes = np.array([axes])
        for t in range(num_frames):
            gt = _to_numpy_float_array(frames[0, t].permute(1, 2, 0).clamp(0, 1))
            recon = _to_numpy_float_array(output.reconstructed_frames[0, t].permute(1, 2, 0).clamp(0, 1))
            diff = np.abs(recon - gt)
            axes[t, 0].imshow(gt)
            axes[t, 0].set_title(f"t={t} GT")
            axes[t, 1].imshow(recon)
            axes[t, 1].set_title(f"t={t} Recon")
            axes[t, 2].imshow(diff)
            axes[t, 2].set_title(f"t={t} |diff|")
            for c in range(3):
                axes[t, c].axis("off")
        fig.tight_layout()
        fig.savefig(vis_dir / f"epoch_{epoch:03d}_reconstruction.png", dpi=180)
        plt.close(fig)

        # --- 2) Feature maps: teacher shared / restored / |diff| per level ---
        fig, axes = plt.subplots(num_levels, 3, figsize=(12, 4 * num_levels))
        if num_levels == 1:
            axes = np.array([axes])
        for lv in range(num_levels):
            teacher = _to_numpy_float_array(shared_sequences[lv][0, 0].mean(dim=0))
            student = _to_numpy_float_array(output.mdvsc_output.restored_sequences[lv][0, 0].mean(dim=0))
            diff = np.abs(student - teacher)
            axes[lv, 0].imshow(teacher, cmap="viridis")
            axes[lv, 0].set_title(f"lv{lv} shared (teacher)")
            axes[lv, 1].imshow(student, cmap="viridis")
            axes[lv, 1].set_title(f"lv{lv} MDVSC restored")
            axes[lv, 2].imshow(diff, cmap="magma")
            axes[lv, 2].set_title(f"lv{lv} |diff|")
            for c in range(3):
                axes[lv, c].axis("off")
        fig.tight_layout()
        fig.savefig(vis_dir / f"epoch_{epoch:03d}_feature_maps.png", dpi=180)
        plt.close(fig)

        # --- 3) Detection recovery: projected teacher / recovered / |diff| ---
        fig, axes = plt.subplots(num_levels, 3, figsize=(12, 4 * num_levels))
        if num_levels == 1:
            axes = np.array([axes])
        for lv in range(num_levels):
            teacher = _to_numpy_float_array(projected_sequences[lv][0, 0].mean(dim=0))
            student = _to_numpy_float_array(output.det_recovery_sequences[lv][0, 0].mean(dim=0))
            diff = np.abs(student - teacher)
            axes[lv, 0].imshow(teacher, cmap="viridis")
            axes[lv, 0].set_title(f"lv{lv} projected (teacher)")
            axes[lv, 1].imshow(student, cmap="viridis")
            axes[lv, 1].set_title(f"lv{lv} det recovery")
            axes[lv, 2].imshow(diff, cmap="magma")
            axes[lv, 2].set_title(f"lv{lv} |diff|")
            for c in range(3):
                axes[lv, c].axis("off")
        fig.tight_layout()
        fig.savefig(vis_dir / f"epoch_{epoch:03d}_det_recovery.png", dpi=180)
        plt.close(fig)

        # --- 4) Entropy masks ---
        mdvsc_out = output.mdvsc_output
        fig, axes = plt.subplots(num_levels, 4, figsize=(16, 3.5 * num_levels))
        if num_levels == 1:
            axes = np.array([axes])
        for lv in range(num_levels):
            c_mask = _to_numpy_float_array(mdvsc_out.common_masks[lv][0].mean(dim=0))
            i_mask = _to_numpy_float_array(mdvsc_out.individual_masks[lv][0, 0].mean(dim=0))
            c_ent = _to_numpy_float_array(mdvsc_out.common_entropy_maps[lv][0].mean(dim=0))
            i_ent = _to_numpy_float_array(mdvsc_out.individual_entropy_maps[lv][0, 0].mean(dim=0))
            axes[lv, 0].imshow(c_ent, cmap="inferno")
            axes[lv, 0].set_title(f"lv{lv} common entropy")
            axes[lv, 1].imshow(c_mask, cmap="gray", vmin=0, vmax=1)
            axes[lv, 1].set_title(f"lv{lv} common mask")
            axes[lv, 2].imshow(i_ent, cmap="inferno")
            axes[lv, 2].set_title(f"lv{lv} indiv entropy")
            axes[lv, 3].imshow(i_mask, cmap="gray", vmin=0, vmax=1)
            axes[lv, 3].set_title(f"lv{lv} indiv mask")
            for c in range(4):
                axes[lv, c].axis("off")
        fig.tight_layout()
        fig.savefig(vis_dir / f"epoch_{epoch:03d}_entropy_masks.png", dpi=180)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_stage4_training(config: MDVSCStage4TrainConfig) -> dict[str, Any]:
    trainer = Stage4Trainer(config)
    return trainer.run()
