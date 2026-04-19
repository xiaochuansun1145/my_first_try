"""Stage-2.1 trainer: Stage2 + Detail Bypass (compressed stage1 for reconstruction).

Reuses loss utilities from stage2_trainer and adds stage1 feature extraction.
"""

from __future__ import annotations

import json
import math
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

from src.semantic_rtdetr.detector.rtdetr_baseline import RTDetrBaseline
from src.semantic_rtdetr.semantic_comm.stage2_1_model import Stage2_1MDVSC, Stage2_1Output
from src.semantic_rtdetr.training.stage2_1_config import MDVSCStage2_1TrainConfig
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


class Stage2_1Trainer:
    def __init__(self, config: MDVSCStage2_1TrainConfig):
        self.config = config
        self.output_dir = Path(config.output.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _set_seed(config.optimization.seed)

        print("[stage2.1] Loading RT-DETR teacher", flush=True)
        self.baseline = RTDetrBaseline(
            config.detector.hf_name,
            device=config.detector.device,
            local_path=config.detector.local_path,
            cache_dir=config.detector.cache_dir,
        )
        for parameter in self.baseline.model.parameters():
            parameter.requires_grad = False
        self.baseline.model.eval()

        print("[stage2.1] Building Stage2_1MDVSC model", flush=True)
        self.model = Stage2_1MDVSC(
            backbone_channels=config.mdvsc.backbone_channels,
            shared_channels=config.mdvsc.shared_channels,
            reconstruction_hidden_channels=config.mdvsc.reconstruction_hidden_channels,
            reconstruction_detail_channels=config.mdvsc.reconstruction_detail_channels,
            reconstruction_head_type=config.mdvsc.reconstruction_head_type,
            reconstruction_use_checkpoint=config.mdvsc.reconstruction_use_checkpoint,
            stage1_channels=config.mdvsc.stage1_channels,
            detail_latent_channels=config.mdvsc.detail_latent_channels,
            detail_spatial_size=config.mdvsc.detail_spatial_size,
        ).to(self.baseline.device)
        self._load_initialization()
        self.parameter_counts = self._summarize_parameter_counts()

        self.amp_dtype = _resolve_amp_dtype(config.optimization.amp_dtype)
        self.amp_enabled = bool(config.optimization.use_amp and self.baseline.device.type == "cuda")
        self.scaler_enabled = bool(self.amp_enabled and self.amp_dtype == torch.float16)
        self.grad_scaler = torch.amp.GradScaler("cuda", enabled=self.scaler_enabled)
        if config.optimization.use_amp and not self.amp_enabled:
            print("[stage2.1] AMP requested but CUDA unavailable; falling back to FP32", flush=True)
        elif self.amp_enabled:
            print(
                f"[stage2.1] AMP enabled with dtype={config.optimization.amp_dtype} "
                f"(grad_scaler={'on' if self.scaler_enabled else 'off'})",
                flush=True,
            )

        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler = None
        self.scheduler_step_per_batch = False

        print("[stage2.1] Building train/val datasets", flush=True)
        train_dataset, val_dataset = build_train_val_datasets(config.data, seed=config.optimization.seed)
        self.dataset_info = {
            "dataset_name": config.data.dataset_name,
            "train_num_samples": len(train_dataset),
            "val_num_samples": len(val_dataset) if val_dataset is not None else 0,
            "train_source_path": config.data.train_source_path,
            "val_source_path": config.data.val_source_path,
            "train_dataset": _dataset_summary(train_dataset),
            "val_dataset": _dataset_summary(val_dataset) if val_dataset is not None else None,
        }
        print(
            f"[stage2.1] Dataset ready: train={self.dataset_info['train_num_samples']}, "
            f"val={self.dataset_info['val_num_samples']}",
            flush=True,
        )
        pin_memory = self.baseline.device.type == "cuda"
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.optimization.batch_size,
            shuffle=True,
            num_workers=config.optimization.num_workers,
            pin_memory=pin_memory,
        )
        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=config.optimization.batch_size,
                shuffle=False,
                num_workers=config.optimization.num_workers,
                pin_memory=pin_memory,
            )
        print("[stage2.1] Initialization complete", flush=True)

    # -----------------------------------------------------------------------
    # Main training loop
    # -----------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        metrics_path = self.output_dir / "metrics.jsonl"
        config_path = self.output_dir / "resolved_config.json"
        dataset_info_path = self.output_dir / "dataset_info.json"
        config_path.write_text(json.dumps(self.config.to_dict(), indent=2), encoding="utf-8")
        dataset_info_path.write_text(json.dumps(self.dataset_info, indent=2), encoding="utf-8")

        total_epochs = self.config.optimization.epochs
        if total_epochs <= 0:
            raise ValueError("epochs must be positive")

        train_steps_per_epoch = self._steps_per_epoch(self.train_loader)
        for p in self.model.parameters():
            p.requires_grad = True
        parameters = list(self.model.parameters())
        lr = self.config.optimization.lr
        self.optimizer = self._build_optimizer(parameters, lr)
        self.scheduler, self.scheduler_step_per_batch = self._build_scheduler(
            optimizer=self.optimizer, epochs=total_epochs, steps_per_epoch=train_steps_per_epoch, max_lr=lr,
        )

        best_val_loss: float | None = None
        last_summary: dict[str, Any] | None = None
        print(f"[stage2.1] Starting joint training for {total_epochs} epochs", flush=True)

        for epoch in range(1, total_epochs + 1):
            print(f"[stage2.1] Epoch {epoch}/{total_epochs}", flush=True)

            train_metrics = self._run_epoch(self.train_loader, training=True, epoch=epoch)
            val_metrics = None
            if self.val_loader is not None:
                val_metrics = self._run_epoch(self.val_loader, training=False, epoch=epoch)

            if self.scheduler is not None and not self.scheduler_step_per_batch:
                self.scheduler.step()

            summary: dict[str, Any] = {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "lr": float(self.optimizer.param_groups[0]["lr"]),
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(summary, ensure_ascii=False) + "\n")

            current_val = val_metrics["total_loss"] if val_metrics is not None else train_metrics["total_loss"]
            if best_val_loss is None or current_val < best_val_loss:
                best_val_loss = current_val
                self._save_checkpoint("best.pt", epoch, summary)

            if epoch % self.config.optimization.save_every_epochs == 0:
                self._save_checkpoint(f"epoch_{epoch:03d}.pt", epoch, summary)
            self._save_checkpoint("latest.pt", epoch, summary)
            last_summary = summary

        final_summary: dict[str, Any] = {
            "output_dir": str(self.output_dir),
            "best_val_loss": best_val_loss,
            "amp_enabled": self.amp_enabled,
            "amp_dtype": self.config.optimization.amp_dtype,
            "dataset": self.dataset_info,
            "epochs": total_epochs,
            "parameter_counts": self.parameter_counts,
            "last_epoch": last_summary,
        }
        (self.output_dir / "final_summary.json").write_text(
            json.dumps(final_summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return final_summary

    # -----------------------------------------------------------------------
    # Epoch runner
    # -----------------------------------------------------------------------

    def _run_epoch(self, dataloader: DataLoader, training: bool, epoch: int) -> dict[str, float]:
        self.model.train(training)

        num_levels = len(self.config.mdvsc.backbone_channels)
        metric_sums: defaultdict[str, float] = defaultdict(float)
        max_steps = self.config.optimization.max_steps_per_epoch
        steps = 0
        progress = tqdm(dataloader, desc=f"epoch {epoch} {'train' if training else 'val'}", leave=False)

        for batch_index, frames in enumerate(progress, start=1):
            if max_steps is not None and batch_index > max_steps:
                break

            frames = frames.to(self.baseline.device, non_blocking=True)
            batch_size, time_steps, _, height, width = frames.shape
            flat_frames = frames.view(batch_size * time_steps, *frames.shape[2:])

            if training:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=self.baseline.device.type,
                dtype=self.amp_dtype,
                enabled=self.amp_enabled,
            ):
                # Extract teacher features: stage1 + backbone raw + projected
                with torch.no_grad():
                    detector_inputs = self.baseline.prepare_frame_tensor_batch(flat_frames)
                    stage1_flat, raw_features, teacher_bundle = self.baseline.extract_all_backbone_features(
                        detector_inputs
                    )

                # Reshape to sequences
                backbone_sequences = [feat.view(batch_size, time_steps, *feat.shape[1:]) for feat in raw_features]
                projected_sequences = [feat.view(batch_size, time_steps, *feat.shape[1:]) for feat in teacher_bundle.feature_maps]
                stage1_sequences = stage1_flat.view(batch_size, time_steps, *stage1_flat.shape[1:])

                # Forward pass
                model_outputs = self.model(backbone_sequences, stage1_sequences, output_size=(height, width))

                # Compute losses
                loss_dict = self._compute_losses(
                    frames=frames,
                    projected_sequences=projected_sequences,
                    model_outputs=model_outputs,
                    num_levels=num_levels,
                )

            _ensure_finite_tensor("reconstructed_frames", model_outputs.reconstructed_frames)
            for loss_name, loss_value in loss_dict.items():
                _ensure_finite_tensor(f"loss.{loss_name}", loss_value)

            if training:
                if self.scaler_enabled:
                    self.grad_scaler.scale(loss_dict["total"]).backward()
                    self.grad_scaler.unscale_(self.optimizer)
                else:
                    loss_dict["total"].backward()
                trainable_parameters = [p for p in self.model.parameters() if p.requires_grad]
                clip_grad_norm_(trainable_parameters, self.config.optimization.grad_clip_norm)
                if self.scaler_enabled:
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                else:
                    self.optimizer.step()
                if self.scheduler is not None and self.scheduler_step_per_batch:
                    self.scheduler.step()

            detached = {name: float(value.detach().item()) for name, value in loss_dict.items()}
            detached["detail_tx_ratio"] = model_outputs.detail_transmission_ratio
            for name, value in detached.items():
                metric_sums[name] += value
            steps += 1

            if batch_index == 1 and self._should_save_visualizations(epoch):
                self._save_visualizations(
                    epoch=epoch,
                    split="train" if training else "val",
                    frames=frames,
                    projected_sequences=projected_sequences,
                    model_outputs=model_outputs,
                )

            if training and batch_index % self.config.optimization.log_every == 0:
                progress.set_postfix(
                    total=f"{detached['total']:.4f}",
                    recon_mse=f"{detached['recon_mse']:.4f}",
                    det_rec=f"{detached['det_recovery']:.4f}",
                    detail_tx=f"{detached['detail_tx_ratio']:.3%}",
                )

        if steps == 0:
            raise ValueError("No batches processed in epoch")

        avg_metrics: dict[str, float] = {}
        for key in metric_sums:
            if key == "detail_tx_ratio":
                avg_metrics[key] = metric_sums[key] / steps
            else:
                avg_metrics[f"{key}_loss" if key != "total" else "total_loss"] = metric_sums[key] / steps

        # Derived quality metrics
        avg_recon_mse = metric_sums["recon_mse"] / steps
        avg_metrics["recon_psnr"] = _psnr_from_mse(avg_recon_mse)
        avg_metrics["recon_ssim"] = 1.0 - metric_sums["recon_ssim"] / steps

        return avg_metrics

    # -----------------------------------------------------------------------
    # Loss computation
    # -----------------------------------------------------------------------

    def _compute_losses(
        self,
        frames: torch.Tensor,
        projected_sequences: list[torch.Tensor],
        model_outputs: Stage2_1Output,
        num_levels: int,
    ) -> dict[str, torch.Tensor]:
        device = frames.device
        frames_fp32 = frames.float()
        reconstructed = model_outputs.reconstructed_frames.float()

        # Reconstruction losses
        recon_l1 = F.l1_loss(reconstructed, frames_fp32)
        recon_mse = F.mse_loss(reconstructed, frames_fp32)
        recon_ssim = _ssim_loss(reconstructed, frames_fp32, downsample_factor=self.config.loss.ssim_downsample_factor)
        recon_edge = _gradient_edge_loss(reconstructed, frames_fp32)

        # Detection recovery loss
        det_recovery = torch.zeros((), device=device, dtype=torch.float32)
        for level in range(num_levels):
            level_mse = F.mse_loss(
                model_outputs.det_recovery_sequences[level].float(),
                projected_sequences[level].float(),
            )
            det_recovery = det_recovery + float(self.config.loss.level_recovery_weights[level]) * level_mse

        total = (
            self.config.loss.recon_l1_weight * recon_l1
            + self.config.loss.recon_mse_weight * recon_mse
            + self.config.loss.recon_ssim_weight * recon_ssim
            + self.config.loss.recon_edge_weight * recon_edge
            + self.config.loss.det_recovery_weight * det_recovery
        )

        return {
            "total": total,
            "recon_l1": recon_l1,
            "recon_mse": recon_mse,
            "recon_ssim": recon_ssim,
            "recon_edge": recon_edge,
            "det_recovery": det_recovery,
        }

    # -----------------------------------------------------------------------
    # Optimizer / scheduler
    # -----------------------------------------------------------------------

    def _build_optimizer(self, parameters: list[torch.nn.Parameter], lr: float) -> torch.optim.Optimizer:
        optimizer_name = self.config.optimization.optimizer.lower()
        betas = (self.config.optimization.adam_beta1, self.config.optimization.adam_beta2)
        if optimizer_name == "adamw":
            return torch.optim.AdamW(parameters, lr=lr, betas=betas, weight_decay=self.config.optimization.weight_decay)
        if optimizer_name == "adam":
            return torch.optim.Adam(parameters, lr=lr, betas=betas, weight_decay=self.config.optimization.weight_decay)
        raise ValueError(f"Unsupported optimizer: {self.config.optimization.optimizer}")

    def _build_scheduler(
        self, optimizer: torch.optim.Optimizer, epochs: int, steps_per_epoch: int, max_lr: float,
    ) -> tuple[torch.optim.lr_scheduler.LRScheduler | None, bool]:
        scheduler_type = self.config.optimization.scheduler.lower()
        if scheduler_type == "constant":
            return None, False
        if scheduler_type == "onecycle":
            return (
                OneCycleLR(
                    optimizer,
                    max_lr=max_lr,
                    epochs=max(epochs, 1),
                    steps_per_epoch=max(steps_per_epoch, 1),
                    pct_start=self.config.optimization.onecycle_pct_start,
                    div_factor=self.config.optimization.onecycle_div_factor,
                    final_div_factor=self.config.optimization.onecycle_final_div_factor,
                    anneal_strategy="cos",
                ),
                True,
            )
        if scheduler_type != "cosine":
            raise ValueError(f"Unsupported scheduler: {self.config.optimization.scheduler}")

        total_epochs = max(epochs, 1)
        warmup_epochs = max(0, min(self.config.optimization.warmup_epochs, total_epochs - 1))
        eta_min = float(optimizer.param_groups[0]["lr"]) * self.config.optimization.min_lr_ratio
        if warmup_epochs == 0:
            return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=eta_min), False
        warmup = LinearLR(optimizer, start_factor=self.config.optimization.warmup_start_factor, end_factor=1.0, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=max(total_epochs - warmup_epochs, 1), eta_min=eta_min)
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]), False

    def _steps_per_epoch(self, dataloader: DataLoader) -> int:
        if self.config.optimization.max_steps_per_epoch is None:
            return len(dataloader)
        return min(len(dataloader), self.config.optimization.max_steps_per_epoch)

    # -----------------------------------------------------------------------
    # Checkpoint / initialization
    # -----------------------------------------------------------------------

    def _save_checkpoint(self, file_name: str, epoch: int, summary: dict[str, Any]) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler is not None else None,
            "grad_scaler_state": self.grad_scaler.state_dict() if self.scaler_enabled else None,
            "config": self.config.to_dict(),
            "summary": summary,
        }
        torch.save(checkpoint, self.output_dir / file_name)

    def _load_initialization(self) -> None:
        init = self.config.initialization
        if init.full_checkpoint:
            state_dict = self._load_model_state(init.full_checkpoint)
            self.model.load_state_dict(state_dict, strict=init.strict)
        if init.stage2_checkpoint:
            self._load_stage2_weights(init.stage2_checkpoint, strict=init.strict)

    def _load_stage2_weights(self, checkpoint_path: str, strict: bool) -> None:
        """Load compatible weights from a stage-2 checkpoint (SharedEncoder, DetRecovery, ReconHead)."""
        state_dict = self._load_model_state(checkpoint_path)
        current_state = self.model.state_dict()
        compatible_prefixes = (
            "shared_encoder.",
            "det_recovery_head.",
            "reconstruction_refinement_heads.",
            "reconstruction_head.",
        )
        filtered = {
            key: value
            for key, value in state_dict.items()
            if any(key.startswith(p) for p in compatible_prefixes)
            and key in current_state
            and current_state[key].shape == value.shape
        }
        if strict and not filtered:
            raise ValueError(f"No matching params in {checkpoint_path}")
        if filtered:
            self.model.load_state_dict(filtered, strict=False)
            print(f"[stage2.1] Loaded {len(filtered)} params from stage-2 checkpoint", flush=True)

    @staticmethod
    def _load_model_state(checkpoint_path: str) -> dict[str, torch.Tensor]:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            return checkpoint["model_state"]
        if isinstance(checkpoint, dict):
            return checkpoint
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    # -----------------------------------------------------------------------
    # Parameter summary
    # -----------------------------------------------------------------------

    def _summarize_parameter_counts(self) -> dict[str, int]:
        counts = {
            "total": sum(p.numel() for p in self.model.parameters()),
            "shared_encoder": sum(p.numel() for p in self.model.shared_encoder.parameters()),
            "det_recovery_head": sum(p.numel() for p in self.model.det_recovery_head.parameters()),
            "detail_compressor": sum(p.numel() for p in self.model.detail_compressor.parameters()),
            "detail_decompressor": sum(p.numel() for p in self.model.detail_decompressor.parameters()),
            "reconstruction_head": sum(p.numel() for p in self.model.reconstruction_head.parameters()),
            "reconstruction_refinement_heads": sum(p.numel() for p in self.model.reconstruction_refinement_heads.parameters()),
            "trainable_at_init": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        }
        print(f"[stage2.1] Parameter counts: {json.dumps(counts, indent=2)}", flush=True)
        return counts

    # -----------------------------------------------------------------------
    # Visualization
    # -----------------------------------------------------------------------

    def _should_save_visualizations(self, epoch: int) -> bool:
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
        model_outputs: Stage2_1Output,
    ) -> None:
        import matplotlib.pyplot as plt

        vis_dir = self.output_dir / "visualizations" / split
        vis_dir.mkdir(parents=True, exist_ok=True)

        max_frames = min(self.config.output.visualization_num_frames, frames.shape[1])

        # --- Reconstruction quality ---
        orig = _to_numpy_float_array(frames[0, :max_frames].permute(0, 2, 3, 1))
        recon = _to_numpy_float_array(model_outputs.reconstructed_frames[0, :max_frames].permute(0, 2, 3, 1))
        base = _to_numpy_float_array(model_outputs.reconstructed_base_frames[0, :max_frames].permute(0, 2, 3, 1))
        hf = _to_numpy_float_array(model_outputs.reconstructed_high_frequency_residuals[0, :max_frames].abs().mean(dim=1))

        fig, axes = plt.subplots(4, max_frames, figsize=(4 * max_frames, 12))
        if max_frames == 1:
            axes = np.array([[axes[0]], [axes[1]], [axes[2]], [axes[3]]])
        for t in range(max_frames):
            axes[0, t].imshow(orig[t])
            axes[0, t].set_title(f"original t={t}")
            axes[0, t].axis("off")
            axes[1, t].imshow(recon[t])
            axes[1, t].set_title(f"reconstructed t={t}")
            axes[1, t].axis("off")
            axes[2, t].imshow(base[t])
            axes[2, t].set_title(f"base t={t}")
            axes[2, t].axis("off")
            axes[3, t].imshow(hf[t], cmap="magma")
            axes[3, t].set_title(f"hf residual t={t}")
            axes[3, t].axis("off")
        fig.tight_layout()
        fig.savefig(vis_dir / f"epoch_{epoch:03d}_reconstruction.png", dpi=180)
        plt.close(fig)

        # --- Feature recovery heatmaps ---
        num_levels = len(projected_sequences)
        fig, axes = plt.subplots(num_levels, 3, figsize=(12, 4 * num_levels))
        if num_levels == 1:
            axes = np.array([axes])
        for level in range(num_levels):
            teacher_map = _to_numpy_float_array(projected_sequences[level][0, 0].mean(dim=0))
            student_map = _to_numpy_float_array(model_outputs.det_recovery_sequences[level][0, 0].mean(dim=0))
            diff_map = np.abs(student_map - teacher_map)
            axes[level, 0].imshow(teacher_map, cmap="viridis")
            axes[level, 0].set_title(f"level {level} projected (teacher)")
            axes[level, 1].imshow(student_map, cmap="viridis")
            axes[level, 1].set_title(f"level {level} det recovery")
            axes[level, 2].imshow(diff_map, cmap="magma")
            axes[level, 2].set_title(f"level {level} abs diff")
            for col in range(3):
                axes[level, col].axis("off")
        fig.tight_layout()
        fig.savefig(vis_dir / f"epoch_{epoch:03d}_feature_recovery.png", dpi=180)
        plt.close(fig)

        # --- Detail compressed packet visualization ---
        detail = model_outputs.detail_compressed[0, 0]  # [C_latent, Hs, Ws]
        detail_mean = _to_numpy_float_array(detail.mean(dim=0))
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(detail_mean, cmap="viridis")
        ax.set_title(f"detail packet mean ({detail.shape[0]}ch, {detail.shape[1]}x{detail.shape[2]})")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(vis_dir / f"epoch_{epoch:03d}_detail_packet.png", dpi=180)
        plt.close(fig)


def run_stage2_1_training(config: MDVSCStage2_1TrainConfig) -> dict[str, Any]:
    trainer = Stage2_1Trainer(config)
    return trainer.run()
