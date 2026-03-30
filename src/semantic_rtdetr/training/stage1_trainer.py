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
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.semantic_rtdetr.contracts import EncoderFeatureBundle
from src.semantic_rtdetr.detector.rtdetr_baseline import RTDetrBaseline
from src.semantic_rtdetr.semantic_comm.mdvsc import MDVSCOutput, ProjectMDVSC
from src.semantic_rtdetr.training.stage1_config import MDVSCStage1TrainConfig
from src.semantic_rtdetr.training.stage1_data import build_train_val_datasets


def _dataset_summary(dataset) -> dict[str, Any]:
    base_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
    if hasattr(base_dataset, "summary"):
        summary = dict(base_dataset.summary())
        summary["visible_samples"] = len(dataset)
        if isinstance(dataset, Subset):
            summary["subset_type"] = "random_split"
        return summary
    return {"visible_samples": len(dataset)}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _bundle_to_sequences(bundle: EncoderFeatureBundle, batch_size: int, time_steps: int) -> list[torch.Tensor]:
    sequences: list[torch.Tensor] = []
    for feature_map in bundle.feature_maps:
        sequences.append(feature_map.view(batch_size, time_steps, *feature_map.shape[1:]))
    return sequences


def _sequences_to_bundle(sequences: list[torch.Tensor], reference: EncoderFeatureBundle) -> EncoderFeatureBundle:
    flattened_feature_maps = [sequence.reshape(sequence.shape[0] * sequence.shape[1], *sequence.shape[2:]) for sequence in sequences]
    return EncoderFeatureBundle(
        feature_maps=flattened_feature_maps,
        spatial_shapes=reference.spatial_shapes.clone(),
        level_start_index=reference.level_start_index.clone(),
        strides=list(reference.strides) if reference.strides is not None else None,
    )


class Stage1Trainer:
    def __init__(self, config: MDVSCStage1TrainConfig):
        self.config = config
        self.output_dir = Path(config.output.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _set_seed(config.optimization.seed)

        print("[stage1] Loading RT-DETR teacher", flush=True)

        self.baseline = RTDetrBaseline(
            config.detector.hf_name,
            device=config.detector.device,
            local_path=config.detector.local_path,
            cache_dir=config.detector.cache_dir,
        )
        for parameter in self.baseline.model.parameters():
            parameter.requires_grad = False
        self.baseline.model.eval()

        self.model = ProjectMDVSC(
            feature_channels=config.mdvsc.feature_channels,
            latent_dims=config.mdvsc.latent_dims,
            common_keep_ratios=config.mdvsc.common_keep_ratios,
            individual_keep_ratios=config.mdvsc.individual_keep_ratios,
            block_sizes=config.mdvsc.block_sizes,
        ).to(self.baseline.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.optimization.lr,
            weight_decay=config.optimization.weight_decay,
        )
        self.scheduler = self._build_scheduler()

        print("[stage1] Building train/val datasets", flush=True)
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
            f"[stage1] Dataset ready: train={self.dataset_info['train_num_samples']} samples, "
            f"val={self.dataset_info['val_num_samples']} samples",
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
            print("[stage1] DataLoader initialization complete", flush=True)

    def run(self) -> dict[str, Any]:
        metrics_path = self.output_dir / "metrics.jsonl"
        config_path = self.output_dir / "resolved_config.json"
        dataset_info_path = self.output_dir / "dataset_info.json"
        config_path.write_text(json.dumps(self.config.to_dict(), indent=2), encoding="utf-8")
        dataset_info_path.write_text(json.dumps(self.dataset_info, indent=2), encoding="utf-8")

        best_val_loss: float | None = None
        last_summary: dict[str, Any] | None = None
        for epoch in range(1, self.config.optimization.epochs + 1):
            print(f"[stage1] Starting epoch {epoch}/{self.config.optimization.epochs}", flush=True)
            train_metrics = self._run_epoch(self.train_loader, training=True, epoch=epoch)
            val_metrics = None
            if self.val_loader is not None:
                val_metrics = self._run_epoch(self.val_loader, training=False, epoch=epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            summary = {
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

        final_summary = {
            "output_dir": str(self.output_dir),
            "best_val_loss": best_val_loss,
            "dataset": self.dataset_info,
            "last_epoch": last_summary,
        }
        (self.output_dir / "final_summary.json").write_text(
            json.dumps(final_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return final_summary

    def _run_epoch(self, dataloader: DataLoader, training: bool, epoch: int) -> dict[str, float]:
        self.model.train(training)
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

            with torch.no_grad():
                detector_inputs = self.baseline.prepare_frame_tensor_batch(flat_frames)
                teacher_bundle = self.baseline.extract_encoder_feature_bundle(detector_inputs)
                teacher_outputs = self.baseline.predict(detector_inputs)

            if training:
                self.optimizer.zero_grad(set_to_none=True)

            target_sequences = _bundle_to_sequences(teacher_bundle, batch_size, time_steps)
            model_outputs = self.model(
                target_sequences,
                output_size=(height, width),
                apply_masks=self.config.mdvsc.apply_masks,
                channel_mode=self.config.mdvsc.channel_mode,
                snr_db=self.config.mdvsc.snr_db,
            )
            loss_dict = self._compute_losses(
                frames=frames,
                target_sequences=target_sequences,
                model_outputs=model_outputs,
                detector_inputs=detector_inputs,
                teacher_bundle=teacher_bundle,
                teacher_outputs=teacher_outputs,
            )

            if training:
                loss_dict["total"].backward()
                clip_grad_norm_(self.model.parameters(), self.config.optimization.grad_clip_norm)
                self.optimizer.step()

            detached = {name: float(value.detach().item()) for name, value in loss_dict.items()}
            for name, value in detached.items():
                metric_sums[name] += value

            metric_sums["common_active_ratio"] += float(
                np.mean([stat.common_active_ratio for stat in model_outputs.level_stats])
            )
            metric_sums["individual_active_ratio"] += float(
                np.mean([stat.individual_active_ratio for stat in model_outputs.level_stats])
            )
            steps += 1

            if batch_index == 1 and self._should_save_visualizations(epoch):
                self._save_visualizations(
                    epoch=epoch,
                    split="train" if training else "val",
                    frames=frames,
                    target_sequences=target_sequences,
                    model_outputs=model_outputs,
                )

            if training and batch_index % self.config.optimization.log_every == 0:
                progress.set_postfix(total=f"{detached['total']:.4f}", feature=f"{detached['feature']:.4f}")

        if steps == 0:
            raise ValueError("No batches were processed in the current epoch")

        return {
            "total_loss": metric_sums["total"] / steps,
            "feature_loss": metric_sums["feature"] / steps,
            "recon_l1_loss": metric_sums["recon_l1"] / steps,
            "recon_mse_loss": metric_sums["recon_mse"] / steps,
            "detection_logit_loss": metric_sums["detection_logit"] / steps,
            "detection_box_loss": metric_sums["detection_box"] / steps,
            "common_active_ratio": metric_sums["common_active_ratio"] / steps,
            "individual_active_ratio": metric_sums["individual_active_ratio"] / steps,
        }

    def _compute_losses(
        self,
        frames: torch.Tensor,
        target_sequences: list[torch.Tensor],
        model_outputs: MDVSCOutput,
        detector_inputs: dict[str, torch.Tensor],
        teacher_bundle: EncoderFeatureBundle,
        teacher_outputs,
    ) -> dict[str, torch.Tensor]:
        feature_loss = torch.zeros((), device=frames.device)
        for level_weight, restored_sequence, target_sequence in zip(
            self.config.loss.level_loss_weights,
            model_outputs.restored_sequences,
            target_sequences,
        ):
            feature_loss = feature_loss + float(level_weight) * F.smooth_l1_loss(restored_sequence, target_sequence)

        reconstructed_frames = model_outputs.reconstructed_frames
        recon_l1_loss = F.l1_loss(reconstructed_frames, frames)
        recon_mse_loss = F.mse_loss(reconstructed_frames, frames)

        detection_logit_loss = torch.zeros((), device=frames.device)
        detection_box_loss = torch.zeros((), device=frames.device)
        if self.config.loss.detection_logit_weight > 0.0 or self.config.loss.detection_box_weight > 0.0:
            student_bundle = _sequences_to_bundle(model_outputs.restored_sequences, teacher_bundle)
            student_outputs = self.baseline.forward_from_encoder_feature_bundle(detector_inputs, student_bundle)
            detection_logit_loss = F.mse_loss(student_outputs.logits, teacher_outputs.logits.detach())
            detection_box_loss = F.l1_loss(student_outputs.pred_boxes, teacher_outputs.pred_boxes.detach())

        total_loss = (
            self.config.loss.feature_loss_weight * feature_loss
            + self.config.loss.recon_l1_weight * recon_l1_loss
            + self.config.loss.recon_mse_weight * recon_mse_loss
            + self.config.loss.detection_logit_weight * detection_logit_loss
            + self.config.loss.detection_box_weight * detection_box_loss
        )

        return {
            "total": total_loss,
            "feature": feature_loss,
            "recon_l1": recon_l1_loss,
            "recon_mse": recon_mse_loss,
            "detection_logit": detection_logit_loss,
            "detection_box": detection_box_loss,
        }

    def _save_checkpoint(self, file_name: str, epoch: int, summary: dict[str, Any]) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler is not None else None,
            "config": self.config.to_dict(),
            "summary": summary,
        }
        torch.save(checkpoint, self.output_dir / file_name)

    def _build_scheduler(self):
        scheduler_type = self.config.optimization.scheduler.lower()
        if scheduler_type == "constant":
            return None
        if scheduler_type != "cosine":
            raise ValueError(f"Unsupported scheduler type: {self.config.optimization.scheduler}")

        total_epochs = max(self.config.optimization.epochs, 1)
        warmup_epochs = max(0, min(self.config.optimization.warmup_epochs, total_epochs - 1))
        eta_min = self.config.optimization.lr * self.config.optimization.min_lr_ratio

        if warmup_epochs == 0:
            return CosineAnnealingLR(self.optimizer, T_max=total_epochs, eta_min=eta_min)

        warmup = LinearLR(
            self.optimizer,
            start_factor=self.config.optimization.warmup_start_factor,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=max(total_epochs - warmup_epochs, 1),
            eta_min=eta_min,
        )
        return SequentialLR(self.optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

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
        target_sequences: list[torch.Tensor],
        model_outputs: MDVSCOutput,
    ) -> None:
        import matplotlib.pyplot as plt

        vis_dir = self.output_dir / "visualizations" / split
        vis_dir.mkdir(parents=True, exist_ok=True)

        max_frames = min(self.config.output.visualization_num_frames, frames.shape[1])
        original_frames = frames[0, :max_frames].detach().cpu().permute(0, 2, 3, 1).numpy()
        reconstructed_frames = model_outputs.reconstructed_frames[0, :max_frames].detach().cpu().permute(0, 2, 3, 1).numpy()

        figure, axes = plt.subplots(2, max_frames, figsize=(4 * max_frames, 8))
        if max_frames == 1:
            axes = np.array([[axes[0]], [axes[1]]])
        for frame_index in range(max_frames):
            axes[0, frame_index].imshow(original_frames[frame_index])
            axes[0, frame_index].set_title(f"original t={frame_index}")
            axes[0, frame_index].axis("off")
            axes[1, frame_index].imshow(reconstructed_frames[frame_index])
            axes[1, frame_index].set_title(f"reconstructed t={frame_index}")
            axes[1, frame_index].axis("off")
        figure.tight_layout()
        figure.savefig(vis_dir / f"epoch_{epoch:03d}_reconstruction.png", dpi=180)
        plt.close(figure)

        num_levels = len(target_sequences)
        figure, axes = plt.subplots(num_levels, 3, figsize=(12, 4 * num_levels))
        if num_levels == 1:
            axes = np.array([axes])
        for level_index in range(num_levels):
            teacher_map = target_sequences[level_index][0, 0].detach().cpu().mean(dim=0).numpy()
            student_map = model_outputs.restored_sequences[level_index][0, 0].detach().cpu().mean(dim=0).numpy()
            diff_map = np.abs(student_map - teacher_map)
            axes[level_index, 0].imshow(teacher_map, cmap="viridis")
            axes[level_index, 0].set_title(f"level {level_index} teacher")
            axes[level_index, 1].imshow(student_map, cmap="viridis")
            axes[level_index, 1].set_title(f"level {level_index} restored")
            axes[level_index, 2].imshow(diff_map, cmap="magma")
            axes[level_index, 2].set_title(f"level {level_index} abs diff")
            for column in range(3):
                axes[level_index, column].axis("off")
        figure.tight_layout()
        figure.savefig(vis_dir / f"epoch_{epoch:03d}_feature_maps.png", dpi=180)
        plt.close(figure)

        figure, axes = plt.subplots(num_levels, 2, figsize=(12, 3 * num_levels))
        if num_levels == 1:
            axes = np.array([axes])
        for level_index in range(num_levels):
            common_mask = model_outputs.common_masks[level_index][0, :, 0, 0].detach().cpu().numpy()[None, :]
            individual_mask = model_outputs.individual_masks[level_index][0, 0, 0].detach().cpu().numpy()
            axes[level_index, 0].imshow(common_mask, cmap="gray", aspect="auto", vmin=0.0, vmax=1.0)
            axes[level_index, 0].set_title(f"level {level_index} common mask")
            axes[level_index, 1].imshow(individual_mask, cmap="gray", vmin=0.0, vmax=1.0)
            axes[level_index, 1].set_title(f"level {level_index} individual mask t=0")
            axes[level_index, 0].set_ylabel(f"level {level_index}")
            axes[level_index, 0].set_yticks([])
            axes[level_index, 1].set_yticks([])
        figure.tight_layout()
        figure.savefig(vis_dir / f"epoch_{epoch:03d}_masks.png", dpi=180)
        plt.close(figure)


def run_stage1_training(config: MDVSCStage1TrainConfig) -> dict[str, Any]:
    trainer = Stage1Trainer(config)
    return trainer.run()