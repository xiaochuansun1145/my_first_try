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
from src.semantic_rtdetr.semantic_comm.mdvsc import MDVSCOutput, ProjectMDVSC
from src.semantic_rtdetr.training.stage1_config import MDVSCStage1TrainConfig
from src.semantic_rtdetr.training.stage1_data import build_train_val_datasets

PHASE_RECONSTRUCTION_PRETRAIN = "reconstruction_pretrain"
PHASE_MDVSC_BOOTSTRAP = "mdvsc_bootstrap"
PHASE_JOINT_TRAINING = "joint_training"


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


def _gaussian_kernel(kernel_size: int, sigma: float, channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coordinates = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
    kernel_1d = torch.exp(-(coordinates.pow(2)) / (2 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.expand(channels, 1, kernel_size, kernel_size).contiguous()


def _ssim_loss(prediction: torch.Tensor, target: torch.Tensor, kernel_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape for SSIM")

    batch_size, time_steps, channels, height, width = prediction.shape
    prediction_2d = prediction.view(batch_size * time_steps, channels, height, width)
    target_2d = target.view(batch_size * time_steps, channels, height, width)
    kernel = _gaussian_kernel(kernel_size, sigma, channels, prediction.device, prediction.dtype)
    padding = kernel_size // 2

    mu_prediction = F.conv2d(prediction_2d, kernel, padding=padding, groups=channels)
    mu_target = F.conv2d(target_2d, kernel, padding=padding, groups=channels)

    mu_prediction_sq = mu_prediction.pow(2)
    mu_target_sq = mu_target.pow(2)
    mu_prediction_target = mu_prediction * mu_target

    sigma_prediction_sq = F.conv2d(prediction_2d * prediction_2d, kernel, padding=padding, groups=channels) - mu_prediction_sq
    sigma_target_sq = F.conv2d(target_2d * target_2d, kernel, padding=padding, groups=channels) - mu_target_sq
    sigma_prediction_target = F.conv2d(
        prediction_2d * target_2d,
        kernel,
        padding=padding,
        groups=channels,
    ) - mu_prediction_target

    c1 = 0.01**2
    c2 = 0.03**2
    ssim_numerator = (2 * mu_prediction_target + c1) * (2 * sigma_prediction_target + c2)
    ssim_denominator = (mu_prediction_sq + mu_target_sq + c1) * (sigma_prediction_sq + sigma_target_sq + c2)
    ssim_map = ssim_numerator / ssim_denominator.clamp_min(1e-6)
    return 1.0 - ssim_map.mean()


def _gradient_edge_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape for gradient edge loss")

    prediction_dx = prediction[..., :, :, 1:] - prediction[..., :, :, :-1]
    target_dx = target[..., :, :, 1:] - target[..., :, :, :-1]
    prediction_dy = prediction[..., :, 1:, :] - prediction[..., :, :-1, :]
    target_dy = target[..., :, 1:, :] - target[..., :, :-1, :]
    return F.l1_loss(prediction_dx, target_dx) + F.l1_loss(prediction_dy, target_dy)


def _resolve_amp_dtype(amp_dtype: str) -> torch.dtype:
    normalized = amp_dtype.lower()
    if normalized == "float16":
        return torch.float16
    if normalized == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported AMP dtype: {amp_dtype}")


def _to_numpy_float_array(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def _ensure_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    if torch.isfinite(tensor).all():
        return

    detached = tensor.detach().float()
    finite_mask = torch.isfinite(detached)
    finite_values = detached[finite_mask]
    min_value = float(finite_values.min().item()) if finite_values.numel() > 0 else float("nan")
    max_value = float(finite_values.max().item()) if finite_values.numel() > 0 else float("nan")
    raise ValueError(
        f"Non-finite values detected in {name}: "
        f"finite={int(finite_mask.sum().item())}/{detached.numel()}, min={min_value}, max={max_value}"
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
            reconstruction_hidden_channels=config.mdvsc.reconstruction_hidden_channels,
            reconstruction_detail_channels=config.mdvsc.reconstruction_detail_channels,
        ).to(self.baseline.device)
        self._load_initialization()
        self.parameter_counts = self._summarize_parameter_counts()
        self.amp_dtype = _resolve_amp_dtype(config.optimization.amp_dtype)
        self.amp_enabled = bool(config.optimization.use_amp and self.baseline.device.type == "cuda")
        self.scaler_enabled = bool(self.amp_enabled and self.amp_dtype == torch.float16)
        self.grad_scaler = torch.amp.GradScaler("cuda", enabled=self.scaler_enabled)
        if config.optimization.use_amp and not self.amp_enabled:
            print("[stage1] AMP requested but CUDA is unavailable; falling back to FP32", flush=True)
        elif self.amp_enabled:
            print(
                f"[stage1] AMP enabled on CUDA with dtype={config.optimization.amp_dtype} "
                f"(grad_scaler={'on' if self.scaler_enabled else 'off'})",
                flush=True,
            )
        self.optimizer = None
        self.scheduler = None
        self.scheduler_step_per_batch = False

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
        global_epoch = 0
        phase_sequence: list[tuple[str, int]] = []
        if self.config.optimization.reconstruction_pretrain_epochs > 0:
            phase_sequence.append((PHASE_RECONSTRUCTION_PRETRAIN, self.config.optimization.reconstruction_pretrain_epochs))
        if self.config.optimization.mdvsc_bootstrap_epochs > 0:
            phase_sequence.append((PHASE_MDVSC_BOOTSTRAP, self.config.optimization.mdvsc_bootstrap_epochs))
        if self.config.optimization.epochs > 0:
            phase_sequence.append((PHASE_JOINT_TRAINING, self.config.optimization.epochs))
        if not phase_sequence:
            raise ValueError("At least one stage must have a positive epoch count")

        tracked_best_phase = PHASE_JOINT_TRAINING if self.config.optimization.epochs > 0 else phase_sequence[-1][0]
        train_steps_per_epoch = self._steps_per_epoch(self.train_loader)

        for phase_name, phase_epochs in phase_sequence:
            self.optimizer, self.scheduler, self.scheduler_step_per_batch = self._build_phase_optimizer(
                phase=phase_name,
                steps_per_epoch=train_steps_per_epoch,
            )
            if phase_name == PHASE_RECONSTRUCTION_PRETRAIN:
                print("[stage1] Starting reconstruction-head pretraining with frozen RT-DETR teacher", flush=True)
            elif phase_name == PHASE_MDVSC_BOOTSTRAP:
                print("[stage1] Starting MDVSC bootstrap with frozen reconstruction head", flush=True)
            else:
                print("[stage1] Starting joint MDVSC + reconstruction training", flush=True)

            for phase_epoch in range(1, phase_epochs + 1):
                global_epoch += 1
                print(
                    f"[stage1] Starting {phase_name} epoch {phase_epoch}/{phase_epochs} "
                    f"(global {global_epoch})",
                    flush=True,
                )
                train_metrics = self._run_epoch(
                    self.train_loader,
                    training=True,
                    epoch=global_epoch,
                    phase=phase_name,
                )
                val_metrics = None
                if self.val_loader is not None:
                    val_metrics = self._run_epoch(
                        self.val_loader,
                        training=False,
                        epoch=global_epoch,
                        phase=phase_name,
                    )

                if self.scheduler is not None and not self.scheduler_step_per_batch:
                    self.scheduler.step()

                summary = {
                    "epoch": global_epoch,
                    "phase": phase_name,
                    "phase_epoch": phase_epoch,
                    "train": train_metrics,
                    "val": val_metrics,
                    "lr": float(self.optimizer.param_groups[0]["lr"]),
                }
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(summary, ensure_ascii=False) + "\n")

                current_val = val_metrics["total_loss"] if val_metrics is not None else train_metrics["total_loss"]
                if phase_name == tracked_best_phase and (best_val_loss is None or current_val < best_val_loss):
                    best_val_loss = current_val
                    self._save_checkpoint("best.pt", global_epoch, summary)

                if global_epoch % self.config.optimization.save_every_epochs == 0:
                    self._save_checkpoint(f"epoch_{global_epoch:03d}.pt", global_epoch, summary)
                self._save_checkpoint("latest.pt", global_epoch, summary)
                last_summary = summary

        final_summary = {
            "output_dir": str(self.output_dir),
            "best_val_loss": best_val_loss,
            "amp_enabled": self.amp_enabled,
            "amp_dtype": self.config.optimization.amp_dtype,
            "dataset": self.dataset_info,
            "reconstruction_pretrain_epochs": self.config.optimization.reconstruction_pretrain_epochs,
            "mdvsc_bootstrap_epochs": self.config.optimization.mdvsc_bootstrap_epochs,
            "joint_training_epochs": self.config.optimization.epochs,
            "parameter_counts": self.parameter_counts,
            "phase_sequence": [{"phase": phase_name, "epochs": epochs} for phase_name, epochs in phase_sequence],
            "last_epoch": last_summary,
        }
        (self.output_dir / "final_summary.json").write_text(
            json.dumps(final_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return final_summary

    def _run_epoch(self, dataloader: DataLoader, training: bool, epoch: int, phase: str) -> dict[str, float]:
        self.model.train(training)
        metric_sums: defaultdict[str, float] = defaultdict(float)
        max_steps = self.config.optimization.max_steps_per_epoch
        steps = 0
        progress = tqdm(dataloader, desc=f"epoch {epoch} {phase} {'train' if training else 'val'}", leave=False)

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
                with torch.no_grad():
                    detector_inputs = self.baseline.prepare_frame_tensor_batch(flat_frames)
                    teacher_bundle = self.baseline.extract_projected_backbone_feature_bundle(detector_inputs)
                    teacher_outputs = None
                    if phase == PHASE_JOINT_TRAINING and (
                        self.config.loss.detection_logit_weight > 0.0 or self.config.loss.detection_box_weight > 0.0
                    ):
                        teacher_outputs = self.baseline.predict(detector_inputs)

                target_sequences = _bundle_to_sequences(teacher_bundle, batch_size, time_steps)
                if phase == PHASE_RECONSTRUCTION_PRETRAIN:
                    model_outputs = self.model.reconstruct_from_feature_sequences(
                        target_sequences,
                        output_size=(height, width),
                    )
                else:
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
                    phase=phase,
                )

            _ensure_finite_tensor("reconstructed_frames", model_outputs.reconstructed_frames)
            _ensure_finite_tensor("reconstructed_base_frames", model_outputs.reconstructed_base_frames)
            _ensure_finite_tensor(
                "reconstructed_high_frequency_residuals",
                model_outputs.reconstructed_high_frequency_residuals,
            )
            for loss_name, loss_value in loss_dict.items():
                _ensure_finite_tensor(f"loss.{loss_name}", loss_value)

            if training:
                if self.scaler_enabled:
                    self.grad_scaler.scale(loss_dict["total"]).backward()
                    self.grad_scaler.unscale_(self.optimizer)
                else:
                    loss_dict["total"].backward()
                trainable_parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
                clip_grad_norm_(trainable_parameters, self.config.optimization.grad_clip_norm)
                if self.scaler_enabled:
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                else:
                    self.optimizer.step()
                if self.scheduler is not None and self.scheduler_step_per_batch:
                    self.scheduler.step()

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
            "recon_ssim_loss": metric_sums["recon_ssim"] / steps,
            "recon_edge_loss": metric_sums["recon_edge"] / steps,
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
        phase: str,
    ) -> dict[str, torch.Tensor]:
        frames_fp32 = frames.float()
        feature_loss = torch.zeros((), device=frames.device, dtype=torch.float32)
        if phase in {PHASE_MDVSC_BOOTSTRAP, PHASE_JOINT_TRAINING}:
            for level_weight, restored_sequence, target_sequence in zip(
                self.config.loss.level_loss_weights,
                model_outputs.restored_sequences,
                target_sequences,
            ):
                feature_loss = feature_loss + float(level_weight) * F.smooth_l1_loss(
                    restored_sequence.float(),
                    target_sequence.float(),
                )

        reconstructed_frames = model_outputs.reconstructed_frames.float()
        recon_l1_loss = torch.zeros((), device=frames.device, dtype=torch.float32)
        recon_mse_loss = torch.zeros((), device=frames.device, dtype=torch.float32)
        recon_ssim_loss = torch.zeros((), device=frames.device, dtype=torch.float32)
        recon_edge_loss = torch.zeros((), device=frames.device, dtype=torch.float32)
        if phase in {PHASE_RECONSTRUCTION_PRETRAIN, PHASE_JOINT_TRAINING}:
            recon_l1_loss = F.l1_loss(reconstructed_frames, frames_fp32)
            recon_mse_loss = F.mse_loss(reconstructed_frames, frames_fp32)
            recon_ssim_loss = _ssim_loss(reconstructed_frames, frames_fp32)
            recon_edge_loss = _gradient_edge_loss(reconstructed_frames, frames_fp32)

        detection_logit_loss = torch.zeros((), device=frames.device, dtype=torch.float32)
        detection_box_loss = torch.zeros((), device=frames.device, dtype=torch.float32)
        if (
            phase == PHASE_JOINT_TRAINING
            and teacher_outputs is not None
            and (self.config.loss.detection_logit_weight > 0.0 or self.config.loss.detection_box_weight > 0.0)
        ):
            student_bundle = _sequences_to_bundle(model_outputs.detection_sequences, teacher_bundle)
            student_outputs = self.baseline.forward_from_encoder_feature_bundle(detector_inputs, student_bundle)
            detection_logit_loss = F.mse_loss(
                student_outputs.logits.float(),
                teacher_outputs.logits.detach().float(),
            )
            detection_box_loss = F.l1_loss(
                student_outputs.pred_boxes.float(),
                teacher_outputs.pred_boxes.detach().float(),
            )

        total_loss = (
            self.config.loss.feature_loss_weight * feature_loss
            + self.config.loss.recon_l1_weight * recon_l1_loss
            + self.config.loss.recon_mse_weight * recon_mse_loss
            + self.config.loss.recon_ssim_weight * recon_ssim_loss
            + self.config.loss.recon_edge_weight * recon_edge_loss
            + self.config.loss.detection_logit_weight * detection_logit_loss
            + self.config.loss.detection_box_weight * detection_box_loss
        )

        return {
            "total": total_loss,
            "feature": feature_loss,
            "recon_l1": recon_l1_loss,
            "recon_mse": recon_mse_loss,
            "recon_ssim": recon_ssim_loss,
            "recon_edge": recon_edge_loss,
            "detection_logit": detection_logit_loss,
            "detection_box": detection_box_loss,
        }

    def _build_phase_optimizer(self, phase: str, steps_per_epoch: int):
        if phase == PHASE_RECONSTRUCTION_PRETRAIN:
            self._set_phase_trainability(phase)
            parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
            lr = self.config.optimization.reconstruction_pretrain_lr
        elif phase == PHASE_MDVSC_BOOTSTRAP:
            self._set_phase_trainability(phase)
            parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
            lr = self.config.optimization.mdvsc_bootstrap_lr
        else:
            self._set_phase_trainability(phase)
            parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
            lr = self.config.optimization.lr

        if not parameters:
            raise ValueError(f"No trainable parameters found for phase: {phase}")

        optimizer = self._build_optimizer(parameters=parameters, lr=lr)
        phase_epochs = self._phase_epochs(phase)
        scheduler, scheduler_step_per_batch = self._build_scheduler(
            optimizer=optimizer,
            epochs=phase_epochs,
            steps_per_epoch=steps_per_epoch,
            max_lr=lr,
        )
        return optimizer, scheduler, scheduler_step_per_batch

    def _build_optimizer(self, parameters: list[torch.nn.Parameter], lr: float) -> torch.optim.Optimizer:
        optimizer_name = self.config.optimization.optimizer.lower()
        betas = (self.config.optimization.adam_beta1, self.config.optimization.adam_beta2)
        if optimizer_name == "adamw":
            return torch.optim.AdamW(
                parameters,
                lr=lr,
                betas=betas,
                weight_decay=self.config.optimization.weight_decay,
            )
        if optimizer_name == "adam":
            return torch.optim.Adam(
                parameters,
                lr=lr,
                betas=betas,
                weight_decay=self.config.optimization.weight_decay,
            )
        raise ValueError(f"Unsupported optimizer type: {self.config.optimization.optimizer}")

    def _set_phase_trainability(self, phase: str) -> None:
        reconstruction_trainable = phase in {PHASE_RECONSTRUCTION_PRETRAIN, PHASE_JOINT_TRAINING}
        mdvsc_trainable = phase in {PHASE_MDVSC_BOOTSTRAP, PHASE_JOINT_TRAINING}
        detection_adaptation_trainable = phase == PHASE_JOINT_TRAINING

        for parameter in self.model.reconstruction_head.parameters():
            parameter.requires_grad = reconstruction_trainable
        for parameter in self.model.reconstruction_refinement_heads.parameters():
            parameter.requires_grad = reconstruction_trainable
        for parameter in self.model.detection_refinement_heads.parameters():
            parameter.requires_grad = detection_adaptation_trainable
        for parameter in self.model.level_modules.parameters():
            parameter.requires_grad = mdvsc_trainable

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

    def _build_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        steps_per_epoch: int,
        max_lr: float,
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
            raise ValueError(f"Unsupported scheduler type: {self.config.optimization.scheduler}")

        total_epochs = max(epochs, 1)
        warmup_epochs = max(0, min(self.config.optimization.warmup_epochs, total_epochs - 1))
        eta_min = float(optimizer.param_groups[0]["lr"]) * self.config.optimization.min_lr_ratio

        if warmup_epochs == 0:
            return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=eta_min), False

        warmup = LinearLR(
            optimizer,
            start_factor=self.config.optimization.warmup_start_factor,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=max(total_epochs - warmup_epochs, 1),
            eta_min=eta_min,
        )
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]), False

    def _phase_epochs(self, phase: str) -> int:
        if phase == PHASE_RECONSTRUCTION_PRETRAIN:
            return self.config.optimization.reconstruction_pretrain_epochs
        if phase == PHASE_MDVSC_BOOTSTRAP:
            return self.config.optimization.mdvsc_bootstrap_epochs
        return self.config.optimization.epochs

    def _steps_per_epoch(self, dataloader: DataLoader) -> int:
        if self.config.optimization.max_steps_per_epoch is None:
            return len(dataloader)
        return min(len(dataloader), self.config.optimization.max_steps_per_epoch)

    def _summarize_parameter_counts(self) -> dict[str, int]:
        counts = {
            "total": sum(parameter.numel() for parameter in self.model.parameters()),
            "level_modules": sum(parameter.numel() for parameter in self.model.level_modules.parameters()),
            "reconstruction_head": sum(parameter.numel() for parameter in self.model.reconstruction_head.parameters()),
            "detection_refinement_heads": sum(
                parameter.numel() for parameter in self.model.detection_refinement_heads.parameters()
            ),
            "reconstruction_refinement_heads": sum(
                parameter.numel() for parameter in self.model.reconstruction_refinement_heads.parameters()
            ),
        }
        counts["trainable_at_init"] = sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)
        return counts

    def _load_initialization(self) -> None:
        initialization = self.config.initialization
        if initialization.full_checkpoint:
            state_dict = self._load_model_state(initialization.full_checkpoint)
            self.model.load_state_dict(state_dict, strict=initialization.strict)
        if initialization.reconstruction_checkpoint:
            self._load_component_checkpoint(
                checkpoint_path=initialization.reconstruction_checkpoint,
                prefixes=["reconstruction_head.", "reconstruction_refinement_heads."],
                strict=initialization.strict,
            )
        if initialization.transmission_checkpoint:
            self._load_component_checkpoint(
                checkpoint_path=initialization.transmission_checkpoint,
                prefixes=["level_modules."],
                strict=initialization.strict,
            )

    def _load_component_checkpoint(self, checkpoint_path: str, prefixes: list[str], strict: bool) -> None:
        state_dict = self._load_model_state(checkpoint_path)
        current_state = self.model.state_dict()
        filtered_state = {
            key: value
            for key, value in state_dict.items()
            if any(key.startswith(prefix) for prefix in prefixes)
            and key in current_state
            and current_state[key].shape == value.shape
        }
        if strict and not filtered_state:
            raise ValueError(f"No matching parameters found when loading {checkpoint_path}")
        self.model.load_state_dict(filtered_state, strict=False)

    @staticmethod
    def _load_model_state(checkpoint_path: str) -> dict[str, torch.Tensor]:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            return checkpoint["model_state"]
        if isinstance(checkpoint, dict):
            return checkpoint
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

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
        original_frames = _to_numpy_float_array(frames[0, :max_frames].permute(0, 2, 3, 1))
        reconstructed_frames = _to_numpy_float_array(
            model_outputs.reconstructed_frames[0, :max_frames].permute(0, 2, 3, 1)
        )

        reconstructed_base_frames = _to_numpy_float_array(
            model_outputs.reconstructed_base_frames[0, :max_frames].permute(0, 2, 3, 1)
        )
        high_frequency_maps = _to_numpy_float_array(
            model_outputs.reconstructed_high_frequency_residuals[0, :max_frames].abs().mean(dim=1)
        )

        figure, axes = plt.subplots(4, max_frames, figsize=(4 * max_frames, 12))
        if max_frames == 1:
            axes = np.array([[axes[0]], [axes[1]], [axes[2]], [axes[3]]])
        for frame_index in range(max_frames):
            axes[0, frame_index].imshow(original_frames[frame_index])
            axes[0, frame_index].set_title(f"original t={frame_index}")
            axes[0, frame_index].axis("off")
            axes[1, frame_index].imshow(reconstructed_frames[frame_index])
            axes[1, frame_index].set_title(f"reconstructed t={frame_index}")
            axes[1, frame_index].axis("off")
            axes[2, frame_index].imshow(reconstructed_base_frames[frame_index])
            axes[2, frame_index].set_title(f"base t={frame_index}")
            axes[2, frame_index].axis("off")
            axes[3, frame_index].imshow(high_frequency_maps[frame_index], cmap="magma")
            axes[3, frame_index].set_title(f"high-frequency residual t={frame_index}")
            axes[3, frame_index].axis("off")
        figure.tight_layout()
        figure.savefig(vis_dir / f"epoch_{epoch:03d}_reconstruction.png", dpi=180)
        plt.close(figure)

        num_levels = len(target_sequences)
        figure, axes = plt.subplots(num_levels, 3, figsize=(12, 4 * num_levels))
        if num_levels == 1:
            axes = np.array([axes])
        for level_index in range(num_levels):
            teacher_map = _to_numpy_float_array(target_sequences[level_index][0, 0].mean(dim=0))
            student_map = _to_numpy_float_array(model_outputs.restored_sequences[level_index][0, 0].mean(dim=0))
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
            common_mask = _to_numpy_float_array(model_outputs.common_masks[level_index][0].mean(dim=0))
            individual_mask = _to_numpy_float_array(model_outputs.individual_masks[level_index][0, 0].mean(dim=0))
            axes[level_index, 0].imshow(common_mask, cmap="gray", vmin=0.0, vmax=1.0)
            axes[level_index, 0].set_title(f"level {level_index} common mask")
            axes[level_index, 1].imshow(individual_mask, cmap="gray", vmin=0.0, vmax=1.0)
            axes[level_index, 1].set_title(f"level {level_index} individual mask t=0")
            axes[level_index, 0].set_ylabel(f"level {level_index}")
            axes[level_index, 0].axis("off")
            axes[level_index, 1].axis("off")
        figure.tight_layout()
        figure.savefig(vis_dir / f"epoch_{epoch:03d}_masks.png", dpi=180)
        plt.close(figure)


def run_stage1_training(config: MDVSCStage1TrainConfig) -> dict[str, Any]:
    trainer = Stage1Trainer(config)
    return trainer.run()