from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from src.semantic_rtdetr.training.stage1_trainer import (
    PHASE_JOINT_TRAINING,
    PHASE_MDVSC_BOOTSTRAP,
    PHASE_RECONSTRUCTION_PRETRAIN,
    Stage1Trainer,
    _ensure_finite_tensor,
    _gradient_edge_loss,
    _resolve_amp_dtype,
    _ssim_loss,
    _to_numpy_float_array,
)


class Stage1TrainerUtilityTest(unittest.TestCase):
    def test_ssim_loss_is_zero_for_identical_inputs(self) -> None:
        frames = torch.rand(2, 3, 3, 32, 32)
        loss = _ssim_loss(frames, frames)
        self.assertLess(float(loss.item()), 1e-6)

    def test_ssim_loss_increases_for_shifted_inputs(self) -> None:
        frames = torch.rand(1, 2, 3, 32, 32)
        shifted = (frames * 0.5).clamp(0.0, 1.0)
        loss = _ssim_loss(frames, shifted)
        self.assertGreater(float(loss.item()), 0.01)

    def test_gradient_edge_loss_is_zero_for_identical_inputs(self) -> None:
        frames = torch.rand(1, 2, 3, 32, 32)
        loss = _gradient_edge_loss(frames, frames)
        self.assertLess(float(loss.item()), 1e-6)

    def test_gradient_edge_loss_increases_for_blurred_inputs(self) -> None:
        frames = torch.rand(1, 2, 3, 32, 32)
        blurred = torch.nn.functional.avg_pool2d(
            frames.view(-1, 3, 32, 32),
            kernel_size=3,
            stride=1,
            padding=1,
        ).view_as(frames)
        loss = _gradient_edge_loss(frames, blurred)
        self.assertGreater(float(loss.item()), 0.01)

    def test_resolve_amp_dtype_supports_float16_and_bfloat16(self) -> None:
        self.assertIs(_resolve_amp_dtype("float16"), torch.float16)
        self.assertIs(_resolve_amp_dtype("bfloat16"), torch.bfloat16)

    def test_resolve_amp_dtype_rejects_unknown_value(self) -> None:
        with self.assertRaises(ValueError):
            _resolve_amp_dtype("float32")

    def test_to_numpy_float_array_promotes_half_precision(self) -> None:
        tensor = torch.rand(2, 3, dtype=torch.float16)
        array = _to_numpy_float_array(tensor)
        self.assertEqual(array.dtype, np.float32)

    def test_ensure_finite_tensor_rejects_nan(self) -> None:
        with self.assertRaises(ValueError):
            _ensure_finite_tensor("nan_tensor", torch.tensor([1.0, float("nan")]))

    def test_phase_epochs_return_configured_values(self) -> None:
        trainer = Stage1Trainer.__new__(Stage1Trainer)
        trainer.config = SimpleNamespace(
            optimization=SimpleNamespace(
                reconstruction_pretrain_epochs=3,
                mdvsc_bootstrap_epochs=5,
                epochs=7,
            )
        )

        self.assertEqual(trainer._phase_epochs(PHASE_RECONSTRUCTION_PRETRAIN), 3)
        self.assertEqual(trainer._phase_epochs(PHASE_MDVSC_BOOTSTRAP), 5)
        self.assertEqual(trainer._phase_epochs(PHASE_JOINT_TRAINING), 7)

    def test_steps_per_epoch_honors_max_steps(self) -> None:
        trainer = Stage1Trainer.__new__(Stage1Trainer)
        trainer.config = SimpleNamespace(optimization=SimpleNamespace(max_steps_per_epoch=3))

        self.assertEqual(trainer._steps_per_epoch([1, 2, 3, 4, 5]), 3)

        trainer.config = SimpleNamespace(optimization=SimpleNamespace(max_steps_per_epoch=None))
        self.assertEqual(trainer._steps_per_epoch([1, 2, 3, 4, 5]), 5)

    def test_build_scheduler_returns_onecycle_per_batch_flag(self) -> None:
        trainer = Stage1Trainer.__new__(Stage1Trainer)
        trainer.config = SimpleNamespace(
            optimization=SimpleNamespace(
                scheduler="onecycle",
                onecycle_pct_start=0.2,
                onecycle_div_factor=10.0,
                onecycle_final_div_factor=100.0,
                warmup_epochs=0,
                warmup_start_factor=0.2,
                min_lr_ratio=0.1,
            )
        )
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = AdamW([parameter], lr=1.0e-3)

        scheduler, per_batch = trainer._build_scheduler(
            optimizer=optimizer,
            epochs=4,
            steps_per_epoch=6,
            max_lr=1.0e-3,
        )

        self.assertIsInstance(scheduler, OneCycleLR)
        self.assertTrue(per_batch)


if __name__ == "__main__":
    unittest.main()