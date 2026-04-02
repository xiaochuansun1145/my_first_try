from __future__ import annotations

import unittest

import numpy as np
import torch

from src.semantic_rtdetr.training.stage1_trainer import (
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


if __name__ == "__main__":
    unittest.main()