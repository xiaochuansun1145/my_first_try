from __future__ import annotations

import unittest

import torch

from src.semantic_rtdetr.training.stage1_trainer import _gradient_edge_loss, _ssim_loss


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


if __name__ == "__main__":
    unittest.main()