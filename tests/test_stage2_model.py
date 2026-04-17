from __future__ import annotations

import unittest

import torch

from src.semantic_rtdetr.semantic_comm.stage2_model import (
    DetRecoveryHead,
    SharedEncoder,
    Stage2MDVSC,
    Stage2Output,
)


class SharedEncoderTest(unittest.TestCase):
    def test_output_shapes(self) -> None:
        encoder = SharedEncoder(backbone_channels=[32, 64, 128], shared_channels=16)
        features = [
            torch.randn(4, 32, 8, 8),
            torch.randn(4, 64, 4, 4),
            torch.randn(4, 128, 2, 2),
        ]
        outputs = encoder(features)
        self.assertEqual(len(outputs), 3)
        self.assertEqual(outputs[0].shape, (4, 16, 8, 8))
        self.assertEqual(outputs[1].shape, (4, 16, 4, 4))
        self.assertEqual(outputs[2].shape, (4, 16, 2, 2))

    def test_single_level(self) -> None:
        encoder = SharedEncoder(backbone_channels=[64], shared_channels=32)
        features = [torch.randn(2, 64, 4, 4)]
        outputs = encoder(features)
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].shape, (2, 32, 4, 4))


class DetRecoveryHeadTest(unittest.TestCase):
    def test_output_shapes(self) -> None:
        head = DetRecoveryHead(channels=16, num_levels=3)
        features = [
            torch.randn(4, 16, 8, 8),
            torch.randn(4, 16, 4, 4),
            torch.randn(4, 16, 2, 2),
        ]
        outputs = head(features)
        self.assertEqual(len(outputs), 3)
        for inp, out in zip(features, outputs):
            self.assertEqual(inp.shape, out.shape)

    def test_no_bias_in_conv(self) -> None:
        head = DetRecoveryHead(channels=16, num_levels=2)
        for level_head in head.heads:
            conv = level_head[0]
            self.assertIsNone(conv.bias)


class Stage2MDVSCTest(unittest.TestCase):
    def _make_model(self, **kwargs):
        defaults = dict(
            backbone_channels=[32, 64, 128],
            shared_channels=16,
            reconstruction_hidden_channels=16,
            reconstruction_detail_channels=16,
        )
        defaults.update(kwargs)
        return Stage2MDVSC(**defaults)

    def _make_sequences(self, batch=2, time=3):
        return [
            torch.randn(batch, time, 32, 8, 8),
            torch.randn(batch, time, 64, 4, 4),
            torch.randn(batch, time, 128, 2, 2),
        ]

    def test_forward_output_shapes(self) -> None:
        model = self._make_model()
        sequences = self._make_sequences(batch=2, time=3)
        output = model(sequences, output_size=(64, 64))

        self.assertIsInstance(output, Stage2Output)
        # Shared sequences: [B, T, shared_ch, H, W]
        self.assertEqual(output.shared_sequences[0].shape, (2, 3, 16, 8, 8))
        self.assertEqual(output.shared_sequences[1].shape, (2, 3, 16, 4, 4))
        self.assertEqual(output.shared_sequences[2].shape, (2, 3, 16, 2, 2))
        # Det recovery: same shape as shared
        for s, d in zip(output.shared_sequences, output.det_recovery_sequences):
            self.assertEqual(s.shape, d.shape)
        # Reconstruction: same shape as shared
        for s, r in zip(output.shared_sequences, output.reconstruction_sequences):
            self.assertEqual(s.shape, r.shape)
        # Reconstructed frames: [B, T, 3, H, W]
        self.assertEqual(output.reconstructed_frames.shape, (2, 3, 3, 64, 64))
        self.assertEqual(output.reconstructed_base_frames.shape, (2, 3, 3, 64, 64))
        self.assertEqual(output.reconstructed_high_frequency_residuals.shape, (2, 3, 3, 64, 64))

    def test_backward_pass(self) -> None:
        model = self._make_model()
        sequences = self._make_sequences(batch=1, time=2)
        output = model(sequences, output_size=(64, 64))

        loss = output.reconstructed_frames.mean() + output.det_recovery_sequences[0].mean()
        loss.backward()

        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        self.assertTrue(has_grad, "Expected at least one parameter to have a non-zero gradient")

    def test_wrong_level_count_raises(self) -> None:
        model = self._make_model()
        bad_sequences = [torch.randn(1, 2, 32, 8, 8)]  # only 1 level instead of 3
        with self.assertRaises(ValueError):
            model(bad_sequences, output_size=(64, 64))

    def test_standard_reconstruction_head(self) -> None:
        model = self._make_model(reconstruction_head_type="standard")
        sequences = self._make_sequences(batch=1, time=2)
        output = model(sequences, output_size=(64, 64))
        self.assertEqual(output.reconstructed_frames.shape, (1, 2, 3, 64, 64))

    def test_reconstructed_frames_in_valid_range(self) -> None:
        model = self._make_model()
        sequences = self._make_sequences(batch=1, time=2)
        output = model(sequences, output_size=(64, 64))
        self.assertTrue(output.reconstructed_frames.min() >= 0.0)
        self.assertTrue(output.reconstructed_frames.max() <= 1.0)

    def test_det_recovery_head_frozen_in_recon_pretrain(self) -> None:
        """Verify that DetRecoveryHead params can be frozen independently."""
        model = self._make_model()
        # Freeze det_recovery_head
        for p in model.det_recovery_head.parameters():
            p.requires_grad = False
        model.det_recovery_head.eval()

        sequences = self._make_sequences(batch=1, time=2)
        output = model(sequences, output_size=(64, 64))
        loss = output.reconstructed_frames.mean()
        loss.backward()

        # SharedEncoder should have gradients
        has_encoder_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.shared_encoder.parameters()
        )
        self.assertTrue(has_encoder_grad)

        # DetRecoveryHead should NOT have gradients
        for p in model.det_recovery_head.parameters():
            self.assertIsNone(p.grad)

    def test_parameter_count_reasonable(self) -> None:
        """Stage2MDVSC with real backbone channels should be compact."""
        model = Stage2MDVSC(
            backbone_channels=[512, 1024, 2048],
            shared_channels=256,
            reconstruction_hidden_channels=160,
            reconstruction_detail_channels=64,
        )
        total = sum(p.numel() for p in model.parameters())
        # SharedEncoder + DetRecovery + ReconRefinement + ReconHead
        # Should be well under 5M params
        self.assertLess(total, 5_000_000, f"Model has {total} params, expected < 5M")
        self.assertGreater(total, 100_000, f"Model has {total} params, expected > 100K")


if __name__ == "__main__":
    unittest.main()
