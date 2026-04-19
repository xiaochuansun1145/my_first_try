from __future__ import annotations

import unittest

import torch

from src.semantic_rtdetr.semantic_comm.stage2_1_model import (
    DetailAwareLightReconstructionHead,
    DetailCompressor,
    DetailDecompressor,
    Stage2_1MDVSC,
    Stage2_1Output,
)


class DetailCompressorTest(unittest.TestCase):
    def test_output_shape(self) -> None:
        compressor = DetailCompressor(in_channels=256, detail_latent_channels=32, detail_spatial_size=20)
        x = torch.randn(4, 256, 40, 40)
        out = compressor(x)
        self.assertEqual(out.shape, (4, 32, 20, 20))

    def test_small_input(self) -> None:
        compressor = DetailCompressor(in_channels=64, detail_latent_channels=8, detail_spatial_size=5)
        x = torch.randn(2, 64, 10, 10)
        out = compressor(x)
        self.assertEqual(out.shape, (2, 8, 5, 5))

    def test_gradient_flows(self) -> None:
        compressor = DetailCompressor(in_channels=32, detail_latent_channels=8, detail_spatial_size=4)
        x = torch.randn(1, 32, 16, 16, requires_grad=True)
        out = compressor(x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(x.grad.abs().sum() > 0)


class DetailDecompressorTest(unittest.TestCase):
    def test_output_shape(self) -> None:
        decompressor = DetailDecompressor(detail_latent_channels=32, hidden_channels=160)
        packet = torch.randn(4, 32, 5, 5)
        out = decompressor(packet, target_size=(20, 20))
        self.assertEqual(out.shape, (4, 160, 20, 20))

    def test_no_upsample_when_size_matches(self) -> None:
        decompressor = DetailDecompressor(detail_latent_channels=8, hidden_channels=16)
        packet = torch.randn(2, 8, 10, 10)
        out = decompressor(packet, target_size=(10, 10))
        self.assertEqual(out.shape, (2, 16, 10, 10))


class Stage2_1MDVSCTest(unittest.TestCase):
    def _make_model(self, **kwargs):
        defaults = dict(
            backbone_channels=[32, 64, 128],
            shared_channels=16,
            reconstruction_hidden_channels=16,
            reconstruction_detail_channels=16,
            stage1_channels=24,
            detail_latent_channels=8,
            detail_spatial_size=4,
        )
        defaults.update(kwargs)
        return Stage2_1MDVSC(**defaults)

    def _make_inputs(self, batch=2, time=3):
        backbone_sequences = [
            torch.randn(batch, time, 32, 8, 8),
            torch.randn(batch, time, 64, 4, 4),
            torch.randn(batch, time, 128, 2, 2),
        ]
        stage1_sequences = torch.randn(batch, time, 24, 16, 16)
        return backbone_sequences, stage1_sequences

    def test_forward_output_shapes(self) -> None:
        model = self._make_model()
        backbone_seq, stage1_seq = self._make_inputs(batch=2, time=3)
        output = model(backbone_seq, stage1_seq, output_size=(64, 64))

        self.assertIsInstance(output, Stage2_1Output)
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
        # Detail compressed: [B, T, C_latent, Hs, Ws]
        self.assertEqual(output.detail_compressed.shape, (2, 3, 8, 4, 4))

    def test_detail_transmission_ratio(self) -> None:
        model = self._make_model()
        backbone_seq, stage1_seq = self._make_inputs(batch=1, time=2)
        output = model(backbone_seq, stage1_seq, output_size=(64, 64))

        self.assertIsInstance(output.detail_transmission_ratio, float)
        # Ratio should be small (detail packet << main features)
        self.assertGreater(output.detail_transmission_ratio, 0.0)
        self.assertLess(output.detail_transmission_ratio, 0.5)

    def test_backward_pass(self) -> None:
        model = self._make_model()
        backbone_seq, stage1_seq = self._make_inputs(batch=1, time=2)
        output = model(backbone_seq, stage1_seq, output_size=(64, 64))

        loss = output.reconstructed_frames.mean() + output.det_recovery_sequences[0].mean()
        loss.backward()

        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        self.assertTrue(has_grad, "Expected at least one parameter to have a non-zero gradient")

    def test_detail_compressor_receives_gradient(self) -> None:
        model = self._make_model()
        backbone_seq, stage1_seq = self._make_inputs(batch=1, time=2)
        output = model(backbone_seq, stage1_seq, output_size=(64, 64))

        loss = output.reconstructed_frames.mean()
        loss.backward()

        has_compressor_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.detail_compressor.parameters()
        )
        self.assertTrue(has_compressor_grad, "DetailCompressor should receive gradients from reconstruction loss")

    def test_wrong_level_count_raises(self) -> None:
        model = self._make_model()
        bad_sequences = [torch.randn(1, 2, 32, 8, 8)]
        stage1_seq = torch.randn(1, 2, 24, 16, 16)
        with self.assertRaises(ValueError):
            model(bad_sequences, stage1_seq, output_size=(64, 64))

    def test_reconstructed_frames_in_valid_range(self) -> None:
        model = self._make_model()
        backbone_seq, stage1_seq = self._make_inputs(batch=1, time=2)
        output = model(backbone_seq, stage1_seq, output_size=(64, 64))
        self.assertTrue(output.reconstructed_frames.min() >= 0.0)
        self.assertTrue(output.reconstructed_frames.max() <= 1.0)

    def test_parameter_count_reasonable(self) -> None:
        """Stage2_1MDVSC with real backbone channels should be compact."""
        model = Stage2_1MDVSC(
            backbone_channels=[512, 1024, 2048],
            shared_channels=256,
            reconstruction_hidden_channels=160,
            reconstruction_detail_channels=64,
            stage1_channels=256,
            detail_latent_channels=32,
            detail_spatial_size=20,
        )
        total = sum(p.numel() for p in model.parameters())
        # Stage2 ~< 5M + small detail bypass overhead
        self.assertLess(total, 6_000_000, f"Model has {total} params, expected < 6M")
        self.assertGreater(total, 100_000, f"Model has {total} params, expected > 100K")

    def test_only_standard_head_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._make_model(reconstruction_head_type="standard")


if __name__ == "__main__":
    unittest.main()
