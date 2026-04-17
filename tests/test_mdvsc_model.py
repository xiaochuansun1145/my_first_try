from __future__ import annotations

import unittest

import torch

from src.semantic_rtdetr.semantic_comm.mdvsc import ProjectMDVSC


class ProjectMDVSCTest(unittest.TestCase):
    def test_reconstruction_head_direct_path_returns_expected_shape(self) -> None:
        model = ProjectMDVSC(
            feature_channels=[16, 16, 16],
            latent_dims=[8, 8, 8],
            common_keep_ratios=[0.5, 0.5, 0.5],
            individual_keep_ratios=[0.25, 0.25, 0.25],
            block_sizes=[4, 2, 1],
        )
        feature_sequences = [
            torch.randn(2, 3, 16, 8, 8),
            torch.randn(2, 3, 16, 4, 4),
            torch.randn(2, 3, 16, 2, 2),
        ]

        outputs = model.reconstruct_from_feature_sequences(feature_sequences, output_size=(64, 64))

        self.assertEqual(outputs.reconstructed_frames.shape, (2, 3, 3, 64, 64))
        self.assertEqual(outputs.restored_sequences[0].shape, feature_sequences[0].shape)
        self.assertEqual(outputs.detection_sequences[0].shape, feature_sequences[0].shape)
        self.assertEqual(outputs.reconstruction_sequences[0].shape, feature_sequences[0].shape)
        self.assertEqual(outputs.reconstructed_base_frames.shape, (2, 3, 3, 64, 64))
        self.assertEqual(outputs.reconstructed_high_frequency_residuals.shape, (2, 3, 3, 64, 64))

    def test_forward_preserves_multiscale_shapes(self) -> None:
        model = ProjectMDVSC(
            feature_channels=[16, 16, 16],
            latent_dims=[8, 8, 8],
            common_keep_ratios=[0.5, 0.5, 0.5],
            individual_keep_ratios=[0.25, 0.25, 0.25],
            block_sizes=[4, 2, 1],
        )
        inputs = [
            torch.randn(2, 3, 16, 8, 8),
            torch.randn(2, 3, 16, 4, 4),
            torch.randn(2, 3, 16, 2, 2),
        ]

        outputs = model(inputs, output_size=(64, 64), apply_masks=False, channel_mode="identity")

        self.assertEqual(len(outputs.restored_sequences), 3)
        self.assertEqual(outputs.restored_sequences[0].shape, inputs[0].shape)
        self.assertEqual(outputs.restored_sequences[1].shape, inputs[1].shape)
        self.assertEqual(outputs.restored_sequences[2].shape, inputs[2].shape)
        self.assertEqual(outputs.detection_sequences[0].shape, inputs[0].shape)
        self.assertEqual(outputs.reconstruction_sequences[0].shape, inputs[0].shape)
        self.assertEqual(outputs.reconstructed_frames.shape, (2, 3, 3, 64, 64))

    def test_mask_stats_stay_in_unit_interval(self) -> None:
        model = ProjectMDVSC(
            feature_channels=[8, 8, 8],
            latent_dims=[4, 4, 4],
            common_keep_ratios=[0.5, 0.5, 0.5],
            individual_keep_ratios=[0.25, 0.25, 0.25],
            block_sizes=[2, 2, 1],
        )
        inputs = [
            torch.randn(1, 2, 8, 4, 4),
            torch.randn(1, 2, 8, 2, 2),
            torch.randn(1, 2, 8, 1, 1),
        ]

        outputs = model(inputs, output_size=(32, 32), apply_masks=True, channel_mode="identity")

        for stat in outputs.level_stats:
            self.assertGreaterEqual(stat.common_active_ratio, 0.0)
            self.assertLessEqual(stat.common_active_ratio, 1.0)
            self.assertGreaterEqual(stat.individual_active_ratio, 0.0)
            self.assertLessEqual(stat.individual_active_ratio, 1.0)

    def test_masks_use_channelwise_block_layout(self) -> None:
        model = ProjectMDVSC(
            feature_channels=[8, 8, 8],
            latent_dims=[4, 4, 4],
            common_keep_ratios=[0.5, 0.5, 0.5],
            individual_keep_ratios=[0.25, 0.25, 0.25],
            block_sizes=[2, 2, 1],
        )
        inputs = [
            torch.randn(1, 2, 8, 4, 4),
            torch.randn(1, 2, 8, 2, 2),
            torch.randn(1, 2, 8, 1, 1),
        ]

        outputs = model(inputs, output_size=(32, 32), apply_masks=True, channel_mode="identity")

        self.assertEqual(outputs.common_masks[0].shape, (1, 4, 4, 4))
        self.assertEqual(outputs.individual_masks[0].shape, (1, 2, 4, 4, 4))
        common_level0 = outputs.common_masks[0][0, 0]
        individual_level0 = outputs.individual_masks[0][0, 0, 0]
        self.assertTrue(torch.equal(common_level0[0::2, 0::2], common_level0[1::2, 1::2]))
        self.assertTrue(torch.equal(individual_level0[0::2, 0::2], individual_level0[1::2, 1::2]))

    def test_light_reconstruction_head_output_shape(self) -> None:
        model = ProjectMDVSC(
            feature_channels=[16, 16, 16],
            latent_dims=[8, 8, 8],
            common_keep_ratios=[0.5, 0.5, 0.5],
            individual_keep_ratios=[0.25, 0.25, 0.25],
            block_sizes=[4, 2, 1],
            reconstruction_head_type="light",
        )
        feature_sequences = [
            torch.randn(1, 2, 16, 8, 8),
            torch.randn(1, 2, 16, 4, 4),
            torch.randn(1, 2, 16, 2, 2),
        ]
        outputs = model.reconstruct_from_feature_sequences(feature_sequences, output_size=(64, 64))
        self.assertEqual(outputs.reconstructed_frames.shape, (1, 2, 3, 64, 64))
        self.assertEqual(outputs.reconstructed_base_frames.shape, (1, 2, 3, 64, 64))
        self.assertEqual(outputs.reconstructed_high_frequency_residuals.shape, (1, 2, 3, 64, 64))

    def test_standard_reconstruction_head_still_works(self) -> None:
        model = ProjectMDVSC(
            feature_channels=[16, 16, 16],
            latent_dims=[8, 8, 8],
            common_keep_ratios=[0.5, 0.5, 0.5],
            individual_keep_ratios=[0.25, 0.25, 0.25],
            block_sizes=[4, 2, 1],
            reconstruction_head_type="standard",
        )
        feature_sequences = [
            torch.randn(1, 2, 16, 8, 8),
            torch.randn(1, 2, 16, 4, 4),
            torch.randn(1, 2, 16, 2, 2),
        ]
        outputs = model.reconstruct_from_feature_sequences(feature_sequences, output_size=(64, 64))
        self.assertEqual(outputs.reconstructed_frames.shape, (1, 2, 3, 64, 64))

    def test_light_head_backward_pass(self) -> None:
        model = ProjectMDVSC(
            feature_channels=[16, 16, 16],
            latent_dims=[8, 8, 8],
            common_keep_ratios=[0.5, 0.5, 0.5],
            individual_keep_ratios=[0.25, 0.25, 0.25],
            block_sizes=[4, 2, 1],
            reconstruction_head_type="light",
        )
        feature_sequences = [
            torch.randn(1, 2, 16, 8, 8),
            torch.randn(1, 2, 16, 4, 4),
            torch.randn(1, 2, 16, 2, 2),
        ]
        outputs = model.reconstruct_from_feature_sequences(feature_sequences, output_size=(64, 64))
        loss = outputs.reconstructed_frames.mean()
        loss.backward()
        # Verify hf_scale gradient flows
        self.assertIsNotNone(model.reconstruction_head.hf_scale.grad)

    def test_light_head_with_checkpoint(self) -> None:
        model = ProjectMDVSC(
            feature_channels=[16, 16, 16],
            latent_dims=[8, 8, 8],
            common_keep_ratios=[0.5, 0.5, 0.5],
            individual_keep_ratios=[0.25, 0.25, 0.25],
            block_sizes=[4, 2, 1],
            reconstruction_head_type="light",
            reconstruction_use_checkpoint=True,
        )
        model.train()
        feature_sequences = [
            torch.randn(1, 2, 16, 8, 8),
            torch.randn(1, 2, 16, 4, 4),
            torch.randn(1, 2, 16, 2, 2),
        ]
        outputs = model.reconstruct_from_feature_sequences(feature_sequences, output_size=(64, 64))
        loss = outputs.reconstructed_frames.mean()
        loss.backward()
        self.assertEqual(outputs.reconstructed_frames.shape, (1, 2, 3, 64, 64))


if __name__ == "__main__":
    unittest.main()