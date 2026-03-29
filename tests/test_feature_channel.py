from __future__ import annotations

import unittest

import torch

from src.semantic_rtdetr.config import ChannelConfig
from src.semantic_rtdetr.contracts import EncoderFeatureBundle
from src.semantic_rtdetr.semantic_comm.channel import build_feature_channel
from src.semantic_rtdetr.semantic_comm.codec import build_feature_semantic_codec
from src.semantic_rtdetr.config import SemComConfig


class FeatureChannelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle = EncoderFeatureBundle(
            feature_maps=[
                torch.ones(1, 4, 8, 8),
                torch.ones(1, 4, 4, 4) * 2.0,
            ],
            spatial_shapes=torch.tensor([[8, 8], [4, 4]]),
            level_start_index=torch.tensor([0, 64]),
            strides=[8, 16],
        )
        self.image_size = (64, 64)

    def test_identity_channel_preserves_feature_values(self) -> None:
        codec = build_feature_semantic_codec(SemComConfig(selected_levels=[0, 1]))
        packet = codec.encode(self.bundle)
        channel = build_feature_channel(ChannelConfig(mode="identity"))

        result = channel.transmit(packet, self.image_size)

        self.assertTrue(torch.equal(result.received_packet.feature_bundle.feature_maps[0], self.bundle.feature_maps[0]))
        self.assertEqual(result.metrics.feature_mse, 0.0)
        self.assertEqual(result.metrics.channel_mode, "identity")

    def test_awgn_channel_keeps_shapes_and_injects_noise(self) -> None:
        codec = build_feature_semantic_codec(SemComConfig(selected_levels=[0, 1]))
        packet = codec.encode(self.bundle)
        channel = build_feature_channel(ChannelConfig(mode="awgn", snr_db=10.0, seed=7))

        result = channel.transmit(packet, self.image_size)

        self.assertEqual(result.received_packet.feature_bundle.feature_maps[0].shape, self.bundle.feature_maps[0].shape)
        self.assertEqual(result.received_packet.feature_bundle.level_start_index.tolist(), [0, 64])
        self.assertGreater(result.metrics.feature_mse, 0.0)
        self.assertEqual(result.metrics.channel_mode, "awgn")

    def test_codec_can_select_subset_levels_and_restore_full_bundle(self) -> None:
        codec = build_feature_semantic_codec(SemComConfig(selected_levels=[1]))

        packet = codec.encode(self.bundle)
        restored = codec.decode(packet, self.bundle)

        self.assertEqual(packet.selected_levels, [1])
        self.assertEqual(packet.bypassed_levels, [0])
        self.assertEqual(len(packet.feature_bundle.feature_maps), 1)
        self.assertTrue(torch.equal(restored.feature_maps[0], self.bundle.feature_maps[0]))
        self.assertTrue(torch.equal(restored.feature_maps[1], self.bundle.feature_maps[1]))


if __name__ == "__main__":
    unittest.main()