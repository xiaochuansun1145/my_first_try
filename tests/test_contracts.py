from __future__ import annotations

import unittest

import torch

from src.semantic_rtdetr.contracts import EncoderFeatureBundle


class EncoderFeatureBundleTest(unittest.TestCase):
    def test_contract_tracks_multiscale_shapes(self) -> None:
        bundle = EncoderFeatureBundle(
            feature_maps=[
                torch.zeros(1, 256, 80, 80),
                torch.zeros(1, 256, 40, 40),
                torch.zeros(1, 256, 20, 20),
            ],
            spatial_shapes=torch.tensor([[80, 80], [40, 40], [20, 20]]),
            level_start_index=torch.tensor([0, 6400, 8000]),
            strides=[8, 16, 32],
        )

        contract = bundle.contract()

        self.assertEqual(contract.flattened_sequence_length, 8400)
        self.assertEqual(contract.levels[0].stride, 8)
        self.assertEqual(contract.levels[1].height, 40)
        self.assertEqual(contract.level_start_index, [0, 6400, 8000])


if __name__ == "__main__":
    unittest.main()