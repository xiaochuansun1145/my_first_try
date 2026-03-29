from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.semantic_rtdetr.training.stage1_config import Stage1DataConfig
from src.semantic_rtdetr.training.stage1_data import VideoGOPDataset


class Stage1DataTest(unittest.TestCase):
    def test_imagenet_vid_like_layout_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sequence_dir = root / "Data" / "VID" / "train" / "ILSVRC2015_VID_train_0000" / "ILSVRC2015_train_00000000"
            sequence_dir.mkdir(parents=True)

            for index in range(6):
                image = Image.new("RGB", (32, 32), color=(index * 20, index * 20, index * 20))
                image.save(sequence_dir / f"{index:06d}.JPEG")

            dataset = VideoGOPDataset(
                Stage1DataConfig(
                    dataset_name="imagenet_vid",
                    train_source_path=str(root),
                    recursive=True,
                    gop_size=4,
                    frame_height=32,
                    frame_width=32,
                    frame_stride=1,
                    gop_stride=1,
                    max_sources=1,
                    max_samples=2,
                ),
                str(root),
            )

            sample = dataset[0]
            self.assertEqual(len(dataset), 2)
            self.assertEqual(tuple(sample.shape), (4, 3, 32, 32))

    def test_fractional_subset_can_take_partial_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for sequence_index in range(5):
                sequence_dir = root / f"seq_{sequence_index:02d}"
                sequence_dir.mkdir(parents=True)
                for frame_index in range(5):
                    image = Image.new("RGB", (24, 24), color=(sequence_index * 20, frame_index * 20, 0))
                    image.save(sequence_dir / f"{frame_index:06d}.jpg")

            dataset = VideoGOPDataset(
                Stage1DataConfig(
                    train_source_path=str(root),
                    recursive=True,
                    subset_seed=7,
                    source_fraction=0.4,
                    sample_fraction=0.5,
                    gop_size=4,
                    frame_height=24,
                    frame_width=24,
                    frame_stride=1,
                    gop_stride=1,
                ),
                str(root),
            )

            self.assertGreaterEqual(len(dataset), 1)
            self.assertLess(len(dataset), 10)


if __name__ == "__main__":
    unittest.main()