from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from src.semantic_rtdetr.training.stage1_config import load_stage1_config


class Stage1ConfigTest(unittest.TestCase):
    def test_load_stage1_config_reads_initialization_and_optimizer_options(self) -> None:
        yaml_text = textwrap.dedent(
            """
            optimization:
              optimizer: adam
              scheduler: onecycle
              adam_beta2: 0.98

            initialization:
              reconstruction_checkpoint: outputs/reconstruction/best.pt
              transmission_checkpoint: outputs/transmission/best.pt
              strict: true
            """
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "stage1.yaml"
            config_path.write_text(yaml_text, encoding="utf-8")
            config = load_stage1_config(config_path)

        self.assertEqual(config.optimization.optimizer, "adam")
        self.assertEqual(config.optimization.scheduler, "onecycle")
        self.assertAlmostEqual(config.optimization.adam_beta2, 0.98)
        self.assertEqual(config.initialization.reconstruction_checkpoint, "outputs/reconstruction/best.pt")
        self.assertEqual(config.initialization.transmission_checkpoint, "outputs/transmission/best.pt")
        self.assertTrue(config.initialization.strict)


if __name__ == "__main__":
    unittest.main()