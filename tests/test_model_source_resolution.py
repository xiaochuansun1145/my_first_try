from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.semantic_rtdetr.detector.rtdetr_baseline import RTDetrBaseline


class ModelSourceResolutionTest(unittest.TestCase):
    def test_local_model_directory_takes_priority(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            resolved = RTDetrBaseline._resolve_model_source("PekingU/rtdetr_r50vd", model_dir)
            self.assertEqual(resolved, str(model_dir))

    def test_missing_local_directory_falls_back_to_repo_name(self) -> None:
        resolved = RTDetrBaseline._resolve_model_source(
            "PekingU/rtdetr_r50vd",
            "pretrained/does_not_exist",
        )
        self.assertEqual(resolved, "PekingU/rtdetr_r50vd")


if __name__ == "__main__":
    unittest.main()