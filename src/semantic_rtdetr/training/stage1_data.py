from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, random_split
from torchvision.transforms.functional import to_tensor

from src.semantic_rtdetr.training.stage1_config import Stage1DataConfig

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@dataclass(frozen=True)
class GOPSample:
    source_index: int
    start_index: int


class VideoGOPDataset(Dataset[torch.Tensor]):
    def __init__(self, config: Stage1DataConfig, source_path: str | None):
        if not source_path:
            raise ValueError("A data source path is required for stage-1 training")

        self.config = config
        self.source_path = Path(source_path)
        self.frame_size = (config.frame_width, config.frame_height)
        self.sources = self._collect_sources(self.source_path, recursive=config.recursive)
        self.sources = self._apply_source_fraction(self.sources)
        if config.max_sources is not None:
            self.sources = self.sources[: config.max_sources]
        self.source_frames: list[list[Any]] = []
        self.samples: list[GOPSample] = []

        for source in self.sources:
            frames = self._load_source_frames(source)
            if config.max_frames_per_source is not None:
                frames = frames[: config.max_frames_per_source]

            required_frames = 1 + (config.gop_size - 1) * config.frame_stride
            if len(frames) < required_frames:
                continue

            source_index = len(self.source_frames)
            self.source_frames.append(frames)
            max_start = len(frames) - (config.gop_size - 1) * config.frame_stride
            for start_index in range(0, max_start, config.gop_stride):
                end_index = start_index + (config.gop_size - 1) * config.frame_stride
                if end_index < len(frames):
                    self.samples.append(GOPSample(source_index=source_index, start_index=start_index))

        if not self.samples:
            raise ValueError(f"No valid GOP samples were found under {self.source_path}")

        self.samples = self._apply_sample_fraction(self.samples)

        if config.max_samples is not None:
            self.samples = self.samples[: config.max_samples]
            if not self.samples:
                raise ValueError("max_samples truncated the dataset to zero samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> torch.Tensor:
        sample = self.samples[index]
        source_frames = self.source_frames[sample.source_index]
        frame_tensors: list[torch.Tensor] = []

        for offset in range(self.config.gop_size):
            frame_index = sample.start_index + offset * self.config.frame_stride
            frame = self._read_frame(source_frames[frame_index])
            frame = frame.resize(self.frame_size, Image.BILINEAR)
            frame_tensors.append(to_tensor(frame))

        return torch.stack(frame_tensors, dim=0)

    @staticmethod
    def _collect_sources(root: Path, recursive: bool) -> list[Path]:
        if root.is_file():
            return [root]

        if not root.is_dir():
            raise FileNotFoundError(f"Data source not found: {root}")

        sources: list[Path] = []
        direct_images = [path for path in sorted(root.iterdir()) if path.suffix.lower() in IMAGE_EXTENSIONS]
        if direct_images:
            sources.append(root)

        iterator = root.rglob("*") if recursive else root.iterdir()
        for path in sorted(iterator):
            if path == root:
                continue
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                sources.append(path)
            elif path.is_dir():
                child_images = [child for child in path.iterdir() if child.suffix.lower() in IMAGE_EXTENSIONS]
                if child_images:
                    sources.append(path)

        unique_sources: list[Path] = []
        seen: set[Path] = set()
        for source in sources:
            if source not in seen:
                unique_sources.append(source)
                seen.add(source)
        return unique_sources

    def _apply_source_fraction(self, sources: list[Path]) -> list[Path]:
        fraction = float(self.config.source_fraction)
        if fraction <= 0.0 or fraction > 1.0:
            raise ValueError("source_fraction must be in the range (0, 1]")
        if fraction >= 1.0 or len(sources) <= 1:
            return sources

        count = max(1, int(len(sources) * fraction))
        indices = list(range(len(sources)))
        random.Random(self.config.subset_seed).shuffle(indices)
        selected_indices = sorted(indices[:count])
        return [sources[index] for index in selected_indices]

    def _apply_sample_fraction(self, samples: list[GOPSample]) -> list[GOPSample]:
        fraction = float(self.config.sample_fraction)
        if fraction <= 0.0 or fraction > 1.0:
            raise ValueError("sample_fraction must be in the range (0, 1]")
        if fraction >= 1.0 or len(samples) <= 1:
            return samples

        count = max(1, int(len(samples) * fraction))
        indices = list(range(len(samples)))
        random.Random(self.config.subset_seed).shuffle(indices)
        selected_indices = sorted(indices[:count])
        return [samples[index] for index in selected_indices]

    @staticmethod
    def _load_source_frames(source: Path) -> list[Any]:
        if source.is_dir():
            return [path for path in sorted(source.iterdir()) if path.suffix.lower() in IMAGE_EXTENSIONS]

        if source.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported source format: {source}")

        import cv2

        capture = cv2.VideoCapture(str(source))
        frames: list[np.ndarray] = []
        while True:
            success, frame = capture.read()
            if not success:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        capture.release()
        return frames

    @staticmethod
    def _read_frame(frame_entry: Any) -> Image.Image:
        if isinstance(frame_entry, Path):
            return Image.open(frame_entry).convert("RGB")
        if isinstance(frame_entry, np.ndarray):
            return Image.fromarray(frame_entry)
        raise TypeError(f"Unsupported frame entry type: {type(frame_entry)!r}")


def build_train_val_datasets(
    config: Stage1DataConfig,
    seed: int,
) -> tuple[Dataset[torch.Tensor], Dataset[torch.Tensor] | None]:
    train_dataset = VideoGOPDataset(config, config.train_source_path)

    if config.val_source_path:
        val_dataset = VideoGOPDataset(config, config.val_source_path)
        return train_dataset, val_dataset

    if config.train_val_split <= 0.0 or len(train_dataset) < 2:
        return train_dataset, None

    val_size = max(1, int(len(train_dataset) * config.train_val_split))
    train_size = len(train_dataset) - val_size
    if train_size <= 0:
        return train_dataset, None

    split_generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size], generator=split_generator)
    return train_subset, val_subset