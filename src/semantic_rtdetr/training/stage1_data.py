from __future__ import annotations

import hashlib
import json
import os
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
import random

import torch
from PIL import Image
from torch.utils.data import Dataset, random_split
from torchvision.transforms.functional import to_tensor

from src.semantic_rtdetr.training.stage1_config import Stage1DataConfig

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
CACHE_VERSION = 1

@dataclass(frozen=True)
class IndexedSource:
    source_path: str
    source_type: str
    frame_entries: tuple[str, ...]
    frame_count: int


@dataclass(frozen=True)
class ActiveSource:
    source_path: str
    source_type: str
    frame_entries: tuple[str, ...]
    frame_count: int
    candidate_sample_count: int
    active_sample_count: int


class VideoGOPDataset(Dataset[torch.Tensor]):
    def __init__(self, config: Stage1DataConfig, source_path: str | None):
        if not source_path:
            raise ValueError("A data source path is required for stage-1 training")

        self.config = config
        self.source_path = Path(source_path)
        self.frame_size = (config.frame_width, config.frame_height)
        self.required_frames = 1 + (config.gop_size - 1) * config.frame_stride
        self.index_cache_path: str | None = None
        self.index_cache_hit = False
        print(
            f"[stage1_data] Building dataset from {self.source_path} "
            f"(dataset={config.dataset_name}, recursive={config.recursive})",
            flush=True,
        )

        indexed_sources = self._load_or_build_index()
        self.total_indexed_sources = len(indexed_sources)
        selected_sources = self._apply_source_fraction(indexed_sources)
        if config.max_sources is not None:
            selected_sources = selected_sources[: config.max_sources]
        self.total_selected_sources = len(selected_sources)

        self.active_sources: list[ActiveSource] = []
        self.sample_prefix_sums: list[int] = []
        self.total_candidate_samples = 0
        running_total = 0
        for source in selected_sources:
            candidate_sample_count = self._count_candidate_samples(source.frame_count)
            if candidate_sample_count <= 0:
                continue

            active_sample_count = self._count_active_samples(candidate_sample_count)
            self.total_candidate_samples += candidate_sample_count
            running_total += active_sample_count
            self.active_sources.append(
                ActiveSource(
                    source_path=source.source_path,
                    source_type=source.source_type,
                    frame_entries=source.frame_entries,
                    frame_count=source.frame_count,
                    candidate_sample_count=candidate_sample_count,
                    active_sample_count=active_sample_count,
                )
            )
            self.sample_prefix_sums.append(running_total)

        if not self.active_sources:
            raise ValueError(f"No valid GOP samples were found under {self.source_path}")

        self.total_usable_sources = len(self.active_sources)
        self.total_active_samples = running_total
        if config.max_samples is not None:
            self.total_active_samples = min(self.total_active_samples, config.max_samples)
            if self.total_active_samples <= 0:
                raise ValueError("max_samples truncated the dataset to zero samples")

        print(
            f"[stage1_data] Indexed {self.total_indexed_sources} sources, selected {len(selected_sources)}, "
            f"usable {len(self.active_sources)}; candidate GOPs={self.total_candidate_samples}, "
            f"active GOPs={self.total_active_samples}",
            flush=True,
        )

    def __len__(self) -> int:
        return self.total_active_samples

    def summary(self) -> dict[str, Any]:
        return {
            "dataset_name": self.config.dataset_name,
            "source_path": str(self.source_path),
            "indexed_sources": self.total_indexed_sources,
            "selected_sources": self.total_selected_sources,
            "usable_sources": self.total_usable_sources,
            "candidate_samples": self.total_candidate_samples,
            "active_samples": self.total_active_samples,
            "index_cache_path": self.index_cache_path,
            "index_cache_hit": self.index_cache_hit,
        }

    def __getitem__(self, index: int) -> torch.Tensor:
        if index < 0 or index >= len(self):
            raise IndexError(index)

        source_index = bisect_right(self.sample_prefix_sums, index)
        source = self.active_sources[source_index]
        previous_prefix = 0 if source_index == 0 else self.sample_prefix_sums[source_index - 1]
        local_sample_index = index - previous_prefix
        raw_position_index = self._map_active_to_candidate_position(source, local_sample_index)
        start_index = raw_position_index * self.config.gop_stride
        frame_tensors: list[torch.Tensor] = []

        for offset in range(self.config.gop_size):
            frame_index = start_index + offset * self.config.frame_stride
            frame = self._read_frame(source, frame_index)
            frame = frame.resize(self.frame_size, Image.BILINEAR)
            frame_tensors.append(to_tensor(frame))

        return torch.stack(frame_tensors, dim=0)

    @staticmethod
    def _resolve_cache_dir(cache_dir: str | None) -> Path | None:
        if cache_dir is None:
            return None

        candidate = Path(cache_dir)
        if candidate.is_absolute():
            return candidate
        repo_root = Path(__file__).resolve().parents[3]
        return repo_root / candidate

    @classmethod
    def _cache_key(cls, root: Path, config: Stage1DataConfig) -> str:
        payload = {
            "version": CACHE_VERSION,
            "dataset_name": config.dataset_name,
            "source_path": str(root.resolve()),
            "recursive": config.recursive,
            "max_frames_per_source": config.max_frames_per_source,
        }
        raw_key = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha1(raw_key).hexdigest()

    def _load_or_build_index(self) -> list[IndexedSource]:
        cache_dir = self._resolve_cache_dir(self.config.index_cache_dir)
        cache_path = None
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / f"{self._cache_key(self.source_path, self.config)}.json"
            self.index_cache_path = str(cache_path)
            if cache_path.exists():
                self.index_cache_hit = True
                print(f"[stage1_data] Loading source index cache: {cache_path}", flush=True)
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                return [
                    IndexedSource(
                        source_path=item["source_path"],
                        source_type=item["source_type"],
                        frame_entries=tuple(item.get("frame_entries") or []),
                        frame_count=int(item["frame_count"]),
                    )
                    for item in cached.get("sources") or []
                ]

        print("[stage1_data] Scanning source tree and building index", flush=True)
        raw_sources = self._collect_sources(
            self.source_path,
            recursive=self.config.recursive,
            dataset_name=self.config.dataset_name,
        )
        indexed_sources = [self._index_source(source) for source in raw_sources]

        if cache_path is not None:
            payload = {
                "version": CACHE_VERSION,
                "dataset_name": self.config.dataset_name,
                "source_path": str(self.source_path.resolve()),
                "recursive": self.config.recursive,
                "max_frames_per_source": self.config.max_frames_per_source,
                "sources": [
                    {
                        "source_path": item.source_path,
                        "source_type": item.source_type,
                        "frame_entries": list(item.frame_entries),
                        "frame_count": item.frame_count,
                    }
                    for item in indexed_sources
                ],
            }
            cache_path.write_text(json.dumps(payload), encoding="utf-8")
            print(f"[stage1_data] Wrote source index cache: {cache_path}", flush=True)

        return indexed_sources

    @classmethod
    def _collect_sources(cls, root: Path, recursive: bool, dataset_name: str) -> list[Path]:
        if root.is_file():
            return [root]

        if not root.is_dir():
            raise FileNotFoundError(f"Data source not found: {root}")

        if not recursive:
            sources: list[Path] = []
            direct_images = [path for path in sorted(root.iterdir()) if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
            if direct_images:
                sources.append(root)
            for path in sorted(root.iterdir()):
                if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                    sources.append(path)
                elif path.is_dir():
                    child_images = [child for child in path.iterdir() if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS]
                    if child_images:
                        sources.append(path)
            return sources

        return cls._walk_sources(root, dataset_name)

    @classmethod
    def _walk_sources(cls, root: Path, dataset_name: str) -> list[Path]:
        sources: list[Path] = []
        for current_root, dir_names, file_names in os.walk(root, topdown=True):
            current_path = Path(current_root)
            image_names = sorted(
                file_name for file_name in file_names if Path(file_name).suffix.lower() in IMAGE_EXTENSIONS
            )
            if image_names:
                sources.append(current_path)
                dir_names[:] = []
                continue

            if dataset_name != "imagenet_vid":
                video_names = sorted(
                    file_name for file_name in file_names if Path(file_name).suffix.lower() in VIDEO_EXTENSIONS
                )
                for file_name in video_names:
                    sources.append(current_path / file_name)
        return sources

    def _index_source(self, source: Path) -> IndexedSource:
        if source.is_dir():
            frame_entries = tuple(
                path.name for path in sorted(source.iterdir()) if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            if self.config.max_frames_per_source is not None:
                frame_entries = frame_entries[: self.config.max_frames_per_source]
            return IndexedSource(
                source_path=str(source.resolve()),
                source_type="image_dir",
                frame_entries=frame_entries,
                frame_count=len(frame_entries),
            )

        frame_count = self._get_video_frame_count(source)
        if self.config.max_frames_per_source is not None:
            frame_count = min(frame_count, self.config.max_frames_per_source)
        return IndexedSource(
            source_path=str(source.resolve()),
            source_type="video_file",
            frame_entries=(),
            frame_count=frame_count,
        )

    def _apply_source_fraction(self, sources: list[IndexedSource]) -> list[IndexedSource]:
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

    def _count_active_samples(self, candidate_sample_count: int) -> int:
        fraction = float(self.config.sample_fraction)
        if fraction <= 0.0 or fraction > 1.0:
            raise ValueError("sample_fraction must be in the range (0, 1]")
        if fraction >= 1.0 or candidate_sample_count <= 1:
            return candidate_sample_count

        return max(1, int(candidate_sample_count * fraction))

    def _count_candidate_samples(self, frame_count: int) -> int:
        if frame_count < self.required_frames:
            return 0

        max_start = frame_count - (self.config.gop_size - 1) * self.config.frame_stride
        return (max_start + self.config.gop_stride - 1) // self.config.gop_stride

    def _map_active_to_candidate_position(self, source: ActiveSource, local_sample_index: int) -> int:
        if source.active_sample_count >= source.candidate_sample_count:
            return local_sample_index

        centered_position = ((2 * local_sample_index + 1) * source.candidate_sample_count) // (2 * source.active_sample_count)
        offset_seed = f"{self.config.subset_seed}:{source.source_path}".encode("utf-8")
        offset = int(hashlib.sha1(offset_seed).hexdigest(), 16) % source.candidate_sample_count
        return (centered_position + offset) % source.candidate_sample_count

    @staticmethod
    def _get_video_frame_count(source: Path) -> int:
        import cv2

        capture = cv2.VideoCapture(str(source))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        return frame_count

    @staticmethod
    def _read_video_frame(source_path: str, frame_index: int) -> Image.Image:
        import cv2

        capture = cv2.VideoCapture(source_path)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = capture.read()
        capture.release()
        if not success:
            raise IndexError(f"Could not read frame {frame_index} from {source_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame)

    @staticmethod
    def _read_frame(source: ActiveSource, frame_index: int) -> Image.Image:
        if source.source_type == "image_dir":
            frame_name = source.frame_entries[frame_index]
            return Image.open(Path(source.source_path) / frame_name).convert("RGB")
        if source.source_type == "video_file":
            return VideoGOPDataset._read_video_frame(source.source_path, frame_index)
        raise TypeError(f"Unsupported source type: {source.source_type!r}")


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