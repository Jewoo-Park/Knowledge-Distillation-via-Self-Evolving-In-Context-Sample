from pathlib import Path
from typing import Literal
import random
import re
from pydantic import BaseModel, ConfigDict, Field, model_validator
from src.utils.storage import read_jsonl, file_hash, digest


class Sample(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    sample_id: str = Field(min_length=1)
    dataset_name: str = Field(min_length=1)
    split: Literal["train", "validation", "test"]
    video_path: str | None = None
    frame_paths: list[str] = Field(default_factory=list)
    question: str = Field(min_length=1)
    choices: dict[str, str] = Field(default_factory=dict)
    ground_truth: str = Field(min_length=1)
    metadata: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_choices(self):
        if bool(self.video_path) == bool(self.frame_paths):
            raise ValueError("Provide exactly one of video_path or frame_paths")
        if self.frame_paths and (len(self.frame_paths) < 2 or len(set(self.frame_paths)) != len(self.frame_paths)):
            raise ValueError("Frame input requires at least two distinct ordered paths")
        if self.choices:
            if len(self.choices) < 2 or any(not re.fullmatch(r"[A-Z]", k) for k in self.choices):
                raise ValueError("MCQ requires at least two single uppercase option letters")
            if self.ground_truth not in self.choices:
                raise ValueError("Ground truth is not an option")
        return self

    def model_input(self):
        # Labels and scene metadata cannot cross the inference boundary.
        return self.model_dump(exclude={"ground_truth", "metadata", "split", "dataset_name"})


class VideoQADataset:
    def __init__(self, manifest, split):
        self.manifest = Path(manifest).resolve()
        self.samples = [Sample.model_validate(row) for row in read_jsonl(self.manifest)]
        self._validate_samples(split)

    def _validate_samples(self, split):
        if not self.samples or any(s.split != split for s in self.samples):
            raise ValueError(f"Empty or mixed split manifest: {self.manifest}")
        if len({s.sample_id for s in self.samples}) != len(self.samples):
            raise ValueError("Duplicate sample IDs")
        for sample in self.samples:
            paths = [sample.video_path] if sample.video_path else sample.frame_paths
            resolved = []
            for value in paths:
                path = Path(value)
                if not path.is_absolute():
                    path = self.manifest.parent / path
                path = path.resolve()
                if not path.is_file():
                    raise FileNotFoundError(path)
                resolved.append(str(path))
            if sample.video_path:
                sample.video_path = resolved[0]
            else:
                sample.frame_paths = resolved

    def subset(self, count=None, seed=42):
        ordered = sorted(self.samples, key=lambda s: s.sample_id)
        if count is None:
            return ordered
        if count < 1 or count > len(ordered):
            raise ValueError("Subset count must be between 1 and dataset size")
        return sorted(random.Random(seed).sample(ordered, count), key=lambda s: s.sample_id)


def media_hash(sample, cache=None):
    """Hash all supplied media bytes, including frames not selected for decoding."""
    key = (sample.video_path, tuple(sample.frame_paths))
    if cache is not None and key in cache:
        return cache[key]
    value = file_hash(sample.video_path) if sample.video_path else digest(
        {'kind': 'ordered_frames', 'sha256': [file_hash(path) for path in sample.frame_paths]})
    if cache is not None:
        cache[key] = value
    return value


def check_split_overlap(datasets, media_cache=None):
    ids, videos = {}, {}
    media_cache = {} if media_cache is None else media_cache
    for split, samples in datasets.items():
        for sample in samples:
            if sample.split != split:
                raise ValueError("Split mismatch")
            key = sample.sample_id
            video = media_hash(sample, media_cache)
            if key in ids:
                raise ValueError(f"Repeated sample ID: {key}")
            if video in videos and videos[video] != split:
                raise ValueError("Identical video or frame content across splits")
            ids[key], videos[video] = split, split
    return {split: [s.sample_id for s in samples] for split, samples in datasets.items()}
