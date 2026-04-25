from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import urllib.request

import pandas as pd


class VocabLoader:
    REPO_BASE = "https://raw.githubusercontent.com/Jonathangadeaharder/IdeaProjects/master/src/backend/data"

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or Path(tempfile.gettempdir()) / "mlx_subtitler_vocab"

    def _build_url(self, level: str) -> str:
        return f"{self.REPO_BASE}/{level}_vokabeln.csv"

    def _download(self, level: str) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        dest = self.cache_dir / f"{level}_vokabeln.csv"
        if not dest.exists():
            urllib.request.urlretrieve(self._build_url(level), str(dest))
        return dest

    def load(self, levels: list[str] | None = None) -> set[str]:
        if levels is None:
            levels = ["B1"]

        vocab: set[str] = set()
        for level in levels:
            path = self._download(level)
            df = pd.read_csv(path, header=None)
            words = df.iloc[:, 0].astype(str).str.strip().str.lower().tolist()
            vocab.update(words)
        return vocab
