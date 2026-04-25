from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mlx_subtitler.models import Segment
from mlx_subtitler.vocab_loader import VocabLoader


def test_load_downloads_and_parses(tmp_path):
    csv_content = "hallo\nwelt\ntest\n"
    cache = tmp_path / "cache"
    cache.mkdir()

    with patch.object(VocabLoader, "_download") as mock_dl:
        csv_path = cache / "B1_vokabeln.csv"
        csv_path.write_text(csv_content)
        mock_dl.return_value = csv_path

        loader = VocabLoader(cache_dir=cache)
        vocab = loader.load(["B1"])

    assert "hallo" in vocab
    assert "welt" in vocab
    assert "test" in vocab


def test_load_uses_cache(tmp_path):
    csv_path = tmp_path / "B1_vokabeln.csv"
    csv_path.write_text("hallo\nwelt\n")

    loader = VocabLoader(cache_dir=tmp_path)
    vocab = loader.load(["B1"])
    assert "hallo" in vocab


def test_load_empty_levels(tmp_path):
    loader = VocabLoader(cache_dir=tmp_path)
    vocab = loader.load([])
    assert vocab == set()


def test_load_handles_failure(tmp_path):
    with patch("mlx_subtitler.vocab_loader.urllib.request.urlretrieve", side_effect=Exception("network error")):
        loader = VocabLoader(cache_dir=tmp_path)
        with pytest.raises(Exception, match="network error"):
            loader.load(["B1"])


def test_build_url_format():
    loader = VocabLoader()
    url = loader._build_url("B2")
    assert url == "https://raw.githubusercontent.com/Jonathangadeaharder/IdeaProjects/master/src/backend/data/B2_vokabeln.csv"


def test_load_with_none_levels_uses_default(tmp_path):
    csv_content = "Haus\nAuto\n"

    def mock_urlretrieve(url, path):
        Path(path).write_text(csv_content)

    with patch("mlx_subtitler.vocab_loader.urllib.request.urlretrieve", side_effect=mock_urlretrieve):
        loader = VocabLoader(cache_dir=tmp_path / "vocab")
        vocab = loader.load(None)
    assert "haus" in vocab
    assert "auto" in vocab
