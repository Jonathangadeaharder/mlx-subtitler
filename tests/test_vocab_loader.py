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


def test_init_default_cache_dir():
    import tempfile

    vl = VocabLoader()
    assert vl.cache_dir.name == "mlx_subtitler_vocab"
    assert vl.cache_dir.parent == Path(tempfile.gettempdir())


def test_init_custom_cache_dir(tmp_path):
    vl = VocabLoader(cache_dir=tmp_path / "custom")
    assert vl.cache_dir == tmp_path / "custom"


def test_download_creates_dir_with_parents(tmp_path):
    with patch.object(VocabLoader, "_build_url", return_value="http://x"), \
         patch("mlx_subtitler.vocab_loader.urllib.request.urlretrieve"):
        vl = VocabLoader(cache_dir=tmp_path / "sub" / "dir")
        vl._download("B1")
        assert (tmp_path / "sub" / "dir").exists()


def test_download_calls_urlretrieve_with_correct_url(tmp_path):
    with patch.object(VocabLoader, "_build_url", return_value="http://example.com/B1.csv") as mock_url, \
         patch("mlx_subtitler.vocab_loader.urllib.request.urlretrieve") as mock_retrieve:
        cache = tmp_path / "cache"
        cache.mkdir(parents=True)
        vl = VocabLoader(cache_dir=cache)
        vl._download("B1")
        mock_url.assert_called_with("B1")
        mock_retrieve.assert_called_once()
        assert mock_retrieve.call_args[0][0] == "http://example.com/B1.csv"


def test_download_skips_if_file_exists(tmp_path):
    with patch.object(VocabLoader, "_build_url", return_value="http://x"), \
         patch("mlx_subtitler.vocab_loader.urllib.request.urlretrieve") as mock_retrieve:
        cache = tmp_path / "cache"
        cache.mkdir()
        (cache / "B1_vokabeln.csv").write_text("existing")
        vl = VocabLoader(cache_dir=cache)
        vl._download("B1")
        mock_retrieve.assert_not_called()


def test_download_returns_correct_path(tmp_path):
    with patch.object(VocabLoader, "_build_url", return_value="http://x"), \
         patch("mlx_subtitler.vocab_loader.urllib.request.urlretrieve"):
        cache = tmp_path / "cache"
        vl = VocabLoader(cache_dir=cache)
        result = vl._download("B2")
        assert result == cache / "B2_vokabeln.csv"


def test_load_default_levels_calls_download_with_b1(tmp_path):
    with patch.object(VocabLoader, "_download") as mock_dl:
        csv_path = tmp_path / "f.csv"
        csv_path.write_text("hallo\n")
        mock_dl.return_value = csv_path
        vl = VocabLoader(cache_dir=tmp_path)
        vl.load()
        mock_dl.assert_called_with("B1")


def test_load_strips_and_lowercases(tmp_path):
    csv_content = "  Hallo  \nWELT\n"
    csv_path = tmp_path / "B1_vokabeln.csv"
    csv_path.write_text(csv_content)
    loader = VocabLoader(cache_dir=tmp_path)
    vocab = loader.load(["B1"])
    assert "hallo" in vocab
    assert "welt" in vocab


def test_load_multiple_levels(tmp_path):
    with patch.object(VocabLoader, "_download") as mock_dl:
        b1_path = tmp_path / "b1.csv"
        b1_path.write_text("hallo\n")
        b2_path = tmp_path / "b2.csv"
        b2_path.write_text("welt\n")
        mock_dl.side_effect = [b1_path, b2_path]
        vl = VocabLoader(cache_dir=tmp_path)
        vocab = vl.load(["B1", "B2"])
        assert "hallo" in vocab
        assert "welt" in vocab
        assert mock_dl.call_count == 2
        mock_dl.assert_any_call("B1")
        mock_dl.assert_any_call("B2")
