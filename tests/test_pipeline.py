from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mlx_subtitler.models import Segment
from mlx_subtitler.pipeline import Pipeline


def _segments():
    return [
        Segment(index=1, start=0.0, end=2.5, text="Hallo"),
        Segment(index=2, start=2.5, end=5.0, text="Welt"),
    ]


def _make_transcriber_mock():
    mock = MagicMock()
    mock.transcribe.return_value = _segments()
    return mock


@patch("mlx_subtitler.pipeline.Transcriber")
def test_transcribe_only(mock_cls, tmp_path):
    mock_cls.return_value = _make_transcriber_mock()

    audio = tmp_path / "test.m4a"
    audio.write_bytes(b"\x00")
    out = tmp_path / "out.srt"

    p = Pipeline(translate=False, filter_vocab=False, output_format="srt")
    result = p.run(audio, out)
    assert result == out
    assert out.exists()


@patch("mlx_subtitler.translator.Translator")
@patch("mlx_subtitler.vocab_filter.VocabFilter")
@patch("mlx_subtitler.vocab_loader.VocabLoader")
@patch("mlx_subtitler.pipeline.Transcriber")
def test_full_with_filter(mock_tc, mock_vl, mock_vf, mock_tr):
    mock_tc.return_value = _make_transcriber_mock()

    mock_loader = MagicMock()
    mock_loader.load.return_value = {"hallo", "welt"}
    mock_vl.return_value = mock_loader

    mock_filter = MagicMock()
    mock_filter.filter.return_value = _segments()
    mock_vf.return_value = mock_filter

    mock_translator = MagicMock()
    mock_translator.translate.return_value = [
        Segment(index=1, start=0.0, end=2.5, text="Hallo", translation="Hello"),
        Segment(index=2, start=2.5, end=5.0, text="Welt", translation="World"),
    ]
    mock_tr.return_value = mock_translator

    with patch("mlx_subtitler.subtitle_writer.SubtitleWriter.write_srt", return_value=Path("out.srt")):
        p = Pipeline(output_format="srt")
        p.run(Path("test.m4a"), Path("out.srt"))

    mock_loader.load.assert_called_once()
    mock_filter.filter.assert_called_once()
    mock_translator.translate.assert_called_once()


@patch("mlx_subtitler.translator.Translator")
@patch("mlx_subtitler.pipeline.Transcriber")
def test_skip_filter(mock_tc, mock_tr):
    mock_tc.return_value = _make_transcriber_mock()

    mock_translator = MagicMock()
    mock_translator.translate.return_value = _segments()
    mock_tr.return_value = mock_translator

    with patch("mlx_subtitler.subtitle_writer.SubtitleWriter.write_srt", return_value=Path("out.srt")):
        p = Pipeline(filter_vocab=False, output_format="srt")
        p.run(Path("test.m4a"), Path("out.srt"))

    mock_translator.translate.assert_called_once()


@patch("mlx_subtitler.pipeline.Transcriber")
def test_auto_output_path(mock_cls, tmp_path):
    mock_cls.return_value = _make_transcriber_mock()

    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"\x00")
    expected = tmp_path / "audio.srt"

    p = Pipeline(translate=False, filter_vocab=False, output_format="srt")
    result = p.run(audio)
    assert result == expected


@patch("mlx_subtitler.pipeline.Transcriber")
def test_run_batch_processes_directory(mock_cls, tmp_path):
    mock_cls.return_value = _make_transcriber_mock()
    (tmp_path / "a.m4a").write_bytes(b"\x00")
    (tmp_path / "b.mp3").write_bytes(b"\x00")
    (tmp_path / "c.txt").write_bytes(b"\x00")
    p = Pipeline(translate=False, filter_vocab=False, output_format="srt")
    results = p.run_batch(tmp_path)
    assert len(results) == 2
    assert all(r.suffix == ".srt" for r in results)


@patch("mlx_subtitler.pipeline.Transcriber")
def test_run_batch_with_output_dir(mock_cls, tmp_path):
    mock_cls.return_value = _make_transcriber_mock()
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "a.m4a").write_bytes(b"\x00")
    out_dir = tmp_path / "output"
    p = Pipeline(translate=False, filter_vocab=False, output_format="srt")
    results = p.run_batch(audio_dir, out_dir)
    assert len(results) == 1
    assert results[0].parent == out_dir


@patch("mlx_subtitler.pipeline.Transcriber")
def test_run_batch_empty_directory(mock_cls, tmp_path):
    mock_cls.return_value = _make_transcriber_mock()
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    p = Pipeline(translate=False, filter_vocab=False, output_format="srt")
    results = p.run_batch(empty_dir)
    assert results == []
