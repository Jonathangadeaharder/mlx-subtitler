from __future__ import annotations

from pathlib import Path

import pytest

from mlx_subtitler.models import Segment
from mlx_subtitler.subtitle_writer import SubtitleWriter


def _translated_segments():
    return [
        Segment(index=1, start=0.0, end=2.5, text="Hallo Welt", translation="Hola mundo"),
        Segment(index=2, start=2.5, end=5.0, text="Wie geht es dir", translation="¿Cómo estás"),
    ]


def _no_translation_segments():
    return [
        Segment(index=1, start=0.0, end=2.5, text="Hallo Welt"),
        Segment(index=2, start=2.5, end=5.0, text="Wie geht es dir"),
    ]


def test_srt_uses_translation(tmp_path):
    out = tmp_path / "out.srt"
    SubtitleWriter.write_srt(_translated_segments(), out)
    content = out.read_text()
    assert "Hola mundo" in content
    assert "¿Cómo estás" in content


def test_srt_uses_original_when_no_translation(tmp_path):
    out = tmp_path / "out.srt"
    SubtitleWriter.write_srt(_no_translation_segments(), out)
    content = out.read_text()
    assert "Hallo Welt" in content
    assert "Wie geht es dir" in content


def test_srt_uses_original_when_requested(tmp_path):
    out = tmp_path / "out.srt"
    SubtitleWriter.write_srt(_translated_segments(), out, use_translation=False)
    content = out.read_text()
    assert "Hallo Welt" in content


def test_srt_has_timestamps(tmp_path):
    out = tmp_path / "out.srt"
    SubtitleWriter.write_srt(_translated_segments(), out)
    content = out.read_text()
    assert "-->" in content
    assert "00:00:00,000" in content


def test_vtt_format(tmp_path):
    out = tmp_path / "out.vtt"
    SubtitleWriter.write_vtt(_translated_segments(), out)
    content = out.read_text()
    assert content.startswith("WEBVTT")
    assert "-->" in content
    assert "Hola mundo" in content


def test_returns_path(tmp_path):
    out = tmp_path / "out.srt"
    result = SubtitleWriter.write_srt(_translated_segments(), out)
    assert result == out
    assert out.exists()
