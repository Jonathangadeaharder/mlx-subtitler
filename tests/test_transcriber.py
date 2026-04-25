from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from mlx_subtitler.models import Segment
from mlx_subtitler.transcriber import Transcriber


def _mock_whisper_result():
    return {
        "segments": [
            {"start": 0.0, "end": 2.5, "text": "  Hallo Welt  "},
            {"start": 2.5, "end": 5.0, "text": "Wie geht es dir"},
        ]
    }


@patch("mlx_subtitler.transcriber.mlx_whisper.transcribe", return_value=_mock_whisper_result())
def test_transcribe_returns_segments(mock_transcribe):
    t = Transcriber()
    segments = t.transcribe(Path("test.m4a"))
    assert len(segments) == 2
    assert isinstance(segments[0], Segment)
    assert segments[0].index == 1


@patch("mlx_subtitler.transcriber.mlx_whisper.transcribe", return_value=_mock_whisper_result())
def test_transcribe_strips_whitespace(mock_transcribe):
    t = Transcriber()
    segments = t.transcribe(Path("test.m4a"))
    assert segments[0].text == "Hallo Welt"


@patch("mlx_subtitler.transcriber.mlx_whisper.transcribe", return_value=_mock_whisper_result())
def test_transcribe_passes_language(mock_transcribe):
    t = Transcriber(language="de")
    t.transcribe(Path("test.m4a"))
    call_kwargs = mock_transcribe.call_args
    assert call_kwargs[1]["language"] == "de" or (len(call_kwargs[0]) > 1 and False)


@patch("mlx_subtitler.transcriber.mlx_whisper.transcribe", return_value=_mock_whisper_result())
def test_transcribe_auto_detect_no_language(mock_transcribe):
    t = Transcriber()
    t.transcribe(Path("test.m4a"))
    call_kwargs = mock_transcribe.call_args[1]
    assert "language" not in call_kwargs
