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


def test_init_stores_model_default():
    t = Transcriber()
    assert t.model == "mlx-community/whisper-large-v3-turbo"
    assert t.language is None


def test_init_stores_custom_params():
    t = Transcriber(model="other-model", language="de")
    assert t.model == "other-model"
    assert t.language == "de"


@patch("mlx_subtitler.transcriber.mlx_whisper.transcribe", return_value=_mock_whisper_result())
def test_transcribe_passes_model_in_kwargs(mock_transcribe):
    t = Transcriber(model="test-model", language="de")
    t.transcribe(Path("/audio.wav"))
    call_kwargs = mock_transcribe.call_args[1]
    assert call_kwargs["path_or_hf_repo"] == "test-model"
    assert call_kwargs["condition_on_previous_text"] is False
    assert call_kwargs["language"] == "de"


@patch("mlx_subtitler.transcriber.mlx_whisper.transcribe")
def test_transcribe_passes_audio_path_as_string(mock_transcribe):
    mock_transcribe.return_value = {"segments": []}
    t = Transcriber()
    t.transcribe(Path("/audio.wav"))
    first_arg = mock_transcribe.call_args[0][0]
    assert first_arg == "/audio.wav"
    assert isinstance(first_arg, str)


@patch("mlx_subtitler.transcriber.mlx_whisper.transcribe")
def test_transcribe_extracts_segment_fields(mock_transcribe):
    mock_transcribe.return_value = {
        "segments": [
            {"start": 1.5, "end": 3.0, "text": "  hello  "},
        ]
    }
    t = Transcriber()
    result = t.transcribe(Path("/a.wav"))
    assert result[0].start == 1.5
    assert result[0].end == 3.0
    assert result[0].text == "hello"


@patch("mlx_subtitler.transcriber.mlx_whisper.transcribe")
def test_transcribe_uses_get_with_default_empty_list(mock_transcribe):
    mock_transcribe.return_value = {}
    t = Transcriber()
    result = t.transcribe(Path("/a.wav"))
    assert result == []


@patch("mlx_subtitler.transcriber.mlx_whisper.transcribe")
def test_transcribe_enumerates_from_1(mock_transcribe):
    mock_transcribe.return_value = {
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "a"},
            {"start": 1.0, "end": 2.0, "text": "b"},
        ]
    }
    t = Transcriber()
    result = t.transcribe(Path("/a.wav"))
    assert result[0].index == 1
    assert result[1].index == 2


@patch("mlx_subtitler.transcriber.mlx_whisper.transcribe")
def test_transcribe_condition_on_previous_text_is_false(mock_transcribe):
    mock_transcribe.return_value = {"segments": []}
    t = Transcriber()
    t.transcribe(Path("/a.wav"))
    call_kwargs = mock_transcribe.call_args[1]
    assert "condition_on_previous_text" in call_kwargs
    assert call_kwargs["condition_on_previous_text"] is False


@patch("mlx_subtitler.transcriber.mlx_whisper.transcribe")
def test_transcribe_omits_language_when_none(mock_transcribe):
    mock_transcribe.return_value = {"segments": []}
    t = Transcriber(language=None)
    t.transcribe(Path("/a.wav"))
    call_kwargs = mock_transcribe.call_args[1]
    assert "language" not in call_kwargs


@patch("mlx_subtitler.transcriber.mlx_whisper.transcribe")
def test_transcribe_includes_language_when_set(mock_transcribe):
    mock_transcribe.return_value = {"segments": []}
    t = Transcriber(language="fr")
    t.transcribe(Path("/a.wav"))
    call_kwargs = mock_transcribe.call_args[1]
    assert call_kwargs["language"] == "fr"


@patch("mlx_subtitler.transcriber.mlx_whisper.transcribe")
def test_transcribe_strips_text(mock_transcribe):
    mock_transcribe.return_value = {
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "  spaces  "},
        ]
    }
    t = Transcriber()
    result = t.transcribe(Path("/a.wav"))
    assert result[0].text == "spaces"


@patch("mlx_subtitler.transcriber.mlx_whisper.transcribe")
def test_transcribe_preserves_start_end(mock_transcribe):
    mock_transcribe.return_value = {
        "segments": [
            {"start": 2.5, "end": 5.0, "text": "x"},
        ]
    }
    t = Transcriber()
    result = t.transcribe(Path("/a.wav"))
    assert result[0].start == 2.5
    assert result[0].end == 5.0


@patch("mlx_subtitler.transcriber.mlx_whisper.transcribe")
def test_transcribe_returns_list_of_segment(mock_transcribe):
    mock_transcribe.return_value = {
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "a"},
            {"start": 1.0, "end": 2.0, "text": "b"},
            {"start": 2.0, "end": 3.0, "text": "c"},
        ]
    }
    t = Transcriber()
    result = t.transcribe(Path("/a.wav"))
    assert len(result) == 3
    assert all(isinstance(s, Segment) for s in result)
