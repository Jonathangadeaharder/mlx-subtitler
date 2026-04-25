from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_cli_transcribe_command(tmp_path):
    audio = tmp_path / "test.m4a"
    audio.write_bytes(b"\x00")
    with patch("cli.Pipeline") as mock_cls:
        mock_p = MagicMock()
        mock_p.run.return_value = tmp_path / "test.srt"
        mock_cls.return_value = mock_p
        from cli import main

        main(["transcribe", str(audio), "--no-translate", "--no-filter"])
        mock_cls.assert_called_once()
        mock_p.run.assert_called_once()


def test_cli_batch_command(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "a.m4a").write_bytes(b"\x00")
    with patch("cli.Pipeline") as mock_cls:
        mock_p = MagicMock()
        mock_p.run_batch.return_value = [tmp_path / "a.srt"]
        mock_cls.return_value = mock_p
        from cli import main

        main(["batch", str(audio_dir), "--no-translate"])
        mock_p.run_batch.assert_called_once()


def test_cli_custom_model_flag(tmp_path):
    audio = tmp_path / "test.m4a"
    audio.write_bytes(b"\x00")
    with patch("cli.Pipeline") as mock_cls:
        mock_p = MagicMock()
        mock_p.run.return_value = tmp_path / "test.srt"
        mock_cls.return_value = mock_p
        from cli import main

        main(["transcribe", str(audio), "--model", "custom-model", "--no-translate", "--no-filter"])
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["whisper_model"] == "custom-model"


def test_cli_vtt_format(tmp_path):
    audio = tmp_path / "test.m4a"
    audio.write_bytes(b"\x00")
    with patch("cli.Pipeline") as mock_cls:
        mock_p = MagicMock()
        mock_p.run.return_value = tmp_path / "test.vtt"
        mock_cls.return_value = mock_p
        from cli import main

        main(["transcribe", str(audio), "--format", "vtt", "--no-translate", "--no-filter"])
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["output_format"] == "vtt"
