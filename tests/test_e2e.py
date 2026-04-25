from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mlx_subtitler.models import Segment
from mlx_subtitler.pipeline import Pipeline


def test_e2e_transcribe_translate_write_srt(tmp_path):
    segments = [
        Segment(index=1, start=0.0, end=3.0, text="Hallo Welt"),
        Segment(index=2, start=3.0, end=6.0, text="Guten Morgen"),
    ]
    translated = [
        Segment(index=1, start=0.0, end=3.0, text="Hallo Welt", translation="Hola mundo"),
        Segment(index=2, start=3.0, end=6.0, text="Guten Morgen", translation="Buenos días"),
    ]
    audio = tmp_path / "test.m4a"
    audio.write_bytes(b"\x00")
    with patch("mlx_subtitler.pipeline.Transcriber") as mock_tc, \
         patch("mlx_subtitler.translator.Translator") as mock_tr:
        mock_t = MagicMock()
        mock_t.transcribe.return_value = segments
        mock_tc.return_value = mock_t
        mock_translator = MagicMock()
        mock_translator.translate.return_value = translated
        mock_tr.return_value = mock_translator
        p = Pipeline(src_lang="de", tgt_lang="es", filter_vocab=False, translate=True, output_format="srt")
        out = p.run(audio)
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "Hola mundo" in content
    assert "Buenos días" in content
    assert "-->" in content
    assert "00:00:00,000" in content


def test_e2e_transcribe_filter_translate_write(tmp_path):
    segments = [
        Segment(index=1, start=0.0, end=3.0, text="Hallo"),
        Segment(index=2, start=3.0, end=6.0, text="Unbekannteswort hier"),
    ]
    filtered = [Segment(index=1, start=3.0, end=6.0, text="Unbekannteswort hier")]
    translated = [Segment(index=1, start=3.0, end=6.0, text="Unbekannteswort hier", translation="Palabra desconocida")]
    audio = tmp_path / "test.m4a"
    audio.write_bytes(b"\x00")
    with patch("mlx_subtitler.pipeline.Transcriber") as mock_tc, \
         patch("mlx_subtitler.vocab_filter.VocabFilter") as mock_vf_cls, \
         patch("mlx_subtitler.vocab_loader.VocabLoader") as mock_vl_cls, \
         patch("mlx_subtitler.translator.Translator") as mock_tr:
        mock_t = MagicMock()
        mock_t.transcribe.return_value = segments
        mock_tc.return_value = mock_t
        mock_loader = MagicMock()
        mock_loader.load.return_value = {"hallo"}
        mock_vl_cls.return_value = mock_loader
        mock_filter = MagicMock()
        mock_filter.filter.return_value = filtered
        mock_vf_cls.return_value = mock_filter
        mock_translator = MagicMock()
        mock_translator.translate.return_value = translated
        mock_tr.return_value = mock_translator
        p = Pipeline(src_lang="de", tgt_lang="es", filter_vocab=True, vocab_levels=["B1"], translate=True, output_format="srt")
        out = p.run(audio)
    content = out.read_text(encoding="utf-8")
    assert "Palabra desconocida" in content
    assert "Hallo" not in content
