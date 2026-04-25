from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mlx_subtitler.models import Segment
from mlx_subtitler.translator import Translator


def _make_segments():
    return [
        Segment(index=1, start=0.0, end=2.5, text="Hallo Welt"),
        Segment(index=2, start=2.5, end=5.0, text="Wie geht es dir"),
    ]


def _make_translator():
    t = Translator.__new__(Translator)
    t.model_name = "Helsinki-NLP/opus-mt-tc-big-de-es"
    t.device = "cpu"
    t.tokenizer = MagicMock()
    t.model = MagicMock()
    return t


def test_translate_fills_translation():
    t = _make_translator()
    t.tokenizer.batch_decode.return_value = ["Hola mundo", "¿Cómo estás"]
    t.model.generate.return_value = MagicMock()

    result = t.translate(_make_segments())
    assert result[0].translation == "Hola mundo"
    assert result[1].translation == "¿Cómo estás"


def test_translate_respects_batch_size():
    t = _make_translator()
    t.tokenizer.batch_decode.return_value = ["translated"] * 3
    t.model.generate.return_value = MagicMock()

    segs = [Segment(index=i, start=0.0, end=1.0, text=f"text{i}") for i in range(3)]
    t.translate(segs, batch_size=2)
    assert t.tokenizer.call_count == 2


def test_translate_empty_segments():
    t = _make_translator()
    result = t.translate([])
    assert result == []


def test_model_name_format():
    assert "Helsinki-NLP/opus-mt-tc-big-de-es" == f"Helsinki-NLP/opus-mt-tc-big-de-es"
    assert "Helsinki-NLP/opus-mt-tc-big-en-fr" == f"Helsinki-NLP/opus-mt-tc-big-en-fr"


def test_init_with_explicit_device():
    import sys

    mock_transformers = MagicMock()
    mock_model = MagicMock()
    mock_transformers.MarianMTModel.from_pretrained.return_value = mock_model
    mock_transformers.MarianTokenizer.from_pretrained.return_value = MagicMock()

    with patch.dict(sys.modules, {"transformers": mock_transformers}):
        t = Translator(src_lang="de", tgt_lang="es", device="cpu")
    assert t.device == "cpu"
    mock_model.to.assert_called_once_with("cpu")


def test_init_auto_device_selects_mps():
    import sys

    mock_torch = MagicMock()
    mock_torch.backends.mps.is_available.return_value = True
    mock_transformers = MagicMock()
    mock_model = MagicMock()
    mock_transformers.MarianMTModel.from_pretrained.return_value = mock_model
    mock_transformers.MarianTokenizer.from_pretrained.return_value = MagicMock()

    with patch.dict(sys.modules, {"torch": mock_torch, "transformers": mock_transformers}):
        t = Translator(src_lang="de", tgt_lang="es", device="auto")
    assert t.device == "mps"
    mock_model.to.assert_called_once_with("mps")


def test_init_auto_device_falls_back_to_cpu_without_torch():
    import sys

    mock_transformers = MagicMock()
    mock_model = MagicMock()
    mock_transformers.MarianMTModel.from_pretrained.return_value = mock_model
    mock_transformers.MarianTokenizer.from_pretrained.return_value = MagicMock()

    with patch.dict(sys.modules, {"torch": None, "transformers": mock_transformers}):
        t = Translator(src_lang="de", tgt_lang="es", device="auto")
    assert t.device == "cpu"
    mock_model.to.assert_called_once_with("cpu")
