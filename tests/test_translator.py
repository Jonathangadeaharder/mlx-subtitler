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


def test_init_stores_src_and_tgt_lang():
    import sys

    mock_transformers = MagicMock()
    mock_transformers.MarianMTModel.from_pretrained.return_value = MagicMock()
    mock_transformers.MarianTokenizer.from_pretrained.return_value = MagicMock()

    with patch.dict(sys.modules, {"transformers": mock_transformers}):
        t = Translator(src_lang="en", tgt_lang="fr", device="cpu")
    assert t.src_lang == "en"
    assert t.tgt_lang == "fr"


def test_init_builds_model_name():
    import sys

    mock_transformers = MagicMock()
    mock_transformers.MarianMTModel.from_pretrained.return_value = MagicMock()
    mock_transformers.MarianTokenizer.from_pretrained.return_value = MagicMock()

    with patch.dict(sys.modules, {"transformers": mock_transformers}):
        t = Translator(src_lang="de", tgt_lang="es", device="cpu")
    assert t.model_name == "Helsinki-NLP/opus-mt-tc-big-de-es"


def test_init_loads_tokenizer_and_model():
    import sys

    mock_transformers = MagicMock()
    mock_tok = MagicMock()
    mock_model = MagicMock()
    mock_transformers.MarianMTModel.from_pretrained.return_value = mock_model
    mock_transformers.MarianTokenizer.from_pretrained.return_value = mock_tok

    with patch.dict(sys.modules, {"transformers": mock_transformers}):
        t = Translator(src_lang="de", tgt_lang="es", device="cpu")
    mock_transformers.MarianTokenizer.from_pretrained.assert_called_once_with("Helsinki-NLP/opus-mt-tc-big-de-es")
    mock_transformers.MarianMTModel.from_pretrained.assert_called_once_with("Helsinki-NLP/opus-mt-tc-big-de-es")
    mock_model.to.assert_called_once_with("cpu")


def test_translate_returns_empty_for_empty_input():
    t = _make_translator()
    result = t.translate([])
    assert result == []


def test_translate_passes_batch_to_tokenizer():
    t = _make_translator()
    mock_inputs = MagicMock()
    t.tokenizer.return_value = mock_inputs
    t.model.generate.return_value = [1, 2, 3]
    t.tokenizer.batch_decode.return_value = ["translated"]
    segs = [Segment(index=1, start=0.0, end=1.0, text="Hallo")]
    t.translate(segs)
    t.tokenizer.assert_called_once_with(["Hallo"], return_tensors="pt", padding=True, truncation=True)


def test_translate_moves_inputs_to_device():
    t = _make_translator()
    t.device = "mps"
    mock_inputs = MagicMock()
    t.tokenizer.return_value = mock_inputs
    t.model.generate.return_value = [1]
    t.tokenizer.batch_decode.return_value = ["x"]
    segs = [Segment(index=1, start=0.0, end=1.0, text="a")]
    t.translate(segs)
    mock_inputs.to.assert_called_once_with("mps")


def test_translate_generates_with_max_new_tokens():
    t = _make_translator()
    mock_inputs = MagicMock()
    t.tokenizer.return_value = mock_inputs
    t.model.generate.return_value = [1]
    t.tokenizer.batch_decode.return_value = ["x"]
    segs = [Segment(index=1, start=0.0, end=1.0, text="a")]
    t.translate(segs)
    gen_kwargs = t.model.generate.call_args[1]
    assert gen_kwargs["max_new_tokens"] == 512


def test_translate_batch_decode_args():
    t = _make_translator()
    t.tokenizer.return_value = MagicMock()
    t.model.generate.return_value = [1, 2]
    t.tokenizer.batch_decode.return_value = ["hello", "world"]
    segs = [Segment(index=1, start=0.0, end=1.0, text="a")]
    t.translate(segs)
    t.tokenizer.batch_decode.assert_called_once_with([1, 2], skip_special_tokens=True)


def test_translate_preserves_segment_fields():
    t = _make_translator()
    t.tokenizer.return_value = MagicMock()
    t.model.generate.return_value = [1]
    t.tokenizer.batch_decode.return_value = ["hola"]
    segs = [Segment(index=1, start=2.5, end=5.0, text="Hallo")]
    result = t.translate(segs)
    assert result[0].index == 1
    assert result[0].start == 2.5
    assert result[0].end == 5.0
    assert result[0].text == "Hallo"
    assert result[0].translation == "hola"


def test_translate_batches_correctly():
    t = _make_translator()
    t.tokenizer.return_value = MagicMock()
    t.model.generate.return_value = [1]
    t.tokenizer.batch_decode.return_value = ["x"]
    segs = [Segment(index=i, start=0.0, end=1.0, text=f"text{i}") for i in range(5)]
    t.translate(segs, batch_size=2)
    assert t.tokenizer.call_count == 3


def test_translate_passes_correct_texts_to_tokenizer():
    t = _make_translator()
    t.tokenizer.return_value = MagicMock()
    t.model.generate.return_value = [1]
    t.tokenizer.batch_decode.return_value = ["x", "y"]
    segs = [
        Segment(index=1, start=0.0, end=1.0, text="Hallo"),
        Segment(index=2, start=1.0, end=2.0, text="Welt"),
    ]
    t.translate(segs, batch_size=32)
    first_call_texts = t.tokenizer.call_args[0][0]
    assert first_call_texts == ["Hallo", "Welt"]


def test_translate_default_batch_size_is_32():
    t = _make_translator()
    t.tokenizer.return_value = MagicMock()
    t.model.generate.return_value = [1]
    t.tokenizer.batch_decode.return_value = ["x"]
    segs = [Segment(index=1, start=0.0, end=1.0, text="a")]
    t.translate(segs)
    assert t.tokenizer.call_count == 1


def test_translate_with_torch_no_grad():
    t = _make_translator()
    t.tokenizer.return_value = MagicMock()
    t.model.generate.return_value = [1]
    t.tokenizer.batch_decode.return_value = ["x"]
    segs = [Segment(index=1, start=0.0, end=1.0, text="a")]

    import sys
    mock_torch = sys.modules["torch"]
    t.translate(segs)
    mock_torch.no_grad.assert_called()


def test_translate_zip_preserves_order():
    t = _make_translator()
    t.tokenizer.return_value = MagicMock()
    t.model.generate.return_value = [1, 2]
    t.tokenizer.batch_decode.return_value = ["first", "second"]
    segs = [
        Segment(index=1, start=0.0, end=1.0, text="A"),
        Segment(index=2, start=1.0, end=2.0, text="B"),
    ]
    result = t.translate(segs)
    assert result[0].translation == "first"
    assert result[1].translation == "second"


def test_init_default_langs():
    import sys

    mock_transformers = MagicMock()
    mock_transformers.MarianMTModel.from_pretrained.return_value = MagicMock()
    mock_transformers.MarianTokenizer.from_pretrained.return_value = MagicMock()

    with patch.dict(sys.modules, {"transformers": mock_transformers}):
        t = Translator(device="cpu")
    assert t.src_lang == "de"
    assert t.tgt_lang == "es"
    assert t.model_name == "Helsinki-NLP/opus-mt-tc-big-de-es"


def test_init_model_name_format():
    import sys

    mock_transformers = MagicMock()
    mock_transformers.MarianMTModel.from_pretrained.return_value = MagicMock()
    mock_transformers.MarianTokenizer.from_pretrained.return_value = MagicMock()

    with patch.dict(sys.modules, {"transformers": mock_transformers}):
        t = Translator(src_lang="en", tgt_lang="fr", device="cpu")
    assert t.model_name == "Helsinki-NLP/opus-mt-tc-big-en-fr"


def test_init_device_explicit_cpu():
    import sys

    mock_transformers = MagicMock()
    mock_model = MagicMock()
    mock_transformers.MarianMTModel.from_pretrained.return_value = mock_model
    mock_transformers.MarianTokenizer.from_pretrained.return_value = MagicMock()

    with patch.dict(sys.modules, {"transformers": mock_transformers}):
        t = Translator(device="cpu")
    assert t.device == "cpu"


def test_init_device_explicit_mps():
    import sys

    mock_transformers = MagicMock()
    mock_model = MagicMock()
    mock_transformers.MarianMTModel.from_pretrained.return_value = mock_model
    mock_transformers.MarianTokenizer.from_pretrained.return_value = MagicMock()

    with patch.dict(sys.modules, {"transformers": mock_transformers}):
        t = Translator(device="mps")
    assert t.device == "mps"
    mock_model.to.assert_called_once_with("mps")


def test_translate_generate_receives_spread_inputs():
    t = _make_translator()
    mock_inputs = MagicMock()
    t.tokenizer.return_value = mock_inputs
    t.model.generate.return_value = [1]
    t.tokenizer.batch_decode.return_value = ["x"]
    segs = [Segment(index=1, start=0.0, end=1.0, text="a")]
    t.translate(segs)
    gen_call = t.model.generate.call_args
    assert gen_call[1].get("max_new_tokens") == 512
    assert "**inputs" not in str(gen_call)
    mock_inputs.to.assert_called_once_with("cpu")


def test_translate_spread_inputs_not_passed_directly():
    t = _make_translator()
    mock_inputs = MagicMock()
    mock_inputs_keys = MagicMock()
    mock_inputs.keys.return_value = ["input_ids", "attention_mask"]
    t.tokenizer.return_value = mock_inputs
    t.model.generate.return_value = [1]
    t.tokenizer.batch_decode.return_value = ["x"]
    segs = [Segment(index=1, start=0.0, end=1.0, text="a")]
    t.translate(segs)
    gen_args, gen_kwargs = t.model.generate.call_args
    assert gen_args == ()
    assert "max_new_tokens" in gen_kwargs
