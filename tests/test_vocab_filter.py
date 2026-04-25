from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from mlx_subtitler.models import Segment
from mlx_subtitler.vocab_filter import VocabFilter


def _make_token(text: str, lemma: str, pos_: str, is_punct: bool = False, is_stop: bool = False):
    tok = MagicMock()
    tok.text = text
    tok.lemma_ = lemma
    tok.pos_ = pos_
    tok.is_punct = is_punct
    tok.is_stop = is_stop
    return tok


def _make_doc(tokens: list):
    return tokens


@patch("mlx_subtitler.vocab_filter.spacy.load")
def test_removes_all_known_words(mock_load):
    mock_nlp = MagicMock()
    mock_nlp.return_value = _make_doc([
        _make_token("hallo", "hallo", "NOUN"),
        _make_token("welt", "welt", "NOUN"),
    ])
    mock_load.return_value = mock_nlp

    vf = VocabFilter(vocab={"hallo", "welt"})
    segs = [Segment(index=1, start=0.0, end=2.5, text="hallo welt")]
    result = vf.filter(segs)
    assert result == []


@patch("mlx_subtitler.vocab_filter.spacy.load")
def test_keeps_segment_with_unknown_word(mock_load):
    mock_nlp = MagicMock()
    mock_nlp.return_value = _make_doc([
        _make_token("hallo", "hallo", "NOUN"),
        _make_token("quantenphysik", "quantenphysik", "NOUN"),
    ])
    mock_load.return_value = mock_nlp

    vf = VocabFilter(vocab={"hallo"})
    segs = [Segment(index=1, start=0.0, end=2.5, text="hallo Quantenphysik")]
    result = vf.filter(segs)
    assert len(result) == 1


@patch("mlx_subtitler.vocab_filter.spacy.load")
def test_reindexes_output(mock_load):
    mock_nlp = MagicMock()
    mock_nlp.return_value = _make_doc([
        _make_token("quantenphysik", "quantenphysik", "NOUN"),
    ])
    mock_load.return_value = mock_nlp

    vf = VocabFilter(vocab=set())
    segs = [
        Segment(index=5, start=0.0, end=2.5, text="a"),
        Segment(index=10, start=2.5, end=5.0, text="b"),
    ]
    result = vf.filter(segs)
    assert [s.index for s in result] == [1, 2]


@patch("mlx_subtitler.vocab_filter.spacy.load")
def test_skips_punctuation(mock_load):
    mock_nlp = MagicMock()
    mock_nlp.return_value = _make_doc([
        _make_token("!", "!", "PUNCT", is_punct=True),
    ])
    mock_load.return_value = mock_nlp

    vf = VocabFilter(vocab=set())
    segs = [Segment(index=1, start=0.0, end=1.0, text="!")]
    result = vf.filter(segs)
    assert result == []


@patch("mlx_subtitler.vocab_filter.spacy.load")
def test_empty_input(mock_load):
    mock_nlp = MagicMock()
    mock_load.return_value = mock_nlp

    vf = VocabFilter(vocab={"hallo"})
    result = vf.filter([])
    assert result == []


@patch("mlx_subtitler.vocab_filter.spacy.load")
def test_init_calls_spacy_load_with_exact_args(mock_load):
    mock_load.return_value = MagicMock()
    vf = VocabFilter(vocab={"hallo"})
    mock_load.assert_called_once_with("de_core_news_lg", disable=["parser", "ner"])


@patch("mlx_subtitler.vocab_filter.spacy.load")
def test_init_stores_vocab(mock_load):
    mock_load.return_value = MagicMock()
    vf = VocabFilter(vocab={"hallo", "welt"})
    assert vf.vocab == {"hallo", "welt"}


@patch("mlx_subtitler.vocab_filter.spacy.load")
def test_filter_passes_seg_text_to_nlp(mock_load):
    mock_nlp = MagicMock()
    mock_token = _make_token("Hallo", "hallo", "NOUN")
    mock_nlp.return_value = [mock_token]
    mock_load.return_value = mock_nlp
    vf = VocabFilter(vocab={"hallo"})
    vf.filter([Segment(index=1, start=0.0, end=1.0, text="Hallo Welt")])
    mock_nlp.assert_called_with("Hallo Welt")


@patch("mlx_subtitler.vocab_filter.spacy.load")
def test_filter_skips_stop_words(mock_load):
    mock_nlp = MagicMock()
    stop_token = _make_token("der", "der", "DET", is_stop=True)
    good_token = _make_token("Haus", "haus", "NOUN")
    mock_nlp.return_value = [stop_token, good_token]
    mock_load.return_value = mock_nlp
    vf = VocabFilter(vocab={"haus"})
    result = vf.filter([Segment(index=1, start=0.0, end=1.0, text="das Haus")])
    assert result == []


@patch("mlx_subtitler.vocab_filter.spacy.load")
def test_filter_does_not_break_on_skip(mock_load):
    mock_nlp = MagicMock()
    punct_token = _make_token(".", ".", "PUNCT", is_punct=True)
    good_token = _make_token("Welt", "welt", "NOUN")
    mock_nlp.return_value = [punct_token, good_token]
    mock_load.return_value = mock_nlp
    vf = VocabFilter(vocab={"welt"})
    result = vf.filter([Segment(index=1, start=0.0, end=1.0, text=". Welt")])
    assert len(result) == 0


@patch("mlx_subtitler.vocab_filter.spacy.load")
def test_filter_continues_past_skip_pos(mock_load):
    mock_nlp = MagicMock()
    num_token = _make_token("drei", "drei", "NUM")
    good_token = _make_token("Haus", "haus", "NOUN")
    mock_nlp.return_value = [num_token, good_token]
    mock_load.return_value = mock_nlp
    vf = VocabFilter(vocab={"haus"})
    result = vf.filter([Segment(index=1, start=0.0, end=1.0, text="drei Haus")])
    assert len(result) == 0


@patch("mlx_subtitler.vocab_filter.spacy.load")
def test_filter_skips_empty_lemma(mock_load):
    mock_nlp = MagicMock()
    empty_token = MagicMock()
    empty_token.is_punct = False
    empty_token.is_stop = False
    empty_token.pos_ = "NOUN"
    empty_token.lemma_ = "  "
    mock_nlp.return_value = [empty_token]
    mock_load.return_value = mock_nlp
    vf = VocabFilter(vocab=set())
    result = vf.filter([Segment(index=1, start=0.0, end=1.0, text="x")])
    assert result == []


@patch("mlx_subtitler.vocab_filter.spacy.load")
def test_filter_lemma_lowered(mock_load):
    mock_nlp = MagicMock()
    token = MagicMock()
    token.is_punct = False
    token.is_stop = False
    token.pos_ = "NOUN"
    token.lemma_ = "Haus"
    mock_nlp.return_value = [token]
    mock_load.return_value = mock_nlp
    vf = VocabFilter(vocab={"haus"})
    result = vf.filter([Segment(index=1, start=0.0, end=1.0, text="Haus")])
    assert result == []


@patch("mlx_subtitler.vocab_filter.spacy.load")
def test_filter_keeps_unknown_lemma_case_insensitive(mock_load):
    mock_nlp = MagicMock()
    token = MagicMock()
    token.is_punct = False
    token.is_stop = False
    token.pos_ = "NOUN"
    token.lemma_ = "Quantenphysik"
    mock_nlp.return_value = [token]
    mock_load.return_value = mock_nlp
    vf = VocabFilter(vocab={"haus"})
    result = vf.filter([Segment(index=1, start=0.0, end=1.0, text="Quantenphysik")])
    assert len(result) == 1


@patch("mlx_subtitler.vocab_filter.spacy.load")
def test_filter_reindexes_output_preserves_fields(mock_load):
    mock_nlp = MagicMock()
    t1 = _make_token("quantenphysik", "quantenphysik", "NOUN")
    t2 = _make_token("quantenmechanik", "quantenmechanik", "NOUN")
    mock_nlp.side_effect = [[t1], [t2]]
    mock_load.return_value = mock_nlp
    vf = VocabFilter(vocab=set())
    segs = [
        Segment(index=99, start=1.5, end=3.0, text="Quantenphysik"),
        Segment(index=100, start=4.0, end=5.5, text="Quantenmechanik"),
    ]
    result = vf.filter(segs)
    assert result[0].start == 1.5
    assert result[0].end == 3.0
    assert result[0].text == "Quantenphysik"
    assert result[0].index == 1
    assert result[1].start == 4.0
    assert result[1].end == 5.5
    assert result[1].text == "Quantenmechanik"
    assert result[1].index == 2


@patch("mlx_subtitler.vocab_filter.spacy.load")
def test_filter_preserves_translation_field(mock_load):
    mock_nlp = MagicMock()
    token = _make_token("quantenphysik", "quantenphysik", "NOUN")
    mock_nlp.return_value = [token]
    mock_load.return_value = mock_nlp
    vf = VocabFilter(vocab=set())
    segs = [Segment(index=1, start=0.0, end=1.0, text="Quantenphysik", translation="quantum physics")]
    result = vf.filter(segs)
    assert result[0].translation == "quantum physics"


@patch("mlx_subtitler.vocab_filter.spacy.load")
def test_filter_all_known_drops_segment(mock_load):
    mock_nlp = MagicMock()
    t1 = _make_token("hallo", "hallo", "NOUN")
    t2 = _make_token("welt", "welt", "NOUN")
    mock_nlp.return_value = [t1, t2]
    mock_load.return_value = mock_nlp
    vf = VocabFilter(vocab={"hallo", "welt"})
    result = vf.filter([Segment(index=1, start=0.0, end=1.0, text="Hallo Welt")])
    assert result == []


@patch("mlx_subtitler.vocab_filter.spacy.load")
def test_filter_mixed_known_unknown_keeps_segment(mock_load):
    mock_nlp = MagicMock()
    t1 = _make_token("hallo", "hallo", "NOUN")
    t2 = _make_token("quantenphysik", "quantenphysik", "NOUN")
    mock_nlp.return_value = [t1, t2]
    mock_load.return_value = mock_nlp
    vf = VocabFilter(vocab={"hallo"})
    result = vf.filter([Segment(index=1, start=0.0, end=1.0, text="hallo Quantenphysik")])
    assert len(result) == 1
    assert result[0].text == "hallo Quantenphysik"


@patch("mlx_subtitler.vocab_filter.spacy.load")
def test_filter_drops_segment_with_no_lemmas(mock_load):
    mock_nlp = MagicMock()
    mock_nlp.return_value = []
    mock_load.return_value = mock_nlp
    vf = VocabFilter(vocab=set())
    result = vf.filter([Segment(index=1, start=0.0, end=1.0, text="...")])
    assert result == []


@patch("mlx_subtitler.vocab_filter.spacy.load")
def test_init_custom_spacy_model(mock_load):
    mock_load.return_value = MagicMock()
    vf = VocabFilter(vocab={"x"}, spacy_model="de_core_news_sm")
    mock_load.assert_called_once_with("de_core_news_sm", disable=["parser", "ner"])


@patch("mlx_subtitler.vocab_filter.spacy.load")
def test_filter_continue_vs_break_on_skip(mock_load):
    mock_nlp = MagicMock()
    punct_token = _make_token(",", ",", "PUNCT", is_punct=True)
    good_token = _make_token("Quantenphysik", "quantenphysik", "NOUN")
    unknown_token = _make_token("Unbekannt", "unbekannt", "NOUN")
    mock_nlp.return_value = [punct_token, good_token, unknown_token]
    mock_load.return_value = mock_nlp
    vf = VocabFilter(vocab={"quantenphysik"})
    result = vf.filter([Segment(index=1, start=0.0, end=1.0, text=", Quantenphysik Unbekannt")])
    assert len(result) == 1
    assert result[0].text == ", Quantenphysik Unbekannt"
