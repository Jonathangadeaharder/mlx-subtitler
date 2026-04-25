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
