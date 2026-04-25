from __future__ import annotations

import pytest
from mlx_subtitler.models import Segment


def test_segment_creation():
    seg = Segment(index=1, start=0.0, end=2.5, text="Hello world")
    assert seg.index == 1
    assert seg.start == 0.0
    assert seg.end == 2.5
    assert seg.text == "Hello world"
    assert seg.translation is None


def test_segment_with_translation():
    seg = Segment(index=1, start=0.0, end=2.5, text="Hallo", translation="Hello")
    assert seg.translation == "Hello"


def test_segment_frozen():
    seg = Segment(index=1, start=0.0, end=2.5, text="Hello")
    with pytest.raises(AttributeError):
        seg.text = "changed"


def test_segment_equality():
    a = Segment(index=1, start=0.0, end=2.5, text="Hello")
    b = Segment(index=1, start=0.0, end=2.5, text="Hello")
    assert a == b


def test_segment_duration():
    seg = Segment(index=1, start=1.0, end=4.5, text="Hello")
    assert seg.duration == 3.5
