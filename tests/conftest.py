from __future__ import annotations

import pytest
from mlx_subtitler.models import Segment


@pytest.fixture
def sample_segments() -> list[Segment]:
    return [
        Segment(index=1, start=0.0, end=2.5, text="Hallo Welt"),
        Segment(index=2, start=2.5, end=5.0, text="Wie geht es dir"),
        Segment(index=3, start=5.0, end=7.5, text="Das ist ein Test"),
    ]


@pytest.fixture
def translated_segments() -> list[Segment]:
    return [
        Segment(index=1, start=0.0, end=2.5, text="Hallo Welt", translation="Hola mundo"),
        Segment(index=2, start=2.5, end=5.0, text="Wie geht es dir", translation="¿Cómo estás"),
        Segment(index=3, start=5.0, end=7.5, text="Das ist ein Test", translation="Esto es una prueba"),
    ]
