from __future__ import annotations

import sys
from unittest.mock import MagicMock

for _mod in (
    "mlx_whisper",
    "mlx",
    "mlx.core",
    "spacy",
    "torch",
    "transformers",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import pytest
from mlx_subtitler.models import Segment

collect_ignore = []
if "mutmut" in sys.modules:
    collect_ignore = ["test_cli.py", "test_e2e.py"]


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
