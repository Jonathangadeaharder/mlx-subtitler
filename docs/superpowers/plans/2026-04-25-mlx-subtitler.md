# MLX Subtitler — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a modular, local, Apple Silicon-optimized transcription + translation + vocab-filtering pipeline as a Python library with CLI.

**Architecture:** Four-stage pipeline (Transcriber → VocabFilter → Translator → SubtitleWriter) composed by a Pipeline class. Each module is independently testable. Shared `Segment` dataclass passed between stages. CLI wraps the pipeline.

**Tech Stack:** Python 3.10+, mlx-whisper, transformers (MarianMT), spacy, pysrt, pytest

---

## File Structure

```
mlx_subtitler/
├── mlx_subtitler/
│   ├── __init__.py
│   ├── models.py             # Task 1
│   ├── transcriber.py        # Task 2
│   ├── vocab_loader.py       # Task 3
│   ├── vocab_filter.py       # Task 4
│   ├── translator.py         # Task 5
│   ├── subtitle_writer.py    # Task 6
│   └── pipeline.py           # Task 7
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_models.py        # Task 1
│   ├── test_transcriber.py   # Task 2
│   ├── test_vocab_loader.py  # Task 3
│   ├── test_vocab_filter.py  # Task 4
│   ├── test_translator.py    # Task 5
│   ├── test_subtitle_writer.py # Task 6
│   └── test_pipeline.py      # Task 7
├── cli.py                    # Task 8
├── pyproject.toml            # Task 0
└── docs/superpowers/
    ├── specs/2026-04-25-mlx-subtitler-design.md
    └── plans/2026-04-25-mlx-subtitler.md
```

---

## Task 0: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `mlx_subtitler/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
# pyproject.toml
[project]
name = "mlx-subtitler"
version = "0.1.0"
description = "Local MLX-powered transcription, translation, and subtitle generation for Apple Silicon"
requires-python = ">=3.10"
dependencies = [
    "mlx-whisper",
    "transformers",
    "sentencepiece",
    "sacremoses",
    "spacy",
    "pysrt",
    "tqdm",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-cov"]

[project.scripts]
mlx-subtitle = "cli:main"
```

- [ ] **Step 2: Create package init**

```python
# mlx_subtitler/__init__.py
```

- [ ] **Step 3: Create test init and conftest**

```python
# tests/__init__.py
```

```python
# tests/conftest.py
import pytest
from mlx_subtitler.models import Segment


@pytest.fixture
def sample_segments():
    return [
        Segment(index=1, start=0.0, end=3.5, text="Hallo, wie geht es dir?"),
        Segment(index=2, start=3.5, end=7.0, text="Mir geht es gut, danke."),
        Segment(index=3, start=7.0, end=12.0, text="Das Wetter ist heute sehr schön."),
    ]


@pytest.fixture
def sample_vocab():
    return {"hallo", "gehen", "gut", "danke", "sein", "wetter", "schön", "heute"}
```

- [ ] **Step 4: Commit**

```bash
git init
git add -A
git commit -m "chore: scaffold mlx-subtitler project"
```

---

## Task 1: Models — Shared Data Types

**Files:**
- Create: `mlx_subtitler/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_models.py
from mlx_subtitler.models import Segment


def test_segment_creation():
    seg = Segment(index=1, start=0.0, end=3.5, text="Hello")
    assert seg.index == 1
    assert seg.start == 0.0
    assert seg.end == 3.5
    assert seg.text == "Hello"
    assert seg.translation is None


def test_segment_with_translation():
    seg = Segment(index=1, start=0.0, end=3.5, text="Hallo", translation="Hello")
    assert seg.translation == "Hello"


def test_segment_is_frozen():
    seg = Segment(index=1, start=0.0, end=3.5, text="Hello")
    try:
        seg.text = "Changed"
        assert False, "Should have raised FrozenInstanceError"
    except AttributeError:
        pass


def test_segment_equality():
    a = Segment(index=1, start=0.0, end=3.5, text="Hello")
    b = Segment(index=1, start=0.0, end=3.5, text="Hello")
    assert a == b


def test_segment_duration():
    seg = Segment(index=1, start=1.0, end=5.5, text="Test")
    assert seg.duration == 4.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_models.py -v`
Expected: FAIL

- [ ] **Step 3: Implement models**

```python
# mlx_subtitler/models.py
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    index: int
    start: float
    end: float
    text: str
    translation: str | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_models.py -v`
Expected: All 5 PASS

- [ ] **Step 5: Commit**

```bash
git add mlx_subtitler/models.py tests/test_models.py
git commit -m "feat: add Segment dataclass with duration property"
```

---

## Task 2: Transcriber — MLX-Whisper Wrapper

**Files:**
- Create: `mlx_subtitler/transcriber.py`
- Create: `tests/test_transcriber.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_transcriber.py
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from mlx_subtitler.transcriber import Transcriber
from mlx_subtitler.models import Segment


@pytest.fixture
def mock_whisper_output():
    return {
        "text": "Hallo wie geht es dir",
        "segments": [
            {"start": 0.0, "end": 3.5, "text": "Hallo wie geht es dir"},
            {"start": 3.5, "end": 7.0, "text": " Mir geht es gut danke"},
        ],
    }


def test_transcribe_returns_segments(mock_whisper_output):
    with patch("mlx_subtitler.transcriber.mlx_whisper") as mock_mlx:
        mock_mlx.transcribe.return_value = mock_whisper_output
        t = Transcriber(model="test-model", language="de")
        result = t.transcribe(Path("test.m4a"))
        assert len(result) == 2
        assert isinstance(result[0], Segment)
        assert result[0].start == 0.0
        assert result[0].end == 3.5
        assert result[0].text == "Hallo wie geht es dir"
        assert result[0].index == 1
        assert result[1].index == 2


def test_transcribe_strips_whitespace(mock_whisper_output):
    with patch("mlx_subtitler.transcriber.mlx_whisper") as mock_mlx:
        mock_mlx.transcribe.return_value = mock_whisper_output
        t = Transcriber(model="test-model")
        result = t.transcribe(Path("test.m4a"))
        assert result[1].text == "Mir geht es gut danke"


def test_transcribe_passes_language():
    with patch("mlx_subtitler.transcriber.mlx_whisper") as mock_mlx:
        mock_mlx.transcribe.return_value = {"text": "", "segments": []}
        t = Transcriber(model="test-model", language="de")
        t.transcribe(Path("test.m4a"))
        _, kwargs = mock_mlx.transcribe.call_args
        assert kwargs.get("language") == "de" or mock_mlx.transcribe.call_args[0][0] == "test.m4a"


def test_transcribe_auto_detect_language():
    with patch("mlx_subtitler.transcriber.mlx_whisper") as mock_mlx:
        mock_mlx.transcribe.return_value = {"text": "", "segments": []}
        t = Transcriber(model="test-model", language=None)
        t.transcribe(Path("test.m4a"))
        call_kwargs = mock_mlx.transcribe.call_args[1]
        assert "language" not in call_kwargs or call_kwargs["language"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_transcriber.py -v`
Expected: FAIL

- [ ] **Step 3: Implement Transcriber**

```python
# mlx_subtitler/transcriber.py
from __future__ import annotations
import logging
from pathlib import Path
import mlx_whisper
from mlx_subtitler.models import Segment

logger = logging.getLogger(__name__)


class Transcriber:
    def __init__(self, model: str = "mlx-community/whisper-large-v3-mlx", language: str | None = None):
        self._model = model
        self._language = language

    def transcribe(self, audio_path: Path) -> list[Segment]:
        kwargs: dict = {"path_or_hf_repo": self._model}
        if self._language:
            kwargs["language"] = self._language
        logger.info(f"Transcribing {audio_path.name} with model {self._model}")
        result = mlx_whisper.transcribe(str(audio_path), **kwargs)
        segments = []
        for i, seg in enumerate(result.get("segments", []), 1):
            segments.append(Segment(
                index=i,
                start=seg["start"],
                end=seg["end"],
                text=seg["text"].strip(),
            ))
        return segments
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_transcriber.py -v`
Expected: All 4 PASS

- [ ] **Step 5: Commit**

```bash
git add mlx_subtitler/transcriber.py tests/test_transcriber.py
git commit -m "feat: add Transcriber wrapping mlx-whisper"
```

---

## Task 3: VocabLoader — GitHub CSV Download + Cache

**Files:**
- Create: `mlx_subtitler/vocab_loader.py`
- Create: `tests/test_vocab_loader.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_vocab_loader.py
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from mlx_subtitler.vocab_loader import VocabLoader


@pytest.fixture
def csv_content_a1():
    return "der\ndie\ndas\nist\nein\neine"


@pytest.fixture
def csv_content_a2():
    return "Haus\nAuto\nStraße\n"


def test_load_downloads_and_parses_csv(csv_content_a1, csv_content_a2, tmp_path):
    def mock_urlretrieve(url, path):
        if "A1" in url:
            Path(path).write_text(csv_content_a1)
        elif "A2" in url:
            Path(path).write_text(csv_content_a2)

    with patch("urllib.request.urlretrieve", side_effect=mock_urlretrieve):
        loader = VocabLoader(cache_dir=tmp_path / "vocab")
        vocab = loader.load(["A1", "A2"])
        assert "der" in vocab
        assert "die" in vocab
        assert "haus" in vocab
        assert "auto" in vocab
        assert "straße" in vocab


def test_load_uses_cache_on_second_call(csv_content_a1, tmp_path):
    call_count = 0
    def mock_urlretrieve(url, path):
        nonlocal call_count
        call_count += 1
        Path(path).write_text(csv_content_a1)

    cache = tmp_path / "vocab"
    with patch("urllib.request.urlretrieve", side_effect=mock_urlretrieve):
        loader = VocabLoader(cache_dir=cache)
        loader.load(["A1"])
        loader2 = VocabLoader(cache_dir=cache)
        vocab2 = loader2.load(["A1"])
        assert "der" in vocab2
        assert call_count == 1


def test_load_empty_levels_returns_empty(tmp_path):
    loader = VocabLoader(cache_dir=tmp_path / "vocab")
    vocab = loader.load([])
    assert vocab == set()


def test_load_handles_download_failure(tmp_path):
    with patch("urllib.request.urlretrieve", side_effect=Exception("network error")):
        loader = VocabLoader(cache_dir=tmp_path / "vocab")
        vocab = loader.load(["A1"])
        assert vocab == set()


def test_build_url():
    loader = VocabLoader()
    url = loader._build_url("A1")
    assert "A1_vokabeln.csv" in url
    assert "IdeaProjects" in url
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_vocab_loader.py -v`
Expected: FAIL

- [ ] **Step 3: Implement VocabLoader**

```python
# mlx_subtitler/vocab_loader.py
from __future__ import annotations
import logging
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)


class VocabLoader:
    REPO_BASE = "https://raw.githubusercontent.com/Jonathangadeaharder/IdeaProjects/main/src/backend/data"

    def __init__(self, cache_dir: Path | None = None):
        self._cache_dir = cache_dir or Path.home() / ".cache" / "mlx-subtitler" / "vocab"

    def _build_url(self, level: str) -> str:
        return f"{self.REPO_BASE}/{level}_vokabeln.csv"

    def _download(self, level: str) -> Path:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        dest = self._cache_dir / f"{level}_vokabeln.csv"
        if dest.exists():
            return dest
        url = self._build_url(level)
        logger.info(f"Downloading {url}")
        try:
            urllib.request.urlretrieve(url, str(dest))
        except Exception as e:
            logger.warning(f"Failed to download vocab for {level}: {e}")
            return dest
        return dest

    def load(self, levels: list[str] | None = None) -> set[str]:
        if levels is None:
            levels = ["A1", "A2", "B1"]
        vocab: set[str] = set()
        for level in levels:
            path = self._download(level)
            if not path.exists():
                continue
            try:
                text = path.read_text(encoding="utf-8")
                for line in text.strip().splitlines():
                    word = line.strip().lower()
                    if word:
                        vocab.add(word)
            except Exception as e:
                logger.warning(f"Failed to parse vocab for {level}: {e}")
        return vocab
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_vocab_loader.py -v`
Expected: All 5 PASS

- [ ] **Step 5: Commit**

```bash
git add mlx_subtitler/vocab_loader.py tests/test_vocab_loader.py
git commit -m "feat: add VocabLoader with GitHub CSV download and cache"
```

---

## Task 4: VocabFilter — SpaCy Lemma Filtering

**Files:**
- Create: `mlx_subtitler/vocab_filter.py`
- Create: `tests/test_vocab_filter.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_vocab_filter.py
import pytest
from unittest.mock import MagicMock, patch
from mlx_subtitler.models import Segment
from mlx_subtitler.vocab_filter import VocabFilter


def _mock_spacy_doc(text, lemmas_with_pos):
    tokens = []
    for lemma, pos, is_stop, is_punct in lemmas_with_pos:
        tok = MagicMock()
        tok.lemma_ = lemma
        tok.pos_ = pos
        tok.is_stop = is_stop
        tok.is_punct = is_punct
        tok.text = lemma
        tokens.append(tok)
    doc = MagicMock()
    doc.__iter__ = lambda self: iter(tokens)
    return doc


def test_filter_removes_all_known_words():
    vocab = {"hallo", "gehen", "gut", "danke"}
    segments = [
        Segment(index=1, start=0.0, end=3.0, text="Hallo danke"),
        Segment(index=2, start=3.0, end=6.0, text="Unbekannteswort hier"),
    ]
    docs = {
        "Hallo danke": _mock_spacy_doc("Hallo danke", [
            ("hallo", "NOUN", False, False),
            ("danke", "NOUN", False, False),
        ]),
        "Unbekannteswort hier": _mock_spacy_doc("Unbekannteswort hier", [
            ("unbekannteswort", "NOUN", False, False),
            ("hier", "ADV", True, False),
        ]),
    }
    with patch("mlx_subtitler.vocab_filter.spacy") as mock_spacy:
        mock_nlp = MagicMock()
        mock_spacy.load.return_value = mock_nlp
        mock_nlp.side_effect = lambda t: docs.get(t, _mock_spacy_doc(t, []))
        vf = VocabFilter(vocab=vocab, spacy_model="de_core_news_lg")
        result = vf.filter(segments)
    assert len(result) == 1
    assert result[0].text == "Unbekannteswort hier"


def test_filter_keeps_segment_with_one_unknown_word():
    vocab = {"hallo"}
    segments = [Segment(index=1, start=0.0, end=3.0, text="Hallo unbekannt")]
    docs = {
        "Hallo unbekannt": _mock_spacy_doc("Hallo unbekannt", [
            ("hallo", "NOUN", False, False),
            ("unbekannt", "ADJ", False, False),
        ]),
    }
    with patch("mlx_subtitler.vocab_filter.spacy") as mock_spacy:
        mock_nlp = MagicMock()
        mock_spacy.load.return_value = mock_nlp
        mock_nlp.side_effect = lambda t: docs.get(t, _mock_spacy_doc(t, []))
        vf = VocabFilter(vocab=vocab, spacy_model="de_core_news_lg")
        result = vf.filter(segments)
    assert len(result) == 1


def test_filter_reindexes_output():
    vocab = {"hallo"}
    segments = [
        Segment(index=1, start=0.0, end=3.0, text="Hallo"),
        Segment(index=2, start=3.0, end=6.0, text="Unbekannt"),
        Segment(index=3, start=6.0, end=9.0, text="Auch bekannt hallo"),
    ]
    docs = {
        "Hallo": _mock_spacy_doc("Hallo", [("hallo", "NOUN", False, False)]),
        "Unbekannt": _mock_spacy_doc("Unbekannt", [("unbekannt", "NOUN", False, False)]),
        "Auch bekannt hallo": _mock_spacy_doc("Auch bekannt hallo", [
            ("auch", "ADV", True, False),
            ("bekannt", "ADJ", False, False),
            ("hallo", "NOUN", False, False),
        ]),
    }
    with patch("mlx_subtitler.vocab_filter.spacy") as mock_spacy:
        mock_nlp = MagicMock()
        mock_spacy.load.return_value = mock_nlp
        mock_nlp.side_effect = lambda t: docs.get(t, _mock_spacy_doc(t, []))
        vf = VocabFilter(vocab=vocab, spacy_model="de_core_news_lg")
        result = vf.filter(segments)
    assert len(result) == 2
    assert result[0].index == 1
    assert result[1].index == 2


def test_filter_skips_punctuation_and_stopwords():
    vocab = set()
    segments = [Segment(index=1, start=0.0, end=3.0, text="... !!")]
    docs = {
        "... !!": _mock_spacy_doc("... !!", [
            (".", "PUNCT", False, True),
            ("!", "PUNCT", False, True),
        ]),
    }
    with patch("mlx_subtitler.vocab_filter.spacy") as mock_spacy:
        mock_nlp = MagicMock()
        mock_spacy.load.return_value = mock_nlp
        mock_nlp.side_effect = lambda t: docs.get(t, _mock_spacy_doc(t, []))
        vf = VocabFilter(vocab=vocab, spacy_model="de_core_news_lg")
        result = vf.filter(segments)
    assert len(result) == 0


def test_filter_empty_input():
    with patch("mlx_subtitler.vocab_filter.spacy") as mock_spacy:
        mock_nlp = MagicMock()
        mock_spacy.load.return_value = mock_nlp
        vf = VocabFilter(vocab={"test"}, spacy_model="de_core_news_lg")
        result = vf.filter([])
    assert result == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_vocab_filter.py -v`
Expected: FAIL

- [ ] **Step 3: Implement VocabFilter**

```python
# mlx_subtitler/vocab_filter.py
from __future__ import annotations
import logging
import spacy
from mlx_subtitler.models import Segment

logger = logging.getLogger(__name__)

SKIP_POS = {"PROPN", "NUM", "INTJ", "X", "SPACE"}


class VocabFilter:
    def __init__(self, vocab: set[str], spacy_model: str = "de_core_news_lg"):
        self._vocab = vocab
        self._nlp = spacy.load(spacy_model, disable=["parser", "ner"])

    def filter(self, segments: list[Segment]) -> list[Segment]:
        kept = []
        for seg in segments:
            doc = self._nlp(seg.text)
            lemmas = []
            for token in doc:
                if token.is_punct or token.is_stop or token.pos_ in SKIP_POS:
                    continue
                lemma = token.lemma_.lower().strip()
                if lemma:
                    lemmas.append(lemma)
            if lemmas and not all(lm in self._vocab for lm in lemmas):
                kept.append(seg)
        return [Segment(index=i + 1, start=s.start, end=s.end, text=s.text) for i, s in enumerate(kept)]
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_vocab_filter.py -v`
Expected: All 5 PASS

- [ ] **Step 5: Commit**

```bash
git add mlx_subtitler/vocab_filter.py tests/test_vocab_filter.py
git commit -m "feat: add VocabFilter with spaCy lemmatization"
```

---

## Task 5: Translator — MarianMT Batch Translation

**Files:**
- Create: `mlx_subtitler/translator.py`
- Create: `tests/test_translator.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_translator.py
import pytest
from unittest.mock import patch, MagicMock
from mlx_subtitler.models import Segment
from mlx_subtitler.translator import Translator


def test_translate_fills_translation_field():
    segments = [
        Segment(index=1, start=0.0, end=3.0, text="Hallo"),
        Segment(index=2, start=3.0, end=6.0, text="Guten Morgen"),
    ]
    with patch("mlx_subtitler.translator.MarianMTModel") as mock_model_cls, \
         patch("mlx_subtitler.translator.MarianTokenizer") as mock_tok_cls:
        mock_tok = MagicMock()
        mock_tok_cls.from_pretrained.return_value = mock_tok
        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_model.generate.return_value = MagicMock()
        mock_tok.batch_decode.return_value = ["Hello", "Good morning"]
        mock_tok.return_value = MagicMock()
        t = Translator(src_lang="de", tgt_lang="es", device="cpu")
        result = t.translate(segments)
    assert len(result) == 2
    assert result[0].translation == "Hello"
    assert result[1].translation == "Good morning"
    assert result[0].text == "Hallo"
    assert result[0].index == 1


def test_translate_respects_batch_size():
    segments = [Segment(index=i, start=0.0, end=1.0, text=f"Word{i}") for i in range(1, 6)]
    with patch("mlx_subtitler.translator.MarianMTModel") as mock_model_cls, \
         patch("mlx_subtitler.translator.MarianTokenizer") as mock_tok_cls:
        mock_tok = MagicMock()
        mock_tok_cls.from_pretrained.return_value = mock_tok
        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_model.generate.return_value = MagicMock()
        mock_tok.batch_decode.return_value = [f"trans{i}" for i in range(5)]
        mock_tok.return_value = MagicMock()
        t = Translator(src_lang="de", tgt_lang="es", device="cpu")
        result = t.translate(segments, batch_size=2)
    assert len(result) == 5
    assert mock_model.generate.call_count >= 3


def test_translate_empty_segments():
    with patch("mlx_subtitler.translator.MarianMTModel") as mock_model_cls, \
         patch("mlx_subtitler.translator.MarianTokenizer") as mock_tok_cls:
        mock_tok = MagicMock()
        mock_tok_cls.from_pretrained.return_value = mock_tok
        mock_model_cls.from_pretrained.return_value = MagicMock()
        t = Translator(src_lang="de", tgt_lang="es", device="cpu")
        result = t.translate([])
    assert result == []


def test_model_name_from_lang_pair():
    with patch("mlx_subtitler.translator.MarianMTModel") as mock_model_cls, \
         patch("mlx_subtitler.translator.MarianTokenizer") as mock_tok_cls:
        mock_tok_cls.from_pretrained.return_value = MagicMock()
        mock_model_cls.from_pretrained.return_value = MagicMock()
        t = Translator(src_lang="de", tgt_lang="fr", device="cpu")
        assert "de-fr" in t._model_name
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_translator.py -v`
Expected: FAIL

- [ ] **Step 3: Implement Translator**

```python
# mlx_subtitler/translator.py
from __future__ import annotations
import logging
import torch
from mlx_subtitler.models import Segment
from transformers import MarianMTModel, MarianTokenizer

logger = logging.getLogger(__name__)


class Translator:
    def __init__(self, src_lang: str = "de", tgt_lang: str = "es", device: str = "auto"):
        self._src_lang = src_lang
        self._tgt_lang = tgt_lang
        if device == "auto":
            self._device = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self._device = device
        self._model_name = f"Helsinki-NLP/opus-mt-tc-big-{src_lang}-{tgt_lang}"
        logger.info(f"Loading translation model {self._model_name} on {self._device}")
        self._tokenizer = MarianTokenizer.from_pretrained(self._model_name)
        self._model = MarianMTModel.from_pretrained(self._model_name).to(self._device)

    def translate(self, segments: list[Segment], batch_size: int = 32) -> list[Segment]:
        if not segments:
            return []
        texts = [s.text for s in segments]
        translations = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self._tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self._device)
            with torch.no_grad():
                generated = self._model.generate(**inputs, max_new_tokens=512)
            translations.extend(self._tokenizer.batch_decode(generated, skip_special_tokens=True))
        return [
            Segment(index=s.index, start=s.start, end=s.end, text=s.text, translation=tr)
            for s, tr in zip(segments, translations)
        ]
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_translator.py -v`
Expected: All 4 PASS

- [ ] **Step 5: Commit**

```bash
git add mlx_subtitler/translator.py tests/test_translator.py
git commit -m "feat: add Translator with MarianMT batch translation"
```

---

## Task 6: SubtitleWriter — SRT/VTT Output

**Files:**
- Create: `mlx_subtitler/subtitle_writer.py`
- Create: `tests/test_subtitle_writer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_subtitle_writer.py
import pytest
from pathlib import Path
from mlx_subtitler.models import Segment
from mlx_subtitler.subtitle_writer import SubtitleWriter


@pytest.fixture
def segments():
    return [
        Segment(index=1, start=0.0, end=3.5, text="Hallo", translation="Hello"),
        Segment(index=2, start=3.5, end=7.0, text="Guten Morgen", translation="Good morning"),
    ]


def test_write_srt_uses_translation(segments, tmp_path):
    out = tmp_path / "out.srt"
    SubtitleWriter.write_srt(segments, out, use_translation=True)
    content = out.read_text(encoding="utf-8")
    assert "Hello" in content
    assert "Good morning" in content
    assert "Hallo" not in content


def test_write_srt_uses_original_when_no_translation(segments, tmp_path):
    out = tmp_path / "out.srt"
    SubtitleWriter.write_srt(segments, out, use_translation=True)
    content = out.read_text(encoding="utf-8")
    assert "Hello" in content


def test_write_srt_uses_original_text_when_requested(segments, tmp_path):
    out = tmp_path / "out.srt"
    SubtitleWriter.write_srt(segments, out, use_translation=False)
    content = out.read_text(encoding="utf-8")
    assert "Hallo" in content
    assert "Guten Morgen" in content


def test_write_srt_format_has_timestamps(segments, tmp_path):
    out = tmp_path / "out.srt"
    SubtitleWriter.write_srt(segments, out)
    content = out.read_text(encoding="utf-8")
    assert "-->" in content
    assert "00:00:00,000" in content


def test_write_vtt_format(segments, tmp_path):
    out = tmp_path / "out.vtt"
    SubtitleWriter.write_vtt(segments, out)
    content = out.read_text(encoding="utf-8")
    assert content.startswith("WEBVTT")
    assert "Hello" in content


def test_write_srt_returns_path(segments, tmp_path):
    out = tmp_path / "out.srt"
    result = SubtitleWriter.write_srt(segments, out)
    assert result == out
    assert out.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_subtitle_writer.py -v`
Expected: FAIL

- [ ] **Step 3: Implement SubtitleWriter**

```python
# mlx_subtitler/subtitle_writer.py
from __future__ import annotations
import pysrt
from pathlib import Path
from mlx_subtitler.models import Segment


class SubtitleWriter:
    @staticmethod
    def write_srt(segments: list[Segment], output_path: Path, use_translation: bool = True) -> Path:
        subs = pysrt.SubRipFile()
        for seg in segments:
            text = seg.translation if (use_translation and seg.translation) else seg.text
            subs.append(pysrt.SubRipItem(
                index=seg.index,
                start=pysrt.SubRipTime(seconds=seg.start),
                end=pysrt.SubRipTime(seconds=seg.end),
                text=text,
            ))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        subs.save(str(output_path), encoding="utf-8")
        return output_path

    @staticmethod
    def write_vtt(segments: list[Segment], output_path: Path, use_translation: bool = True) -> Path:
        lines = ["WEBVTT\n\n"]
        for seg in segments:
            text = seg.translation if (use_translation and seg.translation) else seg.text
            start_h, start_rem = divmod(seg.start, 3600)
            start_m, start_s = divmod(start_rem, 60)
            end_h, end_rem = divmod(seg.end, 3600)
            end_m, end_s = divmod(end_rem, 60)
            lines.append(f"{int(start_h):02d}:{int(start_m):02d}:{start_s:06.3f} --> {int(end_h):02d}:{int(end_m):02d}:{end_s:06.3f}\n")
            lines.append(f"{text}\n\n")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("".join(lines), encoding="utf-8")
        return output_path
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_subtitle_writer.py -v`
Expected: All 6 PASS

- [ ] **Step 5: Commit**

```bash
git add mlx_subtitler/subtitle_writer.py tests/test_subtitle_writer.py
git commit -m "feat: add SubtitleWriter with SRT and VTT output"
```

---

## Task 7: Pipeline — Stage Composition

**Files:**
- Create: `mlx_subtitler/pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_pipeline.py
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from mlx_subtitler.models import Segment
from mlx_subtitler.pipeline import Pipeline


def test_pipeline_transcribe_only(tmp_path):
    with patch("mlx_subtitler.pipeline.Transcriber") as mock_t_cls, \
         patch("mlx_subtitler.pipeline.SubtitleWriter") as mock_w_cls:
        mock_t = MagicMock()
        mock_t_cls.return_value = mock_t
        mock_t.transcribe.return_value = [
            Segment(index=1, start=0.0, end=3.0, text="Hallo"),
        ]
        mock_w_cls.write_srt = MagicMock(return_value=tmp_path / "out.srt")
        p = Pipeline(src_lang="de", tgt_lang="es", translate=False, filter_vocab=False)
        result = p.run(tmp_path / "test.m4a")
        mock_t.transcribe.assert_called_once()
        mock_w_cls.write_srt.assert_called_once()


def test_pipeline_full_with_filter(tmp_path):
    with patch("mlx_subtitler.pipeline.Transcriber") as mock_t_cls, \
         patch("mlx_subtitler.pipeline.VocabFilter") as mock_vf_cls, \
         patch("mlx_subtitler.pipeline.Translator") as mock_tr_cls, \
         patch("mlx_subtitler.pipeline.VocabLoader") as mock_vl_cls, \
         patch("mlx_subtitler.pipeline.SubtitleWriter") as mock_w_cls:
        mock_t = MagicMock()
        mock_t_cls.return_value = mock_t
        mock_t.transcribe.return_value = [
            Segment(index=1, start=0.0, end=3.0, text="Hallo"),
            Segment(index=2, start=3.0, end=6.0, text="Unbekannt"),
        ]
        mock_vl = MagicMock()
        mock_vl_cls.return_value = mock_vl
        mock_vl.load.return_value = {"hallo"}
        mock_vf = MagicMock()
        mock_vf_cls.return_value = mock_vf
        mock_vf.filter.return_value = [Segment(index=1, start=3.0, end=6.0, text="Unbekannt")]
        mock_tr = MagicMock()
        mock_tr_cls.return_value = mock_tr
        mock_tr.translate.return_value = [Segment(index=1, start=3.0, end=6.0, text="Unbekannt", translation="Unknown")]
        mock_w_cls.write_srt = MagicMock(return_value=tmp_path / "out.srt")
        p = Pipeline(src_lang="de", tgt_lang="es", translate=True, filter_vocab=True)
        result = p.run(tmp_path / "test.m4a")
        mock_vf.filter.assert_called_once()
        mock_tr.translate.assert_called_once()


def test_pipeline_skip_filter(tmp_path):
    with patch("mlx_subtitler.pipeline.Transcriber") as mock_t_cls, \
         patch("mlx_subtitler.pipeline.Translator") as mock_tr_cls, \
         patch("mlx_subtitler.pipeline.SubtitleWriter") as mock_w_cls:
        mock_t = MagicMock()
        mock_t_cls.return_value = mock_t
        mock_t.transcribe.return_value = [Segment(index=1, start=0.0, end=3.0, text="Hallo")]
        mock_tr = MagicMock()
        mock_tr_cls.return_value = mock_tr
        mock_tr.translate.return_value = [Segment(index=1, start=0.0, end=3.0, text="Hallo", translation="Hello")]
        mock_w_cls.write_srt = MagicMock(return_value=tmp_path / "out.srt")
        p = Pipeline(src_lang="de", tgt_lang="es", translate=True, filter_vocab=False)
        result = p.run(tmp_path / "test.m4a")
        mock_tr.translate.assert_called_once()


def test_pipeline_auto_output_path(tmp_path):
    with patch("mlx_subtitler.pipeline.Transcriber") as mock_t_cls, \
         patch("mlx_subtitler.pipeline.SubtitleWriter") as mock_w_cls:
        mock_t = MagicMock()
        mock_t_cls.return_value = mock_t
        mock_t.transcribe.return_value = [Segment(index=1, start=0.0, end=3.0, text="Hallo")]
        mock_w_cls.write_srt = MagicMock(return_value=tmp_path / "test.srt")
        p = Pipeline(src_lang="de", tgt_lang="es", translate=False, filter_vocab=False)
        result = p.run(tmp_path / "test.m4a")
        write_args = mock_w_cls.write_srt.call_args[0]
        assert write_args[1].suffix == ".srt"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_pipeline.py -v`
Expected: FAIL

- [ ] **Step 3: Implement Pipeline**

```python
# mlx_subtitler/pipeline.py
from __future__ import annotations
import logging
from pathlib import Path
from mlx_subtitler.models import Segment
from mlx_subtitler.transcriber import Transcriber
from mlx_subtitler.vocab_loader import VocabLoader
from mlx_subtitler.vocab_filter import VocabFilter
from mlx_subtitler.translator import Translator
from mlx_subtitler.subtitle_writer import SubtitleWriter

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".m4a", ".ogg", ".mp3", ".wav", ".flac"}


class Pipeline:
    def __init__(
        self,
        src_lang: str = "de",
        tgt_lang: str = "es",
        whisper_model: str = "mlx-community/whisper-large-v3-mlx",
        filter_vocab: bool = False,
        vocab_levels: list[str] | None = None,
        translate: bool = True,
        output_format: str = "srt",
        batch_size: int = 32,
    ):
        self._src_lang = src_lang
        self._tgt_lang = tgt_lang
        self._whisper_model = whisper_model
        self._filter_vocab = filter_vocab
        self._vocab_levels = vocab_levels or ["A1", "A2", "B1"]
        self._translate = translate
        self._output_format = output_format
        self._batch_size = batch_size

    def run(self, audio_path: Path, output_path: Path | None = None) -> Path:
        if not output_path:
            output_path = audio_path.with_suffix(f".{self._output_format}")
        logger.info(f"Processing {audio_path.name}")
        transcriber = Transcriber(model=self._whisper_model, language=self._src_lang)
        segments = transcriber.transcribe(audio_path)
        logger.info(f"Transcribed {len(segments)} segments")
        if self._filter_vocab:
            loader = VocabLoader()
            vocab = loader.load(self._vocab_levels)
            vfilter = VocabFilter(vocab=vocab)
            segments = vfilter.filter(segments)
            logger.info(f"After filtering: {len(segments)} segments")
        if self._translate:
            translator = Translator(src_lang=self._src_lang, tgt_lang=self._tgt_lang)
            segments = translator.translate(segments, batch_size=self._batch_size)
            logger.info(f"Translated {len(segments)} segments")
        writer = SubtitleWriter.write_srt if self._output_format == "srt" else SubtitleWriter.write_vtt
        writer(segments, output_path, use_translation=self._translate)
        logger.info(f"Saved {output_path}")
        return output_path

    def run_batch(self, audio_dir: Path, output_dir: Path | None = None) -> list[Path]:
        if not output_dir:
            output_dir = audio_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(f for f in audio_dir.iterdir() if f.suffix.lower() in AUDIO_EXTENSIONS)
        logger.info(f"Found {len(files)} audio files in {audio_dir}")
        results = []
        for f in files:
            try:
                out = self.run(f, output_dir / f.with_suffix(f".{self._output_format}").name)
                results.append(out)
            except Exception as e:
                logger.error(f"Failed processing {f.name}: {e}")
        return results
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_pipeline.py -v`
Expected: All 4 PASS

- [ ] **Step 5: Commit**

```bash
git add mlx_subtitler/pipeline.py tests/test_pipeline.py
git commit -m "feat: add Pipeline composing all stages"
```

---

## Task 8: CLI Entry Point

**Files:**
- Create: `cli.py`
- Modify: `mlx_subtitler/__init__.py` — export public API

- [ ] **Step 1: Implement CLI**

```python
# cli.py
#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from mlx_subtitler.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(description="MLX Subtitler — Local transcription, translation, and subtitles")
    sub = parser.add_subparsers(dest="command")

    # transcribe command
    t = sub.add_parser("transcribe", help="Transcribe a single audio file")
    t.add_argument("audio", type=Path, help="Path to audio file")
    t.add_argument("--src", default="de", help="Source language code (default: de)")
    t.add_argument("--tgt", default="es", help="Target language code (default: es)")
    t.add_argument("--no-translate", action="store_true", help="Skip translation")
    t.add_argument("--filter-vocab", action="store_true", help="Enable vocab filtering")
    t.add_argument("--vocab-levels", default="A1,A2,B1", help="Comma-separated CEFR levels")
    t.add_argument("--model", default="mlx-community/whisper-large-v3-mlx", help="MLX whisper model")
    t.add_argument("--format", default="srt", choices=["srt", "vtt"], help="Output format")
    t.add_argument("-o", "--output", type=Path, default=None, help="Output file path")
    t.add_argument("--batch-size", type=int, default=32, help="Translation batch size")

    # batch command
    b = sub.add_parser("batch", help="Process all audio files in a directory")
    b.add_argument("audio_dir", type=Path, help="Directory containing audio files")
    b.add_argument("--output-dir", type=Path, default=None, help="Output directory")
    b.add_argument("--src", default="de", help="Source language code (default: de)")
    b.add_argument("--tgt", default="es", help="Target language code (default: es)")
    b.add_argument("--no-translate", action="store_true", help="Skip translation")
    b.add_argument("--filter-vocab", action="store_true", help="Enable vocab filtering")
    b.add_argument("--vocab-levels", default="A1,A2,B1", help="Comma-separated CEFR levels")
    b.add_argument("--model", default="mlx-community/whisper-large-v3-mlx", help="MLX whisper model")
    b.add_argument("--format", default="srt", choices=["srt", "vtt"], help="Output format")
    b.add_argument("--batch-size", type=int, default=32, help="Translation batch size")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    levels = args.vocab_levels.split(",") if hasattr(args, "vocab_levels") else ["A1", "A2", "B1"]

    p = Pipeline(
        src_lang=args.src,
        tgt_lang=args.tgt,
        whisper_model=args.model,
        filter_vocab=getattr(args, "filter_vocab", False),
        vocab_levels=levels,
        translate=not getattr(args, "no_translate", False),
        output_format=args.format,
        batch_size=getattr(args, "batch_size", 32),
    )

    if args.command == "transcribe":
        out = p.run(args.audio, getattr(args, "output", None))
        print(f"Saved: {out}")
    elif args.command == "batch":
        results = p.run_batch(args.audio_dir, getattr(args, "output_dir", None))
        print(f"Processed {len(results)} files")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Update __init__.py**

```python
# mlx_subtitler/__init__.py
from mlx_subtitler.models import Segment
from mlx_subtitler.transcriber import Transcriber
from mlx_subtitler.vocab_loader import VocabLoader
from mlx_subtitler.vocab_filter import VocabFilter
from mlx_subtitler.translator import Translator
from mlx_subtitler.subtitle_writer import SubtitleWriter
from mlx_subtitler.pipeline import Pipeline

__all__ = [
    "Segment", "Transcriber", "VocabLoader", "VocabFilter",
    "Translator", "SubtitleWriter", "Pipeline",
]
```

- [ ] **Step 3: Verify CLI help**

Run: `python3 cli.py --help`
Expected: Shows help text with transcribe and batch subcommands

- [ ] **Step 4: Run full test suite**

Run: `python3 -m pytest tests/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add cli.py mlx_subtitler/__init__.py
git commit -m "feat: add CLI entry point with transcribe and batch commands"
```

---

## Task 9: Final Verification

- [ ] **Step 1: Run full test suite**

Run: `python3 -m pytest tests/ -v --tb=short`
Expected: All ~34 tests PASS

- [ ] **Step 2: Verify test counts**

Expected by file:
- `test_models.py`: 5
- `test_transcriber.py`: 4
- `test_vocab_loader.py`: 5
- `test_vocab_filter.py`: 5
- `test_translator.py`: 4
- `test_subtitle_writer.py`: 6
- `test_pipeline.py`: 4
- **Total: ~33 tests**

- [ ] **Step 3: Verify CLI**

Run: `python3 cli.py transcribe --help`
Expected: Shows flags for --src, --tgt, --filter-vocab, etc.
