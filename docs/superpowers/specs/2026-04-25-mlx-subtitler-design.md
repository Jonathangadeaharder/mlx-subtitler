# MLX Subtitler — Design Spec

## Goal

A modular, local, Apple Silicon-optimized transcription + translation + vocab-filtering pipeline that converts audio files into subtitle files (SRT/VTT). Replaces the Colab-based `subtitle_ultimate.py` with a reusable Python library and CLI tool.

## Architecture

Four-stage pipeline with decoupled modules, each independently testable and usable:

```
Audio → [Transcriber] → [VocabFilter] → [Translator] → [SubtitleWriter] → .srt/.vtt
              ↓                ↓                 ↓                ↓
        mlx-whisper        spaCy + CSV     MarianMT (MPS)      pysrt
```

Each stage produces a `list[Segment]` (or filtered subset), passing data through a shared dataclass. The `Pipeline` class composes stages. The CLI wraps the pipeline.

## Tech Stack

- **Transcription:** `mlx-whisper` with `mlx-community/whisper-large-v3-mlx` model
- **Lemmatization:** `spacy` with `de_core_news_lg` (German)
- **Translation:** `transformers` MarianMT (`Helsinki-NLP/opus-mt-tc-big-{src}-{tgt}`) on MPS/CPU
- **Subtitle output:** `pysrt`
- **Language:** Python 3.10+
- **Hardware target:** Apple Silicon (MPS backend), unified memory
- **Testing:** pytest with mocked models

## File Structure

```
mlx_subtitler/
├── mlx_subtitler/
│   ├── __init__.py           # Package init, exports public API
│   ├── models.py             # Dataclasses: Segment, SubtitleResult
│   ├── transcriber.py        # Transcriber class wrapping mlx-whisper
│   ├── vocab_loader.py       # Download + cache vocab CSVs from GitHub
│   ├── vocab_filter.py       # VocabFilter: spaCy lemmatize + filter by vocab set
│   ├── translator.py         # Translator: MarianMT batch translation, configurable lang pair
│   ├── subtitle_writer.py    # SubtitleWriter: write SRT/VTT from segments
│   └── pipeline.py           # Pipeline: compose stages, run full pipeline
├── tests/
│   ├── __init__.py
│   ├── conftest.py           # Shared fixtures (sample segments, mock models)
│   ├── test_models.py        # Tests for dataclasses
│   ├── test_transcriber.py   # Tests for Transcriber (mocked mlx-whisper)
│   ├── test_vocab_loader.py  # Tests for CSV download + parsing
│   ├── test_vocab_filter.py  # Tests for filtering logic
│   ├── test_translator.py    # Tests for Translator (mocked MarianMT)
│   ├── test_subtitle_writer.py # Tests for SRT/VTT output
│   └── test_pipeline.py      # Integration tests between stages
├── cli.py                    # CLI entry point
├── pyproject.toml            # Project config, dependencies
└── docs/
    └── superpowers/
        └── specs/
            └── 2026-04-25-mlx-subtitler-design.md  # This file
```

## Module Specifications

### models.py — Shared Data Types

```python
@dataclass(frozen=True)
class Segment:
    index: int          # 1-based subtitle index
    start: float        # Start time in seconds
    end: float          # End time in seconds
    text: str           # Segment text (original language)
    translation: str | None = None  # Translated text (if translated)
```

All modules pass `list[Segment]` between stages. The `translation` field is `None` until the Translator stage fills it.

### transcriber.py — MLX-Whisper Transcription

```python
class Transcriber:
    def __init__(self, model: str = "mlx-community/whisper-large-v3-mlx", language: str | None = None):
        """Initialize with MLX model repo. Language auto-detected if None."""

    def transcribe(self, audio_path: Path) -> list[Segment]:
        """Transcribe audio file. Returns segments with timestamps."""
```

- Uses `mlx_whisper.transcribe()` under the hood
- Converts raw whisper output to `Segment` dataclasses
- Language parameter passed through to mlx-whisper (auto-detect if None)

### vocab_loader.py — Vocabulary Loading

```python
class VocabLoader:
    REPO_URL = "https://raw.githubusercontent.com/Jonathangadeaharder/IdeaProjects/main/src/backend/data"

    def __init__(self, cache_dir: Path | None = None):
        """Initialize with optional cache directory for downloaded CSVs."""

    def load(self, levels: list[str] = ["A1", "A2", "B1"]) -> set[str]:
        """Download (if not cached) and load vocab CSVs. Returns set of lowercase words."""
```

- Downloads CSV files from GitHub on first use, caches locally
- Each CSV has words in column 0
- Returns a `set[str]` of lowercase stripped words

### vocab_filter.py — Vocabulary-Based Filtering

```python
class VocabFilter:
    def __init__(self, vocab: set[str], spacy_model: str = "de_core_news_lg"):
        """Initialize with vocab set and spaCy model name."""

    def filter(self, segments: list[Segment]) -> list[Segment]:
        """Remove segments where ALL lemmatized content words are in vocab."""
```

- Uses spaCy for lemmatization
- Skips punctuation, stopwords, proper nouns, numbers, interjections
- Keeps a segment if it has at least one lemma NOT in the vocab set
- Re-indexes output segments (sequential from 1)

### translator.py — MarianMT Translation

```python
class Translator:
    def __init__(self, src_lang: str = "de", tgt_lang: str = "es", device: str = "auto"):
        """Initialize translator. Device: 'auto' uses MPS if available, else CPU."""

    def translate(self, segments: list[Segment], batch_size: int = 32) -> list[Segment]:
        """Translate segment texts. Returns new Segments with translation field filled."""
```

- Loads `Helsinki-NLP/opus-mt-tc-big-{src}-{tgt}` model
- Auto-detects MPS availability (falls back to CPU)
- Batch translation for efficiency
- Returns new `Segment` instances with `translation` populated
- The `text` field keeps the original language text

### subtitle_writer.py — Output Formatting

```python
class SubtitleWriter:
    @staticmethod
    def write_srt(segments: list[Segment], output_path: Path, use_translation: bool = True) -> Path:
        """Write SRT file. If use_translation and translations exist, use translated text."""

    @staticmethod
    def write_vtt(segments: list[Segment], output_path: Path, use_translation: bool = True) -> Path:
        """Write WebVTT file."""
```

- Uses `pysrt` for SRT generation
- Supports both original text and translated text output
- `use_translation=True` writes the `translation` field if available, falls back to `text`

### pipeline.py — Stage Composition

```python
class Pipeline:
    def __init__(
        self,
        src_lang: str = "de",
        tgt_lang: str = "es",
        whisper_model: str = "mlx-community/whisper-large-v3-mlx",
        filter_vocab: bool = False,
        vocab_levels: list[str] = ["A1", "A2", "B1"],
        output_format: str = "srt",
        batch_size: int = 32,
    ):
        """Configure the full pipeline."""

    def run(self, audio_path: Path, output_path: Path | None = None) -> Path:
        """Run full pipeline on a single audio file. Returns path to output subtitle."""

    def run_batch(self, audio_dir: Path, output_dir: Path | None = None) -> list[Path]:
        """Process all audio files in a directory. Returns list of output paths."""
```

- Composes all 4 stages
- `filter_vocab=False` skips the vocab filter stage entirely
- Auto-generates output filename from input (e.g., `audio.m4a` → `audio.srt`)

## CLI

```bash
# Full pipeline with translation
python cli.py transcribe audio.m4a --src de --tgt es -o output.srt

# Transcription only (no translation)
python cli.py transcribe audio.m4a --src de --no-translate

# With vocab filtering
python cli.py transcribe audio.m4a --src de --tgt es --filter-vocab --vocab-levels A1,B1

# Batch processing
python cli.py batch ./audio_folder/ --src de --tgt es --filter-vocab

# Custom whisper model
python cli.py transcribe audio.m4a --model mlx-community/whisper-large-v3-turbo-mlx
```

Flags:
- `--src` / `--tgt`: Source and target language codes (default: `de` / `es`)
- `--no-translate`: Skip translation, output original transcription
- `--filter-vocab`: Enable vocab filtering
- `--vocab-levels`: Comma-separated CEFR levels (default: `A1,A2,B1`)
- `--model`: MLX whisper model repo (default: `mlx-community/whisper-large-v3-mlx`)
- `--format`: Output format `srt` or `vtt` (default: `srt`)
- `-o` / `--output`: Output file path (auto-generated if omitted)
- `--batch-size`: Translation batch size (default: 32)

## Error Handling

- Missing audio file → clear error message, skip in batch mode
- Missing spaCy model → print install instructions, exit
- Missing MarianMT model for language pair → auto-download from HuggingFace
- Vocab CSV download failure → warn and skip filtering
- Partial transcription (empty segments) → warn, produce empty subtitle or skip

## Testing Strategy

Each module tested in isolation with mocked heavy dependencies:
- **Transcriber:** Mock `mlx_whisper.transcribe()` to return fixture segments
- **VocabFilter:** Use small inline vocab sets, no GitHub download needed
- **Translator:** Mock `MarianMTModel.generate()` to return fixture translations
- **SubtitleWriter:** Write to temp files, verify format
- **Pipeline:** Integration tests with all stages mocked
- **VocabLoader:** Mock HTTP requests, test CSV parsing and caching

Target: 30+ tests, all runnable without GPU or model downloads.
