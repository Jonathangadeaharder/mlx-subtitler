from __future__ import annotations

import logging
from pathlib import Path

from mlx_subtitler.models import Segment
from mlx_subtitler.transcriber import Transcriber
from mlx_subtitler.subtitle_writer import SubtitleWriter

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(
        self,
        src_lang: str = "de",
        tgt_lang: str = "es",
        whisper_model: str = "mlx-community/whisper-large-v3-turbo",
        filter_vocab: bool = True,
        vocab_levels: list[str] | None = None,
        translate: bool = True,
        output_format: str = "srt",
        batch_size: int = 32,
    ) -> None:
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.whisper_model = whisper_model
        self.filter_vocab = filter_vocab
        self.vocab_levels = vocab_levels
        self.translate_flag = translate
        self.output_format = output_format
        self.batch_size = batch_size

    def _auto_output_path(self, audio_path: Path) -> Path:
        ext = self.output_format
        return audio_path.with_suffix(f".{ext}")

    def run(self, audio_path: Path, output_path: Path | None = None) -> Path:
        audio_path = Path(audio_path)
        if output_path is None:
            output_path = self._auto_output_path(audio_path)

        transcriber = Transcriber(model=self.whisper_model, language=self.src_lang)
        segments = transcriber.transcribe(audio_path)
        logger.info("Transcribed %d segments", len(segments))

        if self.filter_vocab:
            from mlx_subtitler.vocab_filter import VocabFilter
            from mlx_subtitler.vocab_loader import VocabLoader

            loader = VocabLoader()
            vocab = loader.load(self.vocab_levels)
            vf = VocabFilter(vocab)
            segments = vf.filter(segments)
            logger.info("Filtered to %d segments", len(segments))

        if self.translate_flag:
            from mlx_subtitler.translator import Translator

            translator = Translator(src_lang=self.src_lang, tgt_lang=self.tgt_lang)
            segments = translator.translate(segments, batch_size=self.batch_size)

        writer_fn = SubtitleWriter.write_srt if self.output_format == "srt" else SubtitleWriter.write_vtt
        writer_fn(segments, output_path)
        logger.info("Wrote %s", output_path)
        return output_path

    def run_batch(
        self,
        audio_dir: Path,
        output_dir: Path | None = None,
    ) -> list[Path]:
        audio_dir = Path(audio_dir)
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        extensions = {".m4a", ".mp3", ".wav", ".flac", ".aac"}
        results: list[Path] = []
        for f in sorted(audio_dir.iterdir()):
            if f.suffix.lower() in extensions:
                out = None
                if output_dir is not None:
                    out = output_dir / f.with_suffix(f".{self.output_format}").name
                results.append(self.run(f, out))
        return results
