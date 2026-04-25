#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from mlx_subtitler.pipeline import Pipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlx-subtitler",
        description="MLX-powered transcription, vocab filtering, and translation pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--src", default="de", help="Source language code (default: de)")
    common.add_argument("--tgt", default="es", help="Target language code (default: es)")
    common.add_argument("--model", default="mlx-community/whisper-large-v3-turbo", help="Whisper model")
    common.add_argument("--no-translate", action="store_true", help="Skip translation")
    common.add_argument("--no-filter", dest="filter_vocab", action="store_false", help="Skip vocab filtering")
    common.add_argument("--vocab-levels", nargs="+", default=None, help="Vocab levels (e.g. B1 B2)")
    common.add_argument("--format", dest="output_format", default="srt", choices=["srt", "vtt"], help="Output format")
    common.add_argument("-o", "--output", default=None, help="Output file/directory path")
    common.add_argument("--batch-size", type=int, default=32, help="Translation batch size")

    transcribe_parser = subparsers.add_parser("transcribe", parents=[common], help="Transcribe a single audio file")
    transcribe_parser.add_argument("audio", type=Path, help="Path to audio file")

    batch_parser = subparsers.add_parser("batch", parents=[common], help="Batch process a directory")
    batch_parser.add_argument("audio_dir", type=Path, help="Directory containing audio files")

    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = _build_parser()
    args = parser.parse_args(argv)

    pipeline = Pipeline(
        src_lang=args.src,
        tgt_lang=args.tgt,
        whisper_model=args.model,
        filter_vocab=args.filter_vocab,
        vocab_levels=args.vocab_levels,
        translate=not args.no_translate,
        output_format=args.output_format,
        batch_size=args.batch_size,
    )

    if args.command == "transcribe":
        output = Path(args.output) if args.output else None
        result = pipeline.run(args.audio, output)
        print(f"Output: {result}")
    elif args.command == "batch":
        output_dir = Path(args.output) if args.output else None
        results = pipeline.run_batch(args.audio_dir, output_dir)
        for r in results:
            print(f"Output: {r}")


if __name__ == "__main__":
    main()
