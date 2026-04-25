from __future__ import annotations

from pathlib import Path

import mlx_whisper

from mlx_subtitler.models import Segment


class Transcriber:
    def __init__(
        self,
        model: str = "mlx-community/whisper-large-v3-turbo",
        language: str | None = None,
    ) -> None:
        self.model = model
        self.language = language

    def transcribe(self, audio_path: Path) -> list[Segment]:
        kwargs: dict = {
            "path_or_hf_repo": self.model,
            "condition_on_previous_text": False,
        }
        if self.language is not None:
            kwargs["language"] = self.language

        result = mlx_whisper.transcribe(str(audio_path), **kwargs)

        segments: list[Segment] = []
        for i, seg in enumerate(result.get("segments", []), 1):
            segments.append(
                Segment(
                    index=i,
                    start=seg["start"],
                    end=seg["end"],
                    text=seg["text"].strip(),
                )
            )
        return segments
