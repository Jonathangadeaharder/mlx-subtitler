from __future__ import annotations

from pathlib import Path

import pysrt

from mlx_subtitler.models import Segment


class SubtitleWriter:
    @staticmethod
    def write_srt(
        segments: list[Segment],
        output_path: Path,
        use_translation: bool = True,
    ) -> Path:
        subs = pysrt.SubRipFile()
        for seg in segments:
            text = seg.translation if (use_translation and seg.translation) else seg.text
            subs.append(
                pysrt.SubRipItem(
                    index=seg.index,
                    start=pysrt.SubRipTime(seconds=seg.start),
                    end=pysrt.SubRipTime(seconds=seg.end),
                    text=text,
                )
            )
        output_path = Path(output_path)
        subs.save(str(output_path), encoding="utf-8")
        return output_path

    @staticmethod
    def write_vtt(
        segments: list[Segment],
        output_path: Path,
        use_translation: bool = True,
    ) -> Path:
        output_path = Path(output_path)
        lines = ["WEBVTT", ""]
        for seg in segments:
            text = seg.translation if (use_translation and seg.translation) else seg.text
            start_fmt = SubtitleWriter._format_vtt_time(seg.start)
            end_fmt = SubtitleWriter._format_vtt_time(seg.end)
            lines.append(f"{start_fmt} --> {end_fmt}")
            lines.append(text)
            lines.append("")
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path

    @staticmethod
    def _format_vtt_time(seconds: float) -> str:
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{hrs:02d}:{mins:02d}:{secs:02d}.{ms:03d}"
