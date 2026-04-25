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
