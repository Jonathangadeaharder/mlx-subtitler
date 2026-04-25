from __future__ import annotations

import spacy

from mlx_subtitler.models import Segment

SKIP_POS = {"PROPN", "NUM", "INTJ", "X", "SPACE"}


class VocabFilter:
    def __init__(
        self,
        vocab: set[str],
        spacy_model: str = "de_core_news_lg",
    ) -> None:
        self.vocab = vocab
        self.nlp = spacy.load(spacy_model, disable=["parser", "ner"])

    def filter(self, segments: list[Segment]) -> list[Segment]:
        kept: list[Segment] = []
        for seg in segments:
            doc = self.nlp(seg.text)
            lemmas: list[str] = []
            for token in doc:
                if token.is_punct or token.is_stop or token.pos_ in SKIP_POS:
                    continue
                lemma = token.lemma_.lower().strip()
                if lemma:
                    lemmas.append(lemma)
            if lemmas and not all(lm in self.vocab for lm in lemmas):
                kept.append(seg)

        return [
            Segment(index=i + 1, start=s.start, end=s.end, text=s.text, translation=s.translation)
            for i, s in enumerate(kept)
        ]
