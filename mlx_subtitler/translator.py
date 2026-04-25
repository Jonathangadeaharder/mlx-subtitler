from __future__ import annotations

from mlx_subtitler.models import Segment


class Translator:
    def __init__(
        self,
        src_lang: str = "de",
        tgt_lang: str = "es",
        device: str = "auto",
    ) -> None:
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.model_name = f"Helsinki-NLP/opus-mt-tc-big-{src_lang}-{tgt_lang}"

        if device == "auto":
            try:
                import torch

                self.device = "mps" if torch.backends.mps.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        from transformers import MarianMTModel, MarianTokenizer

        self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        self.model = MarianMTModel.from_pretrained(self.model_name).to(self.device)

    def translate(
        self,
        segments: list[Segment],
        batch_size: int = 32,
    ) -> list[Segment]:
        if not segments:
            return segments

        import torch

        texts = [s.text for s in segments]
        translations: list[str] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            with torch.no_grad():
                generated = self.model.generate(**inputs, max_new_tokens=512)
            translations.extend(
                self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            )

        return [
            Segment(
                index=s.index,
                start=s.start,
                end=s.end,
                text=s.text,
                translation=tr,
            )
            for s, tr in zip(segments, translations)
        ]
