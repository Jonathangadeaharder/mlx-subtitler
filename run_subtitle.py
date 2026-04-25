#!/usr/bin/env python3
"""One-shot runner: transcribe + filter B1 + translate DE→ES → SRT"""
import sys
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import mlx_whisper
import spacy
import pysrt
import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

AUDIO = "/Users/jonathangadeaharder/Downloads/Desperate Housewives S08E23 S.to.m4a"
OUTPUT = "/Users/jonathangadeaharder/Downloads/Desperate Housewives S08E23 S.to.srt"
VOCAB_CSV = "https://raw.githubusercontent.com/Jonathangadeaharder/IdeaProjects/master/src/backend/data/B1_vokabeln.csv"
MODEL_WHISPER = "mlx-community/whisper-large-v3-turbo"
MODEL_MT = "Helsinki-NLP/opus-mt-tc-big-de-es"
SKIP_POS = {"PROPN", "NUM", "INTJ", "X", "SPACE"}


def load_vocab():
    logger.info("Loading B1 vocabulary...")
    import urllib.request
    import tempfile
    tmp = Path(tempfile.mktemp(suffix=".csv"))
    urllib.request.urlretrieve(VOCAB_CSV, str(tmp))
    df = pd.read_csv(tmp, header=None)
    vocab = set(df.iloc[:, 0].astype(str).str.strip().str.lower().tolist())
    tmp.unlink(missing_ok=True)
    logger.info(f"Loaded {len(vocab)} B1 words")
    return vocab


def transcribe():
    logger.info(f"Transcribing with {MODEL_WHISPER}...")
    t0 = time.time()
    result = mlx_whisper.transcribe(
        AUDIO,
        path_or_hf_repo=MODEL_WHISPER,
        language="de",
        condition_on_previous_text=False,
    )
    elapsed = time.time() - t0
    segments = []
    for i, seg in enumerate(result.get("segments", []), 1):
        segments.append({
            "index": i,
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip(),
        })
    logger.info(f"Transcribed {len(segments)} segments in {elapsed:.0f}s")
    return segments


def filter_segments(segments, vocab):
    logger.info(f"Filtering {len(segments)} segments against B1 vocab...")
    nlp = spacy.load("de_core_news_lg", disable=["parser", "ner"])
    kept = []
    for seg in segments:
        doc = nlp(seg["text"])
        lemmas = []
        for token in doc:
            if token.is_punct or token.is_stop or token.pos_ in SKIP_POS:
                continue
            lemma = token.lemma_.lower().strip()
            if lemma:
                lemmas.append(lemma)
        if lemmas and not all(lm in vocab for lm in lemmas):
            kept.append(seg)
    logger.info(f"Kept {len(kept)}/{len(segments)} segments")
    return kept


def translate_segments(segments):
    logger.info(f"Loading translation model {MODEL_MT}...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    tokenizer = MarianTokenizer.from_pretrained(MODEL_MT)
    model = MarianMTModel.from_pretrained(MODEL_MT).to(device)

    texts = [s["text"] for s in segments]
    translations = []
    batch_size = 32
    logger.info(f"Translating {len(texts)} segments in batches of {batch_size}...")
    for i in tqdm(range(0, len(texts), batch_size), desc="Translating"):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=512)
        translations.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))

    for seg, tr in zip(segments, translations):
        seg["translation"] = tr
    return segments


def write_srt(segments):
    subs = pysrt.SubRipFile()
    for seg in segments:
        text = seg.get("translation", seg["text"])
        subs.append(pysrt.SubRipItem(
            index=seg["index"],
            start=pysrt.SubRipTime(seconds=seg["start"]),
            end=pysrt.SubRipTime(seconds=seg["end"]),
            text=text,
        ))
    subs.save(OUTPUT, encoding="utf-8")
    logger.info(f"Saved {OUTPUT}")


if __name__ == "__main__":
    t_total = time.time()

    vocab = load_vocab()
    segments = transcribe()
    segments = filter_segments(segments, vocab)
    segments = translate_segments(segments)
    write_srt(segments)

    logger.info(f"Total time: {time.time() - t_total:.0f}s")
