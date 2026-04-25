from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mlx_subtitler.models import Segment
from mlx_subtitler.pipeline import Pipeline


def _segments():
    return [
        Segment(index=1, start=0.0, end=2.5, text="Hallo"),
        Segment(index=2, start=2.5, end=5.0, text="Welt"),
    ]


def _make_transcriber_mock():
    mock = MagicMock()
    mock.transcribe.return_value = _segments()
    return mock


@patch("mlx_subtitler.pipeline.Transcriber")
def test_transcribe_only(mock_cls, tmp_path):
    mock_cls.return_value = _make_transcriber_mock()

    audio = tmp_path / "test.m4a"
    audio.write_bytes(b"\x00")
    out = tmp_path / "out.srt"

    p = Pipeline(translate=False, filter_vocab=False, output_format="srt")
    result = p.run(audio, out)
    assert result == out
    assert out.exists()


@patch("mlx_subtitler.translator.Translator")
@patch("mlx_subtitler.vocab_filter.VocabFilter")
@patch("mlx_subtitler.vocab_loader.VocabLoader")
@patch("mlx_subtitler.pipeline.Transcriber")
def test_full_with_filter(mock_tc, mock_vl, mock_vf, mock_tr):
    mock_tc.return_value = _make_transcriber_mock()

    mock_loader = MagicMock()
    mock_loader.load.return_value = {"hallo", "welt"}
    mock_vl.return_value = mock_loader

    mock_filter = MagicMock()
    mock_filter.filter.return_value = _segments()
    mock_vf.return_value = mock_filter

    mock_translator = MagicMock()
    mock_translator.translate.return_value = [
        Segment(index=1, start=0.0, end=2.5, text="Hallo", translation="Hello"),
        Segment(index=2, start=2.5, end=5.0, text="Welt", translation="World"),
    ]
    mock_tr.return_value = mock_translator

    with patch("mlx_subtitler.subtitle_writer.SubtitleWriter.write_srt", return_value=Path("out.srt")):
        p = Pipeline(output_format="srt")
        p.run(Path("test.m4a"), Path("out.srt"))

    mock_loader.load.assert_called_once()
    mock_filter.filter.assert_called_once()
    mock_translator.translate.assert_called_once()


@patch("mlx_subtitler.translator.Translator")
@patch("mlx_subtitler.pipeline.Transcriber")
def test_skip_filter(mock_tc, mock_tr):
    mock_tc.return_value = _make_transcriber_mock()

    mock_translator = MagicMock()
    mock_translator.translate.return_value = _segments()
    mock_tr.return_value = mock_translator

    with patch("mlx_subtitler.subtitle_writer.SubtitleWriter.write_srt", return_value=Path("out.srt")):
        p = Pipeline(filter_vocab=False, output_format="srt")
        p.run(Path("test.m4a"), Path("out.srt"))

    mock_translator.translate.assert_called_once()


@patch("mlx_subtitler.pipeline.Transcriber")
def test_auto_output_path(mock_cls, tmp_path):
    mock_cls.return_value = _make_transcriber_mock()

    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"\x00")
    expected = tmp_path / "audio.srt"

    p = Pipeline(translate=False, filter_vocab=False, output_format="srt")
    result = p.run(audio)
    assert result == expected


@patch("mlx_subtitler.pipeline.Transcriber")
def test_run_batch_processes_directory(mock_cls, tmp_path):
    mock_cls.return_value = _make_transcriber_mock()
    (tmp_path / "a.m4a").write_bytes(b"\x00")
    (tmp_path / "b.mp3").write_bytes(b"\x00")
    (tmp_path / "c.txt").write_bytes(b"\x00")
    p = Pipeline(translate=False, filter_vocab=False, output_format="srt")
    results = p.run_batch(tmp_path)
    assert len(results) == 2
    assert all(r.suffix == ".srt" for r in results)


@patch("mlx_subtitler.pipeline.Transcriber")
def test_run_batch_with_output_dir(mock_cls, tmp_path):
    mock_cls.return_value = _make_transcriber_mock()
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "a.m4a").write_bytes(b"\x00")
    out_dir = tmp_path / "output"
    p = Pipeline(translate=False, filter_vocab=False, output_format="srt")
    results = p.run_batch(audio_dir, out_dir)
    assert len(results) == 1
    assert results[0].parent == out_dir


@patch("mlx_subtitler.pipeline.Transcriber")
def test_run_batch_empty_directory(mock_cls, tmp_path):
    mock_cls.return_value = _make_transcriber_mock()
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    p = Pipeline(translate=False, filter_vocab=False, output_format="srt")
    results = p.run_batch(empty_dir)
    assert results == []


def test_init_stores_all_defaults():
    p = Pipeline()
    assert p.src_lang == "de"
    assert p.tgt_lang == "es"
    assert p.whisper_model == "mlx-community/whisper-large-v3-turbo"
    assert p.filter_vocab is True
    assert p.vocab_levels is None
    assert p.translate_flag is True
    assert p.output_format == "srt"
    assert p.batch_size == 32


def test_init_stores_custom_params():
    p = Pipeline(
        src_lang="en",
        tgt_lang="fr",
        whisper_model="custom-model",
        filter_vocab=False,
        vocab_levels=["A1", "A2"],
        translate=False,
        output_format="vtt",
        batch_size=16,
    )
    assert p.src_lang == "en"
    assert p.tgt_lang == "fr"
    assert p.whisper_model == "custom-model"
    assert p.filter_vocab is False
    assert p.vocab_levels == ["A1", "A2"]
    assert p.translate_flag is False
    assert p.output_format == "vtt"
    assert p.batch_size == 16


def test_auto_output_path_srt():
    p = Pipeline(output_format="srt")
    result = p._auto_output_path(Path("/audio/test.m4a"))
    assert result == Path("/audio/test.srt")


def test_auto_output_path_vtt():
    p = Pipeline(output_format="vtt")
    result = p._auto_output_path(Path("/audio/test.m4a"))
    assert result == Path("/audio/test.vtt")


@patch("mlx_subtitler.pipeline.Transcriber")
def test_run_creates_transcriber_with_correct_args(mock_cls, tmp_path):
    mock_cls.return_value = _make_transcriber_mock()
    audio = tmp_path / "test.m4a"
    audio.write_bytes(b"\x00")
    p = Pipeline(whisper_model="my-model", src_lang="fr", translate=False, filter_vocab=False)
    p.run(audio)
    mock_cls.assert_called_once_with(model="my-model", language="fr")


@patch("mlx_subtitler.pipeline.Transcriber")
def test_run_calls_transcriber_transcribe(mock_cls, tmp_path):
    mock_tc = _make_transcriber_mock()
    mock_cls.return_value = mock_tc
    audio = tmp_path / "test.m4a"
    audio.write_bytes(b"\x00")
    p = Pipeline(translate=False, filter_vocab=False)
    p.run(audio)
    mock_tc.transcribe.assert_called_once_with(audio)


@patch("mlx_subtitler.translator.Translator")
@patch("mlx_subtitler.vocab_filter.VocabFilter")
@patch("mlx_subtitler.vocab_loader.VocabLoader")
@patch("mlx_subtitler.pipeline.Transcriber")
def test_run_creates_vocab_loader_with_defaults(mock_tc, mock_vl, mock_vf, mock_tr):
    mock_tc.return_value = _make_transcriber_mock()
    mock_loader = MagicMock()
    mock_loader.load.return_value = {"hallo"}
    mock_vl.return_value = mock_loader
    mock_filter = MagicMock()
    mock_filter.filter.return_value = _segments()
    mock_vf.return_value = mock_filter
    mock_tr.return_value = MagicMock()

    with patch("mlx_subtitler.subtitle_writer.SubtitleWriter.write_srt"):
        p = Pipeline(output_format="srt", translate=False)
        p.run(Path("test.m4a"), Path("out.srt"))

    mock_vl.assert_called_once_with()


@patch("mlx_subtitler.translator.Translator")
@patch("mlx_subtitler.vocab_filter.VocabFilter")
@patch("mlx_subtitler.vocab_loader.VocabLoader")
@patch("mlx_subtitler.pipeline.Transcriber")
def test_run_loads_vocab_with_levels(mock_tc, mock_vl, mock_vf, mock_tr):
    mock_tc.return_value = _make_transcriber_mock()
    mock_loader = MagicMock()
    mock_loader.load.return_value = {"hallo"}
    mock_vl.return_value = mock_loader
    mock_filter = MagicMock()
    mock_filter.filter.return_value = _segments()
    mock_vf.return_value = mock_filter
    mock_tr.return_value = MagicMock()

    with patch("mlx_subtitler.subtitle_writer.SubtitleWriter.write_srt"):
        p = Pipeline(vocab_levels=["A1", "A2"], output_format="srt", translate=False)
        p.run(Path("test.m4a"), Path("out.srt"))

    mock_loader.load.assert_called_once_with(["A1", "A2"])


@patch("mlx_subtitler.translator.Translator")
@patch("mlx_subtitler.vocab_filter.VocabFilter")
@patch("mlx_subtitler.vocab_loader.VocabLoader")
@patch("mlx_subtitler.pipeline.Transcriber")
def test_run_creates_vocab_filter_with_vocab(mock_tc, mock_vl, mock_vf, mock_tr):
    mock_tc.return_value = _make_transcriber_mock()
    mock_loader = MagicMock()
    mock_loader.load.return_value = {"word1", "word2"}
    mock_vl.return_value = mock_loader
    mock_filter = MagicMock()
    mock_filter.filter.return_value = _segments()
    mock_vf.return_value = mock_filter
    mock_tr.return_value = MagicMock()

    with patch("mlx_subtitler.subtitle_writer.SubtitleWriter.write_srt"):
        p = Pipeline(output_format="srt", translate=False)
        p.run(Path("test.m4a"), Path("out.srt"))

    mock_vf.assert_called_once_with({"word1", "word2"})


@patch("mlx_subtitler.translator.Translator")
@patch("mlx_subtitler.vocab_filter.VocabFilter")
@patch("mlx_subtitler.vocab_loader.VocabLoader")
@patch("mlx_subtitler.pipeline.Transcriber")
def test_run_calls_filter_with_segments(mock_tc, mock_vl, mock_vf, mock_tr):
    segs = _segments()
    mock_tc.return_value = _make_transcriber_mock()
    mock_loader = MagicMock()
    mock_loader.load.return_value = {"hallo"}
    mock_vl.return_value = mock_loader
    mock_filter = MagicMock()
    mock_filter.filter.return_value = segs
    mock_vf.return_value = mock_filter
    mock_tr.return_value = MagicMock()

    with patch("mlx_subtitler.subtitle_writer.SubtitleWriter.write_srt"):
        p = Pipeline(output_format="srt", translate=False)
        p.run(Path("test.m4a"), Path("out.srt"))

    mock_filter.filter.assert_called_once_with(segs)


@patch("mlx_subtitler.translator.Translator")
@patch("mlx_subtitler.pipeline.Transcriber")
def test_run_creates_translator_with_langs(mock_tc, mock_tr):
    mock_tc.return_value = _make_transcriber_mock()
    mock_translator = MagicMock()
    mock_translator.translate.return_value = _segments()
    mock_tr.return_value = mock_translator

    with patch("mlx_subtitler.subtitle_writer.SubtitleWriter.write_srt"):
        p = Pipeline(src_lang="en", tgt_lang="fr", filter_vocab=False, output_format="srt")
        p.run(Path("test.m4a"), Path("out.srt"))

    mock_tr.assert_called_once_with(src_lang="en", tgt_lang="fr")


@patch("mlx_subtitler.translator.Translator")
@patch("mlx_subtitler.pipeline.Transcriber")
def test_run_calls_translate_with_batch_size(mock_tc, mock_tr):
    mock_tc.return_value = _make_transcriber_mock()
    mock_translator = MagicMock()
    mock_translator.translate.return_value = _segments()
    mock_tr.return_value = mock_translator

    with patch("mlx_subtitler.subtitle_writer.SubtitleWriter.write_srt"):
        p = Pipeline(filter_vocab=False, output_format="srt", batch_size=16)
        p.run(Path("test.m4a"), Path("out.srt"))

    mock_translator.translate.assert_called_once_with(_segments(), batch_size=16)


@patch("mlx_subtitler.pipeline.Transcriber")
def test_run_uses_srt_writer(mock_cls, tmp_path):
    mock_cls.return_value = _make_transcriber_mock()
    audio = tmp_path / "test.m4a"
    audio.write_bytes(b"\x00")
    out = tmp_path / "out.srt"

    with patch("mlx_subtitler.subtitle_writer.SubtitleWriter.write_srt") as mock_write:
        p = Pipeline(translate=False, filter_vocab=False, output_format="srt")
        p.run(audio, out)
        mock_write.assert_called_once()


@patch("mlx_subtitler.pipeline.Transcriber")
def test_run_uses_vtt_writer(mock_cls, tmp_path):
    mock_cls.return_value = _make_transcriber_mock()
    audio = tmp_path / "test.m4a"
    audio.write_bytes(b"\x00")
    out = tmp_path / "out.vtt"

    with patch("mlx_subtitler.subtitle_writer.SubtitleWriter.write_vtt") as mock_write:
        p = Pipeline(translate=False, filter_vocab=False, output_format="vtt")
        p.run(audio, out)
        mock_write.assert_called_once()


@patch("mlx_subtitler.pipeline.Transcriber")
def test_run_skips_filter_when_disabled(mock_cls, tmp_path):
    mock_cls.return_value = _make_transcriber_mock()
    audio = tmp_path / "test.m4a"
    audio.write_bytes(b"\x00")
    out = tmp_path / "out.srt"

    with patch("mlx_subtitler.vocab_filter.VocabFilter") as mock_vf:
        p = Pipeline(translate=False, filter_vocab=False, output_format="srt")
        p.run(audio, out)
        mock_vf.assert_not_called()


@patch("mlx_subtitler.pipeline.Transcriber")
def test_run_skips_translate_when_disabled(mock_cls, tmp_path):
    mock_cls.return_value = _make_transcriber_mock()
    audio = tmp_path / "test.m4a"
    audio.write_bytes(b"\x00")
    out = tmp_path / "out.srt"

    with patch("mlx_subtitler.translator.Translator") as mock_tr:
        p = Pipeline(translate=False, filter_vocab=False, output_format="srt")
        p.run(audio, out)
        mock_tr.assert_not_called()


@patch("mlx_subtitler.pipeline.Transcriber")
def test_run_returns_output_path(mock_cls, tmp_path):
    mock_cls.return_value = _make_transcriber_mock()
    audio = tmp_path / "test.m4a"
    audio.write_bytes(b"\x00")
    out = tmp_path / "out.srt"

    p = Pipeline(translate=False, filter_vocab=False, output_format="srt")
    result = p.run(audio, out)
    assert result == out


@patch("mlx_subtitler.pipeline.Transcriber")
def test_run_batch_sorts_files(mock_cls, tmp_path):
    mock_cls.return_value = _make_transcriber_mock()
    (tmp_path / "c.m4a").write_bytes(b"\x00")
    (tmp_path / "a.m4a").write_bytes(b"\x00")
    (tmp_path / "b.m4a").write_bytes(b"\x00")
    p = Pipeline(translate=False, filter_vocab=False, output_format="srt")
    results = p.run_batch(tmp_path)
    names = [r.name for r in results]
    assert names == sorted(names)


@patch("mlx_subtitler.pipeline.Transcriber")
def test_run_batch_filters_by_extension(mock_cls, tmp_path):
    mock_cls.return_value = _make_transcriber_mock()
    (tmp_path / "a.m4a").write_bytes(b"\x00")
    (tmp_path / "b.mp3").write_bytes(b"\x00")
    (tmp_path / "c.wav").write_bytes(b"\x00")
    (tmp_path / "d.flac").write_bytes(b"\x00")
    (tmp_path / "e.aac").write_bytes(b"\x00")
    (tmp_path / "f.txt").write_bytes(b"\x00")
    (tmp_path / "g.pdf").write_bytes(b"\x00")
    p = Pipeline(translate=False, filter_vocab=False, output_format="srt")
    results = p.run_batch(tmp_path)
    assert len(results) == 5


@patch("mlx_subtitler.pipeline.Transcriber")
def test_run_batch_uses_output_dir_for_paths(mock_cls, tmp_path):
    mock_cls.return_value = _make_transcriber_mock()
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "a.m4a").write_bytes(b"\x00")
    out_dir = tmp_path / "output"
    p = Pipeline(translate=False, filter_vocab=False, output_format="srt")
    results = p.run_batch(audio_dir, out_dir)
    assert len(results) == 1
    assert results[0].parent == out_dir
    assert results[0].name == "a.srt"


@patch("mlx_subtitler.pipeline.Transcriber")
def test_run_batch_creates_output_dir(mock_cls, tmp_path):
    mock_cls.return_value = _make_transcriber_mock()
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "a.m4a").write_bytes(b"\x00")
    out_dir = tmp_path / "nested" / "output"
    p = Pipeline(translate=False, filter_vocab=False, output_format="srt")
    p.run_batch(audio_dir, out_dir)
    assert out_dir.exists()


@patch("mlx_subtitler.pipeline.Transcriber")
def test_run_batch_no_output_dir_uses_audio_dir(mock_cls, tmp_path):
    mock_cls.return_value = _make_transcriber_mock()
    (tmp_path / "a.m4a").write_bytes(b"\x00")
    p = Pipeline(translate=False, filter_vocab=False, output_format="srt")
    results = p.run_batch(tmp_path)
    assert len(results) == 1
    assert results[0].parent == tmp_path


@patch("mlx_subtitler.pipeline.Transcriber")
def test_run_batch_vtt_format(mock_cls, tmp_path):
    mock_cls.return_value = _make_transcriber_mock()
    (tmp_path / "a.m4a").write_bytes(b"\x00")
    p = Pipeline(translate=False, filter_vocab=False, output_format="vtt")
    results = p.run_batch(tmp_path)
    assert all(r.suffix == ".vtt" for r in results)


@patch("mlx_subtitler.pipeline.Transcriber")
def test_run_passes_path_object_to_transcriber(mock_cls, tmp_path):
    mock_tc = _make_transcriber_mock()
    mock_cls.return_value = mock_tc
    audio = tmp_path / "test.m4a"
    audio.write_bytes(b"\x00")
    p = Pipeline(translate=False, filter_vocab=False)
    p.run(audio)
    called_arg = mock_tc.transcribe.call_args[0][0]
    assert isinstance(called_arg, Path)
    assert called_arg == audio


@patch("mlx_subtitler.translator.Translator")
@patch("mlx_subtitler.pipeline.Transcriber")
def test_run_pipeline_order_transcribe_then_translate(mock_tc, mock_tr):
    call_order = []
    mock_tc_inst = MagicMock()
    mock_tc_inst.transcribe.return_value = _segments()
    mock_tc.return_value = mock_tc_inst
    mock_tc_inst.transcribe.side_effect = lambda *a, **k: (call_order.append("transcribe"), _segments())[1]

    mock_tr_inst = MagicMock()
    mock_tr_inst.translate.return_value = _segments()
    mock_tr.return_value = mock_tr_inst
    mock_tr_inst.translate.side_effect = lambda *a, **k: (call_order.append("translate"), _segments())[1]

    with patch("mlx_subtitler.subtitle_writer.SubtitleWriter.write_srt"):
        p = Pipeline(filter_vocab=False, output_format="srt")
        p.run(Path("test.m4a"), Path("out.srt"))

    assert call_order == ["transcribe", "translate"]


@patch("mlx_subtitler.translator.Translator")
@patch("mlx_subtitler.vocab_filter.VocabFilter")
@patch("mlx_subtitler.vocab_loader.VocabLoader")
@patch("mlx_subtitler.pipeline.Transcriber")
def test_run_pipeline_order_all_steps(mock_tc, mock_vl, mock_vf, mock_tr):
    call_order = []

    mock_tc_inst = MagicMock()
    mock_tc_inst.transcribe.return_value = _segments()
    mock_tc.return_value = mock_tc_inst
    mock_tc_inst.transcribe.side_effect = lambda *a, **k: (call_order.append("transcribe"), _segments())[1]

    mock_loader = MagicMock()
    mock_loader.load.return_value = {"hallo"}
    mock_loader.load.side_effect = lambda *a, **k: (call_order.append("load_vocab"), {"hallo"})[1]
    mock_vl.return_value = mock_loader

    mock_filter = MagicMock()
    mock_filter.filter.return_value = _segments()
    mock_vf.return_value = mock_filter
    mock_filter.filter.side_effect = lambda *a, **k: (call_order.append("filter"), _segments())[1]

    mock_tr_inst = MagicMock()
    mock_tr_inst.translate.return_value = _segments()
    mock_tr.return_value = mock_tr_inst
    mock_tr_inst.translate.side_effect = lambda *a, **k: (call_order.append("translate"), _segments())[1]

    with patch("mlx_subtitler.subtitle_writer.SubtitleWriter.write_srt"):
        p = Pipeline(output_format="srt")
        p.run(Path("test.m4a"), Path("out.srt"))

    assert call_order == ["transcribe", "load_vocab", "filter", "translate"]


@patch("mlx_subtitler.pipeline.Transcriber")
def test_run_writes_segments_to_file(mock_cls, tmp_path):
    mock_cls.return_value = _make_transcriber_mock()
    audio = tmp_path / "test.m4a"
    audio.write_bytes(b"\x00")
    out = tmp_path / "out.srt"
    p = Pipeline(translate=False, filter_vocab=False, output_format="srt")
    p.run(audio, out)
    assert out.exists()
    content = out.read_text()
    assert len(content) > 0


@patch("mlx_subtitler.pipeline.Transcriber")
def test_run_logs_transcribed_count(mock_cls, tmp_path, caplog):
    mock_cls.return_value = _make_transcriber_mock()
    audio = tmp_path / "test.m4a"
    audio.write_bytes(b"\x00")
    out = tmp_path / "out.srt"
    p = Pipeline(translate=False, filter_vocab=False)
    with caplog.at_level(logging.INFO, logger="mlx_subtitler.pipeline"):
        p.run(audio, out)
    transcribed_msgs = [r for r in caplog.records if "Transcribed" in r.message]
    assert len(transcribed_msgs) == 1
    assert "Transcribed 2 segments" == transcribed_msgs[0].message


@patch("mlx_subtitler.translator.Translator")
@patch("mlx_subtitler.vocab_filter.VocabFilter")
@patch("mlx_subtitler.vocab_loader.VocabLoader")
@patch("mlx_subtitler.pipeline.Transcriber")
def test_run_logs_filtered_count(mock_tc, mock_vl, mock_vf, mock_tr, caplog):
    mock_tc.return_value = _make_transcriber_mock()
    mock_loader = MagicMock()
    mock_loader.load.return_value = {"hallo"}
    mock_vl.return_value = mock_loader
    mock_filter = MagicMock()
    mock_filter.filter.return_value = _segments()
    mock_vf.return_value = mock_filter
    mock_tr.return_value = MagicMock()

    with patch("mlx_subtitler.subtitle_writer.SubtitleWriter.write_srt"), \
         caplog.at_level(logging.INFO, logger="mlx_subtitler.pipeline"):
        p = Pipeline(output_format="srt", translate=False)
        p.run(Path("test.m4a"), Path("out.srt"))

    filtered_msgs = [r for r in caplog.records if "Filtered" in r.message]
    assert len(filtered_msgs) == 1
    assert "Filtered to 2 segments" == filtered_msgs[0].message


@patch("mlx_subtitler.pipeline.Transcriber")
def test_run_logs_wrote_output_path(mock_cls, tmp_path, caplog):
    mock_cls.return_value = _make_transcriber_mock()
    audio = tmp_path / "test.m4a"
    audio.write_bytes(b"\x00")
    out = tmp_path / "out.srt"
    p = Pipeline(translate=False, filter_vocab=False)
    with caplog.at_level(logging.INFO, logger="mlx_subtitler.pipeline"):
        p.run(audio, out)
    wrote_msgs = [r for r in caplog.records if "Wrote" in r.message]
    assert len(wrote_msgs) == 1
    assert str(out) in wrote_msgs[0].message


@patch("mlx_subtitler.pipeline.Transcriber")
def test_run_batch_mkdir_with_exist_ok(mock_cls, tmp_path):
    mock_cls.return_value = _make_transcriber_mock()
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "a.m4a").write_bytes(b"\x00")
    out_dir = tmp_path / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    original_mkdir = Path.mkdir

    def mkdir_spy(self, *args, **kwargs):
        return original_mkdir(self, *args, **kwargs)

    with patch.object(Path, "mkdir", autospec=True, side_effect=mkdir_spy) as mock_mkdir:
        p = Pipeline(translate=False, filter_vocab=False, output_format="srt")
        p.run_batch(audio_dir, out_dir)
    mkdir_calls = [c for c in mock_mkdir.call_args_list if str(out_dir) in str(c)]
    assert any(c.kwargs.get("exist_ok") is True for c in mkdir_calls)
