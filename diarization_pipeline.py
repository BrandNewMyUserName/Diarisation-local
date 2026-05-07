#!/usr/bin/env python3
"""Resumable WhisperX batch transcription + diarization pipeline."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import logging
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv


try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


load_dotenv()
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _get_hf_token() -> str:
    token = os.getenv("HF_TOKEN", "").strip()
    if token:
        return token
    try:
        from huggingface_hub import HfFolder

        cached = HfFolder.get_token()
        return cached.strip() if cached else ""
    except Exception:
        return ""


DEFAULT_INPUT_DIR = os.getenv("DIARIZATION_INPUT_DIR", "./audio")

DEFAULT_CONFIG: dict[str, Any] = {
    "input_dir": DEFAULT_INPUT_DIR,
    "output_dir": "./output",
    "log_dir": "./logs",
    "state_dir": "./output/_progress",
    "model_size": "large-v3",
    "language": None,
    "preferred_languages": ["uk", "ru"],
    "fallback_language": "uk",
    "retry_unexpected_language": True,
    "device": "cuda",
    "compute_type": "float16",
    "batch_size": 16,
    "min_speakers": 1,
    "max_speakers": 4,
    "num_speakers": None,
    "diarization_model": "pyannote/speaker-diarization-community-1",
    "offline_diarization_model": "pyannote/speaker-diarization-3.1",
    "offline_diarization_fallback": True,
    "align_words": True,
    "output_format": "all",
    "min_confidence": 0.0,
    "extensions": [
        ".mp3",
        ".wav",
        ".m4a",
        ".ogg",
        ".oga",
        ".opus",
        ".flac",
        ".aac",
        ".wma",
        ".mp4",
        ".mkv",
        ".webm",
    ],
    "hf_token": _get_hf_token(),
    "resume": True,
    "force": False,
    "probe_durations": True,
    "reuse_models": True,
    "adaptive_batch": True,
    "max_retries": 3,
    "config_name": "max_quality_rtx5080",
}


try:
    import psutil

    _PSUTIL_OK = True
except ImportError:
    _PSUTIL_OK = False


_logger: Optional[logging.Logger] = None
_log_path: Optional[Path] = None
_shutdown_requested = False


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _format_seconds(seconds: Optional[float]) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "n/a"
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def _normalize_formats(value: Any) -> list[str]:
    if value == "all":
        return ["txt", "srt", "vtt", "json", "tsv"]
    if isinstance(value, str):
        return [value]
    return list(value)


def _safe_relative_path(path: Path, root: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return Path(path.name)


def _file_key(path: Path, root: Path) -> str:
    return _safe_relative_path(path, root).as_posix()


def _file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _hash_payload(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _config_hash(config: dict[str, Any]) -> str:
    relevant = {
        "model_size": config["model_size"],
        "language": config["language"],
        "preferred_languages": config.get("preferred_languages"),
        "fallback_language": config.get("fallback_language"),
        "retry_unexpected_language": config.get("retry_unexpected_language"),
        "device": config["device"],
        "compute_type": config["compute_type"],
        "min_speakers": config["min_speakers"],
        "max_speakers": config["max_speakers"],
        "num_speakers": config.get("num_speakers"),
        "effective_diarization_model": config.get("effective_diarization_model") or _effective_diarization_model(config),
        "align_words": config["align_words"],
        "output_format": _normalize_formats(config["output_format"]),
        "extensions": sorted(config["extensions"]),
        "config_name": config.get("config_name"),
    }
    return _hash_payload(relevant)


def _hf_cache_model_dir(repo_id: str) -> Path:
    return Path.home() / ".cache" / "huggingface" / "hub" / f"models--{repo_id.replace('/', '--')}"


def _is_hf_model_cached(repo_id: Optional[str]) -> bool:
    if not repo_id:
        return False
    model_dir = _hf_cache_model_dir(repo_id)
    snapshots_dir = model_dir / "snapshots"
    return snapshots_dir.exists() and any(path.is_file() for path in snapshots_dir.rglob("*"))


def _effective_diarization_model(config: dict[str, Any]) -> Optional[str]:
    if config.get("hf_token"):
        return config.get("diarization_model")
    fallback = config.get("offline_diarization_model")
    if config.get("offline_diarization_fallback", True) and _is_hf_model_cached(fallback):
        return fallback
    return config.get("diarization_model")


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        suffix=path.suffix + ".tmp",
        prefix=path.stem + ".",
        dir=path.parent,
        delete=False,
    )
    tmp_path = Path(tmp_handle.name)
    json.dump(payload, tmp_handle, ensure_ascii=False, indent=2)
    tmp_handle.close()
    last_error: Optional[PermissionError] = None
    for attempt in range(10):
        try:
            os.replace(tmp_path, path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.05 * (attempt + 1))
    try:
        if path.exists():
            path.unlink()
        os.replace(tmp_path, path)
        return
    except PermissionError:
        try:
            tmp_path.unlink(missing_ok=True)
        finally:
            raise last_error or PermissionError(f"Cannot replace {path}")


def _looks_like_oom(exc: BaseException) -> bool:
    text = " ".join(str(part) for part in (type(exc).__name__, exc, traceback.format_exc())).lower()
    return "out of memory" in text or "cuda" in text and "memory" in text or "cublas" in text


def setup_logging(log_dir: str) -> logging.Logger:
    global _logger, _log_path

    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _log_path = log_dir_path / f"diarize_{ts}.log"

    logger = logging.getLogger("diarize")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(_log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    _logger = logger
    return logger


def flush_logs() -> None:
    if _logger:
        for handler in _logger.handlers:
            handler.flush()


def _signal_handler(signum: int, frame: Any) -> None:
    global _shutdown_requested
    _shutdown_requested = True
    message = f"Отримано сигнал {signum} — завершення після поточного файлу. Лог: {_log_path}"
    if _logger:
        _logger.warning(message)
    else:
        print(f"\n⚠ {message}", file=sys.stderr)
    flush_logs()


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def get_memory_info(device: str) -> str:
    parts: list[str] = []
    if device == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                parts.append(f"GPU: alloc={alloc:.2f}GB reserved={reserved:.2f}/{total:.2f}GB")
        except Exception:
            pass

    if _PSUTIL_OK:
        try:
            proc = psutil.Process(os.getpid())
            rss = proc.memory_info().rss / 1024**3
            vm = psutil.virtual_memory()
            parts.append(
                f"RAM: proc={rss:.2f}GB sys={vm.used / 1024**3:.1f}/{vm.total / 1024**3:.1f}GB ({vm.percent:.0f}%)"
            )
        except Exception:
            pass
    return "  |  ".join(parts) if parts else "n/a"


def _log_step(logger: logging.Logger, label: str, elapsed: float, device: str) -> None:
    logger.info(f"  {'[' + label + ']':<36} {elapsed:>9.2f}s  |  {get_memory_info(device)}")


class ProgressTracker:
    def __init__(self, state_dir: Path, input_root: Path, config_hash_value: str, logger: logging.Logger):
        self.state_dir = state_dir
        self.input_root = input_root
        self.config_hash = config_hash_value
        self.logger = logger
        self.manifest_path = state_dir / "manifest.json"
        self.events_path = state_dir / "events.jsonl"
        self.quality_csv_path = state_dir / "quality_report.csv"
        self.errors_path = state_dir / "errors.jsonl"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.data = self._load_manifest()

    def _load_manifest(self) -> dict[str, Any]:
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                data.setdefault("files", {})
                data["last_loaded_at"] = _now_iso()
                return data
            except Exception as exc:
                self.logger.warning(f"Не вдалося прочитати manifest.json: {exc}. Створюємо новий.")
        return {
            "schema_version": 2,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "config_hash": self.config_hash,
            "files": {},
            "summary": {},
        }

    def save(self) -> None:
        self.data["updated_at"] = _now_iso()
        self.data["config_hash"] = self.config_hash
        _atomic_write_json(self.manifest_path, self.data)

    def append_event(self, event: dict[str, Any]) -> None:
        event = {"ts": _now_iso(), **event}
        with open(self.events_path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(event, ensure_ascii=False) + "\n")

    def record_for(self, path: Path) -> dict[str, Any]:
        key = _file_key(path, self.input_root)
        return self.data["files"].setdefault(key, {"path": str(path), "key": key})

    def update_file(self, path: Path, **updates: Any) -> dict[str, Any]:
        record = self.record_for(path)
        record.update(updates)
        record["updated_at"] = _now_iso()
        self.append_event({"file": record["key"], **updates})
        self.save()
        return record

    def mark_error(self, path: Path, error: str, error_kind: str = "error", **extra: Any) -> dict[str, Any]:
        record = self.update_file(
            path,
            status="error",
            stage="error",
            error=error,
            error_kind=error_kind,
            completed_at=_now_iso(),
            **extra,
        )
        with open(self.errors_path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.write_quality_csv()
        return record

    def is_completed(self, path: Path, fingerprint: dict[str, Any], config_hash_value: str) -> bool:
        record = self.record_for(path)
        if record.get("status") != "done":
            return False
        if record.get("fingerprint") != fingerprint:
            return False
        if record.get("config_hash") != config_hash_value:
            return False
        output_paths = record.get("output_paths") or []
        return bool(output_paths) and all(Path(p).exists() for p in output_paths)

    def write_quality_csv(self) -> None:
        rows: list[dict[str, Any]] = []
        for record in self.data["files"].values():
            quality = record.get("quality") or {}
            rows.append(
                {
                    "key": record.get("key"),
                    "status": record.get("status"),
                    "duration_sec": record.get("duration_sec"),
                    "processing_time_sec": record.get("processing_time_sec"),
                    "real_time_factor": record.get("real_time_factor"),
                    "detected_language": record.get("detected_language"),
                    "speaker_count": quality.get("speaker_count"),
                    "overall_score": quality.get("overall_score"),
                    "grade": quality.get("grade"),
                    "speaker_assignment_coverage": quality.get("speaker_assignment_coverage"),
                    "unassigned_word_ratio": quality.get("unassigned_word_ratio"),
                    "missing_word_timestamp_ratio": quality.get("missing_word_timestamp_ratio"),
                    "avg_word_score": quality.get("avg_word_score"),
                    "speaker_switches_per_min": quality.get("speaker_switches_per_min"),
                    "short_turn_ratio": quality.get("short_turn_ratio"),
                    "flags": ";".join(quality.get("flags") or []),
                    "error_kind": record.get("error_kind"),
                    "error": record.get("error"),
                }
            )

        self.quality_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.quality_csv_path, "w", encoding="utf-8-sig", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()) if rows else ["key"])
            writer.writeheader()
            writer.writerows(rows)

    def update_summary(self, summary: dict[str, Any]) -> None:
        self.data["summary"] = summary
        self.save()


def probe_audio_duration(path: Path) -> Optional[float]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
        if completed.returncode != 0:
            return None
        value = completed.stdout.strip()
        return float(value) if value else None
    except Exception:
        return None


def find_audio_files(input_dir: str, extensions: list[str]) -> list[Path]:
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Директорія не знайдена: {root.resolve()}")
    ext_set = {e.lower() for e in extensions}
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in ext_set)


def _fmt_srt(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _fmt_vtt(seconds: float) -> str:
    return _fmt_srt(seconds).replace(",", ".")


def _save_txt(result: dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as fp:
        current_speaker = None
        for segment in result.get("segments", []):
            speaker = segment.get("speaker", "")
            if speaker and speaker != current_speaker:
                current_speaker = speaker
                fp.write(f"\n[{speaker}]\n")
            start = _fmt_srt(segment.get("start", 0.0)).replace(",", ".")
            end = _fmt_srt(segment.get("end", 0.0)).replace(",", ".")
            fp.write(f"[{start} → {end}] {segment.get('text', '').strip()}\n")


def _save_srt(result: dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as fp:
        for idx, segment in enumerate(result.get("segments", []), 1):
            speaker = segment.get("speaker", "")
            label = f"[{speaker}] " if speaker else ""
            fp.write(f"{idx}\n")
            fp.write(f"{_fmt_srt(segment.get('start', 0.0))} --> {_fmt_srt(segment.get('end', 0.0))}\n")
            fp.write(f"{label}{segment.get('text', '').strip()}\n\n")


def _save_vtt(result: dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as fp:
        fp.write("WEBVTT\n\n")
        for idx, segment in enumerate(result.get("segments", []), 1):
            speaker = segment.get("speaker", "")
            label = f"<v {speaker}>" if speaker else ""
            fp.write(f"{idx}\n")
            fp.write(f"{_fmt_vtt(segment.get('start', 0.0))} --> {_fmt_vtt(segment.get('end', 0.0))}\n")
            fp.write(f"{label}{segment.get('text', '').strip()}\n\n")


def _save_tsv(result: dict[str, Any], path: Path) -> None:
    def cell(value: Any) -> str:
        return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()

    with open(path, "w", encoding="utf-8") as fp:
        fp.write("start\tend\tspeaker\ttext\n")
        for segment in result.get("segments", []):
            fp.write(
                f"{segment.get('start', 0.0):.3f}\t{segment.get('end', 0.0):.3f}\t"
                f"{cell(segment.get('speaker', ''))}\t{cell(segment.get('text', ''))}\n"
            )


def _save_results(result: dict[str, Any], output_dir: Path, base_name: str, formats: list[str]) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    allowed = {"txt", "srt", "vtt", "json", "tsv"}
    for fmt in formats:
        if fmt not in allowed:
            continue
        out_path = output_dir / f"{base_name}.{fmt}"
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        if fmt == "json":
            with open(tmp_path, "w", encoding="utf-8") as fp:
                json.dump(result, fp, ensure_ascii=False, indent=2)
        elif fmt == "txt":
            _save_txt(result, tmp_path)
        elif fmt == "srt":
            _save_srt(result, tmp_path)
        elif fmt == "vtt":
            _save_vtt(result, tmp_path)
        elif fmt == "tsv":
            _save_tsv(result, tmp_path)
        os.replace(tmp_path, out_path)
        saved.append(str(out_path))
    return saved


def _output_location(audio_path: Path, input_root: Path, output_root: Path) -> tuple[Path, str]:
    relative = _safe_relative_path(audio_path, input_root)
    subdir = output_root / relative.parent
    short_hash = hashlib.sha1(str(relative).encode("utf-8")).hexdigest()[:8]
    return subdir, f"{audio_path.stem}__{short_hash}"


def _collect_word_stats(result: dict[str, Any]) -> tuple[list[dict[str, Any]], int, int]:
    words: list[dict[str, Any]] = []
    segment_count = 0
    speaker_segment_count = 0
    for segment in result.get("segments", []):
        segment_count += 1
        if segment.get("speaker"):
            speaker_segment_count += 1
        for word in segment.get("words") or []:
            words.append(word)
    return words, segment_count, speaker_segment_count


def _safe_duration(start: Any, end: Any) -> float:
    try:
        return max(0.0, float(end) - float(start))
    except Exception:
        return 0.0


def _diarization_rows(diarize_segments: Any) -> list[dict[str, Any]]:
    if diarize_segments is None:
        return []
    if hasattr(diarize_segments, "to_dict"):
        try:
            return list(diarize_segments.to_dict("records"))
        except Exception:
            return []
    if isinstance(diarize_segments, list):
        return [row for row in diarize_segments if isinstance(row, dict)]
    return []


def calculate_quality(result: dict[str, Any], diarize_segments: Any, duration_sec: Optional[float]) -> dict[str, Any]:
    segments = result.get("segments", [])
    words, segment_count, speaker_segment_count = _collect_word_stats(result)
    speech_duration = sum(_safe_duration(seg.get("start"), seg.get("end")) for seg in segments)
    speaker_duration: dict[str, float] = defaultdict(float)
    speaker_switches = 0
    last_speaker = None
    short_turns = 0
    speaker_turns = 0

    for segment in segments:
        speaker = segment.get("speaker")
        dur = _safe_duration(segment.get("start"), segment.get("end"))
        if speaker:
            speaker_duration[str(speaker)] += dur
            speaker_turns += 1
            if dur < 0.75:
                short_turns += 1
            if last_speaker and speaker != last_speaker:
                speaker_switches += 1
            last_speaker = speaker

    diar_rows = _diarization_rows(diarize_segments)
    diarized_duration = sum(_safe_duration(row.get("start"), row.get("end")) for row in diar_rows)

    word_count = len(words)
    assigned_words = sum(1 for word in words if word.get("speaker"))
    timestamped_words = sum(1 for word in words if word.get("start") is not None and word.get("end") is not None)
    word_scores = [float(word["score"]) for word in words if isinstance(word.get("score"), (int, float))]
    avg_word_score = sum(word_scores) / len(word_scores) if word_scores else None

    assignment_coverage = assigned_words / word_count if word_count else speaker_segment_count / segment_count if segment_count else 0.0
    unassigned_word_ratio = 1.0 - assignment_coverage if word_count else 1.0 - assignment_coverage
    missing_timestamp_ratio = 1.0 - timestamped_words / word_count if word_count else 0.0
    diarization_coverage = diarized_duration / speech_duration if speech_duration > 0 else 0.0
    switches_per_min = speaker_switches / max((speech_duration or duration_sec or 1.0) / 60.0, 1e-6)
    short_turn_ratio = short_turns / speaker_turns if speaker_turns else 0.0
    dominant_ratio = max(speaker_duration.values()) / sum(speaker_duration.values()) if speaker_duration else 0.0

    flags: list[str] = []
    score = 100.0
    if not speaker_duration:
        flags.append("no_speaker_labels")
        score -= 35
    if assignment_coverage < 0.85:
        flags.append("low_speaker_assignment_coverage")
    score -= max(0.0, 0.98 - assignment_coverage) * 35
    if missing_timestamp_ratio > 0.15:
        flags.append("many_missing_word_timestamps")
    score -= missing_timestamp_ratio * 15
    if avg_word_score is not None and avg_word_score < 0.55:
        flags.append("low_alignment_score")
        score -= (0.55 - avg_word_score) * 35
    if diarization_coverage and diarization_coverage < 0.80:
        flags.append("low_diarization_speech_coverage")
        score -= (0.80 - diarization_coverage) * 20
    if switches_per_min > 18:
        flags.append("many_speaker_switches")
        score -= min(12.0, (switches_per_min - 18) * 0.5)
    if short_turn_ratio > 0.35:
        flags.append("many_short_turns")
        score -= min(10.0, short_turn_ratio * 12)
    if len(speaker_duration) > 3 and dominant_ratio < 0.45:
        flags.append("possible_speaker_oversplitting")
        score -= 5
    if segment_count == 0 or not "".join(seg.get("text", "") for seg in segments).strip():
        flags.append("empty_transcript")
        score = min(score, 35)

    score = round(max(0.0, min(100.0, score)), 1)
    if score >= 90:
        grade = "A"
    elif score >= 80:
        grade = "B"
    elif score >= 65:
        grade = "C"
    else:
        grade = "D"

    return {
        "overall_score": score,
        "grade": grade,
        "flags": flags,
        "speaker_count": len(speaker_duration),
        "speaker_durations_sec": {speaker: round(value, 3) for speaker, value in sorted(speaker_duration.items())},
        "speaker_assignment_coverage": round(assignment_coverage, 4),
        "unassigned_word_ratio": round(unassigned_word_ratio, 4),
        "missing_word_timestamp_ratio": round(missing_timestamp_ratio, 4),
        "avg_word_score": round(avg_word_score, 4) if avg_word_score is not None else None,
        "diarization_speech_coverage": round(diarization_coverage, 4),
        "speaker_switches_per_min": round(switches_per_min, 3),
        "short_turn_ratio": round(short_turn_ratio, 4),
        "dominant_speaker_ratio": round(dominant_ratio, 4),
        "word_count": word_count,
        "segment_count": segment_count,
        "speech_duration_sec": round(speech_duration, 3),
        "diarized_duration_sec": round(diarized_duration, 3),
    }


class WhisperXRuntime:
    def __init__(self, config: dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.whisperx = None
        self.asr_model = None
        self.align_models: dict[str, tuple[Any, Any]] = {}
        self.diarize_model = None
        self.diarization_model_name: Optional[str] = None

    @property
    def device(self) -> str:
        return self.config["device"]

    def _import_whisperx(self) -> Any:
        if self.whisperx is None:
            import whisperx

            self.whisperx = whisperx
        return self.whisperx

    def clear_cuda(self) -> None:
        gc.collect()
        if self.device == "cuda":
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass

    def unload_asr(self) -> None:
        if self.asr_model is not None:
            del self.asr_model
            self.asr_model = None
        self.clear_cuda()

    def unload_alignment(self) -> None:
        self.align_models.clear()
        self.clear_cuda()

    def unload_diarization(self) -> None:
        if self.diarize_model is not None:
            del self.diarize_model
            self.diarize_model = None
        self.clear_cuda()

    def unload_all(self) -> None:
        self.unload_asr()
        self.unload_alignment()
        self.unload_diarization()

    def load_asr(self) -> Any:
        whisperx = self._import_whisperx()
        if self.asr_model is None:
            self.logger.info(
                f"Завантаження WhisperX ASR: {self.config['model_size']} | "
                f"{self.device}/{self.config['compute_type']} | batch={self.config['batch_size']}"
            )
            self.asr_model = whisperx.load_model(
                self.config["model_size"],
                self.device,
                compute_type=self.config["compute_type"],
                language=self.config["language"],
            )
        return self.asr_model

    def load_align_model(self, language_code: str) -> tuple[Any, Any]:
        whisperx = self._import_whisperx()
        if language_code not in self.align_models:
            self.logger.info(f"Завантаження alignment-моделі для мови: {language_code}")
            self.align_models[language_code] = whisperx.load_align_model(language_code=language_code, device=self.device)
        return self.align_models[language_code]

    def load_diarization(self) -> Any:
        if self.diarize_model is None:
            from whisperx.diarize import DiarizationPipeline as WxDiarizationPipeline

            model_name = self.config.get("effective_diarization_model") or _effective_diarization_model(self.config)
            token = self.config.get("hf_token") or None
            offline = not token and model_name == self.config.get("offline_diarization_model")
            old_offline = os.environ.get("HF_HUB_OFFLINE")
            if offline:
                os.environ["HF_HUB_OFFLINE"] = "1"
            try:
                self.logger.info(f"Завантаження pyannote diarization pipeline: {model_name}")
                self.diarize_model = WxDiarizationPipeline(model_name=model_name, token=token, device=self.device)
                self.diarization_model_name = model_name
            finally:
                if offline:
                    if old_offline is None:
                        os.environ.pop("HF_HUB_OFFLINE", None)
                    else:
                        os.environ["HF_HUB_OFFLINE"] = old_offline
        return self.diarize_model

    def maybe_release_for_low_memory(self) -> None:
        if not self.config.get("reuse_models", True):
            self.unload_all()


def transcribe_file(
    audio_path: Path,
    config: dict[str, Any],
    logger: logging.Logger,
    tracker: ProgressTracker,
    runtime: WhisperXRuntime,
    index: int,
    total_files: int,
    config_hash_value: str,
    duration_sec: Optional[float],
) -> dict[str, Any]:
    output_root = Path(config["output_dir"])
    input_root = Path(config["input_dir"])
    output_dir, base_name = _output_location(audio_path, input_root, output_root)
    file_start = time.perf_counter()
    fingerprint = _file_fingerprint(audio_path)

    tracker.update_file(
        audio_path,
        status="running",
        stage="start",
        started_at=_now_iso(),
        index=index,
        total_files=total_files,
        fingerprint=fingerprint,
        config_hash=config_hash_value,
        duration_sec=duration_sec,
        output_dir=str(output_dir),
        output_base=base_name,
        attempts=(tracker.record_for(audio_path).get("attempts") or 0) + 1,
        error=None,
        error_kind=None,
    )

    logger.info("")
    logger.info("=" * 80)
    logger.info(f"ФАЙЛ [{index}/{total_files}]: {audio_path}")
    logger.info(f"Тривалість: {_format_seconds(duration_sec)} | Початок: {_now_iso()}")
    logger.info("=" * 80)

    try:
        whisperx = runtime._import_whisperx()

        t0 = time.perf_counter()
        tracker.update_file(audio_path, stage="load_audio")
        logger.info("▶ Крок 1/5: Завантаження аудіо")
        audio = whisperx.load_audio(str(audio_path))
        _log_step(logger, "Крок 1 · завантаження аудіо", time.perf_counter() - t0, config["device"])

        t0 = time.perf_counter()
        tracker.update_file(audio_path, stage="transcribe")
        logger.info("▶ Крок 2/5: Транскрибація WhisperX")
        asr_model = runtime.load_asr()
        result = asr_model.transcribe(audio, batch_size=config["batch_size"], language=config["language"])
        detected_lang = result.get("language", config["language"])
        language_retry: Optional[dict[str, Any]] = None
        preferred_languages = set(config.get("preferred_languages") or [])
        fallback_language = config.get("fallback_language")
        if (
            config.get("retry_unexpected_language", True)
            and config.get("language") is None
            and fallback_language
            and preferred_languages
            and detected_lang
            and detected_lang not in preferred_languages
        ):
            logger.warning(
                f"   ⚠ Неочікувана мова '{detected_lang}' для Telegram UA/RU набору — "
                f"повторюємо ASR з language='{fallback_language}'"
            )
            language_retry = {"initial_detected_language": detected_lang, "fallback_language": fallback_language}
            result = asr_model.transcribe(audio, batch_size=config["batch_size"], language=fallback_language)
            result.setdefault("pipeline_warnings", []).append(
                f"unexpected_language_retry: {detected_lang} -> {fallback_language}"
            )
            detected_lang = result.get("language", fallback_language) or fallback_language
        logger.info(f"   Визначена мова: {detected_lang} | Сегментів: {len(result.get('segments', []))}")
        _log_step(logger, "Крок 2 · транскрибація", time.perf_counter() - t0, config["device"])

        if not config.get("reuse_models", True):
            runtime.unload_asr()

        t0 = time.perf_counter()
        if config["align_words"] and detected_lang:
            tracker.update_file(audio_path, stage="align")
            logger.info("▶ Крок 3/5: Вирівнювання слів")
            try:
                model_a, metadata = runtime.load_align_model(detected_lang)
                result = whisperx.align(
                    result.get("segments", []),
                    model_a,
                    metadata,
                    audio,
                    config["device"],
                    return_char_alignments=False,
                )
                _log_step(logger, "Крок 3 · вирівнювання", time.perf_counter() - t0, config["device"])
            except Exception as exc:
                logger.warning(f"   ⚠ Вирівнювання недоступне для '{detected_lang}': {exc} — продовжуємо")
                result.setdefault("pipeline_warnings", []).append(f"alignment_failed: {exc}")
                _log_step(logger, "Крок 3 · вирівнювання (помилка)", time.perf_counter() - t0, config["device"])
        else:
            logger.info("▶ Крок 3/5: Вирівнювання пропущено")
            _log_step(logger, "Крок 3 · вирівнювання (пропущено)", time.perf_counter() - t0, config["device"])

        if not config.get("reuse_models", True):
            runtime.unload_alignment()

        t0 = time.perf_counter()
        diarize_segments = None
        speaker_embeddings = None
        if not config.get("effective_diarization_model"):
            logger.warning("   ⚠ Модель діаризації не налаштована — діаризацію пропущено")
            result.setdefault("pipeline_warnings", []).append("diarization_skipped_no_model")
        else:
            tracker.update_file(audio_path, stage="diarize")
            logger.info("▶ Крок 4/5: Діаризація pyannote")
            diarize_model = runtime.load_diarization()
            diarize_kwargs: dict[str, Any] = {}
            if config.get("num_speakers"):
                diarize_kwargs["num_speakers"] = config["num_speakers"]
            else:
                diarize_kwargs["min_speakers"] = config["min_speakers"]
                diarize_kwargs["max_speakers"] = config["max_speakers"]
            # Enable speaker embedding extraction for speaker tracking
            diarize_kwargs["return_embeddings"] = True
            diarize_segments = diarize_model(audio, **diarize_kwargs)
            
            # Extract speaker embeddings if available
            speaker_embeddings = None
            if isinstance(diarize_segments, tuple) and len(diarize_segments) >= 2:
                diarize_segments, speaker_embeddings = diarize_segments[0], diarize_segments[1]
                if speaker_embeddings:
                    logger.info(f"   ✓ Отримано speaker embeddings для {len(speaker_embeddings)} speaker(s)")
            
            result = whisperx.assign_word_speakers(diarize_segments, result)
            speakers = sorted({seg.get("speaker", "") for seg in result.get("segments", []) if seg.get("speaker")})
            logger.info(f"   Знайдено спікерів: {len(speakers)} → {speakers}")
        _log_step(logger, "Крок 4 · діаризація", time.perf_counter() - t0, config["device"])

        if not config.get("reuse_models", True):
            runtime.unload_diarization()

        t0 = time.perf_counter()
        tracker.update_file(audio_path, stage="quality")
        logger.info("▶ Крок 5/5: QA-оцінка та збереження")
        quality = calculate_quality(result, diarize_segments, duration_sec)
        result["quality"] = quality
        
        # Store speaker embeddings if extracted
        if speaker_embeddings:
            result["speaker_embeddings"] = {
                speaker_id: emb.tolist() if hasattr(emb, 'tolist') else emb
                for speaker_id, emb in speaker_embeddings.items()
            }
        
        result["source"] = {
            "path": str(audio_path),
            "relative_key": _file_key(audio_path, input_root),
            "fingerprint": fingerprint,
            "duration_sec": duration_sec,
        }
        result["pipeline"] = {
            "model_size": config["model_size"],
            "language_requested": config["language"],
            "language_detected": detected_lang,
            "language_retry": language_retry,
            "preferred_languages": config.get("preferred_languages"),
            "fallback_language": config.get("fallback_language"),
            "diarization_model": runtime.diarization_model_name or config.get("effective_diarization_model"),
            "device": config["device"],
            "compute_type": config["compute_type"],
            "batch_size": config["batch_size"],
            "config_hash": config_hash_value,
            "completed_at": _now_iso(),
        }

        saved = _save_results(result, output_dir, base_name, _normalize_formats(config["output_format"]))
        elapsed = time.perf_counter() - file_start
        real_time_factor = elapsed / duration_sec if duration_sec and duration_sec > 0 else None
        for output_path in saved:
            logger.info(f"   ✓ {output_path}")
        logger.info(f"   QA: {quality['overall_score']}/100 ({quality['grade']}) | flags={quality['flags'] or 'none'}")
        _log_step(logger, "Крок 5 · QA+збереження", time.perf_counter() - t0, config["device"])

        record = tracker.update_file(
            audio_path,
            status="done",
            stage="done",
            completed_at=_now_iso(),
            processing_time_sec=round(elapsed, 3),
            real_time_factor=round(real_time_factor, 4) if real_time_factor is not None else None,
            detected_language=detected_lang,
            output_paths=saved,
            quality=quality,
            error=None,
            error_kind=None,
        )
        tracker.write_quality_csv()

        logger.info(f"Загальний час файлу: {_format_seconds(elapsed)} | RTF={real_time_factor:.3f}" if real_time_factor else f"Загальний час файлу: {_format_seconds(elapsed)}")
        flush_logs()
        return record

    except Exception as exc:
        elapsed = time.perf_counter() - file_start
        error_kind = "cuda_oom" if _looks_like_oom(exc) else "error"
        logger.error(f"✗ ПОМИЛКА при обробці '{audio_path.name}' через {_format_seconds(elapsed)}: {exc}")
        logger.debug(traceback.format_exc())
        runtime.clear_cuda()
        record = tracker.mark_error(
            audio_path,
            error=str(exc),
            error_kind=error_kind,
            processing_time_sec=round(elapsed, 3),
            fingerprint=fingerprint,
            config_hash=config_hash_value,
            duration_sec=duration_sec,
        )
        flush_logs()
        return record


def _progress_summary(tracker: ProgressTracker, files: list[Path]) -> dict[str, Any]:
    statuses = defaultdict(int)
    total_duration = 0.0
    done_duration = 0.0
    processing_time = 0.0
    scores: list[float] = []

    for file_path in files:
        record = tracker.record_for(file_path)
        status = record.get("status", "pending")
        statuses[status] += 1
        duration = record.get("duration_sec")
        if isinstance(duration, (int, float)) and math.isfinite(duration):
            total_duration += duration
            if status in {"done", "skipped"}:
                done_duration += duration
        elapsed = record.get("processing_time_sec")
        if isinstance(elapsed, (int, float)):
            processing_time += elapsed
        score = (record.get("quality") or {}).get("overall_score")
        if isinstance(score, (int, float)):
            scores.append(float(score))

    eta = None
    if done_duration > 0 and processing_time > 0 and total_duration > done_duration:
        seconds_per_audio_second = processing_time / done_duration
        eta = (total_duration - done_duration) * seconds_per_audio_second

    return {
        "total_files": len(files),
        "statuses": dict(statuses),
        "total_duration_sec": round(total_duration, 3) if total_duration else None,
        "done_duration_sec": round(done_duration, 3) if done_duration else None,
        "progress_by_duration": round(done_duration / total_duration, 4) if total_duration else None,
        "processing_time_sec": round(processing_time, 3),
        "eta_sec": round(eta, 3) if eta is not None else None,
        "average_quality_score": round(sum(scores) / len(scores), 2) if scores else None,
        "updated_at": _now_iso(),
    }


def process_directory(config: dict[str, Any]) -> int:
    logger = setup_logging(config["log_dir"])
    input_root = Path(config["input_dir"])
    output_root = Path(config["output_dir"])
    state_dir = Path(config["state_dir"])
    config["effective_diarization_model"] = _effective_diarization_model(config)
    cfg_hash = _config_hash(config)
    tracker = ProgressTracker(state_dir, input_root, cfg_hash, logger)

    logger.info("╔══════════════════════════════════════════════════════════════════════╗")
    logger.info("║       WhisperX — resumable batch transcription + diarization         ║")
    logger.info("╚══════════════════════════════════════════════════════════════════════╝")
    logger.info(f"Вхідна директорія : {input_root.resolve()}")
    logger.info(f"Вихідна директорія: {output_root.resolve()}")
    logger.info(f"Progress state    : {state_dir.resolve()}")
    logger.info(f"Лог-файл          : {_log_path}")
    logger.info(
        f"Модель: {config['model_size']} | Пристрій: {config['device']} ({config['compute_type']}) | "
        f"batch={config['batch_size']} | Мова: {config['language'] or 'auto'}"
    )
    logger.info(
        f"Спікери: {config.get('num_speakers') or str(config['min_speakers']) + '–' + str(config['max_speakers'])} | "
        f"Diarization: {config.get('effective_diarization_model')} | "
        f"Вирівнювання: {'так' if config['align_words'] else 'ні'} | Формати: {_normalize_formats(config['output_format'])}"
    )

    if not config.get("hf_token") and config.get("effective_diarization_model") == config.get("diarization_model"):
        message = "HF_TOKEN не знайдено, і cached offline diarization fallback недоступний."
        if config.get("dry_run"):
            logger.warning(message)
        else:
            logger.error(message)
            return 2
    if not config.get("hf_token") and config.get("effective_diarization_model") == config.get("offline_diarization_model"):
        logger.warning(
            "HF_TOKEN не знайдено — використовуємо cached offline fallback "
            f"{config.get('effective_diarization_model')}. Для максимальної якості community-1 додайте HF_TOKEN."
        )

    try:
        audio_files = find_audio_files(config["input_dir"], config["extensions"])
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 2

    limit = config.get("limit")
    if limit:
        audio_files = audio_files[: int(limit)]

    if not audio_files:
        logger.warning(f"Аудіофайлів не знайдено у: {config['input_dir']}")
        return 1

    logger.info(f"Знайдено аудіофайлів: {len(audio_files)}")
    logger.info(f"Config hash: {cfg_hash}")
    tracker.append_event({"event": "run_started", "config_hash": cfg_hash, "total_files": len(audio_files)})

    if config.get("probe_durations", True):
        logger.info("Пробуємо тривалість файлів через ffprobe для ETA...")
        for idx, audio_path in enumerate(audio_files, 1):
            record = tracker.record_for(audio_path)
            duration = record.get("duration_sec")
            fingerprint = _file_fingerprint(audio_path)
            if record.get("fingerprint") != fingerprint or not isinstance(duration, (int, float)):
                duration = probe_audio_duration(audio_path)
            tracker.update_file(
                audio_path,
                status=record.get("status", "pending"),
                stage=record.get("stage", "queued"),
                index=idx,
                total_files=len(audio_files),
                source_path=str(audio_path),
                fingerprint=fingerprint,
                duration_sec=duration,
                config_hash=cfg_hash,
            )
            if idx % 100 == 0 or idx == len(audio_files):
                logger.info(f"  ffprobe: {idx}/{len(audio_files)}")

    if config.get("dry_run"):
        summary = _progress_summary(tracker, audio_files)
        tracker.update_summary(summary)
        logger.info(f"Dry run завершено. Total duration: {_format_seconds(summary.get('total_duration_sec'))}")
        return 0

    runtime = WhisperXRuntime(config, logger)
    ok_count = 0
    skipped_count = 0
    err_count = 0
    session_start = time.perf_counter()

    try:
        for idx, audio_path in enumerate(audio_files, 1):
            fingerprint = _file_fingerprint(audio_path)
            existing = tracker.record_for(audio_path)
            duration = existing.get("duration_sec") if isinstance(existing.get("duration_sec"), (int, float)) else None

            if _shutdown_requested:
                logger.warning("Зупинку запрошено користувачем. Manifest вже збережено — можна продовжити запуском тієї ж команди.")
                break

            if config.get("resume", True) and not config.get("force") and tracker.is_completed(audio_path, fingerprint, cfg_hash):
                skipped_count += 1
                tracker.update_file(
                    audio_path,
                    status="done",
                    stage="done",
                    index=idx,
                    total_files=len(audio_files),
                    last_resume_skip_at=_now_iso(),
                    resume_skip_count=(existing.get("resume_skip_count") or 0) + 1,
                )
                logger.info(f"[{idx}/{len(audio_files)}] SKIP done: {audio_path.name}")
                continue

            summary = _progress_summary(tracker, audio_files)
            logger.info(
                f"Прогрес: files done={summary['statuses'].get('done', 0)} skipped={summary['statuses'].get('skipped', 0)} "
                f"errors={summary['statuses'].get('error', 0)} / {len(audio_files)} | "
                f"duration={summary.get('progress_by_duration') or 0:.1%} | ETA={_format_seconds(summary.get('eta_sec'))}"
            )

            attempt = 0
            while True:
                attempt += 1
                record = transcribe_file(audio_path, config, logger, tracker, runtime, idx, len(audio_files), cfg_hash, duration)
                if record.get("status") == "done":
                    ok_count += 1
                    break
                if (
                    record.get("error_kind") == "cuda_oom"
                    and config.get("adaptive_batch", True)
                    and attempt < int(config.get("max_retries", 3))
                    and int(config["batch_size"]) > 1
                ):
                    old_batch = int(config["batch_size"])
                    config["batch_size"] = max(1, old_batch // 2)
                    logger.warning(f"CUDA OOM: зменшуємо batch_size {old_batch} → {config['batch_size']} і повторюємо файл")
                    runtime.unload_all()
                    continue
                err_count += 1
                break

            summary = _progress_summary(tracker, audio_files)
            tracker.update_summary(summary)
            logger.info(
                f"Після файлу [{idx}/{len(audio_files)}]: avg QA={summary.get('average_quality_score')} | "
                f"ETA={_format_seconds(summary.get('eta_sec'))} | state={tracker.manifest_path}"
            )
            flush_logs()
    finally:
        runtime.unload_all()

    session_elapsed = time.perf_counter() - session_start
    summary = _progress_summary(tracker, audio_files)
    summary.update(
        {
            "session_elapsed_sec": round(session_elapsed, 3),
            "session_elapsed_human": _format_seconds(session_elapsed),
            "session_ok": ok_count,
            "session_skipped": skipped_count,
            "session_errors": err_count,
            "log_path": str(_log_path),
            "manifest_path": str(tracker.manifest_path),
            "quality_csv_path": str(tracker.quality_csv_path),
        }
    )
    tracker.update_summary(summary)

    logger.info("")
    logger.info("=" * 80)
    logger.info("СЕСІЯ ЗАВЕРШЕНА")
    logger.info(f"Загальний час сесії: {_format_seconds(session_elapsed)}")
    logger.info(f"Успішно: {ok_count} | Пропущено resume: {skipped_count} | Помилок: {err_count}")
    logger.info(f"Average QA score: {summary.get('average_quality_score')}")
    logger.info(f"Manifest: {tracker.manifest_path}")
    logger.info(f"Quality CSV: {tracker.quality_csv_path}")
    logger.info(f"Лог: {_log_path}")
    logger.info("=" * 80)
    flush_logs()
    return 1 if err_count else 0


def build_config_from_args(argv: Optional[list[str]] = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="WhisperX — resumable high-quality batch transcription + diarization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_dir", nargs="?", default=DEFAULT_CONFIG["input_dir"], help="Директорія з аудіофайлами")
    parser.add_argument("--output-dir", default=DEFAULT_CONFIG["output_dir"], help="Директорія для результатів")
    parser.add_argument("--log-dir", default=DEFAULT_CONFIG["log_dir"], help="Директорія для логів")
    parser.add_argument("--state-dir", default=DEFAULT_CONFIG["state_dir"], help="Директорія manifest/progress/quality")
    parser.add_argument("--model", default=DEFAULT_CONFIG["model_size"], dest="model_size")
    parser.add_argument("--language", default="auto", help="uk, ru, en або auto")
    parser.add_argument("--preferred-languages", nargs="+", default=DEFAULT_CONFIG["preferred_languages"], help="Очікувані мови для auto-detect")
    parser.add_argument("--fallback-language", default=DEFAULT_CONFIG["fallback_language"], help="Мова повтору, якщо auto-detect дав неочікувану мову")
    parser.add_argument("--device", default=DEFAULT_CONFIG["device"], choices=["cuda", "cpu", "mps"])
    parser.add_argument(
        "--compute-type",
        default=DEFAULT_CONFIG["compute_type"],
        choices=["float16", "float32", "int8", "int8_float16", "int8_float32"],
        dest="compute_type",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"], dest="batch_size")
    parser.add_argument("--min-speakers", type=int, default=DEFAULT_CONFIG["min_speakers"], dest="min_speakers")
    parser.add_argument("--max-speakers", type=int, default=DEFAULT_CONFIG["max_speakers"], dest="max_speakers")
    parser.add_argument("--num-speakers", type=int, default=None, dest="num_speakers", help="Точна кількість спікерів, якщо відома")
    parser.add_argument("--diarization-model", default=DEFAULT_CONFIG["diarization_model"], help="Основна pyannote модель діаризації")
    parser.add_argument("--offline-diarization-model", default=DEFAULT_CONFIG["offline_diarization_model"], help="Cached fallback модель без HF_TOKEN")
    parser.add_argument("--no-offline-diarization-fallback", action="store_true", help="Не використовувати cached fallback без HF_TOKEN")
    parser.add_argument("--formats", nargs="+", default=DEFAULT_CONFIG["output_format"], choices=["txt", "srt", "vtt", "json", "tsv", "all"], dest="output_format")
    parser.add_argument("--extensions", nargs="+", default=DEFAULT_CONFIG["extensions"], help="Розширення аудіофайлів")
    parser.add_argument("--no-align", action="store_true", help="Пропустити word alignment")
    parser.add_argument("--no-resume", action="store_true", help="Не пропускати вже завершені файли")
    parser.add_argument("--force", action="store_true", help="Перезаписати навіть completed файли")
    parser.add_argument("--dry-run", action="store_true", help="Тільки інвентаризація + ETA metadata, без Whisper")
    parser.add_argument("--limit", type=int, default=None, help="Обробити тільки перші N файлів")
    parser.add_argument("--no-probe-durations", action="store_true", help="Не запускати ffprobe для ETA")
    parser.add_argument("--reload-models-per-file", action="store_true", help="Економити VRAM, перезавантажуючи моделі на кожному файлі")
    parser.add_argument("--no-adaptive-batch", action="store_true", help="Не зменшувати batch_size автоматично після CUDA OOM")
    parser.add_argument("--no-language-retry", action="store_true", help="Не повторювати ASR при неочікуваній auto-detect мові")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_CONFIG["max_retries"], dest="max_retries")

    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.min_speakers < 1:
        parser.error("--min-speakers must be >= 1")
    if args.max_speakers < args.min_speakers:
        parser.error("--max-speakers must be >= --min-speakers")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    cfg = DEFAULT_CONFIG.copy()
    cfg["input_dir"] = args.input_dir
    cfg["output_dir"] = args.output_dir
    cfg["log_dir"] = args.log_dir
    cfg["state_dir"] = args.state_dir
    cfg["model_size"] = args.model_size
    cfg["language"] = None if args.language.lower() in {"", "auto", "none", "null"} else args.language
    cfg["preferred_languages"] = args.preferred_languages
    cfg["fallback_language"] = args.fallback_language
    cfg["device"] = args.device
    cfg["compute_type"] = args.compute_type
    cfg["batch_size"] = args.batch_size
    cfg["min_speakers"] = args.min_speakers
    cfg["max_speakers"] = args.max_speakers
    cfg["num_speakers"] = args.num_speakers
    cfg["diarization_model"] = args.diarization_model
    cfg["offline_diarization_model"] = args.offline_diarization_model
    cfg["offline_diarization_fallback"] = not args.no_offline_diarization_fallback
    cfg["output_format"] = args.output_format
    cfg["extensions"] = [ext if ext.startswith(".") else f".{ext}" for ext in args.extensions]
    cfg["align_words"] = not args.no_align
    cfg["resume"] = not args.no_resume
    cfg["force"] = args.force
    cfg["dry_run"] = args.dry_run
    cfg["limit"] = args.limit
    cfg["probe_durations"] = not args.no_probe_durations
    cfg["reuse_models"] = not args.reload_models_per_file
    cfg["adaptive_batch"] = not args.no_adaptive_batch
    cfg["retry_unexpected_language"] = not args.no_language_retry
    cfg["max_retries"] = args.max_retries
    cfg["hf_token"] = _get_hf_token()
    return cfg


def main(argv: Optional[list[str]] = None) -> int:
    return process_directory(build_config_from_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())