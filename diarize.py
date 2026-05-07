#!/usr/bin/env python3
"""
WhisperX — пакетна транскрибація + діаризація аудіозаписів

Використання:
  python diarize.py <вхідна_директорія> [параметри]

Приклад:
  python diarize.py ./audio --output-dir ./output --model large-v3 --device cuda

HF_TOKEN береться з файлу .env (HF_TOKEN=hf_...)
"""

import gc
import json
import logging
import os
import signal
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────────────────────
# Завантаження .env
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# КОНФІГУРАЦІЯ ЗА ЗАМОВЧУВАННЯМ
# ─────────────────────────────────────────────────────────────────────────────
dir_path = ""

DEFAULT_CONFIG: dict = {
    # Вхідна директорія з аудіофайлами (обходить усі підпапки)
    "input_dir": dir_path,

    # Директорія для збереження транскрипцій
    "output_dir": "./output",

    # Директорія для лог-файлів
    "log_dir": "./logs",

    # ── Модель Whisper ──────────────────────────────────────────────────────
    # Варіанти: tiny | base | small | medium | large-v1 | large-v2 | large-v3
    "model_size": "large-v3",

    # ── Мова ────────────────────────────────────────────────────────────────
    # ISO 639-1: "uk", "ru", "en" … або None для автовизначення (повільніше)
    "language": "uk",

    # ── Пристрій ────────────────────────────────────────────────────────────
    # "cuda" | "mps" | "cpu"
    "device": "cuda",

    # Тип обчислень: "float16" (GPU), "int8" (CPU), "float32" (точніше)
    "compute_type": "int8",

    # Більший batch_size = швидше, але більше VRAM. При помилці — зменшити
    "batch_size": 8,

    # ── Токен HuggingFace ────────────────────────────────────────────────────
    # Береться з .env файлу: HF_TOKEN=hf_...
    # Потрібно прийняти умови: pyannote/speaker-diarization-3.1
    "hf_token": os.getenv("HF_TOKEN", ""),

    # ── Діаризація ──────────────────────────────────────────────────────────
    "min_speakers": 1,
    "max_speakers": 4,

    # Прив'язка кожного слова до точного таймкоду (повільніше, але точніше)
    "align_words": True,

    # ── Формати виводу ──────────────────────────────────────────────────────
    # "txt" | "srt" | "vtt" | "json" | "tsv" | "all"
    "output_format": "txt",

    # Порог впевненості: відкинути сегменти нижче (0.0 = залишити все)
    "min_confidence": 0.0,

    # Розширення файлів для обробки
    "extensions": [".mp3", ".wav"],
}

# ─────────────────────────────────────────────────────────────────────────────
# УТИЛІТИ ПАМЯТІ
# ─────────────────────────────────────────────────────────────────────────────
try:
    import psutil
    _PSUTIL_OK = True
except ImportError:
    _PSUTIL_OK = False


def get_memory_info(device: str) -> str:
    """Повертає розширений рядок з використанням Python RAM, GPU та системної RAM."""
    parts: list[str] = []

    if device == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / 1024 ** 3
                reserved = torch.cuda.memory_reserved() / 1024 ** 3
                parts.append(f"GPU: alloc={alloc:.2f}GB reserved={reserved:.2f}GB")
        except Exception:
            pass

    if _PSUTIL_OK:
        try:
            proc = psutil.Process(os.getpid())
            rss = proc.memory_info().rss / 1024 ** 3
            vm = psutil.virtual_memory()
            parts.append(
                f"RAM: proc={rss:.2f}GB "
                f"sys={vm.used / 1024 ** 3:.1f}/{vm.total / 1024 ** 3:.1f}GB "
                f"({vm.percent:.0f}%)"
            )
        except Exception:
            pass

    return "  |  ".join(parts) if parts else "n/a"


def get_python_process_memory_gb() -> Optional[float]:
    """Пам'ять лише поточного процесу Python (RSS), GB."""
    if not _PSUTIL_OK:
        return None
    try:
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / 1024 ** 3
    except Exception:
        return None


def get_gpu_memory_info_gb(device: str) -> tuple[Optional[float], Optional[float]]:
    """Повертає (allocated, reserved) GPU пам'ять у GB для CUDA."""
    if device != "cuda":
        return None, None
    try:
        import torch
        if not torch.cuda.is_available():
            return None, None
        alloc = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        return alloc, reserved
    except Exception:
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# НАЛАШТУВАННЯ ЛОГУВАННЯ
# ─────────────────────────────────────────────────────────────────────────────
_logger: Optional[logging.Logger] = None
_log_path: Optional[Path] = None


def setup_logging(log_dir: str) -> logging.Logger:
    global _logger, _log_path

    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _log_path = log_dir_path / f"diarize_{ts}.log"

    logger = logging.getLogger("diarize")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(_log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    _logger = logger
    return logger


def flush_logs() -> None:
    if _logger:
        for handler in _logger.handlers:
            handler.flush()


# ─────────────────────────────────────────────────────────────────────────────
# ОБРОБКА СИГНАЛІВ (аварійне завершення)
# ─────────────────────────────────────────────────────────────────────────────
_shutdown_requested = False


def _signal_handler(signum: int, frame) -> None:
    global _shutdown_requested
    _shutdown_requested = True
    msg = (
        f"Отримано сигнал {signum} — завершення після поточного файлу. "
        f"Лог: {_log_path}"
    )
    if _logger:
        _logger.warning(msg)
    else:
        print(f"\n⚠ {msg}", file=sys.stderr)
    flush_logs()


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ─────────────────────────────────────────────────────────────────────────────
# ФОРМАТЕРИ ВИВОДУ
# ─────────────────────────────────────────────────────────────────────────────
def _fmt_srt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _fmt_vtt(seconds: float) -> str:
    return _fmt_srt(seconds).replace(",", ".")


def _save_txt(result: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        current_speaker = None
        for seg in result["segments"]:
            speaker = seg.get("speaker", "")
            if speaker and speaker != current_speaker:
                current_speaker = speaker
                f.write(f"\n[{speaker}]\n")
            start = _fmt_srt(seg["start"]).replace(",", ".")
            end = _fmt_srt(seg["end"]).replace(",", ".")
            f.write(f"[{start} → {end}] {seg['text'].strip()}\n")


def _save_srt(result: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(result["segments"], 1):
            speaker = seg.get("speaker", "")
            label = f"[{speaker}] " if speaker else ""
            f.write(f"{i}\n")
            f.write(f"{_fmt_srt(seg['start'])} --> {_fmt_srt(seg['end'])}\n")
            f.write(f"{label}{seg['text'].strip()}\n\n")


def _save_vtt(result: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for i, seg in enumerate(result["segments"], 1):
            speaker = seg.get("speaker", "")
            label = f"<v {speaker}>" if speaker else ""
            f.write(f"{i}\n")
            f.write(f"{_fmt_vtt(seg['start'])} --> {_fmt_vtt(seg['end'])}\n")
            f.write(f"{label}{seg['text'].strip()}\n\n")


def _save_tsv(result: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("start\tend\tspeaker\ttext\n")
        for seg in result["segments"]:
            speaker = seg.get("speaker", "")
            f.write(
                f"{seg['start']:.3f}\t{seg['end']:.3f}\t"
                f"{speaker}\t{seg['text'].strip()}\n"
            )


def _save_results(result: dict, base_name: str, output_dir: Path, fmt_list: list[str]) -> list[str]:
    saved: list[str] = []
    _ALLOWED = {"txt", "srt", "vtt", "json", "tsv"}
    for fmt in fmt_list:
        if fmt not in _ALLOWED:
            continue
        out_path = output_dir / f"{base_name}.{fmt}"
        if fmt == "json":
            with open(out_path, "w", encoding="utf-8") as fp:
                json.dump(result, fp, ensure_ascii=False, indent=2)
        elif fmt == "txt":
            _save_txt(result, out_path)
        elif fmt == "srt":
            _save_srt(result, out_path)
        elif fmt == "vtt":
            _save_vtt(result, out_path)
        elif fmt == "tsv":
            _save_tsv(result, out_path)
        saved.append(str(out_path))
    return saved


# ─────────────────────────────────────────────────────────────────────────────
# ДОПОМІЖНІ ФУНКЦІЇ ЛОГУВАННЯ
# ─────────────────────────────────────────────────────────────────────────────
def _log_step(logger: logging.Logger, label: str, elapsed: float, device: str) -> None:
    mem = get_memory_info(device)
    logger.info(f"  {'[' + label + ']':<32}  {elapsed:>8.2f}s  |  {mem}")
    py_mem = get_python_process_memory_gb()
    gpu_alloc, gpu_reserved = get_gpu_memory_info_gb(device)
    py_part = f"Python RSS={py_mem:.2f}GB" if py_mem is not None else "Python RSS=n/a"
    if gpu_alloc is not None and gpu_reserved is not None:
        gpu_part = f"GPU alloc={gpu_alloc:.2f}GB reserved={gpu_reserved:.2f}GB"
    else:
        gpu_part = "GPU=n/a"
    logger.info(f"  {'[Resource detail]':<32}  {'':>8}   {py_part}  |  {gpu_part}")


# ─────────────────────────────────────────────────────────────────────────────
# ОБРОБКА ОДНОГО ФАЙЛУ
# ─────────────────────────────────────────────────────────────────────────────
def transcribe_file(audio_path: Path, config: dict, logger: logging.Logger) -> bool:
    """Транскрибує та діаризує один аудіофайл. Повертає True при успіху."""
    import whisperx
    from whisperx.diarize import DiarizationPipeline as WxDiarizationPipeline

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    device = config["device"]
    file_start = time.perf_counter()

    logger.info("")
    logger.info("=" * 72)
    logger.info(f"  ФАЙЛ: {audio_path}")
    logger.info(f"  Початок: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 72)

    try:
        # ── Крок 1: Завантаження аудіо ─────────────────────────────────────
        t0 = time.perf_counter()
        logger.info("▶ Крок 1/4: Завантаження аудіо...")
        audio = whisperx.load_audio(str(audio_path))
        _log_step(logger, "Крок 1 · завантаження аудіо", time.perf_counter() - t0, device)

        # ── Крок 2: Транскрибація ───────────────────────────────────────────
        t0 = time.perf_counter()
        logger.info("▶ Крок 2/4: Транскрибація (Whisper)...")
        model = whisperx.load_model(
            config["model_size"],
            device,
            compute_type=config["compute_type"],
            language=config["language"],
        )
        result = model.transcribe(
            audio,
            batch_size=config["batch_size"],
            language=config["language"],
        )
        detected_lang = result.get("language", config["language"])
        logger.info(f"   Визначена мова: {detected_lang}  |  Сегментів: {len(result['segments'])}")

        del model
        gc.collect()
        if device == "cuda":
            import torch
            torch.cuda.empty_cache()

        _log_step(logger, "Крок 2 · транскрибація", time.perf_counter() - t0, device)

        # ── Крок 3: Вирівнювання слів ───────────────────────────────────────
        t0 = time.perf_counter()
        if config["align_words"]:
            logger.info("▶ Крок 3/4: Вирівнювання слів (alignment)...")
            try:
                model_a, metadata = whisperx.load_align_model(
                    language_code=detected_lang,
                    device=device,
                )
                result = whisperx.align(
                    result["segments"],
                    model_a,
                    metadata,
                    audio,
                    device,
                    return_char_alignments=False,
                )
                del model_a
                gc.collect()
                if device == "cuda":
                    import torch
                    torch.cuda.empty_cache()
                _log_step(logger, "Крок 3 · вирівнювання", time.perf_counter() - t0, device)
            except Exception as exc:
                logger.warning(
                    f"   ⚠ Вирівнювання недоступне для '{detected_lang}': {exc}"
                    " → продовжуємо без вирівнювання"
                )
                _log_step(logger, "Крок 3 · вирівнювання (помилка)", time.perf_counter() - t0, device)
        else:
            logger.info("▶ Крок 3/4: Вирівнювання пропущено")
            _log_step(logger, "Крок 3 · вирівнювання (пропущено)", time.perf_counter() - t0, device)

        # ── Крок 4: Діаризація ──────────────────────────────────────────────
        t0 = time.perf_counter()
        token = config.get("hf_token", "")
        if not token:
            logger.warning("   ⚠ HF_TOKEN не вказано — діаризацію пропущено")
            _log_step(logger, "Крок 4 · діаризація (пропущено)", time.perf_counter() - t0, device)
        else:
            logger.info("▶ Крок 4/4: Діаризація (pyannote)...")
            try:
                diarize_model = WxDiarizationPipeline(
                    token=token,
                    device=device,
                )
                diarize_segments = diarize_model(
                    audio,
                    min_speakers=config["min_speakers"],
                    max_speakers=config["max_speakers"],
                )
                result = whisperx.assign_word_speakers(diarize_segments, result)
                speakers = sorted({s.get("speaker", "?") for s in result["segments"]})
                logger.info(f"   Знайдено спікерів: {len(speakers)} → {speakers}")

                del diarize_model
                gc.collect()
                if device == "cuda":
                    import torch
                    torch.cuda.empty_cache()

                _log_step(logger, "Крок 4 · діаризація", time.perf_counter() - t0, device)
            except Exception as exc:
                logger.error(f"   ✗ Помилка діаризації: {exc}")
                _log_step(logger, "Крок 4 · діаризація (помилка)", time.perf_counter() - t0, device)

        # ── Збереження результатів ──────────────────────────────────────────
        t0 = time.perf_counter()
        of = config["output_format"]
        if of == "all":
            fmt_list = ["txt", "srt", "vtt", "json", "tsv"]
        elif isinstance(of, (list, tuple)):
            fmt_list = list(of)
        else:
            fmt_list = [of]

        saved = _save_results(result, audio_path.stem, output_dir, fmt_list)
        for s in saved:
            logger.info(f"   ✓ {s}")
        _log_step(logger, "Збереження результатів", time.perf_counter() - t0, device)

        total = time.perf_counter() - file_start
        logger.info(f"  {'─' * 52}")
        logger.info(f"  Загальний час обробки файлу: {total:.2f}s  ({total / 60:.1f} хв)")
        logger.info(f"  {'─' * 52}")
        flush_logs()
        return True

    except Exception as exc:
        elapsed = time.perf_counter() - file_start
        logger.error(f"  ✗ ПОМИЛКА при обробці '{audio_path.name}' (через {elapsed:.2f}s):")
        logger.error(traceback.format_exc())
        flush_logs()
        return False


# ─────────────────────────────────────────────────────────────────────────────
# ПОШУК АУДІОФАЙЛІВ
# ─────────────────────────────────────────────────────────────────────────────
def find_audio_files(input_dir: str, extensions: list[str]) -> list[Path]:
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Директорія не знайдена: {root.resolve()}")
    ext_set = {e.lower() for e in extensions}
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in ext_set)


# ─────────────────────────────────────────────────────────────────────────────
# ГОЛОВНИЙ ЦИКЛ
# ─────────────────────────────────────────────────────────────────────────────
def process_directory(config: dict) -> None:
    logger = setup_logging(config["log_dir"])

    logger.info("╔══════════════════════════════════════════════════════════════════════╗")
    logger.info("║        WhisperX — пакетна транскрибація + діаризація                ║")
    logger.info("╚══════════════════════════════════════════════════════════════════════╝")
    logger.info(f"Вхідна директорія : {Path(config['input_dir']).resolve()}")
    logger.info(f"Вихідна директорія: {Path(config['output_dir']).resolve()}")
    logger.info(f"Лог-файл          : {_log_path}")
    logger.info(
        f"Модель: {config['model_size']}  |  "
        f"Пристрій: {config['device']} ({config['compute_type']})  |  "
        f"Мова: {config['language'] or 'автовизначення'}"
    )
    logger.info(
        f"Спікери: {config['min_speakers']}–{config['max_speakers']}  |  "
        f"Вирівнювання: {'так' if config['align_words'] else 'ні'}  |  "
        f"Формати: {config['output_format']}"
    )

    if not config.get("hf_token"):
        logger.warning("⚠ HF_TOKEN не знайдено в .env — діаризація буде пропущена для всіх файлів")

    try:
        audio_files = find_audio_files(config["input_dir"], config["extensions"])
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return

    if not audio_files:
        logger.warning(f"Аудіофайлів не знайдено у: {config['input_dir']}")
        return

    logger.info(f"\nЗнайдено файлів: {len(audio_files)}")
    for f in audio_files:
        logger.info(f"  • {f}")

    ok_count = 0
    err_count = 0
    session_start = time.perf_counter()

    for idx, audio_path in enumerate(audio_files, 1):
        if _shutdown_requested:
            logger.warning(
                f"Зупинено на {idx - 1}/{len(audio_files)} за запитом завершення. "
                f"Оброблено: {ok_count} успішно, {err_count} помилок."
            )
            break

        logger.info(f"\n[{idx}/{len(audio_files)}] → {audio_path.name}")

        if transcribe_file(audio_path, config, logger):
            ok_count += 1
        else:
            err_count += 1

        flush_logs()

    session_elapsed = time.perf_counter() - session_start
    logger.info("")
    logger.info("=" * 72)
    logger.info("  СЕСІЯ ЗАВЕРШЕНА")
    logger.info(f"  Загальний час сесії : {session_elapsed:.1f}s  ({session_elapsed / 60:.1f} хв)")
    logger.info(f"  Успішно             : {ok_count}")
    logger.info(f"  Помилок             : {err_count}")
    logger.info(f"  Лог збережено       : {_log_path}")
    logger.info("=" * 72)
    flush_logs()


# ─────────────────────────────────────────────────────────────────────────────
# ТОЧКА ВХОДУ
# ─────────────────────────────────────────────────────────────────────────────
def _build_config_from_args() -> dict:
    import argparse

    parser = argparse.ArgumentParser(
        description="WhisperX — пакетна транскрибація + діаризація аудіо (mp3/wav)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=DEFAULT_CONFIG["input_dir"],
        help="Директорія з аудіофайлами (рекурсивний обхід)",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_CONFIG["output_dir"],
        help="Директорія для збереження транскрипцій",
    )
    parser.add_argument(
        "--log-dir",
        default=DEFAULT_CONFIG["log_dir"],
        help="Директорія для лог-файлів",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_CONFIG["model_size"],
        choices=["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"],
        dest="model_size",
        help="Розмір моделі Whisper",
    )
    parser.add_argument(
        "--language",
        default=DEFAULT_CONFIG["language"],
        help="Мова (uk, ru, en … або порожньо для автовизначення)",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_CONFIG["device"],
        choices=["cuda", "cpu", "mps"],
    )
    parser.add_argument(
        "--compute-type",
        default=DEFAULT_CONFIG["compute_type"],
        choices=["float16", "float32", "int8"],
        dest="compute_type",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_CONFIG["batch_size"],
        dest="batch_size",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=DEFAULT_CONFIG["min_speakers"],
        dest="min_speakers",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=DEFAULT_CONFIG["max_speakers"],
        dest="max_speakers",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=DEFAULT_CONFIG["output_format"],
        choices=["txt", "srt", "vtt", "json", "tsv", "all"],
        dest="output_format",
    )
    parser.add_argument(
        "--no-align",
        action="store_true",
        help="Пропустити вирівнювання слів",
    )

    args = parser.parse_args()
    cfg = DEFAULT_CONFIG.copy()
    requested_input = args.input_dir
    # Якщо користувач залишив шаблонний аргумент ./audio, але папки нема,
    # використовуємо DEFAULT_CONFIG["input_dir"] як безпечний fallback.
    if requested_input in {"./audio", "audio"} and not Path(requested_input).exists():
        requested_input = DEFAULT_CONFIG["input_dir"]
    cfg["input_dir"] = requested_input
    cfg["output_dir"] = args.output_dir
    cfg["log_dir"] = args.log_dir
    cfg["model_size"] = args.model_size
    cfg["language"] = args.language or None
    cfg["device"] = args.device
    cfg["compute_type"] = args.compute_type
    cfg["batch_size"] = args.batch_size
    cfg["min_speakers"] = args.min_speakers
    cfg["max_speakers"] = args.max_speakers
    cfg["output_format"] = args.output_format
    cfg["align_words"] = not args.no_align
    return cfg


if __name__ == "__main__":
    process_directory(_build_config_from_args())
