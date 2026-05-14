# Windows 11 Quick Start

Цей проєкт запускає WhisperX для пакетної транскрибації та діаризації Telegram-аудіо з resumable progress tracking, ETA і QA-оцінкою по кожному файлу.

Коротко про артефакти запуску:

- детальний лог: `logs/diarize_YYYYMMDD_HHMMSS.log`;
- resumable manifest: `output/_progress/manifest.json`;
- події прогресу: `output/_progress/events.jsonl`;
- проблемні файли: `output/_progress/errors.jsonl`;
- QA-таблиця: `output/_progress/quality_report.csv`.

## 1) Python 3.12
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2) FFmpeg 7.x
```powershell
# завантажити/розпакувати FFmpeg 7.x у C:\tools\ffmpeg7
# має бути: C:\tools\ffmpeg7\bin\ffmpeg.exe
```

## 3) Додати FFmpeg у PATH (вручну, без коду)
- Натисніть `Win + R`
- Введіть `SystemPropertiesAdvanced` і натисніть Enter
- Натисніть `Environment Variables...`
- У блоці **User variables** виберіть `Path` -> `Edit`
- Натисніть `New` і додайте: `C:\tools\ffmpeg7\bin` або `ffmpeg7.exe`
- Натисніть `OK` у всіх вікнах

```powershell
# перезапустити Cursor/термінал, потім перевірити:
where ffmpeg
where ffprobe
ffmpeg -version
ffprobe -version
```

## 4) HF token

Скопіюйте `.env.example` у `.env` і заповніть токен локально. Не комітьте `.env`.

```env
HF_TOKEN=<your-hugging-face-token>
# Optional default input folder if you do not pass an input path on CLI:
DIARIZATION_INPUT_DIR=<path-to-audio-folder>
```

Також потрібно прийняти умови моделей pyannote у Hugging Face, інакше діаризація не запуститься.

## 5) Dry run / інвентаризація

```powershell
python .\diarize.py <path-to-audio-folder> --dry-run
```

## 6) Тестовий прогін

```powershell
python .\diarize.py <path-to-audio-folder> --limit 5 --output-dir .\output-test --state-dir .\output-test\_progress --log-dir .\logs-test
```

Перевірити якість:

```powershell
Import-Csv .\output-test\_progress\quality_report.csv | Sort-Object overall_score | Select-Object -First 10
```

## 7) Повний max-quality запуск

```powershell
python .\diarize.py <path-to-audio-folder>
```

За замовчуванням використовується:

- вхідна папка: CLI аргумент `<path-to-audio-folder>` або `DIARIZATION_INPUT_DIR` у `.env`;
- модель: `large-v3`;
- мова: auto-detect для змішаних українських/російських записів;
- language guard: якщо короткий запис помилково визначено не як `uk`/`ru`, ASR повторюється з fallback `uk`;
- GPU: CUDA `float16`, початковий `batch_size=16`;
- diarization speakers: `min=1`, `max=4`;
- output formats: `txt`, `srt`, `vtt`, `json`, `tsv`.

Якщо процес зупинився, повторіть ту саму команду. Завершені файли будуть пропущені за manifest/config hash/output checks.

