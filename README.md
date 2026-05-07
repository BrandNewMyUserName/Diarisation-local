# Windows 11 Quick Start

Коротко про логування: для кожного запуску створюється детальний запис у `logs/diarize_YYYYMMDD_HHMMSS.log` (етап обробки, час етапу, загальний час, пам'ять Python RSS і GPU).

## 1) Python
```powershell
python -m venv .venv
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
```env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxx
```

## 5) Шлях до аудіо через код (`dir_path`)
У `diarize.py` на строчці 36 вкажіть:

```python
dir_path = 

DEFAULT_CONFIG: dict = {
    "input_dir": dir_path,
    # ...
}
```

Після цього запуск:

```powershell
python .\diarize.py
```

