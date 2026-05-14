#!/usr/bin/env python3
"""Check which files failed to parse."""

import json
from pathlib import Path

from filename_parser import parse_telegram_filename

# Load manifest
manifest = json.load(open("output/_progress/manifest.json", encoding="utf-8"))

# Find files that failed to parse
failed = []
for file_key in manifest.get("files", {}).keys():
    filename = Path(file_key).name
    if not parse_telegram_filename(filename):
        failed.append(filename)

print(f"Found {len(failed)} parsing failures out of {len(manifest['files'])}")
print("\nFirst 10 failures:")
for fname in failed[:10]:
    print(f"  {fname}")
