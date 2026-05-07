---
name: copilot-diarization-max-quality
format: AMD
owner: GitHub Copilot
qa_owner: GitHub Copilot
status: template
created: 2026-05-07
---

# AMD Task Ledger — WhisperX max-quality diarization

## 0. Trigger

Process a local audio folder through WhisperX with resumable transcription, word alignment, speaker diarization, progress tracking, and per-file quality scoring.

## 1. Non-negotiable constraints

- Quality is higher priority than runtime.
- Use `large-v3` unless a local test proves it cannot run.
- Use CUDA `float16` on a suitable GPU; fall back only when required.
- Keep raw private audio, generated transcripts, logs, archives, and `.env` out of git.
- The current Copilot chat is the controller and final QA gate.
- The run must be resumable after interruption.

## 2. Runtime configuration

- ASR model: `large-v3`.
- Language: `auto` for mixed-language audio.
- Language guard: if auto-detect returns a language outside expected values, retry ASR with fallback language.
- Device: `cuda` by default.
- Compute: `float16` for GPU quality/speed balance.
- Initial batch size: `16`; adaptive retry halves it only after CUDA OOM.
- Alignment: enabled.
- Diarization: pyannote via WhisperX, `min_speakers=1`, `max_speakers=4`.
- Outputs: `txt`, `srt`, `vtt`, `json`, `tsv`.
- Progress state: `output/_progress/manifest.json`, `events.jsonl`, `quality_report.csv`, `errors.jsonl`.

## 3. P0 tasks

- [ ] Inspect repository, audio inventory, hardware, FFmpeg, and git state.
- [ ] Confirm `HF_TOKEN` exists locally and pyannote terms are accepted.
- [ ] Run dry-run inventory and duration probe.
- [ ] Run a small pilot and inspect `quality_report.csv`.
- [ ] Complete the full production run.
- [ ] Verify resumability with a repeat run.
- [ ] Archive reports/transcripts outside git if needed.

## 4. P1 tasks

- [ ] Compare forced language vs auto-detect if language output looks suspicious.
- [ ] If speaker oversplitting is common, rerun affected files with tighter speaker limits.
- [ ] If CUDA memory fails repeatedly, lower batch size or switch only failing files to a lower-memory compute type.

## 5. Acceptance criteria

- Every supported audio file is either `done`, validly skipped by resume, or recorded in `errors.jsonl` with a clear error.
- Every completed file has output artifacts and a quality record.
- Progress can resume with the same command without reprocessing completed files.
- Final summary includes processed count, errors, average QA score, manifest path, quality report path, and production log path.

## 6. QA rubric

For each completed file, Copilot checks:

- `overall_score` target: A/B preferred; C files are review/rerun candidates; D files are failures unless the source audio is empty/noisy.
- `speaker_assignment_coverage` should be high for diarized speech.
- `unassigned_word_ratio`, `missing_word_timestamp_ratio`, `many_speaker_switches`, and `many_short_turns` flags identify likely bad diarization.
- Empty transcript, missing speaker labels, or missing outputs are blockers.
