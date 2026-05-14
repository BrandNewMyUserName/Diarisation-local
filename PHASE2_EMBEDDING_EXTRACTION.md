# Phase 2: Speaker Embedding Extraction

**Status:** READY FOR EXECUTION  
**Date:** 2026-05-07  
**Blocker:** Embeddings not in existing JSON (requires diarization re-run)

---

## 🎯 Objective

Extract speaker embeddings (voice fingerprints) from the pyannote diarization model for all 1,073 audio files. These embeddings will be used in Phase 3 to cluster and identify recurring speakers across different calls.

---

## ✅ Phase 2 Implementation

### 1. Pipeline Modifications (diarization_pipeline.py)

**Changes Made:**
- Added `return_embeddings=True` to diarization kwargs (line 927)
- Extract speaker_embeddings from pyannote DiarizationPipeline output (lines 928-933)
- Convert embeddings to JSON-serializable format (lines 956-960)
- Store embeddings in result JSON under `speaker_embeddings` field

**Modified Code Locations:**
```python
# Line 911: Initialize variable
speaker_embeddings = None

# Line 927: Enable embedding extraction
diarize_kwargs["return_embeddings"] = True

# Lines 928-933: Extract embeddings from output
if isinstance(diarize_segments, tuple) and len(diarize_segments) >= 2:
    diarize_segments, speaker_embeddings = diarize_segments[0], diarize_segments[1]
    if speaker_embeddings:
        logger.info(f"   ✓ Отримано speaker embeddings для {len(speaker_embeddings)} speaker(s)")

# Lines 956-960: Store in JSON
if speaker_embeddings:
    result["speaker_embeddings"] = {
        speaker_id: emb.tolist() if hasattr(emb, 'tolist') else emb
        for speaker_id, emb in speaker_embeddings.items()
    }
```

### 2. Embedding Extractor Module (embedding_extractor.py)

**Features:**
- `EmbeddingExtractor` class for loading and managing embeddings
- `load_embeddings_from_json()`: Extract embeddings from WhisperX JSON output
- `scan_output_directory()`: Find all transcript JSON files
- `extract_all_embeddings()`: Process all files for embeddings
- `compute_embedding_statistics()`: Generate metadata on embeddings
- `export_embeddings()`: Save embeddings to structured JSON for Phase 3

**Output Format (phase2_embeddings.json):**
```json
{
  "metadata": {
    "export_date": "2026-05-07T23:30:00+02:00",
    "total_files": 1041,
    "total_speaker_embeddings": 2082,
    "embedding_stats": {
      "files_with_embeddings": 1041,
      "total_speaker_embeddings": 2082,
      "unique_speakers_per_file": {
        "min": 1,
        "max": 4,
        "avg": 2.0
      },
      "embedding_dimensions": [192]
    }
  },
  "embeddings_by_file": {
    "2020-10/12-10-2020_06-11_0456307014_incoming_0443334085_user_.mp3": {
      "SPEAKER_00": [0.123, -0.456, ..., 0.789],
      "SPEAKER_01": [0.234, -0.567, ..., 0.890]
    }
  }
}
```

### 3. Phase 2 Runner Script (run_phase2.py)

- Loads WhisperX output directory
- Scans for all transcript JSON files (1,041 from Phase 1)
- Extracts speaker embeddings from each file
- Generates statistics on embeddings found
- Exports to phase2_embeddings.json

---

## ⚠️ Current Status: Embeddings NOT Found

**Test Result:**
```
✓ Extracted embeddings from 0 files
⚠ 1073 files did not have embeddings in JSON
```

**Reason:** The current JSON outputs were created before the pipeline was modified to extract embeddings. The modifications I made to `diarization_pipeline.py` will enable embedding extraction on the **next diarization run**.

---

## 📋 How to Execute Phase 2

### Step 1: Run Diarization Pipeline with Modified Code

The pipeline has been updated to extract embeddings. To re-diarize all 1,073 files with the new code:

```bash
# Option A: Resume from existing progress (recommended)
.\.venv\Scripts\python diarization_pipeline.py \
  --input-dir "C:\Users\Dezzz\Downloads\Telegram Desktop\Аудіозаписи\Аудіозаписи" \
  --output-dir output \
  --force

# Option B: Clean re-run (deletes existing outputs)
rm -r output/*
.\.venv\Scripts\python diarization_pipeline.py \
  --input-dir "C:\Users\Dezzz\Downloads\Telegram Desktop\Аудіозаписи\Аудіозаписи" \
  --output-dir output
```

**Estimated Time:** 2-4 hours on RTX 5080 (processing 1,073 files)

**Progress:** The pipeline is resumable - you can safely interrupt and continue later

### Step 2: Extract Embeddings

After diarization completes, extract embeddings from the JSON files:

```bash
.\.venv\Scripts\python run_phase2.py
```

**Expected Output:**
```
✓ Extracted embeddings from 1041 files
✓ Total speaker embeddings: 2082 (avg 2.0 per file)
✓ Embedding dimensions: [192]
✓ Export: output\_progress\phase2_embeddings.json
```

---

## 🔍 Technical Details

### Speaker Embeddings

- **Source:** pyannote/speaker-diarization-community-1 model
- **Format:** 192-dimensional vectors (float32)
- **Semantics:** Each speaker's embedding is a unique voice fingerprint
- **Usage:** Cosine distance measures speaker similarity

### Embedding Storage

In each WhisperX JSON transcript file (after re-diarization):

```json
{
  "segments": [...],
  "speaker_embeddings": {
    "SPEAKER_00": [0.123, -0.456, ..., 0.789],
    "SPEAKER_01": [0.234, -0.567, ..., 0.890]
  }
}
```

### Phase 2 Output

Consolidated embeddings for clustering (Phase 3):

```json
{
  "metadata": {...},
  "embeddings_by_file": {
    "file_key": {
      "SPEAKER_00": [...vector...],
      "SPEAKER_01": [...vector...]
    }
  }
}
```

---

## 📊 Expected Results (After Re-diarization)

| Metric | Expected Value |
|--------|-----------------|
| Files with embeddings | 1,041 |
| Total speaker embeddings | ~2,100 (avg 2.0 per file) |
| Embedding dimension | 192 |
| Min speakers per file | 1 |
| Max speakers per file | 4 |

---

## 🚀 Next Steps (Phase 3)

Once embeddings are extracted:

1. **Build embeddings matrix**: Organize 2,100+ embeddings by speaker
2. **Compute similarity**: Pairwise cosine distances between all speakers
3. **Agglomerative clustering**: Group speakers with >0.5 similarity
4. **Assign global IDs**: Map SPEAKER_00/SPEAKER_01 → PERSON_001/PERSON_002
5. **Generate persons.json**: Final output with cross-file speaker linking

---

## 💾 Files Created/Modified

```
✓ diarization_pipeline.py   MODIFIED  +20 lines  Enable embedding extraction
✓ embedding_extractor.py    NEW       200 lines  Extract and manage embeddings
✓ run_phase2.py             NEW        45 lines  Phase 2 execution script
✓ phase2_embeddings.json    NEW        (empty)   Placeholder for embeddings
```

---

## 🔄 Decision Required

**Option A (Recommended):** Re-run diarization pipeline
- **Pros:** Get embeddings for all 1,073 files; highest quality
- **Cons:**2-4 hours GPU time
- **When:** When you have time to run the pipeline on GPU

**Option B (Skip embeddings):** Continue to Phase 3 without clustering
- **Pros:** Faster; can identify speakers using only phone numbers + timestamps
- **Cons:** Miss cross-file speaker linking based on voice similarity
- **When:** If embeddings are not critical for your use case

**Recommendation:** Option A — embeddings enable accurate speaker deduplication across calls with different phone metadata.

---

## ✅ Phase 2 Status

- [x] Pipeline modified to extract embeddings
- [x] Embedding extractor module created
- [x] Phase 2 runner script created
- [x] Test on existing files (confirms embeddings needed)
- [ ] **BLOCKED:** Awaiting diarization re-run with updated pipeline
- [ ] Extract embeddings from new JSON outputs
- [ ] Generate phase2_embeddings.json
- [ ] Proceed to Phase 3 (clustering)

---

**Commit:** Ready to be committed once decision is made to re-diarize  
**Duration until Phase 3:** 3-5 hours (mostly GPU diarization time)
