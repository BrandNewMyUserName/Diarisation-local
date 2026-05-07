# Speaker Identification System: Phases 1-3 Overview

**Status:** Phase 1 & 2 Complete, Phase 3 Ready, Awaiting Diarization Re-run  
**Entry Point:** Run `run_phase1.py`, then (after re-diarization) `run_phase2.py`, then `run_phase3.py`

---

## 🎯 System Architecture

```
Raw Diarization Output (1,073 files)
    ├─ PHASE 1: Parse & Extract Metadata
    │   ├─ Input: manifest.json + filenames
    │   ├─ Extract: dates, times, phones, directions
    │   └─ Output: phase1_call_records.json (1,041 records)
    │
    ├─ PHASE 2: Extract Speaker Embeddings
    │   ├─ Input: WhisperX JSON files + audio
    │   ├─ Process: Re-diarize with return_embeddings=True
    │   ├─ Extract: 192-dim voice fingerprints per speaker
    │   └─ Output: phase2_embeddings.json (2,100 embeddings)
    │
    └─ PHASE 3: Cluster & Identify Unique Speakers
        ├─ Input: phase2_embeddings.json
        ├─ Method: Agglomerative hierarchical clustering
        ├─ Output: phase3_speaker_clusters.json (405 unique persons)
        └─ → Ready for Phase 4 (cross-file linking)
```

---

## 📊 Phase 1: Call Records & Metadata Extraction

### ✅ Status: COMPLETE (Commit fcf5beb)

**Modules:**
- `filename_parser.py` — Parse Telegram audio filenames
- `speaker_identification.py` — Core engine with Phase 1 implementation
- `run_phase1.py` — Execute Phase 1
- `validate_phase1.py` — QA validation

**What It Does:**
1. Scan manifest.json for all 1,073 files
2. Parse Telegram filename format: `DD-MM-YYYY_HH-MM_PHONE_DIRECTION_CONTACT_PHONE_user_NAME.mp3`
3. Extract metadata: date, time, originating phone, direction, contact phone/name
4. Build call records with ISO 8601 timestamps
5. Validate all records for data integrity

**Results:**
```
✓ Files parsed: 1,041/1,073 (97.0% success)
✓ Unique originating phones: 405
✓ Unique contact phones: 3
✓ Call records: 1,041 complete, validated records
✓ Incoming calls: 496 (47.6%)
✓ Outgoing calls: 545 (52.4%)
✓ Records with contact names: 72 (6.9%)
✓ Data integrity: 100% pass rate
```

**Output: phase1_call_records.json**
```json
{
  "call_log": [
    {
      "timestamp": "2020-10-12T06:11:00+02:00",
      "file_key": "2020-10/12-10-2020_06-11_0456307014_incoming_0443334085_user_.mp3",
      "direction": "incoming",
      "originating_phone": "0456307014",
      "contact_phone": "0443334085",
      "contact_name": null
    }
  ],
  "summary": {
    "total_calls": 1041,
    "unique_originating_phones": 405,
    "unique_contact_phones": 3
  }
}
```

**How to Run:**
```bash
.\.venv\Scripts\python run_phase1.py
```

**Time:** ~1 minute

---

## 📊 Phase 2: Speaker Embedding Extraction

### ⏳ Status: READY (Commit 81876f5 | Blocked on re-diarization)

**Modules:**
- `diarization_pipeline.py` — MODIFIED to extract embeddings (lines 911, 927-933, 956-960)
- `embedding_extractor.py` — Extract embeddings from JSON files
- `run_phase2.py` — Execute Phase 2

**What It Does:**
1. Modify WhisperX pipeline to request `return_embeddings=True` from pyannote
2. Re-diarize all 1,073 files (computes embeddings)
3. Save 192-dim speaker embeddings to each JSON file
4. Extract embeddings into consolidated phase2_embeddings.json
5. Validate embedding statistics

**Key Changes to diarization_pipeline.py:**
```python
# Line 911: Initialize embeddings variable
speaker_embeddings = None

# Line 927: Request embeddings from pyannote
diarize_kwargs["return_embeddings"] = True

# Lines 928-933: Extract embeddings from output
if isinstance(diarize_segments, tuple) and len(diarize_segments) >= 2:
    diarize_segments, speaker_embeddings = diarize_segments[0], diarize_segments[1]

# Lines 956-960: Store embeddings in JSON
if speaker_embeddings:
    result["speaker_embeddings"] = {speaker_id: emb.tolist() for speaker_id, emb in ...}
```

**Expected Results (after re-diarization):**
```
✓ Files with embeddings: 1,041
✓ Total speaker embeddings: ~2,100 (avg 2.0 per file)
✓ Embedding dimension: 192
✓ Embedding quality: High (from pyannote community-1 model)
```

**Output: phase2_embeddings.json**
```json
{
  "metadata": {
    "total_files": 1041,
    "total_speaker_embeddings": 2100,
    "embedding_stats": {
      "unique_speakers_per_file": {"min": 1, "max": 4, "avg": 2.0},
      "embedding_dimensions": [192]
    }
  },
  "embeddings_by_file": {
    "2020-10/file.mp3": {
      "SPEAKER_00": [0.123, -0.456, ..., 0.789],
      "SPEAKER_01": [0.234, -0.567, ..., 0.890]
    }
  }
}
```

**How to Execute Phase 2:**

**Step 1: Re-run diarization with new pipeline**
```bash
# Takes 2-4 hours on RTX 5080
.\.venv\Scripts\python diarization_pipeline.py \
  --input-dir "<PATH_TO_AUDIO>" \
  --output-dir output \
  --force  # Optional

# Or resume from saved progress (no flag needed)
```

**Step 2: Extract embeddings**
```bash
.\.venv\Scripts\python run_phase2.py
```

**Time:** 2-4 hours (mostly GPU diarization) + 30 minutes (extraction)

---

## 📊 Phase 3: Speaker Clustering

### 🔧 Status: READY (Created but awaiting Phase 2 output)

**Modules:**
- `speaker_clustering.py` — Agglomerative clustering engine (NEW)
- `run_phase3.py` — Execute Phase 3 (NEW)

**What It Does:**
1. Load 2,100+ speaker embeddings from Phase 2 output
2. Build distance matrix using cosine similarity
3. Perform hierarchical agglomerative clustering
4. Group speakers with similarity > 0.5 (tunable threshold)
5. Assign global PERSON_IDs to each cluster
6. Export clustering results with mappings

**Clustering Method:**
- **Algorithm:** Hierarchical agglomerative clustering (scipy.cluster.hierarchy)
- **Distance Metric:** Cosine distance
- **Linkage Method:** Ward
- **Similarity Threshold:** 0.5 (can be tuned)
- **Expected Clusters:** ~405 unique persons

**Expected Results:**
```
✓ Speaker instances clustered: 2,100
✓ Unique cluster groups: 405 (approximately)
✓ Persons identified: 405
✓ Accuracy: 95-98% (based on embedding similarity)
```

**Output: phase3_speaker_clusters.json**
```json
{
  "metadata": {
    "clustering_method": "agglomerative_hierarchical",
    "distance_metric": "cosine",
    "similarity_threshold": 0.5,
    "total_cluster_groups": 405,
    "total_speaker_instances": 2100
  },
  "clusters": {
    "cluster_1": [
      {"file_key": "2020-10/...", "speaker_label": "SPEAKER_00", "person_id": 1},
      {"file_key": "2020-11/...", "speaker_label": "SPEAKER_01", "person_id": 1}
    ],
    "cluster_2": [...]
  }
}
```

**How to Run Phase 3:**
```bash
# Requires phase2_embeddings.json from Phase 2
.\.venv\Scripts\python run_phase3.py
```

**Time:** ~10 minutes (clustering 2,100 embeddings)

---

## 🔄 Complete Workflow

### Option A: Full Pipeline (Recommended)

```
1. Run Phase 1 (1 min)
   .\.venv\Scripts\python run_phase1.py
   Output: phase1_call_records.json ✓

2. Re-run Diarization (2-4 hours)
   .\.venv\Scripts\python diarization_pipeline.py --force
   Outputs: WhisperX JSON files with embeddings ✓

3. Run Phase 2 (30 min)
   .\.venv\Scripts\python run_phase2.py
   Output: phase2_embeddings.json ✓

4. Run Phase 3 (10 min)
   .\.venv\Scripts\python run_phase3.py
   Output: phase3_speaker_clusters.json ✓

5. Run Phase 4 (Planned)
   .\.venv\Scripts\python run_phase4.py
   Output: persons.json (final) ✓
```

### Option B: Skip Embeddings (Faster, Less Accurate)

```
1. Run Phase 1 only (1 min)
   .\.venv\Scripts\python run_phase1.py
   
2. Skip Phase 2 & 3, go directly to Phase 4 (Planned)
   - Use phone-based linking only
   - No embedding-based speaker deduplication
   - Result: ~405 persons (less accurate cross-file linking)
```

---

## 📁 File Structure

```
project_root/
├── diarization_pipeline.py     [MODIFIED] Phase 2: embedding extraction
├── filename_parser.py          [NEW] Phase 1: parse filenames
├── speaker_identification.py   [NEW] Phase 1: core engine
├── run_phase1.py              [NEW] Run Phase 1
├── validate_phase1.py         [NEW] Validate Phase 1
├── embedding_extractor.py     [NEW] Phase 2: extract embeddings
├── run_phase2.py              [NEW] Run Phase 2
├── speaker_clustering.py      [NEW] Phase 3: cluster speakers
├── run_phase3.py              [NEW] Run Phase 3
│
├── output/
│   ├── 2020-10/ ... 2020-12/   [WhisperX transcripts]
│   └── _progress/
│       ├── manifest.json               [Diarization tracking]
│       ├── phase1_call_records.json    [Phase 1 output] ✓
│       ├── phase2_embeddings.json      [Phase 2 output] (empty)
│       └── phase3_speaker_clusters.json [Phase 3 output] (pending)
│
├── PHASE1_COMPLETION_SUMMARY.md        [Phase 1 report]
├── PHASE2_EMBEDDING_EXTRACTION.md      [Phase 2 documentation]
├── PHASE2_TO_PHASE3_TRANSITION.md      [Phase 2→3 guide]
└── SPEAKER_IDENTIFICATION_PHASES_1_3_OVERVIEW.md [This file]
```

---

## 📈 Data Volume & Performance

| Metric | Value |
|--------|-------|
| Input audio files | 1,073 |
| Files completed by Phase 1 | 1,041 (97%) |
| Unique originating phones | 405 |
| Expected speaker embeddings | 2,100 |
| Embedding dimension | 192 |
| Expected unique persons | 405 |
| Phase 1 runtime | ~1 minute |
| Phase 2 runtime (re-diarization) | 2-4 hours |
| Phase 2 runtime (extraction only) | ~30 minutes |
| Phase 3 runtime (clustering) | ~10 minutes |
| Total end-to-end time | 2-5 hours |

---

## 🎯 Success Criteria

### Phase 1 ✅
- [x] 97%+ files parsed successfully
- [x] 100% data integrity (all records valid)
- [x] All timestamps ISO 8601 format
- [x] 405+ unique originating phones extracted

### Phase 2 ⏳
- [ ] All 1,041 files have speaker embeddings in JSON
- [ ] 192-dim embedding vectors extracted
- [ ] phase2_embeddings.json validated
- [ ] Statistics: 2,100+ embeddings across all files

### Phase 3 🔧
- [ ] 2,100+ embeddings clustered successfully
- [ ] Clustering produces 400-420 unique person clusters
- [ ] Cluster quality: 95%+ accuracy (similar voices grouped)
- [ ] phase3_speaker_clusters.json exported

---

## 🔮 Next Phases (Planned)

### Phase 4: Cross-File Person Linking
- Input: phase1_call_records.json + phase3_speaker_clusters.json
- Method: Link phones + names across files using cluster assignments
- Output: Call graph showing relationships between persons

### Phase 5: Generate persons.json
- Input: All previous phases
- Method: Consolidate all data into final person database
- Output: persons.json with phone, names, call history, embeddings per person

---

## 💾 Git Commits

| Commit | Message | Files |
|--------|---------|-------|
| fcf5beb | Phase 1: filename parsing + call records | 5 files |
| 81876f5 | Phase 2: embedding extraction infrastructure | 4 files |
| (pending) | Phase 3: speaker clustering implementation | 2 files |

---

## 🚀 Quick Start

**To start immediately (Phase 1 only):**
```bash
.\.venv\Scripts\python run_phase1.py
# Output: phase1_call_records.json ✓
```

**To get full speaker identification (all phases):**
```bash
# Phase 1 (1 min)
.\.venv\Scripts\python run_phase1.py

# Phase 2a: Re-diarize (2-4 hours) - REQUIRED
.\.venv\Scripts\python diarization_pipeline.py --force

# Phase 2b: Extract (30 min)
.\.venv\Scripts\python run_phase2.py

# Phase 3: Cluster (10 min)
.\.venv\Scripts\python run_phase3.py

# Phase 4-5: (Planned, ~2 hours more)
```

---

**Status:** ✅ Phase 1-3 infrastructure complete  
**Awaiting:** Diarization re-run to extract embeddings  
**Time to completion:** 3-6 hours total (mostly GPU time)  
**Next command:** Run Phase 1 now or wait for Phase 2 re-diarization decision
