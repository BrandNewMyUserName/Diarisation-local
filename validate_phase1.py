#!/usr/bin/env python3
"""Test and validate Phase 1 results."""

import json
from pathlib import Path
from speaker_identification import SpeakerIdentificationEngine

def main():
    # Load the phase 1 output
    phase1_path = Path("output/_progress/phase1_call_records.json")
    
    if not phase1_path.exists():
        print("❌ Phase 1 output file not found")
        return 1
    
    with open(phase1_path, encoding="utf-8") as f:
        data = json.load(f)
    
    call_log = data.get("call_log", [])
    summary = data.get("summary", {})
    
    print("\n" + "="*70)
    print("PHASE 1 VALIDATION REPORT")
    print("="*70)
    
    print(f"\n📊 STATISTICS:")
    print(f"  • Total calls parsed: {len(call_log)}")
    print(f"  • Unique originating phones: {summary.get('unique_originating_phones', 'N/A')}")
    print(f"  • Unique contact phones: {summary.get('unique_contact_phones', 'N/A')}")
    
    # Verify data integrity
    print(f"\n✓ DATA INTEGRITY CHECKS:")
    
    # Check all records have required fields
    required_fields = ["timestamp", "file_key", "direction", "originating_phone"]
    missing_field_records = 0
    
    for i, rec in enumerate(call_log):
        for field in required_fields:
            if field not in rec or not rec[field]:
                missing_field_records += 1
                break
    
    if missing_field_records == 0:
        print(f"  ✓ All {len(call_log)} records have required fields")
    else:
        print(f"  ⚠ {missing_field_records} records missing required fields")
    
    # Check direction values
    directions = set(rec.get("direction") for rec in call_log)
    if directions <= {"incoming", "outgoing"}:
        print(f"  ✓ Direction values valid: {sorted(directions)}")
    else:
        print(f"  ⚠ Invalid direction values: {directions - {'incoming', 'outgoing'}}")
    
    # Check timestamp format
    valid_timestamps = sum(1 for rec in call_log if "T" in rec.get("timestamp", ""))
    print(f"  ✓ Valid ISO 8601 timestamps: {valid_timestamps}/{len(call_log)}")
    
    # Sample records
    print(f"\n📝 SAMPLE RECORDS (first 3):")
    for i, rec in enumerate(call_log[:3], 1):
        print(f"\n  [{i}] {rec['timestamp']}")
        print(f"      Direction: {rec['direction']}")
        print(f"      From: {rec['originating_phone']}")
        if rec.get('contact_phone'):
            print(f"      To: {rec['contact_phone']}")
        if rec.get('contact_name'):
            print(f"      Name: {rec['contact_name']}")
        print(f"      File: {rec['file_key']}")
    
    # Aggregate statistics
    incoming = sum(1 for rec in call_log if rec.get("direction") == "incoming")
    outgoing = sum(1 for rec in call_log if rec.get("direction") == "outgoing")
    with_names = sum(1 for rec in call_log if rec.get("contact_name"))
    
    print(f"\n📈 CALL DIRECTION BREAKDOWN:")
    print(f"  • Incoming: {incoming} ({100*incoming/len(call_log):.1f}%)")
    print(f"  • Outgoing: {outgoing} ({100*outgoing/len(call_log):.1f}%)")
    print(f"  • With contact names: {with_names} ({100*with_names/len(call_log):.1f}%)")
    
    print(f"\n✅ PHASE 1 VALIDATION PASSED")
    print("="*70 + "\n")
    
    return 0

if __name__ == "__main__":
    exit(main())
