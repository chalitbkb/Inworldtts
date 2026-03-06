#!/usr/bin/env python3
"""
Kaggle Notebook: Download Mozilla Common Voice Thai v24.0
=========================================================
This notebook downloads the Mozilla Common Voice Scripted Speech dataset (Thai)
from the Mozilla Data Collective API, extracts only validated clips, converts
them to WAV (24000Hz mono), and prepares a train.jsonl for InworldTTS training.

Steps:
  1. Download dataset from Mozilla Data Collective API
  2. Extract the tar.gz archive
  3. Filter only validated clips using validated.tsv
  4. Convert MP3 -> WAV (24000Hz, mono) using ffmpeg
  5. Generate train.jsonl in InworldTTS format
  6. Save as Kaggle output for reuse in other notebooks

Usage:
  - Upload this script as a Kaggle Notebook
  - Enable GPU (optional for this step, but needed later for training)
  - Enable Internet Access (required for downloading)
  - Run all cells
"""

import json
import os
import subprocess
import sys

import pandas as pd
import requests

# ==============================================================================
# Configuration
# ==============================================================================

# Mozilla Data Collective API credentials
MDC_API_KEY = "6799035d029e9bcf52cea5e84637baf19f377d387d3ab1f3ef4f4cc0235f5ccf"
MDC_DATASET_ID = "cmj8u3pvx00r9nxxb1yo5z53z"
MDC_API_BASE = "https://datacollective.mozillafoundation.org/api"

# Paths (Kaggle environment)
KAGGLE_WORKING = "/kaggle/working"
DOWNLOAD_DIR = os.path.join(KAGGLE_WORKING, "mcv_thai_raw")
EXTRACT_DIR = os.path.join(KAGGLE_WORKING, "mcv_thai_extracted")
OUTPUT_WAV_DIR = os.path.join(KAGGLE_WORKING, "dataset", "th_wavs")
OUTPUT_JSONL = os.path.join(KAGGLE_WORKING, "dataset", "train.jsonl")

# Audio settings
TARGET_SAMPLE_RATE = 24000
TARGET_CHANNELS = 1  # mono

# Limit for prototyping (set to None for all validated clips)
# Recommended: start with 5000-10000 clips for initial testing
MAX_CLIPS = 10000

# ==============================================================================
# Step 1: Download dataset from Mozilla Data Collective
# ==============================================================================

print("=" * 60)
print("Step 1: Downloading Mozilla Common Voice Thai v24.0")
print("=" * 60)

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Request download URL from API
print("Requesting download URL from Mozilla Data Collective API...")
response = requests.post(
    f"{MDC_API_BASE}/datasets/{MDC_DATASET_ID}/download",
    headers={
        "Authorization": f"Bearer {MDC_API_KEY}",
        "Content-Type": "application/json",
    },
)

if response.status_code != 200:
    print(f"Error: API returned status {response.status_code}")
    print(f"Response: {response.text}")
    sys.exit(1)

data = response.json()
download_url = data.get("downloadUrl") or data.get("download_url") or data.get("url")

if not download_url:
    print(f"Error: Could not find download URL in response: {data}")
    sys.exit(1)

print(f"Download URL obtained successfully!")

# Download the tar.gz file
tar_path = os.path.join(DOWNLOAD_DIR, "cv_thai_v24.tar.gz")

if os.path.exists(tar_path):
    print(f"File already exists: {tar_path}, skipping download.")
else:
    print(f"Downloading dataset (~8GB). This may take 5-15 minutes on Kaggle...")
    subprocess.run(
        ["wget", "-O", tar_path, "--progress=dot:giga", download_url],
        check=True,
    )
    print(f"Download complete: {tar_path}")

# ==============================================================================
# Step 2: Extract the archive
# ==============================================================================

print("\n" + "=" * 60)
print("Step 2: Extracting archive")
print("=" * 60)

os.makedirs(EXTRACT_DIR, exist_ok=True)

# Check if already extracted
tsv_candidates = []
for root, dirs, files in os.walk(EXTRACT_DIR):
    for f in files:
        if f == "validated.tsv":
            tsv_candidates.append(os.path.join(root, f))

if tsv_candidates:
    print(f"Already extracted. Found validated.tsv at: {tsv_candidates[0]}")
else:
    print("Extracting tar.gz (this may take a few minutes)...")
    subprocess.run(
        ["tar", "-xzf", tar_path, "-C", EXTRACT_DIR],
        check=True,
    )
    print("Extraction complete!")

    # Re-scan for validated.tsv
    for root, dirs, files in os.walk(EXTRACT_DIR):
        for f in files:
            if f == "validated.tsv":
                tsv_candidates.append(os.path.join(root, f))
                
    # Free up space by deleting the 8GB tar archive immediately
    if os.path.exists(tar_path):
        print("Freeing up disk space: Deleting tar.gz archive...")
        os.remove(tar_path)

if not tsv_candidates:
    print("Error: Could not find validated.tsv after extraction!")
    print("Directory contents:")
    for root, dirs, files in os.walk(EXTRACT_DIR):
        for f in files[:20]:
            print(f"  {os.path.join(root, f)}")
    sys.exit(1)

validated_tsv_path = tsv_candidates[0]
cv_base_dir = os.path.dirname(validated_tsv_path)
clips_dir = os.path.join(cv_base_dir, "clips")

print(f"Dataset directory: {cv_base_dir}")
print(f"Clips directory: {clips_dir}")

# ==============================================================================
# Step 3: Filter validated clips
# ==============================================================================

print("\n" + "=" * 60)
print("Step 3: Reading validated clips")
print("=" * 60)

df = pd.read_csv(validated_tsv_path, sep="\t")
print(f"Total validated clips: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"\nSample data:")
print(df.head(3).to_string())

# Apply limit for prototyping
if MAX_CLIPS and MAX_CLIPS < len(df):
    print(f"\nLimiting to {MAX_CLIPS} clips for prototyping.")
    df = df.head(MAX_CLIPS)

# ==============================================================================
# Step 4: Convert MP3 -> WAV (24000Hz, mono) and cleanup MP3s
# ==============================================================================

print("\n" + "=" * 60)
print(f"Step 4: Converting {len(df)} MP3 files to WAV ({TARGET_SAMPLE_RATE}Hz)")
print("=" * 60)
print("Note: Original MP3 files will be deleted immediately after conversion to save disk space.")

os.makedirs(OUTPUT_WAV_DIR, exist_ok=True)

successful = 0
failed = 0
jsonl_lines = []

for idx, row in df.iterrows():
    # Determine MP3 filename
    mp3_filename = row.get("path", row.get("filename", ""))
    if not mp3_filename:
        failed += 1
        continue

    mp3_path = os.path.join(clips_dir, mp3_filename)
    wav_filename = mp3_filename.replace(".mp3", ".wav")
    wav_path = os.path.join(OUTPUT_WAV_DIR, wav_filename)

    # Skip if already converted
    if os.path.exists(wav_path):
        successful += 1
        jsonl_lines.append(
            json.dumps(
                {
                    "transcript": str(row["sentence"]),
                    "language": "th",
                    "wav_path": wav_path,
                    "duration": 5.0, # Approximate/fallback for skipped ones
                    "sample_rate": TARGET_SAMPLE_RATE
                },
                ensure_ascii=False,
            )
        )
        # Still try to delete mp3 if it somehow exists to save space
        if os.path.exists(mp3_path):
            os.remove(mp3_path)
        continue

    # Prepare to extract duration with ffprobe if conversion succeeds
    
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", mp3_path,
            "-ar", str(TARGET_SAMPLE_RATE),
            "-ac", str(TARGET_CHANNELS),
            wav_path,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if result.returncode == 0:
        # Get exact duration using ffprobe
        probe_result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", wav_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        try:
            duration = float(probe_result.stdout.strip())
        except ValueError:
            duration = 0.0 # fallback

        # Filter out clips over 30 seconds per README spec
        if duration > 30.0:
            failed += 1
            if os.path.exists(wav_path):
                os.remove(wav_path)
            if os.path.exists(mp3_path):
                os.remove(mp3_path)
            continue

        successful += 1
        jsonl_lines.append(
            json.dumps(
                {
                    "transcript": str(row["sentence"]),
                    "language": "th",
                    "wav_path": wav_path,
                    "duration": round(duration, 2),
                    "sample_rate": TARGET_SAMPLE_RATE
                },
                ensure_ascii=False,
            )
        )
        # Delete MP3 after successful conversion to free up Kaggle disk space
        os.remove(mp3_path)
    else:
        failed += 1

    # Progress update every 500 files
    if (idx + 1) % 500 == 0:
        print(f"  Progress: {idx + 1}/{len(df)} | OK: {successful} | Failed: {failed}")

print(f"\nConversion complete!")
print(f"  Successful: {successful}")
print(f"  Failed: {failed}")

# ==============================================================================
# Step 5: Generate train.jsonl
# ==============================================================================

print("\n" + "=" * 60)
print("Step 5: Generating train.jsonl for InworldTTS")
print("=" * 60)

os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)

with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
    f.write("\n".join(jsonl_lines))

print(f"Written {len(jsonl_lines)} entries to: {OUTPUT_JSONL}")

# ==============================================================================
# Summary
# ==============================================================================

print("\n" + "=" * 60)
print("✅ COMPLETE! Dataset is ready for InworldTTS training.")
print("=" * 60)
print(f"  WAV files:   {OUTPUT_WAV_DIR}")
print(f"  JSONL file:  {OUTPUT_JSONL}")
print(f"  Total clips: {len(jsonl_lines)}")
print(f"  Sample rate: {TARGET_SAMPLE_RATE}Hz")
print(f"  Channels:    mono")
print()
print("Next steps:")
print("  1. Clone InworldTTS repo in a new Kaggle Notebook")
print("  2. Add this notebook's output as a dataset")
print("  3. Run data vectorization: python tools/data/data_vectorization.py")
print("  4. Start training: python tts/training/sft/sft.py")
