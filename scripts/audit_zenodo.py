"""
scripts/audit_zenodo.py
───────────────────────
Comprehensive dataset audit for Zenodo Yoga Landmarks dataset.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "zenodo_yoga_landmarks - zenodo_yoga_landmarks.csv"

if not CSV_PATH.exists():
    print(f"Error: {CSV_PATH} does not exist.")
    sys.exit(1)

print(f"Loading {CSV_PATH.name}...")
df = pd.read_csv(CSV_PATH)
print(f"Shape: {df.shape}")

metadata_cols = ["image_path", "asana_label", "correctness_label", "combined_label", "group_id"]
landmark_cols = [c for c in df.columns if c not in metadata_cols]

print(f"\nTotal columns: {len(df.columns)}")
print(f"Metadata columns: {metadata_cols}")
print(f"Landmark columns: {len(landmark_cols)} ({landmark_cols[0]} to {landmark_cols[-1]})")

# 1. Null / NaN / Inf checks
print("\n--- Missing / Infinite Value Check ---")
meta_nulls = df[metadata_cols].isnull().sum()
print("Metadata nulls:\n", meta_nulls)

lm_nulls = df[landmark_cols].isnull().sum().sum()
print(f"Landmark NaN count: {lm_nulls}")

lm_vals = df[landmark_cols].values
inf_count = np.isinf(lm_vals).sum()
print(f"Landmark Inf count: {inf_count}")

# 2. Check value ranges for landmarks
# MediaPipe outputs: (x, y, z, visibility) for 33 landmarks -> 132 features
print("\n--- Landmark Value Ranges by Channel (assuming 33 landmarks * 4: x, y, z, vis) ---")
for channel_idx, channel_name in enumerate(["x", "y", "z", "visibility"]):
    cols_ch = [landmark_cols[i * 4 + channel_idx] for i in range(33)]
    vals_ch = df[cols_ch].values
    print(f"Channel {channel_name} ({cols_ch[0]}..{cols_ch[-1]}): min={vals_ch.min():.4f}, max={vals_ch.max():.4f}, mean={vals_ch.mean():.4f}, std={vals_ch.std():.4f}")

# 3. Label distribution
print("\n--- Asana Labels ---")
print(df["asana_label"].value_counts())

print("\n--- Correctness Labels ---")
print(df["correctness_label"].value_counts())

print("\n--- Asana x Correctness ---")
print(pd.crosstab(df["asana_label"], df["correctness_label"], margins=True))

# 4. Group ID analysis
print("\n--- Group ID Semantics ---")
print(f"Unique group_ids: {df['group_id'].nunique()}")
print("Top 25 group_ids:")
print(df["group_id"].value_counts().head(25))

# 5. Combined label analysis
print("\n--- Combined Label Semantics ---")
print(f"Unique combined_labels: {df['combined_label'].nunique()}")
print("Top 25 combined_labels:")
print(df["combined_label"].value_counts().head(25))

# 6. Image path inspection
print("\n--- Image Path Inspection ---")
print("Sample image paths:")
for p in df["image_path"].head(10):
    print("  ", p)

# Check if image_path can extract video/session/user identifiers
# e.g., extract folder names or base names
import re
paths = df["image_path"].astype(str)
folders = paths.apply(lambda p: str(Path(p).parent) if "/" in p or "\\" in p else "")
print(f"\nUnique folders from image_path: {folders.nunique()}")
print(folders.value_counts().head(15))

# 7. Exact duplicate rows
print("\n--- Duplicate Rows Check ---")
dup_all = df.duplicated().sum()
dup_lm = df.duplicated(subset=landmark_cols).sum()
print(f"Exact row duplicates (all cols): {dup_all}")
print(f"Duplicate landmark vectors: {dup_lm}")

# Check landmark duplicates with different labels
if dup_lm > 0:
    dups = df[df.duplicated(subset=landmark_cols, keep=False)].sort_values(by="image_path")
    print(f"Number of rows involved in duplicate landmark vectors: {len(dups)}")

# 8. Date and Burst / Sequence Analysis
print("\n--- Date & Burst Sequence Analysis ---")
filenames = df["image_path"].apply(lambda x: Path(x).name)
df["date"] = filenames.apply(lambda fn: fn[:8])
df["burst_min"] = filenames.apply(lambda fn: fn[:13]) # YYYYMMDD_HHMM

print("Cross-tab Asana vs Date:")
print(pd.crosstab(df["asana_label"], df["date"]))

print("\nBurst Counts per Asana and Correctness:")
for asana, grp in df.groupby("asana_label"):
    right_bursts = grp[grp["correctness_label"] == "Right"]["burst_min"].nunique()
    wrong_bursts = grp[grp["correctness_label"] == "Wrong"]["burst_min"].nunique()
    total_bursts = grp["burst_min"].nunique()
    print(f"  {asana:22s} Total={len(grp):4d} | Bursts={total_bursts:2d} (Right: {right_bursts:2d}, Wrong: {wrong_bursts:2d})")

