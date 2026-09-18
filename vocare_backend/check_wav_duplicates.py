import os
import hashlib
import pandas as pd
from collections import defaultdict

CSV_PATH = r"..\vocal_features.csv"
WAV_ROOT = r"C:\healthcare_model2\data"

print("Loading CSV...")
df = pd.read_csv(CSV_PATH)

print("Indexing WAV files...")

wav_map = {}

for root, dirs, files in os.walk(WAV_ROOT):
    for file in files:
        if file.lower().endswith(".wav"):
            wav_map[file.lower()] = os.path.join(root, file)

print("WAVs indexed:", len(wav_map))

hash_groups = defaultdict(list)

print("\nHashing labeled WAV files...")

for i, row in df.iterrows():

    filename = os.path.basename(row["file_path"]).lower()
    path = wav_map.get(filename)

    if path is None:
        continue

    sha = hashlib.md5()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            sha.update(chunk)

    hash_groups[sha.hexdigest()].append({
        "patient_id": row["patient_id"],
        "label": row["label"],
        "filename": filename
    })

    if (i + 1) % 2000 == 0:
        print(f"Processed {i + 1}/{len(df)}")

duplicate_groups = [
    group
    for group in hash_groups.values()
    if len(group) > 1
]

duplicate_recordings = sum(
    len(group)
    for group in duplicate_groups
)

cross_patient = []

cross_label = []

for group in duplicate_groups:

    patients = {
        x["patient_id"]
        for x in group
    }

    labels = {
        x["label"]
        for x in group
    }

    if len(patients) > 1:
        cross_patient.append(group)

    if len(labels) > 1:
        cross_label.append(group)

print("\n" + "=" * 60)
print("WAV DUPLICATE ANALYSIS")
print("=" * 60)

print("CSV rows:", len(df))
print("Unique WAV hashes:", len(hash_groups))
print("Duplicate groups:", len(duplicate_groups))
print("Recordings inside duplicate groups:", duplicate_recordings)

print(
    "Duplicate groups spanning multiple patients:",
    len(cross_patient)
)

print(
    "Duplicate groups containing different labels:",
    len(cross_label)
)

print("=" * 60)

if cross_patient:

    print("\nFirst 10 cross-patient duplicate groups:")

    for group in cross_patient[:10]:

        print("\n--- DUPLICATE GROUP ---")

        for item in group:
            print(
                item["patient_id"],
                item["label"],
                item["filename"]
            )