import os
import csv

# Folders to scan for filenames
base_dir = r"c:\xampp\htdocs\id_photo_validator\data\id_photo_dataset\val"
subfolders = ["high_quality", "low_quality"]

# Output CSV filename
output_csv = "val_filenames.csv"

rows = []
for label in subfolders:
    folder_path = os.path.join(base_dir, label)
    # Collect only files (skip nested dirs, if any)
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            rows.append((label, filename))

with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["quality", "filename"])
    writer.writerows(rows)

print(f"Done! {len(rows)} filenames saved to {output_csv}")
