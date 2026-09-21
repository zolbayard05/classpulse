import os
import re
import time
from roboflow import Roboflow

API_KEY = os.environ["ROBOFLOW_API_KEY"]
WORKSPACE = "zolbayar-danzan-zeno"
PROJECT_SLUG = "classpulse-g8l1n"
LABELMAP = {0: "attentive", 1: "distracted", 2: "phone using", 3: "sleeping"}
BASE_DIR = r"C:\Users\NewTech\Documents\classpulse-yolo\Classpulse-1"

rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT_SLUG)

print("Fetching in_dataset=True images (original ground truth)...", flush=True)
gt_images = project.search(in_dataset=True, limit=300, fields=["id", "name"])
print(f"Found {len(gt_images)} ground-truth images", flush=True)
name_to_id = {img["name"]: img["id"] for img in gt_images}

local_label_files = []
for split in ["train", "valid", "test"]:
    d = os.path.join(BASE_DIR, split, "labels")
    for fn in os.listdir(d):
        if fn.endswith(".txt"):
            local_label_files.append(os.path.join(d, fn))

print(f"Found {len(local_label_files)} local backup label files", flush=True)

success, no_match, failed = 0, 0, 0

for path in local_label_files:
    fname = os.path.basename(path)
    # strip "_jpg.rf.<hash>.txt" -> ".jpg"
    m = re.match(r"^(.*)_jpg\.rf\.[0-9a-f]+\.txt$", fname)
    if not m:
        print(f"  !! could not parse filename: {fname}", flush=True)
        no_match += 1
        continue
    orig_name = m.group(1) + ".jpg"

    img_id = name_to_id.get(orig_name)
    if not img_id:
        print(f"  !! no live match for: {orig_name}", flush=True)
        no_match += 1
        continue

    try:
        project.save_annotation(
            annotation_path=path,
            annotation_labelmap=LABELMAP,
            image_id=img_id,
            is_prediction=False,
            annotation_overwrite=True,
        )
        print(f"  OK restored: {orig_name} -> {img_id}", flush=True)
        success += 1
    except Exception as e:
        print(f"  !! FAILED {orig_name}: {e}", flush=True)
        failed += 1

    time.sleep(0.2)

print(f"\nDone. success={success} no_match={no_match} failed={failed}", flush=True)
