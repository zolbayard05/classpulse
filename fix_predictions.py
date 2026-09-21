import os
import time
from roboflow import Roboflow

API_KEY = os.environ["ROBOFLOW_API_KEY"]
WORKSPACE = "zolbayar-danzan-zeno"
PROJECT_SLUG = "classpulse-g8l1n"
LABELMAP = {0: "attentive", 1: "distracted", 2: "phone using", 3: "sleeping"}
LABEL_DIR = r"C:\Users\NewTech\Documents\classpulse-yolo\unlabeled_batch\labels"
MAP_FILE = r"C:\Users\NewTech\Documents\classpulse-yolo\id_name_map.txt"

rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT_SLUG)

pairs = []
with open(MAP_FILE) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        img_id, name = line.split(" ", 1)
        pairs.append((img_id, name))

print(f"Loaded {len(pairs)} id/name pairs", flush=True)

success, skipped, failed = 0, 0, 0

for i, (img_id, name) in enumerate(pairs, 1):
    txt_name = os.path.splitext(name)[0] + ".txt"
    txt_path = os.path.join(LABEL_DIR, txt_name)

    if not os.path.exists(txt_path) or os.path.getsize(txt_path) == 0:
        skipped += 1
        continue

    try:
        project.save_annotation(
            annotation_path=txt_path,
            annotation_labelmap=LABELMAP,
            image_id=img_id,
            is_prediction=True,
            annotation_overwrite=True,
        )
        print(f"[{i}/{len(pairs)}] OK fixed: {name} -> {img_id}", flush=True)
        success += 1
    except Exception as e:
        print(f"[{i}/{len(pairs)}] FAILED {name}: {e}", flush=True)
        failed += 1

    time.sleep(0.15)

print(f"\nDone. success={success} skipped={skipped} failed={failed}", flush=True)
