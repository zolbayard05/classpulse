import os
import json
import time
import requests
from roboflow import Roboflow
from ultralytics import YOLO

API_KEY = os.environ["ROBOFLOW_API_KEY"]
WORKSPACE = "zolbayar-danzan-zeno"
PROJECT_SLUG = "classpulse-g8l1n"
MODEL_PATH = r"C:\Users\NewTech\Documents\classpulse-yolo\Classpulse-1\runs\detect\runs\classpulse_v1\weights\best.pt"
DOWNLOAD_DIR = r"C:\Users\NewTech\Documents\classpulse-yolo\review_batch"
OUT_JSON = r"C:\Users\NewTech\Documents\classpulse-yolo\review_priority.json"
CONF_THRESHOLD = 0.15  # low, so we see everything the model considers

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT_SLUG)
model = YOLO(MODEL_PATH)

def get_job_image_ids(job_id):
    all_ids = []
    after = None
    while True:
        resp = project.get_annotation_job_images(job_id, limit=100, after=after)
        ids = resp.get("imageIds", [])
        all_ids.extend(ids)
        after = resp.get("nextPageToken")
        if not after or not ids:
            break
    return all_ids

JOB_1 = "qYIvdU7E7e0FircMwBKe"   # room_frames (9/17), 199 items
JOB_2 = "ybRSASEgijtiKRH3f6LA"   # room_frames_v2 (9/21), 961 items

print("Fetching job image lists...", flush=True)
ids_1 = get_job_image_ids(JOB_1)
ids_2 = get_job_image_ids(JOB_2)
print(f"job1={len(ids_1)} job2={len(ids_2)}", flush=True)

all_ids = list(dict.fromkeys(ids_1 + ids_2))  # dedupe, preserve order
print(f"total unique candidate images: {len(all_ids)}", flush=True)

results = []
if os.path.exists(OUT_JSON):
    with open(OUT_JSON) as f:
        results = json.load(f)
done_ids = {r["id"] for r in results}

for i, img_id in enumerate(all_ids, 1):
    if img_id in done_ids:
        continue
    try:
        detail = project.image(img_id)
        ann = detail.get("annotation") or {}
        boxes = ann.get("boxes", [])
        if not boxes:
            continue  # empty ones are a separate task, skip here

        name = detail.get("name")
        url = detail["urls"]["original"]
        local_path = os.path.join(DOWNLOAD_DIR, name)
        if not os.path.exists(local_path):
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(resp.content)

        pred = model.predict(source=local_path, conf=CONF_THRESHOLD, verbose=False)[0]
        pred_confs = [float(b.conf.item()) for b in pred.boxes]
        pred_classes = [pred.names[int(b.cls.item())] for b in pred.boxes]

        stored_labels = [b["label"] for b in boxes]
        has_sleeping = "sleeping" in stored_labels
        min_conf = min(pred_confs) if pred_confs else None
        max_conf = max(pred_confs) if pred_confs else None
        avg_conf = sum(pred_confs) / len(pred_confs) if pred_confs else None
        count_mismatch = abs(len(boxes) - len(pred.boxes))

        entry = {
            "id": img_id,
            "name": name,
            "stored_box_count": len(boxes),
            "stored_labels": stored_labels,
            "has_sleeping": has_sleeping,
            "my_pred_count": len(pred.boxes),
            "my_pred_classes": pred_classes,
            "my_min_conf": min_conf,
            "my_avg_conf": avg_conf,
            "count_mismatch": count_mismatch,
        }
        results.append(entry)

        if i % 25 == 0:
            with open(OUT_JSON, "w") as f:
                json.dump(results, f, indent=1)
            print(f"[{i}/{len(all_ids)}] checkpoint saved ({len(results)} analyzed)", flush=True)

    except Exception as e:
        print(f"[{i}/{len(all_ids)}] FAILED {img_id}: {e}", flush=True)

    time.sleep(0.1)

with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=1)

print(f"\nDone. analyzed={len(results)}", flush=True)
