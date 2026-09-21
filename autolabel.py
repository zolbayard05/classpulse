import os
import sys
import time
import requests
from roboflow import Roboflow
from roboflow.util.image_utils import load_labelmap
from ultralytics import YOLO

API_KEY = os.environ["ROBOFLOW_API_KEY"]
WORKSPACE = "zolbayar-danzan-zeno"
PROJECT_SLUG = "classpulse-g8l1n"
MODEL_PATH = r"C:\Users\NewTech\Documents\classpulse-yolo\Classpulse-1\runs\detect\runs\classpulse_v1\weights\best.pt"
DATA_YAML = r"C:\Users\NewTech\Documents\classpulse-yolo\Classpulse-1\data.yaml"
DOWNLOAD_DIR = r"C:\Users\NewTech\Documents\classpulse-yolo\unlabeled_batch"
LABEL_DIR = os.path.join(DOWNLOAD_DIR, "labels")
PROCESSED_LOG = r"C:\Users\NewTech\Documents\classpulse-yolo\processed_ids.txt"
CONF_THRESHOLD = 0.25

os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)

chunk_size = int(sys.argv[1]) if len(sys.argv) > 1 else 30

processed_ids = set()
if os.path.exists(PROCESSED_LOG):
    with open(PROCESSED_LOG) as f:
        processed_ids = set(line.strip() for line in f if line.strip())
print(f"Already processed: {len(processed_ids)}", flush=True)

print("Connecting to Roboflow...", flush=True)
rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT_SLUG)
print("Connected.", flush=True)

print("Fetching unannotated image list...", flush=True)
all_results = project.search(in_dataset=False, limit=300, fields=["id", "name"])
remaining = [r for r in all_results if r["id"] not in processed_ids]
print(f"Total unannotated: {len(all_results)}, remaining to process: {len(remaining)}", flush=True)

batch = remaining[:chunk_size]

LABELMAP = load_labelmap(DATA_YAML)
print("Labelmap:", LABELMAP, flush=True)

print("Loading model...", flush=True)
model = YOLO(MODEL_PATH)
print("Model loaded. Starting batch...", flush=True)

success, failed, skipped = 0, 0, 0
log_f = open(PROCESSED_LOG, "a")

for i, item in enumerate(batch, 1):
    img_id = item["id"]
    img_name = item["name"]
    print(f"[{i}/{len(batch)}] {img_name} ({img_id})", flush=True)

    try:
        detail = project.image(img_id)
        url = detail["urls"]["original"]

        local_img_path = os.path.join(DOWNLOAD_DIR, img_name)
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        with open(local_img_path, "wb") as f:
            f.write(resp.content)

        pred = model.predict(source=local_img_path, conf=CONF_THRESHOLD, verbose=False)[0]

        txt_name = os.path.splitext(img_name)[0] + ".txt"
        txt_path = os.path.join(LABEL_DIR, txt_name)
        with open(txt_path, "w") as f:
            for box in pred.boxes:
                cls_id = int(box.cls.item())
                x, y, w, h = box.xywhn[0].tolist()
                f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

        num_boxes = len(pred.boxes)

        if num_boxes == 0:
            print("    -> 0 boxes detected, skipped (needs manual review)", flush=True)
            skipped += 1
        else:
            project.save_annotation(
                annotation_path=txt_path,
                annotation_labelmap=LABELMAP,
                image_id=img_id,
                is_prediction=True,
                annotation_overwrite=True,
            )
            print(f"    -> {num_boxes} boxes predicted and uploaded as prediction", flush=True)
            success += 1

        log_f.write(img_id + "\n")
        log_f.flush()

    except Exception as e:
        print(f"    !! FAILED: {e}", flush=True)
        failed += 1

    time.sleep(0.2)

log_f.close()
print(f"\nDone. success={success} skipped={skipped} failed={failed}", flush=True)
