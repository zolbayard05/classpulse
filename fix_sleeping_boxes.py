import os
import json
import time
from roboflow import Roboflow

API_KEY = os.environ["ROBOFLOW_API_KEY"]
WORKSPACE = "zolbayar-danzan-zeno"
PROJECT_SLUG = "classpulse-g8l1n"
LABELMAP = {0: "attentive", 1: "distracted", 2: "phone using", 3: "sleeping"}
NAME_TO_IDX = {v: k for k, v in LABELMAP.items()}

rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT_SLUG)

targets = json.load(open(r"C:\Users\NewTech\Documents\classpulse-yolo\sleeping_fix_targets.json"))
print(f"Fixing {len(targets)} images", flush=True)

fixed, skipped, failed = 0, 0, 0

for i, t in enumerate(targets, 1):
    img_id = t["id"]
    try:
        detail = project.image(img_id)
        ann = detail.get("annotation") or {}
        boxes = ann.get("boxes", [])
        w, h = ann.get("width"), ann.get("height")

        kept = [b for b in boxes if b["label"] != "sleeping"]
        if len(kept) == len(boxes):
            skipped += 1
            continue

        lines = []
        for b in kept:
            cls_idx = NAME_TO_IDX[b["label"]]
            cx = (b["x"]) / w
            cy = (b["y"]) / h
            bw = b["width"] / w
            bh = b["height"] / h
            lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        yolo_txt = "\n".join(lines) + ("\n" if lines else "")

        if lines:
            project.save_annotation(
                annotation_path={"name": f"{img_id}.txt", "rawText": yolo_txt},
                annotation_labelmap=LABELMAP,
                image_id=img_id,
                is_prediction=False,
                annotation_overwrite=True,
            )
        fixed += 1
        if i % 25 == 0:
            print(f"[{i}/{len(targets)}] fixed={fixed} skipped={skipped} failed={failed}", flush=True)

    except Exception as e:
        print(f"[{i}/{len(targets)}] FAILED {img_id}: {e}", flush=True)
        failed += 1

    time.sleep(0.1)

print(f"\nDone. fixed={fixed} skipped={skipped} failed={failed}", flush=True)
