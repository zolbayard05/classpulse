import os
import random
import shutil
import time
import requests
from roboflow import Roboflow

API_KEY = os.environ["ROBOFLOW_API_KEY"]
WORKSPACE = "zolbayar-danzan-zeno"
PROJECT_SLUG = "classpulse-g8l1n"
NAME_TO_IDX = {"attentive": 0, "distracted": 1, "phone using": 2, "sleeping": 3}

CACHE_DIRS = [
    r"C:\Users\NewTech\Documents\classpulse-yolo\review_batch",
    r"C:\Users\NewTech\Documents\classpulse-yolo\unlabeled_batch",
]
OUT_ROOT = r"C:\Users\NewTech\Documents\classpulse-yolo\CombinedDataset"

random.seed(42)

rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT_SLUG)

for split in ["train", "valid", "test"]:
    os.makedirs(os.path.join(OUT_ROOT, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUT_ROOT, split, "labels"), exist_ok=True)


def find_cached(name):
    for d in CACHE_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


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


JOB_1 = "qYIvdU7E7e0FircMwBKe"  # room_frames (9/17)
JOB_2 = "ybRSASEgijtiKRH3f6LA"  # room_frames_v2 (9/21)

print("Fetching image lists...", flush=True)
ids_job1 = get_job_image_ids(JOB_1)
ids_job2 = get_job_image_ids(JOB_2)

gt = project.search(in_dataset=True, limit=300, fields=["id", "name"])
ids_gt = [g["id"] for g in gt]

all_ids = list(dict.fromkeys(ids_gt + ids_job1 + ids_job2))
print(f"gt={len(ids_gt)} job1={len(ids_job1)} job2={len(ids_job2)} total_unique={len(all_ids)}", flush=True)

# session-stratified split: keep 9/17 and 9/21 both represented in val/test
random.shuffle(all_ids)

written, empty_skipped, failed = 0, 0, 0

for i, img_id in enumerate(all_ids, 1):
    try:
        detail = project.image(img_id)
        name = detail.get("name")
        ann = detail.get("annotation") or {}
        boxes = ann.get("boxes", [])
        w, h = ann.get("width"), ann.get("height")

        if not boxes or not w or not h:
            empty_skipped += 1
            continue

        local_img = find_cached(name)
        if not local_img:
            url = detail["urls"]["original"]
            local_img = os.path.join(OUT_ROOT, "_tmp_" + name)
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            with open(local_img, "wb") as f:
                f.write(resp.content)

        lines = []
        valid = True
        for b in boxes:
            if b["label"] not in NAME_TO_IDX:
                continue
            try:
                cx = float(b["x"]) / float(w)
                cy = float(b["y"]) / float(h)
                bw = float(b["width"]) / float(w)
                bh = float(b["height"]) / float(h)
            except (TypeError, ValueError):
                valid = False
                break
            lines.append(f"{NAME_TO_IDX[b['label']]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if not valid or not lines:
            empty_skipped += 1
            continue

        r = random.random()
        split = "train" if r < 0.85 else ("valid" if r < 0.95 else "test")

        img_out = os.path.join(OUT_ROOT, split, "images", name)
        lbl_out = os.path.join(OUT_ROOT, split, "labels", os.path.splitext(name)[0] + ".txt")
        shutil.copy(local_img, img_out)
        with open(lbl_out, "w") as f:
            f.write("\n".join(lines) + "\n")

        written += 1
        if i % 100 == 0:
            print(f"[{i}/{len(all_ids)}] written={written} empty_skipped={empty_skipped} failed={failed}", flush=True)

    except Exception as e:
        print(f"[{i}/{len(all_ids)}] FAILED {img_id}: {e}", flush=True)
        failed += 1

    time.sleep(0.05)

print(f"\nDone. written={written} empty_skipped={empty_skipped} failed={failed}", flush=True)
