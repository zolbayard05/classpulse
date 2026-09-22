"""
Classpulse - Camo camera дээр real-time тест хийх script (macOS)

Урьдчилсан бэлтгэл:
  1. Camo Studio-г Mac дээрээ суулгаад, утсаа холбож идэвхжүүл
  2. pip3 install ultralytics opencv-python
  3. best.pt загварын файлыг энэ script-тэй ижил фолдерт хий

Ажиллуулах:
  python3 live_camo_test.py --camera 0
  (Camo камер олдохгүй бол --camera 1, --camera 2 гэх мэтээр туршиж үз)

Гарах: q товч дар
"""

import argparse
import cv2
from ultralytics import YOLO

CLASS_COLORS = {
    "attentive": (206, 255, 0),      # cyan-ish (BGR)
    "distracted": (235, 183, 0),     # blue-ish
    "phone using": (0, 128, 255),    # orange
    "sleeping": (86, 0, 254),        # pink/red
}
CLASS_ORDER = ["attentive", "distracted", "phone using", "sleeping"]

BOX_THICKNESS = 1
LABEL_ALPHA = 0.55       # label арын өнгөний тунгалагшил (0=бүрэн харагдахгүй, 1=бүрэн шаргал)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_THICKNESS = 1
CROSS_CLASS_IOU_THRESH = 0.5   # үүнээс дээш давхцвал зөвхөн хамгийн итгэлтэй class-ыг үлдээнэ
STANDING_ASPECT_RATIO = 1.4    # box-ийн өндөр/өргөн харьцаа үүнээс дээш бол "зогссон" гэж үзнэ
SMOOTH_IOU_THRESH = 0.3        # frame хооронд ижил хүн гэж тооцох хамгийн бага IoU
SMOOTH_ALPHA = 0.3             # EMA smoothing коэффициент (бага байх тусам илүү гөлгөр, удаан хариу үйлдэл)
SMOOTH_MAX_AGE = 10            # frame-ээр илрээгүй track-ыг хэдэн frame-ийн дараа устгах


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def suppress_cross_class_overlaps(dets, iou_thresh=CROSS_CLASS_IOU_THRESH):
    """Нэг хүн дээр өөр class-ийн box давхцвал зөвхөн хамгийн confidence өндөртэйг нь үлдээнэ.
    dets: [(x1,y1,x2,y2,conf,cls_name), ...]
    """
    dets = sorted(dets, key=lambda d: d[4], reverse=True)
    kept = []
    for d in dets:
        box_d = d[:4]
        suppressed = False
        for k in kept:
            if d[5] != k[5] and iou(box_d, k[:4]) > iou_thresh:
                suppressed = True
                break
        if not suppressed:
            kept.append(d)
    return kept


def apply_standing_rule(dets, ratio_thresh=STANDING_ASPECT_RATIO):
    """Sleeping-ээс бусад хүний box өндөр/нарийн (зогссон/явж буй) бол distracted болгоно.
    Зөвхөн тухайн хүний ӨӨРИЙН box-ийн хэлбэрт л тулгуурладаг тул хажуугийн хүнээс
    нөлөөлдөггүй (гинжин false positive үүсгэхгүй)."""
    out = []
    for x1, y1, x2, y2, conf, cls_name in dets:
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        if cls_name != "sleeping" and (h / w) > ratio_thresh:
            cls_name = "distracted"
        out.append((x1, y1, x2, y2, conf, cls_name))
    return out


class ConfidenceSmoother:
    """Frame хоорондоо box-уудыг IoU-аар тааруулж, confidence-ийг EMA-аар тэгшилнэ.
    Ингэснээр нэг хүний тоо frame бүрт үсрэлт хийхгүй, гөлгөр өөрчлөгдөнө."""

    def __init__(self, iou_thresh=SMOOTH_IOU_THRESH, alpha=SMOOTH_ALPHA, max_age=SMOOTH_MAX_AGE):
        self.iou_thresh = iou_thresh
        self.alpha = alpha
        self.max_age = max_age
        self.tracks = []  # {box, cls_name, smoothed_conf, age}

    def update(self, dets):
        used_tracks = set()
        out = []

        for x1, y1, x2, y2, conf, cls_name in dets:
            best_iou, best_i = 0, -1
            for i, t in enumerate(self.tracks):
                if i in used_tracks or t["cls_name"] != cls_name:
                    continue
                v = iou((x1, y1, x2, y2), t["box"])
                if v > best_iou:
                    best_iou, best_i = v, i

            if best_iou > self.iou_thresh:
                t = self.tracks[best_i]
                t["smoothed_conf"] = self.alpha * conf + (1 - self.alpha) * t["smoothed_conf"]
                t["box"] = (x1, y1, x2, y2)
                t["age"] = 0
                used_tracks.add(best_i)
                out.append((x1, y1, x2, y2, t["smoothed_conf"], cls_name))
            else:
                new_track = {"box": (x1, y1, x2, y2), "cls_name": cls_name, "smoothed_conf": conf, "age": 0}
                self.tracks.append(new_track)
                used_tracks.add(len(self.tracks) - 1)
                out.append((x1, y1, x2, y2, conf, cls_name))

        # хайгдаагүй track-уудыг хөгшрүүлж, хэт хуучирсныг хасна
        alive = []
        for i, t in enumerate(self.tracks):
            if i not in used_tracks:
                t["age"] += 1
                if t["age"] > self.max_age:
                    continue
            alive.append(t)
        self.tracks = alive

        return out


def rects_overlap(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def place_label(label_rects, x1, y1, tw, th, frame_h):
    """y1 дээрх label-ийн байрлалыг өмнөх label-үүдтэй давхцахгүй болтол дээш/доош шилжүүлнэ."""
    pad = 4
    top = y1 - th - 2 * pad
    bottom = y1
    step = th + 2 * pad + 2

    for _ in range(40):
        rect = (x1, top, x1 + tw + 2 * pad, bottom)
        if not any(rects_overlap(rect, r) for r in label_rects):
            label_rects.append(rect)
            return rect
        top -= step
        bottom -= step
        if top < 0:
            # box-ийн дотор дээд хэсэгт нь байрлуулах руу шилжинэ
            top = y1 + 2
            bottom = y1 + th + 2 * pad
            rect = (x1, top, x1 + tw + 2 * pad, bottom)
            label_rects.append(rect)
            return rect

    label_rects.append(rect)
    return rect


def draw_summary_hud(frame, counts):
    x0, y0 = 12, 12
    row_h = 24
    panel_w = 190
    panel_h = row_h * len(CLASS_ORDER) + 16

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for i, cls_name in enumerate(CLASS_ORDER):
        cy = y0 + 16 + i * row_h
        color = CLASS_COLORS[cls_name]
        cv2.circle(frame, (x0 + 14, cy), 6, color, -1)
        text = f"{cls_name}: {counts.get(cls_name, 0)}"
        cv2.putText(frame, text, (x0 + 28, cy + 5), FONT, FONT_SCALE, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (Camo эсвэл built-in)")
    parser.add_argument("--model", type=str, default="best.pt", help="YOLO model weights path")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--show-conf", action="store_true", help="Label дээр confidence тоог харуул (анхдагчаар нуугдсан)")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    smoother = ConfidenceSmoother()

    print(f"Opening camera index {args.camera} ...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: camera index {args.camera} нээгдсэнгүй. --camera 1, 2 гэх мэтээр туршиж үз.")
        return

    print("Гарахын тулд 'q' дар.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame уншиж чадсангүй, гарч байна.")
            break

        results = model.predict(source=frame, conf=args.conf, verbose=False)[0]
        h = frame.shape[0]

        raw_dets = []
        for box in results.boxes:
            cls_id = int(box.cls.item())
            cls_name = results.names[cls_id]
            conf = float(box.conf.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            raw_dets.append((x1, y1, x2, y2, conf, cls_name))

        # нэг хүн дээр өөр class-ийн box давхцвал хамгийн итгэлтэйг нь л үлдээнэ
        raw_dets = suppress_cross_class_overlaps(raw_dets)

        # зогссон/явж буй хүнийг distracted болгоно (sleeping-д хэрэглэхгүй)
        raw_dets = apply_standing_rule(raw_dets)

        # frame хоорондын confidence-ийг гөлгөр болгоно (тогтворгүй үсрэлтийг арилгана)
        raw_dets = smoother.update(raw_dets)

        # box-уудыг дээрээс доош эрэмбэлж, label collision-ийг тогтвортой болгоно
        raw_dets.sort(key=lambda d: d[1])

        counts = {c: 0 for c in CLASS_ORDER}
        label_rects = []
        drawn = []  # (color, x1,y1,x2,y2, label, lx1,ly1,lx2,ly2)

        # 1-р эгнээ: геометрийг тооцоолох (юу ч зураагүй)
        for x1, y1, x2, y2, conf, cls_name in raw_dets:
            color = CLASS_COLORS.get(cls_name, (255, 255, 255))
            counts[cls_name] = counts.get(cls_name, 0) + 1

            label = f"{cls_name} {conf:.2f}" if args.show_conf else cls_name
            (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
            lx1, ly1, lx2, ly2 = place_label(label_rects, x1, y1, tw, th, h)

            drawn.append((color, x1, y1, x2, y2, label, lx1, ly1, lx2, ly2))

        # 2-р эгнээ: зөвхөн label-ийн арын өнгийг тунгалаг байдлаар blend хийнэ
        overlay = frame.copy()
        for color, x1, y1, x2, y2, label, lx1, ly1, lx2, ly2 in drawn:
            cv2.rectangle(overlay, (lx1, ly1), (lx2, ly2), color, -1)
        cv2.addWeighted(overlay, LABEL_ALPHA, frame, 1 - LABEL_ALPHA, 0, frame)

        # 3-р эгнээ: box зураас болон текстийг бүрэн тодоор дээр нь зурна
        for color, x1, y1, x2, y2, label, lx1, ly1, lx2, ly2 in drawn:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)
            cv2.putText(frame, label, (lx1 + 4, ly2 - 6), FONT, FONT_SCALE, (0, 0, 0), FONT_THICKNESS, cv2.LINE_AA)

        draw_summary_hud(frame, counts)

        cv2.imshow("Classpulse - Camo Live Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
