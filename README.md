# Classpulse

Ангийн camera-с ирсэн зургийг **attentive / distracted / phone using / sleeping** гэж ангилдаг YOLOv8 object detection систем.

## Бүтэц

- `best.pt` — сүүлд сургасан YOLOv8s загвар (1022 зураг, 9/17 + 9/21 session-ууд хослуулсан)
- `live_camo_test.py` — Camo camera (эсвэл дурын webcam) дээр real-time тест хийх script
- `download_dataset.py` — Roboflow-с анхны version татах
- `autolabel.py` — Roboflow дээрх label хийгдээгүй зурган дээр загвараар автомат annotate хийх
- `restore_ground_truth.py`, `fix_predictions.py`, `fix_sleeping_boxes.py` — Roboflow дээрх annotation засварын script-үүд
- `build_local_dataset.py` — Roboflow-с бүх annotated зургийг татаж, локал YOLO dataset угсрах
- `review_priority.py` — Загварын өөрийн таамаглалтай харьцуулж, эргэлзээтэй/буруу annotation-г олох

## Тохиргоо

```bash
pip install ultralytics opencv-python roboflow requests
export ROBOFLOW_API_KEY="таны_api_key"   # Roboflow script-үүдэд шаардлагатай
```

## Real-time тест (Camo camera)

```bash
python3 live_camo_test.py --camera 0
```

- `--camera N` — камерын device index (Camo олдохгүй бол 0,1,2... туршиж үз)
- `--conf 0.35` — confidence threshold
- `--hide-conf` — label дээр confidence тоо бүү харуул
- Гарах: `q`

## Загварын гүйцэтгэл (test set, 46 зураг)

| Class | mAP50 | mAP50-95 |
|---|---|---|
| attentive | 0.931 | 0.726 |
| distracted | 0.702 | 0.523 |
| phone using | 0.934 | 0.831 |
| **overall** | **0.856** | **0.693** |

**Мэдэгдэж буй сул тал:** distracted class хамгийн сул; sleeping class-ийн жинхэнэ найдвартай байдал хараахан бүрэн баталгаажаагүй (AUTOLABEL-ийн 709 буруу sleeping label-ийг цэвэрлэсний дараа дата цөөрсөн).
