# Dataset Sources

All links verified via Kaggle/GitHub/Roboflow Universe research (2026-08-25). None of these
alone is production-scale — the plan is to merge 2-4 sources per class and rely on
augmentation. Classes with no usable dataset (streetlight state, encroachment) get a
workaround instead of a dataset.

## 1. Garbage / litter piles
- TACO — https://universe.roboflow.com/mohamed-traore-2ekkp/taco-trash-annotations-in-context (1,499 img, YOLO, CC BY 4.0)
- Trash Detection — https://universe.roboflow.com/yolo-hfplh/trash-detection-1fjjc-nd1dg (2,783 img, YOLO)
- YOLOv8 Trash Detection EE4016 — https://universe.roboflow.com/yolov8-trash-detection/yolo-v8-trash-detection-ee4016 (2,527 img, YOLO)
- Kaggle Domestic Trash/Garbage — https://www.kaggle.com/datasets/dataclusterlabs/domestic-trash-garbage-dataset (~9,000 img, classification, needs bbox relabel)
- Master index of 20+ litter datasets: https://github.com/AgaMiko/waste-datasets-review
- Reference repo (YOLOv8 + TACO, trained weights): https://github.com/jeremy-rico/litter-detection
- Note: most items are per-object (bottle/can), not "pile > 1m". Filter/relabel toward
  dense clusters, or treat clusters of nearby detections as one "pile" incident in post-processing.

## 2. Road potholes
- Pothole Detection (Intel Unnati) — https://universe.roboflow.com/intel-unnati-training-program/pothole-detection-bqu6s (2,475 img, YOLO)
- Kaggle Annotated Potholes (Chitholian) — https://www.kaggle.com/datasets/chitholian/annotated-potholes-dataset (665 img, VOC/YOLO)
- Kaggle Road Damage (potholes+cracks+manholes) — https://www.kaggle.com/datasets/lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes (2,009 img, YOLO, 3 classes)
- Roboflow public Pothole set — https://public.roboflow.com/object-detection/pothole
- Reference (exports to TFLite): https://github.com/gbadrain/Pothole-Image-Segmentation-with-YOLOv8
- Synthetic augmentation technique for scarce damage class: PG-GAN pothole patch + Poisson
  blending onto clean road images (arxiv.org/pdf/2005.08628)

## 3. Open/broken manholes
- Open-Manholes (segmentation) — https://universe.roboflow.com/aibased-solution-for-realtime-detection-of-road-anomalies-d6eay/open-manholes (677 img)
- Manhole Cover Dataset YOLO (Broken/Good/Loose/Uncovered) — https://universe.roboflow.com/create-dataset-for-yolo/manhole-cover-dataset-yolo (1,427 img, YOLO, 4 classes)
- Pothole and Manhole Detection (combined) — https://universe.roboflow.com/project-auzdn/pothole-and-manhole-detection (2,482 img)
- Reference architecture (MGB-YOLO = YOLOv5s+MobileNetV3, embedded-deployed, 96.6% acc):
  https://www.nature.com/articles/s41598-023-43173-z
- Smallest/scarcest class — expect to merge all sources and augment heavily.

## 4. Non-working streetlights — NO end-to-end dataset exists
- Street Light detection (localization only, no lit/unlit label) —
  https://universe.roboflow.com/sashank-s/street-light (1,232 img, YOLO)
- LA Bureau of Street Lighting outage records (tabular, not imagery) —
  https://www.kaggle.com/datasets/cityofLA/los-angeles-bureau-of-street-lighting-data
- **Workaround (used in this project, see streetlight/):** train the unified detector to
  localize the lamp head only, then classify lit/unlit with a brightness/HSV heuristic on
  the cropped ROI at night. Self-collect a small verification set from real footage over time.

## 5. Encroachment — no mature single dataset, merge 2-3 partial sources
- StreetVendor-SLI (Zenodo, CC BY) — https://doi.org/10.5281/zenodo.14635548 (2,794 img, 3 vendor classes, YOLO)
- Illegal Parking — https://universe.roboflow.com/parking-amu50/illegal-parking (56 img — seed only)
- FootpathVision (binary encroached/clear) — referenced in a 2026 arXiv benchmarking paper, verify host before use.
- **Workaround for immediate coverage without training:** zero-shot with YOLO-World or
  Grounding DINO prompted with "vendor cart", "parked vehicle on sidewalk", "construction
  debris on road"; use its high-confidence outputs as pseudo-labels to bootstrap a
  supervised YOLO11n encroachment class.

## 6. Billboard & hoarding
- Billboard Object Detection — https://universe.roboflow.com/arslan-ongr8/billboard-xlvz1 (YOLO, CC BY 4.0)
- Additional Roboflow Universe projects: search `class:billboard`, `class:hoarding`
- Reference notebook: https://github.com/Tessellate-Imaging/Monk_Object_Detection/blob/master/application_model_zoo/Example%20-%20Billboard%20(Hoarding%20detection).ipynb
- Compliance/size checking has no ground-truth dataset — add a second stage that measures
  the billboard bbox against a reference object (pole/vehicle/door) for relative scale.

## 7. Cattle on road
- COCO 2017 already has a `cow` class (id 21 / index 19), CC BY 4.0 — usable zero-shot with
  any COCO-pretrained detector, just filter the class.
- Open Images V7 has separate `Cattle` and `Cow` classes, larger coverage, CC BY 4.0 (check
  per-image for CC BY-SA).
- Cow Indian (domain-matched, stray cattle on Indian roads) —
  https://universe.roboflow.com/sakshi-kumari-70uxi/cow-indian (449 img, YOLO, CC BY 4.0)
- Additional small Roboflow sets: search `class:cattle` on Roboflow Universe.
- **Recommendation:** start from COCO/Open-Images pretrained weights (cow class already
  present), fine-tune on Cow Indian + similar sets for domain adaptation to street-camera
  footage and to merge cow into the unified 7-class model.
