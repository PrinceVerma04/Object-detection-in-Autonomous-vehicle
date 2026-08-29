# Civic Services Surveillance & Monitoring

Real-time detection of 7 civic issues from street camera / mobile-phone footage:
garbage piles, potholes, open/broken manholes, non-working streetlights, encroachments,
billboard/hoarding, and cattle on roads — deployable on-device (Android/iOS).

## Architecture decision

One shared detector, not seven separate models. Running seven models per frame on a phone
is impractical; instead a single **YOLO11n** (Ultralytics, ~5-6MB) is fine-tuned on a
7-class unified dataset. YOLO11n was chosen over YOLOv8n/YOLOv10n/YOLO-NAS/MobileNet-SSD/
EfficientDet-Lite/NanoDet after comparing size, mobile CPU/NPU speed, and export tooling —
see the comparison table below. It already starts from COCO weights, which include a `cow`
class, giving cattle detection a head start.

Streetlight state (lit/unlit) has no labeled dataset anywhere, so it's handled as two
stages instead of a class the detector predicts directly: the shared model only localizes
`streetlight_pole`, and `streetlight/classify_lit_state.py` classifies the crop as lit/unlit
by brightness at night. Swap that heuristic for a trained classifier once enough verified
field crops exist.

Encroachment similarly has no mature single dataset — start from the merged partial sources
in DATASETS.md, and optionally bootstrap more via zero-shot YOLO-World/Grounding DINO
pseudo-labeling on your own footage before hand-verifying.

## Project layout

```
configs/data.yaml        7-class schema shared by every dataset/model script
data/download_datasets.py  pulls Roboflow-hosted sources into data/raw/<key>/
data/merge_datasets.py     remaps every source's classes into the unified 7-class set
train.py                   fine-tunes YOLO11n on data/unified/
export.py                  exports to TFLite (int8, Android) / CoreML (iOS) / ONNX
streetlight/classify_lit_state.py   lit/unlit brightness heuristic
reporting/report.py        writes an incident report (crop + JSON) per confirmed detection
reporting/geocode.py       optional Google Maps reverse-geocoding for report addresses
live_monitor.py            live detection loop — webcam, video file, or phone/IP camera
reports/                   incident reports land here (image + JSON per incident)
DATASETS.md                 every dataset source used, with links, sizes, licenses, caveats
```

## Setup

```
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

This was verified end-to-end in a `.venv` in this repo (dependencies install cleanly,
`live_monitor.py` runs a full read-detect-report loop without errors).

To build the training data:
```
export ROBOFLOW_API_KEY=xxxx   # https://app.roboflow.com/settings/api
python data/download_datasets.py
```
Then manually fetch the Kaggle sources (`kaggle datasets download -d <slug>`) and the
Zenodo StreetVendor-SLI set listed in DATASETS.md, drop each into
`data/raw/<key>/{train,valid}/{images,labels}` in Roboflow's YOLO export layout, and run:
```
python data/merge_datasets.py
python train.py --epochs 100
python export.py --weights runs/civic_services/weights/best.pt
```

## Live monitoring — connect a phone or any camera

`live_monitor.py` reads from a webcam, a video file, or a phone/IP camera stream, runs
the detector, applies the streetlight lit/unlit check at night, and writes an incident
report (`reports/<timestamp>_<class>_<camera>.jpg` + `.json`) the first time a class is
seen, then again after a 60s per-class cooldown.

**Connect an Android phone as the camera:**
1. Install the **IP Webcam** app (Google Play), start the server — it shows a URL like
   `http://192.168.1.23:8080`. Phone and computer must be on the same Wi-Fi.
2. `python live_monitor.py --source http://192.168.1.23:8080/video --camera-id phone-1`

**Connect an iPhone:** install an app like **IP Camera Lite**, same idea — it gives you
an MJPEG/RTSP URL to pass as `--source`.

**Laptop/desktop webcam:** `python live_monitor.py --source 0`

**A recorded video:** `python live_monitor.py --source path/to/video.mp4 --no-display`

Add `--weights runs/civic_services/weights/best.pt` once you've trained on the merged
dataset for full 7-class coverage; without it, `live_monitor.py` defaults to the stock
COCO-pretrained `yolo11n.pt`, which only overlaps our classes on `cow` → reported as
`cattle`. Add `--lat`/`--lon` (the camera's fixed location) plus `GOOGLE_MAPS_API_KEY`
in the environment to get a human-readable address in each report instead of raw
coordinates — see the Google Maps discussion above for why that API is used only for
report labeling, not detection.

Once you have a trained model, exported to TFLite/CoreML, `live_monitor.py`'s detect →
post-process → report logic is exactly what to port into the Android (TFLite Task
Library / MediaPipe) or iOS (CoreML + Vision) app — same model, same logic, different
runtime API.

## Native Android app (runs fully on-device, no server)

`android_app/` is a real CameraX + TFLite Android Studio project — no phone-as-IP-camera
step needed, the model runs directly on the phone. It already builds successfully on this
machine (JDK 17 + Android SDK toolchain, see `android_app/README.md` for the exact setup)
and produces `android_app/app/build/outputs/apk/debug/app-debug.apk`. See that README for
how to install it on a Samsung S24 (or any Android phone) over USB via `adb`/`gradlew`.

## Pretrained model comparison (edge/mobile, 2025-2026)

| Model | Size | Mobile speed | Export to TFLite/CoreML | Notes |
|---|---|---|---|---|
| **YOLO11n (chosen)** | ~5-6MB | fastest in family on CPU/NPU | 1-command, official | COCO-pretrained, has `cow` |
| YOLOv8n | ~6MB | slightly slower than v11n | 1-command, most mature docs | fine if v11n unavailable |
| YOLOv10n | ~5.5MB | NMS-free, lowest post-proc latency | via Ultralytics | good alt if NMS is a bottleneck |
| YOLO26n | smaller, NMS/DFL-free | up to 43% faster CPU than v11n | cleanest export path | newest, worth trying if available |
| YOLO-NAS | 5-19MB | good | ONNX easy, TFLite/CoreML need extra steps | less turnkey |
| MobileNet-SSD/SSDLite | ~4.2MB int8 | very fast, lower accuracy | TFLite-native | fallback if v11n too slow |
| EfficientDet-Lite0-2 | 4.6-10MB | slower than SSD, better accuracy | TFLite-native via MediaPipe | good MediaPipe integration |
| NanoDet(-Plus) | 1-1.8MB | extremely fast, tiny | ONNX/NCNN great, TFLite/CoreML manual | smallest footprint option |

**Deployment runtime alternative:** Google's MediaPipe Tasks (Object Detector API) wraps
EfficientDet-Lite/MobileNet-SSD with ready Android/iOS/Web SDKs and a Model Maker for
retraining — a faster time-to-app path if you want to skip raw TFLite integration.

## Known data gaps (see DATASETS.md for full detail)

- **Garbage:** most public datasets label individual litter items, not >1m piles — filter/
  relabel toward dense clusters or merge nearby detections into one incident downstream.
- **Manholes:** smallest dataset pool (dozens-low thousands) — heaviest augmentation needed.
- **Streetlight state & encroachment:** no adequate off-the-shelf dataset; handled via the
  workarounds above rather than training data alone.
