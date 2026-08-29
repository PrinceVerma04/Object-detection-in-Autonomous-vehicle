"""
Pulls every Roboflow-hosted source dataset listed in DATASETS.md into data/raw/<key>/.
Requires a free Roboflow API key: https://app.roboflow.com/settings/api

Usage:
    export ROBOFLOW_API_KEY=xxxx
    python data/download_datasets.py

Kaggle sources (Domestic Trash, Annotated Potholes, Road Damage) and the Zenodo
StreetVendor-SLI set are not on Roboflow — download those separately with the
`kaggle datasets download` CLI / the Zenodo DOI link in DATASETS.md and drop them
under data/raw/<key>/ in the same {images,labels}/{train,val} layout before running
merge_datasets.py.
"""
import os
from roboflow import Roboflow

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")

# (output key, target unified class, workspace slug, project slug, version)
SOURCES = [
    ("taco", "garbage_pile", "mohamed-traore-2ekkp", "taco-trash-annotations-in-context", 16),
    ("trash_detection", "garbage_pile", "yolo-hfplh", "trash-detection-1fjjc-nd1dg", 1),
    ("pothole_intel", "pothole", "intel-unnati-training-program", "pothole-detection-bqu6s", 9),
    ("open_manholes", "manhole_open_broken", "aibased-solution-for-realtime-detection-of-road-anomalies-d6eay", "open-manholes", 1),
    ("manhole_cover", "manhole_open_broken", "create-dataset-for-yolo", "manhole-cover-dataset-yolo", 1),
    ("streetlight", "streetlight_pole", "sashank-s", "street-light", 1),
    ("illegal_parking", "encroachment", "parking-amu50", "illegal-parking", 1),
    ("billboard", "billboard_hoarding", "arslan-ongr8", "billboard-xlvz1", 1),
    ("cow_indian", "cattle", "sakshi-kumari-70uxi", "cow-indian", 1),
]


def main():
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("Set ROBOFLOW_API_KEY first (https://app.roboflow.com/settings/api)")

    rf = Roboflow(api_key=api_key)
    for key, target_class, workspace, project_slug, version in SOURCES:
        out_dir = os.path.join(RAW_DIR, key)
        if os.path.isdir(out_dir):
            print(f"[skip] {key} already downloaded")
            continue
        print(f"[fetch] {key} -> unified class '{target_class}'")
        try:
            project = rf.workspace(workspace).project(project_slug)
            dataset = project.version(version).download("yolov8", location=out_dir)
        except Exception as exc:
            print(f"  failed: {exc}  (project may have moved/renamed — check the URL in DATASETS.md)")


if __name__ == "__main__":
    main()
