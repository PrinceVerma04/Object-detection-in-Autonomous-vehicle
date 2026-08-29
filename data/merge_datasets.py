"""
Merges every dataset under data/raw/<key>/ into a single unified YOLO dataset at
data/unified/, remapping each source's class ids onto the 7-class scheme in
configs/data.yaml. Run after download_datasets.py (and after manually dropping any
Kaggle/Zenodo sources into data/raw/<key>/ in the same layout).

Usage:
    python data/merge_datasets.py
"""
import os
import shutil
import yaml
from pathlib import Path

DATA_DIR = Path(__file__).parent
RAW_DIR = DATA_DIR / "raw"
UNIFIED_DIR = DATA_DIR / "unified"
UNIFIED_CLASSES = yaml.safe_load((DATA_DIR.parent / "configs" / "data.yaml").read_text())["names"]
CLASS_TO_ID = {name: idx for idx, name in UNIFIED_CLASSES.items()}

# key -> unified target class for every box in that source (single-class remap).
TARGET_CLASS_BY_KEY = {
    "taco": "garbage_pile",
    "trash_detection": "garbage_pile",
    "pothole_intel": "pothole",
    "open_manholes": "manhole_open_broken",
    "streetlight": "streetlight_pole",
    "illegal_parking": "encroachment",
    "billboard": "billboard_hoarding",
    "cow_indian": "cattle",
    "street_vendors_india": "encroachment",
    "street_vendors": "encroachment",
    "manhole_broken_v2": "manhole_open_broken",
}

# sources with multiple original classes: only keep source classes whose name contains
# one of these substrings (case-insensitive), still remapped to a single target class.
SOURCE_CLASS_KEEP_FILTER = {
    "manhole_cover": (["broken", "uncover", "open"], "manhole_open_broken"),
    "footpath_guide": (["vendor", "obstacle", "street goods"], "encroachment"),
}


def find_split_dirs(source_dir: Path):
    for split in ("train", "valid", "val", "test"):
        img_dir = source_dir / split / "images"
        lbl_dir = source_dir / split / "labels"
        if img_dir.is_dir() and lbl_dir.is_dir():
            yield "val" if split in ("valid", "val", "test") else "train", img_dir, lbl_dir


def load_source_names(source_dir: Path):
    yaml_path = source_dir / "data.yaml"
    if not yaml_path.is_file():
        return {}
    names = yaml.safe_load(yaml_path.read_text()).get("names", [])
    if isinstance(names, dict):
        return names
    return {i: n for i, n in enumerate(names)}


def remap_label_file(src_label: Path, source_names: dict, keep_filter):
    lines_out = []
    for line in src_label.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        src_id = int(parts[0])
        src_name = str(source_names.get(src_id, "")).lower()

        if keep_filter:
            keywords, target_class = keep_filter
            if not any(k in src_name for k in keywords):
                continue
        else:
            target_class = None  # set by caller via TARGET_CLASS_BY_KEY
        yield target_class, parts[1:]


def merge_one(key: str):
    source_dir = RAW_DIR / key
    if not source_dir.is_dir():
        print(f"[skip] {key}: not downloaded")
        return 0

    keep_filter = SOURCE_CLASS_KEEP_FILTER.get(key)
    default_target = TARGET_CLASS_BY_KEY.get(key)
    source_names = load_source_names(source_dir)
    count = 0

    for split, img_dir, lbl_dir in find_split_dirs(source_dir):
        out_img_dir = UNIFIED_DIR / "images" / split
        out_lbl_dir = UNIFIED_DIR / "labels" / split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        for label_file in lbl_dir.glob("*.txt"):
            img_file = next((p for ext in (".jpg", ".jpeg", ".png") if (p := img_dir / (label_file.stem + ext)).is_file()), None)
            if img_file is None:
                continue

            out_lines = []
            for target_name, coords in remap_label_file(label_file, source_names, keep_filter):
                target_name = target_name or default_target
                if target_name is None:
                    continue
                out_lines.append(f"{CLASS_TO_ID[target_name]} {' '.join(coords)}")

            if not out_lines:
                continue

            new_stem = f"{key}_{label_file.stem}"
            shutil.copy(img_file, out_img_dir / f"{new_stem}{img_file.suffix}")
            (out_lbl_dir / f"{new_stem}.txt").write_text("\n".join(out_lines) + "\n")
            count += 1

    print(f"[merged] {key}: {count} images")
    return count


def main():
    total = 0
    for key in RAW_DIR.iterdir() if RAW_DIR.is_dir() else []:
        if key.is_dir():
            total += merge_one(key.name)
    print(f"Unified dataset: {total} images -> {UNIFIED_DIR}")


if __name__ == "__main__":
    main()
