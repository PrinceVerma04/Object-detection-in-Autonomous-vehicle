"""
No public dataset labels streetlights as lit/unlit (see DATASETS.md #4), so this task is
split in two: the unified YOLO model only localizes the 'streetlight_pole' class, and this
module classifies each crop as lit/unlit from pixel brightness at night. Swap
`is_lit_by_brightness` for a trained binary classifier once enough verified field crops
have been collected.

Usage:
    from streetlight.classify_lit_state import is_lit_by_brightness
    lit = is_lit_by_brightness(frame, bbox)
"""
import cv2
import numpy as np

BRIGHTNESS_THRESHOLD = 140  # mean V-channel (HSV) inside the lamp-head crop, 0-255
HALO_RATIO_THRESHOLD = 0.15  # fraction of crop pixels near-saturated white (glow)


def is_lit_by_brightness(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    mean_brightness = float(v_channel.mean())
    halo_ratio = float((v_channel > 230).mean())

    return mean_brightness > BRIGHTNESS_THRESHOLD or halo_ratio > HALO_RATIO_THRESHOLD
