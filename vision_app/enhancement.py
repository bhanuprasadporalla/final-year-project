import cv2
import numpy as np


def enhance_low_light(
    frame: np.ndarray,
    gamma: float = 1.6,
    clahe_clip_limit: float = 3.0,
    clahe_tile_size: int = 8,
) -> np.ndarray:
    """Enhance low-light frames with luminance CLAHE and gamma brightening."""
    if frame is None or frame.size == 0:
        return frame

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=float(clahe_clip_limit), tileGridSize=(int(clahe_tile_size), int(clahe_tile_size)))
    l_eq = clahe.apply(l_channel)

    merged_lab = cv2.merge((l_eq, a_channel, b_channel))
    enhanced = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

    inv_gamma = 1.0 / max(0.1, float(gamma))
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(enhanced, table)
