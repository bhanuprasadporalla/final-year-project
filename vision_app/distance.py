def estimate_distance_m(real_width_m: float, bbox_pixel_width: float, focal_length_px: float) -> float:
    if bbox_pixel_width <= 1:
        return float("inf")
    return (real_width_m * focal_length_px) / bbox_pixel_width


def classify_zone(distance_m: float, danger_threshold_m: float, warning_threshold_m: float) -> str:
    if distance_m < danger_threshold_m:
        return "Danger"
    if danger_threshold_m <= distance_m <= warning_threshold_m:
        return "Warning"
    return "Safe"


def zone_color(zone: str) -> tuple[int, int, int]:
    if zone == "Danger":
        return (0, 0, 255)
    if zone == "Warning":
        return (0, 165, 255)
    return (0, 200, 0)
