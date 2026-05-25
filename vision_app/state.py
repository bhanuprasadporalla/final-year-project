import time
from collections import defaultdict

import streamlit as st


def init_state() -> None:
    defaults = {
        "running": False,
        "cap": None,
        "active_camera_index": 0,
        "camera_failures": 0,
        "last_frame_time": time.time(),
        "fps_ema": 0.0,
        "frame_count": 0,
        "chart_counter": 0,
        "last_detections": [],
        "confidence_history": [],
        "recent_confidences": [],
        "object_counts": defaultdict(int),
        "danger_object_counts": defaultdict(int),
        "prev_labels_in_frame": set(),
        "prev_danger_labels_in_frame": set(),
        "last_alert_time": {},
        "last_global_alert_time": 0.0,
        "last_alert_status": "No alerts yet",
        "pending_alerts": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
