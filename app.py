import time

import cv2
import pandas as pd
import streamlit as st

from vision_app import (
    KNOWN_WIDTHS_M,
    classify_zone,
    enhance_low_light,
    estimate_distance_m,
    handle_danger_alert_async,
    init_state,
    load_model,
    open_camera,
    zone_color,
)

TELEGRAM_BOT_TOKEN_DEFAULT = "8571508973:AAFPfxA0NS2N8yRH1GnMdJmPyzQo53jZ3n8"
TELEGRAM_CHAT_ID_DEFAULT = "5031780473"
TARGET_CLASSES = {"person", "vehicle"}
PERSON_CLASS_ID = 0
VEHICLE_CLASS_IDS = {1, 2, 3, 5, 7}

st.set_page_config(
    page_title="YOLO11 Distance Zone Monitor",
    page_icon="camera",
    layout="wide",
)

init_state()
if "telegram_token_input" not in st.session_state:
    st.session_state.telegram_token_input = TELEGRAM_BOT_TOKEN_DEFAULT
if "telegram_chat_id_input" not in st.session_state:
    st.session_state.telegram_chat_id_input = TELEGRAM_CHAT_ID_DEFAULT

st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(120deg, #f3f7ff 0%, #eef9f3 100%);
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1rem;
    }
    .title-wrap {
        padding: 0.8rem 1rem;
        border-radius: 14px;
        background: linear-gradient(90deg, rgba(10,85,50,0.10), rgba(40,90,200,0.10));
        border: 1px solid rgba(0,0,0,0.08);
        margin-bottom: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class='title-wrap'>
        <h2 style='margin:0;'>night survilleance system</h2>
        <p style='margin:0.2rem 0 0 0;'>Webcam detection, distance estimation, zone classification, and Telegram danger alerts.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Detection Settings")
    model_path = st.text_input("Model path", value="yolo11n.pt")
    camera_index = st.number_input("Webcam index", min_value=0, max_value=10, value=0, step=1)
    conf_threshold = st.slider("Confidence threshold", min_value=0.1, max_value=0.95, value=0.35, step=0.05)
    inference_size = st.selectbox("Inference image size", options=[320, 416, 512, 640], index=1)
    detect_every_n_frames = st.number_input("Run detection every N frames", min_value=1, max_value=6, value=2, step=1)
    chart_update_every = st.number_input("Chart update interval (frames)", min_value=1, max_value=30, value=5, step=1)
    enable_night_enhancement = st.checkbox("Enable night-vision enhancement", value=True)
    gamma_value = st.slider("Enhancement gamma", min_value=1.0, max_value=3.0, value=1.6, step=0.1)
    clahe_clip_limit = st.slider("Enhancement CLAHE clip", min_value=1.0, max_value=6.0, value=3.0, step=0.5)

    st.subheader("Distance & Zones")
    focal_length_px = st.number_input(
        "Estimated focal length (pixels)",
        min_value=100.0,
        max_value=4000.0,
        value=800.0,
        step=50.0,
        help="Calibrate for better distance accuracy.",
    )
    default_width_m = st.number_input(
        "Default object width (m)",
        min_value=0.05,
        max_value=5.0,
        value=0.5,
        step=0.05,
    )
    danger_threshold_m = st.number_input("Danger threshold (m)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    warning_threshold_m = st.number_input("Warning threshold (m)", min_value=0.2, max_value=15.0, value=2.0, step=0.1)

    st.subheader("Telegram Alerts")
    telegram_token = st.text_input("Bot token", type="password", key="telegram_token_input")
    telegram_chat_id = st.text_input("Chat ID", key="telegram_chat_id_input")
    st.caption("Token can be pasted as plain token or full bot URL; spaces are auto-trimmed.")
    alert_cooldown_sec = st.number_input("Alert cooldown (sec)", min_value=20, max_value=300, value=20, step=1)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Start", use_container_width=True):
            reopen_required = (
                st.session_state.cap is None
                or int(camera_index) != int(st.session_state.active_camera_index)
            )
            if reopen_required:
                if st.session_state.cap is not None:
                    st.session_state.cap.release()
                    st.session_state.cap = None
                cap = open_camera(int(camera_index))
                if cap is not None:
                    st.session_state.cap = cap
                    st.session_state.running = True
                    st.session_state.active_camera_index = int(camera_index)
                    st.session_state.camera_failures = 0
                    st.session_state.last_alert_status = "Monitoring started"
                else:
                    st.session_state.last_alert_status = "Failed to open webcam"
            else:
                st.session_state.running = True

    with col_b:
        if st.button("Stop", use_container_width=True):
            st.session_state.running = False
            if st.session_state.cap is not None:
                st.session_state.cap.release()
                st.session_state.cap = None
            st.session_state.last_detections = []
            st.session_state.pending_alerts = []

if warning_threshold_m < danger_threshold_m:
    st.error("Warning threshold must be greater than or equal to danger threshold.")
    st.stop()

model = load_model(model_path)
model_names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
allowed_class_ids = [
    class_id
    for class_id, class_name in model_names.items()
    if class_id == PERSON_CLASS_ID or class_id in VEHICLE_CLASS_IDS
]

metrics_row = st.columns(10)
metric_fps = metrics_row[0].empty()
metric_total = metrics_row[1].empty()
metric_danger = metrics_row[2].empty()
metric_warning = metrics_row[3].empty()
metric_safe = metrics_row[4].empty()
metric_avg_distance = metrics_row[5].empty()
metric_closest = metrics_row[6].empty()
metric_danger_rate = metrics_row[7].empty()
metric_avg_conf_10f = metrics_row[8].empty()
metric_danger_session = metrics_row[9].empty()

left, right = st.columns([1.6, 1.0])
frame_placeholder = left.empty()
objects_title = right.empty()
objects_placeholder = right.empty()
accuracy_title = right.empty()
accuracy_chart_placeholder = right.empty()
danger_counts_title = right.empty()
danger_counts_placeholder = right.empty()
alert_placeholder = st.empty()

objects_title.subheader("Detected Objects (Current Frame)")
accuracy_title.subheader("Detection Accuracy Trend (Avg Confidence)")
danger_counts_title.subheader("Danger Object Counts (Session)")


@st.fragment(run_every="250ms")
def render_live_panel() -> None:
    try:
        if st.session_state.pending_alerts:
            remaining_alerts = []
            for pending in st.session_state.pending_alerts:
                future = pending["future"]
                label = pending["label"]
                if future.done():
                    try:
                        ok, status = future.result()
                        if ok:
                            st.session_state.last_alert_status = f"Alert sent: {label} | {status}"
                        else:
                            st.session_state.last_alert_status = status
                    except Exception as exc:
                        st.session_state.last_alert_status = f"Telegram alert failed: {exc}"
                else:
                    remaining_alerts.append(pending)
            st.session_state.pending_alerts = remaining_alerts

        if st.session_state.running and st.session_state.cap is None:
            st.session_state.cap = open_camera(int(camera_index))
            st.session_state.active_camera_index = int(camera_index)

        if st.session_state.running and st.session_state.cap is not None:
            ret, frame = st.session_state.cap.read()
            if not ret:
                st.session_state.camera_failures += 1
                if st.session_state.cap is not None:
                    st.session_state.cap.release()
                    st.session_state.cap = None

                if st.session_state.camera_failures <= 10:
                    st.session_state.cap = open_camera(int(camera_index))
                    frame_placeholder.info("Webcam frame dropped. Reconnecting camera...")
                else:
                    st.warning("Webcam disconnected repeatedly. Click Start to retry.")
                    st.session_state.running = False
                    st.session_state.camera_failures = 0
                return

            st.session_state.camera_failures = 0
            processed_frame = frame
            if enable_night_enhancement:
                processed_frame = enhance_low_light(
                    frame,
                    gamma=float(gamma_value),
                    clahe_clip_limit=float(clahe_clip_limit),
                )

            st.session_state.frame_count += 1
            run_inference = (
                st.session_state.frame_count % int(detect_every_n_frames) == 0
                or not st.session_state.last_detections
            )

            if run_inference:
                results = model.predict(
                    processed_frame,
                    conf=conf_threshold,
                    imgsz=int(inference_size),
                    classes=allowed_class_ids if allowed_class_ids else None,
                    verbose=False,
                )
                result = results[0]
                detections = []
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        if cls_id not in allowed_class_ids:
                            continue
                        label = "person" if cls_id == PERSON_CLASS_ID else "vehicle"
                        pixel_width = max(1.0, x2 - x1)
                        real_width_m = (
                            float(default_width_m)
                            if label == "vehicle"
                            else KNOWN_WIDTHS_M.get(label, float(default_width_m))
                        )
                        distance_m = estimate_distance_m(real_width_m, pixel_width, float(focal_length_px))
                        zone = classify_zone(distance_m, float(danger_threshold_m), float(warning_threshold_m))
                        detections.append(
                            {
                                "x1": float(x1),
                                "y1": float(y1),
                                "x2": float(x2),
                                "y2": float(y2),
                                "label": label,
                                "confidence": conf,
                                "distance_m": float(distance_m),
                                "zone": zone,
                            }
                        )
                st.session_state.last_detections = detections
            else:
                detections = st.session_state.last_detections

            records = []
            danger_count = 0
            warning_count = 0
            safe_count = 0
            current_labels = set()
            current_danger_labels = set()
            danger_candidates = []

            for det in detections:
                x1 = det["x1"]
                y1 = det["y1"]
                x2 = det["x2"]
                y2 = det["y2"]
                label = det["label"]
                conf = float(det["confidence"])
                distance_m = float(det["distance_m"])
                zone = det["zone"]
                current_labels.add(label)

                if zone == "Danger":
                    danger_count += 1
                    current_danger_labels.add(label)
                    if run_inference:
                        danger_candidates.append(
                            {
                                "label": label,
                                "distance_m": distance_m,
                                "confidence": conf,
                            }
                        )
                elif zone == "Warning":
                    warning_count += 1
                else:
                    safe_count += 1

                color = zone_color(zone)
                cv2.rectangle(processed_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                text = f"{label} {conf:.2f} | {distance_m:.2f}m | {zone}"
                cv2.putText(
                    processed_frame,
                    text,
                    (int(x1), max(20, int(y1) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )

                records.append(
                    {
                        "Object": label,
                        "Confidence": round(conf, 3),
                        "Distance (m)": round(distance_m, 3),
                        "Zone": zone,
                    }
                )

            # Trigger one snapshot alert per cooldown window using the closest danger object.
            now = time.time()
            if danger_candidates and (
                now - st.session_state.last_global_alert_time >= float(alert_cooldown_sec)
            ):
                target = min(danger_candidates, key=lambda item: item["distance_m"])
                future = handle_danger_alert_async(
                    processed_frame=processed_frame.copy(),
                    object_label=target["label"],
                    distance_m=target["distance_m"],
                    confidence=target["confidence"],
                    bot_token=telegram_token,
                    chat_id=telegram_chat_id,
                    alerts_dir="alerts",
                )
                st.session_state.pending_alerts.append(
                    {"future": future, "label": target["label"]}
                )
                st.session_state.last_global_alert_time = now
                st.session_state.last_alert_status = f"Alert queued: {target['label']}"

            # Count object entries only when a class appears after being absent.
            newly_entered_labels = current_labels - st.session_state.prev_labels_in_frame
            for entered_label in newly_entered_labels:
                st.session_state.object_counts[entered_label] += 1
            st.session_state.prev_labels_in_frame = current_labels

            # Count danger entries only when a class enters danger after being absent from danger.
            newly_danger_labels = current_danger_labels - st.session_state.prev_danger_labels_in_frame
            for danger_label in newly_danger_labels:
                st.session_state.danger_object_counts[danger_label] += 1
            st.session_state.prev_danger_labels_in_frame = current_danger_labels

            now = time.time()
            elapsed = max(1e-6, now - st.session_state.last_frame_time)
            fps = 1.0 / elapsed
            if st.session_state.fps_ema <= 0:
                st.session_state.fps_ema = fps
            else:
                st.session_state.fps_ema = (0.2 * fps) + (0.8 * st.session_state.fps_ema)
            st.session_state.last_frame_time = now

            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, caption="Live Webcam Feed", use_container_width=True)

            df = pd.DataFrame(records)
            if not df.empty:
                objects_placeholder.dataframe(df, use_container_width=True, hide_index=True)
                avg_conf = float(df["Confidence"].mean())
                st.session_state.recent_confidences.append(avg_conf)
                if len(st.session_state.recent_confidences) > 10:
                    st.session_state.recent_confidences = st.session_state.recent_confidences[-10:]
                st.session_state.confidence_history.append(avg_conf)
                if len(st.session_state.confidence_history) > 300:
                    st.session_state.confidence_history = st.session_state.confidence_history[-300:]
            else:
                objects_placeholder.info("No objects detected in this frame.")
                st.session_state.confidence_history.append(float("nan"))
                if len(st.session_state.confidence_history) > 300:
                    st.session_state.confidence_history = st.session_state.confidence_history[-300:]

            st.session_state.chart_counter += 1
            if st.session_state.chart_counter % int(chart_update_every) == 0:
                conf_hist = pd.DataFrame(
                    {
                        "Frame": list(range(len(st.session_state.confidence_history))),
                        "Avg Confidence": st.session_state.confidence_history,
                    }
                )
                accuracy_chart_placeholder.line_chart(conf_hist.set_index("Frame"), use_container_width=True)

                danger_counts_df = pd.DataFrame(
                    {
                        "Object": list(st.session_state.danger_object_counts.keys()),
                        "Danger Count": list(st.session_state.danger_object_counts.values()),
                    }
                )
                if not danger_counts_df.empty:
                    danger_counts_df = danger_counts_df.sort_values("Danger Count", ascending=False)
                    danger_counts_placeholder.dataframe(
                        danger_counts_df, use_container_width=True, hide_index=True
                    )
                else:
                    danger_counts_placeholder.info("No danger objects counted yet.")

            metric_fps.metric("FPS", f"{st.session_state.fps_ema:.1f}")
            metric_total.metric("Total Detected", f"{len(records)}")
            metric_danger.metric("Danger", str(danger_count))
            metric_warning.metric("Warning", str(warning_count))
            metric_safe.metric("Safe", str(safe_count))
            metric_danger_session.metric(
                "Danger (Session)", str(sum(st.session_state.danger_object_counts.values()))
            )
            if records:
                distances = [item["Distance (m)"] for item in records]
                avg_distance = sum(distances) / len(distances)
                closest_distance = min(distances)
                danger_rate = (danger_count / len(records)) * 100.0
                avg_conf_10f = sum(st.session_state.recent_confidences) / max(1, len(st.session_state.recent_confidences))
                metric_avg_distance.metric("Avg Distance", f"{avg_distance:.2f} m")
                metric_closest.metric("Closest", f"{closest_distance:.2f} m")
                metric_danger_rate.metric("Danger Rate", f"{danger_rate:.0f}%")
                metric_avg_conf_10f.metric("Avg Conf (10f)", f"{avg_conf_10f:.2f}")
            else:
                metric_avg_distance.metric("Avg Distance", "N/A")
                metric_closest.metric("Closest", "N/A")
                metric_danger_rate.metric("Danger Rate", "0%")
                metric_avg_conf_10f.metric("Avg Conf (10f)", "N/A")

            alert_placeholder.info(f"Telegram status: {st.session_state.last_alert_status}")
            return

        frame_placeholder.info("Click Start to begin webcam monitoring.")
        alert_placeholder.info(f"Telegram status: {st.session_state.last_alert_status}")
        if st.session_state.cap is not None:
            st.session_state.cap.release()
            st.session_state.cap = None
    except Exception as exc:
        st.session_state.last_alert_status = f"Runtime error: {exc}"
        st.error(f"App runtime error: {exc}")


render_live_panel()
