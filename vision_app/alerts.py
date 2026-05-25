from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import csv

import cv2
import requests

_ALERT_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="telegram_alert")


def _normalize_bot_token(raw_token: str) -> str:
    token = (raw_token or "").strip()
    if token.startswith("https://api.telegram.org/bot"):
        token = token.split("/bot", 1)[1].split("/", 1)[0].strip()
    if token.startswith("bot"):
        token = token[3:]
    return token.strip()


def _normalize_chat_id(raw_chat_id: str) -> str:
    return (raw_chat_id or "").strip()


def send_telegram_alert(
    image_path: str,
    caption: str,
    bot_token: str,
    chat_id: str,
    timeout_sec: float = 5.0,
) -> tuple[bool, str]:
    normalized_token = _normalize_bot_token(bot_token)
    normalized_chat_id = _normalize_chat_id(chat_id)

    if not normalized_token or not normalized_chat_id:
        msg = "Telegram token/chat_id missing. Skipping send."
        print(msg)
        return False, msg

    url = f"https://api.telegram.org/bot{normalized_token}/sendPhoto"
    try:
        with open(image_path, "rb") as image_file:
            response = requests.post(
                url,
                data={"chat_id": normalized_chat_id, "caption": caption},
                files={"photo": (Path(image_path).name, image_file, "image/jpeg")},
                timeout=timeout_sec,
            )
        if response.status_code == 200:
            msg = "Telegram photo alert sent"
            print(msg)
            return True, msg

        details = ""
        try:
            details = response.json().get("description", "")
        except ValueError:
            details = response.text[:200]

        msg = f"Telegram API error: {response.status_code}"
        if details:
            msg = f"{msg} | {details}"
        print(msg)
        return False, msg
    except requests.RequestException as exc:
        msg = f"Telegram request failed: {exc}"
        print(msg)
        return False, msg
    except OSError as exc:
        msg = f"Failed to read image for Telegram: {exc}"
        print(msg)
        return False, msg


def handle_danger_alert(
    processed_frame,
    object_label: str,
    distance_m: float,
    confidence: float,
    bot_token: str,
    chat_id: str,
    alerts_dir: str = "alerts",
) -> tuple[bool, str]:
    timestamp = datetime.now()
    timestamp_text = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    file_timestamp = timestamp.strftime("%Y%m%d_%H%M%S")

    alerts_path = Path(alerts_dir)
    alerts_path.mkdir(parents=True, exist_ok=True)

    image_path = alerts_path / f"alert_{file_timestamp}.jpg"
    saved = cv2.imwrite(str(image_path), processed_frame)
    if not saved:
        msg = f"Failed to save alert snapshot: {image_path}"
        print(msg)
        return False, msg

    log_path = alerts_path / "alerts_log.csv"
    write_header = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(
            log_file,
            fieldnames=["timestamp", "object", "distance_m", "confidence", "image_path"],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": timestamp_text,
                "object": object_label,
                "distance_m": f"{distance_m:.2f}",
                "confidence": f"{confidence:.2f}",
                "image_path": str(image_path),
            }
        )

    caption = (
        f"🚨 Danger Alert\n"
        f"Object: {object_label}\n"
        f"Distance: {distance_m:.2f} m\n"
        f"Confidence: {confidence:.2f}\n"
        f"Time: {timestamp_text}"
    )

    ok, status = send_telegram_alert(
        image_path=str(image_path),
        caption=caption,
        bot_token=bot_token,
        chat_id=chat_id,
    )

    if ok:
        print(f"Alert pipeline success: {image_path}")
        return True, f"Snapshot saved and sent: {image_path.name}"

    print(f"Alert snapshot saved but Telegram failed/skipped: {image_path}")
    return False, f"Snapshot saved: {image_path.name} | {status}"


def handle_danger_alert_async(
    processed_frame,
    object_label: str,
    distance_m: float,
    confidence: float,
    bot_token: str,
    chat_id: str,
    alerts_dir: str = "alerts",
) -> Future[tuple[bool, str]]:
    return _ALERT_EXECUTOR.submit(
        handle_danger_alert,
        processed_frame,
        object_label,
        distance_m,
        confidence,
        bot_token,
        chat_id,
        alerts_dir,
    )
