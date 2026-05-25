import cv2


def open_camera(camera_index: int) -> cv2.VideoCapture | None:
    # Prefer DirectShow on Windows for more stable webcam capture.
    cap = cv2.VideoCapture(int(camera_index), cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(int(camera_index))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap
