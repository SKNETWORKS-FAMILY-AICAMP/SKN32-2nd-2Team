from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FaceDetectionResult:
    detected: bool
    bbox: tuple[int, int, int, int]
    preview_bytes: bytes


def _decode_image(image_bytes: bytes):
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def detect_largest_face(image_bytes: bytes) -> FaceDetectionResult:
    image = _decode_image(image_bytes)
    if image is None:
        return FaceDetectionResult(False, (0, 0, 0, 0), image_bytes)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

    if len(faces) == 0:
        ok, encoded = cv2.imencode(".jpg", image)
        return FaceDetectionResult(False, (0, 0, 0, 0), encoded.tobytes() if ok else image_bytes)

    x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
    preview = image.copy()
    cv2.rectangle(preview, (x, y), (x + w, y + h), (118, 217, 87), 4)
    ok, encoded = cv2.imencode(".jpg", preview)
    return FaceDetectionResult(True, (int(x), int(y), int(w), int(h)), encoded.tobytes() if ok else image_bytes)
