from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

# 🚨 [계약서 19-7-1 §1 위반 해소]
# 무겁고 중복되는 insightface(buffalo_l) 임포트 및 초기화 코드를 완전히 제거했습니다.

@dataclass(frozen=True)
class FaceDetectionResult:
    detected: bool
    bbox: tuple[int, int, int, int]
    preview_bytes: bytes
    score: Optional[float] = None


def _decode_image(image_bytes: bytes):
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


# 🚨 [경량화 기본 검출] OpenCV 내장 기본 Haar Cascade 가중치 파일 로드 (별도 용량 차지 없음)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def _draw_result(image: np.ndarray, bbox: tuple[int, int, int, int], score: Optional[float]):
    """이미지 위에 초록색 박스와 유사도 점수를 그립니다."""
    preview = image.copy()
    x, y, w, h = bbox

    # 1. 초록색 얼굴 박스 그리기 (BGR: (87, 217, 118))
    cv2.rectangle(preview, (x, y), (x + w, y + h), (87, 217, 118), 4)

    # 2. 점수가 존재할 때 글자 그리기
    if score is not None:
        text = f"Accuracy: {score:.1%}"

        # 글자가 화면 상단 밖으로 가려지지 않도록 y 좌표 안전장치 설정
        text_y = y - 15 if y - 15 > 30 else y + 40
        text_x = x + 5

        # 가시성을 극대화하기 위해 검은색 외곽선(그림자 효과) 먼저 그리기
        cv2.putText(preview, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 0), 5, cv2.LINE_AA)

        # 그 위에 선명한 초록색 글씨 얹기
        cv2.putText(preview, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (87, 217, 118), 2, cv2.LINE_AA)

    return preview


def detect_largest_face(image_bytes: bytes, mode: str = "detection",
                        forced_score: Optional[float] = None) -> FaceDetectionResult:
    """OpenCV를 사용하여 얼굴을 가볍게 검출하고 정확도 오버레이를 안전하게 수행합니다."""
    image = _decode_image(image_bytes)
    if image is None:
        return FaceDetectionResult(False, (0, 0, 0, 0), image_bytes)

    # 1. OpenCV 연산을 위해 그레이스케일로 변환 후 얼굴 검출
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        ok, encoded = cv2.imencode(".jpg", image)
        return FaceDetectionResult(False, (0, 0, 0, 0), encoded.tobytes() if ok else image_bytes)

    # 2. 가장 큰 얼굴 선택 (OpenCV 결과 배열은 [x, y, w, h] 구조입니다)
    main_face = max(faces, key=lambda f: f[2] * f[3])
    x, y, w, h = main_face
    face_bbox = (int(x), int(y), int(w), int(h))

    # 3. 정확도 점수 결정
    # 🚨 [계약서 준수] 프론트엔드에서는 임베딩 계산 및 백엔드 중복 로직(dot product 유사도 연산)을 수행하지 않습니다.
    # 백엔드 API 통신 후 리턴받아 아규먼트로 꽂히는 forced_score만 안전하게 매핑합니다.
    score_to_draw = forced_score

    # 4. 이미지 위에 박스와 점수 그리기
    preview = _draw_result(image, face_bbox, score_to_draw)

    # 5. 최종 결과 반환 (.jpg 포맷 인코딩)
    ok, encoded = cv2.imencode(".jpg", preview)
    return FaceDetectionResult(True, face_bbox, encoded.tobytes() if ok else image_bytes, score=score_to_draw)