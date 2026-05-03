import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple, Dict

from .interfaces.face_detector import FaceDetector


class MediaPipeFaceDetector(FaceDetector):
    """Реализация детекции лица через MediaPipe (очень быстрый)."""

    def __init__(self, min_detection_confidence: float = 0.85):
        self.min_detection_confidence = min_detection_confidence

        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,  # 0 — короткое расстояние, 1 — дальнее
            min_detection_confidence=min_detection_confidence
        )

    def detect(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Простая детекция — возвращает (x, y, w, h)"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)

        if not results.detections:
            return None

        # Берём первое (самое уверенное) лицо
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box

        h, w, _ = frame.shape
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)

        return x, y, width, height

    def detect_with_landmarks(self, frame: np.ndarray) -> Optional[Dict]:
        """Полная информация + ключевые точки"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)

        if not results.detections:
            return None

        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box

        h, w, _ = frame.shape
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)

        return {
            "bbox": (x, y, width, height),
            "confidence": detection.score[0],
            "landmarks": detection.location_data.relative_keypoints  # 6 точек
        }