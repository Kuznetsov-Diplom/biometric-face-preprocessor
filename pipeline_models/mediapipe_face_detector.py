import cv2
import mediapipe as mp
import numpy as np
from typing import Optional

from .interfaces.pipeline_step import PipelineStep
from .interfaces.pipeline_context import PipelineContext


class MediaPipeFaceDetector(PipelineStep):
    """Шаг пайплайна: детекция лица"""

    def __init__(self, min_detection_confidence: float = 0.85):
        super().__init__(min_detection_confidence=min_detection_confidence)
        self.min_detection_confidence = min_detection_confidence

        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=min_detection_confidence
        )

    def process(self, context: PipelineContext) -> PipelineContext:
        rgb = cv2.cvtColor(context.frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb)

        if results.detections:
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = context.frame.shape

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            context.face_bbox = (x, y, width, height)

        return context