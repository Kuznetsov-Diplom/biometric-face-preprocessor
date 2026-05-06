import cv2
# Надёжный импорт для Docker slim-образа (избегаем проблемы с mp.solutions)
from mediapipe.python.solutions import face_detection as mp_face_detection

from .interfaces.pipeline_context import PipelineContext
from .interfaces.pipeline_step import PipelineStep


class MediaPipeFaceDetector(PipelineStep):
    """Шаг пайплайна: детекция лица (MediaPipe) — полностью совместим с app.py"""

    def __init__(self, name: str = "face_detection", min_detection_confidence: float = 0.85):
        super().__init__(name=name, min_detection_confidence=min_detection_confidence)
        self.min_detection_confidence = min_detection_confidence

        self.face_detection = mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=min_detection_confidence
        )

    def process(self, context: PipelineContext) -> PipelineContext:
        """Обрабатывает кадр и заполняет все поля, которые использует app.py"""
        rgb = cv2.cvtColor(context.frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb)

        context.current_step = self.name
        context.stabilization = 100.0

        if results.detections:
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = context.frame.shape

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            context.face_bbox = (x, y, width, height)
            context.roi_size = (width, height)
            context.confidence = float(detection.score[0]) if detection.score else 0.0
            context.inter_pupil_distance = 0.0
        else:
            context.confidence = 0.0
            context.roi_size = (0, 0)
            context.face_bbox = None

        return context