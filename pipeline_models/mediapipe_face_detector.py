import cv2
import mediapipe as mp

from .interfaces.pipeline_context import PipelineContext
from .interfaces.pipeline_step import PipelineStep


class MediaPipeFaceDetector(PipelineStep):
    """Шаг 2 + 4 по plan.md: детекция лица + 468 ландмарков (MediaPipe Face Mesh)
    Полностью совместим с текущим app.py"""

    def __init__(self,
                 name: str = "face_detection",
                 min_detection_confidence: float = 0.85,
                 padding_ratio: float = 0.15):
        super().__init__(name=name, min_detection_confidence=min_detection_confidence)
        self.min_detection_confidence = min_detection_confidence
        self.padding_ratio = padding_ratio

        # Face Detection (для быстрого bbox)
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=min_detection_confidence
        )

        # Face Mesh (468 точек) — основной выбор по plan.md
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )

    def process(self, context: PipelineContext) -> PipelineContext:
        """Обрабатывает кадр: bbox + 468 ландмарков + ROI-кроп"""
        context.current_step = self.name

        if context.frame is None:
            return context

        rgb = cv2.cvtColor(context.frame, cv2.COLOR_BGR2RGB)

        # 1. Быстрый bbox (FaceDetection)
        detection_results = self.face_detection.process(rgb)
        context.confidence = 0.0
        context.face_bbox = None
        context.roi_size = (0, 0)

        if detection_results.detections:
            detection = detection_results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = context.frame.shape

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            context.face_bbox = (x, y, width, height)
            context.roi_size = (width, height)
            context.confidence = float(detection.score[0]) if detection.score else 0.0

            # 2. Полные 468 ландмарков (FaceMesh)
            mesh_results = self.face_mesh.process(rgb)
            if mesh_results.multi_face_landmarks:
                landmarks = mesh_results.multi_face_landmarks[0]

                # Сохраняем как dict для удобства (можно потом конвертировать в np.array)
                context.landmarks = {
                    "points": [(lm.x * w, lm.y * h) for lm in landmarks.landmark],
                    "count": 468
                }

                # 3. Кроп ROI с padding
                pad_x = int(width * self.padding_ratio)
                pad_y = int(height * self.padding_ratio)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(w, x + width + pad_x)
                y2 = min(h, y + height + pad_y)

                context.cropped_face = context.frame[y1:y2, x1:x2].copy()

        return context