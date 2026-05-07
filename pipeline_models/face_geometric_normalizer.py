import cv2
import numpy as np

from .interfaces.pipeline_context import PipelineContext
from .interfaces.pipeline_step import PipelineStep


class FaceGeometricNormalizer(PipelineStep):
    """Шаг 3 по plan.md: Геометрическая нормализация (выравнивание по глазам)
    • Выравнивание межзрачковой линии по горизонтали
    • Масштабирование по межзрачковому расстоянию
    • Центрирование + приведение к фиксированному размеру (224×224)"""

    def __init__(self,
                 name: str = "geometric_normalization",
                 target_size: int = 224,
                 target_inter_pupil_distance: float = 100.0):
        super().__init__(name=name,
                         target_size=target_size,
                         target_inter_pupil_distance=target_inter_pupil_distance)
        self.target_size = target_size
        self.target_inter_pupil_distance = target_inter_pupil_distance

    def _get_eye_centers(self, landmarks_points: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
        """Извлекаем центры глаз по iris landmarks (MediaPipe 468 + refine_landmarks)"""
        if len(landmarks_points) < 474:
            # fallback — используем приблизительные точки глаз
            left_eye = np.mean([landmarks_points[i] for i in [33, 133, 160, 158, 144, 145]], axis=0)
            right_eye = np.mean([landmarks_points[i] for i in [362, 263, 387, 385, 373, 374]], axis=0)
        else:
            # точные iris-центры (стандарт MediaPipe)
            left_eye = np.array(landmarks_points[468])
            right_eye = np.array(landmarks_points[473])
        return left_eye, right_eye

    def process(self, context: PipelineContext) -> PipelineContext:
        context.current_step = self.name

        if context.cropped_face is None or context.landmarks is None:
            return context

        landmarks_points = context.landmarks.get("points", [])
        if len(landmarks_points) < 468:
            return context

        # Приводим landmarks к координатам cropped_face
        if context.crop_offset is not None:
            ox, oy = context.crop_offset
            rel_points = [(x - ox, y - oy) for x, y in landmarks_points]
        else:
            rel_points = landmarks_points

        # 1. Центры глаз (теперь в координатах cropped_face!)
        left_eye, right_eye = self._get_eye_centers(rel_points)
        eyes_center = (left_eye + right_eye) / 2
        inter_pupil = np.linalg.norm(right_eye - left_eye)

        # 2. Угол поворота
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))

        # 3. Масштаб
        scale = self.target_inter_pupil_distance / inter_pupil if inter_pupil > 0 else 1.0

        # 4. Аффинное преобразование
        h, w = context.cropped_face.shape[:2]
        center = (w / 2, h / 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, scale)

        # Сдвигаем так, чтобы центр глаз оказался в центре целевого изображения
        rot_mat[0, 2] += (self.target_size / 2) - eyes_center[0] * scale
        rot_mat[1, 2] += (self.target_size / 2) - eyes_center[1] * scale

        # 5. Применяем трансформацию к cropped_face
        aligned = cv2.warpAffine(
            context.cropped_face,
            rot_mat,
            (self.target_size, self.target_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        # 6. Нормализуем ландмарки (применяем ту же матрицу к shifted)
        norm_landmarks = []
        for (x, y) in rel_points:
            pts = np.array([[x, y, 1.0]])
            transformed = pts @ rot_mat.T
            norm_landmarks.append((transformed[0][0], transformed[0][1]))

        # Сохраняем результаты
        context.aligned_face = aligned
        context.normalized_landmarks = np.array(norm_landmarks, dtype=np.float32)
        context.inter_pupil_distance = float(inter_pupil * scale)

        return context