from collections.abc import Iterator
from typing import Optional

import cv2
import numpy as np

from .interfaces.pipeline_context import PipelineContext
from .interfaces.pipeline_step import PipelineStep
from .interfaces.video_input import VideoInput


class BiometricPreprocessorPipeline:
    """Главный пайплайн — builder-style API"""

    def __init__(self):
        self._steps: list[PipelineStep] = []
        self._video_source: Optional[VideoInput] = None
        self._preview_size = (640, 480)          # фиксированный размер для всех preview

    def add_step(self, step: PipelineStep) -> "BiometricPreprocessorPipeline":
        self._steps.append(step)
        return self

    def add_source(self, video_input: VideoInput) -> "BiometricPreprocessorPipeline":
        self._video_source = video_input
        return self

    @property
    def steps(self) -> list[PipelineStep]:
        return self._steps

    def update_step_params(self, step_name: str, **new_params) -> None:
        for step in self._steps:
            if step.name == step_name:
                step.update_params(**new_params)
                return
        raise ValueError(f"Шаг {step_name} не найден")

    def process_single_frame(self, frame: np.ndarray, stop_after_step: Optional[str] = None) -> PipelineContext:
        context = PipelineContext(frame_idx=0, frame=frame.copy())

        for step in self._steps:
            context = step(context)
            if stop_after_step and step.name == stop_after_step:
                break
        return context

    def _center_image(self, img: np.ndarray) -> np.ndarray:
        """Центрирует любое изображение (в т.ч. 448×448) в фиксированном canvas 640×480"""
        h, w = img.shape[:2]
        canvas_h, canvas_w = self._preview_size
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        y_offset = (canvas_h - h) // 2
        x_offset = (canvas_w - w) // 2

        canvas[y_offset:y_offset + h, x_offset:x_offset + w] = img
        return canvas

    def get_preview_frame(self, context: PipelineContext, preview_step: str) -> np.ndarray:
        """Всегда возвращает изображение одного размера → окно больше не летает"""
        if preview_step == "raw":
            return cv2.resize(context.frame, self._preview_size)

        elif preview_step == "frame_preprocessing":
            return cv2.resize(context.frame, self._preview_size) if context.frame is not None else np.zeros((*self._preview_size, 3), dtype=np.uint8)

        elif preview_step == "face_detection":
            overlay = self.get_overlay_frame(context)
            return cv2.resize(overlay, self._preview_size)

        elif preview_step == "geometric_normalization":
            if context.aligned_face is not None:
                # Увеличиваем до 448×448 — лицо становится крупным и стабильным
                large_aligned = cv2.resize(context.aligned_face, (448, 448), interpolation=cv2.INTER_LINEAR)
                return self._center_image(large_aligned)
            return np.zeros((*self._preview_size, 3), dtype=np.uint8)

        elif preview_step == "temporal_landmarks_smoother":
            if hasattr(context, 'aligned_face') and context.aligned_face is not None:
                preview = context.aligned_face.copy()
                if hasattr(context, 'smoothed_landmarks') and context.smoothed_landmarks is not None:
                    # Рисуем сглаженные ландмарки (красные точки)
                    for (x, y) in context.smoothed_landmarks.astype(int):
                        cv2.circle(preview, (x, y), 2, (0, 0, 255), -1)   # красные точки
                    # Опционально: соединяем в контур (для наглядности)
                    cv2.polylines(preview, [context.smoothed_landmarks.astype(int)], False, (0, 255, 0), 1)
                return preview
            else:
                return context.aligned_face if hasattr(context, 'aligned_face') else context.current_frame

        # fallback
        return cv2.resize(self.get_overlay_frame(context), self._preview_size)

    def get_overlay_frame(self, context: PipelineContext) -> np.ndarray:
        """Оверлей bbox + 468 точек"""
        overlay = context.frame.copy()

        if context.face_bbox:
            x, y, w, h = context.face_bbox
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 3)

        if context.landmarks and "points" in context.landmarks:
            for (lx, ly) in context.landmarks["points"]:
                cx, cy = int(lx), int(ly)
                cv2.circle(overlay, (cx, cy), 1, (0, 255, 0), -1)

        return overlay

    def run(self, max_frames: Optional[int] = None) -> Iterator[PipelineContext]:
        if not self._video_source:
            raise ValueError("Сначала добавь источник видео")
        for frame_idx, frame in self._video_source:
            context = self.process_single_frame(frame)
            yield context
            if max_frames is not None and frame_idx >= max_frames - 1:
                break

    def process_all(self, max_frames: Optional[int] = None) -> list[PipelineContext]:
        return list(self.run(max_frames=max_frames))