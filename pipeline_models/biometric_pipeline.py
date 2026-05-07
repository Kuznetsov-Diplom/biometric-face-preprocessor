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
        """Обрабатывает кадр до указанного шага (или до конца).
        stop_after_step=None → полный пайплайн (как было раньше)"""
        context = PipelineContext(frame_idx=0, frame=frame.copy())

        for step in self._steps:
            context = step(context)   # шаг вызывается как callable
            if stop_after_step and step.name == stop_after_step:
                break  # останавливаемся после нужного шага

        return context

    def get_preview_frame(self, context: PipelineContext, preview_step: str) -> np.ndarray:
        """Возвращает именно ту картинку, которую хочет видеть пользователь"""
        if preview_step == "raw":
            return context.frame.copy()

        elif preview_step == "frame_preprocessing":
            return context.frame.copy() if context.frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)

        elif preview_step == "face_detection":
            # после детекции — оригинальный кадр с оверлеем
            return self.get_overlay_frame(context)

        elif preview_step == "geometric_normalization":
            # после выравнивания — aligned_face
            if context.aligned_face is not None:
                return context.aligned_face.copy()
            return context.frame.copy()  # fallback

        # по умолчанию — полный оверлей
        return self.get_overlay_frame(context)

    def get_overlay_frame(self, context: PipelineContext) -> np.ndarray:
        """Готовый кадр с оверлеем (bbox + 468 точек) — используется для face_detection"""
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