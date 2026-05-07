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
        context = PipelineContext(frame_idx=0, frame=frame.copy())

        for step in self._steps:
            context = step(context)
            if stop_after_step and step.name == stop_after_step:
                break
        return context

    def _center_image(self, small_img: np.ndarray, canvas_size: tuple[int, int]) -> np.ndarray:
        """Центрирует 224×224 (или любое маленькое) изображение в большом canvas"""
        h, w = small_img.shape[:2]
        canvas_h, canvas_w = canvas_size

        # Чёрный canvas того же размера, что и оригинальный кадр
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        # Центрируем
        y_offset = (canvas_h - h) // 2
        x_offset = (canvas_w - w) // 2

        canvas[y_offset:y_offset + h, x_offset:x_offset + w] = small_img
        return canvas

    def get_preview_frame(self, context: PipelineContext, preview_step: str) -> np.ndarray:
        """Возвращает картинку нужного шага + стабилизированный размер"""
        if preview_step == "raw":
            return context.frame.copy()

        elif preview_step == "frame_preprocessing":
            return context.frame.copy() if context.frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)

        elif preview_step == "face_detection":
            return self.get_overlay_frame(context)

        elif preview_step == "geometric_normalization":
            if context.aligned_face is not None:
                # Привязываем 224×224 к размеру оригинального кадра (чтобы не летало)
                h, w = context.frame.shape[:2]
                return self._center_image(context.aligned_face, (h, w))
            return context.frame.copy()

        # fallback
        return self.get_overlay_frame(context)

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