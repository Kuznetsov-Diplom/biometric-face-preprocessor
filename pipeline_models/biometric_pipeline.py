from typing import List, Optional, Iterator
from .interfaces.pipeline_context import PipelineContext
from .interfaces.pipeline_step import PipelineStep
from .interfaces.video_input import VideoInput
import cv2
import numpy as np


class BiometricPreprocessorPipeline:
    """Главный пайплайн — builder-style API"""

    def __init__(self):
        self._steps: List[PipelineStep] = []
        self._video_source: Optional[VideoInput] = None

    def add_step(self, step: PipelineStep) -> "BiometricPreprocessorPipeline":
        self._steps.append(step)
        return self

    def add_source(self, video_input: VideoInput) -> "BiometricPreprocessorPipeline":
        self._video_source = video_input
        return self

    @property
    def steps(self) -> List[PipelineStep]:
        """Для сайта — список шагов, чтобы переключать отображаемые слои"""
        return self._steps

    def update_step_params(self, step_name: str, **new_params) -> None:
        """Динамическая смена параметров (слайдеры в демо)"""
        for step in self._steps:
            if step.name == step_name:
                step.update_params(**new_params)
                return
        raise ValueError(f"Шаг {step_name} не найден")

    def process_single_frame(self, frame: np.ndarray) -> PipelineContext:
        """Удобно для live-камеры в веб-приложении"""
        context = PipelineContext(frame_idx=0, frame=frame.copy())
        for step in self._steps:
            context = step(context)
        return context

    def get_overlay_frame(self, context: PipelineContext) -> np.ndarray:
        """Готовый кадр с зелёным оверлеем (bbox пока — остальное добавим позже)"""
        overlay = context.frame.copy()
        if context.face_bbox:
            x, y, w, h = context.face_bbox
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 3)
            # TODO: когда добавим landmarks — сюда же circles
        return overlay

    def run(self, max_frames: Optional[int] = None) -> Iterator[PipelineContext]:
        """Генератор — если хочешь обрабатывать потоково"""
        if not self._video_source:
            raise ValueError("Сначала добавь источник видео через .add_source()")

        for frame_idx, frame in self._video_source:
            context = self.process_single_frame(frame)
            yield context

            if max_frames is not None and frame_idx >= max_frames - 1:
                break

    def process_all(self, max_frames: Optional[int] = None) -> list[PipelineContext]:
        """Удобный метод — сразу возвращает список всех результатов"""
        return list(self.run(max_frames=max_frames))