from typing import List, Optional, Iterator
from .interfaces.pipeline_context import PipelineContext
from .interfaces.pipeline_step import PipelineStep
from .interfaces.video_input import VideoInput

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

    def run(self, max_frames: Optional[int] = None) -> Iterator[PipelineContext]:
        """Генератор — если хочешь обрабатывать потоково"""
        if not self._video_source:
            raise ValueError("Сначала добавь источник видео через .add_source()")

        for frame_idx, frame in self._video_source:
            context = PipelineContext(frame_idx=frame_idx, frame=frame.copy())

            for step in self._steps:
                context = step(context)

            yield context

            if max_frames is not None and frame_idx >= max_frames - 1:
                break

    def process_all(self, max_frames: Optional[int] = None) -> list[PipelineContext]:
        """Удобный метод — сразу возвращает список всех результатов (без for снаружи)"""
        return list(self.run(max_frames=max_frames))