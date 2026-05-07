import numpy as np

from .interfaces.pipeline_context import PipelineContext
from .interfaces.pipeline_step import PipelineStep


class TemporalLandmarksSmoother(PipelineStep):
    """Шаг 5 по plan.md: Стабилизация ключевых точек
    Экспоненциальное скользящее среднее (EMA) для снижения jitter (~35% по плану)"""

    def __init__(self,
                 name: str = "temporal_landmarks_smoother",
                 alpha: float = 0.75):   # 0.7–0.85 — хороший баланс скорость/стабильность
        super().__init__(name=name)
        self.alpha = alpha
        self.prev_smoothed: np.ndarray | None = None

    def process(self, context: PipelineContext) -> PipelineContext:
        context.current_step = self.name

        # Если ландмарки ещё не нормализованы — пропускаем
        if not hasattr(context, 'normalized_landmarks') or context.normalized_landmarks is None:
            self.prev_smoothed = None
            context.smoothed_landmarks = None
            return context

        current = np.asarray(context.normalized_landmarks, dtype=np.float32)

        if self.prev_smoothed is None:
            smoothed = current.copy()
        else:
            # EMA-сглаживание
            smoothed = self.alpha * current + (1 - self.alpha) * self.prev_smoothed

        context.smoothed_landmarks = smoothed
        self.prev_smoothed = smoothed.copy()

        return context