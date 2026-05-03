from abc import ABC, abstractmethod
from .pipeline_context import PipelineContext

class PipelineStep(ABC):
    """Абстрактный обработчик (твой Handler)"""

    def __init__(self, **kwargs):
        self.params = kwargs

    @abstractmethod
    def process(self, context: PipelineContext) -> PipelineContext:
        """Основной метод — каждый шаг получает context и возвращает обновлённый"""
        pass

    def __call__(self, context: PipelineContext) -> PipelineContext:
        """Удобный синтаксис: step(context)"""
        return self.process(context)