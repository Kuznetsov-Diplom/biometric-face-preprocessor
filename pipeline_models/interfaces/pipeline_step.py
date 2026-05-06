from abc import ABC, abstractmethod

from .pipeline_context import PipelineContext


class PipelineStep(ABC):
    """Абстрактный обработчик (твой Handler)"""

    def __init__(self, name: str = "unnamed_step", **kwargs):
        self.name = name
        self.params = kwargs

    @abstractmethod
    def process(self, context: PipelineContext) -> PipelineContext:
        """Основной метод — каждый шаг получает context и возвращает обновлённый"""
        pass

    def __call__(self, context: PipelineContext) -> PipelineContext:
        """Удобный синтаксис: step(context)"""
        return self.process(context)

    def update_params(self, **new_params):
        """Динамическая смена параметров для демо-сайта"""
        self.params.update(new_params)
