from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, Tuple

import numpy as np


class VideoInput(ABC):
    """Абстрактный базовый класс для источников видео (файл / камера)."""

    def __init__(self, source: Any):
        self.source = source
        self._cap = None  # VideoCapture будет здесь
        self._frame_count = 0
        self._fps = 0.0

    @abstractmethod
    def get_video(self) -> Iterator[Tuple[int, np.ndarray]]:
        """Возвращает генератор: (номер_кадра, кадр)"""
        pass

    def __iter__(self):
        """Чтобы можно было писать: for frame in video_input:"""
        return self.get_video()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._cap is not None:
            self._cap.release()
