from abc import ABC, abstractmethod
from typing import Iterator, Tuple, Optional
import numpy as np

class FaceDetector(ABC):
    """Абстрактный класс для детекции лица."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Принимает кадр (BGR), возвращает (x, y, w, h) — bounding box лица
        Если лицо не найдено — возвращает None
        """
        pass

    @abstractmethod
    def detect_with_landmarks(self, frame: np.ndarray) -> Optional[dict]:
        """
        Возвращает более полную информацию:
        - bounding box
        - confidence
        - landmarks (если есть)
        """
        pass