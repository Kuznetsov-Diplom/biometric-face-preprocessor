from dataclasses import dataclass, field
from typing import Optional, Any
import numpy as np

@dataclass
class PipelineContext:
    """Объект, который путешествует по всем шагам пайплайна"""
    frame_idx: int
    frame: np.ndarray                     # BGR изображение

    # Результаты работы предыдущих шагов
    face_bbox: Optional[tuple[int, int, int, int]] = None
    landmarks: Optional[dict] = None
    cropped_face: Optional[np.ndarray] = None
    normalized_landmarks: Optional[np.ndarray] = None
    features: Optional[np.ndarray] = None

    # Дополнительные метаданные
    metadata: dict[str, Any] = field(default_factory=dict)