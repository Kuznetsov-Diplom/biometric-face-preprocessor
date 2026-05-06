from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class PipelineContext:
    """Объект, который путешествует по всем шагам пайплайна"""

    frame_idx: int
    frame: np.ndarray  # BGR изображение

    # Результаты работы предыдущих шагов
    face_bbox: Optional[tuple[int, int, int, int]] = None
    landmarks: Optional[dict] = None
    cropped_face: Optional[np.ndarray] = None
    normalized_landmarks: Optional[np.ndarray] = None
    features: Optional[np.ndarray] = None

    # Новые поля специально под демо-сайт (пока заполняются детектором)
    confidence: float = 0.0
    inter_pupil_distance: float = 0.0  # будет заполняться в следующем шаге
    roi_size: tuple[int, int] = (0, 0)
    stabilization: float = 0.0  # 0-100%
    current_step: str = ""

    # Дополнительные метаданные
    metadata: dict[str, Any] = field(default_factory=dict)
