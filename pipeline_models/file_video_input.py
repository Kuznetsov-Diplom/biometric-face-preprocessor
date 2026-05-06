from collections.abc import Iterator
from typing import Tuple

import cv2
import numpy as np

from .interfaces.video_input import VideoInput


class FileVideoInput(VideoInput):
    """Читает видео из файла (mp4, avi, mov и т.д.)."""

    def __init__(self, file_path: str):
        super().__init__(source=file_path)
        self.file_path = file_path

    def get_video(self) -> Iterator[Tuple[int, np.ndarray]]:
        """Генератор кадров из видеофайла."""
        self._cap = cv2.VideoCapture(self.file_path)

        if not self._cap.isOpened():
            raise FileNotFoundError(f"Не удалось открыть видео: {self.file_path}")

        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"📹 Открыт файл: {self.file_path}")
        print(
            f"   Разрешение: {int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))}×"
            f"{int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
        )
        print(f"   FPS: {self._fps:.2f}, кадров: {self._frame_count}")

        frame_idx = 0
        while self._cap.isOpened():
            ret, frame = self._cap.read()
            if not ret:
                break

            yield frame_idx, frame
            frame_idx += 1

        self._cap.release()
        self._cap = None
        print(f"✅ Видео завершено ({frame_idx} кадров прочитано)")
