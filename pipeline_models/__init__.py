# Импортируем всё, что хотим сделать публичным
from .file_video_input import FileVideoInput
from .camera_video_input import CameraVideoInput

# Абстрактный класс тоже можно вынести на верхний уровень
from .interfaces.video_input import VideoInput

# Это список того, что будет доступно при from pipeline_models import *
__all__ = [
    "FileVideoInput",
    "CameraVideoInput",
    "VideoInput",
]