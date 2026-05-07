import cv2

from .interfaces.pipeline_context import PipelineContext
from .interfaces.pipeline_step import PipelineStep


class FramePreprocessor(PipelineStep):
    """Шаг 1: Предобработка каждого кадра (по plan.md)
    • Конвертация в grayscale (рекомендуется)
    • CLAHE
    • Фильтрация (Gaussian)
    """

    def __init__(self,
                 name: str = "frame_preprocessing",
                 use_grayscale: bool = True,
                 clahe_clip_limit: float = 2.0,
                 clahe_tile_grid_size: tuple[int, int] = (8, 8),
                 use_gaussian_filter: bool = True,
                 gaussian_kernel_size: tuple[int, int] = (5, 5)):
        super().__init__(name=name,
                         use_grayscale=use_grayscale,
                         clahe_clip_limit=clahe_clip_limit,
                         clahe_tile_grid_size=clahe_tile_grid_size,
                         use_gaussian_filter=use_gaussian_filter,
                         gaussian_kernel_size=gaussian_kernel_size)
        self.use_grayscale = use_grayscale
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
        self.use_gaussian_filter = use_gaussian_filter
        self.gaussian_kernel_size = gaussian_kernel_size

        # CLAHE создаём в __init__ (как FaceDetection в медиапайпе)
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_grid_size
        )

    def process(self, context: PipelineContext) -> PipelineContext:
        """Основной метод — предобработка кадра, обновляет context.frame"""
        context.current_step = self.name

        if context.frame is None:
            return context

        frame = context.frame.copy()

        # 1. Конвертация в grayscale (если нужно)
        if self.use_grayscale and len(frame.shape) == 3:
            processed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            processed = frame

        # 2. CLAHE
        if len(processed.shape) == 2:  # grayscale
            processed = self.clahe.apply(processed)
        else:
            # Для BGR — CLAHE только на L-канале LAB
            lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = self.clahe.apply(l)
            lab = cv2.merge([l, a, b])
            processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # 3. Gaussian фильтр
        if self.use_gaussian_filter:
            processed = cv2.GaussianBlur(processed, self.gaussian_kernel_size, 0)

        # Если grayscale — возвращаем в BGR (совместимость с детектором и оверлеем)
        if self.use_grayscale and len(processed.shape) == 2:
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

        context.frame = processed
        context.stabilization = 100.0  # placeholder (видео-стабилизация позже)

        return context