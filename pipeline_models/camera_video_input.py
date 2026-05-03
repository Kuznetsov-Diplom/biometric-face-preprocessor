from .interfaces.video_input import VideoInput

class CameraVideoInput(VideoInput):
    def __init__(self, camera_id: int = 0):
        super().__init__(source=camera_id)

    def get_video(self):
        print(f"Открываем камеру #{self.source}")
        # return cv2.VideoCapture(self.source)
        pass