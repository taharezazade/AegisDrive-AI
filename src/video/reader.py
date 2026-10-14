import cv2


class VideoReader:
    def __init__(self, video_path):
        self.video_path = video_path
        self.capture = cv2.VideoCapture(video_path)

        if not self.capture.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

    def read(self):
        return self.capture.read()

    def get_fps(self):
        return self.capture.get(cv2.CAP_PROP_FPS)

    def get_size(self):
        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height

    def release(self):
        self.capture.release()