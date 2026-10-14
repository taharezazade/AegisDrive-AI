import cv2

from src.video.reader import VideoReader
from src.lanes.lane_detector import LaneDetector


video = VideoReader("assets/sample_videos/input.mp4")
detector = LaneDetector()

ok, frame = video.read()

if not ok:
    raise RuntimeError("Video frame could not be read")

mask, lines = detector.detect(frame)
result = detector.draw_lanes(frame, lines)

cv2.imwrite("outputs/tests/lane_detection.jpg", result)
cv2.imwrite("outputs/tests/lane_mask.jpg", mask)

video.release()

print("Lane test completed")
print("Detected lines:", 0 if lines is None else len(lines))