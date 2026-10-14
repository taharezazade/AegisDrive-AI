import cv2

from src.video.reader import VideoReader
from src.detection.detector import ObjectDetector


video = VideoReader("assets/sample_videos/input.mp4")
detector = ObjectDetector()

ok, frame = video.read()

if not ok:
    raise RuntimeError("Could not read video frame")

result = detector.detect(frame)
detections = detector.get_detections(result)

for item in detections:
    x1, y1, x2, y2 = item["bbox"]
    label = f'{item["class_name"]} {item["confidence"]:.2f}'

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        frame,
        label,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

cv2.imwrite("outputs/tests/detection_test.jpg", frame)
video.release()

print(f"Detected {len(detections)} objects")
print("Image saved: outputs/tests/detection_test.jpg")