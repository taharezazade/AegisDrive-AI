from ultralytics import YOLO


class ObjectDetector:
    def __init__(
        self,
        model_path="yolo11n.pt",
        confidence=0.35,
        iou=0.45,
        device="cpu",
    ):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.iou = iou
        self.device = device

    def detect(self, frame):
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )

        return results[0]

    @staticmethod
    def get_detections(result):
        detections = []

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": float(box.conf[0]),
                "class_id": int(box.cls[0]),
                "class_name": result.names[int(box.cls[0])],
            })

        return detections