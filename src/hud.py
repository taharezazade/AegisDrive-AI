import cv2
import numpy as np


class ADASVisualizer:
    def draw_vehicles(self, frame, detections):
        output = frame.copy()

        for item in detections:
            x1, y1, x2, y2 = item["bbox"]
            name = item["class_name"]
            confidence = item["confidence"]

            # رنگ متفاوت برای کامیون و خودرو
            if name == "truck":
                color = (0, 165, 255)
                label = "TRUCK"
            elif name == "bus":
                color = (255, 0, 255)
                label = "BUS"
            elif name == "motorcycle":
                color = (255, 255, 0)
                label = "MOTORCYCLE"
            else:
                color = (0, 255, 0)
                label = "CAR"

            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            text = f"{label} {confidence:.2f}"
            cv2.putText(
                output,
                text,
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

        return output

    def draw_lane_lines(self, frame, lines):
        output = frame.copy()

        if lines is None:
            return output

        for line in lines:
            x1, y1, x2, y2 = map(int, line[0])
            cv2.line(output, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return output

    def draw_path(self, frame, left_points, right_points):
        output = frame.copy()

        if len(left_points) < 2 or len(right_points) < 2:
            return output

        left = np.array(left_points, dtype=np.int32)
        right = np.array(right_points, dtype=np.int32)

        # ساخت محدوده مسیر
        polygon = np.vstack((left, right[::-1]))

        overlay = output.copy()
        cv2.fillPoly(overlay, [polygon], (255, 120, 0))
        output = cv2.addWeighted(overlay, 0.28, output, 0.72, 0)

        # فلش جهت حرکت
        height, width = frame.shape[:2]
        start = (width // 2, height - 40)
        end = (width // 2, int(height * 0.68))

        cv2.arrowedLine(
            output,
            start,
            end,
            (255, 180, 0),
            6,
            tipLength=0.3,
        )

        return output