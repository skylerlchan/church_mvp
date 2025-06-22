import os
import cv2
import pandas as pd
import numpy as np
import requests
from io import BytesIO
from ultralytics import YOLO

class ParqSpotChecker:
    def __init__(self, camera_id, model_path='yolov8m.pt', folder='Opening', conf_thresh=0.5):
        self.camera_id = camera_id
        self.model = YOLO(model_path)
        self.folder = folder
        self.conf_thresh = conf_thresh
        self.image = None
        self.annotated_image = None
        self.car_boxes = []
        self.parking_points = []
        os.makedirs(folder, exist_ok=True)

    def fetch_live_image(self):
        url = f"https://webcams.nyctmc.org/api/cameras/{self.camera_id}/image"
        r = requests.get(url)
        if r.status_code != 200:
            raise Exception(f"❌ Could not fetch image from: {url}")
        img_bytes = np.frombuffer(BytesIO(r.content).read(), np.uint8)
        self.image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        return self.image

    def run_yolo(self):
        results = self.model(self.image)
        self.car_boxes = []
        for box in results[0].boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = box
            if int(cls) == 2 and conf >= self.conf_thresh:
                self.car_boxes.append((x1, y1, x2, y2, conf))
        return self.car_boxes

    def load_parking_points(self, csv_path):
        df = pd.read_csv(csv_path)
        df = df[df['image_id'] == self.camera_id]
        self.parking_points = df[['x', 'y']].values.tolist()
        return self.parking_points

    def check_open_spots(self):
        open_count = 0
        img_copy = self.image.copy()

        def point_inside_box(pt, box):
            x, y = pt
            x1, y1, x2, y2, _ = box
            return x1 <= x <= x2 and y1 <= y <= y2

        for idx, pt in enumerate(self.parking_points):
            match = next((b for b in self.car_boxes if point_inside_box(pt, b)), None)
            occupied = match is not None
            color = (0, 255, 0) if not occupied else (0, 0, 255)
            label = f"{idx+1}: {'Open' if not occupied else 'Taken'}"
            if occupied:
                label += f" ({match[4]:.2f})"
            else:
                open_count += 1
            cv2.circle(img_copy, pt, 8, color, -1)
            cv2.putText(img_copy, label, (pt[0] + 5, pt[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        self.annotated_image = img_copy
        return open_count, len(self.parking_points)

    def show_result(self, title="Result"):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(self.annotated_image, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
        plt.show()

    def save_annotated_image(self):
        out_path = os.path.join(self.folder, f"{self.camera_id}_live_output.jpg")
        cv2.imwrite(out_path, self.annotated_image)
        print(f"✅ Saved annotated image to: {out_path}")

    def manual_label_open_spots(self, resized_width=800):
        print("🟢 Click open parking spots. Press 'q' to finish, 'r' to reset.")
        img = self.annotated_image.copy()
        h, w = img.shape[:2]
        scale = resized_width / w
        resized = cv2.resize(img, (resized_width, int(h * scale)))
        clone = resized.copy()
        clicked = []

        def click_handler(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                cv2.circle(resized, (x, y), 5, (0, 255, 0), -1)
                clicked.append((x, y))
                cv2.imshow("Manual Label", resized)

        cv2.namedWindow("Manual Label")
        cv2.setMouseCallback("Manual Label", click_handler)

        while True:
            cv2.imshow("Manual Label", resized)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                resized = clone.copy()
                clicked = []

        cv2.destroyAllWindows()
        original_points = [(int(x / scale), int(y / scale)) for (x, y) in clicked]
        return original_points

    def save_manual_points(self, points, csv_name='parking_points.csv'):
        csv_path = os.path.join(self.folder, csv_name)
        df_new = pd.DataFrame(points, columns=["x", "y"])
        df_new.insert(0, "image_id", self.camera_id)
        if os.path.exists(csv_path):
            df_existing = pd.read_csv(csv_path)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new
        df_combined.to_csv(csv_path, index=False)
        print(f"✅ Saved {len(df_new)} points to {csv_path}")

    def draw_yolo_boxes(self, title="YOLO Car Detections"):
        img = self.image.copy()
        for i, (x1, y1, x2, y2, conf) in enumerate(self.car_boxes):
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            label = f"Car {i+1} ({conf:.2f})"
            cv2.putText(img, label, (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        self.annotated_image = img
        self.show_result(title)

