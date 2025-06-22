import os
import cv2
import requests
import pandas as pd
import numpy as np
import folium
from io import BytesIO
from ultralytics import YOLO

# === CONFIG ===
CAMERA_IDS = [
    "ed12f2da-9f65-44f0-a162-123c412716c6",
    "6fc602cf-72c5-40cc-9a42-9a844de7bc1a",
    "07b8616e-373e-4ec9-89cc-11cad7d59fcb",
    "ebec9de9-3f56-477a-a413-2e0a09b2b6ba",
    "7ae8f58a-84e9-4c9f-9e90-daa2dc548f20",
    "547cd268-58f9-4a84-a235-dbaa0432d79a"
]

POINTS_CSV = "manual_open_spots.csv"
CAMERA_CSV = "nyc_cameras_full.csv"
MODEL_PATH = "yolov8m.pt"
OUTPUT_FOLDER = "annotated_output"
MAP_PATH = "index.html"

# === Load model and data
model = YOLO(MODEL_PATH)
spot_df = pd.read_csv(POINTS_CSV)
camera_df = pd.read_csv(CAMERA_CSV, encoding="ISO-8859-1")
camera_df = camera_df[camera_df["uuid"].isin(CAMERA_IDS)]

# === Helper Functions
def fetch_image(camera_id):
    url = f"https://webcams.nyctmc.org/api/cameras/{camera_id}/image"
    r = requests.get(url)
    if r.status_code != 200:
        raise Exception(f"❌ Failed to fetch image: {camera_id}")
    img_bytes = np.frombuffer(BytesIO(r.content).read(), np.uint8)
    return cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

def point_in_box(point, box):
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2

def check_open_spots(camera_id, img, spots):
    results = model(img)
    car_boxes = []
    for box in results[0].boxes.data.cpu().numpy():
        x1, y1, x2, y2, conf, cls = box
        if int(cls) == 2:
            car_boxes.append((x1, y1, x2, y2))

    statuses = []
    img_copy = img.copy()
    for i, pt in enumerate(spots):
        is_taken = any(point_in_box(pt, b) for b in car_boxes)
        status = "Taken" if is_taken else "Open"
        statuses.append((i + 1, status))
        color = (0, 0, 255) if is_taken else (0, 255, 0)
        cv2.circle(img_copy, pt, 6, color, -1)
        cv2.putText(img_copy, f"{i+1}", (pt[0]+5, pt[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return statuses, img_copy

# === Process + annotate all cameras
summary = []
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

for cam_id in CAMERA_IDS:
    print(f"\n📸 Processing {cam_id}")
    try:
        points = spot_df[spot_df['image_id'] == cam_id][['x', 'y']].values.tolist()
        if not points:
            print("⚠️ No points found.")
            continue

        img = fetch_image(cam_id)
        statuses, annotated_img = check_open_spots(cam_id, img, points)

        open_count = sum(1 for _, s in statuses if s == "Open")
        total = len(statuses)
        out_path = os.path.join(OUTPUT_FOLDER, f"{cam_id}.jpg")
        cv2.imwrite(out_path, annotated_img)

        summary.append({
            "id": cam_id,
            "open": open_count,
            "total": total,
            "image_path": out_path
        })

        print(f"✅ {open_count}/{total} open — saved to {out_path}")

    except Exception as e:
        print(f"❌ Skipped {cam_id}: {e}")

# === Create map
map_center = [camera_df["latitude"].mean(), camera_df["longitude"].mean()]
m = folium.Map(location=map_center, zoom_start=13, tiles="CartoDB positron")

for _, row in camera_df.iterrows():
    cam_id = row["uuid"]
    cam_summary = next((s for s in summary if s["id"] == cam_id), None)
    if not cam_summary:
        continue

    image_filename = os.path.basename(cam_summary["image_path"])
    popup_html = f"""
    <strong>{row.get('name', 'Unknown Camera')}</strong><br>
    {cam_summary['open']} Open / {cam_summary['total']} Spots<br>
    <img src='{OUTPUT_FOLDER}/{image_filename}' width="300px">
    """

    folium.Marker(
        location=[row["latitude"], row["longitude"]],
        popup=folium.Popup(popup_html, max_width=320),
        icon=folium.Icon(color="green" if cam_summary["open"] > 0 else "red", icon="camera", prefix="fa")
    ).add_to(m)

# === Save final map
m.save(MAP_PATH)
print(f"\n🗺️ Map saved to: {MAP_PATH}")
