import csv
from ParqSpotChecker import ParqSpotChecker

camera_ids = [
    "ed12f2da-9f65-44f0-a162-123c412716c6",
    "6fc602cf-72c5-40cc-9a42-9a844de7bc1a",
    "63e79f0b-7dea-4c8e-864c-f3315f9cc62c",
    "07b8616e-373e-4ec9-89cc-11cad7d59fcb",
    "ebec9de9-3f56-477a-a413-2e0a09b2b6ba",
    "7ae8f58a-84e9-4c9f-9e90-daa2dc548f20",
    "547cd268-58f9-4a84-a235-dbaa0432d79a"
]

# Will store all (image_id, x, y) triples here
all_open_points = []

for camera_id in camera_ids:
    print(f"\n📸 Processing camera: {camera_id}")
    try:
        checker = ParqSpotChecker(camera_id=camera_id, folder="Opening", conf_thresh=0.3)
        checker.fetch_live_image()
        checker.run_yolo()

        checker.draw_yolo_boxes("Click Open Spots (Avoid Cars)")
        points = checker.manual_label_open_spots()

        for (x, y) in points:
            all_open_points.append((camera_id, x, y))

    except Exception as e:
        print(f"⚠️ Skipping camera {camera_id} due to error: {e}")

# Save all collected points to a new CSV file
output_csv = "manual_open_spots.csv"
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_id", "x", "y"])  # Header
    writer.writerows(all_open_points)

print(f"\n✅ All open spot points saved to {output_csv}")
