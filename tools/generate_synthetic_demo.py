import os
import csv
import cv2
import numpy as np

os.makedirs("synthetic_demo/images", exist_ok=True)

rows = []
for i in range(5):
    img = np.zeros((256,256,3), dtype=np.uint8)
    cv2.rectangle(img, (50,50), (200,200), (0,255,0), -1)
    
    filename = f"roof_{i}.jpg"
    cv2.imwrite(f"synthetic_demo/images/{filename}", img)

    rows.append([f"id_{i}", "12.9000", "77.6000", filename])

with open("synthetic_demo/test_rooftop_data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id","lat","long","rooftop_image"])
    writer.writerows(rows)

print("Synthetic data created!")
