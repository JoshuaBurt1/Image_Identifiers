from ultralytics import YOLO
import numpy as np
import os
from PIL import Image

model = YOLO('C:/Users/jburt/Desktop/Resources/imageIdentifiers/Image_Identifier_Comparison/yolo_image_identifier/runs/classify/train6/weights/best.pt')  # load a custom model
folder_path = 'C:/Users/jburt/Desktop/Resources/imageIdentifiers/Image_Identifier_Comparison/testImages'
image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]

for img_name in image_files:
    img_path = os.path.join(folder_path, img_name)
    img = Image.open(img_path)

    results = model(img)
    names_dict = results[0].names
    probs = results[0].probs.data.tolist()

    # Get top 5 predictions (index, prob)
    top5 = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:5]

    print(f"\n🖼️ Image: {img_name}")
    for idx, prob in top5:
        class_name = names_dict[idx]
        print(f"{class_name:<40} → {prob:.4f}")
