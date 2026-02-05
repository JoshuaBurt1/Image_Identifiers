import numpy as np
import os
from PIL import Image
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

folder_path = 'C:/Users/jburt/Desktop/Resources/imageIdentifiers/Image_Identifier_Comparison/testImages'

image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]

model = load_model('mushroom_model.h5')


# Load class indices mapping
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)

# Reverse mapping to get index → class name
index_to_class = {v: k for k, v in class_indices.items()}


for img_name in image_files:
    img_path = os.path.join(folder_path, img_name)
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    predicted_class_name = index_to_class[predicted_class]
    confidence = predictions[0][predicted_class]


    print(f"Image: {img_name}")
    print(f"Predicted class: {predicted_class_name} (index: {predicted_class})")
    print(f"Confidence: {confidence:.5f}")
    print('-' * 40)