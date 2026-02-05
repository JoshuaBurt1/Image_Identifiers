# predict_console.py
import torch
import torch.nn as nn
from torchvision import transforms, models
import os
from PIL import Image
import numpy as np

# CONSOLE PREDICTION CODE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model
model = models.resnet18(weights=None)

# Load class names and model weights
checkpoint = torch.load('class_names.pth', map_location=device)
class_names = checkpoint['class_names']
num_classes = len(class_names)

# Modify final layer to match training
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model = model.to(device)
model.eval()

# Prediction transform (same as validation transform)
predict_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# change the path according to your setup
folder_path = 'C:/Users/jburt/Desktop/Resources/imageIdentifiers/Image_Identifier_Comparison/testImages'

image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]

# Batch processing for faster predictions
batch_size = 16
images_batch = []
names_batch = []

for i, img_name in enumerate(image_files):
    img_path = os.path.join(folder_path, img_name)
    img = Image.open(img_path).convert('RGB')
    img_tensor = predict_transform(img)
    
    images_batch.append(img_tensor)
    names_batch.append(img_name)
    
    # Process batch when full or at the end
    if len(images_batch) == batch_size or i == len(image_files) - 1:
        batch_tensor = torch.stack(images_batch).to(device)
        
        with torch.no_grad():
            outputs = model(batch_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            for j, (probs, img_name) in enumerate(zip(probabilities, names_batch)):
                probs_list = probs.cpu().numpy().tolist()
                predicted_idx = np.argmax(probs_list)
                predicted_class = class_names[predicted_idx]
                confidence = probs_list[predicted_idx]
                
                print(f"Image: {img_name}")
                print("Class names:", class_names)
                print("Probabilities:", [f"{p:.4f}" for p in probs_list])
                print(f"Predicted class: {predicted_class} (confidence: {confidence:.4f})")
                print('-' * 40)
        
        # Reset batch
        images_batch = []
        names_batch = []