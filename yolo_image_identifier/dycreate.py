import os
import yaml

train_dir = 'C:/Users/jburt/Desktop/yolo_improved/mushroom_dataset/train'
val_dir = 'C:/Users/jburt/Desktop/yolo_improved/mushroom_dataset/val'

class_names = sorted(os.listdir(train_dir))

data_yaml = {
    'train': train_dir,
    'val': val_dir,
    'nc': len(class_names),
    'names': class_names
}

with open('data.yaml', 'w') as f:
    yaml.dump(data_yaml, f)

print("data.yaml created successfully!")
