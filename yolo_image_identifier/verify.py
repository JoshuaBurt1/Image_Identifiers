import os

label_dirs = [
    'C:/Users/jburt/Desktop/yolo_improved/mushroom_dataset/train',
    'C:/Users/jburt/Desktop/yolo_improved/mushroom_dataset/val'
]

nc = 200  # number of classes

for label_dir in label_dirs:
    for root, _, files in os.walk(label_dir):
        for file in files:
            if file.endswith('.txt'):
                with open(os.path.join(root, file), 'r') as f:
                    for line in f:
                        class_idx = int(line.split()[0])
                        if class_idx < 0 or class_idx >= nc:
                            print(f"Invalid class index {class_idx} in file {os.path.join(root, file)}")