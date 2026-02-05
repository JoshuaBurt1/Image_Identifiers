This is an image identification model creator.

_train.py trains an image identification model (RUN THIS FIRST).
predict_console.py is used to test a model's accuracy using test images in testImages folder.

Modify the file paths according to your setup:
_train.py : modify the data path -> data='C:/Users/jimmy/Linux/rewrite/mushroom_dataset',
predict_console.py : modify the model path -> model = YOLO('C:/Users/jimmy/Linux/rewrite/runs/classify/train/weights/best.pt')  
                     modify the folder_path -> folder_path = 'C:/Users/jimmy/Linux/rewrite/testImages'

