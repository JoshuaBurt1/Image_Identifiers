from ultralytics import YOLO
import torch
import time
import multiprocessing


def main():
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model
    model = YOLO('yolo11m-cls.pt')

    # Train model
    model.train(
        data='C:/Users/jburt/Desktop/yolo_improved/mushroom_dataset',
        epochs=100,
        imgsz=224,
        batch=16,
        lr0=0.01,
        patience=3, #this is for early stopping if lack of progress
        save_period=10,
        device=device,
        augment=True,
        mixup=0.1,
        copy_paste=0.1,
        plots=True,
        val=True,
        cache=True
    )

    # Validate
    metrics = model.val()
    print(f"Validation accuracy: {metrics.top1}")

    # Export
    model.export(format='onnx')

if __name__ == "__main__":
    start_time = time.time()  # Start timer
    multiprocessing.freeze_support()  # Optional but safe
    main()
    end_time = time.time()  # End timer
    total_time = end_time - start_time
    print(f"Training completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes).")