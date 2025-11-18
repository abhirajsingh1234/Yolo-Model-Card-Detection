from ultralytics import YOLO
import os

# Path to your dataset
dataset_path = r"C:\Users\ACER\OneDrive\डेस्कटॉप\E-Card Detection\training_Data\Sumit_ka_Kaam.v1-version_1.yolov8\data.yaml"

# Choose model size: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
# Recommendation: Start with 'n' for fast training, use 's' or 'm' for better accuracy
model_size = 'n'

# Load a pretrained YOLOv8 model
model = YOLO(f'yolov8{model_size}.pt')

# Train the model
results = model.train(
    data=dataset_path,           # Path to data.yaml
    epochs=50,                   # Number of training epochs (adjust as needed)
    imgsz=640,                    # Image size (640 is standard)
    batch=4,                     # Batch size (reduce if you get memory errors)
    patience=20,                  # Early stopping patience
    save=True,                    # Save training checkpoints
    device='cpu',                   # Use GPU 0 (use 'cpu' if no GPU available)
    workers=2,                    # Number of worker threads
    project='card_detection',     # Project name
    name='exp',                   # Experiment name
    exist_ok=True,                # Overwrite existing project
    pretrained=True,              # Use pretrained weights
    optimizer='auto',             # Optimizer (auto, SGD, Adam, AdamW)
    verbose=True,                 # Verbose output
    seed=42,                      # Random seed for reproducibility
    deterministic=True,           # Deterministic mode
    single_cls=False,             # Train as single-class (use if only one card type)
    rect=False,                   # Rectangular training
    cos_lr=False,                 # Cosine learning rate scheduler
    close_mosaic=10,              # Disable mosaic augmentation for final epochs
    resume=False,                 # Resume training from last checkpoint
    amp=True,                     # Automatic Mixed Precision training
    fraction=1.0,                 # Dataset fraction to train on (1.0 = all data)
    profile=False,                # Profile ONNX and TensorRT speeds
    freeze=None,                  # Freeze first n layers, or freeze list of layers
    # Learning rate settings
    lr0=0.01,                     # Initial learning rate
    lrf=0.01,                     # Final learning rate (lr0 * lrf)
    momentum=0.937,               # SGD momentum/Adam beta1
    weight_decay=0.0005,          # Optimizer weight decay
    warmup_epochs=3.0,            # Warmup epochs
    warmup_momentum=0.8,          # Warmup initial momentum
    warmup_bias_lr=0.1,           # Warmup initial bias learning rate
    # Augmentation settings
    hsv_h=0.015,                  # HSV-Hue augmentation
    hsv_s=0.7,                    # HSV-Saturation augmentation
    hsv_v=0.4,                    # HSV-Value augmentation
    degrees=0.0,                  # Rotation (+/- deg)
    translate=0.1,                # Translation (+/- fraction)
    scale=0.5,                    # Scale (+/- gain)
    shear=0.0,                    # Shear (+/- deg)
    perspective=0.0,              # Perspective (+/- fraction)
    flipud=0.0,                   # Flip up-down (probability)
    fliplr=0.5,                   # Flip left-right (probability)
    mosaic=1.0,                   # Mosaic augmentation (probability)
    mixup=0.0,                    # Mixup augmentation (probability)
    copy_paste=0.0,               # Copy-paste augmentation (probability)
)

# Print training results
print("\n" + "="*50)
print("TRAINING COMPLETED!")
print("="*50)
print(f"Results saved to: {results.save_dir}")
print(f"Best model: {results.save_dir}/weights/best.pt")
print(f"Last model: {results.save_dir}/weights/last.pt")

# Validate the model
print("\n" + "="*50)
print("VALIDATING MODEL...")
print("="*50)
metrics = model.val()

# Print validation metrics
print(f"\nmAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")

# Export the model (optional)
print("\n" + "="*50)
print("EXPORTING MODEL...")
print("="*50)

# Export to ONNX format (for deployment)
model.export(format='onnx')
print("Model exported to ONNX format")

# Test on a sample image (optional)
# Uncomment the lines below to test on a sample image
# test_image = r"C:\Users\ACER\OneDrive\डेस्कटॉप\E-Card Detection\training_Data\Sumit_ka_Kaam.v1-version_1.yolov8\images\test\sample.jpg"
# results = model.predict(test_image, save=True, conf=0.5)
# print(f"Test results saved to: {results[0].save_dir}")

print("\n" + "="*50)
print("ALL DONE! 🎉")
print("="*50)