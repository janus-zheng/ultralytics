''' autodocstring '''
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)

# Train the model
model.train(data='/home/jupyter/data/color_dataset/deepfashion_mixed_tokopedia',
            epochs=100, imgsz=160)
