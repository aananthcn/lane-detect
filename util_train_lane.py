from ultralytics import YOLO

# Load the base segmentation model
model = YOLO("yolo11n-seg.pt")

# Train the model using your downloaded dataset
# Point to the data.yaml file inside your 'Pavement-feature-2' folder
results = model.train(
    data="Pavement-feature-2/data.yaml", 
    epochs=50, 
    imgsz=640
)