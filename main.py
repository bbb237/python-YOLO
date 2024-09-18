# from ultralytics import YOLO

# import requests
# import zipfile
# import os

# # Example function to download a file
# def download_file(url, output_path):
#     response = requests.get(url)
#     with open(output_path, 'wb') as file:
#         file.write(response.content)

# # URL and output path will be inputs
# url = ''  # Replace this with the actual URL input
# output_path = 'images.zip'
# download_file(url, output_path)


# # Load a pretrained YOLO model
# model = YOLO("yolov8n.pt")

# Download COCO val
import torch
torch.hub.download_url_to_file('https://ultralytics.com/assets/coco2017val.zip', 'tmp.zip')  # download (780M - 5000 images)
!unzip -q tmp.zip -d datasets && rm tmp.zip  # unzip

from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")

# Train the model
results = model.train(
    data="config.yaml",  # path to dataset YAML
    epochs=5,  # number of training epochs
    imgsz=640,  # training image size
    device="cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("/Users/kalea/Desktop/skyeyez-ai/Serena_Williams_at_2013_US_Open.jpg")
results[0].show()

# # Export the model to ONNX format
# path = model.export(format="onnx")  # return path to exported model

# # Use the model
# results = model.train(data='coco8.yaml', epochs=3)  # train the model
# results = model.val()  # evaluate model performance on the validation set
# results = model('https://ultralytics.com/images/bus.jpg')  # predict on an image
# results = model.export(format='onnx')  # export the model to ONNX format