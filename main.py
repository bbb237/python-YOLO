from ultralytics import YOLO

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


# Load pretrained model
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

# Export the model to ONNX format
path = model.export(format="onnx")  # return path to exported model


# # Use the model on coco8 dataset
# results = model.train(data='coco8.yaml', epochs=5, device="cpu")  # train the model
# results = model.val()  # evaluate model performance on the validation set
# results = model('/Users/kalea/Desktop/skyeyez-ai/Dalmatian1.jpg')  # predict on an image
# results = model.export(format='onnx')  # export the model to ONNX format