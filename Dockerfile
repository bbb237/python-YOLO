# Use an official Python runtime as a parent image
FROM python:3.12.3-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir torch torchvision requests opencv-python pillow ultralytics

# Run the application
CMD ["python", "main.py"]
