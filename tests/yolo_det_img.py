from ultralytics import YOLO
import cv2

# Load a pre-trained YOLOv8 detection model
# 'yolov8n.pt' is the nano version, which is fast and lightweight.
model = YOLO('yolov8n.pt')

# Path to your input image
image_path = 'img/example_car_person.jpg'

# Run inference on the image
# The model will automatically download if it's not present.
results = model(image_path)

# Show the results
# The .show() method from ultralytics will display the image with bounding boxes
# and labels drawn on it.
results[0].show()


# for example_car_person.jpg the result should be
# image 1/1 /home/gr_intern/yolo_test/img/example_car_person.jpg: 480x640 1 person, 2 cars, 1 truck, 29.8ms