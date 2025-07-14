from ultralytics import YOLO
import cv2
import math

# Load a pre-trained YOLOv8n model
# The model will be downloaded automatically the first time you run this.
model = YOLO("yolov8n.pt")

# Object classes (for COCO dataset, which yolov8n is trained on)
classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

# Start webcam
cap = cv2.VideoCapture(0)  # 0 is usually the default webcam. Change if you have multiple.
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit the webcam feed.")

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame.")
        break

    # Perform object detection on the current frame
    # stream=True processes the video stream more efficiently
    results = model(img, stream=True)

    # Process results
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Confidence score
            confidence = math.ceil((box.conf[0] * 100)) / 100
            print(f"Confidence: {confidence}")

            # Class name
            cls = int(box.cls[0])
            class_name = classNames[cls]
            print(f"Class Name: {class_name}")

            # Put class name and confidence on the image
            org = (x1, y1 - 10)  # Position for text
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.8
            color = (255, 255, 255)  # White text
            thickness = 2
            cv2.putText(img, f"{class_name} {confidence}", org, font, fontScale, color, thickness, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('YOLO Object Detection', img)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()