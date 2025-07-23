# qualcomm model:
# https://aihub.qualcomm.com/models/yolov8_seg

from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict, deque

# Load the pre-trained YOLOv8 segmentation model
model = YOLO('yolov8n-seg.pt')

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Dictionary to store the classification history for each tracked object ID
# key: track_id, value: deque of class names
track_history = defaultdict(lambda: deque(maxlen=5)) # Store history for the last 5 frames
# The deque acts as a sliding window to keep track of recent classifications

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break
    
    # Run object tracking with segmentation on the current frame
    results = model.track(source=frame, tracker="bytetrack.yaml", conf=0.25, iou=0.5, verbose=False)

    # Process the results
    for result in results:
        if result.masks is not None and result.boxes.id is not None:
            masks = result.masks.data
            boxes = result.boxes
            classes = result.names
            track_ids = result.boxes.id.cpu().numpy().astype(int)

            # Iterate through each detected object
            for i, (box, mask) in enumerate(zip(boxes, masks)):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # Get the detected class and the unique track ID
                detected_class_id = int(box.cls[0].cpu().numpy())
                detected_class_name = classes[detected_class_id]
                track_id = track_ids[i]

                # Update the classification history for this track ID
                track_history[track_id].append(detected_class_name)

                # Determine the most likely class based on the history
                # We'll use a simple "majority vote" over the last few frames
                if len(track_history[track_id]) > 0:
                    # Get the most common class in the history
                    most_common_class = max(set(track_history[track_id]), key=track_history[track_id].count)
                    final_class_name = most_common_class
                else:
                    final_class_name = detected_class_name

                # Get the confidence score
                score = float(box.conf[0].cpu().numpy())

                # Create a colored mask for the object
                mask_np = mask.cpu().numpy()
                mask_np = (mask_np * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Seed the random number generator with the track ID for consistent color
                np.random.seed(track_id)
                color = [int(c) for c in np.random.randint(0, 255, size=3)]
                
                overlay = np.zeros_like(frame)
                cv2.drawContours(overlay, contours, -1, color, -1)
                frame = cv2.addWeighted(frame, 1, overlay, 0.5, 0)
                
                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Put the class label and unique track ID
                label = f'{final_class_name} ID:{track_id} conf:{score:.2f}'
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the resulting frame
    cv2.imshow("YOLOv8 Segmentation & Tracking", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()