import cv2
import numpy as np
import os

def test_opencv_installation():
    print(f"OpenCV Version: {cv2.__version__}")
    print("OpenCV installed successfully!")

def display_image_test():
    print("\n--- Image Display Test ---")
    # Create a simple black image with a white rectangle
    width, height = 400, 300
    image = np.zeros((height, width, 3), dtype=np.uint8) # Black image
    # Draw a white rectangle
    cv2.rectangle(image, (50, 50), (350, 250), (255, 255, 255), -1) # White, filled
    # Add some text
    cv2.putText(image, "Hello, OpenCV!", (70, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) # Green text

    image_filename = "test_image.png"
    cv2.imwrite(image_filename, image) # Save the image temporarily

    print(f"Generated a test image: {image_filename}")
    print("Displaying image. Press any key to close the window...")

    try:
        cv2.imshow("OpenCV Test Image", image)
        cv2.waitKey(0) # Wait indefinitely until a key is pressed
        cv2.destroyAllWindows()
        print("Image display test complete.")
    except Exception as e:
        print(f"Error during image display: {e}")
    finally:
        if os.path.exists(image_filename):
            os.remove(image_filename) # Clean up the temporary file
            print(f"Cleaned up {image_filename}")

def video_playback_test():
    print("\n--- Video Playback Test ---")
    # For this test, you'll need a sample video file.
    # Replace 'path/to/your/video.mp4' with an actual video file on your computer.
    # If you don't have one readily available, you can download a sample from the internet,
    # or skip this test for now.
    video_path = "path/to/your/video.mp4" # <--- IMPORTANT: Change this!

    if not os.path.exists(video_path):
        print(f"WARNING: Video file not found at '{video_path}'. Skipping video playback test.")
        print("Please update 'video_path' in the script to a valid video file on your system.")
        return

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'.")
        return

    print(f"Playing video: '{os.path.basename(video_path)}'. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream or error reading frame.")
                break

            # Optionally resize frame for display
            # frame = cv2.resize(frame, (640, 480))

            cv2.imshow("OpenCV Video Playback", frame)

            # Press 'q' to quit
            if cv2.waitKey(25) & 0xFF == ord('q'): # 25ms delay between frames
                break
        print("Video playback test complete.")
    except Exception as e:
        print(f"Error during video playback: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

def webcam_test():
    print("\n--- Webcam Test ---")
    print("Opening webcam. Press 'q' to quit.")
    # 0 refers to the default webcam. If you have multiple, try 1, 2, etc.
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam. Make sure your webcam is connected and not in use.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from webcam. Exiting...")
                break

            # You can add some simple processing here, e.g., grayscale
            # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # cv2.imshow("Webcam (Grayscale)", gray_frame)

            cv2.imshow("OpenCV Webcam Feed", frame)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        print("Webcam test complete.")
    except Exception as e:
        print(f"Error during webcam access: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_opencv_installation()

    print("\nStarting tests...")
    display_image_test()
    video_playback_test()
    webcam_test()
    print("\nAll tests attempted.")
    print("If windows appeared and closed as expected, OpenCV is likely working well!")