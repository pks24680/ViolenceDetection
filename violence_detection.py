import cv2
import torch
import numpy as np
from transformers import AutoModelForVideoClassification, AutoFeatureExtractor

# Check if CUDA is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the model and feature extractor from Hugging Face
model_name = "MCG-NJU/videomae-large"  # Model name from Hugging Face
model = AutoModelForVideoClassification.from_pretrained(model_name).to(device)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

model.eval()

#(0 for webcam or provide a video file path)
video_capture = cv2.VideoCapture(0)

# Number of frames to use for prediction (typically set to 16 for VideoMAE)
frame_window_size = 16
frame_buffer = []
frame_skip = 10  # Skip more frames to reduce load
frame_count = 0  # Counter to track frames

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Skip frames to reduce processing load
    if frame_count % frame_skip == 0:
        # Resize the frame to a smaller size (e.g., 128x128) for faster processing
        frame_resized = cv2.resize(frame, (128, 128))
        frame_buffer.append(frame_resized)

    # Ensure buffer has exactly `frame_window_size` frames before running inference
    if len(frame_buffer) == frame_window_size:
        # Convert frame buffer to the format required by the feature extractor
        inputs = feature_extractor([frame_buffer], return_tensors="pt").to(device)

        # Run inference with the model
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=-1).item()

        # Clear the buffer after making the prediction
        frame_buffer = []

        # Display the result (adjust class labels as needed)
        action_label = "Violence Detected" if predicted_class == 1 else "No Violence"
        print(f"Action Label: {action_label}")

        # Add the label to the frame
        cv2.putText(frame, action_label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame with the label
    cv2.imshow("Violence Detection", frame)

    # Press 'q' to exit the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Increment frame counter
    frame_count += 1

video_capture.release()
cv2.destroyAllWindows()
