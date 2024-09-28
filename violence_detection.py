import cv2
import torch
import numpy as np
from transformers import AutoModelForVideoClassification, AutoFeatureExtractor

# Check if CUDA is available and set device - this line checks for GPU if available. Otherwise uses CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the model and feature extractor from Hugging Face
model_name = "MCG-NJU/videomae-large"
model = AutoModelForVideoClassification.from_pretrained(model_name).to(device)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

model.eval()

#(0 for webcam or provide a video file path)
video_capture = cv2.VideoCapture(0)

# Number of frames to use for prediction - considering CPU used, accuracy is compensated for speed of inference
frame_window_size = 16
frame_buffer = []
frame_skip = 10  # Skip more frames to reduce load
frame_count = 0

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    if frame_count % frame_skip == 0:
        # Resize the frame to a smaller size (e.g., 128x128) for faster processing
        frame_resized = cv2.resize(frame, (128, 128))
        frame_buffer.append(frame_resized)

    if len(frame_buffer) == frame_window_size:
        inputs = feature_extractor([frame_buffer], return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=-1).item()

        # Clear the buffer after making the prediction
        frame_buffer = []

        action_label = "Violence Detected" if predicted_class == 1 else "No Violence"
        print(f"Action Label: {action_label}")

        cv2.putText(frame, action_label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Violence Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

video_capture.release()
cv2.destroyAllWindows()
