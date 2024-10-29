import os
import cv2
import torch
from transformers import VideoMAEForVideoClassification, AutoFeatureExtractor
from sklearn.metrics import classification_report
from datasets import IterableDataset

def video_generator(folder, num_frames=16):
    for label in os.listdir(folder):
        label_folder = os.path.join(folder, label)
        for video_file in os.listdir(label_folder):
            video_path = os.path.join(label_folder, video_file)
            frames = []
            video = cv2.VideoCapture(video_path)
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            
            sampling_rate = max(1, frame_count // num_frames)

            for i in range(num_frames):
                video.set(cv2.CAP_PROP_POS_FRAMES, i * sampling_rate)
                ret, frame = video.read()
                if not ret:
                    break
                frame_resized = cv2.resize(frame, (224, 224))
                frames.append(frame_resized)
            
            video.release()

            if len(frames) < num_frames:
                print(f"Warning: {video_path} has fewer than {num_frames} frames. Padding with last frame.")
                while len(frames) < num_frames:
                    frames.append(frames[-1])

            label_value = 0 if label == "violence" else 1
            yield {"video": frames, "label": label_value, "path": video_path}

class VideoIterableDataset(IterableDataset):
    def __init__(self, folder):
        self.folder = folder

    def __iter__(self):
        return video_generator(self.folder)

test_folder = "dataset/test"
test_dataset = VideoIterableDataset(test_folder)
model_name_or_path = "./fine_tuned_videomae"
model = VideoMAEForVideoClassification.from_pretrained(model_name_or_path)

base_model_name = "MCG-NJU/videomae-base"
feature_extractor = AutoFeatureExtractor.from_pretrained(base_model_name)
model.eval()

correct_predictions = 0
total_predictions = 0
predicted_labels = []
true_labels = []

for idx, batch in enumerate(test_dataset):
    video_path = batch["path"]
    frames = batch["video"]
    true_label = batch["label"]

    inputs = feature_extractor(frames, return_tensors="pt", padding=True)

    inputs = {"pixel_values": inputs["pixel_values"]}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()

    predicted_labels.append(predicted_class)
    true_labels.append(true_label)

    if predicted_class == true_label:
        correct_predictions += 1
    total_predictions += 1

    print(f"Processed video {idx+1}: Predicted = {predicted_class}, Actual = {true_label}")

accuracy = correct_predictions / total_predictions
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(true_labels, predicted_labels, target_names=["Violence", "No Violence"]))
