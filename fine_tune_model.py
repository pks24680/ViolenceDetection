import os
import cv2
import torch
from transformers import VideoMAEForVideoClassification, AutoFeatureExtractor, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from datasets import IterableDataset
import numpy as np

def load_videos_from_folder(folder):
    videos = []
    labels = []
    for label in os.listdir(folder):
        label_folder = os.path.join(folder, label)
        for video_file in os.listdir(label_folder):
            video_path = os.path.join(label_folder, video_file)
            videos.append(video_path)
            labels.append(1 if label == "violence" else 0) 
    return videos, labels

def check_video_format(video_path):
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return False
    
    codec = int(video.get(cv2.CAP_PROP_FOURCC))
    codec_str = "".join([chr((codec >> 8 * i) & 0xFF) for i in range(4)])
    print(f"Video: {video_path}, Codec: {codec_str}")
    
    if codec_str != 'mp4v':  # Adjust this based on your needs (e.g., check for H.264 or other codecs)
        print(f"Warning: {video_path} may not be in the correct format.")
        return False
    
    video.release()
    return True

# Preprocess the video frames
def preprocess_video(video_path, feature_extractor, num_frames=16):
    video = cv2.VideoCapture(video_path)
    
    # Check if the video is opened successfully
    if not video.isOpened():
        print(f"Warning: {video_path} may not be in the correct format.")
        return None  # Return None to indicate failure to read the video
    
    frames = []
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame sampling rate to get `num_frames` evenly spaced frames
    sampling_rate = max(1, frame_count // num_frames)
    
    for i in range(num_frames):
        video.set(cv2.CAP_PROP_POS_FRAMES, i * sampling_rate)
        ret, frame = video.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, (224, 224))  # Resize frames to 224x224
        frames.append(frame_resized)
    
    video.release()
    
    if len(frames) < num_frames:
        print(f"Warning: {video_path} has fewer than {num_frames} frames. Padding with last frame.")
        # Repeat the last frame if there aren't enough frames
        while len(frames) < num_frames:
            frames.append(frames[-1])
    
    try:
        # Use feature extractor to preprocess frames
        inputs = feature_extractor(frames, return_tensors="pt")
        return inputs['pixel_values'] if 'pixel_values' in inputs else None
    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
        return None

# Custom Iterable Dataset
class VideoIterableDataset(IterableDataset):
    def __init__(self, video_paths, labels, feature_extractor):
        self.video_paths = video_paths
        self.labels = labels
        self.feature_extractor = feature_extractor
        self.current_epoch = 0  # Use a different name

    def set_epoch(self, epoch):
        self.current_epoch = epoch 

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, index):
        video_path = self.video_paths[index]
        label = self.labels[index]

        frames = preprocess_video(video_path, self.feature_extractor)
        if frames is None:
            return None  # Handle invalid videos gracefully
        
        return {
            'pixel_values': frames.squeeze(),  # Ensure correct shape
            'labels': torch.tensor(label, dtype=torch.long)
        }

    def __iter__(self):
        for video_path, label in zip(self.video_paths, self.labels):
            if not check_video_format(video_path):
                print(f"Skipping video {video_path} due to format issues.")
                continue

            frames = preprocess_video(video_path, self.feature_extractor)
            if frames is None:
                print(f"Skipping video {video_path} due to missing or invalid frames.")
                continue

            yield {'pixel_values': frames.squeeze(), 'labels': torch.tensor(label, dtype=torch.long)}

# Load dataset
videos, labels = load_videos_from_folder("dataset")

# Split the dataset into training and testing
train_videos, test_videos, train_labels, test_labels = train_test_split(videos, labels, test_size=0.2, random_state=42)

# Initialize the feature extractor
model_name = "MCG-NJU/videomae-base"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

# Create the custom dataset instances
train_dataset = VideoIterableDataset(train_videos, train_labels, feature_extractor)
eval_dataset = VideoIterableDataset(test_videos, test_labels, feature_extractor)

# Load the model for video classification
model = VideoMAEForVideoClassification.from_pretrained(model_name)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    evaluation_strategy="epoch",
    logging_dir='./logs',
    max_steps=525  # Adjust based on your dataset size and batch size
)

# Define the Trainer for fine-tuning
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Fine-tune the model
trainer.train()

# After training, save the model
trainer.save_model("./fine_tuned_videomae")

print("Model training and fine-tuning complete.")
