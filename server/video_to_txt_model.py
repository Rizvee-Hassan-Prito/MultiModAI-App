
from transformers import AutoImageProcessor, TimesformerForVideoClassification
import numpy as np
import torch
import cv2

def hugf_vid_to_txt_Model(video):
    
    processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k600")
    model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k600")

    # Read video and extract frames using OpenCV
    #video_path = ""
    cap = cv2.VideoCapture(video)

    frames = []
    frame_count = 0
    num_frames_to_extract = 16 

    # Read evenly spaced frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames_to_extract, dtype=int)

    for idx in frame_indices:
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break

    # Preprocess and predict
    inputs = processor(images=frames, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    predicted_class_idx = logits.argmax(-1).item()

    return  model.config.id2label[predicted_class_idx]
