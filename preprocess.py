import numpy as np
import torch
from transformers import BertTokenizer, BertModel

def preprocess_text(text_file):
    try:
        # Read the text file content
        file_extension = text_file.filename.split('.')[-1]
        text_data = text_file.read().decode("utf-8")

        if file_extension == "textonly":
            # Handle the `.textonly` file (simple text processing)
            text_features = get_bert_embeddings(text_data)

        elif file_extension == "annotprocessed":
            # Handle `.annotprocessed` (annotated/structured data)
            # For example, parse annotations and convert it to features
            text_features = np.random.randn(1, 768)  # Dummy feature
        else:
            raise ValueError("Unsupported text file format")

        return text_features

    except Exception as e:
        print(f"Error in preprocess_text: {e}")
        return None  # Return None if preprocessing fails

def preprocess_audio(audio_file):
    try:
        audio_data = audio_file.read()  # Read audio file
        # Example: Random audio features (replace with real feature extraction)
        audio_features = np.random.randn(1, 40)  # Example: Random features
        return audio_features
    except Exception as e:
        print(f"Error in preprocess_audio: {e}")
        return None

def preprocess_video(video_file):
    try:
        video_data = video_file.read()  # Read video file
        # Example: Random video features (replace with real feature extraction)
        video_features = np.random.randn(1, 512)  # Example: Random features
        return video_features
    except Exception as e:
        print(f"Error in preprocess_video: {e}")
        return None

def get_bert_embeddings(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()
