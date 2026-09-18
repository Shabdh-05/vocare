from transformers import AutoFeatureExtractor, HubertModel
import soundfile as sf
import torch

# Test WAV file
p = r"C:\Users\msdha\Downloads\archive (1)\patient-vocal-dataset\patient-vocal-dataset\Normal\4-a_h.wav"

# Load audio
audio, sr = sf.read(p)

print("Sample rate:", sr)
print("Samples:", len(audio))

# Load HuBERT feature extractor and model
processor = AutoFeatureExtractor.from_pretrained(
    "facebook/hubert-base-ls960"
)

model = HubertModel.from_pretrained(
    "facebook/hubert-base-ls960"
)

# Prepare audio for HuBERT
x = processor(
    audio,
    sampling_rate=sr,
    return_tensors="pt"
).input_values

print("Input shape:", x.shape)

# Extract HuBERT embeddings
with torch.no_grad():
    y = model(x).last_hidden_state

print("HuBERT output shape:", y.shape)

print("HuBERT test completed successfully!")