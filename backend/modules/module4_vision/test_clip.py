import torch
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"

print("Loading CLIP model...")

processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model = CLIPModel.from_pretrained(MODEL_NAME)

model.eval()

print("CLIP loaded successfully!")

if torch.cuda.is_available():
    print("Device: CUDA")
else:
    print("Device: CPU")
    