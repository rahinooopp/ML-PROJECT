import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image


class CLIPFeatureExtractor:
    def __init__(self):
        self.model_name = "openai/clip-vit-base-patch32"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading CLIP on {self.device}...")

        self.processor = CLIPProcessor.from_pretrained(
            self.model_name
        )

        self.model = CLIPModel.from_pretrained(
            self.model_name
        )

        self.model.to(self.device)
        self.model.eval()

        print("CLIP loaded successfully!")

    def extract_features(self, image_path):
        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            features = self.model.get_image_features(**inputs)

        features = features / features.norm(
            p=2, dim=-1, keepdim=True
        )

        return features.cpu()


if __name__ == "__main__":
    extractor = CLIPFeatureExtractor()

    test_image = Image.new(
        "RGB",
        (224, 224),
        color="white"
    )

    test_image.save("clip_test_image.jpg")

    features = extractor.extract_features(
        "clip_test_image.jpg"
    )

    print("Feature extraction successful!")
    print("Feature shape:", tuple(features.shape))