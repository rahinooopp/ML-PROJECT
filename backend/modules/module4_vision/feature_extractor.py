
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


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
            p=2,
            dim=-1,
            keepdim=True
        )

        return features.cpu().numpy().squeeze()


def main():
    project_root = Path(__file__).resolve().parents[3]

    real_dir = project_root / "datasets" / "real"
    ai_dir = project_root / "datasets" / "ai_generated"

    output_dir = project_root / "backend" / "modules" / "module4_vision" / "features"
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = CLIPFeatureExtractor()

    features_list = []
    labels_list = []

    image_groups = [
        (real_dir, 1, "Real"),
        (ai_dir, 0, "AI-generated"),
    ]

    for folder, label, class_name in image_groups:
        image_paths = sorted(folder.glob("*.jpg"))

        print(f"\nProcessing {class_name} images: {len(image_paths)}")

        for image_path in image_paths:
            try:
                features = extractor.extract_features(image_path)

                features_list.append(features)
                labels_list.append(label)

                print(f"Processed: {image_path.name}")

            except Exception as error:
                print(f"Failed: {image_path.name} - {error}")

    if not features_list:
        print("Error: No features were extracted.")
        return

    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int64)

    np.savez_compressed(
        output_dir / "image_features.npz",
        X=X,
        y=y
    )

    print("\nFeature extraction completed!")
    print("Features shape:", X.shape)
    print("Labels shape:", y.shape)
    print("Real images:", int(np.sum(y == 1)))
    print("AI-generated images:", int(np.sum(y == 0)))
    print("Saved to:", output_dir / "image_features.npz")


if __name__ == "__main__":
    main()


