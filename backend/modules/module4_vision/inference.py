
from pathlib import Path

import joblib
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


# 1. Model settings
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

BASE_DIR = Path(__file__).resolve().parent

CLASSIFIER_PATH = (
    BASE_DIR / "models" / "clip_logistic_regression.joblib"
)

# Change this path to test a different image
IMAGE_PATH = BASE_DIR.parent.parent.parent / "datasets" / "ai_generated" / "ai_1.jpg"


def predict_image():
    # 2. Check that the image exists
    if not IMAGE_PATH.exists():
        print(f"Error: Image not found:\n{IMAGE_PATH}")
        return

    if not CLASSIFIER_PATH.exists():
        print(f"Error: Classifier not found:\n{CLASSIFIER_PATH}")
        return

    # 3. Select CPU or GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 4. Load the trained classifier
    print("Loading trained classifier...")
    classifier = joblib.load(CLASSIFIER_PATH)

    # 5. Load the CLIP model and processor
    print("Loading CLIP model...")
    clip_model = CLIPModel.from_pretrained(
        CLIP_MODEL_NAME
    ).to(device)

    processor = CLIPProcessor.from_pretrained(
        CLIP_MODEL_NAME
    )

    clip_model.eval()

    # 6. Open the image
    print(f"Analyzing image:\n{IMAGE_PATH}")

    with Image.open(IMAGE_PATH) as image:
        image = image.convert("RGB")

        inputs = processor(
            images=image,
            return_tensors="pt"
        )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # 7. Extract image features using CLIP
    with torch.no_grad():
        features = clip_model.get_image_features(
            pixel_values=inputs["pixel_values"]
        )

        # Normalize features exactly as in training
        features = F.normalize(features, p=2, dim=-1)

    features = features.cpu().numpy()

    # 8. Predict the image class
    prediction = classifier.predict(features)[0]
    probabilities = classifier.predict_proba(features)[0]

    class_names = {
        0: "AI-generated",
        1: "Real"
    }

    predicted_name = class_names[int(prediction)]

    print("\n" + "=" * 40)
    print("        SENTINELLENS RESULTS")
    print("=" * 40)

    print(f"Prediction: {predicted_name}")

    # Display the classifier's score for each class
    for class_id, probability in zip(
        classifier.classes_,
        probabilities
    ):
        class_name = class_names[int(class_id)]

        print(
            f"{class_name} score: "
            f"{probability * 100:.2f}%"
        )

    print("=" * 40)

    print(
        "\nNote: This is an experimental result. "
        "The classifier was trained on only 20 images, "
        "so the prediction may be inaccurate."
    )


if __name__ == "__main__":
    predict_image()
