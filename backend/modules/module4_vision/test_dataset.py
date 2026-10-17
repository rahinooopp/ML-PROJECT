
from datasets import load_dataset
from pathlib import Path

DATASET_NAME = "Parveshiiii/AI-vs-Real"

PROJECT_ROOT = Path(__file__).resolve().parents[3]

REAL_DIR = PROJECT_ROOT / "datasets" / "real"
AI_DIR = PROJECT_ROOT / "datasets" / "ai_generated"

REAL_DIR.mkdir(parents=True, exist_ok=True)
AI_DIR.mkdir(parents=True, exist_ok=True)

print("Connecting to dataset...")

dataset = load_dataset(
    DATASET_NAME,
    split="train",
    streaming=True
)

saved_real = 0
saved_ai = 0
max_per_class = 10

print("Saving up to 10 real images and 10 AI-generated images...")

for sample in dataset:
    label = int(sample["binary_label"])
    image = sample["image"].convert("RGB")

    if label == 1 and saved_real < max_per_class:
        image.save(REAL_DIR / f"real_{saved_real + 1}.jpg")
        saved_real += 1
        print(f"Saved real image: {saved_real}")

    elif label == 0 and saved_ai < max_per_class:
        image.save(AI_DIR / f"ai_{saved_ai + 1}.jpg")
        saved_ai += 1
        print(f"Saved AI-generated image: {saved_ai}")

    if saved_real == max_per_class and saved_ai == max_per_class:
        break

print("\nDataset sampling finished!")
print("Real images saved:", saved_real)
print("AI-generated images saved:", saved_ai)
print("Real folder:", REAL_DIR)
print("AI folder:", AI_DIR)
