
from pathlib import Path

import numpy as np
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    module_dir = Path(__file__).resolve().parent
    features_file = module_dir / "features" / "image_features.npz"
    models_dir = module_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Loading extracted features...")

    with np.load(features_file) as data:
        X = data["X"]
        y = data["y"]

    print("Features shape:", X.shape)
    print("Labels shape:", y.shape)
    print("Real images:", int(np.sum(y == 1)))
    print("AI-generated images:", int(np.sum(y == 0)))

    if len(X) < 10 or len(np.unique(y)) != 2:
        raise ValueError("Not enough data or one class is missing.")

    # Keep the test set separate from the training data.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            random_state=42,
        ),
    )

    print("\nTraining Logistic Regression...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\nEvaluation results")
    print("------------------")
    print(f"Accuracy: {accuracy:.2%}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["AI-generated", "Real"],
            zero_division=0,
        )
    )

    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    model_path = models_dir / "clip_logistic_regression.joblib"
    dump(model, model_path)

    print("\nTraining completed!")
    print("Model saved to:", model_path)
    print(
        "\nIMPORTANT: This is a tiny preliminary experiment, "
        "not a reliable measure of real-world performance."
    )


if __name__ == "__main__":
    main()

