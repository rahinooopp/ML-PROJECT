# SentinelLens - Module 4: Vision

## Overview

This module provides an experimental image authenticity classifier.
It uses CLIP image embeddings and a Logistic Regression classifier
to classify an image as either AI-generated or real.

## Current Pipeline

1. Load an input image.
2. Process the image using the CLIP processor.
3. Extract image features using the pretrained CLIP model.
4. Load the trained Logistic Regression classifier.
5. Predict whether the image is AI-generated or real.
6. Display the prediction and classification scores.

## Dataset

The current experimental dataset contains 20 images:

- 10 real images
- 10 AI-generated images

The dataset is stored in the repository root:

- datasets/real/
- datasets/ai_generated/

This dataset is intended for initial testing, not final evaluation.

## Requirements

Install the module dependencies from the repository root:

    python -m pip install -r backend/modules/module4_vision/requirements.txt

## Run Inference

From the repository root, run:

    python backend/modules/module4_vision/inference.py

The first run may download the pretrained CLIP model.
An internet connection may be required if the model is not cached.

## Train the Classifier

From the repository root, run:

    python backend/modules/module4_vision/feature_extractor.py

Then run:

    python backend/modules/module4_vision/train_classifier.py

Feature extraction and training may take time, depending on the computer.

## Important Limitations

- This is an experimental prototype.
- The classifier was trained using only 20 images.
- The current test split is very small.
- Classification scores are not calibrated probabilities.
- Results may not generalize to unseen images or different image-generation methods.
- More diverse data and a larger independent test set are required
  before making reliability claims.

## Current Scope

This module is an initial standalone prototype.
Integration with the main SentinelLens backend is a separate step.
