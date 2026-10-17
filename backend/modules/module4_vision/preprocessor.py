from PIL import Image


def preprocess_image(image_path):
    """
    Load an image and resize it to 224x224 pixels.
    """

    image = Image.open(image_path).convert("RGB")

    image = image.resize((224, 224))

    return image