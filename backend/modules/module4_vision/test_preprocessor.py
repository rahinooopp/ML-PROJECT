from PIL import Image
from preprocessor import preprocess_image

# Create a temporary test image
test_image = Image.new("RGB", (500, 300), color="white")
test_image.save("test_image.jpg")

# Preprocess the image
result = preprocess_image("test_image.jpg")

# Print the results
print("Image processed successfully!")
print("Image size:", result.size)
print("Image mode:", result.mode)