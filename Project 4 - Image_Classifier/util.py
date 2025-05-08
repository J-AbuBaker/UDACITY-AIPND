import tensorflow as tf
from PIL import Image
import numpy as np

def process_image(image, image_size=224):
    """
    Preprocesses an input image for use in a machine learning model.

    This function performs the following steps:
        1. Converts the input image to a float32 TensorFlow tensor.
        2. Resizes the image to the specified dimensions (image_size x image_size).
        3. Normalizes pixel values to the range [0, 1].
        4. Converts the image to a NumPy array and removes singleton dimensions.

    Args:
        image (Tensor or array-like): The input image to preprocess.
                                      Expected shape: (height, width, channels).
        image_size (int): The target size to resize the image's height and width.

    Returns:
        np.ndarray: The preprocessed image as a NumPy array with pixel values in [0, 1]
                    and shape (image_size, image_size, channels), or (image_size, image_size)
                    if it's a grayscale image.
    """
    # Convert the image to a TensorFlow tensor with float32 data type
    image = tf.cast(image, tf.float32)

    # Resize the image to the target size (image_size, image_size)
    image = tf.image.resize(image, [image_size, image_size])

    # Normalize the image by scaling pixel values to [0, 1]
    image /= 255.0

    # Convert the image back to a NumPy array and remove any singleton dimensions
    image = image.numpy().squeeze()

    return image


def predict(image_path, model, top_k=5, class_names=None):
    """
    Classifies a single image, returns the top-k predicted classes and their indices, 
    and prints the top-k predictions.

    Parameters:
    - image_path: Path to the image file
    - model: Trained model to make predictions
    - top_k: Number of top predictions to return (default is 5)
    - class_names: Dictionary mapping class indices to class names

    Returns:
    - probs: List of top-k probabilities
    - classes: List of top-k class names
    """
    # Open the image using PIL
    img = Image.open(image_path)

    # Process the image using the process_image function
    img_array = np.array(img)
    processed_image = process_image(img_array, image_size=224)

    # Add batch dimension to the processed image
    img_array_expanded = np.expand_dims(processed_image, axis=0)

    # Get predictions from the model
    predictions = model.predict(img_array_expanded)

    # Get the top-k predicted indices and probabilities
    top_k_indices = np.argsort(predictions[0])[-top_k:][::-1]
    top_k_probs = predictions[0][top_k_indices]

    # Map the indices to class names (assuming class_names is available)
    top_classes = [class_names[str(idx)] for idx in top_k_indices]

    return top_k_probs, top_classes
