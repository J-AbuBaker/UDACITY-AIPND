import argparse
import tensorflow as tf
import tensorflow_hub as hub
import json
import tf_keras
from tf_keras.models import load_model
from util import process_image, predict


# Command-line arguments parsing
parser = argparse.ArgumentParser(description="Flower classification using a trained model.")

parser.add_argument('image_path', type=str, help="Path to the input image")
parser.add_argument('model_path', type=str, help="Path to the saved Keras model (.h5 file)")
parser.add_argument('--top_k', type=int, default=5, help="Number of top predictions to return")
parser.add_argument('--category_names', type=str, help="Path to a JSON file with class name mappings")

args = parser.parse_args()

# Load the model
custom_objects = {
    'KerasLayer': hub.KerasLayer  # Register the custom layer
}

model = load_model(args.model_path, custom_objects=custom_objects)

# Load category names from JSON file if provided
if args.category_names:
    with open(args.category_names, 'r') as f:
        class_names = json.load(f)
else:
    class_names = None

# Make predictions
top_k_probs, top_classes = predict(args.image_path, model, top_k=args.top_k, class_names=class_names)

# Display the results
print("Top predictions:")
for i in range(args.top_k):
    print(f"{top_classes[i]}: {top_k_probs[i]:.4f}")

