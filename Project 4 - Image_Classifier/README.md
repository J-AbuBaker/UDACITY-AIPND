# Project 4: Image Classifier

This repository contains **Project 4** of the AI Programming with Python Nanodegree. The objective is to build an image classification model that can identify different species of flowers from images using a pre-trained deep learning model.

## Overview

In this project, a deep learning classifier is trained on flower images using transfer learning. The model leverages a pre-trained image recognition network from TensorFlow’s `tf.keras.applications`, which is then customized and fine-tuned to perform flower classification across 102 categories.

## Core Components

* **Data Preprocessing**: Images are resized, normalized, and split into training, validation, and test datasets.

* **Model Architecture**: A pre-trained CNN (**MobileNetV2**) is used as a base model. The top layer is replaced with a custom classifier suitable for the 102-class flower dataset.

* **Training Process**: The model is compiled and trained using TensorFlow/Keras, with accuracy and loss tracked across epochs.

* **Evaluation**: The final model is evaluated on a test dataset to assess its performance and generalization ability.

* **Prediction Function**: A prediction function is implemented to classify a new image and return the top probable classes.

## Category Mapping (JSON File)

To provide human-readable names for flower categories, a file named `cat_to_name.json` is used. This JSON file maps numeric category labels (e.g., "21") to actual flower names (e.g., "fire lily").

Example:

```json
{
  "0": "pink primrose",
  "1": "hard-leaved pocket orchid"
}
```

This file is loaded as follows:

```python
import json
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
```

The mapping is applied during prediction to return class names instead of category numbers.

## Model Saving and Loading

After training, the model is saved using:

```python
model.save('model.h5')
```

It can be loaded later for inference or further training using:

```python
from tensorflow.keras.models import load_model
model = load_model('model.h5')
```

## Tools and Libraries

* Python 3.x
* TensorFlow / Keras
* NumPy
* Matplotlib
* Jupyter Notebook

## Evaluation

The final classifier is tested on unseen data, and performance is evaluated using classification accuracy. Additionally, a function is provided to visualize predictions alongside top-k class probabilities.
