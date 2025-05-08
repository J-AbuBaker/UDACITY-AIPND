# Project 1: Classifying Pet Images

This repository contains **Project 1** from the AI Programming with Python Nanodegree. The goal of this project is to build an image classification application that uses a pre-trained Convolutional Neural Network (CNN) to identify whether a pet image is of a dog, and if so, what breed it is.

## Overview

The project evaluates the performance of different CNN architectures (`resnet`, `alexnet`, `vgg`) on classifying pet images. It compares the predicted labels from the classifier to the actual labels extracted from image filenames and provides a detailed performance report including correct and incorrect classifications.

## Main Features

* **Command Line Interface**: Accepts user input for image directory, CNN model architecture, and a file of dog names.
* **Label Extraction**: Parses image filenames to determine ground-truth pet labels.
* **Image Classification**: Classifies images using a selected pre-trained CNN model from PyTorch.
* **Performance Analysis**: Tracks and displays detailed classification metrics such as match percentages and breed accuracy.
* **Dog Detection**: Identifies whether the image is a dog and if the classifier correctly detected it.

## Functional Workflow

1. **Input Parsing**:
   Retrieves user input via command line:

   * `--dir`: Image folder path (default = `'pet_images/'`)
   * `--arch`: CNN architecture to use (`resnet`, `alexnet`, or `vgg`)
   * `--dogfile`: File containing valid dog breed names

2. **Label Creation**:

   * `get_pet_labels.py`: Extracts clean labels from image filenames

3. **Classification**:

   * `classify_images.py`: Uses `classifier.py` to generate predictions and compares them to the ground-truth labels

4. **Dog Verification**:

   * `adjust_results4_isadog.py`: Checks if both the pet label and classifier label correspond to dogs using `dognames.txt`

5. **Statistics Calculation**:

   * `calculates_results_stats.py`: Computes summary metrics like accuracy and match rates

6. **Reporting**:

   * `print_results.py`: Displays overall performance, as well as incorrectly classified dogs and breeds if enabled

## Tools and Technologies

* Python 3.x
* PyTorch (pre-trained models)
* argparse
* PIL (Pillow)
* torchvision
* NumPy

## Output Metrics

* Number of dog and non-dog images
* Percentage of:

  * Correct dog classifications
  * Correct breed classifications
  * Correct non-dog identifications
  * Total matches between labels
