# MNIST Dataset Prediction

This repository contains the code for training and predicting digits from the MNIST dataset using a Convolutional Neural Network (CNN). The MNIST dataset is a well-known dataset in the machine learning community, consisting of 28x28 grayscale images of handwritten digits (0-9).

## Table of Contents

- [Introduction](#introduction)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Predicting Custom Images](#predicting-custom-images)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The goal of this project is to build a deep learning model that can accurately classify handwritten digits from the MNIST dataset. The model is built using TensorFlow/Keras and is capable of predicting digits from custom images as well.

## Model Architecture

The model is a simple CNN that consists of the following layers:
- **Conv2D**: Convolutional layers to extract features from the input images.
- **MaxPooling2D**: Pooling layers to reduce the spatial dimensions of the feature maps.
- **Flatten**: Converts the 2D matrix data to a 1D vector.
- **Dense**: Fully connected layers to perform the classification.

### Summary of the Model

```
Conv2D -> MaxPooling2D -> Conv2D -> MaxPooling2D -> Flatten -> Dense -> Dense
```

## Dataset

The MNIST dataset contains 70,000 images of handwritten digits, split into:
- **Training Set**: 60,000 images
- **Test Set**: 10,000 images

Each image is 28x28 pixels and is labeled with a digit from 0 to 9.

## Installation

To get started, clone this repository and install the required dependencies.

```bash
git clone https://github.com/alihassanml/Mnist-Dataset-Prediction.git
cd Mnist-Dataset-Prediction
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the model on the MNIST dataset, run the following command:

```bash
python train.py
```

### Loading a Pre-trained Model

If you already have a trained model, you can load it and use it for predictions:

```python
import tensorflow as tf

model = tf.keras.models.load_model('mnist_model.h5')
```

### Predicting Custom Images

To predict a digit from a custom image, follow these steps:

1. Place your custom image in the project directory.
2. Run the following script:

```python
from predict_custom_image import predict_digit

image_path = 'custom_digit.png'
predicted_digit = predict_digit(image_path)
print(f'The predicted digit is: {predicted_digit}')
```

Make sure the custom image is a 28x28 grayscale image. If not, the script will resize and preprocess it accordingly.

## Results

The model achieves over 99% accuracy on the MNIST test set. Below are some sample predictions:

![Sample Predictions](sample_predictions.png)

## Contributing

Contributions are welcome! If you have any improvements, suggestions, or new features to add, feel free to open an issue or create a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
