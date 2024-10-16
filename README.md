# Food101 Image Classification

This project implements a food image classification model using the Food101 dataset from TensorFlow Datasets. The model outperforms the DeepFood model and achieves an accuracy of **80%**. Helper functions are used from an external script to streamline the process.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Dataset](#dataset)
- [Helper Functions](#helper-functions)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Introduction
This project involves building a deep learning model for food image classification using the Food101 dataset. The model is designed to classify food images into 101 different categories and achieves an accuracy of **80%**, surpassing the DeepFood model's performance.

## Features
- **TensorFlow and Keras Integration:** Utilizes TensorFlow's high-level Keras API for building and training the model.
- **Food101 Dataset:** Directly loads the Food101 dataset from TensorFlow Datasets.
- **Efficient Training:** Used data augmentation and transfer learning techniques to improve model performance.

## Dataset
The Food101 dataset, which contains 101,000 images across 101 different categories, is loaded directly from TensorFlow Datasets.

```python
import tensorflow_datasets as tfds

# Load the Food101 dataset
(ds_train, ds_test), ds_info = tfds.load('food101', split=['train', 'validation'], shuffle_files = true , with_info=True, as_supervised=True)
```

More information on the dataset can be found on the [TensorFlow Datasets Food101 page](https://www.tensorflow.org/datasets/catalog/food101).

## Helper Functions
The project uses helper functions for image preprocessing, model training, and evaluation. These functions can be accessed by downloading the script using the following command:

```bash
!wget https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/extras/helper_functions.py
```

This script contains utility functions to simplify tasks like plotting loss curves and loading data.

## Usage

### Running the Model
To run the food classification model, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/food101_classification.git
    cd food101_classification
    ```

2. Download the helper functions:
    ```bash
    !wget https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/extras/helper_functions.py
    ```

3. Run the training script:
    ```bash
    jupyter notebook food101_classification.ipynb
    ```

This will train the model using the Food101 dataset and display the accuracy after training.

## Model Architecture
The model was built using the following components:
- **Convolutional Neural Networks (CNNs):** Multiple convolutional layers with pooling and dropout for feature extraction.
- **Transfer Learning:** Pre-trained models (e.g., EfficientNet or ResNet) were used for faster training and improved performance.
- **Data Augmentation:** Applied techniques like random flips, rotations, and zooms to improve model generalization.

## Evaluation
The model achieved the following performance:
- **Accuracy:** 80% on the validation dataset, outperforming the DeepFood model.

```bash
Final Accuracy: 80%
```

## Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request if you have ideas for improvements.

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
