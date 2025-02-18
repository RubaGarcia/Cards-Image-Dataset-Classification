# Card Image Classification

This repository contains a Jupyter Notebook (`cards-classification.ipynb`) that demonstrates the process of building and training a convolutional neural network (CNN) for classifying images of playing cards. The model is designed to recognize 53 different classes of cards, including the standard 52 cards and a joker.

## Overview

The notebook includes the following key steps:

1. **Data Loading and Preprocessing**:
   - The dataset is loaded using TensorFlow's `image_dataset_from_directory` function.
   - Images are resized to a uniform size of 200x200 pixels.
   - Data augmentation techniques such as random rotation, zoom, contrast, brightness, and translation are applied to increase the robustness of the model.

2. **Model Architecture**:
   - The CNN model is built using TensorFlow and Keras.
   - The model consists of multiple convolutional layers followed by batch normalization and max-pooling layers.
   - The model includes dense layers with dropout regularization to prevent overfitting.
   - The final layer uses a softmax activation function to output probabilities for each of the 53 classes.

3. **Model Training**:
   - The model is compiled with the Adam optimizer and sparse categorical cross-entropy loss.
   - Training is performed with early stopping and learning rate reduction on plateau to optimize performance.
   - The model is trained for 100 epochs, with validation accuracy and loss monitored.

4. **Evaluation**:
   - The model's performance is evaluated on a validation dataset.
   - The notebook includes visualizations of the training process, including accuracy and loss plots.

## Requirements

To run the notebook, you need the following Python packages:

- TensorFlow
- NumPy
- Matplotlib
- Pandas
- Scikit-learn

You can install the required packages using pip:

```bash
pip install tensorflow numpy matplotlib pandas scikit-learn
```

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/card-image-classification.git
   cd card-image-classification
   ```

2. Open the Jupyter Notebook:

   ```bash
   jupyter notebook cards-classification.ipynb
   ```

3. Follow the steps in the notebook to load the dataset, build the model, and train it.

## Dataset

The dataset used in this project is assumed to be stored in a directory structure where each subdirectory corresponds to a class of cards. The dataset should be split into training, validation, and test sets.

## Results

The model achieves high accuracy on the validation set, demonstrating its effectiveness in classifying card images. The notebook includes visualizations of the training process and the model's performance.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any suggestions or improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- TensorFlow and Keras for providing the tools to build and train the model.
- The dataset providers for making the card images available for training.

---

Feel free to explore the notebook and adapt it to your own image classification tasks!
