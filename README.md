# CNN for Image Classification with Patch-Based Feature Activation

## Overview

This project focuses on the development and comparison of various convolutional neural network architectures for image classification using a subset of the Caltech-101 dataset. The dataset consists of images from 5 different classes, and the images are preprocessed by resizing them to 224x224 pixels. 

## Dataset

The dataset is a subset of Caltech-101 consisting of color images from 5 different classes. The images are train-validation-test separated and resized to 224x224x3 pixels for this task.


### Architecture Designs

1. **Architecture-1 (4-layer CNN):**
    - **Layer 1:** 8 filters of size 11x11, stride 4, padding 0, followed by 3x3 max pooling with stride 2.
    - **Layer 2:** 16 filters of size 5x5, stride 1, padding 0, followed by 3x3 max pooling with stride 2.
    - **Fully Connected Layer 1:** 128 nodes, ReLU activation.
    - **Output Layer:** 5 nodes, Softmax activation.

2. **Architecture-2 (5-layer CNN):**
    - **Layers 1 & 2:** Same as Architecture-1.
    - **Layer 3:** 32 filters of size 3x3, stride 1, padding 0, followed by 3x3 max pooling with stride 2.
    - **Fully Connected Layer 1:** 128 nodes, ReLU activation.
    - **Output Layer:** 5 nodes, Softmax activation.

3. **Architecture-3 (6-layer CNN):**
    - **Layers 1 & 2:** Same as Architecture-1.
    - **Layer 3:** 32 filters of size 3x3, stride 1, padding 0 (no max pooling).
    - **Layer 4:** 64 filters of size 3x3, stride 1, padding 0, followed by 3x3 max pooling with stride 2.
    - **Fully Connected Layer 1:** 128 nodes, ReLU activation.
    - **Output Layer:** 5 nodes, Softmax activation.

### Evaluation Metrics

Each architecture is evaluated based on:
- **Training and Validation Accuracy**
- **Confusion Matrix**
- **Number of Parameters**

### Tasks Breakdown

1. **Training & Validation Accuracy**:
   - Train all three CNN architectures and record the training and validation accuracy for each architecture.
   - Compare results with the FCNN from Task 1.

2. **Parameter Count**:
   - Compute and compare the number of parameters for both CNNs and FCNNs.

3. **Best Model Selection**:
   - Select the architecture with the highest validation accuracy.
   - Evaluate its performance on the test dataset and present the confusion matrix.

4. **Feature Maps**:
   - Select one image from the training set and plot the feature maps:
     - **All 8 feature maps** from the first convolutional layer.
     - **8 feature maps** selected from the other convolutional layers of the best architecture.
   - Provide observations from the feature maps.

5. **Neuron Activation & Visualization**:
   - For one image from each of the 5 classes in the training set:
     - Pass the image through the best CNN architecture.
     - Identify the neuron in the last convolutional layer that is maximally activated.
     - Trace the patch in the image that causes the neuron to fire.
     - Visualize the patches for each image that maximally activate the neuron.

## Results & Observations

- **Architecture Comparison**: The CNN architectures are compared in terms of accuracy and parameter efficiency.
- **Feature Map Analysis**: Insights are gained from analyzing the feature maps of the best-performing CNN model.
- **Neuron Activation Visualization**: Patches in the images that cause neurons to fire are visualized for further understanding of the network's decision-making process.

## Conclusion

The project explores the design and implementation of convolutional neural networks and provides insights into their performance and behavior, as well as a comparison with fully connected networks.

## How to Run

1. Clone the repository.
2. Ensure you have Python and the necessary libraries (TensorFlow, Keras, etc.) installed.
3. Run the scripts for training and evaluation.

## Dependencies

- Python 3.x
- TensorFlow or PyTorch
- NumPy
- Matplotlib (for visualization)
- PIL (for image preprocessing)

---

