# AI Engineer IBM Projects

This repository contains a collection of projects completed as part of an AI engineering curriculum. These projects cover fundamental concepts in deep learning and machine learning, from building a convolutional neural network (CNN) for image classification to applying transfer learning for waste classification and predicting match outcomes in League of Legends.

## Projects and What I've Learned

### 1. Fashion MNIST Project (`FashionMNISTProject.ipynb`)

- **Objective:** Classify clothing items from the Fashion MNIST dataset using a custom-built CNN.
- **Key Learnings:**
  - **PyTorch Fundamentals:** How to use `torch`, `nn.Module`, and `torch.utils.data.Dataset` to build a deep learning pipeline.
  - **Convolutional Neural Networks (CNNs):** Understanding the role of convolutional layers (`Conv2d`), pooling layers (`MaxPool2d`), and fully connected layers (`Linear`) in image classification.
  - **Batch Normalization:** Implementing `BatchNorm2d` to stabilize and accelerate the training process.
  - **Data Preprocessing:** Using `torchvision.transforms` to resize and convert images to tensors.

### 2. Waste Classification Using Transfer Learning (`Final Proj-Classify Waste Products Using TL FT.ipynb`)

- **Objective:** Classify waste products as either "recyclable" or "organic" using a pre-trained VGG16 model.
- **Key Learnings:**
  - **Transfer Learning:** Leveraging a pre-trained model (VGG16) to solve a new image classification problem with a smaller dataset.
  - **Fine-Tuning:** Unfreezing some layers of the pre-trained model to fine-tune it on the specific task, improving its performance.
  - **Keras and TensorFlow:** Using the Keras API within TensorFlow to build and train the model.
  - **Data Augmentation:** Using `ImageDataGenerator` to create more training data and prevent overfitting.

### 3. League of Legends Match Predictor (`Final Project League of Legends Match Predictor.ipynb`)

- **Objective:** Predict the winner of a League of Legends match using logistic regression.
- **Key Learnings:**
  - **Logistic Regression:** Implementing a logistic regression model from scratch using PyTorch.
  - **Feature Importance:** Evaluating which in-game statistics are most predictive of a win.
  - **Model Evaluation:** Using metrics like accuracy, confusion matrix, and ROC curves to evaluate the model's performance.
  - **Hyperparameter Tuning:** Experimenting with different learning rates to find the optimal one for the model.
  - **Regularization:** Using L2 regularization (weight decay) to prevent overfitting.

### 4. AI Capstone Project: Implment and Test a PyTorch-Based Classifier (`AI-capstone-project/Lab_M2L2_Implement_and_Test_a_PyTorch-Based_Classifier.ipynb`)

- **Objective:** Build, train, and evaluate a PyTorch-based CNN for classifying satellite images.
- **Key Learnings:**
  - **PyTorch `nn.Sequential`:** Building a CNN model in PyTorch using a sequential container.
  - **Training Loop:** Implementing a complete training and validation loop from scratch, including forward pass, loss calculation, backpropagation, and optimizer steps.
  - **Model Evaluation:** Tracking training and validation accuracy and loss to visualize the learning process and identify overfitting.
  - **Reproducibility:** Setting random seeds for Python, NumPy, and PyTorch to ensure reproducible results.

---

_This repository showcases my ability to apply a variety of machine learning and deep learning techniques to solve real-world problems._
