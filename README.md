# Flower Classifier using PyTorch and ResNet-18

This project implements a flower image classification model using PyTorch and the ResNet-18 architecture. The model is trained on a custom flower dataset, and it classifies flowers into two categories, **daisy** and **dandelion**.

## Project Overview

This repository contains a pipeline for training a deep learning model to classify flower images. It uses the pre-trained ResNet-18 model, fine-tuned for the flower classification task. The model is trained on a dataset of labeled flower images, with two classes: **daisy** and **dandelion**.

### Features

- **Data Augmentation**: Random cropping, resizing, and horizontal flips applied to training data for better generalization.
- **Transfer Learning**: ResNet-18 model pretrained on ImageNet, fine-tuned for flower classification.
- **GPU Support**: Training can be accelerated using CUDA, making the model more efficient.

## Dataset

The dataset consists of images divided into training and validation sets, each containing two categories: **daisy** and **dandelion**. These images are preprocessed using data normalization based on mean and standard deviation calculated from the dataset.

### Dataset Path

- Training Dataset: `ImgDataset/train`
- Validation Dataset: `ImgDataset/valid`

## Model Architecture

This project uses **ResNet-18**, a residual neural network pretrained on ImageNet. The model's final fully connected layer is modified to output two classes corresponding to the flowers.

## Training Process

- **Optimizer**: SGD (Stochastic Gradient Descent) with learning rate = 0.001, momentum = 0.9.
- **Loss Function**: Cross-Entropy Loss for multi-class classification.
- **Epochs**: 10 epochs are run to fine-tune the model.
- **Batch Size**: Batch size of 64 images.

### Training Loop

- The model goes through the **train** and **validation** phases. In the training phase, the model learns from the data using backpropagation. In the validation phase, the model's performance is evaluated on unseen data.

## Evaluation

During training, the model is evaluated on accuracy and loss for both the training and validation sets. After training, the model is saved as `Flower_classification_model.pth`.

Metrics tracked:
- Accuracy
- Loss
- Precision
- Recall
- F1 Score
- ROC-AUC
