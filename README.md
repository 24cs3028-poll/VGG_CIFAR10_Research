# VGG CIFAR-10 Research

## Overview

This project compares VGG-11, VGG-13, and VGG-16 convolutional neural network architectures on the CIFAR-10 image classification dataset, with and without Batch Normalization.

The goal is to study how architecture depth and Batch Normalization affect model performance.

## Models

Six configurations are evaluated:

- VGG-11 without Batch Normalization
- VGG-11 with Batch Normalization
- VGG-13 without Batch Normalization
- VGG-13 with Batch Normalization
- VGG-16 without Batch Normalization
- VGG-16 with Batch Normalization

## Dataset

CIFAR-10 contains 60,000 colour images across 10 classes. The experiments use a training/validation/test split of 45,000 / 5,000 / 10,000 images.

## Methodology

- Framework: PyTorch
- Dataset: CIFAR-10
- Optimizer: Stochastic Gradient Descent (SGD)
- Momentum: 0.9
- Weight decay: 0.0005
- Initial learning rate: 0.05
- Learning-rate schedule: Cosine Annealing
- Weight initialization: Kaiming initialization
- Training duration: 5 epochs per configuration

## Evaluation

The experiments record training and validation metrics, test accuracy, model parameter counts, and classification metrics. Accuracy/loss curves and a confusion matrix are also used to analyse performance.

Results should be interpreted in the context of the five-epoch training budget. They are preliminary experiments, not a definitive conclusion about whether Batch Normalization improves VGG models.

## Repository Contents

- Research notebook: model implementation, training, and evaluation
- Results: experiment summaries and training histories
- Model checkpoints: saved weights, where provided
- `requirements.txt`: Python dependencies

## How to Run

1. Install the dependencies listed in `requirements.txt`.
2. Open the research notebook in Jupyter or Google Colab.
3. Run the notebook cells in order.
4. Review the evaluation tables and plots.

For GPU training, use a compatible CUDA environment or Google Colab GPU runtime.

## Limitations

The models were trained for only five epochs per configuration. Results may change with longer training, hyperparameter tuning, and repeated runs with different random seeds.

## Research Background and References

This project is an educational implementation and comparative experimental study based on the VGG convolutional neural network architecture. The original architecture was introduced by Karen Simonyan and Andrew Zisserman. This project adapts the VGG-11, VGG-13, and VGG-16 configurations for CIFAR-10 and experimentally compares models with and without Batch Normalization.

The original research paper and dataset reference are credited below. The experiments, implementation, evaluation, and reported results in this repository belong to this project and should not be interpreted as results reported by the original paper.

### 1. Original VGG Research Paper

**Title:** Very Deep Convolutional Networks for Large-Scale Image Recognition

**Authors:** Karen Simonyan and Andrew Zisserman

**Publication:** International Conference on Learning Representations (ICLR), 2015

**Publisher / Organization:** Computational and Biological Learning Society (as listed in the University of Oxford Research Archive)

**Original paper:** https://arxiv.org/abs/1409.1556

**Official VGG research page:** https://www.robots.ox.ac.uk/~vgg/research/very_deep/

### 2. CIFAR-10 Dataset Reference

**Title:** Learning Multiple Layers of Features from Tiny Images

**Author:** Alex Krizhevsky

**Year:** 2009

**Publication type:** Technical report, University of Toronto

**Original report:** https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf

**Dataset website:** https://www.cs.toronto.edu/~kriz/cifar.html

### Project Author

**Implementation and experiments:** Kavya Rathore

This repository is an independent educational implementation and experimental comparison inspired by the cited VGG research. The original papers and dataset are credited to their respective authors. This project does not claim authorship of the original VGG architecture or the CIFAR-10 dataset.
