 # VGG Architecture Comparison on CIFAR-10

## Project Overview

This project compares VGG11, VGG13, and VGG16 convolutional neural network (CNN) architectures on the CIFAR-10 image classification dataset. Each architecture is evaluated with and without Batch Normalization (BN) to study its effect on classification performance.

The models are implemented using PyTorch and evaluated using training, validation, and test metrics.

## Objectives

- Implement VGG11, VGG13, and VGG16 architectures.
- Compare each architecture with and without Batch Normalization.
- Train and evaluate all six model configurations on CIFAR-10.
- Analyze accuracy, parameter counts, and class-wise classification performance.
- Visualize and compare experimental results.

## Dataset

**CIFAR-10** contains 60,000 color images of size 32 × 32 pixels across 10 classes.

The dataset is divided into:

- Training: 45,000 images
- Validation: 5,000 images
- Testing: 10,000 images

The ten classes are airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

## Model Configurations

Six configurations are evaluated:

1. VGG11 without Batch Normalization
2. VGG11 with Batch Normalization
3. VGG13 without Batch Normalization
4. VGG13 with Batch Normalization
5. VGG16 without Batch Normalization
6. VGG16 with Batch Normalization

## Methodology

1. Load and preprocess the CIFAR-10 dataset.
2. Create training, validation, and test datasets.
3. Implement the VGG architectures using PyTorch.
4. Apply Kaiming weight initialization.
5. Train the models using Cross-Entropy Loss and SGD with momentum.
6. Use a Cosine Annealing learning-rate scheduler.
7. Evaluate the trained models on validation and test datasets.
8. Compare the results using tables, plots, and classification metrics.

### Training Configuration

- Optimizer: SGD
- Initial learning rate: 0.05
- Momentum: 0.9
- Weight decay: 0.0005
- Learning-rate scheduler: CosineAnnealingLR
- Epochs in the reported experiments: 5
- Loss function: CrossEntropyLoss

## Evaluation Metrics

The project evaluates models using:

- Training accuracy
- Validation accuracy
- Test accuracy
- Model parameter count
- Confusion matrix
- Precision, recall, and F1-score
- Training-history visualizations

## Experimental Results

| Model | Batch Normalization | Test Accuracy |
|---|---|---:|
| VGG11 | No | 77.71% |
| VGG11 | Yes | 68.96% |
| VGG13 | No | 75.11% |
| VGG13 | Yes | 39.79% |
| VGG16 | No | 70.85% |
| VGG16 | Yes | 33.32% |

**Important:** These are preliminary results from five training epochs. The low accuracy of some Batch Normalization configurations should be investigated with longer training and additional experiments. These results do not establish that Batch Normalization is generally worse.

## Project Files

The repository may contain:

- `VGG_CIFAR10.ipynb` — notebook containing model implementation, training, and evaluation.
- `requirements.txt` — Python dependencies.
- `preliminary_results.csv` — preliminary experiment results.
- `final_validation_results.csv` — validation results.
- `final_test_results.csv` — test results.
- `final_training_history.csv` — training-history data.
- Figures and plots — visualizations of model performance.

*Update these filenames if your actual uploaded files use different names.*

## Installation

Clone the repository:

```bash
git clone https://github.com/24cs3028-poll/VGG-CIFAR10-Research.git
cd VGG-CIFAR10-Research
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

1. Open `VGG_CIFAR10.ipynb` in Google Colab or Jupyter Notebook.
2. Install the required dependencies if necessary.
3. Run the notebook cells in order.
4. Allow the models to train and evaluate.
5. Review the generated result tables, metrics, and visualizations.

Training time depends on the selected hardware and configuration.

## Limitations and Future Work

- Train for more epochs to improve convergence.
- Investigate the low accuracy of the Batch Normalization models.
- Experiment with learning rates, batch sizes, and optimizers.
- Add repeated runs to assess result variability.
- Compare the models using additional evaluation metrics.

## Reference

Simonyan, K. and Zisserman, A. (2015). *Very Deep Convolutional Networks for Large-Scale Image Recognition.*

Paper: https://arxiv.org/abs/1409.1556

## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Google Colab
