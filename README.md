# VGG Architectures on CIFAR-10

## Project summary
This project implements and compares VGG-11, VGG-13, and VGG-16 on CIFAR-10, each with and without Batch Normalization (BN). It explores the VGG design idea of stacking small 3×3 convolutions and uses BN as the main ablation.

This is a reduced-scale CIFAR-10 study inspired by Simonyan and Zisserman's VGG paper. It is **not** a direct reproduction of the original ImageNet benchmark or training recipe.

## Research questions
1. How does measured performance change across VGG-11, VGG-13, and VGG-16?
2. What changes when Batch Normalization is enabled or disabled?
3. How do parameter count and accuracy compare?
4. Which CIFAR-10 classes are more difficult for the best measured model?

## Dataset and split
CIFAR-10 has 60,000 32×32 RGB images across ten classes. This notebook uses 45,000 training images, 5,000 validation images, and 10,000 test images.

## Method
- Framework: PyTorch and TorchVision
- Architectures: VGG-11, VGG-13, VGG-16
- Ablation: with and without Batch Normalization
- Loss: CrossEntropyLoss
- Optimizer: SGD (learning rate 0.05, momentum 0.9)
- Weight decay: 0.0005
- Scheduler: CosineAnnealingLR
- Initialization: Kaiming initialization
- Reported final experiments: 5 epochs per model
- Seed: the notebook sets `SEED = 42`; verify that all relevant random sources and data-loader workers use it in the environment where you rerun.

## Recorded results
The following test accuracies were recorded in the project conversation and should be treated as preliminary, five-epoch results:

| Model | BatchNorm | Test accuracy | Parameters |
|---|---|---:|---:|
| VGG11 | No | 77.71% | 9,225,610 |
| VGG11 | Yes | 68.96% | 9,231,114 |
| VGG13 | No | 75.11% | 9,410,122 |
| VGG13 | Yes | 39.79% | 9,416,010 |
| VGG16 | No | 70.85% | 14,719,818 |
| VGG16 | Yes | 33.32% | 14,728,266 |

These results are not a general conclusion that Batch Normalization hurts performance. The poor deeper BN results should be investigated with longer training, learning-rate checks, confirmation of BN placement and `train()`/`eval()` usage, and repeated runs.

## Reproduce the experiments
1. Open `VGG_CIFAR10_Research.ipynb` in Google Colab.
2. Select **Runtime → Change runtime type → T4 GPU** if available.
3. Run the notebook cells in order.
4. Check that the training, validation, test, and plots complete without errors.
5. Record any changed configuration (epochs, batch size, augmentation, learning rate, seed) in your report.

Install dependencies locally with:
```bash
pip install -r requirements.txt
```

## Files
- `VGG_CIFAR10_Research.ipynb`: notebook supplied for the project.
- `requirements.txt`: Python dependencies.
- `VGG_CIFAR10_Research_Report.pdf`: four-page IEEE-style draft report.

## Limitations
- Five epochs is a short training budget and does not establish convergence.
- CIFAR-10 and the adapted classifier differ from the original VGG ImageNet experiment.
- Only one reported run per configuration is available here; run-to-run variability is unknown.
- The recorded results should be regenerated in Colab before final submission if the instructor requires fresh execution evidence.

## Reference
K. Simonyan and A. Zisserman, “Very Deep Convolutional Networks for Large-Scale Image Recognition,” arXiv:1409.1556, 2014 (revised 2015). https://arxiv.org/abs/1409.1556

## Third-party code
This notebook uses standard PyTorch and TorchVision APIs. If you used or adapted an official or third-party implementation beyond standard library building blocks, add its repository URL and describe the parts consulted here and in the report.
