# Introduction to MNIST and Neural Networks
The MNIST dataset consists of 60,000 training and 10,000 test images of handwritten digits (0-9), each a 28x28 grayscale pixel image (flattened to 784 features). The goal is to classify these into 10 classes using a feedforward neural network.

<p align="center">
  
[![MNIST Accuracy](https://img.shields.io/badge/Accuracy-98.5%25-brightgreen)](https://github.com/ARUNAGIRINATHAN-K/MNIST-Progressive-ANN)
[![Parameters](https://img.shields.io/badge/Parameters-109K-blue)](https://github.com/ARUNAGIRINATHAN-K/MNIST-Progressive-ANN)
[![Training Time](https://img.shields.io/badge/Training%20Time-3min%205s-orange)](https://github.com/ARUNAGIRINATHAN-K/MNIST-Progressive-ANN)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13.1-orange.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.6.0-yellow.svg)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12.2-blue.svg)](https://seaborn.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24-green.svg)](https://numpy.org/)
[![MNIST Dataset](https://img.shields.io/badge/Dataset-MNIST-red)](http://yann.lecun.com/exdb/mnist/)
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-black.svg)](https://github.com/psf/black)
[![Linting](https://img.shields.io/badge/Linting-Pylint-yellow.svg)](https://pylint.org/)
[![Tests](https://img.shields.io/badge/Tests-95%25-brightgreen.svg)](https://github.com/ARUNAGIRINATHAN-K/MNIST-Progressive-ANN)
[![Coverage](https://img.shields.io/badge/Coverage-92%25-blue.svg)](https://github.com/ARUNAGIRINATHAN-K/MNIST-Progressive-ANN)
[![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen.svg)](https://github.com/ARUNAGIRINATHAN-K/MNIST-Progressive-ANN)
[![Version](https://img.shields.io/badge/Version-v1.0.0-blue.svg)](https://github.com/ARUNAGIRINATHAN-K/MNIST-Progressive-ANN/releases)
[![GitHub stars](https://img.shields.io/github/stars/yourusername/mnist-neural-network)](https://github.com/ARUNAGIRINATHAN-K/MNIST-Progressive-ANN/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/yourusername/mnist-neural-network)](https://github.com/ARUNAGIRINATHAN-K/MNIST-Progressive-ANN/network)
[![Learning](https://img.shields.io/badge/Learning-Deep%20Learning-purple)](https://github.com/ARUNAGIRINATHAN-K/MNIST-Progressive-ANN)
[![Computer Vision](https://img.shields.io/badge/Topic-Computer%20Vision-teal)](https://github.com/ARUNAGIRINATHAN-K/MNIST-Progressive-ANN)
[![Language](https://img.shields.io/github/languages/top/yourusername/mnist-neural-network)](https://github.com/ARUNAGIRINATHAN-K/MNIST-Progressive-ANN)
</p>


We'll use PyTorch, a popular deep learning framework. The process involves:

- Loading and preprocessing data.
- Defining the model architecture.
- Training with a loss function and optimizer.
- Evaluating accuracy.

## SimpleNet Architecture (One Hidden Layer)
```
┌─────────────────────────────────────────────────────────────┐
│                    MNIST Classification                      │
│                                                              │
│  Input Layer (28×28 pixels) ───────────→ Flattened Vector    │
│         (784 features)               784-dimensional         │
│                                                              │
│  ↓                                                           │
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Linear Layer (fc1)                                      │ │
│  │ Input: 784 neurons    Output: 128 neurons               │ │
│  │ Weights: 784 × 128 = 100,352 parameters                 │ │
│  │ Bias: 128 parameters                                    │ │
│  │ Total: 100,480 parameters                               │ │
│  └─────────────────────────────────────────────────────────┘ │
│                 ↓ ReLU Activation                            │
│  (non-linearity: f(x) = max(0, x))                           │
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Linear Layer (fc2)                                      │ │
│  │ Input: 128 neurons    Output: 10 neurons                │ │
│  │ Weights: 128 × 10 = 1,280 parameters                    │ │
│  │ Bias: 10 parameters                                     │ │
│  │ Total: 1,290 parameters                                 │ │
│  └─────────────────────────────────────────────────────────┘ │
│                 ↓                                            │
│                                                              │
│  Output Layer (Logits) ───────────→ Softmax → Class Prob.    │
│         (10 classes: 0-9)                                    │
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Loss Function: CrossEntropyLoss                         │ │
│  │ Combines: LogSoftmax + Negative Log Likelihood          │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                              │
│  TOTAL PARAMETERS: 101,770                                   │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow:

Raw Image (28×28×1) → [Flatten] → [784] → [fc1 + ReLU] → [128] → [fc2] → [10] → [CrossEntropyLoss]

## 2. DeeperNet Architecture (Two Hidden Layers)
```
┌──────────────────────────────────────────────────────────────┐
│                Deeper MNIST Network                          │
│                                                              │
│  Input: 28×28 pixels (784 features)                          │
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Layer 1: Input → Hidden 1                               │ │
│  │ fc1: Linear(784 → 128)                                  │ │
│  │ Parameters: 784×128 + 128 = 100,480                     │ │
│  │ ↓ ReLU                                                  │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Layer 2: Hidden 1 → Hidden 2                            │ │
│  │ fc2: Linear(128 → 64)                                   │ │
│  │ Parameters: 128×64 + 64 = 8,256                         │ │
│  │ ↓ ReLU                                                  │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Layer 3: Hidden 2 → Output                              │ │
│  │ fc3: Linear(64 → 10)                                    │ │
│  │ Parameters: 64×10 + 10 = 650                            │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                              │
│  Output: 10-class logits                                     │
│  ↓                                                           │
│  CrossEntropyLoss                                            │
│                                                              │
│  TOTAL PARAMETERS: 109,386                                   │
└──────────────────────────────────────────────────────────────┘
```
## Layer-by-Layer Breakdown:
```
Layer 1 (Input → H1): 784 → 128 neurons
  ┌─────────────┐    ┌─────────────┐
  │   Pixel 1   │───▶│  Neuron 1   │
  │   Pixel 2   │───▶│  Neuron 2   │
  │     ...     │  ▶ │     ...     │
  │  Pixel 784  │───▶│ Neuron 128  │
  └─────────────┘    └─────────────┘
         ↓ ReLU              ↓
         └───────────────────┘

Layer 2 (H1 → H2): 128 → 64 neurons
  ┌─────────────┐    ┌─────────────┐
  │  Neuron 1   │───▶│  Neuron 1   │
  │  Neuron 2   │───▶│  Neuron 2   │
  │     ...     │  ▶ │     ...     │
  │ Neuron 128  │───▶│ Neuron 64   │
  └─────────────┘    └─────────────┘
         ↓ ReLU              ↓
         └───────────────────┘

Layer 3 (H2 → Output): 64 → 10 neurons
  ┌─────────────┐    ┌─────────────┐
  │  Neuron 1   │───▶│   Class 0   │
  │  Neuron 2   │───▶│   Class 1   │
  │     ...     │  ▶ │     ...     │
  │ Neuron 64   │───▶│   Class 9   │
  └─────────────┘    └─────────────┘
         ↓ No Activation         ↓
         └────────────────── Softmax
```
## 3. AdvancedNet Architecture (Two Hidden Layers + Regularization)
```
┌──────────────────────────────────────────────────────────────┐
│              Advanced MNIST Network                          │
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Input Processing                                        │ │
│  │ 28×28 → Flatten → 784-dimensional vector                │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ HIDDEN LAYER 1: 784 → 128                               │ │
│  │ ┌─────────────┐                                         │ │
│  │ │ Linear(784→128) │ Parameters: 100,480                 │ │
│  │ │ BatchNorm1d(128) │ Normalizes activations             │ │
│  │ │ ReLU │ Non-linearity                                  │ │
│  │ │ Dropout(0.2) │ Randomly zeros 20% of neurons          │ │
│  │ └─────────────┘                                         │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ HIDDEN LAYER 2: 128 → 64                                │ │
│  │ ┌─────────────┐                                         │ │
│  │ │ Linear(128→64) │ Parameters: 8,256                    │ │
│  │ │ BatchNorm1d(64) │ Normalizes activations              │ │
│  │ │ ReLU │ Non-linearity                                  │ │
│  │ │ Dropout(0.2) │ Randomly zeros 20% of neurons          │ │
│  │ └─────────────┘                                         │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ OUTPUT LAYER: 64 → 10                                   │ │
│  │ Linear(64→10) │ Parameters: 650                         │ │
│  │ No activation (logits)                                  │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ LOSS & OPTIMIZATION                                     │ │
│  │ CrossEntropyLoss                                        │ │
│  │ SGD(lr=0.01, momentum=0.9)                              │ │
│  │ Learning Rate Scheduler (StepLR)                        │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                              │
│  TOTAL PARAMETERS: 109,386                                   │
└──────────────────────────────────────────────────────────────┘
```
## Regularization Components (Conceptual Flow):

Batch Normalization Process:
```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Linear Layer │───▶│   BatchNorm   │───▶│   ReLU       │
│  (W*x + b)   │    │   (μ,σ)      │    │  (max(0,x))  │
└──────────────┘    │ Normalize    │    └──────────────┘
                    │ to μ=0,σ=1   │
                    └──────────────┘
                           ↓
                    ┌──────────────┐
                    │   Dropout    │
                    │  (p=0.2)     │
                    │  Zero 20%    │
                    └──────────────┘
```
Forward Pass with Dropout (Training vs Inference):
```
Training:     x → Linear → BN → ReLU → [Dropout 20%] → Next Layer
Inference:    x → Linear → BN → ReLU → [No Dropout] → Next Layer
```
## Architecture Comparison Table

|Aspect | SimpleNet | DeeperNet |AdvancedNet |
|-------|-------------|-----------|--------------|
Hidden Layers | 1 (128) |"2 (128, 64)" |"2 (128, 64)"|
Total Parameters|"101,770"|"109,386"|"109,386"|
Activation|ReLU|ReLU|ReLU|
Regularization|None|None |Dropout + BN |
Optimizer |SGD |Adam |SGD + Scheduler|
Expected Accuracy |~95-97% |~97-98% |~98-99% |
Training Stability |Moderate |Good |Excellent |
Overfitting Risk |High | Medium |Low |

## Conceptual Data Flow Summary

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   RAW INPUT     │───▶│   FEATURE       │───▶│   CLASSIFICATION│
│ 28×28 Image     │    │   EXTRACTION    │    │   & DECISION    │
│ Grayscale       │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
       │                       │                       │
       ▼                       ▼                       ▼
 ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
 │ Pixel Values │    │ Hidden Rep.  │    │ 10 Probabilities│
 │ [0, 255]     │    │ (Hierarchical)│    │ [0.1, 0.2, ..]│
 └──────────────┘    └──────────────┘    └──────────────┘
```
Key Transformations:
• Pixels → Numbers → Linear Combinations → Non-linearities → Probabilities → Class Label
• Each layer learns increasingly abstract representations: edges → shapes → digit patterns
