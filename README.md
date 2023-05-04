# RegNet: Self-Regulated Network for Image Classification

This repository contains the implementation of the RegNet model proposed in the paper "RegNet: Self-Regulated Network for Image Classification" by Ilija Radosavovic et al.

## Introduction

RegNet is a neural network architecture designed for image classification tasks. The key idea behind RegNet is to develop a network that can self-regulate its depth, width, and resolution during training, based on the computational and memory constraints. RegNet achieves state-of-the-art performance on several image classification benchmarks while requiring fewer computational resources than other popular models.

## Requirements

* Python 3.x
* PyTorch 1.x
* torchvision
* NumPy
* tqdm
* argparse

## Datasets

The code is designed to work with the following datasets:

* CIFAR-10
* CIFAR-100
* ImageNet

To use the CIFAR-10 or CIFAR-100 datasets, simply download the datasets from https://www.cs.toronto.edu/~kriz/cifar.html and place them in a directory named 'data' in the root directory of this repository.

To use the ImageNet dataset, download the dataset from http://www.image-net.org/challenges/LSVRC/2012/nonpub-downloads and follow the instructions provided by PyTorch's ImageNet example to prepare the dataset.

## Usage

To train the model and evaluate it on the test set, follow these steps:

1. Clone the repository to your local machine:

```bash
git clone https://github.com/<username>/regnet-image-classification.git
```

2. Run the training script:

* To train a RegNetY model on CIFAR-10 or CIFAR-100, run the following command:

```bash
python train.py --dataset cifar10 --model regnety --epochs 100 --batch-size 128 --lr 0.1
```

* To train a RegNetY model on ImageNet, run the following command:

```bash
python train.py --dataset imagenet --model regnety --epochs 100 --batch-size 256 --lr 0.1
```

3. The trained model will be saved to the ./saved_models directory with a file name of regnet.pt.

4. Run the evaluation script:

```bash
python evaluate.py --model_path saved_models/regnet.pt
```

5. The test accuracy of the model will be printed to the console.
