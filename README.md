> [!TIP]
> Star the repo if you found it usefull.


# GAN Implementation in Keras and PyTorch

This repository provides implementations of a Generative Adversarial Network (GAN) using both Keras and PyTorch. The PyTorch implementation yielded better results with optimized training time.

## Project Overview

This project focuses on understanding and training GANs with a custom synthetic dataset. The dataset consists of images of the letter "T" in various forms, where the horizontal line of the "T" gradually shifts downward.

## Dataset Generation

You can easily generate your own dataset by adjusting the parameters for the number of images and the save directory:

```python
# Number of images to generate
num_images = 1000  # Adjust as needed
save_dir = 'T_dataset'
```

## Model Architecture:
### Generator
The generator is defined as a series of transpose convolutional layers that progressively upscale the noise input into a synthetic "T" image.
```
nn.Sequential(
    nn.ConvTranspose2d(nz, 256, 4, 1, 0, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(True),
    
    nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),

    nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),

    nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
    nn.Tanh()
)
```
### Discriminator
The discriminator takes the generated image and classifies it as real or fake. It uses convolutional layers and Leaky ReLU activations for effective feature extraction and classification.
```
nn.Sequential(
    nn.Conv2d(1, 64, 4, 2, 1, bias=False),
    nn.LeakyReLU(0.2, inplace=True),

    nn.Conv2d(64, 128, 4, 2, 1, bias=False),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.2, inplace=True),

    nn.Conv2d(128, 256, 4, 2, 1, bias=False),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(0.2, inplace=True),

    nn.Conv2d(256, 1, 4, 1, 0, bias=False),
    nn.Sigmoid()
)
```
## Training
You can adjust hyperparameters to experiment with the training process and enhance results:

```
# Hyperparameters
nz = 120  # Size of the latent vector (input to the generator)
lr = 0.0001  # Learning rate
batch_size = 128
num_epochs = 1200
```

## Results and Observations
>PyTorch Implementation: The PyTorch model provided superior results, with faster and more efficient training.
>Dataset Variability: By adjusting the dataset, you can explore how GANs learn diverse patterns even within small modifications, such as shifting elements of the "T" shape.

## Usage
Clone the repository:
```
git clone https://github.com/adysinghh/T--T-GAN.git
```

