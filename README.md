# Image Colorization Project with GANs 🎨🖌️ 

## Project Overview

This project aims to implementing image colorization using Generative Adversarial Networks (GANs) with a U-Net architecture. worked with a curated dataset of grayscale images paired with their corresponding colored images, also implementing the process of training a GAN to predict colors for grayscale inputs.

### Objectives
- Exploring the architecture and capabilities of U-Net GANs.
- Training a U-Net GAN model on a custom dataset for colorization tasks.
- Evaluating the performance of the trained model.

## Repository Structure

- `src/`: Contains the source code for the project.
  - `data.py`: Custom data loader for handling the image colorization dataset.
  - `ganloss.py`: Implementation of GAN loss functions.
  - `models.py`: Implementation of the U-Net architecture for the generator and discriminator.
  - `weight_initializer.py`: Script for initializing model weights.
- `images/`: Directory containing the images for training and testing.

