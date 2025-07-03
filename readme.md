# DreamUHD: Frequency Enhanced Variational Autoencoder for Ultra-High-Definition Image Restoration [AAAI 2025]

This is the official PyTorch implementation for the paper **"DreamUHD: Frequency Enhanced Variational Autoencoder for Ultra-High-Definition Image Restoration"**.

## Abstract
Existing ultra-high-definition (UHD) image restoration methods often struggle with inconsistent results due to downsampling operations. DreamUHD aims to address these challenges by leveraging the powerful latent space representation and reconstruction capabilities of Variational Autoencoders (VAEs).

We propose a frequency-enhanced VAE framework for UHD image restoration (FEVAE-UHD) that integrates frequency priors to overcome three primary challenges of applying VAEs to this task:

1. Parameter Efficiency: High-performance VAEs have a large number of parameters. We design a Fourier-based lightweight frequency learning mechanism (FE-VAE) to significantly reduce parameter and computational costs.

2. Domain Gap: A VAE pre-trained on clean images encounters a domain gap when processing degraded images. We introduce a wavelet-based adapter (WTA) that integrates degraded image information into the pre-trained VAE via frequency-aware adaptive modulation.

3. High-Frequency Detail Loss: The latent encoding process in a VAE can lead to the loss of high-frequency information. Our adapter injects high-frequency details into the VAE decoder to enhance the restored images.

Our method surpasses state-of-the-art approaches both qualitatively and quantitatively across various UHD image restoration tasks, including deblurring, dehazing, low-light enhancement, and demoiréing.

## Framework Overview

The FEVAE-UHD framework consists of three main components: a frozen Frequency-Enhanced VAE (FE-VAE), a Wavelet Transform-based Adapter (WTA), and an arbitrary image restoration network (IRNet) that operates within the latent space.



Figure: An overview of the FEVAE-UHD framework.

Key Features
VAE-based UHD Restoration Framework: To the best of our knowledge, this is the first work to introduce VAEs into the UHD image restoration task, utilizing their powerful latent space representation to improve consistency.

Frequency-Enhanced VAE (FE-VAE): Employs novel Space-Frequency Adaptive Decomposition (SFAD) and Frequency-Aware Feature Extraction (FAFE) modules to significantly reduce computational and parameter costs while maintaining strong representational capability.

Wavelet Transform-based Adapter (WTA): A plug-and-play module that supplements high-frequency details and bridges the domain gap without altering the original VAE, through frequency replay encoding and high-frequency injection decoding.

State-of-the-Art Performance: Achieves superior results on multiple UHD image restoration benchmarks, including dehazing, deblurring, low-light enhancement, and demoiréing.

Results Showcase
Our method generates high-quality visual results across various UHD restoration tasks.

Image Deblurring (UHD-Blur)

FEVAE-UHD restores the sharpest details, such as text on walls and license plate numbers.

Image Dehazing (UHD-Haze)

FEVAE-UHD generates results with minimal haze residue.

Low-Light Image Enhancement (UHD-LL)

FEVAE-UHD produces results with the most natural color fidelity.

Installation
Clone this repository:

git clone https://github.com/lyd-2022/dreamUHD.git
cd dreamUHD

Create and activate a Conda environment:

conda create -n dreamuhd python=3.8
conda activate dreamuhd

Install the dependencies:

pip install -r requirements.txt

How to Use
Dataset Preparation
Please follow the instructions in the ./datasets directory to prepare the datasets.

Training
Our model is trained using a two-stage strategy:

Stage 1: Train the FE-VAE

# Run the script to train the Frequency-Enhanced VAE
python train_fe_vae.py --config configs/fe_vae_train.yaml

Stage 2: Train the Full Restoration Model

# Freeze FE-VAE weights and train the adapter and IRNet
python train_restoration.py --config configs/deblurring_train.yaml

Testing
# Evaluate the model on the UHD-Blur test set
python test.py --config configs/deblurring_test.yaml --weights /path/to/your/checkpoint.pth

Citation
If you use our work in your research, please cite our paper:

@inproceedings{liu2025dreamuhd,
  title={DreamUHD: Frequency Enhanced Variational Autoencoder for Ultra-High-Definition Image Restoration},
  author={Liu, Yidi and Li, Dong and Xiao, Jie and Bao, Yuanfei and Xu, Senyan and Fu, Xueyang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025}
}

Acknowledgements
This work was supported by the National Natural Science Foundation of China (NSFC) under Grants 62225207, 62436008, 62422609, and 62276243.