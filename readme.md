# DreamUHD: Frequency Enhanced Variational Autoencoder for Ultra-High-Definition Image Restoration

## Introduction
This repository provides the official implementation for the paper, "DreamUHD: Frequency Enhanced Variational Autoencoder for Ultra-High-Definition Image Restoration."  DreamUHD is a novel framework designed to address the challenges inherent in Ultra-High-Definition (UHD) image restoration. By leveraging the powerful latent space representation and reconstruction capabilities of Variational Autoencoders (VAEs) and integrating frequency priors, our method effectively restores high-quality UHD images while maintaining computational efficiency. 



## Key Features
- VAE-Based UHD Image Restoration: To the best of our knowledge, this is the first work to introduce VAEs into the domain of UHD image restoration. By operating in the compact latent space of a VAE, our framework enhances restoration consistency and significantly reduces computational overhead.
- Frequency-Enhanced VAE (FE-VAE): We propose a novel Fourier-based, frequency-enhanced VAE that is both lightweight and powerful. By incorporating the global perceptual capabilities of the Fourier domain, FE-VAE achieves a substantial reduction in parameter count and computational cost without compromising its representational power.
- Wavelet Transform-based Adapter (WTA): A wavelet-based adapter is introduced to supplement the high-frequency details essential for high-fidelity image restoration. This module effectively bridges the domain gap between the pre-trained VAE and degraded images by combining spatial and frequency information.
- State-of-the-Art Performance: Our proposed method has been extensively evaluated on a variety of UHD image restoration tasks, including low-light enhancement, image deblurring, image dehazing, and moiré pattern removal. In all cases, DreamUHD achieves state-of-the-art results, outperforming existing methods both qualitatively and quantitatively. 


## Framework Overview
The DreamUHD framework is composed of three main components:
- A frozen, pre-trained FE-VAE: This serves as the backbone of our model, providing an efficient and compact latent space for image representation. 
- A Wavelet Transform-based Adapter (WTA): This module works in tandem with the FE-VAE to inject high-frequency information and mitigate the domain gap.
- An arbitrary restoration network (IRNet): This is a lightweight network that performs the actual restoration task within the latent space. 





