# ICAIF24: Diffusion Model for Financial Time Series with TailGAN Loss
This project implements a diffusion model for generating high-fidelity intraday log-return series, optimized for financial risk metrics such as Value at Risk (VaR) and Expected Shortfall (ES). By incorporating the TailGAN loss, this approach focuses on accurate tail distribution modeling, which is critical in finance for risk assessment and stress testing.

## Introduction
Traditional generative models often fail to accurately capture the tail behavior of financial returns, which can lead to underestimation of risk. This project combines a diffusion model framework with a TailGAN-inspired loss function to better capture these extreme values and provide more robust modeling of VaR and ES.

## Features
- Diffusion Model: A diffusion-based approach that models the forward and reverse processes for generating realistic time-series data.
- TailGAN Loss: Custom loss function designed to optimize tail distributions, improving the accuracy of VaR and ES estimations.
- Flexible Architecture: Easily adjustable network dimensions and configurations to accommodate different types of financial time series.

## Model Architecture
The core model is a score-based diffusion model, built with PyTorch, where the loss function is adapted from TailGAN to enhance the focus on tail distribution.

### Main Components:
- Score Network: A neural network to estimate the gradient of the data distribution.
- Diffusion Process: A forward and reverse process that progressively transforms data distributions.
- TailGAN Loss: Custom loss function that emphasizes the model’s ability to capture tail behavior, crucial for risk assessment.
