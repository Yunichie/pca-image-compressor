# PCA Image Compressor

A desktop application demonstrating Principal Component Analysis (PCA) for image compression, specifically designed to show the mathematical concepts in linear algebra.

## Overview

This application visualizes how eigenvalues and eigenvectors can be used for dimensionality reduction and data compression through PCA. It serves as a practical demonstration of key linear algebra concepts including:

- Matrix decomposition
- Eigenvalues and eigenvectors
- Covariance matrices
- Dimensionality reduction
- Linear transformations

## How It Works

The compression algorithm in PCAImageCompressor follows these steps:

1. **Image Preparation**
    - Converts image to grayscale
    - Reshapes 2D image matrix into a data matrix where each row is a pixel

2. **Data Standardization**
    - Computes mean of pixel values
    - Centers data by subtracting mean

3. **Eigendecomposition**
    - Calculates covariance matrix of centered data
    - Computes eigenvalues and eigenvectors using [np.linalg.eigh](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html)
    - Sorts eigenvectors by descending eigenvalue magnitude

4. **Dimensionality Reduction**
    - Selects top N principal components (eigenvectors)
    - Projects data onto reduced eigenvector basis

5. **Reconstruction**
    - Projects compressed data back to original space
    - Adds mean back to recover pixel values

## Mathematical Foundation

### 1. Data Preparation
- Input image matrix $I_{m×n}$ is converted to data matrix $X$ where each row represents a pixel
- Data is centered by subtracting mean: $X_{centered} = X - \mu$

### 2. Covariance Matrix
- Compute covariance matrix: $C = \frac{1}{n}X_{centered}^TX_{centered}$
- $C$ is symmetric positive semidefinite, size $n×n$

### 3. Eigendecomposition
- Find eigenvalues $\lambda_i$ and eigenvectors $v_i$ solving: $Cv_i = \lambda_iv_i$
- Sort eigenvalues in descending order: $\lambda_1 ≥ \lambda_2 ≥ ... ≥ \lambda_n$
- Corresponding eigenvectors form orthonormal basis

### 4. Dimensionality Reduction
- Select top $k$ eigenvectors to form projection matrix $P_k$
- Project data: $X_{compressed} = X_{centered}P_k$
- Reconstruction: $X_{reconstructed} = X_{compressed}P_k^T + \mu$

### 5. Quality Metrics
- Mean Squared Error (MSE): $MSE = \frac{1}{mn}\sum(X - X_{reconstructed})^2$
- Peak Signal-to-Noise Ratio: $PSNR = 20\log_{10}(\frac{MAX_I}{\sqrt{MSE}})$
- Structural Similarity Index (SSIM): Measures structural similarity between images

### 6. Compression Analysis
- Explained variance ratio: $\frac{\lambda_i}{\sum\lambda_j}$
- Cumulative explained variance: $\sum_{i=1}^k\frac{\lambda_i}{\sum\lambda_j}$
- Compression ratio: $\frac{mn}{k(m+n)}$

## Key Linear Algebra Concepts

1. **Matrix Operations**
    - Matrix multiplication
    - Transpose
    - Covariance calculation

2. **Eigendecomposition**
    - Eigenvalues represent variance along principal components
    - Eigenvectors form orthogonal basis for data projection
    - Relationship to singular value decomposition (SVD)

3. **Dimensionality Reduction**
    - Linear transformations
    - Basis change
    - Projection onto subspace

## Features
- Interactive visualization of eigenvalue spectrum
- Real-time compression with adjustable components
- Analysis of explained variance
- Quality metrics calculation
- Compression ratio estimation

## Requirements

numpy
Pillow
PyQt5
matplotlib

## License

[MIT License](https://spdx.org/licenses/MIT.html)