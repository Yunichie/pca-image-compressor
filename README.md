# PCA Image Compressor

A desktop application demonstrating Principal Component Analysis (PCA) for image compression, specifically designed to show the mathematical concepts in linear algebra.

## Overview

This application visualizes how eigenvalues and eigenvectors can be used for dimensionality reduction and data compression through PCA. It serves as a practical demonstration of key linear algebra concepts including:

- Matrix decomposition
- Eigenvalues and eigenvectors
- Covariance matrices
- Dimensionality reduction
- Linear transformations

The application supports compression of both grayscale and color images, allowing users to explore how PCA can be applied to different types of image data.

## How It Works

The compression algorithm in `PCAImageCompressor` follows these steps:

1. **Image Preparation**
    - Loads the image, converting it to grayscale or RGB format as needed.
    - Reshapes the 2D (grayscale) or 3D (color) image matrix into a data matrix where each row represents a pixel or a patch of pixels.

2. **Data Standardization**
    - Computes the mean of the pixel values.
    - Centers the data by subtracting the mean.

3. **Eigendecomposition**
    - Calculates the covariance matrix of the centered data.
    - Computes eigenvalues and eigenvectors using `np.linalg.eigh`.
    - Sorts eigenvectors by descending eigenvalue magnitude.

4. **Dimensionality Reduction**
    - Selects top N principal components (eigenvectors).
    - Projects data onto the reduced eigenvector basis for each color channel if applicable.

5. **Reconstruction**
    - Projects compressed data back to the original space.
    - Adds the mean back to recover pixel values.
    - Reassembles the image from the reconstructed data.

## Mathematical Foundation

### 1. Data Preparation
- For a grayscale image, input image matrix $I_{m \times n}$ is converted to data matrix $X$, where each row represents a pixel or a patch.
- For a color image, each color channel (Red, Green, Blue) is processed separately.
- Data is centered by subtracting the mean: $X_{\text{centered}} = X - \mu$.

### 2. Covariance Matrix
- Compute covariance matrix: $C = \frac{1}{n} X_{\text{centered}}^T X_{\text{centered}}$.
- $C$ is symmetric and positive semi-definite.

### 3. Eigendecomposition
- Find eigenvalues $\lambda_i$ and eigenvectors $v_i$ solving: $C v_i = \lambda_i v_i$.
- Sort eigenvalues in descending order: $\lambda_1 \geq \lambda_2 \geq \dots \geq \lambda_n$.
- Corresponding eigenvectors form an orthonormal basis.

### 4. Dimensionality Reduction
- Select top $k$ eigenvectors to form projection matrix $P_k$.
- Project data: $X_{\text{compressed}} = X_{\text{centered}} P_k$.
- For color images, this process is applied to each color channel independently.

### 5. Reconstruction
- Reconstruct data: $X_{\text{reconstructed}} = X_{\text{compressed}} P_k^T + \mu$.
- Reassemble the image from the reconstructed data.

### 6. Quality Metrics
- Mean Squared Error (MSE): $MSE = \frac{1}{mn} \sum (X - X_{reconstructed})^2$.
- Peak Signal-to-Noise Ratio (PSNR): $PSNR = 20 \log_{10} \left( \frac{MAX_I}{\sqrt{\text{MSE}}} \right)$.
- Structural Similarity Index (SSIM): Measures structural similarity between images.

### 7. Compression Analysis
- Explained variance ratio: $\frac{\lambda_i}{\sum \lambda_j}$.
- Cumulative explained variance: $\sum_{i=1}^{k} \frac{\lambda_i}{\sum \lambda_j}$.
- Compression ratio: $\frac{\text{Original Size}}{\text{Compressed Size}}$.

## Key Linear Algebra Concepts

1. **Matrix Operations**
    - Matrix multiplication
    - Transpose
    - Covariance calculation

2. **Eigendecomposition**
    - Eigenvalues represent variance along principal components
    - Eigenvectors form an orthogonal basis for data projection

3. **Dimensionality Reduction**
    - Linear transformations
    - Basis change
    - Projection onto subspace

## Features

- Compress both grayscale and color images using PCA.
- Visualization of the eigenvalue spectrum.
- Real-time compression with adjustable components.
- Analysis of explained variance.
- Quality metrics calculation (MSE, PSNR, SSIM).
- Compression ratio estimation.

## Requirements

- numpy
- Pillow
- PyQt5
- matplotlib
- scikit-image

## License

[MIT License](https://spdx.org/licenses/MIT.html)