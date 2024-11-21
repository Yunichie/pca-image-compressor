import numpy as np
from typing import Dict

def calculate_compression_quality(original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, float]:
    """
    Calculate compression quality metrics.

    Args:
        original: Original image array
        reconstructed: Reconstructed image array

    Returns:
        Dictionary containing MSE, PSNR, and SSIM metrics
    """
    # Calculate MSE
    mse = np.mean((original - reconstructed) ** 2)

    # Calculate PSNR
    psnr = 20 * np.log10(255 / np.sqrt(mse)) if mse > 0 else float('inf')

    # Calculate SSIM (simplified version)
    def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2

        mean1 = np.mean(img1)
        mean2 = np.mean(img2)
        std1 = np.std(img1)
        std2 = np.std(img2)
        cov = np.mean((img1 - mean1) * (img2 - mean2))

        ssim = ((2 * mean1 * mean2 + c1) * (2 * cov + c2)) / \
               ((mean1 ** 2 + mean2 ** 2 + c1) * (std1 ** 2 + std2 ** 2 + c2))
        return ssim

    ssim = compute_ssim(original, reconstructed)

    return {
        'MSE': mse,
        'PSNR': psnr,
        'SSIM': ssim
    }