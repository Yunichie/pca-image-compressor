import numpy as np
from typing import Dict
from skimage.metrics import structural_similarity as ssim

def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute the Structural Similarity Index (SSIM) between two images.

    Args:
        img1: Original image array.
        img2: Reconstructed image array.

    Returns:
        SSIM value.
    """
    if img1.ndim == 2:
        # Grayscale images
        return ssim(img1, img2, data_range=img2.max() - img2.min())
    else:
        # Color images: compute SSIM for each channel and average
        ssim_total = 0
        for i in range(img1.shape[2]):
            ssim_channel = ssim(img1[:, :, i], img2[:, :, i], data_range=img2[:, :, i].max() - img2[:, :, i].min())
            ssim_total += ssim_channel
        return ssim_total / img1.shape[2]

def calculate_compression_quality(original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Calculate compression quality metrics for each color channel.

    Args:
        original: Original image array
        reconstructed: Reconstructed image array

    Returns:
        Dictionary containing MSE, PSNR, and SSIM metrics per channel
    """
    metrics = {}
    channels = ['Red', 'Green', 'Blue'] if original.ndim == 3 else ['Grayscale']

    for idx, channel in enumerate(channels):
        orig = original[:, :, idx] if original.ndim == 3 else original
        recon = reconstructed[:, :, idx] if reconstructed.ndim == 3 else reconstructed

        mse = np.mean((orig - recon) ** 2)
        psnr = 20 * np.log10(255 / np.sqrt(mse)) if mse > 0 else float('inf')
        channel_ssim = compute_ssim(orig, recon)

        metrics[channel] = {
            'MSE': mse,
            'PSNR': psnr,
            'SSIM': channel_ssim
        }

    return metrics