import numpy as np
from PIL import Image
from typing import Tuple, Dict, Optional

class PCAImageCompressor:
    """
    A class for compressing grayscale images using Principal Component Analysis (PCA).

    This implementation uses eigenvalue decomposition to reduce the dimensionality
    of image data while preserving as much variance as possible.
    """

    def __init__(self):
        """Initialize the PCA Image Compressor with empty state"""
        self.eigenvectors: Optional[np.ndarray] = None
        self.eigenvalues: Optional[np.ndarray] = None  # Added eigenvalues attribute
        self.mean: Optional[np.ndarray] = None
        self.original_shape: Optional[Tuple[int, int]] = None
        self.n_components: Optional[int] = None
        self.patch_size: Optional[int] = None

    def _validate_image(self, image_data: np.ndarray) -> None:
        """
        Validate the input image data.

        Args:
            image_data: Input image array

        Raises:
            ValueError: If image data is invalid
        """
        if not isinstance(image_data, np.ndarray):
            raise ValueError("Image data must be a numpy array")
        if len(image_data.shape) != 2:
            raise ValueError("Image must be grayscale (2D array)")

    def _standardize(self, image_data: np.ndarray) -> np.ndarray:
        """
        Center the data by subtracting the mean.

        Args:
            image_data: Input image array

        Returns:
            Centered image data
        """
        self.mean = np.mean(image_data, axis=0)
        return image_data - self.mean

    def _calculate_explained_variance_ratio(self) -> np.ndarray:
        """
        Calculate the ratio of variance explained by each component.

        Returns:
            Array of explained variance ratios
        """
        if self.eigenvalues is None:
            raise ValueError("No eigenvalues available. Must compress an image first.")
        return self.eigenvalues / np.sum(self.eigenvalues)

    def _extract_patches(self, image_data: np.ndarray) -> np.ndarray:
        """
        Extract overlapping patches from the image using the specified patch size.

        Args:
            image_data: Input image array

        Returns:
            Array of patches
        """
        h, w = image_data.shape
        patch_size = self.patch_size
        patches = []
        for i in range(0, h - patch_size + 1):
            for j in range(0, w - patch_size + 1):
                patch = image_data[i:i + patch_size, j:j + patch_size]
                patches.append(patch)
        return np.array(patches)

    def _reconstruct_image_from_patches(self, patches: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Reconstruct the image from patches.

        Args:
            patches: Array of image patches
            image_shape: Original image shape

        Returns:
            Reconstructed image array
        """
        h, w = image_shape
        patch_size = self.patch_size
        reconstructed_image = np.zeros((h, w))
        patch_counts = np.zeros((h, w))

        idx = 0
        for i in range(0, h - patch_size + 1):
            for j in range(0, w - patch_size + 1):
                reconstructed_image[i:i + patch_size, j:j + patch_size] += patches[idx]
                patch_counts[i:i + patch_size, j:j + patch_size] += 1
                idx += 1

        # Average overlapping regions
        reconstructed_image /= patch_counts
        return reconstructed_image

    def compress(self, image_path: str, n_components: int, patch_size: int) -> Tuple[np.ndarray, float]:
        """
        Compress the image using PCA.

        Args:
            image_path: Path to the image file
            n_components: Number of principal components to keep

        Returns:
            Tuple containing:
                - Compressed image data
                - Compression ratio

        Raises:
            ValueError: If n_components is invalid or image cannot be processed
        """
        # Load and convert image to grayscale
        try:
            image = Image.open(image_path).convert('L')
        except Exception as e:
            raise ValueError(f"Failed to load image: {str(e)}")

        image_data = np.array(image).astype(np.float64)
        self._validate_image(image_data)

        # Store the original shape and patch size
        self.original_shape = image_data.shape
        self.n_components = n_components
        self.patch_size = patch_size

        # Extract patches using the specified patch size
        patches = self._extract_patches(image_data)

        # Flatten patches to create the data matrix X
        X = patches.reshape(patches.shape[0], -1)

        # Standardize the data
        X_centered = self._standardize(X)

        # Calculate covariance matrix with normalization by 1/n
        n_samples = X_centered.shape[0]
        covariance_matrix = (X_centered.T @ X_centered) / n_samples

        # Calculate eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Sort eigenvalues and eigenvectors in descending order
        idx = eigenvalues.argsort()[::-1]
        self.eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Store top n_components eigenvectors
        self.eigenvectors = eigenvectors[:, :n_components]

        # Project data onto eigenvectors
        compressed_data = X_centered @ self.eigenvectors

        # Calculate compression ratio
        original_size = X.size
        compressed_size = compressed_data.size + self.eigenvectors.size
        compression_ratio = original_size / compressed_size

        return compressed_data, compression_ratio

        # except np.linalg.LinAlgError as e:
        #     raise ValueError(f"Linear algebra computation failed: {str(e)}")

    def decompress(self, compressed_data: np.ndarray) -> np.ndarray:
        """
        Decompress the image data.

        Args:
            compressed_data: Compressed image data

        Returns:
            Reconstructed image array

        Raises:
            ValueError: If compression hasn't been performed or data is invalid
        """
        if any(attr is None for attr in [self.eigenvectors, self.mean, self.original_shape]):
            raise ValueError("Must compress an image before decompressing")

        try:
            # Reconstruct the data
            X_reconstructed = compressed_data @ self.eigenvectors.T
            X_reconstructed += self.mean

            # Reshape reconstructed data back into patches
            patch_size = self.patch_size
            patches = X_reconstructed.reshape(-1, patch_size, patch_size)

            # Reconstruct image from patches
            reconstructed_image = self._reconstruct_image_from_patches(patches, self.original_shape)

            # Clip values to valid pixel range
            reconstructed_image = np.clip(reconstructed_image, 0, 255)

            return reconstructed_image.astype(np.uint8)

        except Exception as e:
            raise ValueError(f"Failed to decompress image: {str(e)}")

    def get_compression_stats(self) -> Dict[str, float]:
        """
        Get statistics about the current compression.

        Returns:
            Dictionary containing compression statistics

        Raises:
            ValueError: If no compression has been performed
        """
        if self.eigenvalues is None or self.n_components is None:
            raise ValueError("No compression has been performed yet")

        explained_variance_ratios = self._calculate_explained_variance_ratio()

        return {
            'n_components': self.n_components,
            'explained_variance_ratio': np.sum(explained_variance_ratios[:self.n_components]),
            'individual_ratios': explained_variance_ratios[:self.n_components],
            'total_eigenvalues': len(self.eigenvalues)
        }