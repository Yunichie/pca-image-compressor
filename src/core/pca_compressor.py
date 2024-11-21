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

    def compress(self, image_path: str, n_components: int) -> Tuple[np.ndarray, float]:
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

        image_data = np.array(image)
        self._validate_image(image_data)

        # Validate n_components
        max_components = min(image_data.shape)
        if not 1 <= n_components <= max_components:
            raise ValueError(
                f"n_components must be between 1 and {max_components}"
            )

        self.original_shape = image_data.shape
        self.n_components = n_components

        # Reshape image to 2D array (pixels x features)
        X = image_data.reshape(-1, image_data.shape[1])

        # Standardize the data
        X_centered = self._standardize(X)

        try:
            # Calculate covariance matrix
            covariance_matrix = np.cov(X_centered.T)

            # Calculate eigenvectors and eigenvalues
            eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

            # Sort eigenvectors by eigenvalues in descending order
            idx = eigenvalues.argsort()[::-1]
            self.eigenvalues = eigenvalues[idx]  # Store sorted eigenvalues
            eigenvectors = eigenvectors[:, idx]

            # Store top n_components eigenvectors
            self.eigenvectors = eigenvectors[:, :n_components]

            # Project data onto eigenvectors
            compressed_data = np.dot(X_centered, self.eigenvectors)

            # Calculate compression ratio
            original_size = X.shape[0] * X.shape[1]
            compressed_size = (
                    compressed_data.shape[0] * compressed_data.shape[1] +
                    self.eigenvectors.shape[0] * self.eigenvectors.shape[1]
            )
            compression_ratio = original_size / compressed_size

            return compressed_data, compression_ratio

        except np.linalg.LinAlgError as e:
            raise ValueError(f"Linear algebra computation failed: {str(e)}")

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
            reconstructed = np.dot(compressed_data, self.eigenvectors.T)
            reconstructed = reconstructed + self.mean

            # Reshape back to original image dimensions
            reconstructed = reconstructed.reshape(self.original_shape)

            # Clip values to valid pixel range
            reconstructed = np.clip(reconstructed, 0, 255)

            return reconstructed.astype(np.uint8)

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