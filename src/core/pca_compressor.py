import numpy as np
from PIL import Image
from typing import Tuple, Dict, Optional, Any


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
        if image_data.ndim == 2:
            # Grayscale image
            pass
        elif image_data.ndim == 3 and image_data.shape[2] in [3, 4]:
            # Color image (RGB or RGBA)
            pass
        else:
            raise ValueError("Image must be a 2D grayscale or 3D color image")

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

    def _reconstruct_image_from_patches(self, patches: np.ndarray, image_shape: Tuple[int, ...]) -> np.ndarray:
        """
        Reconstruct the image from patches.

        Args:
            patches: Array of image patches
            image_shape: Original image shape (h,w) or (h,w,c)

        Returns:
            Reconstructed image array
        """
        h, w = image_shape[:2]
        patch_size = self.patch_size
        reconstructed_image = np.zeros((h, w))  # Always 2D since we handle channels separately
        patch_counts = np.zeros((h, w))

        idx = 0
        for i in range(0, h - patch_size + 1):
            for j in range(0, w - patch_size + 1):
                reconstructed_image[i:i + patch_size, j:j + patch_size] += patches[idx]
                patch_counts[i:i + patch_size, j:j + patch_size] += 1
                idx += 1

        # Avoid division by zero
        mask = patch_counts > 0
        reconstructed_image[mask] /= patch_counts[mask]

        return reconstructed_image

    def compress(self, image_path: str, n_components: int, patch_size: int, to_grayscale: bool = False) -> Tuple[Any, float]:
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
            image = Image.open(image_path)
            if to_grayscale:
                image = image.convert('L')
            else:
                image = image.convert('RGB')  # Ensure image is in RGB format
        except Exception as e:
            raise ValueError(f"Failed to load image: {str(e)}")

        image_data = np.array(image).astype(np.float64)
        self._validate_image(image_data)

        self.original_shape = image_data.shape
        self.n_components = n_components
        self.patch_size = patch_size

        if image_data.ndim == 2:
            # Grayscale image processing
            patches = self._extract_patches(image_data)
            X = patches.reshape(patches.shape[0], -1)
            X_centered = self._standardize(X)
            n_samples = X_centered.shape[0]
            covariance_matrix = (X_centered.T @ X_centered) / n_samples

            eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            self.eigenvalues = eigenvalues
            self.eigenvectors = eigenvectors[:, :n_components]
            self.mean = self.mean  # Already set in _standardize()

            compressed_data = X_centered @ self.eigenvectors

            # Calculate compression ratio
            original_size = X.size
            compressed_size = compressed_data.size + self.eigenvectors.size
            compression_ratio = original_size / compressed_size

            return compressed_data, compression_ratio
        else:
            # Color image processing
            # Initialize lists instead of trying to append to numpy arrays
            eigenvalues_list = []
            eigenvectors_list = []
            mean_list = []
            compressed_data = []

            for c in range(image_data.shape[2]):
                channel_data = image_data[:, :, c]
                patches = self._extract_patches(channel_data)
                X = patches.reshape(patches.shape[0], -1)
                X_centered = self._standardize(X)
                n_samples = X_centered.shape[0]
                covariance_matrix = (X_centered.T @ X_centered) / n_samples

                eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
                idx = eigenvalues.argsort()[::-1]
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]

                # Store in lists instead of appending to numpy arrays
                eigenvalues_list.append(eigenvalues)
                eigenvectors_list.append(eigenvectors[:, :n_components])
                mean_list.append(self.mean)

                compressed_c = X_centered @ eigenvectors[:, :n_components]
                compressed_data.append(compressed_c)

            # Store the lists in class attributes
            self.eigenvalues = eigenvalues_list
            self.eigenvectors = eigenvectors_list
            self.mean = mean_list

            original_size = image_data.size
            compressed_size = sum(cd.size for cd in compressed_data) + sum(ev.size for ev in eigenvectors_list)
            compression_ratio = original_size / compressed_size

            return compressed_data, compression_ratio

        # except np.linalg.LinAlgError as e:
        #     raise ValueError(f"Linear algebra computation failed: {str(e)}")

    # src/core/pca_compressor.py

    def decompress(self, compressed_data: Any) -> np.ndarray:
        """
        Decompress the image data.

        Args:
            compressed_data: Compressed image data

        Returns:
            Reconstructed image array

        Raises:
            ValueError: If compression hasn't been performed or data is invalid
        """
        if self.eigenvectors is None or self.mean is None or self.original_shape is None:
            raise ValueError("Must compress an image before decompressing")

        if len(self.original_shape) == 2:
            # Grayscale image reconstruction
            X_reconstructed = compressed_data @ self.eigenvectors.T
            X_reconstructed += self.mean
            patches = X_reconstructed.reshape(-1, self.patch_size, self.patch_size)
            reconstructed_image = self._reconstruct_image_from_patches(patches, self.original_shape)
        else:
            # Color image reconstruction
            reconstructed_channels = []
            for i in range(len(compressed_data)):
                X_reconstructed = compressed_data[i] @ self.eigenvectors[i].T
                X_reconstructed += self.mean[i]
                patches = X_reconstructed.reshape(-1, self.patch_size, self.patch_size)
                # Reconstruct each channel independently using same shape
                reconstructed_channel = self._reconstruct_image_from_patches(
                    patches, self.original_shape[:2]
                )
                reconstructed_channels.append(reconstructed_channel)

            # Stack channels after reconstruction
            reconstructed_image = np.stack(reconstructed_channels, axis=2)

        reconstructed_image = np.clip(reconstructed_image, 0, 255)
        return reconstructed_image.astype(np.uint8)

        # except Exception as e:
        #     raise ValueError(f"Failed to decompress image: {str(e)}")

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