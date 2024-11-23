#!/usr/bin/env python
from PyQt5.QtCore import QThread, pyqtSignal
from src.core.pca_compressor import PCAImageCompressor
import numpy as np
from PIL import Image
import os
from src.utils.metrics import calculate_compression_quality

class CompressionWorker(QThread):
    """Worker thread to handle compression operations"""
    finished = pyqtSignal(tuple)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(self, image_path, n_components, patch_size, to_grayscale):
        super().__init__()
        self.image_path = image_path
        self.n_components = n_components
        self.patch_size = patch_size
        self.to_grayscale = to_grayscale
        self.compressor = PCAImageCompressor()

    def run(self):
        try:
            # Compression
            self.progress.emit(25)
            compressed_data, compression_ratio = self.compressor.compress(
                self.image_path, self.n_components, self.patch_size, self.to_grayscale
            )

            # Get compression statistics
            compression_stats = self.compressor.get_compression_stats()

            # Decompression
            self.progress.emit(75)
            reconstructed = self.compressor.decompress(compressed_data)

            # Load original image with the same mode as reconstructed
            if self.to_grayscale:
                original = np.array(Image.open(self.image_path).convert('L'))
            else:
                original = np.array(Image.open(self.image_path).convert('RGB'))

            # Calculate quality metrics
            quality_metrics = calculate_compression_quality(original, reconstructed)

            # Calculate file sizes
            original_size = os.path.getsize(self.image_path)

            # Save temporary compressed file to get its size
            temp_compressed_path = "temp_compressed.jpg"
            Image.fromarray(reconstructed).save(temp_compressed_path)
            compressed_size = os.path.getsize(temp_compressed_path)
            os.remove(temp_compressed_path)

            self.progress.emit(100)
            self.finished.emit((
                reconstructed,
                compression_ratio,
                quality_metrics,
                original_size,
                compressed_size,
                compression_stats,
                self.compressor.eigenvalues
            ))

        except Exception as e:
            self.error.emit(str(e))