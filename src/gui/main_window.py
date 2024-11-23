#!/usr/bin/env python
import sys
import os
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QSpinBox, QProgressBar, QMessageBox,
                             QTabWidget, QGridLayout, QCheckBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from src.gui.worker import CompressionWorker

class ImageCompressorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced PCA Image Compressor")
        self.setGeometry(100, 100, 1400, 900)

        # Initialize variables
        self.image_path = None
        self.compression_worker = None
        self.reconstructed = None

        self.init_ui()

    def init_ui(self):
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Create top controls layout
        controls_layout = QHBoxLayout()

        # Add load image button
        self.load_btn = QPushButton("Load Image")
        self.load_btn.clicked.connect(self.load_image)
        controls_layout.addWidget(self.load_btn)

        # Add components control
        components_layout = QHBoxLayout()
        components_label = QLabel("Principal Components:")
        self.components_spin = QSpinBox()
        self.components_spin.setRange(1, 500)
        self.components_spin.setValue(50)
        components_layout.addWidget(components_label)
        components_layout.addWidget(self.components_spin)
        controls_layout.addLayout(components_layout)

        # Add patch size control
        patch_size_layout = QHBoxLayout()
        patch_size_label = QLabel("Patch Size:")
        self.patch_size_spin = QSpinBox()
        self.patch_size_spin.setRange(1, 50)
        self.patch_size_spin.setValue(8)
        patch_size_layout.addWidget(patch_size_label)
        patch_size_layout.addWidget(self.patch_size_spin)
        controls_layout.addLayout(patch_size_layout)

        # Add grayscale checkbox
        self.grayscale_checkbox = QCheckBox("Convert to Grayscale")
        self.grayscale_checkbox.setChecked(True)
        controls_layout.addWidget(self.grayscale_checkbox)

        # Add compress button
        self.compress_btn = QPushButton("Compress")
        self.compress_btn.clicked.connect(self.compress_image)
        self.compress_btn.setEnabled(False)
        controls_layout.addWidget(self.compress_btn)

        # Add save button
        self.save_btn = QPushButton("Save Result")
        self.save_btn.clicked.connect(self.save_result)
        self.save_btn.setEnabled(False)
        controls_layout.addWidget(self.save_btn)

        main_layout.addLayout(controls_layout)

        # Add progress bar
        self.progress_bar = QProgressBar()
        main_layout.addWidget(self.progress_bar)

        # Create tab widget for different views
        tab_widget = QTabWidget()

        # Image comparison tab
        image_tab = QWidget()
        image_layout = QHBoxLayout(image_tab)

        # Original image display
        original_layout = QVBoxLayout()
        self.original_label = QLabel("Original Image")
        self.original_image = QLabel()
        self.original_image.setAlignment(Qt.AlignCenter)
        self.original_size_label = QLabel()
        original_layout.addWidget(self.original_label)
        original_layout.addWidget(self.original_image)
        original_layout.addWidget(self.original_size_label)
        image_layout.addLayout(original_layout)

        # Compressed image display
        compressed_layout = QVBoxLayout()
        self.compressed_label = QLabel("Compressed Image")
        self.compressed_image = QLabel()
        self.compressed_image.setAlignment(Qt.AlignCenter)
        self.compressed_size_label = QLabel()
        compressed_layout.addWidget(self.compressed_label)
        compressed_layout.addWidget(self.compressed_image)
        compressed_layout.addWidget(self.compressed_size_label)
        image_layout.addLayout(compressed_layout)

        tab_widget.addTab(image_tab, "Image Comparison")

        # Analysis tab
        analysis_tab = QWidget()
        analysis_layout = QGridLayout(analysis_tab)

        # Quality metrics section
        self.metrics_label = QLabel()
        self.metrics_label.setAlignment(Qt.AlignLeft)
        analysis_layout.addWidget(QLabel("Quality Metrics:"), 0, 0)
        analysis_layout.addWidget(self.metrics_label, 1, 0)

        # Compression stats section
        self.stats_label = QLabel()
        self.stats_label.setAlignment(Qt.AlignLeft)
        analysis_layout.addWidget(QLabel("Compression Statistics:"), 0, 1)
        analysis_layout.addWidget(self.stats_label, 1, 1)

        # Add eigenvalue spectrum plot
        plot_label = QLabel("Eigenvalue Spectrum")
        analysis_layout.addWidget(plot_label, 2, 0, 1, 2)

        # Create figure and canvas for plotting
        self.figure = Figure(figsize=(10, 5))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        analysis_layout.addWidget(self.canvas, 3, 0, 1, 2)

        tab_widget.addTab(analysis_tab, "Analysis")

        main_layout.addWidget(tab_widget)

    def format_size(self, size_bytes):
        """Convert bytes to human readable format"""
        for unit in ['B', 'KB', 'MB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} GB"

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "", "Images (*.png *.jpg *.bmp)"
        )

        if file_name:
            self.image_path = file_name
            # Display original image
            pixmap = QPixmap(file_name)
            scaled_pixmap = pixmap.scaled(
                500, 500, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.original_image.setPixmap(scaled_pixmap)

            # Display original file size
            original_size = os.path.getsize(file_name)
            self.original_size_label.setText(f"Size: {self.format_size(original_size)}")

            self.compress_btn.setEnabled(True)

            # Update components spinner maximum
            img = Image.open(file_name).convert('L')
            max_components = min(img.size)
            self.components_spin.setMaximum(max_components)

            # Update patch size spinner maximum
            max_patch_size = min(img.size)
            self.patch_size_spin.setMaximum(max_patch_size)

    def compress_image(self):
        if not self.image_path:
            return

        n_components = self.components_spin.value()
        patch_size = self.patch_size_spin.value()
        to_grayscale = self.grayscale_checkbox.isChecked()

        # Disable controls during compression
        self.compress_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        self.components_spin.setEnabled(False)
        self.patch_size_spin.setEnabled(False)

        # Reset progress bar
        self.progress_bar.setValue(0)

        # Start compression in worker thread
        self.compression_worker = CompressionWorker(
            self.image_path, n_components, patch_size, to_grayscale
        )
        self.compression_worker.finished.connect(self.compression_finished)
        self.compression_worker.progress.connect(self.progress_bar.setValue)
        self.compression_worker.error.connect(self.show_error)
        self.compression_worker.start()

    def compression_finished(self, result):
        (reconstructed, compression_ratio, quality_metrics,
         original_size, compressed_size, compression_stats, eigenvalues) = result

        self.reconstructed = reconstructed

        # Handle both grayscale and color images
        if reconstructed.ndim == 2:
            # Grayscale image
            h, w = reconstructed.shape
            q_img = QImage(reconstructed.data, w, h, w, QImage.Format_Grayscale8)
        else:
            # Color image
            h, w, ch = reconstructed.shape
            bytes_per_line = ch * w
            q_img = QImage(reconstructed.data.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(
            500, 500, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.compressed_image.setPixmap(scaled_pixmap)

        # Update size labels
        self.compressed_size_label.setText(f"Size: {self.format_size(compressed_size)}")

        # Update quality metrics display
        metrics_text = ""
        for channel, metrics in quality_metrics.items():
            metrics_text += (
                f"{channel} Channel:\n"
                f"  Mean Squared Error: {metrics['MSE']:.2f}\n"
                f"  Peak Signal-to-Noise Ratio: {metrics['PSNR']:.2f} dB\n"
                f"  Structural Similarity Index: {metrics['SSIM']:.3f}\n\n"
            )
        metrics_text += f"Size Reduction: {((original_size - compressed_size) / original_size * 100):.1f}%"
        self.metrics_label.setText(metrics_text)

        # Update compression statistics display
        stats_text = (
            f"Number of Components: {compression_stats['n_components']}\n"
            f"Total Explained Variance: {compression_stats['explained_variance_ratio']:.2%}\n"
            f"Compression Ratio: {compression_ratio:.2f}x\n"
            f"Total Eigenvalues: {compression_stats['total_eigenvalues']}"
        )
        self.stats_label.setText(stats_text)

        # Plot eigenvalue spectrum and explained variance
        self.plot_analysis(eigenvalues, compression_stats['individual_ratios'])

        # Re-enable controls
        self.compress_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.components_spin.setEnabled(True)
        self.patch_size_spin.setEnabled(True)
        self.save_btn.setEnabled(True)

    def plot_analysis(self, eigenvalues, explained_variance_ratios):
        self.ax.clear()

        # Create figure with two subplots
        self.figure.clear()
        gs = self.figure.add_gridspec(1, 2)
        ax1 = self.figure.add_subplot(gs[0, 0])
        ax2 = self.figure.add_subplot(gs[0, 1])

        # Check if we're dealing with grayscale or color image
        is_color = isinstance(eigenvalues, list)
        channels = ['Red', 'Green', 'Blue'] if is_color else ['Grayscale']
        colors = ['r', 'g', 'b']

        if is_color:
            # Color image - eigenvalues is a list of arrays
            for idx, channel in enumerate(channels):
                ev = eigenvalues[idx]
                ev_to_plot = ev[:50]
                ax1.plot(range(1, len(ev_to_plot) + 1), ev_to_plot,
                         color=colors[idx], label=f"{channel} Channel")

                cumulative_variance = np.cumsum(explained_variance_ratios[idx][:50])
                ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance,
                         color=colors[idx], label=f"{channel} Channel")
        else:
            # Grayscale image - eigenvalues is a single array
            ev_to_plot = eigenvalues[:50]
            ax1.plot(range(1, len(ev_to_plot) + 1), ev_to_plot,
                     color='k', label="Grayscale")

            cumulative_variance = np.cumsum(explained_variance_ratios[:50])
            ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance,
                     color='k', label="Grayscale")

        # Configure plots
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Eigenvalue')
        ax1.set_title('Eigenvalue Spectrum\n(Top 50 Components)')
        ax1.grid(True)
        ax1.legend()

        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Explained Variance')
        ax2.set_title('Cumulative Explained Variance Ratio')
        ax2.grid(True)
        ax2.set_ylim(0, 1)
        ax2.legend()

        # Update the canvas
        self.figure.tight_layout()
        self.canvas.draw()

    def save_result(self):
        if self.reconstructed is None:
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Compressed Image", "", "Images (*.png *.jpg *.bmp)"
        )

        if file_name:
            try:
                Image.fromarray(self.reconstructed).save(file_name)
                QMessageBox.information(self, "Success", "Image saved successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save image: {str(e)}")

    def show_error(self, error_message):
        QMessageBox.critical(self, "Error", error_message)
        self.compress_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.components_spin.setEnabled(True)