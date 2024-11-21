#!/usr/bin/env python
import sys
from PyQt5.QtWidgets import QApplication
from src.gui.main_window import ImageCompressorGUI

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageCompressorGUI()
    window.show()
    sys.exit(app.exec_())