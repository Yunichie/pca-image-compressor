# build_spec.py
import sys
import os
import platform

from PyInstaller.building.api import PYZ, EXE
from PyInstaller.building.build_main import Analysis
from PyInstaller.building.osx import BUNDLE

block_cipher = None

# Specify OS-specific excludes to reduce size
excludes = [
    'tkinter', 'unittest', 'email', 'html', 'http', 'xml',
    'pydoc', 'doctest', 'argparse', 'datetime', 'zipfile',
    'urllib', 'threading', 'logging', 'distutils'
]

# Hidden imports needed for the app
hidden_imports = [
    'numpy',
    'PIL',
    'PIL._imagingtk',
    'PIL._tkinter_finder',
    'PyQt5',
    'matplotlib'
]

# Project root directory (where main.py is located)
project_root = os.path.dirname(os.path.dirname(__file__))

a = Analysis(
    [os.path.join(project_root, 'main.py')],
    pathex=[project_root],
    binaries=[],
    datas=[],
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False
)

# Remove unnecessary binaries/data
def remove_from_list(source_list, patterns):
    for file_name in list(source_list):
        for pattern in patterns:
            if pattern in str(file_name):
                source_list.remove(file_name)
                break

# Remove unnecessary Qt plugins and files to reduce size
remove_from_list(a.binaries, [
    'Qt5DBus',
    'Qt5Network',
    'Qt5Qml',
    'Qt5Quick',
    'Qt5Svg',
    'Qt5Designer',
    'libGL',
])

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Define executable
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='ImageCompressor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,  # Strip binaries
    upx=True,  # Enable UPX compression
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    icon=None  # Add icon path if you have one
)

# Add macOS bundle for .app
if platform.system() == 'Darwin':
    app = BUNDLE(
        exe,
        name='ImageCompressor.app',
        icon=None,  # Add .icns file path for macOS
        bundle_identifier=None,
        info_plist={
            'NSHighResolutionCapable': 'True',
            'LSBackgroundOnly': 'False',
            'CFBundleName': 'ImageCompressor',
            'CFBundleDisplayName': 'PCA Image Compressor',
            'CFBundleShortVersionString': '1.0.0',
        },
    )