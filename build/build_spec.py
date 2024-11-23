#!/usr/bin/env python
import os
import platform
import subprocess


def build():
    main_script = os.path.abspath('main.py')

    # PyInstaller options
    options = [
        '--name=AdvancedPCAImageCompressor',
        '--onefile',
        '--windowed',
        '--noconfirm',
        main_script,
    ]

    # Platform-specific options
    system = platform.system()
    if system == 'Darwin':
        options += [
            # '--icon=path/to/icon.icns',
            '--codesign-identity=-',
            '--osx-bundle-identifier=com.icikiwir.advancedpcaimagecompressor',
        ]
    elif system == 'Windows':
        # options += [
        #     '--icon=path/to/icon.ico',
        # ]
        # Ensure UPX is available
        if not os.path.exists('upx.exe'):
            download_upx()
        options.append(f'--upx-dir={os.path.abspath(".")}')
    else:
        # Linux-specific options
        pass

    # Hidden imports
    hidden_imports = [
        'numpy',
        'PIL',
        'PyQt5',
        'matplotlib',
        'scikit-image'
    ]
    for hidden_import in hidden_imports:
        options.append(f'--hidden-import={hidden_import}')

    # Run PyInstaller
    cmd = ['pyinstaller'] + options
    subprocess.run(cmd)


def download_upx():
    """Download and set up UPX"""
    import urllib.request
    import zipfile
    import shutil

    UPX_VERSION = "4.2.1"
    UPX_URL = f"https://github.com/upx/upx/releases/download/v{UPX_VERSION}/upx-{UPX_VERSION}-win64.zip"
    print("Downloading UPX...")

    # Download UPX
    zip_path = "upx.zip"
    urllib.request.urlretrieve(UPX_URL, zip_path)
    print("Extracting UPX...")

    # Extract UPX
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("upx_temp")

    # Move UPX executable
    upx_exe = f"upx_temp/upx-{UPX_VERSION}-win64/upx.exe"
    shutil.move(upx_exe, "upx.exe")

    # Clean up
    os.remove(zip_path)
    shutil.rmtree("upx_temp")
    print("UPX setup completed")


if __name__ == '__main__':
    build()