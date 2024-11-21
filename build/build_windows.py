# build_windows.py
import os
import subprocess
import urllib.request
import zipfile
import shutil
import sys


def download_upx():
    """Download and setup UPX"""
    # Latest stable UPX version
    UPX_VERSION = "4.2.1"
    UPX_URL = f"https://github.com/upx/upx/releases/download/v{UPX_VERSION}/upx-{UPX_VERSION}-win64.zip"

    print("Downloading UPX...")

    # Download with progress bar
    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = int(downloaded * 100 / total_size)
        sys.stdout.write(f"\rDownloading: {percent}%")
        sys.stdout.flush()

    try:
        # Download zip file
        zip_path = "upx.zip"
        urllib.request.urlretrieve(UPX_URL, zip_path, show_progress)
        print("\nExtracting UPX...")

        # Extract
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("upx_temp")

        # Move executable to current directory
        upx_exe = f"upx_temp/upx-{UPX_VERSION}-win64/upx.exe"
        shutil.move(upx_exe, "upx.exe")

        # Cleanup
        os.remove(zip_path)
        shutil.rmtree("upx_temp")
        print("UPX setup completed")

    except Exception as e:
        print(f"Error downloading UPX: {e}")
        sys.exit(1)


def build_windows():
    # Install UPX if not present
    if not os.path.exists('upx.exe'):
        download_upx()

    # Enable Python optimization
    os.environ['PYTHONOPTIMIZE'] = '2'

    # Build command
    cmd = [
        'pyinstaller',
        '--clean',
        '--windowed',
        '--onefile',
        '--noconfirm',
        '--upx-dir=.',
        'build_spec.py'
    ]

    subprocess.run(cmd)


if __name__ == '__main__':
    build_windows()