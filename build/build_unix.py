# build_unix.py
import os
import platform
import subprocess


def build_unix():
    # Detect OS
    system = platform.system()

    # Base command
    cmd = [
        'pyinstaller',
        '--clean',
        '--windowed',
        '--onefile',
        '--noconfirm',
    ]

    # OS-specific additions
    if system == 'Darwin':
        cmd.extend([
            '--target-architecture', 'universal2',
            '--codesign-identity', '-',
        ])

    cmd.append('build_spec.py')

    # Run build
    subprocess.run(cmd)


if __name__ == '__main__':
    build_unix()