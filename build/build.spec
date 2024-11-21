# build_spec.py
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
    'PyQt5',
    'matplotlib'
]

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=hidden_imports,
    hookspath=[],
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

# Remove unnecessary Qt plugins and files
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
    icon='app_icon.ico'  # Add an icon if you have one
)