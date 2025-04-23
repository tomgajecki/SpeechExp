# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules
import matlab

block_cipher = None

# Get MATLAB paths
matlab_root = "C:\\Program Files\\Matlab"
matlab_bin = os.path.join(matlab_root, 'bin', 'win64')
matlab_runtime = os.path.join(matlab_root, 'runtime', 'win64')
matlab_engine = os.path.join(matlab_root, 'extern', 'engines', 'python')

# Collect MATLAB binaries
matlab_binaries = []
for folder in [matlab_bin, matlab_runtime]:
    if os.path.exists(folder):
        for file in os.listdir(folder):
            if file.endswith('.dll'):
                matlab_binaries.append((os.path.join(folder, file), '.'))

# Collect all necessary data files
datas = [
    ('assets', 'assets'),
    ('core/matlab', 'core/matlab'),
    ('hsm/audio', 'hsm/audio'),
    ('models', 'models'),
    (matlab_engine, 'matlab'),
]

# Collect all necessary hidden imports
hiddenimports = [
    'matlab',
    'matlab.engine',
    'torchaudio',
    'torch',
    'numpy',
    'PyQt5',
    'PyQt5.QtCore',
    'PyQt5.QtGui',
    'PyQt5.QtWidgets',
] + collect_submodules('torch') + collect_submodules('matlab')

a = Analysis(
    ['main.py'],
    pathex=[
        os.path.abspath(SPECPATH),
        os.path.join(matlab_root, 'extern', 'engines', 'python'),
        os.path.join(matlab_root, 'extern', 'bin', 'win64'),
    ],
    binaries=matlab_binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='SpeechExp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Keep True for now to see any errors
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/icon.ico'
) 