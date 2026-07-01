# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files

sv_ttk_datas = collect_data_files('sv_ttk')

a = Analysis(
    ['analyse_voicekey.py'],
    pathex=[],
    binaries=[],
    datas=sv_ttk_datas,
    hiddenimports=['pyaudio', 'scipy.signal', 'scipy.io.wavfile', 'scipy.signal.windows', 'scipy.interpolate', 'sv_ttk'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # This app only needs numpy/scipy/pandas/matplotlib/pyaudio/sv_ttk; the build
    # environment also has an unrelated ML stack installed that PyInstaller's
    # static analysis otherwise pulls in, bloating the exe by ~150MB for nothing.
    excludes=['torch', 'torchvision', 'torchaudio', 'llvmlite', 'numba',
              'onnx', 'onnxruntime', 'sklearn', 'IPython', 'jupyter', 'notebook',
              'tensorflow', 'tensorboard', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6',
              'lxml', 'Cython'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='VoiceKeyAnalyzer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
