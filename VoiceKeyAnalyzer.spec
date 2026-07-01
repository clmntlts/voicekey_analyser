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
    excludes=[],
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
