# -*- mode: python ; coding: utf-8 -*-

import os

# PyQt6 플러그인 경로 찾기
pyqt6_path = None
try:
    import PyQt6
    pyqt6_path = os.path.dirname(PyQt6.__file__)
except ImportError:
    pass

# Qt 플러그인 데이터
qt_datas = []
if pyqt6_path:
    qt6_plugins = os.path.join(pyqt6_path, 'Qt6', 'plugins')
    if os.path.exists(qt6_plugins):
        for subdir in ['platforms', 'styles', 'imageformats', 'iconengines']:
            full_path = os.path.join(qt6_plugins, subdir)
            if os.path.exists(full_path):
                qt_datas.append((full_path, os.path.join('PyQt6', 'Qt6', 'plugins', subdir)))


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('src/gui/styles.qss', 'src/gui'),
        ('models', 'models'),
    ] + qt_datas,
    hiddenimports=[
        'PyQt6', 'PyQt6.QtCore', 'PyQt6.QtGui', 'PyQt6.QtWidgets', 'PyQt6.sip',
        'easyocr', 'texify', 'texify.inference', 'texify.model.model', 'texify.model.processor',
        'fitz', 'fitz.fitz',
        'docx', 'docx.oxml', 'docx.oxml.ns',
        'pyhwpx', 'latex2mathml', 'latex2mathml.converter',
        'torch', 'cv2', 'img2table',
        'PIL', 'PIL.Image', 'numpy', 'tqdm', 'pydantic',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'tkinter'],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

# 폴더 모드 (빠른 실행, 안정적)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='YM-OCR',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,   # 디버깅용 - 정상 작동 확인 후 False로 변경
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='YM-OCR',
)
