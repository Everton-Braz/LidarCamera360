@echo off
rem Build a standalone LidarCameraCalibrator.exe (no Python needed to run it).
setlocal
cd /d "%~dp0"

where python >nul 2>&1 || (echo Python is required to BUILD the exe. & pause & exit /b 1)
python -m PyInstaller --version >nul 2>&1 || python -m pip install pyinstaller

python -m PyInstaller --noconfirm --clean ^
  --noconsole --onefile ^
  --name LidarCameraCalibrator ^
  --collect-all customtkinter ^
  --collect-submodules rosbags ^
  --hidden-import scipy.optimize ^
  --hidden-import scipy.spatial.transform ^
  --exclude-module matplotlib ^
  --exclude-module pandas ^
  --exclude-module PySide2 ^
  --exclude-module PySide6 ^
  --exclude-module PyQt6 ^
  --exclude-module PyQt5 ^
  --exclude-module IPython ^
  --exclude-module notebook ^
  --exclude-module tests ^
  gui.py

if exist "dist\LidarCameraCalibrator.exe" (
  echo.
  echo   Built: dist\LidarCameraCalibrator.exe
) else (
  echo.
  echo   Build failed, see the output above.
)
pause
