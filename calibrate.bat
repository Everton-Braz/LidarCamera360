@echo off
rem Lidar-Camera calibrator launcher.
setlocal
cd /d "%~dp0"
set "LOG=%TEMP%\lidar-camera-calibrator.log"
if exist "%LOG%" del "%LOG%" >nul 2>&1

set "PY="
where py >nul 2>&1 && set "PY=py -3"
if not defined PY (where python >nul 2>&1 && set "PY=python")
if not defined PY (
  echo.
  echo   Python was not found in PATH. Install Python 3.9+ and try again.
  echo.
  pause
  exit /b 1
)

%PY% -c "import customtkinter, numpy, scipy, cv2, rosbags" 1>nul 2>"%LOG%"
if errorlevel 1 (
  echo.
  echo   Missing dependencies. Install them with:
  echo.
  echo       pip install -r requirements.txt
  echo.
  type "%LOG%"
  echo.
  pause
  exit /b 1
)

set "PYW="
where pythonw >nul 2>&1 && set "PYW=pythonw"
if defined PYW (
  start "Lidar-Camera calibrator" "%PYW%" "%~dp0gui.py"
) else (
  %PY% "%~dp0gui.py"
  if errorlevel 1 pause
)
exit /b 0
