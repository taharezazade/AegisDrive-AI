@echo off
setlocal EnableExtensions

chcp 65001 >nul
title AegisDrive-AI - Installer

cd /d "%~dp0"

echo.
echo ==========================================
echo   AegisDrive-AI
echo   Dependency Installer
echo ==========================================
echo.

REM Step 1: Check Python Launcher
where py >nul 2>&1
if errorlevel 1 (
echo ERROR: Python Launcher was not found.
echo Install Python 3.11 and try again.
goto INSTALL_FAILED
)

REM Step 2: Check requirements file
if not exist "requirements.txt" (
echo ERROR: requirements.txt was not found.
goto INSTALL_FAILED
)

REM Step 3: Create virtual environment if needed
if not exist ".venv\Scripts\python.exe" (
echo Creating Python 3.11 virtual environment...
py -3.11 -m venv .venv

```
if errorlevel 1 (
    echo ERROR: Failed to create the virtual environment.
    echo Make sure Python 3.11 is installed.
    goto INSTALL_FAILED
)
```

)

set "PYTHON_EXE=%CD%.venv\Scripts\python.exe"

REM Step 4: Verify Python
echo.
echo Checking Python version...
"%PYTHON_EXE%" --version

if errorlevel 1 (
echo ERROR: Virtual environment Python is not working.
goto INSTALL_FAILED
)

REM Step 5: Upgrade pip
echo.
echo Upgrading pip...
"%PYTHON_EXE%" -m pip install --upgrade pip

if errorlevel 1 (
echo WARNING: pip could not be upgraded.
echo Continuing with the existing pip installation.
)

REM Step 6: Install dependencies
echo.
echo Installing project dependencies...
"%PYTHON_EXE%" -m pip install -r requirements.txt

if not errorlevel 1 goto INSTALL_SUCCESS

echo.
echo WARNING: Installation from the default package index failed.
echo Retrying with the Tsinghua mirror...

"%PYTHON_EXE%" -m pip install -r requirements.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple/

if not errorlevel 1 goto INSTALL_SUCCESS

echo.
echo WARNING: Tsinghua mirror failed.
echo Retrying with the Aliyun mirror...

"%PYTHON_EXE%" -m pip install -r requirements.txt --index-url https://mirrors.aliyun.com/pypi/simple/

if not errorlevel 1 goto INSTALL_FAILED_VERIFICATION

echo.
echo ERROR: All configured installation attempts failed.
goto INSTALL_FAILED

:INSTALL_SUCCESS
echo.
echo ==========================================
echo   INSTALLATION COMPLETED
echo ==========================================
echo.

REM Step 7: Verify the installed libraries
echo Checking installed libraries...

"%PYTHON_EXE%" -c "import cv2, numpy, torch; print('OpenCV:', cv2.**version**); print('NumPy:', numpy.**version**); print('PyTorch:', torch.**version**); print('CUDA available:', torch.cuda.is_available())"

if errorlevel 1 goto INSTALL_FAILED_VERIFICATION

"%PYTHON_EXE%" -c "import ultralytics; print('Ultralytics:', ultralytics.**version**)" 2>nul

if errorlevel 1 (
echo INFO: Ultralytics is not installed or could not be imported.
echo This is acceptable only if it is not required by this project.
)

"%PYTHON_EXE%" -c "import streamlit; print('Streamlit:', streamlit.**version**)" 2>nul

if errorlevel 1 (
echo INFO: Streamlit is not installed or could not be imported.
echo This is acceptable only if it is not required by this project.
)

"%PYTHON_EXE%" -c "import pandas; print('Pandas:', pandas.**version**)" 2>nul

if errorlevel 1 (
echo INFO: Pandas is not installed or could not be imported.
echo This is acceptable only if it is not required by this project.
)

echo.
echo ==========================================
echo   DEPENDENCY CHECK COMPLETED
echo ==========================================
echo.
echo Review any INFO messages above.
echo Verify the model-specific dependencies before running YOLOP.
echo.
pause
exit /b 0

:INSTALL_FAILED_VERIFICATION
echo.
echo ERROR: Installation or verification failed.
echo Review the error messages above.
goto INSTALL_FAILED

:INSTALL_FAILED
echo.
echo ==========================================
echo   INSTALLATION FAILED
echo ==========================================
echo.
echo Check Python 3.11, your internet connection,
echo package compatibility, and requirements.txt.
echo.
pause
exit /b 1
