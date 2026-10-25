@echo off
TITLE Deepfake Voice & Video Suite - Environment Setup
SETLOCAL EnableDelayedExpansion

echo =======================================================================
echo          DEEPFAKE VOICE & VIDEO SUITE - SYSTEM BOOTSTRAP
echo =======================================================================
echo Targeted HW: NVIDIA RTX GPU (CUDA Execution Provider)
echo Environment: Python 3.10 | Strict numpy^<2.0.0
echo =======================================================================
echo.

:: 1. Verify Python Installation
python --version 2>NUL | findstr /C:"3.10" >NUL
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python 3.10 is required but not detected in PATH.
    echo Please install Python 3.10.x and ensure it is added to system PATH.
    pause
    exit /b 1
)
echo [OK] Python 3.10 detected.

:: 2. Create Directory Structure
echo.
echo [1/5] Initializing Directory Structure...
if not exist "video_pipeline\models" mkdir "video_pipeline\models"
if not exist "voice_pipeline\trained_voices" mkdir "voice_pipeline\trained_voices"
if not exist "configs" mkdir "configs"
echo [OK] Directory tree established.

:: 3. Create Virtual Environment
echo.
echo [2/5] Creating Python Virtual Environment (.venv)...
if exist ".venv" (
    echo [.venv] existing environment found. Skipping creation.
) else (
    python -m venv .venv
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
)

:: 4. Activate VENV & Upgrade Pip
echo.
echo [3/5] Activating VENV and upgrading core tools...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
echo [OK] Base build tools updated.

:: 5. Install CUDA PyTorch & ONNX Runtime GPU
echo.
echo [4/5] Installing CUDA-enabled PyTorch and dependencies...
:: Installing PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

:: 6. Install Core Requirements with Strict NumPy Constraint
echo.
echo [5/5] Installing Suite Dependencies from requirements.txt...
pip install -r requirements.txt

:: Enforce numpy<2.0.0 post-check
echo.
echo Enforcing Strict Numpy Version Cap...
pip install "numpy<2.0.0" --force-reinstall --no-deps

echo.
echo =======================================================================
echo              SETUP COMPLETED SUCCESSFULLY!
echo =======================================================================
echo  To activate this environment manually, run:
echo      .venv\Scripts\activate.bat
echo =======================================================================
echo.
pause
