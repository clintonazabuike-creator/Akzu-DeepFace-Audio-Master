@echo off
TITLE Akzu-DeepFace-Audio-Master Suite Orchestration
SETLOCAL EnableDelayedExpansion

echo =======================================================================
echo          AKZU-DEEPFACE-AUDIO-MASTER - SUITE ORCHESTRATION
echo =======================================================================
echo Hardware Target : NVIDIA RTX GPU (CUDA Execution Provider)
echo Environment     : Python 3.10 (.venv)
echo =======================================================================
echo.

:: 1. Verify Virtual Environment
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment (.venv) not found.
    echo Please run setup.bat first to initialize dependencies.
    pause
    exit /b 1
)

:: 2. Activate Virtual Environment
echo [1/3] Activating Python Virtual Environment...
call .venv\Scripts\activate.bat

:: 3. Prompt user for execution mode selection
echo.
echo Select Execution Target:
echo   [1] Launch Video Pipeline Only (CUDA Face Swap)
echo   [2] Launch Audio Pipeline Only (RVC Live Stream)
echo   [3] Launch Both Concurrent Pipelines
echo.
SET /P CHOICE="Enter your choice (1-3): "

if "%CHOICE%"=="1" (
    goto RUN_VIDEO
) else if "%CHOICE%"=="2" (
    goto RUN_AUDIO
) else if "%CHOICE%"=="3" (
    goto RUN_BOTH
) else (
    echo [ERROR] Invalid selection. Exiting.
    pause
    exit /b 1
)

:RUN_VIDEO
echo.
echo [INFO] Starting Video Pipeline (CUDA Inference)...
python video_pipeline\run_video.py --source video_pipeline\models\source_face.jpg --execution-provider cuda
goto END

:RUN_AUDIO
echo.
echo [INFO] Starting RVC Audio Pipeline (<50ms Latency)...
python voice_pipeline\run_rvc_live.py --model-path voice_pipeline\trained_voices\target_model.pth --f0-method rmvpe
goto END

:RUN_BOTH
echo.
echo [INFO] Launching Concurrent Video and Audio Workspaces...
start "Video Inference CUDA" cmd /k "call .venv\Scripts\activate.bat && python video_pipeline\run_video.py --source video_pipeline\models\source_face.jpg --execution-provider cuda"
start "RVC Audio Stream" cmd /k "call .venv\Scripts\activate.bat && python voice_pipeline\run_rvc_live.py --model-path voice_pipeline\trained_voices\target_model.pth --f0-method rmvpe"
echo [OK] Both pipelines active in separate worker processes.
goto END

:END
echo.
echo =======================================================================
echo Session terminated.
echo =======================================================================
pause
