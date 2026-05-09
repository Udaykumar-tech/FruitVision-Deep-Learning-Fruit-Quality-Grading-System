@echo off
setlocal EnableDelayedExpansion
title FruitVision Demo

cd /d "%~dp0"

echo.
echo ============================================================
echo   FruitVision Demo - Starting Up
echo ============================================================
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.10 or higher from https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%V in ('python --version 2^>^&1') do set PY_VERSION=%%V
echo [OK] Python !PY_VERSION! detected
echo.

echo Checking checkpoints...
if not exist "checkpoints\effnet_swin_pathw_best.pt" goto missing_ckpt
if not exist "checkpoints\dba_vinet_pathw_best.pt"   goto missing_ckpt
if not exist "checkpoints\mobilenetv4_pathw_best.pt" goto missing_ckpt
echo [OK] All 3 model checkpoints found
echo.

echo Checking label map...
if not exist "splits\label_map.json" (
    echo [ERROR] Missing splits\label_map.json
    pause
    exit /b 1
)
echo [OK] Label map found
echo.

echo Checking analysis images...
set IMG_DIR=frontend\assets\analysis
if not exist "%IMG_DIR%\training_curves.png" goto missing_img
if not exist "%IMG_DIR%\pareto_frontier.png" goto missing_img
if not exist "%IMG_DIR%\confusion_matrices_grid.png" goto missing_img
if not exist "%IMG_DIR%\per_class_recall_heatmap.png" goto missing_img
if not exist "%IMG_DIR%\formalin_comparison.png" goto missing_img
if not exist "%IMG_DIR%\focus_distributions.png" goto missing_img
if not exist "%IMG_DIR%\gradcam_all_models_per_class.png" goto missing_img
if not exist "%IMG_DIR%\inference_speed.png" goto missing_img
echo [OK] All 8 analysis images found
echo.

echo Checking PyTorch...
python -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo [WARN] PyTorch not installed. Run: pip install torch torchvision
    echo Or for CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    echo.
    pause
    exit /b 1
)

for /f "delims=" %%V in ('python -c "import torch; print(torch.__version__)"') do set TORCH_VERSION=%%V
for /f "delims=" %%V in ('python -c "import torch; print('CUDA' if torch.cuda.is_available() else 'CPU')"') do set TORCH_DEVICE=%%V
echo [OK] PyTorch !TORCH_VERSION! on !TORCH_DEVICE!
echo.

echo Checking other dependencies...
python -c "import fastapi, uvicorn, timm, kornia, cv2, PIL, pytorch_grad_cam" >nul 2>&1
if errorlevel 1 (
    echo [WARN] Missing dependencies. Installing now...
    echo.
    python -m pip install -q fastapi "uvicorn[standard]" python-multipart pillow opencv-python-headless timm grad-cam kornia
    if errorlevel 1 (
        echo [ERROR] Dependency install failed. Try manually:
        echo   pip install fastapi uvicorn[standard] python-multipart pillow opencv-python-headless timm grad-cam kornia
        echo.
        pause
        exit /b 1
    )
    echo [OK] Dependencies installed
    echo.
) else (
    echo [OK] All dependencies present
    echo.
)

echo ============================================================
echo   Starting FastAPI server on http://localhost:8000
echo ============================================================
echo.
echo Browser will open in 4 seconds. Press Ctrl+C in this window to stop the server.
echo.

start "" /b cmd /c "timeout /t 4 /nobreak >nul && start http://localhost:8000"

cd backend
python main.py

echo.
echo Server stopped.
pause
exit /b 0


:missing_ckpt
echo [ERROR] Missing model checkpoints in checkpoints\ folder.
echo Required files:
echo   - effnet_swin_pathw_best.pt
echo   - dba_vinet_pathw_best.pt
echo   - mobilenetv4_pathw_best.pt
echo.
echo These are too large for normal sharing. See README for transfer instructions.
echo.
pause
exit /b 1


:missing_img
echo [ERROR] Missing analysis images in %IMG_DIR%
echo Required: training_curves.png, pareto_frontier.png, confusion_matrices_grid.png,
echo           per_class_recall_heatmap.png, formalin_comparison.png, focus_distributions.png,
echo           gradcam_all_models_per_class.png, inference_speed.png
echo.
pause
exit /b 1