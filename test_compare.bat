@echo off
title DINOv3 ONNX vs GGML vs Candle Comparison
cd /d "%~dp0"

echo === DINOv3 ONNX vs GGML vs Candle Comparison (all CUDA) ===
echo.

if not exist "models\dinov3_vits16.bin" (
    echo [ERROR] GGML model not found: models\dinov3_vits16.bin
    echo Run first: powershell -ExecutionPolicy Bypass -File run_ggml_demo.ps1
    pause
    exit /b 1
)

if not exist "models\model.onnx" (
    echo [ERROR] ONNX F32 model not found: models\model.onnx
    pause
    exit /b 1
)

if not exist "models\dinov3_vits16.safetensors" (
    echo [ERROR] Candle model not found: models\dinov3_vits16.safetensors
    pause
    exit /b 1
)

echo [1/2] Building (release, CUDA)...
cargo build --release -p differ-tauri --features candle-cuda --quiet
if %errorlevel% neq 0 (
    echo [ERROR] Build failed!
    pause
    exit /b 1
)

echo [2/2] Running comparison over all images in test\ ...
echo.
cargo run --release -p differ-tauri --features candle-cuda --example compare_demo --quiet
echo.

if %errorlevel% equ 0 (
    echo === PASS ===
) else (
    echo === FAILED (exit code: %errorlevel%) ===
)
pause
