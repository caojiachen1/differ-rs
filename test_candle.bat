@echo off
title DINOv3 Candle Backend Test
cd /d "%~dp0"

echo === DINOv3 Candle Backend Test (CUDA) ===
echo.

if not exist "models\dinov3_vits16.safetensors" (
    echo [ERROR] Model not found: models\dinov3_vits16.safetensors
    echo Please provide a safetensors model file or set CANDLE_MODEL_PATH env var.
    pause
    exit /b 1
)

echo [1/2] Building...
cargo build --release -p differ-tauri --features candle-cuda --quiet
if %errorlevel% neq 0 (
    echo [ERROR] Build failed!
    pause
    exit /b 1
)

echo [2/2] Running inference...
echo.
cargo run --release -p differ-tauri --features candle-cuda --example candle_demo --quiet
echo.

if %errorlevel% equ 0 (
    echo === PASS ===
) else (
    echo === FAILED (exit code: %errorlevel%) ===
)
pause
