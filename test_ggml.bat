@echo off
title DINOv3 GGML Backend Test
cd /d "%~dp0"

echo === DINOv3 GGML Backend Test ===
echo.

if not exist "models\dinov3_vits16.bin" (
    echo [ERROR] Model not found: models\dinov3_vits16.bin
    echo Run first: powershell -ExecutionPolicy Bypass -File run_ggml_demo.ps1
    pause
    exit /b 1
)

echo [1/2] Building (CUDA, release)...
cargo build --release -p differ-tauri --quiet
if %errorlevel% neq 0 (
    echo [ERROR] Build failed!
    pause
    exit /b 1
)

echo [2/2] Running inference...
echo.
set "GGML_MODEL_PATH=%~dp0models\dinov3_vits16.bin"
cargo run --release -p differ-tauri --example ggml_demo --quiet
echo.

if %errorlevel% equ 0 (
    echo === PASS ===
) else (
    echo === FAILED (exit code: %errorlevel%) ===
)
pause
