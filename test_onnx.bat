@echo off
title DINOv3 ONNX Backend Test
cd /d "%~dp0"

echo === DINOv3 ONNX Backend Test ===
echo.

if not exist "models\model.onnx" (
    if not exist "models\model_q4.onnx" (
        echo [WARN] No local model found, onnx_demo will download model_q4.onnx
    ) else (
        echo [INFO] Using quantized model: models\model_q4.onnx
    )
) else (
    echo [INFO] Using unquantized F32 model: models\model.onnx
)

echo [1/2] Building (release)...
cargo build --release -p differ-tauri --features onnx --quiet
if %errorlevel% neq 0 (
    echo [ERROR] Build failed!
    pause
    exit /b 1
)

echo [2/2] Running inference...
echo.
cargo run --release -p differ-tauri --features onnx --example onnx_demo --quiet
echo.

if %errorlevel% equ 0 (
    echo === PASS ===
) else (
    echo === FAILED (exit code: %errorlevel%) ===
)
pause
