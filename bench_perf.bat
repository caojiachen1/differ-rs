@echo off
setlocal enabledelayedexpansion
rem ============================================================
rem  DINOv3 backend performance benchmark: ONNX vs GGML
rem
rem  Usage:  bench_perf.bat [N_IMAGES]
rem    N_IMAGES  batch_extract workload size (default 200)
rem
rem  Optional env vars:
rem    ONNX_MODEL_PATH  (default models\model.onnx)
rem    GGML_MODEL_PATH  (default models\dinov3_vits16.bin)
rem ============================================================

set N_IMAGES=%1
if "%N_IMAGES%"=="" set N_IMAGES=200

cd /d "%~dp0"

set RESULT_DIR=bench_results
if not exist "%RESULT_DIR%" mkdir "%RESULT_DIR%"

echo ============================================================
echo  DINOv3 Performance Benchmark  (N_IMAGES=%N_IMAGES%)
echo ============================================================
echo.

echo [1/4] Building benchmarks (release, ggml-cuda + onnx)...
cargo build --release -p differ-tauri --example bench_perf --example compare_demo
if errorlevel 1 (
    echo BUILD FAILED
    exit /b 1
)
echo.

set BIN=target\release\examples\bench_perf.exe

echo [2/4] Benchmarking ONNX backend...
echo ------------------------------------------------------------
"%BIN%" onnx %N_IMAGES% > "%RESULT_DIR%\onnx.txt" 2>&1
if errorlevel 1 (
    echo ONNX benchmark FAILED, see %RESULT_DIR%\onnx.txt
    type "%RESULT_DIR%\onnx.txt"
    exit /b 1
)
type "%RESULT_DIR%\onnx.txt"
echo.

echo [3/4] Benchmarking GGML backend...
echo ------------------------------------------------------------
"%BIN%" ggml %N_IMAGES% > "%RESULT_DIR%\ggml.txt" 2>&1
if errorlevel 1 (
    echo GGML benchmark FAILED, see %RESULT_DIR%\ggml.txt
    type "%RESULT_DIR%\ggml.txt"
    exit /b 1
)
type "%RESULT_DIR%\ggml.txt"
echo.

echo [4/4] Cross-backend accuracy + head-to-head (compare_demo)...
echo ------------------------------------------------------------
target\release\examples\compare_demo.exe > "%RESULT_DIR%\compare.txt" 2>&1
if errorlevel 1 (
    echo compare_demo FAILED, see %RESULT_DIR%\compare.txt
    type "%RESULT_DIR%\compare.txt"
    exit /b 1
)
type "%RESULT_DIR%\compare.txt"
echo.

echo ============================================================
echo  SUMMARY
echo ============================================================
echo --- ONNX ---
findstr /c:"[load]" /c:"avg" /c:"img/s" "%RESULT_DIR%\onnx.txt"
echo --- GGML ---
findstr /c:"[load]" /c:"avg" /c:"img/s" "%RESULT_DIR%\ggml.txt"
echo --- Accuracy ---
findstr /c:"max |sim diff|" /c:"speedup" /c:"batch vs single" "%RESULT_DIR%\compare.txt"
echo.
echo Full logs in %RESULT_DIR%\ (onnx.txt, ggml.txt, compare.txt)

endlocal
pause
