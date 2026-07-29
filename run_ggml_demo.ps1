# run_ggml_demo.ps1 - DINOv3 GGML 后端运行脚本
# 用法: .\run_ggml_demo.ps1
#
# 此脚本不会修改任何现有代码文件。
# 流程: 检查模型 -> 配置链接器 -> 构建 -> 运行 GGML 推理 demo
#
# 前置条件:
#   - Rust toolchain (cargo)
#   - Python 3 + PyTorch + NumPy (仅模型转换时需要)
#   - MSVC (Visual Studio 2022/2026)
#
# 已知问题:
#   ggml_vit.c 中 weight_ctx 使用 no_alloc=false，与 GGML 0.17 的
#   ggml_gallocr_alloc_graph 断言冲突 (ggml-alloc.c:1169)。
#   修复方法: 将 ggml_vit.c 第127行 .no_alloc = false 改为 .no_alloc = true

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$ModelBin = Join-Path $Root "models\dinov3_vits16.bin"
$ModelPth = Join-Path $Root "models\dinov3_vits16_pretrain.pth"
$ModelFixed = Join-Path $Root "models\dinov3_vits16_fixed.pth"
$ConvertScript = Join-Path $Root "scripts\convert_weights.py"
$CargoConfig = Join-Path $Root ".cargo\config.toml"

Write-Host "=== DINOv3 GGML Backend Runner ===" -ForegroundColor Cyan
Write-Host "Project root: $Root"
Write-Host ""

# Step 0: 确保 .cargo/config.toml 存在 (Windows 链接 advapi32)
if (-not (Test-Path $CargoConfig)) {
    Write-Host "[0] Creating .cargo/config.toml for Windows linker flags..."
    New-Item -ItemType Directory -Path (Join-Path $Root ".cargo") -Force | Out-Null
    @"
[target.x86_64-pc-windows-msvc]
rustflags = ["-C", "link-arg=advapi32.lib"]
"@ | Set-Content $CargoConfig
}

# Step 1: 检查模型文件
Write-Host "[1/3] Checking model files..." -ForegroundColor Yellow

if (-not (Test-Path $ModelBin)) {
    Write-Host "  Binary weights not found, attempting conversion..."

    if (-not (Test-Path $ModelPth)) {
        Write-Host "  ERROR: PyTorch checkpoint not found at:" -ForegroundColor Red
        Write-Host "    $ModelPth"
        Write-Host ""
        Write-Host "  Please download the model first:"
        Write-Host "    python -c `"from huggingface_hub import hf_hub_download; hf_hub_download('Fanqi-Lin-IR/dinov3_vits16_pretrain', 'dinov3_vits16_pretrain.pth', local_dir='models')`""
        exit 1
    }

    # 修复 register_tokens 键名 (storage_tokens -> register_tokens)
    if (-not (Test-Path $ModelFixed)) {
        Write-Host "  Fixing checkpoint key names (storage_tokens -> register_tokens)..."
        python -c "import torch; sd = torch.load(r'$ModelPth', map_location='cpu', weights_only=True); sd['register_tokens'] = sd.pop('storage_tokens', sd.get('register_tokens')); torch.save(sd, r'$ModelFixed')"
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  WARNING: Key fix failed, trying conversion with original file..." -ForegroundColor DarkYellow
            $ModelFixed = $ModelPth
        }
    }

    Write-Host "  Converting weights to raw binary format..."
    python $ConvertScript --input $ModelFixed --output $ModelBin --model vit_small_16
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ERROR: Weight conversion failed!" -ForegroundColor Red
        exit 1
    }
}

$modelSize = (Get-Item $ModelBin).Length / 1MB
Write-Host "  Model ready: $ModelBin ($([math]::Round($modelSize, 1)) MB)" -ForegroundColor Green
Write-Host ""

# Step 2: 构建
Write-Host "[2/3] Building with GGML feature..." -ForegroundColor Yellow
Push-Location $Root
cargo build -p differ-tauri 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ERROR: Build failed!" -ForegroundColor Red
    Pop-Location
    exit 1
}
Write-Host "  Build successful." -ForegroundColor Green
Write-Host ""

# Step 3: 运行 demo
Write-Host "[3/3] Running GGML inference demo..." -ForegroundColor Yellow
Write-Host ""

$env:GGML_MODEL_PATH = $ModelBin
$env:RUST_LOG = "info"
cargo run -p differ-tauri --example ggml_demo 2>&1
$exitCode = $LASTEXITCODE

Pop-Location

Write-Host ""
if ($exitCode -eq 0) {
    Write-Host "=== SUCCESS ===" -ForegroundColor Green
} else {
    Write-Host "=== FAILED (exit code: $exitCode) ===" -ForegroundColor Red
    Write-Host ""
    Write-Host "Known issue: ggml_vit.c weight_ctx uses no_alloc=false, which conflicts" -ForegroundColor DarkYellow
    Write-Host "with GGML 0.17 gallocr assertion. Fix: change line 127 in" -ForegroundColor DarkYellow
    Write-Host "crates/ggml-vit/src/ggml_vit.c from '.no_alloc = false' to '.no_alloc = true'" -ForegroundColor DarkYellow
}
exit $exitCode
