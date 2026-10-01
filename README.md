
## CLI

`differ-cli` exposes the same engine (scan / extract / similar / compare / cache)
as a script- and AI-friendly command line tool. Progress goes to stderr,
`--json` writes a single machine-readable document to stdout, exit code 0 on
success / 1 on runtime errors / 2 on usage errors. Models auto-resolve and
auto-download.

```powershell
cargo build --release -p differ-tauri --bin differ_cli
target\release\differ_cli.exe --help

# List images in a folder (metadata only; machine-readable)
differ_cli.exe scan D:\photos --json

# Extract features into the per-folder cache (CLS-pooled, 384-dim)
differ_cli.exe extract D:\photos
differ_cli.exe extract D:\photos --dump features.json   # path -> vector

# Find images similar to a reference
differ_cli.exe similar D:\photos\ref.jpg D:\photos --threshold 0.9 --json

# Compare two folders (best match per target image)
differ_cli.exe compare D:\libA D:\libB --threshold 0.9 --json

# Feature cache
differ_cli.exe cache stats D:\photos
differ_cli.exe cache clear D:\photos
```

Global options: `--backend ggml|onnx|candle`, `--model <path>`,
`--tier fast|ultra|high` (ggml input resolution: 256/224/518 px),
`--json`, `--quiet`. CLI shares the app's per-folder SQLite cache and the
`DIFFER_FEATURE_MODE=full` / `GGML_VIT_TIER` switches.
