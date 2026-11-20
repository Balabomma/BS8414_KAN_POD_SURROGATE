<#
    Launch this project's BS 8414 Part1 Streamlit app.

    .\run_app.ps1              # http://localhost:8501
    .\run_app.ps1 -Port 8502   # a second app alongside the first

    Copy of the shared launcher in bs8414_KAN_surrogate and its siblings, with
    ONE change: a missing material table is rebuilt by this project's
    export_app_assets_paper.py (the paper's frozen 199, written here only),
    never by ..\export_app_assets.py, which rewrites all four sibling apps'
    tables from whatever the corpus holds today. Do not re-sync from a sibling.
#>
param(
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"
$Project = $PSScriptRoot
Set-Location $Project

# ── the app this project ships ────────────────────────────────────────────
$App = $null
foreach ($candidate in @("app_part1.py", "app_fundiff_part1.py")) {
    if (Test-Path (Join-Path $Project $candidate)) { $App = $candidate; break }
}
if (-not $App) {
    throw "No Part1 Streamlit app in $Project (expected app_part1.py or app_fundiff_part1.py)."
}

# ── this project's own venv, never a shared one ───────────────────────────
$Activate = Join-Path $Project "venv\Scripts\Activate.ps1"
if (-not (Test-Path $Activate)) { throw "No venv at $Activate." }
& $Activate

# ── the shipped material table; the app never reads the FDS corpus ────────
if (-not (Test-Path (Join-Path $Project "app_assets\part1_materials.json"))) {
    Write-Host "Material table missing - exporting it from the FDS corpus once..." -ForegroundColor Yellow
    python (Join-Path $Project "export_app_assets_paper.py")
}

# ── GPU status, so a busy card is visible before the app claims it ────────
$gpu = & nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>$null
if ($LASTEXITCODE -eq 0 -and $gpu) {
    Write-Host "GPU: $gpu" -ForegroundColor DarkGray
    $used = [int](($gpu -split ",")[1] -replace "[^\d]", "")
    if ($used -gt 18000) {
        Write-Host "  >18 GB already in use - a training run is probably active." -ForegroundColor Yellow
        Write-Host "  The app falls back to CPU only if CUDA is unavailable, not if it is merely full." -ForegroundColor Yellow
    }
} else {
    Write-Host "nvidia-smi unavailable - the app will run on CPU." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Launching $App on http://localhost:$Port" -ForegroundColor Green
Write-Host "  Ctrl+C to stop." -ForegroundColor DarkGray
Write-Host ""

# `python -m streamlit`, never the `streamlit.exe` shim: this venv was created
# at D:\VS_projects and moved to C:\VS_projects, and the shim still points at
# the old interpreter path, so it exits 1 with no message (found 2026-11-20).
# `python` here is the activated venv's own interpreter, so -m is unaffected.
python -m streamlit run $App --server.port $Port --server.headless false --browser.gatherUsageStats false
