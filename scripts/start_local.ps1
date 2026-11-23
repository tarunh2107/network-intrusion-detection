<#
.SYNOPSIS
    Starts the NIDS FastAPI backend and Streamlit dashboard locally using the existing virtual environment.

.DESCRIPTION
    Phase 9 Local Fallback Launcher:
    - Runs preflight runtime artifact and checksum verification.
    - Checks port 8000 and 8501 availability (does NOT terminate unrelated processes).
    - Binds both services strictly to 127.0.0.1 loopback.
    - Monitors child process lifecycles and cleans them up cleanly on exit/Ctrl+C.
    - Does NOT alter PowerShell execution policy or global machine settings.
#>

[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

$workspaceRoot = $PSScriptRoot | Split-Path -Parent
Set-Location $workspaceRoot

$pythonExe = Join-Path $workspaceRoot ".venv\Scripts\python.exe"
$streamlitExe = Join-Path $workspaceRoot ".venv\Scripts\streamlit.exe"
$preflightScript = Join-Path $workspaceRoot "scripts\check_runtime_artifacts.py"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "NIDS Local Service Launcher (FastAPI & Streamlit)" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# 1. Check Virtual Environment
if (-not (Test-Path $pythonExe) -or -not (Test-Path $streamlitExe)) {
    Write-Host "[ERROR] Virtual environment not found at .venv. Please create it first:" -ForegroundColor Red
    Write-Host "  py -3.11 -m venv .venv" -ForegroundColor Yellow
    Write-Host "  .\.venv\Scripts\pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}

# 2. Run Artifact Preflight Check
Write-Host "`n[1/4] Running runtime artifact preflight check..." -ForegroundColor Yellow
& $pythonExe $preflightScript
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Preflight artifact verification failed. Aborting startup." -ForegroundColor Red
    exit 1
}

# 3. Check Port Availability (8000 and 8501)
Write-Host "`n[2/4] Checking port availability (8000, 8501)..." -ForegroundColor Yellow
$occupiedPorts = @()
foreach ($port in @(8000, 8501)) {
    $conn = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    if ($null -ne $conn) {
        $occupiedPorts += $port
    }
}

if ($occupiedPorts.Count -gt 0) {
    Write-Host "[ERROR] The following ports are already in use: $($occupiedPorts -join ', ')" -ForegroundColor Red
    Write-Host "Please close the occupying applications. This launcher will NOT kill unrelated processes." -ForegroundColor Red
    exit 1
}
Write-Host "      Ports 8000 and 8501 are available." -ForegroundColor Green

# 4. Start Services with Process Tracking
$apiProc = $null
$dashboardProc = $null

try {
    # Launch FastAPI Backend
    Write-Host "`n[3/4] Launching FastAPI backend on 127.0.0.1:8000..." -ForegroundColor Yellow
    $apiProc = Start-Process -FilePath $pythonExe `
        -ArgumentList "-m", "uvicorn", "app.api.main:app", "--host", "127.0.0.1", "--port", "8000" `
        -WorkingDirectory $workspaceRoot `
        -PassThru

    # Wait for API Health
    $apiReady = $false
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Milliseconds 500
        try {
            $resp = Invoke-WebRequest -Uri "http://127.0.0.1:8000/health" -TimeoutSec 2 -UseBasicParsing -ErrorAction SilentlyContinue
            if ($resp.StatusCode -eq 200) {
                $apiReady = $true
                break
            }
        } catch {}
        if ($apiProc.HasExited) {
            break
        }
    }

    if (-not $apiReady) {
        throw "FastAPI backend failed to start or respond to /health."
    }
    Write-Host "      FastAPI backend is online (PID: $($apiProc.Id))." -ForegroundColor Green

    # Launch Streamlit Dashboard
    Write-Host "`n[4/4] Launching Streamlit dashboard on 127.0.0.1:8501..." -ForegroundColor Yellow
    $dashboardProc = Start-Process -FilePath $streamlitExe `
        -ArgumentList "run", "app/dashboard/main.py", "--server.port", "8501", "--server.address", "127.0.0.1", "--browser.gatherUsageStats", "false" `
        -WorkingDirectory $workspaceRoot `
        -PassThru

    # Wait for Dashboard Health
    $dashboardReady = $false
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Milliseconds 500
        try {
            $resp = Invoke-WebRequest -Uri "http://127.0.0.1:8501/_stcore/health" -TimeoutSec 2 -UseBasicParsing -ErrorAction SilentlyContinue
            if ($resp.StatusCode -eq 200) {
                $dashboardReady = $true
                break
            }
        } catch {}
        if ($dashboardProc.HasExited) {
            break
        }
    }

    if (-not $dashboardReady) {
        throw "Streamlit dashboard failed to start or respond to /_stcore/health."
    }
    Write-Host "      Streamlit dashboard is online (PID: $($dashboardProc.Id))." -ForegroundColor Green

    Write-Host "`n======================================================================" -ForegroundColor Cyan
    Write-Host "SYSTEM IS READY" -ForegroundColor Green
    Write-Host "  * Streamlit Dashboard:    http://127.0.0.1:8501" -ForegroundColor White
    Write-Host "  * FastAPI OpenAPI Docs:   http://127.0.0.1:8000/docs" -ForegroundColor White
    Write-Host "  * API Health Endpoint:    http://127.0.0.1:8000/health" -ForegroundColor White
    Write-Host "======================================================================" -ForegroundColor Cyan
    Write-Host "`nPress Ctrl+C to terminate both services cleanly..." -ForegroundColor Yellow

    # Monitor Child Processes
    while (-not $apiProc.HasExited -and -not $dashboardProc.HasExited) {
        Start-Sleep -Seconds 1
    }

    if ($apiProc.HasExited) {
        Write-Host "`n[NOTICE] FastAPI process exited (ExitCode: $($apiProc.ExitCode))." -ForegroundColor Yellow
    }
    if ($dashboardProc.HasExited) {
        Write-Host "`n[NOTICE] Streamlit dashboard process exited (ExitCode: $($dashboardProc.ExitCode))." -ForegroundColor Yellow
    }
}
catch {
    Write-Host "`n[ERROR] An error occurred during service execution: $_" -ForegroundColor Red
}
finally {
    Write-Host "`nCleaning up child processes..." -ForegroundColor Yellow
    if ($null -ne $dashboardProc -and -not $dashboardProc.HasExited) {
        Write-Host "Stopping Streamlit dashboard (PID: $($dashboardProc.Id))..."
        Stop-Process -Id $dashboardProc.Id -Force -ErrorAction SilentlyContinue
    }
    if ($null -ne $apiProc -and -not $apiProc.HasExited) {
        Write-Host "Stopping FastAPI backend (PID: $($apiProc.Id))..."
        Stop-Process -Id $apiProc.Id -Force -ErrorAction SilentlyContinue
    }
    Write-Host "Services stopped cleanly." -ForegroundColor Green
}
