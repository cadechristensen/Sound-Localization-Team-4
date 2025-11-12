# Baby Cry Detection System - Raspberry Pi Deployment (PowerShell)
# Simple deployment script for Windows
#
# Usage:
#   .\deployment\deploy_to_pi.ps1 -PiHost "bthz@raspberrypi.local"
#   .\deployment\deploy_to_pi.ps1 -PiHost "bthz@raspberrypi.local" -ModelPath "results/binary_classification/train_2025-10-05_01-09-46/model_quantized.pth"

param(
    [Parameter(Mandatory=$true, HelpMessage="Pi host (e.g., bthz@raspberrypi.local)")]
    [string]$PiHost,

    [Parameter(Mandatory=$false, HelpMessage="Path to model file")]
    [string]$ModelPath
)

# Setup
$PROJECT_ROOT = Split-Path -Parent $PSScriptRoot
$DEPLOYMENT_DIR = Join-Path $PROJECT_ROOT "deployment\raspberry_pi"
$PI_DEPLOY_DIR = "~/baby_monitor"

Write-Host "Baby Cry Detection - Raspberry Pi Deployment" -ForegroundColor Cyan
Write-Host "Target: $PiHost" -ForegroundColor Cyan
Write-Host ""

# Step 1: Determine model path
if ([string]::IsNullOrWhiteSpace($ModelPath)) {
    Write-Host "[*] Auto-detecting latest model_quantized.pth..." -ForegroundColor Blue
    $models = Get-ChildItem -Path (Join-Path $PROJECT_ROOT "results") -Recurse -Filter "model_quantized.pth" -ErrorAction SilentlyContinue
    if ($models.Count -eq 0) {
        Write-Host "[ERROR] No model_quantized.pth found!" -ForegroundColor Red
        exit 1
    }
    $ModelPath = ($models | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
} else {
    $ModelPath = Join-Path $PROJECT_ROOT $ModelPath
}

Write-Host "[OK] Model: $(Split-Path -Leaf $ModelPath)" -ForegroundColor Green
Write-Host ""

# Step 2: Verify all files exist
Write-Host "[*] Verifying local files..." -ForegroundColor Blue
$filesToCheck = @(
    (Join-Path $DEPLOYMENT_DIR "realtime_baby_cry_detector.py"),
    (Join-Path $DEPLOYMENT_DIR "robot_baby_monitor.py"),
    (Join-Path $DEPLOYMENT_DIR "sound_localization_interface.py"),
    (Join-Path $DEPLOYMENT_DIR "config_pi.py"),
    (Join-Path $DEPLOYMENT_DIR "audio_filtering.py"),
    (Join-Path $DEPLOYMENT_DIR "requirements-pi.txt"),
    (Join-Path (Join-Path $PROJECT_ROOT "src") "config.py"),
    (Join-Path (Join-Path $PROJECT_ROOT "src") "model.py"),
    (Join-Path (Join-Path $PROJECT_ROOT "src") "audio_filter.py"),
    (Join-Path (Join-Path $PROJECT_ROOT "src") "data_preprocessing.py"),
    $ModelPath
)

$missing = 0
foreach ($file in $filesToCheck) {
    if (Test-Path $file) {
        Write-Host "  [OK] $(Split-Path -Leaf $file)" -ForegroundColor Green
    } else {
        Write-Host "  [ERROR] Missing: $file" -ForegroundColor Red
        $missing++
    }
}

if ($missing -gt 0) {
    Write-Host "[ERROR] $missing files missing!" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Step 3: Create directories on Pi
Write-Host "[*] Creating directories on Pi..." -ForegroundColor Blue
ssh $PiHost "mkdir -p $PI_DEPLOY_DIR/src $PI_DEPLOY_DIR/models" 2>$null
Write-Host "[OK] Directories created" -ForegroundColor Green
Write-Host ""

# Step 4: Copy Python files
Write-Host "[*] Copying Python files..." -ForegroundColor Blue
$pyFiles = @(
    "realtime_baby_cry_detector.py",
    "robot_baby_monitor.py",
    "sound_localization_interface.py",
    "config_pi.py",
    "audio_filtering.py",
    "requirements-pi.txt"
)
foreach ($file in $pyFiles) {
    $source = Join-Path $DEPLOYMENT_DIR $file
    scp $source "$PiHost`:$PI_DEPLOY_DIR/" 2>$null
    Write-Host "  [OK] $file" -ForegroundColor Green
}
Write-Host ""

# Step 5: Copy source package
Write-Host "[*] Copying source package (src/)..." -ForegroundColor Blue
$srcFiles = @("config.py", "model.py", "audio_filter.py", "data_preprocessing.py")
foreach ($file in $srcFiles) {
    $source = Join-Path $PROJECT_ROOT "src" $file
    scp $source "$PiHost`:$PI_DEPLOY_DIR/src/" 2>$null
    Write-Host "  [OK] src/$file" -ForegroundColor Green
}
ssh $PiHost "touch $PI_DEPLOY_DIR/src/__init__.py" 2>$null
Write-Host "  [OK] src/__init__.py" -ForegroundColor Green
Write-Host ""

# Step 6: Copy model
Write-Host "[*] Copying trained model..." -ForegroundColor Blue
$modelName = Split-Path -Leaf $ModelPath
scp $ModelPath "$PiHost`:$PI_DEPLOY_DIR/models/" 2>$null
Write-Host "  [OK] models/$modelName" -ForegroundColor Green
Write-Host ""

# Step 7: Copy documentation
Write-Host "[*] Copying documentation..." -ForegroundColor Blue
$docFiles = @("PI_DEPLOYMENT_STEPS.md", "README_FILTERING.md")
foreach ($file in $docFiles) {
    $source = Join-Path $DEPLOYMENT_DIR $file
    if (Test-Path $source) {
        scp $source "$PiHost`:$PI_DEPLOY_DIR/" 2>$null
        Write-Host "  [OK] $file" -ForegroundColor Green
    }
}
Write-Host ""

# Step 8: Install dependencies
Write-Host "[*] Installing dependencies on Pi..." -ForegroundColor Blue
Write-Host "[*] This may take 10-25 minutes (mostly PyTorch installation)" -ForegroundColor Yellow
ssh $PiHost @"
cd $PI_DEPLOY_DIR
sudo apt-get update -y
sudo apt-get install -y python3-pip python3-venv libasound2-dev libportaudio2 libportaudiocpp0 portaudio19-dev
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-pi.txt
"@ 2>$null
Write-Host "[OK] Dependencies installed" -ForegroundColor Green
Write-Host ""

# Step 9: Verify installation
Write-Host "[*] Verifying installation..." -ForegroundColor Blue
ssh $PiHost "cd $PI_DEPLOY_DIR && source venv/bin/activate && python3 -c 'import torch; print(\"PyTorch version:\", torch.__version__)'" 2>$null
Write-Host "[OK] Installation verified" -ForegroundColor Green
Write-Host ""

Write-Host "========================================================" -ForegroundColor Green
Write-Host "DEPLOYMENT COMPLETE!" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps on your Raspberry Pi:" -ForegroundColor Cyan
Write-Host "  1. SSH to Pi: ssh $PiHost"
Write-Host "  2. Go to project: cd $PI_DEPLOY_DIR"
Write-Host "  3. Activate environment: source venv/bin/activate"
Write-Host "  4. Find audio device: python3 << EOF"
Write-Host "import pyaudio"
Write-Host "p = pyaudio.PyAudio()"
Write-Host "for i in range(p.get_device_count()):"
Write-Host "    info = p.get_device_info_by_index(i)"
Write-Host "    print(f'[{i}] {info[\"name\"]} ({info[\"maxInputChannels\"]} channels)')"
Write-Host "EOF"
Write-Host "  5. Test with simulated audio: python3 realtime_baby_cry_detector.py --model models/$modelName --device cpu --test-mode"
Write-Host ""
