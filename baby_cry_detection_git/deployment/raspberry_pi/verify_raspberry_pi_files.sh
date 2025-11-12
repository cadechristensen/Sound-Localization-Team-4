#!/bin/bash
# Verification script for Raspberry Pi deployment files
# Run this on Raspberry Pi after transferring files

echo "=============================================================================="
echo "RASPBERRY PI DEPLOYMENT - FILE VERIFICATION"
echo "=============================================================================="
echo ""

cd ~/baby_monitor 2>/dev/null || {
    echo "ERROR: Directory ~/baby_monitor does not exist!"
    echo "Please create it first: mkdir -p ~/baby_monitor"
    exit 1
}

echo "Checking directory: $(pwd)"
echo ""

# Check main scripts
echo "Main Scripts:"
echo "-------------"
files=("robot_baby_monitor.py" "realtime_baby_cry_detector.py" "sound_localization_interface.py")
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        size=$(ls -lh "$file" | awk '{print $5}')
        echo "[OK] $file ($size)"
    else
        echo "[MISSING] $file"
    fi
done
echo ""

# Check model and requirements
echo "Model & Dependencies:"
echo "---------------------"
if [ -f "model_best.pth" ]; then
    size=$(ls -lh model_best.pth | awk '{print $5}')
    echo "[OK] model_best.pth ($size)"
else
    echo "[MISSING] model_best.pth"
fi

if [ -f "requirements.txt" ]; then
    echo "[OK] requirements.txt"
else
    echo "[MISSING] requirements.txt"
fi
echo ""

# Check src directory
echo "Source Package (src/):"
echo "----------------------"
src_files=("__init__.py" "config.py" "model.py" "audio_filter.py" "data_preprocessing.py")
for file in "${src_files[@]}"; do
    if [ -f "src/$file" ]; then
        if [ "$file" == "__init__.py" ]; then
            echo "[OK] src/$file"
        else
            size=$(ls -lh "src/$file" | awk '{print $5}')
            echo "[OK] src/$file ($size)"
        fi
    else
        echo "[MISSING] src/$file"
    fi
done
echo ""

# Test Python imports
echo "Python Import Tests:"
echo "--------------------"
python3 -c "from src.config import Config; print('[OK] src.config imports successfully')" 2>/dev/null || echo "[FAIL] src.config import failed"
python3 -c "from src.model import create_model; print('[OK] src.model imports successfully')" 2>/dev/null || echo "[FAIL] src.model import failed"
python3 -c "from src.audio_filter import BabyCryAudioFilter; print('[OK] src.audio_filter imports successfully')" 2>/dev/null || echo "[FAIL] src.audio_filter import failed"
python3 -c "import torch; print('[OK] PyTorch version:', torch.__version__)" 2>/dev/null || echo "[FAIL] PyTorch not installed"
python3 -c "import pyaudio; print('[OK] PyAudio installed')" 2>/dev/null || echo "[FAIL] PyAudio not installed"
echo ""

# Count total files
echo "File Count:"
echo "-----------"
total_py=$(find . -maxdepth 2 -name "*.py" | wc -l)
total_pth=$(find . -maxdepth 1 -name "*.pth" | wc -l)
echo "Python files: $total_py (expected: 8+)"
echo "Model files: $total_pth (expected: 1)"
echo ""

# Check permissions
echo "File Permissions:"
echo "-----------------"
if [ -x "robot_baby_monitor.py" ]; then
    echo "[OK] robot_baby_monitor.py is executable"
else
    echo "[INFO] robot_baby_monitor.py not executable (run: chmod +x robot_baby_monitor.py)"
fi
echo ""

# List audio devices
echo "Audio Devices:"
echo "--------------"
python3 -c "
import pyaudio
p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info['maxInputChannels'] > 0:
        print(f'[{i}] {info[\"name\"]} (channels: {info[\"maxInputChannels\"]})')
p.terminate()
" 2>/dev/null || echo "Unable to list audio devices (PyAudio may not be installed)"
echo ""

# Summary
echo "=============================================================================="
echo "VERIFICATION COMPLETE"
echo "=============================================================================="
echo ""
echo "If all files show [OK], you can run:"
echo ""
echo "  python3 robot_baby_monitor.py \\"
echo "      --model model_best.pth \\"
echo "      --device-index [YOUR_MIC_INDEX] \\"
echo "      --channels 4"
echo ""
echo "=============================================================================="
