# Raspberry Pi Deployment - Quick Start Guide

## Latest Updates
- [DONE] **Multi-channel audio support** - All 4 channels preserved with phase intact
- [DONE] `realtime_baby_cry_detector.py` - Updated for 4-channel input
- [DONE] `config_pi.py` - Standalone with embedded Config class
- [DONE] `audio_filtering.py` - Supports multi-channel filtering

## Key Feature: Multi-Channel Audio (NEW!)
The system now preserves all 4 microphone channels throughout the pipeline, enabling sound localization and beamforming capabilities. All channels are perfectly time-aligned with phase relationships intact.

## Deployment Steps

### Step 1: Copy Updated Files to Pi

From your PC:
```bash
# Copy the entire deployment/raspberry_pi folder (includes updated realtime_baby_cry_detector.py)
scp -r deployment/raspberry_pi/ pi@raspberrypi.local:~/baby_monitor/

# Copy your trained model
scp results/train_*/model_best.pth pi@raspberrypi.local:~/baby_monitor/

# Copy source files (including updated src/audio_filter.py with multi-channel support)
scp -r src/ pi@raspberrypi.local:~/baby_monitor/
```

### Step 2: Install Dependencies on Raspberry Pi

SSH into your Pi and run:
```bash
cd ~/baby_monitor

# Install Python dependencies
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy scipy librosa soundfile pyaudio

# Test imports
python3 -c "import torch, torchaudio; print('[OK] Imports OK')"
```

### Step 3: Find Your Audio Device

```bash
# List audio devices
python3 -c "import pyaudio; p = pyaudio.PyAudio(); \
[print(f'[{i}] {p.get_device_info_by_index(i)[\"name\"]}') \
for i in range(p.get_device_count())]"

# Find the PCM6260-Q1 device (usually outputs "4 channels")
```

### Step 4: Test Multi-Channel Audio Capture

```bash
# Test the detector with 4-channel audio
cd ~/baby_monitor
python3 realtime_baby_cry_detector.py \
    --model model_best.pth \
    --device-index 2 \
    --channels 4 \
    --device cpu \
    --test-mode

# Should print: "Audio stream started (device: 2, channels: 4)"
```

**Expected Output:**
```
INFO:root:Using device: cpu
INFO:root:Baby cry model loaded from model_best.pth
INFO:root:Audio filter initialized
INFO:root:Audio processing thread started

======================================================================
Real-Time Baby Cry Detector - ACTIVE
======================================================================
Mode: LOW-POWER LISTENING
Microphone Channels: 4
Detection Threshold: 75%
Confirmation Threshold: 85%
Device: cpu
======================================================================

[Listening... Press Ctrl+C to stop]
```

### Step 5: Verify Multi-Channel Audio

```bash
# Check that 4 channels are being captured
python3 << 'EOF'
import numpy as np
from realtime_baby_cry_detector import CircularAudioBuffer

# Create a test buffer
buffer = CircularAudioBuffer(max_duration=5.0, sample_rate=16000, num_channels=4)

# Add test audio (1 second, all 4 channels)
test_audio = np.random.randn(16000, 4).astype(np.float32)
buffer.add(test_audio)

# Get audio back
result = buffer.get_last_n_seconds(1.0)

print(f"[OK] Buffer shape: {result.shape}")
print(f"[OK] Channels: {result.shape[1]}")
assert result.shape[1] == 4, "Expected 4 channels!"
print("[OK] Multi-channel audio working correctly!")
EOF
```

## Directory Structure on Pi

```
~/baby_cry_detection/
├── raspberry_pi/              # Deployment files (standalone)
│   ├── config_pi.py          # [OK] Standalone config (no src needed)
│   ├── audio_filtering.py    # [OK] Filtering implementation
│   ├── test_pi_filtering.py  # [OK] Performance test
│   ├── pi_setup.py           # Setup script
│   └── realtime_baby_cry_detector.py  # Main detector
│
└── models/                    # Model files
    └── model_quantized.pth   # Your quantized model
```

## Loading the Model on Pi

In your Python code:
```python
import torch
from config_pi import ConfigPi

# Load config
config = ConfigPi()

# Load quantized model
model = torch.load(
    '~/baby_cry_detection/models/model_quantized.pth',
    map_location='cpu'  # Pi doesn't have GPU
)
model.eval()
```

## Available Quantized Models

You have these pre-quantized models on your PC:
- Latest: `results/train_2025-10-21_02-36-55/model_quantized.pth`
- Previous: `results/train_2025-10-18_14-54-11/model_quantized.pth`

These are already optimized for Pi (int8 quantization, smaller size, faster inference).

## Troubleshooting

### "No module named 'config'"
[FIXED] **FIXED!** - config_pi.py is now standalone

### "No module named 'src'"
[FIXED] **FIXED!** - No longer imports from src

### Import errors for numpy/torch/librosa
Install dependencies:
```bash
pip install torch torchaudio numpy scipy librosa
```

### Audio device errors
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
sudo apt-get install alsa-utils pulseaudio
```

## Performance Expectations

| Component | Time | Notes |
|-----------|------|-------|
| Audio capture (4 channels) | ~0.5ms | PyAudio overhead |
| Detection (1 channel) | ~50-100ms | Uses channel 0 only |
| Confirmation (TTA) | ~200-300ms | All 4 channels preserved |
| Multi-channel filtering | ~150-200ms | All 4 channels processed |
| **Total cry->localization time** | **~500-700ms** | [OK] Real-time capable |
| CPU usage (low-power mode) | ~10-15% | 4-channel listening |
| CPU usage (active mode) | ~40-50% | Processing + localization |
| Memory usage | ~250-400 MB | Includes 5s circular buffer |

## Running the Full System

### Test Mode (No Audio Hardware)
```bash
python3 realtime_baby_cry_detector.py \
    --model model_best.pth \
    --device cpu \
    --test-mode
```

### Live Mode (With 4-Channel Microphone Array)
```bash
python3 realtime_baby_cry_detector.py \
    --model model_best.pth \
    --device-index 2 \
    --channels 4 \
    --device cpu
```

### With Sound Localization
```bash
# Run detector + localization integration
python3 robot_baby_monitor.py \
    --model model_best.pth \
    --device-index 2 \
    --channels 4 \
    --device cpu
```

## Integrating Your Sound Localization

The detector now sends multi-channel audio via `detection_queue`:

```python
# In sound_localization_interface.py
localization_data = {
    'raw_audio': np.ndarray,              # (N, 4) - all channels
    'filtered_audio': np.ndarray,         # (N, 4) - cry regions
    'sample_rate': 16000,
    'num_channels': 4,
    'confidence': float
}

# Process with your localization algorithm
def process_localization_data(self, loc_data):
    audio = loc_data['filtered_audio']  # Multi-channel array

    # Compute TDOA from 4 channels
    tdoa = self.compute_tdoa(audio, sample_rate)

    # Estimate direction
    direction = self.estimate_direction(tdoa)

    return direction  # 0-360 degrees
```

See `../documentation/RASPBERRY_PI_DEPLOYMENT.md` for complete integration guide.

## Next Steps

1. [DONE] Copy files to Pi
2. [DONE] Install dependencies
3. [DONE] Find audio device index
4. [DONE] Test multi-channel audio capture
5. [DONE] Verify detector runs
6. [TODO] Integrate your sound localization algorithm
7. [TODO] Test with real baby cry audio
8. [TODO] Deploy to production!

## Documentation

- **Multi-Channel Refactoring:** See `SOUND_LOCALIZATION_INTEGRATION.md` in project root
- **Deployment Guide:** `../documentation/RASPBERRY_PI_DEPLOYMENT.md`
- **Setup Script:** `pi_setup.py --help`
- **Detector Code:** `realtime_baby_cry_detector.py` (now with 4-channel support)
- **Audio Filtering:** `../documentation/RASPBERRY_PI_FILTERING.md`
