# Baby Cry Detection & Sound Localization System

A complete AI-powered baby monitoring system combining real-time cry detection with sound localization for autonomous robot navigation.

---

## Overview

This system detects baby cries in real-time, filters the audio, and determines the baby's location using a 4-microphone array. Designed for deployment on Raspberry Pi 5 for robotic baby monitoring applications.

### Key Features

- **Real-time Baby Cry Detection** - CNN-Transformer hybrid model
- **Audio Filtering** - Isolates baby cry frequencies (100-3000 Hz)
- **Sound Localization Interface** - Ready for TDOA/beamforming integration
- **Low-Power Listening Mode** - Optimized for battery operation
- **Two-Stage Detection** - Fast initial detection + TTA confirmation
- **Raspberry Pi Optimized** - Designed for edge deployment

---

## System Architecture

```
LOW-POWER LISTENING (1-sec chunks)
    |
    v
CRY DETECTED? (>75%)
    |
    v
CAPTURE CONTEXT (3-5 seconds)
    |
    v
CONFIRM WITH TTA (>85%)
    |
    v
AUDIO FILTERING (noise removal, frequency isolation)
    |
    v
SOUND LOCALIZATION (direction + distance)
    |
    v
ROBOT NAVIGATION
```

---

## Project Structure

```
baby_cry_detection_2/
|
+-- README.md                    # This file
+-- requirements.txt             # Python dependencies
|
+-- src/                         # Core source code
|   +-- config.py               # System configuration
|   +-- model.py                # Neural network architecture
|   +-- audio_filter.py         # Audio processing & filtering
|   +-- data_preprocessing.py   # Audio preprocessing utilities
|   +-- train.py                # Training module
|   +-- evaluate.py             # Evaluation module
|   +-- dataset.py              # Dataset handling
|   +-- utils.py                # Utilities
|
+-- training/                    # Training scripts & docs
|   +-- main.py                 # Main training script
|   +-- EVALUATION_GUIDE.md     # Model evaluation guide
|
+-- deployment/                  # Deployment files
|   +-- raspberry_pi/           # Raspberry Pi deployment
|   |   +-- robot_baby_monitor.py
|   |   +-- realtime_baby_cry_detector.py
|   |   +-- sound_localization_interface.py
|   |   +-- verify_raspberry_pi_files.sh
|   |   +-- *.py (utility scripts)
|   +-- documentation/          # Deployment guides
|       +-- RASPBERRY_PI_DEPLOYMENT.md
|       +-- RASPBERRY_PI_FILES_LIST.md
|       +-- RASPBERRY_PI_SUMMARY.md
|       +-- SOUND_LOCALIZATION_INTEGRATION_GUIDE.md
|       +-- *.md (additional docs)
|
+-- data/                        # Training data
|   +-- cry/
|   +-- cry_ICSD/
|   +-- cry_crycaleb/            # CryCeleb dataset
|   +-- baby_noncry/
|   +-- adult_speech/
|   +-- environmental/
|
+-- results/                     # Training results & models
    +-- train_YYYY-MM-DD_HH-MM-SS/
        +-- model_best.pth      # Best model checkpoint
        +-- logs/
        +-- plots/
```

---

## Quick Start

### 1. Training (Development)

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python training/main.py train

# Evaluate the model
python training/main.py evaluate --model results/train_*/model_best.pth
```

### 2. Deployment (Raspberry Pi)

See complete deployment guide: [deployment/documentation/RASPBERRY_PI_SUMMARY.md](deployment/documentation/RASPBERRY_PI_SUMMARY.md)

**Quick deployment:**
```bash
# Transfer files to Raspberry Pi
scp deployment/raspberry_pi/*.py \
    src/{config,model,audio_filter,data_preprocessing}.py \
    results/train_*/model_best.pth \
    requirements.txt \
    pi@raspberrypi.local:~/baby_monitor/

# On Raspberry Pi
cd ~/baby_monitor
python3 robot_baby_monitor.py --model model_best.pth --device-index 2 --channels 4
```

---

## Hardware Requirements

### Development
- Python 3.8+
- GPU (optional, recommended for training)
- 8GB+ RAM

### Deployment (Raspberry Pi)
- Raspberry Pi 5 (8GB RAM)
- TI PCM6260-Q1 microphone array (4 channels)
- 4x Electret Condenser Microphones (-24dB, Mouser #: 665-AOM-5024L-HD-R)

---

## Model Architecture

- **Input:** 3-second audio clips (16kHz, mono)
- **Features:** Log Mel spectrograms (128 mel bins)
- **Architecture:** CNN + Transformer hybrid
  - CNN layers: Extract spatial features from spectrograms
  - Transformer layers: Capture temporal dependencies
  - Attention pooling: Focus on important frames
- **Output:** Binary classification (cry / non-cry)

### Performance
- **Accuracy:** ~96%
- **Precision:** ~95%
- **Recall:** ~97%
- **Inference time:** 100-200ms (Raspberry Pi CPU)

---

## Dataset

### Training Data (Ratio: 0.968:1 cry:non-cry)
- **Cry samples:** 4,557 files
  - cry/ (822)
  - cry_ICSD/ (2,312)
  - cry_crycaleb/ (1,423)
- **Non-cry samples:** 4,707 files
  - baby_noncry/ (356)
  - adult_speech/ (2,241)
  - environmental/ (2,110)

**Total:** 9,264 audio files

---

## Sound Localization Integration

To integrate your sound localization model:

1. **Edit:** `deployment/raspberry_pi/sound_localization_interface.py`
2. **Find:** Line 63 - "INTEGRATE YOUR SOUND LOCALIZATION MODEL HERE"
3. **Replace** placeholder with your model

See complete guide: [deployment/documentation/SOUND_LOCALIZATION_INTEGRATION_GUIDE.md](deployment/documentation/SOUND_LOCALIZATION_INTEGRATION_GUIDE.md)

---

## Usage Examples

### Training
```bash
# Basic training
python training/main.py train

# Resume from checkpoint
python training/main.py train --resume results/train_*/model_best.pth

# Custom configuration
python training/main.py train --config custom_config.py
```

### Evaluation
```bash
# Evaluate on test set
python training/main.py evaluate --model results/train_*/model_best.pth

# Detailed analysis
python training/main.py analyze --model results/train_*/model_best.pth
```

### Deployment
```bash
# Full system (cry detection + localization)
python deployment/raspberry_pi/robot_baby_monitor.py \
    --model model_best.pth \
    --device-index 2 \
    --channels 4

# Cry detection only
python deployment/raspberry_pi/realtime_baby_cry_detector.py \
    --model model_best.pth \
    --device-index 2
```

---

## Quick Reference

### Start Here
- [docs/analysis/START_HERE.md](docs/analysis/START_HERE.md) - First-time users start here
- [docs/analysis/ANALYSIS_INDEX.md](docs/analysis/ANALYSIS_INDEX.md) - Navigation guide for all docs

### Development
- [QUICK_START.md](QUICK_START.md) - Training commands
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Inference quick reference
- [docs/guides/EVALUATION_GUIDE.md](docs/guides/EVALUATION_GUIDE.md) - Model evaluation

### Deployment
- [deployment/documentation/RASPBERRY_PI_SUMMARY.md](deployment/documentation/RASPBERRY_PI_SUMMARY.md) - Quick setup
- [deployment/documentation/RASPBERRY_PI_DEPLOYMENT.md](deployment/documentation/RASPBERRY_PI_DEPLOYMENT.md) - Complete guide
- [deployment/documentation/SOUND_LOCALIZATION_INTEGRATION_GUIDE.md](deployment/documentation/SOUND_LOCALIZATION_INTEGRATION_GUIDE.md) - Integration

### Detailed Documentation

**docs/analysis/** - Project understanding:
- [ANALYSIS_SUMMARY.txt](docs/analysis/ANALYSIS_SUMMARY.txt) - Executive summary
- [PROJECT_STRUCTURE_ANALYSIS.md](docs/analysis/PROJECT_STRUCTURE_ANALYSIS.md) - Comprehensive breakdown
- [CODEBASE_OVERVIEW.md](docs/analysis/CODEBASE_OVERVIEW.md) - Quick file reference
- [FILE_REFERENCE_CARD.md](docs/analysis/FILE_REFERENCE_CARD.md) - File locations
- [VISUAL_PROJECT_MAP.md](docs/analysis/VISUAL_PROJECT_MAP.md) - Diagrams and visuals

**docs/setup/** - Configuration:
- [SETUP_SUMMARY.md](docs/setup/SETUP_SUMMARY.md) - Configuration guide
- [EVALUATION_STATUS.md](docs/setup/EVALUATION_STATUS.md) - Model status
- [FILTERING_SUMMARY.md](docs/setup/FILTERING_SUMMARY.md) - Audio filtering

**docs/guides/** - How-to guides:
- [INFERENCE_WITH_FILTERING_README.md](docs/guides/INFERENCE_WITH_FILTERING_README.md) - Full inference pipeline

---

## Dependencies

Main dependencies:
- PyTorch >= 2.0.0
- torchaudio >= 2.0.0
- numpy >= 1.24.0
- librosa >= 0.10.0
- pyaudio >= 0.2.13 (for real-time audio)
- scipy >= 1.10.0

See [requirements.txt](requirements.txt) for complete list.

---

## Configuration

Key configuration parameters in `src/config.py`:

```python
# Audio
SAMPLE_RATE = 16000
DURATION = 3.0
N_MELS = 128

# Model
CNN_CHANNELS = [32, 64, 128, 256]
D_MODEL = 256
N_HEADS = 8
N_LAYERS = 4

# Training
BATCH_SIZE = 64
LEARNING_RATE = 5e-5
NUM_EPOCHS = 50
```

---

## License

This project is for educational and research purposes.

---

## Acknowledgments

- CryCeleb2023 dataset contributors
- ICSD dataset contributors
- Open-source audio processing libraries

---

## Support

For deployment issues, see:
- [Deployment Guide](deployment/documentation/RASPBERRY_PI_DEPLOYMENT.md)
- [Troubleshooting Section](deployment/documentation/RASPBERRY_PI_DEPLOYMENT.md#troubleshooting)

For training issues, see:
- [Evaluation Guide](training/EVALUATION_GUIDE.md)

---

## Project Status

- [x] Model training and evaluation
- [x] Audio filtering implementation
- [x] Raspberry Pi deployment system
- [x] Sound localization interface
- [ ] Sound localization model integration (user-specific)
- [ ] Robot navigation integration (user-specific)

---

**Version:** 2.0
**Last Updated:** October 2025
