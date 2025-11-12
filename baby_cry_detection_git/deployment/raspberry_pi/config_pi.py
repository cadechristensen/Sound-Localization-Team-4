"""
Raspberry Pi optimized configuration for baby cry detection.
Lightweight filtering settings for real-time performance.

This is a standalone config file that doesn't require the src directory.
"""

import os
from pathlib import Path


class Config:
    """Base configuration class - embedded for standalone Pi deployment."""

    # Data paths
    DATA_DIR = Path("data")
    RESULTS_DIR = Path("results")

    # Audio processing parameters
    SAMPLE_RATE = 16000
    DURATION = 3.0
    N_MELS = 128
    N_FFT = 2048
    HOP_LENGTH = 512
    F_MIN = 0
    F_MAX = 8000

    # Data augmentation parameters
    NOISE_FACTOR = 0.005
    TIME_STRETCH_RATE = [0.8, 1.2]
    PITCH_SHIFT_STEPS = [-2, 2]

    # Model architecture parameters
    INPUT_CHANNELS = 1
    CNN_CHANNELS = [32, 64, 128, 256]
    CNN_KERNEL_SIZE = 3
    CNN_DROPOUT = 0.2

    # Transformer parameters
    D_MODEL = 256
    N_HEADS = 8
    N_LAYERS = 4
    TRANSFORMER_DROPOUT = 0.1

    # Training parameters
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    NUM_EPOCHS = 60
    PATIENCE = 15
    WARMUP_EPOCHS = 3

    # Data split ratios
    TRAIN_RATIO = 0.6
    VAL_RATIO = 0.20
    TEST_RATIO = 0.20

    # Multi-task learning mode
    MULTI_CLASS_MODE = False

    # Class labels - Binary mode
    BINARY_CLASS_LABELS = {
        0: 'non_cry',
        1: 'cry'
    }

    # Class labels - Multi-class mode
    MULTI_CLASS_LABELS = {
        0: 'cry',
        1: 'adult_speech',
        2: 'baby_noncry',
        3: 'environmental'
    }

    # Active class labels (based on mode)
    @property
    def CLASS_LABELS(self):
        return self.MULTI_CLASS_LABELS if self.MULTI_CLASS_MODE else self.BINARY_CLASS_LABELS

    @property
    def NUM_CLASSES(self):
        return len(self.MULTI_CLASS_LABELS) if self.MULTI_CLASS_MODE else len(self.BINARY_CLASS_LABELS)

    # Supported audio formats
    SUPPORTED_FORMATS = ['.wav', '.ogg', '.mp3', '.flac', '.m4a', '.3gp', '.webm', '.mp4']

    # Device configuration
    DEVICE = 'cuda' if os.getenv('CUDA_AVAILABLE', 'false').lower() == 'true' else 'cpu'
    NUM_WORKERS = 4

    # Loss function parameters
    CRY_WEIGHT_MULTIPLIER = 1.3
    FOCAL_LOSS_GAMMA = 2.5
    FOCAL_LOSS_ALPHA = 0.25

    # Logging configuration
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

    # Raspberry Pi optimization settings
    QUANTIZE_MODEL = True
    OPTIMIZE_FOR_MOBILE = True

    # Inference settings
    CONFIDENCE_THRESHOLD = 0.5
    SLIDING_WINDOW_SIZE = 1.0
    OVERLAP_RATIO = 0.5

    # Raspberry Pi optimization settings
    PI_BUFFER_SIZE = 1024
    PI_SAMPLE_RATE = 16000
    PI_CHANNELS = 1

    # Advanced Filtering Configuration
    USE_ADVANCED_FILTERING = True

    # Voice Activity Detection (VAD) parameters
    VAD_FRAME_LENGTH = 400
    VAD_HOP_LENGTH = 160
    VAD_ENERGY_THRESHOLD = 0.01
    VAD_FREQ_MIN = 200
    VAD_FREQ_MAX = 1000

    # Noise Filtering parameters
    HIGHPASS_CUTOFF = 100
    BANDPASS_LOW = 200
    BANDPASS_HIGH = 2000
    NOISE_REDUCE_STRENGTH = 0.5

    # Deep spectrum features
    USE_DEEP_SPECTRUM = False
    EXTRACT_MFCC_DELTAS = False
    EXTRACT_SPECTRAL_CONTRAST = False
    EXTRACT_CHROMA = False


class ConfigPi(Config):
    """
    Raspberry Pi optimized configuration.

    Key differences from base Config:
    - Reduced filtering complexity for speed
    - Optimized for real-time performance
    - Lower resource usage
    """

    # Advanced Filtering Configuration (Pi-Optimized)
    USE_ADVANCED_FILTERING = True  # Keep enabled - filters are fast

    # Voice Activity Detection (Enabled for efficiency)
    # VAD helps by only processing audio when activity is detected
    VAD_FRAME_LENGTH = 400  # 25ms at 16kHz
    VAD_HOP_LENGTH = 160    # 10ms at 16kHz
    VAD_ENERGY_THRESHOLD = 0.015  # Slightly higher threshold for Pi (reduce false triggers)
    VAD_FREQ_MIN = 250      # Baby cry starts around 250 Hz (tighter range)
    VAD_FREQ_MAX = 800      # Baby cry harmonics (reduced from 1000 for speed)

    # Noise Filtering parameters (Fast Butterworth filters)
    HIGHPASS_CUTOFF = 100   # Remove rumble below 100 Hz (FAST)
    BANDPASS_LOW = 250      # Baby cry frequency range start (tighter for speed)
    BANDPASS_HIGH = 1500    # Baby cry harmonics range end (reduced from 2000)
    NOISE_REDUCE_STRENGTH = 0.3  # Lighter spectral subtraction (0.3 instead of 0.5 for speed)

    # Deep spectrum features (DISABLED for Pi - too slow)
    USE_DEEP_SPECTRUM = False         # Disable - too computationally expensive
    EXTRACT_MFCC_DELTAS = False       # Disable - not needed for real-time
    EXTRACT_SPECTRAL_CONTRAST = False # Disable - too slow
    EXTRACT_CHROMA = False            # Disable - too slow

    # Model optimization
    QUANTIZE_MODEL = True  # Enable quantization (4x faster, 4x smaller)
    OPTIMIZE_FOR_MOBILE = True

    # Pi-specific performance settings
    PI_USE_VAD_GATING = True  # Only run model when VAD detects activity
    PI_VAD_BUFFER_SIZE = 3    # Number of VAD frames to buffer before detection
    PI_BATCH_SIZE = 1         # Process one sample at a time for low latency
    PI_NUM_WORKERS = 0        # Single-threaded for Pi (avoid overhead)

    # Inference settings optimized for Pi
    CONFIDENCE_THRESHOLD = 0.6  # Slightly higher to reduce false positives

    # Memory optimization
    BATCH_SIZE = 1  # Process one at a time to reduce memory
    NUM_WORKERS = 0 # No parallel workers on Pi

    # Audio buffer settings for real-time
    PI_BUFFER_SIZE = 2048      # Larger buffer for stability (was 1024)
    PI_SAMPLE_RATE = 16000     # Standard rate
    PI_CHANNELS = 1            # Mono audio
    PI_AUDIO_CHUNK_DURATION = 0.5  # Process 0.5s chunks

    def __init__(self):
        """Initialize Pi configuration."""
        super().__init__()

        # Override device to CPU (Pi doesn't have GPU)
        self.DEVICE = 'cpu'

        # Ensure filtering is properly configured
        self._validate_pi_settings()

    def _validate_pi_settings(self):
        """Validate Pi-specific settings."""
        # Ensure deep spectrum is disabled
        assert not self.USE_DEEP_SPECTRUM, "Deep spectrum must be disabled on Pi"

        # Ensure quantization is enabled
        assert self.QUANTIZE_MODEL, "Model quantization should be enabled for Pi"

        # Ensure single-threaded processing
        assert self.NUM_WORKERS == 0, "Pi should use single-threaded processing"


def get_filtering_performance_estimate(config: ConfigPi) -> dict:
    """
    Estimate filtering performance on Raspberry Pi.

    Args:
        config: Pi configuration

    Returns:
        Dictionary with performance estimates
    """
    estimates = {
        'high_pass_filter': {
            'time_per_second': '~2ms',
            'overhead': '~0.2%',
            'recommended': True
        },
        'band_pass_filter': {
            'time_per_second': '~2ms',
            'overhead': '~0.2%',
            'recommended': True
        },
        'spectral_subtraction': {
            'time_per_second': '~8ms',
            'overhead': '~0.8%',
            'recommended': True,
            'note': 'Reduced strength (0.3) for speed'
        },
        'vad': {
            'time_per_second': '~3ms',
            'overhead': '~0.3%',
            'recommended': True,
            'benefit': 'Reduces processing by only running model when activity detected'
        },
        'deep_spectrum': {
            'time_per_second': '~200ms',
            'overhead': '~20%',
            'recommended': False,
            'note': 'DISABLED - Too slow for real-time on Pi'
        },
        'total_filtering_overhead': {
            'time_per_second': '~15ms',
            'overhead': '~1.5%',
            'real_time_capable': True,
            'note': 'With basic filters only (high-pass, band-pass, spectral subtraction, VAD)'
        },
        'model_inference': {
            'time_per_second': '~150-250ms',
            'overhead': '~15-25%',
            'note': 'With quantization. Main bottleneck, not filtering.'
        }
    }

    return estimates


def print_pi_filtering_info():
    """Print information about Pi filtering configuration."""
    config = ConfigPi()

    print("\n" + "="*70)
    print("RASPBERRY PI FILTERING CONFIGURATION")
    print("="*70)

    print("\nENABLED FILTERS (Fast & Recommended):")
    print(f"  • High-pass filter: {config.HIGHPASS_CUTOFF} Hz cutoff")
    print(f"  • Band-pass filter: {config.BANDPASS_LOW}-{config.BANDPASS_HIGH} Hz")
    print(f"  • Spectral subtraction: {config.NOISE_REDUCE_STRENGTH} strength")
    print(f"  • Voice Activity Detection: Threshold {config.VAD_ENERGY_THRESHOLD}")

    print("\nDISABLED FILTERS (Too Slow for Real-Time):")
    print(f"  • Deep spectrum features: {config.USE_DEEP_SPECTRUM}")
    print(f"  • MFCC deltas: {config.EXTRACT_MFCC_DELTAS}")
    print(f"  • Spectral contrast: {config.EXTRACT_SPECTRAL_CONTRAST}")
    print(f"  • Chroma features: {config.EXTRACT_CHROMA}")

    print("\nPERFORMANCE ESTIMATES (Raspberry Pi 4):")
    estimates = get_filtering_performance_estimate(config)

    print(f"  Filtering overhead: {estimates['total_filtering_overhead']['time_per_second']}/second")
    print(f"  Model inference: {estimates['model_inference']['time_per_second']}/second")
    print(f"  Real-time capable: {estimates['total_filtering_overhead']['real_time_capable']}")

    print("\nKEY OPTIMIZATIONS:")
    print("  1. VAD gating: Only run model when activity detected (~50% reduction)")
    print("  2. Quantized model: 4x faster inference")
    print("  3. Lightweight filters: <2% overhead")
    print("  4. Single-threaded: Reduced context switching")

    print("\nEXPECTED PERFORMANCE:")
    print("  • Audio chunk: 0.5 seconds")
    print("  • Processing time: ~100-150ms")
    print("  • Real-time factor: 0.2-0.3 (3-5x faster than real-time)")
    print("  • Latency: <200ms (acceptable for baby monitor)")

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    print_pi_filtering_info()
