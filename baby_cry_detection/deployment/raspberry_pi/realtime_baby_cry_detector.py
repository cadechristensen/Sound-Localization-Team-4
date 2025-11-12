"""
Real-Time Baby Cry Detection System for Raspberry Pi 5
Optimized for TI PCM6260-Q1 4-microphone array with low-power listening mode.
Interfaces with sound localization model for robot navigation.
"""

import torch
import torchaudio
import numpy as np
import pyaudio
import queue
import threading
import time
import logging
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, Callable
import multiprocessing as mp
from dataclasses import dataclass

from src.config import Config
from src.model import create_model
from src.audio_filter import BabyCryAudioFilter


@dataclass
class DetectionResult:
    """Container for detection results."""
    is_cry: bool
    confidence: float
    timestamp: float
    audio_buffer: np.ndarray
    filtered_audio: Optional[np.ndarray] = None


class CircularAudioBuffer:
    """Circular buffer for continuous multi-channel audio capture with context."""

    def __init__(self, max_duration: float, sample_rate: int, num_channels: int = 4):
        """
        Initialize circular buffer for multi-channel audio.

        Args:
            max_duration: Maximum duration to store (seconds)
            sample_rate: Audio sample rate
            num_channels: Number of audio channels to preserve
        """
        self.max_samples = int(max_duration * sample_rate)
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        # Store as (num_samples, num_channels) to preserve phase relationships
        self.buffer = np.zeros((self.max_samples, num_channels), dtype=np.float32)
        self.write_idx = 0
        self.is_full = False
        self.lock = threading.Lock()

    def add(self, audio_chunk: np.ndarray):
        """
        Add multi-channel audio chunk to buffer.

        Args:
            audio_chunk: Audio data with shape (num_samples, num_channels)
        """
        with self.lock:
            chunk_len = len(audio_chunk)
            remaining_space = self.max_samples - self.write_idx

            if chunk_len <= remaining_space:
                # Fits without wrapping
                self.buffer[self.write_idx:self.write_idx + chunk_len] = audio_chunk
                self.write_idx += chunk_len
                if self.write_idx >= self.max_samples:
                    self.write_idx = 0
                    self.is_full = True
            else:
                # Need to wrap around
                self.buffer[self.write_idx:] = audio_chunk[:remaining_space]
                overflow = chunk_len - remaining_space
                self.buffer[:overflow] = audio_chunk[remaining_space:]
                self.write_idx = overflow
                self.is_full = True

    def get_last_n_seconds(self, duration: float) -> np.ndarray:
        """
        Get last N seconds of multi-channel audio.

        Args:
            duration: Duration in seconds

        Returns:
            Audio array with shape (num_samples, num_channels)
        """
        with self.lock:
            n_samples = int(duration * self.sample_rate)
            n_samples = min(n_samples, self.max_samples)

            if not self.is_full:
                # Buffer not full yet, return what we have
                return self.buffer[:self.write_idx].copy()
            else:
                # Buffer is full, get last n_samples in circular order
                start_idx = (self.write_idx - n_samples) % self.max_samples

                if start_idx < self.write_idx:
                    # No wrap around
                    return self.buffer[start_idx:self.write_idx].copy()
                else:
                    # Wrap around case
                    return np.vstack([
                        self.buffer[start_idx:],
                        self.buffer[:self.write_idx]
                    ])

    def clear(self):
        """Clear the buffer."""
        with self.lock:
            self.buffer = np.zeros((self.max_samples, self.num_channels), dtype=np.float32)
            self.write_idx = 0
            self.is_full = False


class RealtimeBabyCryDetector:
    """
    Real-time baby cry detection with low-power mode and sound localization integration.
    Designed for Raspberry Pi 5 with TI PCM6260-Q1 microphone array.
    """

    def __init__(
        self,
        model_path: str,
        config: Optional[Config] = None,
        use_tta: bool = False,
        detection_threshold: float = 0.75,
        confirmation_threshold: float = 0.85,
        device: Optional[str] = None,
        audio_device_index: Optional[int] = None,
        num_channels: int = 4
    ):
        """
        Initialize real-time detector.

        Args:
            model_path: Path to trained model checkpoint
            config: Configuration object
            use_tta: Use test-time augmentation (slower but more accurate)
            detection_threshold: Initial detection threshold
            confirmation_threshold: Confirmation threshold for wake-up
            device: Device to run on ('cpu' or 'cuda')
            audio_device_index: PyAudio device index for microphone array
            num_channels: Number of microphone channels (4 for PCM6260-Q1)
        """
        self.config = config or Config()
        self.use_tta = use_tta
        self.detection_threshold = detection_threshold
        self.confirmation_threshold = confirmation_threshold
        self.num_channels = num_channels
        self.audio_device_index = audio_device_index

        # Device setup
        self.device = torch.device(device if device else
                                   ('cuda' if torch.cuda.is_available() else 'cpu'))

        logging.info(f"Using device: {self.device}")

        # Load baby cry detection model
        self.model = create_model(self.config).to(self.device)
        self._load_checkpoint(model_path)
        self.model.eval()
        logging.info(f"Baby cry model loaded from {model_path}")

        # Initialize audio filter
        self.audio_filter = BabyCryAudioFilter(self.config, model_path)
        logging.info("Audio filter initialized")

        # Audio processing setup
        self.chunk_duration = 1.0  # Process 1 second chunks in low-power mode
        self.chunk_size = int(self.chunk_duration * self.config.SAMPLE_RATE)
        self.context_duration = 5.0  # Keep 5 seconds of context

        # Circular buffer for multi-channel audio context (preserves phase)
        self.audio_buffer = CircularAudioBuffer(
            max_duration=self.context_duration,
            sample_rate=self.config.SAMPLE_RATE,
            num_channels=self.num_channels
        )

        # Mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.SAMPLE_RATE,
            n_fft=self.config.N_FFT,
            hop_length=self.config.HOP_LENGTH,
            n_mels=self.config.N_MELS,
            f_min=self.config.F_MIN,
            f_max=self.config.F_MAX
        ).to(self.device)

        # Threading and IPC
        self.audio_queue = queue.Queue(maxsize=100)
        self.detection_queue = mp.Queue(maxsize=10)  # For IPC to localization
        self.is_running = False
        self.low_power_mode = True

        # State tracking
        self.last_detection_time = 0
        self.detection_cooldown = 2.0  # Seconds before re-detection

        # Callbacks
        self.on_cry_detected: Optional[Callable] = None

    def _load_checkpoint(self, model_path: str):
        """Load model weights from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for continuous multi-channel audio capture."""
        if status:
            logging.warning(f"Audio callback status: {status}")

        # Convert bytes to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.float32)

        # Reshape to preserve all channels (num_frames, num_channels)
        # This preserves phase relationships between channels for beamforming
        audio_data = audio_data.reshape(-1, self.num_channels)

        # Add to queue for processing
        try:
            self.audio_queue.put_nowait(audio_data)
        except queue.Full:
            logging.warning("Audio queue full, dropping frame")

        return (in_data, pyaudio.paContinue)

    def preprocess_audio(self, audio: np.ndarray) -> torch.Tensor:
        """
        Preprocess multi-channel audio for model input.

        Takes multi-channel audio and extracts/processes the primary channel
        while preserving the full multi-channel data for localization.

        Args:
            audio: Multi-channel audio numpy array with shape (num_samples, num_channels)

        Returns:
            Preprocessed audio tensor from primary channel (channel 0)
        """
        # Convert to tensor
        if audio.ndim == 1:
            # Mono audio (fallback)
            waveform = torch.from_numpy(audio).float()
        else:
            # Multi-channel audio - use primary channel for detection
            # Keep all channels for phase-preserving localization
            waveform = torch.from_numpy(audio[:, 0]).float()  # Use channel 0 for detection

        # Ensure correct duration
        target_length = int(self.config.DURATION * self.config.SAMPLE_RATE)
        if len(waveform) < target_length:
            waveform = torch.nn.functional.pad(waveform, (0, target_length - len(waveform)))
        elif len(waveform) > target_length:
            waveform = waveform[:target_length]

        return waveform

    def audio_to_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert audio waveform to mel spectrogram."""
        waveform = waveform.to(self.device).unsqueeze(0)

        # Compute mel spectrogram
        mel_spec = self.mel_transform(waveform)

        # Convert to log scale and normalize
        mel_spec = torch.log(mel_spec + 1e-9)
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-9)

        # Add channel dimension
        mel_spec = mel_spec.unsqueeze(0)  # (1, 1, n_mels, time)

        return mel_spec

    def predict_with_tta(self, spectrogram: torch.Tensor, n_augments: int = 3) -> torch.Tensor:
        """Predict with test-time augmentation."""
        predictions = []

        with torch.no_grad():
            # Original prediction
            predictions.append(self.model(spectrogram))

            # Augmented predictions
            for _ in range(n_augments - 1):
                # Time shift
                shift = torch.randint(-5, 6, (1,)).item()
                aug_spec = torch.roll(spectrogram, shifts=shift, dims=-1)

                # Light noise
                noise = torch.randn_like(aug_spec) * 0.005
                aug_spec = aug_spec + noise

                predictions.append(self.model(aug_spec))

        return torch.mean(torch.stack(predictions), dim=0)

    def detect_cry(self, audio: np.ndarray, use_tta: bool = False) -> Tuple[bool, float]:
        """
        Detect baby cry in audio chunk.

        Args:
            audio: Audio numpy array
            use_tta: Whether to use TTA

        Returns:
            Tuple of (is_cry, confidence)
        """
        # Preprocess
        waveform = self.preprocess_audio(audio)
        spectrogram = self.audio_to_spectrogram(waveform)

        # Predict
        with torch.no_grad():
            if use_tta:
                logits = self.predict_with_tta(spectrogram)
            else:
                logits = self.model(spectrogram)

            probs = torch.softmax(logits, dim=1)
            cry_prob = probs[0, 1].item()
            is_cry = cry_prob > self.detection_threshold

        return is_cry, cry_prob

    def confirm_and_filter(self, audio: np.ndarray) -> DetectionResult:
        """
        Confirm detection with TTA and prepare multi-channel audio for localization.

        Args:
            audio: Multi-channel audio numpy array with shape (num_samples, num_channels)

        Returns:
            DetectionResult with preserved multi-channel audio for sound localization
        """
        # Confirm with TTA for higher accuracy (uses primary channel)
        is_cry, confidence = self.detect_cry(audio, use_tta=True)

        filtered_audio = None
        if is_cry and confidence >= self.confirmation_threshold:
            # Apply audio filtering for sound localization
            logging.info("Applying audio filtering for sound localization...")

            # For multi-channel audio, preserve all channels and phase relationships
            if audio.ndim > 1:
                # Use the multi-channel aware filter that preserves phase
                audio_tensor = torch.from_numpy(audio).float()
                filtered_tensor, cry_segments = self.audio_filter.isolate_baby_cry_multichannel(
                    audio_tensor,
                    cry_threshold=0.6
                )
                filtered_audio = filtered_tensor.numpy()
            else:
                # Fallback for mono audio
                audio_tensor = torch.from_numpy(audio).float()
                filtered_tensor, cry_segments = self.audio_filter.isolate_baby_cry(
                    audio_tensor,
                    cry_threshold=0.6
                )
                filtered_audio = filtered_tensor.numpy()

        return DetectionResult(
            is_cry=is_cry,
            confidence=confidence,
            timestamp=time.time(),
            audio_buffer=audio,
            filtered_audio=filtered_audio
        )

    def wake_robot(self, detection: DetectionResult):
        """
        Wake robot from low-power mode and send multi-channel data to sound localization.

        Sends raw multi-channel audio with preserved phase relationships for beamforming
        and sound source localization.

        Args:
            detection: Detection result with multi-channel filtered audio
        """
        logging.info(f"BABY CRY DETECTED! Confidence: {detection.confidence:.2%}")
        logging.info(f"Waking robot from low-power mode... ({self.num_channels}-channel audio)")

        self.low_power_mode = False

        # Prepare data for sound localization
        # Send raw multi-channel audio with phase information preserved
        localization_data = {
            'timestamp': detection.timestamp,
            'confidence': detection.confidence,
            'raw_audio': detection.audio_buffer,  # Full multi-channel audio
            'filtered_audio': detection.filtered_audio,  # Multi-channel filtered (cry regions only)
            'sample_rate': self.config.SAMPLE_RATE,
            'num_channels': self.num_channels,
            'audio_shape': detection.audio_buffer.shape if detection.audio_buffer is not None else None
        }

        # Send to sound localization process via queue
        try:
            self.detection_queue.put(localization_data, timeout=1.0)
            logging.info(f"Multi-channel audio ({self.num_channels} channels) sent to sound localization")
            if detection.filtered_audio is not None:
                logging.info(f"  Filtered audio shape: {detection.filtered_audio.shape}")
        except queue.Full:
            logging.error("Localization queue full, data not sent")

        # Call user callback if set
        if self.on_cry_detected:
            self.on_cry_detected(detection)

    def process_audio_stream(self):
        """Main processing loop for audio stream."""
        logging.info("Audio processing thread started")

        while self.is_running:
            try:
                # Get audio chunk from queue (timeout for responsiveness)
                audio_chunk = self.audio_queue.get(timeout=0.5)

                # Add to circular buffer
                self.audio_buffer.add(audio_chunk)

                # Check cooldown
                if time.time() - self.last_detection_time < self.detection_cooldown:
                    continue

                if self.low_power_mode:
                    # Low-power mode: Quick detection without TTA
                    is_cry, confidence = self.detect_cry(audio_chunk, use_tta=False)

                    if is_cry:
                        logging.info(f"Potential cry detected (confidence: {confidence:.2%})")

                        # Get context audio (last 3-5 seconds)
                        context_audio = self.audio_buffer.get_last_n_seconds(3.0)

                        # Confirm with TTA and filter
                        detection = self.confirm_and_filter(context_audio)

                        if detection.is_cry and detection.confidence >= self.confirmation_threshold:
                            # Wake robot
                            self.wake_robot(detection)
                            self.last_detection_time = time.time()
                        else:
                            logging.info(f"False positive filtered out (conf: {detection.confidence:.2%})")

                else:
                    # Active mode: Robot is navigating
                    # Continue monitoring but don't wake again
                    pass

            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error in audio processing: {e}", exc_info=True)

    def start(self, stream_audio: bool = True):
        """
        Start real-time detection.

        Args:
            stream_audio: Whether to start audio streaming (set False for testing)
        """
        logging.info("Starting real-time baby cry detector...")
        self.is_running = True

        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self.process_audio_stream,
            daemon=True
        )
        self.processing_thread.start()

        if stream_audio:
            # Initialize PyAudio
            self.audio = pyaudio.PyAudio()

            # List available devices
            if self.audio_device_index is None:
                logging.info("Available audio devices:")
                for i in range(self.audio.get_device_count()):
                    info = self.audio.get_device_info_by_index(i)
                    logging.info(f"  [{i}] {info['name']} (channels: {info['maxInputChannels']})")

            # Open audio stream
            device_index = self.audio_device_index

            try:
                self.stream = self.audio.open(
                    format=pyaudio.paFloat32,
                    channels=self.num_channels,
                    rate=self.config.SAMPLE_RATE,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=self.chunk_size,
                    stream_callback=self._audio_callback
                )

                self.stream.start_stream()
                logging.info(f"Audio stream started (device: {device_index}, channels: {self.num_channels})")

            except Exception as e:
                logging.error(f"Error starting audio stream: {e}")
                logging.info("Detector running in test mode without audio input")

        logging.info("Real-time detector running in LOW-POWER MODE")

    def stop(self):
        """Stop real-time detection."""
        logging.info("Stopping real-time detector...")

        self.is_running = False

        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()

        if hasattr(self, 'audio'):
            self.audio.terminate()

        if hasattr(self, 'processing_thread'):
            self.processing_thread.join(timeout=2.0)

        logging.info("Real-time detector stopped")

    def reset_to_low_power(self):
        """Reset detector to low-power mode after robot task completion."""
        logging.info("Resetting to low-power listening mode")
        self.low_power_mode = True
        self.audio_buffer.clear()


def main():
    """Command-line interface for real-time detection."""
    parser = argparse.ArgumentParser(description='Real-Time Baby Cry Detection for Raspberry Pi')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--device-index', type=int, default=None,
                       help='Audio device index for microphone array')
    parser.add_argument('--channels', type=int, default=4,
                       help='Number of microphone channels (default: 4)')
    parser.add_argument('--threshold', type=float, default=0.75,
                       help='Detection threshold (default: 0.75)')
    parser.add_argument('--confirm-threshold', type=float, default=0.85,
                       help='Confirmation threshold for wake-up (default: 0.85)')
    parser.add_argument('--no-tta', action='store_true',
                       help='Disable TTA for confirmation (faster)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu/cuda)')
    parser.add_argument('--test-mode', action='store_true',
                       help='Run in test mode without audio input')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Initialize detector
    detector = RealtimeBabyCryDetector(
        model_path=args.model,
        use_tta=not args.no_tta,
        detection_threshold=args.threshold,
        confirmation_threshold=args.confirm_threshold,
        device=args.device,
        audio_device_index=args.device_index,
        num_channels=args.channels
    )

    # Optional: Set callback
    def on_cry_callback(detection: DetectionResult):
        print(f"\n{'='*70}")
        print(f"BABY CRY ALERT!")
        print(f"  Confidence: {detection.confidence:.1%}")
        print(f"  Timestamp: {time.strftime('%H:%M:%S', time.localtime(detection.timestamp))}")
        print(f"  Filtered audio ready for sound localization")
        print(f"{'='*70}\n")

    detector.on_cry_detected = on_cry_callback

    # Start detector
    try:
        detector.start(stream_audio=not args.test_mode)

        print("\n" + "="*70)
        print("Real-Time Baby Cry Detector - ACTIVE")
        print("="*70)
        print(f"Mode: LOW-POWER LISTENING")
        print(f"Microphone Channels: {args.channels}")
        print(f"Detection Threshold: {args.threshold:.0%}")
        print(f"Confirmation Threshold: {args.confirm_threshold:.0%}")
        print(f"Device: {args.device}")
        print("="*70)
        print("\nPress Ctrl+C to stop\n")

        # Keep running
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\nShutting down...")
        detector.stop()
        print("Detector stopped successfully")


if __name__ == "__main__":
    main()
