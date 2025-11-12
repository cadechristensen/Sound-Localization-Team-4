"""
Advanced audio filtering techniques for baby cry detection.
Implements state-of-the-art filtering methods from recent research (2024-2025).

Based on best practices from:
- Voice Activity Detection (VAD) for cry segmentation
- High-pass and band-pass filtering for noise removal
- Spectral subtraction for background noise reduction
- Deep spectrum features for noise robustness
"""

import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import librosa
from scipy import signal
from typing import Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

try:
    from .config_pi import Config
except ImportError:
    try:
        from config_pi import Config  # type: ignore
    except ImportError:
        # Fallback to regular config if config_pi not available
        try:
            from .config import Config
        except ImportError:
            from config import Config  # type: ignore


class VoiceActivityDetector:
    """
    Voice Activity Detection (VAD) for cry detection and segmentation.
    Detects and segments baby cry sounds from silent/background periods.
    """

    def __init__(self,
                 sample_rate: int = 16000,
                 frame_length: int = 400,  # 25ms at 16kHz
                 hop_length: int = 160,    # 10ms at 16kHz
                 energy_threshold: float = 0.01,
                 freq_min: int = 200,       # Baby cry range starts around 200 Hz
                 freq_max: int = 1000):     # Baby cry harmonics up to ~1000 Hz
        """
        Initialize VAD with baby cry-specific parameters.

        Args:
            sample_rate: Audio sample rate
            frame_length: Frame length in samples (25ms default)
            hop_length: Hop length in samples (10ms default)
            energy_threshold: Energy threshold for activity detection
            freq_min: Minimum frequency for baby cry detection
            freq_max: Maximum frequency for baby cry detection
        """
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.energy_threshold = energy_threshold
        self.freq_min = freq_min
        self.freq_max = freq_max

    def compute_energy(self, waveform: torch.Tensor) -> np.ndarray:
        """
        Compute short-time energy of the waveform.

        Args:
            waveform: Input audio waveform

        Returns:
            Energy values per frame
        """
        waveform_np = waveform.numpy() if isinstance(waveform, torch.Tensor) else waveform

        # Compute energy in overlapping frames
        frames = librosa.util.frame(waveform_np,
                                    frame_length=self.frame_length,
                                    hop_length=self.hop_length)
        energy = np.sum(frames ** 2, axis=0) / self.frame_length

        return energy

    def compute_zero_crossing_rate(self, waveform: torch.Tensor) -> np.ndarray:
        """
        Compute zero-crossing rate (useful for distinguishing voiced/unvoiced).

        Args:
            waveform: Input audio waveform

        Returns:
            Zero-crossing rate per frame
        """
        waveform_np = waveform.numpy() if isinstance(waveform, torch.Tensor) else waveform

        zcr = librosa.feature.zero_crossing_rate(
            waveform_np,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )

        return zcr[0]

    def compute_spectral_energy_in_band(self, waveform: torch.Tensor) -> np.ndarray:
        """
        Compute energy in baby cry frequency band (200-1000 Hz).

        Args:
            waveform: Input audio waveform

        Returns:
            Energy in cry band per frame
        """
        waveform_np = waveform.numpy() if isinstance(waveform, torch.Tensor) else waveform

        # Compute STFT
        stft = librosa.stft(waveform_np,
                           n_fft=self.frame_length * 2,
                           hop_length=self.hop_length)

        # Get frequencies
        freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.frame_length * 2)

        # Find indices in cry frequency band
        band_mask = (freqs >= self.freq_min) & (freqs <= self.freq_max)

        # Compute energy in band
        band_energy = np.sum(np.abs(stft[band_mask, :]) ** 2, axis=0)

        return band_energy

    def detect_activity(self, waveform: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect voice activity in the waveform using multiple features.

        Args:
            waveform: Input audio waveform

        Returns:
            Tuple of (activity_mask, confidence_scores)
        """
        # Compute multiple features
        energy = self.compute_energy(waveform)
        zcr = self.compute_zero_crossing_rate(waveform)
        band_energy = self.compute_spectral_energy_in_band(waveform)

        # Find minimum length to align all features
        min_len = min(len(energy), len(zcr), len(band_energy))

        # Truncate all features to same length
        energy = energy[:min_len]
        zcr = zcr[:min_len]
        band_energy = band_energy[:min_len]

        # Normalize features
        energy_norm = (energy - energy.min()) / (energy.max() - energy.min() + 1e-8)
        zcr_norm = (zcr - zcr.min()) / (zcr.max() - zcr.min() + 1e-8)
        band_energy_norm = (band_energy - band_energy.min()) / (band_energy.max() - band_energy.min() + 1e-8)

        # Combine features (weighted)
        # High energy + moderate ZCR + high band energy = cry
        confidence = 0.4 * energy_norm + 0.2 * (1 - zcr_norm) + 0.4 * band_energy_norm

        # Threshold
        activity_mask = confidence > self.energy_threshold

        # Apply morphological operations to remove small gaps
        activity_mask = self._smooth_mask(activity_mask, min_duration_frames=5)

        return activity_mask, confidence

    def _smooth_mask(self, mask: np.ndarray, min_duration_frames: int = 5) -> np.ndarray:
        """
        Smooth activity mask to remove short gaps and spurious detections.

        Args:
            mask: Binary activity mask
            min_duration_frames: Minimum duration for activity segments

        Returns:
            Smoothed mask
        """
        from scipy.ndimage import binary_closing, binary_opening

        # Close small gaps
        mask = binary_closing(mask, structure=np.ones(min_duration_frames))

        # Remove small detections
        mask = binary_opening(mask, structure=np.ones(min_duration_frames))

        return mask

    def segment_audio(self, waveform: torch.Tensor) -> list:
        """
        Segment audio into cry and non-cry regions.

        Args:
            waveform: Input audio waveform

        Returns:
            List of (start_sample, end_sample, is_cry) tuples
        """
        activity_mask, _ = self.detect_activity(waveform)

        # Convert frame indices to sample indices
        segments = []
        in_segment = False
        start_frame = 0

        for i, is_active in enumerate(activity_mask):
            if is_active and not in_segment:
                start_frame = i
                in_segment = True
            elif not is_active and in_segment:
                start_sample = start_frame * self.hop_length
                end_sample = i * self.hop_length
                segments.append((start_sample, end_sample, True))
                in_segment = False

        # Handle last segment
        if in_segment:
            start_sample = start_frame * self.hop_length
            end_sample = len(waveform)
            segments.append((start_sample, end_sample, True))

        return segments


class NoiseFilter:
    """
    Advanced noise filtering using multiple techniques:
    - High-pass filtering for low-frequency noise
    - Band-pass filtering for baby cry frequency range
    - Spectral subtraction for background noise
    """

    def __init__(self,
                 sample_rate: int = 16000,
                 highpass_cutoff: int = 100,      # Remove rumble < 100 Hz
                 bandpass_low: int = 200,          # Baby cry range
                 bandpass_high: int = 2000,        # Baby cry harmonics
                 noise_reduce_strength: float = 0.5):
        """
        Initialize noise filter.

        Args:
            sample_rate: Audio sample rate
            highpass_cutoff: High-pass filter cutoff frequency (Hz)
            bandpass_low: Band-pass filter low cutoff (Hz)
            bandpass_high: Band-pass filter high cutoff (Hz)
            noise_reduce_strength: Spectral subtraction strength (0-1)
        """
        self.sample_rate = sample_rate
        self.highpass_cutoff = highpass_cutoff
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high
        self.noise_reduce_strength = noise_reduce_strength

    def apply_highpass_filter(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply high-pass FIR filter to remove low-frequency noise.

        Args:
            waveform: Input audio waveform

        Returns:
            Filtered waveform
        """
        waveform_np = waveform.numpy() if isinstance(waveform, torch.Tensor) else waveform

        # Design high-pass Butterworth filter
        nyquist = self.sample_rate / 2
        normalized_cutoff = self.highpass_cutoff / nyquist

        # 5th order Butterworth filter
        b, a = signal.butter(5, normalized_cutoff, btype='high')

        # Apply filter
        filtered = signal.filtfilt(b, a, waveform_np)

        # Make a copy to ensure positive strides (filtfilt can produce negative strides)
        return torch.tensor(filtered.copy(), dtype=torch.float32)

    def apply_bandpass_filter(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply band-pass filter to focus on baby cry frequency range.

        Args:
            waveform: Input audio waveform

        Returns:
            Filtered waveform
        """
        waveform_np = waveform.numpy() if isinstance(waveform, torch.Tensor) else waveform

        # Design band-pass Butterworth filter
        nyquist = self.sample_rate / 2
        low = self.bandpass_low / nyquist
        high = self.bandpass_high / nyquist

        # 4th order Butterworth filter
        b, a = signal.butter(4, [low, high], btype='band')

        # Apply filter
        filtered = signal.filtfilt(b, a, waveform_np)

        # Make a copy to ensure positive strides
        return torch.tensor(filtered.copy(), dtype=torch.float32)

    def spectral_subtraction(self,
                            waveform: torch.Tensor,
                            noise_profile: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply spectral subtraction for background noise reduction.
        Estimates noise from silent regions or uses provided noise profile.

        Args:
            waveform: Input audio waveform
            noise_profile: Optional pre-computed noise spectrum

        Returns:
            Denoised waveform
        """
        waveform_np = waveform.numpy() if isinstance(waveform, torch.Tensor) else waveform

        # Compute STFT
        n_fft = 1024
        hop_length = 256
        stft = librosa.stft(waveform_np, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        # Estimate noise spectrum from first 10 frames (typically quieter)
        if noise_profile is None:
            noise_spectrum = np.mean(magnitude[:, :10], axis=1, keepdims=True)
        else:
            noise_profile_np = noise_profile.numpy() if isinstance(noise_profile, torch.Tensor) else noise_profile
            noise_spectrum = noise_profile_np.reshape(-1, 1)

        # Spectral subtraction with floor
        alpha = self.noise_reduce_strength
        subtracted = magnitude - alpha * noise_spectrum

        # Apply spectral floor to avoid artifacts
        floor = 0.1 * magnitude
        subtracted = np.maximum(subtracted, floor)

        # Reconstruct signal
        stft_denoised = subtracted * np.exp(1j * phase)
        waveform_denoised = librosa.istft(stft_denoised, hop_length=hop_length)

        return torch.tensor(waveform_denoised, dtype=torch.float32)

    # COMMENTED OUT: Wiener filter introduces phase distortion which breaks sound localization.
    # The filtfilt-based filters and spectral subtraction preserve phase, which is critical
    # for maintaining time-of-arrival (TOA) differences between microphone channels.
    #
    # def apply_wiener_filter(self, waveform: torch.Tensor) -> torch.Tensor:
    #     """
    #     Apply Wiener filtering for noise reduction.
    #
    #     WARNING: This filter introduces phase distortion and should NOT be used
    #     if the audio will be used for sound localization. Use spectral_subtraction instead.
    #
    #     Args:
    #         waveform: Input audio waveform
    #
    #     Returns:
    #         Filtered waveform
    #     """
    #     from scipy.signal import wiener
    #
    #     waveform_np = waveform.numpy() if isinstance(waveform, torch.Tensor) else waveform
    #
    #     # Apply Wiener filter with small window
    #     filtered = wiener(waveform_np, mysize=5)
    #
    #     # Make a copy to ensure contiguous array
    #     return torch.tensor(np.ascontiguousarray(filtered), dtype=torch.float32)

    def filter_audio(self, waveform: torch.Tensor, use_spectral_sub: bool = True) -> torch.Tensor:
        """
        Apply complete noise filtering pipeline.

        Args:
            waveform: Input audio waveform
            use_spectral_sub: Whether to apply spectral subtraction

        Returns:
            Filtered waveform
        """
        # Step 1: High-pass filter to remove rumble
        filtered = self.apply_highpass_filter(waveform)

        # Step 2: Band-pass filter for cry frequency range
        filtered = self.apply_bandpass_filter(filtered)

        # Step 3: Spectral subtraction for background noise
        if use_spectral_sub:
            filtered = self.spectral_subtraction(filtered)

        return filtered


class DeepSpectrumFeatures:
    """
    Extract deep spectrum features that are robust to noise.
    Based on research showing deep spectrum features outperform traditional
    features in noisy environments (F1: 0.613 in real-world conditions).
    """

    def __init__(self,
                 sample_rate: int = 16000,
                 n_mels: int = 128,
                 n_fft: int = 2048,
                 hop_length: int = 512):
        """
        Initialize deep spectrum feature extractor.

        Args:
            sample_rate: Audio sample rate
            n_mels: Number of mel bins
            n_fft: FFT size
            hop_length: Hop length
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

    def extract_gammatone_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract gammatone spectrogram (perceptually motivated).
        More robust to noise than standard mel-spectrograms.

        Args:
            waveform: Input audio waveform

        Returns:
            Gammatone spectrogram
        """
        # Note: Full gammatone implementation requires additional libraries
        # This is a mel-spectrogram approximation with perceptual scaling
        waveform_np = waveform.numpy() if isinstance(waveform, torch.Tensor) else waveform

        # Use perceptual weighting
        mel_spec = librosa.feature.melspectrogram(
            y=waveform_np,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=50,
            fmax=8000,
            htk=True  # HTK formula is closer to gammatone
        )

        # Apply perceptual loudness scaling (similar to gammatone)
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)

        return torch.tensor(log_mel, dtype=torch.float32)

    def extract_mfcc_with_deltas(self, waveform: torch.Tensor, n_mfcc: int = 40) -> torch.Tensor:
        """
        Extract MFCC with delta and delta-delta features.
        Captures temporal dynamics important for cry detection.

        Args:
            waveform: Input audio waveform
            n_mfcc: Number of MFCC coefficients

        Returns:
            Combined MFCC features with deltas
        """
        waveform_np = waveform.numpy() if isinstance(waveform, torch.Tensor) else waveform

        # Extract MFCC
        mfcc = librosa.feature.mfcc(
            y=waveform_np,
            sr=self.sample_rate,
            n_mfcc=n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

        # Compute deltas
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        # Stack all features
        combined = np.vstack([mfcc, delta, delta2])

        return torch.tensor(combined, dtype=torch.float32)

    def extract_spectral_contrast(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract spectral contrast features.
        Captures differences between peaks and valleys in spectrum.

        Args:
            waveform: Input audio waveform

        Returns:
            Spectral contrast features
        """
        waveform_np = waveform.numpy() if isinstance(waveform, torch.Tensor) else waveform

        contrast = librosa.feature.spectral_contrast(
            y=waveform_np,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

        return torch.tensor(contrast, dtype=torch.float32)

    def extract_chroma_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract chroma features (pitch class profiles).

        Args:
            waveform: Input audio waveform

        Returns:
            Chroma features
        """
        waveform_np = waveform.numpy() if isinstance(waveform, torch.Tensor) else waveform

        chroma = librosa.feature.chroma_cqt(
            y=waveform_np,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )

        return torch.tensor(chroma, dtype=torch.float32)

    def extract_all_features(self, waveform: torch.Tensor) -> dict:
        """
        Extract all deep spectrum features for robust classification.

        Args:
            waveform: Input audio waveform

        Returns:
            Dictionary of feature tensors
        """
        return {
            'gammatone': self.extract_gammatone_spectrogram(waveform),
            'mfcc_deltas': self.extract_mfcc_with_deltas(waveform),
            'spectral_contrast': self.extract_spectral_contrast(waveform),
            'chroma': self.extract_chroma_features(waveform)
        }


class AudioFilteringPipeline:
    """
    Complete audio filtering pipeline integrating all best practices.
    Combines VAD, noise filtering, and deep spectrum features.
    """

    def __init__(self, config: Config = Config()):
        """
        Initialize complete filtering pipeline.

        Args:
            config: Configuration object
        """
        self.config = config

        # Initialize components
        self.vad = VoiceActivityDetector(
            sample_rate=config.SAMPLE_RATE,
            freq_min=config.CRY_F0_MIN if hasattr(config, 'CRY_F0_MIN') else 200,
            freq_max=config.CRY_F0_MAX if hasattr(config, 'CRY_F0_MAX') else 1000
        )

        self.noise_filter = NoiseFilter(
            sample_rate=config.SAMPLE_RATE
        )

        self.deep_features = DeepSpectrumFeatures(
            sample_rate=config.SAMPLE_RATE,
            n_mels=config.N_MELS,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH
        )

    def preprocess_audio(self,
                        waveform: torch.Tensor,
                        apply_vad: bool = False,
                        apply_filtering: bool = True,
                        extract_deep_features: bool = False) -> dict:
        """
        Complete preprocessing pipeline with all filtering techniques.

        Args:
            waveform: Input audio waveform
            apply_vad: Whether to apply VAD for segmentation
            apply_filtering: Whether to apply noise filtering
            extract_deep_features: Whether to extract deep spectrum features

        Returns:
            Dictionary containing processed audio and features
        """
        results = {'original': waveform}

        # Step 1: Voice Activity Detection
        if apply_vad:
            activity_mask, confidence = self.vad.detect_activity(waveform)
            segments = self.vad.segment_audio(waveform)
            results['vad_mask'] = activity_mask
            results['vad_confidence'] = confidence
            results['vad_segments'] = segments

        # Step 2: Noise Filtering
        if apply_filtering:
            filtered = self.noise_filter.filter_audio(waveform)
            results['filtered'] = filtered
        else:
            results['filtered'] = waveform

        # Step 3: Deep Spectrum Features (optional)
        if extract_deep_features:
            features = self.deep_features.extract_all_features(results['filtered'])
            results['deep_features'] = features

        return results
